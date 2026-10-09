"""Internal document RAG.

ingest: parse front matter -> PHI gate (reject the file before any text leaves the machine) -> neutralize injection
-> heading-aware chunks -> embed -> persistent Chroma (one collection per embedding model; dims differ).
search: dense top 20 + BM25 top 20 -> reciprocal rank fusion -> Pinecone-hosted rerank -> threshold -> top k.

  python -m app.rag ingest
  python -m app.rag search "what does Medicare require for anti-amyloid coverage?"
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import threading
from pathlib import Path

import httpx
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from . import config as settings
from .guardrails import neutralize_injection, phi_findings

PINECONE = "https://api.pinecone.io"
PINECONE_HEADERS = {"X-Pinecone-API-Version": "2025-04", "Content-Type": "application/json"}
CANDIDATES = 20
RRF_K = 60
# Tuned on evals/retrieval.jsonl: answerable top-1 rerank scores >= 0.96, the unanswerable question 0.18.
RERANK_MIN_SCORE = float(os.environ.get("RERANK_MIN_SCORE", "0.3"))
# Fallback when reranking is off or fails: keeps every answerable top-1 (min 0.52), but cannot reject off-corpus questions.
COSINE_MIN = float(os.environ.get("COSINE_MIN", "0.5"))
SUFFIXES = {".md", ".txt", ".pdf"}
MAX_UPLOAD_BYTES = 2_000_000
STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "is", "are", "what", "how", "do", "does", "we",
        "our", "which", "with", "by", "at", "be", "it", "this", "that", "many", "much"}

_splitter_md = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True)
# ~800 tokens with ~120 overlap at about 4 characters per token; avoids a tokenizer download.
_splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=480)
_lock = threading.Lock()


class PineconeEmbeddings(Embeddings):
    def __init__(self, model: str = "llama-text-embed-v2"):
        self.model = model

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 96):
            r = httpx.post(f"{PINECONE}/embed", timeout=60, headers={**PINECONE_HEADERS, "Api-Key": settings.pinecone_key()},
                           json={"model": self.model, "parameters": {"input_type": input_type, "truncate": "END"},
                                 "inputs": [{"text": t} for t in texts[i:i + 96]]})
            r.raise_for_status()
            out += [d["values"] for d in r.json()["data"]]
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


def embeddings(provider: str, model: str) -> Embeddings:
    if provider == "pinecone":
        return PineconeEmbeddings(model)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model, api_key=settings.openai_key(), check_embedding_ctx_length=False)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "document"


class Index:
    def __init__(self, provider: str, model: str):
        from langchain_chroma import Chroma

        self.provider, self.model = provider, model
        settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(collection_name=f"docs-{provider}-{_slug(model)}", embedding_function=embeddings(provider, model),
                            persist_directory=str(settings.CHROMA_DIR), collection_metadata={"hnsw:space": "cosine"})
        self._bm25 = None

    @property
    def collection(self):
        return self.store._collection

    def _existing(self, doc_id: str) -> dict | None:
        got = self.collection.get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
        return got["metadatas"][0] if got["metadatas"] else None

    def add(self, doc: dict) -> dict:
        existing = self._existing(doc["doc_id"])
        if existing and existing.get("content_hash") == doc["content_hash"]:
            return {"ok": True, "doc_id": doc["doc_id"], "chunks": self.count(doc["doc_id"]), "flags": doc["flags"], "skipped": True}
        if existing and existing.get("source") == "corpus" and doc["source"] != "corpus":
            return {"ok": False, "reason": f"doc_id '{doc['doc_id']}' belongs to a curated corpus document; uploads cannot replace it."}
        chunks = chunk(doc)
        with _lock:
            if existing:
                self.collection.delete(where={"doc_id": doc["doc_id"]})
            self.store.add_texts([c["text"] for c in chunks], metadatas=[c["metadata"] for c in chunks],
                                 ids=[c["id"] for c in chunks])
            self._bm25 = None
        return {"ok": True, "doc_id": doc["doc_id"], "chunks": len(chunks), "flags": doc["flags"], "skipped": False}

    def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return self.collection.count()
        return len(self.collection.get(where={"doc_id": doc_id}, include=[])["ids"])

    def all_chunks(self) -> tuple[list[str], list[str], list[dict]]:
        got = self.collection.get(include=["documents", "metadatas"])
        return got["ids"], got["documents"], got["metadatas"]

    def bm25(self):
        with _lock:
            if self._bm25 is None:
                ids, texts, metas = self.all_chunks()
                self._bm25 = (BM25Okapi([_tokens(t) for t in texts]) if texts else None, ids, texts, metas)
            return self._bm25

    def search(self, query: str, k: int, rerank: bool, min_score: float) -> list[Document]:
        if self.count() == 0:
            ingest_corpus(self)
        dense = self.store.similarity_search_with_score(query, k=CANDIDATES)
        cosine = {d.metadata["chunk_id"]: 1 - dist for d, dist in dense}
        by_id = {d.metadata["chunk_id"]: d for d, _ in dense}
        fused: dict[str, float] = {}
        for rank, (d, _) in enumerate(dense):
            fused[d.metadata["chunk_id"]] = 1 / (RRF_K + rank + 1)
        model, ids, texts, metas = self.bm25()
        if model:
            scores = model.get_scores(_tokens(query))
            top = sorted(range(len(ids)), key=lambda i: -scores[i])[:CANDIDATES]
            for rank, i in enumerate(t for t in top if scores[t] > 0):
                fused[ids[i]] = fused.get(ids[i], 0) + 1 / (RRF_K + rank + 1)
                by_id.setdefault(ids[i], Document(page_content=texts[i], metadata=metas[i]))
        order = sorted(fused, key=lambda i: -fused[i])
        if rerank:
            try:
                scored = _rerank(query, [(i, by_id[i].page_content) for i in order])
                return [_hit(by_id[i], s, "rerank", cosine.get(i)) for i, s in scored if s >= min_score][:k]
            except (httpx.HTTPError, KeyError, ValueError, TypeError):
                pass
        return [_hit(by_id[i], cosine[i], "cosine", cosine[i]) for i in order if i in cosine and cosine[i] >= COSINE_MIN][:k]


def _hit(doc: Document, score: float, kind: str, cosine: float | None) -> Document:
    meta = {**doc.metadata, "score": round(float(score), 4), "score_kind": kind,
            "cosine": round(cosine, 4) if cosine is not None else None}
    return Document(page_content=doc.page_content, metadata=meta, id=doc.metadata["chunk_id"])


def _rerank(query: str, docs: list[tuple[str, str]]) -> list[tuple[str, float]]:
    key = settings.pinecone_key()
    if not key or not docs:
        raise ValueError("rerank unavailable")
    r = httpx.post(f"{PINECONE}/rerank", timeout=30, headers={**PINECONE_HEADERS, "Api-Key": key},
                   json={"model": settings.RERANK_MODEL, "query": query, "top_n": len(docs), "return_documents": False,
                         "documents": [{"id": i, "text": t} for i, t in docs], "parameters": {"truncate": "END"}})
    r.raise_for_status()
    return [(docs[d["index"]][0], float(d["score"])) for d in r.json()["data"]]


# ---------------- parsing and chunking ----------------

def _front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[m.end():]


def _decode(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="replace")


def parse(filename: str, data: bytes, source: str, user: str | None = None) -> dict:
    """Returns a document dict, or {"reason": ...} when the PHI gate rejects it. Nothing is sent anywhere here."""
    raw = _decode(filename, data)
    phi = phi_findings(raw)
    if phi:
        return {"reason": f"Rejected before embedding: PHI patterns found ({', '.join(phi)}). Remove identifiers and re-upload."}
    meta, body = _front_matter(raw)
    lines, removed = [], 0
    for line in body.splitlines():
        clean, n = neutralize_injection(line)
        lines.append(clean)
        removed += n
    flags = [f"injection: {removed} instruction-like sentence(s) removed"] if removed else []
    stem = Path(filename).stem
    return {
        "doc_id": _slug(meta.get("doc_id") or stem), "title": meta.get("title") or stem.replace("-", " ").title(),
        "doc_type": meta.get("doc_type", "document"), "date": meta.get("date", ""),
        "synthetic": str(meta.get("synthetic", "false")).lower() == "true", "body": "\n".join(lines),
        "content_hash": hashlib.sha256(data).hexdigest()[:16], "source": source, "uploaded_by": user or "",
        "flags": flags,
    }


def chunk(doc: dict) -> list[dict]:
    out = []
    for section in _splitter_md.split_text(doc["body"]):
        heading = section.metadata.get("h3") or section.metadata.get("h2") or section.metadata.get("h1") or "Notice"
        for piece in _splitter.split_text(section.page_content):
            if len(piece.strip()) < 20:
                continue
            n = len(out) + 1
            chunk_id = f"DOC:{doc['doc_id']}#{n}"
            out.append({"id": chunk_id, "text": f"{doc['title']} - {heading}\n{piece.strip()}", "metadata": {
                "chunk_id": chunk_id, "doc_id": doc["doc_id"], "title": doc["title"], "doc_type": doc["doc_type"],
                "date": doc["date"], "section": heading, "synthetic": doc["synthetic"], "content_hash": doc["content_hash"],
                "source": doc["source"], "uploaded_by": doc["uploaded_by"], "injection_flag": bool(doc["flags"]), "chunk": n}})
    return out


# ---------------- public contract ----------------

_indexes: dict[tuple[str, str], Index] = {}
_index_lock = threading.Lock()


def index(provider: str | None = None, model: str | None = None) -> Index:
    key = (provider or settings.EMBED_PROVIDER, model or settings.EMBED_MODEL)
    with _index_lock:
        if key not in _indexes:
            _indexes[key] = Index(*key)
        return _indexes[key]


class DocumentRetriever(BaseRetriever):
    """Hybrid retriever over internal documents. A LangChain retriever so each call is a retriever span in LangSmith."""
    provider: str = settings.EMBED_PROVIDER
    model: str = settings.EMBED_MODEL
    k: int = settings.RETRIEVE_TOP_K
    rerank: bool = settings.RERANK_ON
    min_score: float = RERANK_MIN_SCORE

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return index(self.provider, self.model).search(query, self.k, self.rerank, self.min_score)


def ingest_corpus(idx: Index | None = None) -> dict:
    idx = idx or index()
    results = []
    for path in sorted(p for p in settings.CORPUS_DIR.iterdir() if p.suffix.lower() in SUFFIXES):
        doc = parse(path.name, path.read_bytes(), "corpus")
        results.append({"file": path.name, **(idx.add(doc) if "reason" not in doc else {"ok": False, "reason": doc["reason"]})})
    return {"collection": idx.collection.name, "documents": len(results), "chunks": idx.count(),
            "embedded": sum(1 for r in results if r.get("ok") and not r.get("skipped")),
            "skipped_unchanged": sum(1 for r in results if r.get("skipped")), "results": results}


def ingest_bytes(filename: str, data: bytes, user: str) -> dict:
    name = Path(filename or "").name
    if Path(name).suffix.lower() not in SUFFIXES:
        return {"ok": False, "reason": "Only .md, .txt and .pdf files are accepted."}
    if not data or len(data) > MAX_UPLOAD_BYTES:
        return {"ok": False, "reason": "File is empty or larger than 2 MB."}
    doc = parse(name, data, "upload", user)
    if "reason" in doc:
        return {"ok": False, "reason": doc["reason"]}
    return index().add(doc)


def list_documents() -> list[dict]:
    _, _, metas = index().all_chunks()
    docs: dict[str, dict] = {}
    for m in metas:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "title": m["title"], "doc_type": m["doc_type"],
                                          "date": m["date"], "chunks": 0, "synthetic": m["synthetic"],
                                          "source": m.get("source"), "injection_flag": False})
        d["chunks"] += 1
        d["injection_flag"] |= bool(m.get("injection_flag"))
    return sorted(docs.values(), key=lambda d: d["doc_id"])


def remove_document(doc_id: str) -> int:
    idx = index()
    n = idx.count(doc_id)
    with _lock:
        idx.collection.delete(where={"doc_id": doc_id})
        idx._bm25 = None
    return n


def search(query: str, k: int | None = None, config=None) -> list[dict]:
    retriever = DocumentRetriever(k=k or settings.RETRIEVE_TOP_K)
    docs = retriever.invoke(query, config=config)
    return [{"id": d.metadata["chunk_id"], "doc_id": d.metadata["doc_id"], "title": d.metadata["title"],
             "section": d.metadata["section"], "date": d.metadata.get("date"), "text": d.page_content,
             "score": d.metadata["score"], "score_kind": d.metadata["score_kind"]} for d in docs]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ingest"
    if cmd == "ingest":
        report = ingest_corpus()
        for r in report["results"]:
            status = "skipped (unchanged)" if r.get("skipped") else f"{r.get('chunks')} chunks" if r.get("ok") else r["reason"]
            print(f"{r['file']:42} {status} {' '.join(r.get('flags', []))}")
        print(f"collection {report['collection']}: {report['chunks']} chunks; embedded {report['embedded']}, "
              f"unchanged {report['skipped_unchanged']}")
    elif cmd == "search":
        for hit in search(" ".join(sys.argv[2:])):
            print(f"{hit['score']:.3f} {hit['score_kind']:6} {hit['id']:44} {hit['section']}")
    elif cmd == "upload":
        print(ingest_bytes(sys.argv[2], Path(sys.argv[2]).read_bytes(), "cli"))
