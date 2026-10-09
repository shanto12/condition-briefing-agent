"""Retrieval eval on evals/retrieval.jsonl: recall@5 and MRR per embedding model, with and without reranking,
plus the unanswerable case (passes only if nothing clears the threshold).

  python -m evals.run_retrieval_eval               # text-embedding-3-large and -3-small, rerank on/off
  python -m evals.run_retrieval_eval --langsmith   # also upload dataset 'briefing-retrieval' and run evaluate()
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from app import config, rag

CASES = Path(__file__).parent / "retrieval.jsonl"
RESULTS = Path(__file__).parent / "results"
MODELS = ["text-embedding-3-large", "text-embedding-3-small"]
K = 5


def load() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def score_case(case: dict, hits: list[dict]) -> dict:
    docs = [h["doc_id"] for h in hits]
    if not case["answerable"]:
        return {"id": case["id"], "answerable": False, "pass": not hits, "top": docs[:1],
                "top_score": hits[0]["score"] if hits else None}
    rank = next((i + 1 for i, d in enumerate(docs[:K]) if d == case["expected_doc_id"]), None)
    return {"id": case["id"], "answerable": True, "hit": rank is not None, "rr": 1 / rank if rank else 0.0, "rank": rank,
            "top_score": hits[0]["score"] if hits else None}


def run(model: str, rerank: bool, cases: list[dict]) -> dict:
    idx = rag.index("openai", model)
    if idx.count() == 0:
        rag.ingest_corpus(idx)
    retriever = rag.DocumentRetriever(provider="openai", model=model, k=K, rerank=rerank)
    rows, latencies = [], []
    for case in cases:
        t0 = time.time()
        docs = retriever.invoke(case["question"])
        latencies.append(time.time() - t0)
        rows.append(score_case(case, [d.metadata for d in docs]))
    answerable = [r for r in rows if r["answerable"]]
    return {"model": model, "rerank": rerank,
            "threshold": rag.RERANK_MIN_SCORE if rerank else rag.COSINE_MIN,
            "recall@5": round(sum(r["hit"] for r in answerable) / len(answerable), 3),
            "mrr": round(sum(r["rr"] for r in answerable) / len(answerable), 3),
            "unanswerable_pass": all(r["pass"] for r in rows if not r["answerable"]),
            "min_answerable_top_score": min((r["top_score"] for r in answerable if r["top_score"] is not None), default=None),
            "p50_ms": round(sorted(latencies)[len(latencies) // 2] * 1000), "rows": rows}


def langsmith_experiments(cases: list[dict]) -> None:
    from langsmith import Client, evaluate

    client = Client(api_key=config._secret("LANGSMITH_API_KEY", "codex-langsmith-personal-key"))
    name = "briefing-retrieval"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="Internal-document retrieval questions with expected doc_id (synthetic corpus).")
        client.create_examples(dataset_id=ds.id, inputs=[{"question": c["question"]} for c in cases],
                               outputs=[{"expected_doc_id": c["expected_doc_id"], "answerable": c["answerable"]} for c in cases])

    def recall_at_5(run, example):
        docs = (run.outputs or {}).get("doc_ids", [])
        if not example.outputs["answerable"]:
            return {"key": "recall@5", "score": int(not docs)}
        return {"key": "recall@5", "score": int(example.outputs["expected_doc_id"] in docs[:K])}

    def reciprocal_rank(run, example):
        docs = (run.outputs or {}).get("doc_ids", [])
        if not example.outputs["answerable"]:
            return {"key": "rr", "score": int(not docs)}
        rank = next((i + 1 for i, d in enumerate(docs) if d == example.outputs["expected_doc_id"]), None)
        return {"key": "rr", "score": 1 / rank if rank else 0}

    for model in MODELS:
        for rerank in (True, False):
            retriever = rag.DocumentRetriever(provider="openai", model=model, k=K, rerank=rerank)

            def target(inputs: dict, retriever=retriever) -> dict:
                return {"doc_ids": [d.metadata["doc_id"] for d in retriever.invoke(inputs["question"])]}

            evaluate(target, data=name, evaluators=[recall_at_5, reciprocal_rank], client=client,
                     experiment_prefix=f"retrieval-{model}-{'rerank' if rerank else 'norerank'}",
                     metadata={"embed_model": model, "rerank": rerank, "k": K})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--model", action="append", help="limit to these embedding models")
    args = parser.parse_args()
    cases = load()
    reports = [run(m, rr, cases) for m in (args.model or MODELS) for rr in (True, False)]
    print(f"{'embedding model':24} {'rerank':6} {'recall@5':>8} {'MRR':>6} {'unanswerable':>12} {'min top score':>13} {'p50 ms':>7}")
    for r in reports:
        print(f"{r['model']:24} {'on' if r['rerank'] else 'off':6} {r['recall@5']:>8} {r['mrr']:>6} "
              f"{'pass' if r['unanswerable_pass'] else 'FAIL':>12} {r['min_answerable_top_score']!s:>13} {r['p50_ms']:>7}")
        for row in r["rows"]:
            if (row["answerable"] and row["rank"] != 1) or (not row["answerable"] and not row["pass"]):
                print(f"    {row['id']}: rank={row.get('rank')} top_score={row['top_score']}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"retrieval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(reports, indent=1))
    print(f"saved {out}")
    if args.langsmith:
        langsmith_experiments(cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
