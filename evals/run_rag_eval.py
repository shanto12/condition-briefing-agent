"""RAG and follow-up test suite (MVP). Code scorers only; each case names the component and files it exercises.

  python -m evals.run_rag_eval                    # all cases, fresh isolated data dir (evals/.evaldata/rag)
  python -m evals.run_rag_eval --suite rag        # upload guardrails and search
  python -m evals.run_rag_eval --suite followup   # follow-up research agent on a seeded finished session
  python -m evals.run_rag_eval --case rag-03-search-payer
  python -m evals.run_rag_eval --probe            # report which contracts have landed; runs nothing
  python -m evals.run_rag_eval --langsmith        # upload dataset 'briefing-rag-followup' and run an experiment

Follow-up cases run on a seeded finished session (graph.update_state), so they test the follow-up agent alone
without a full briefing. Required checks decide pass/fail; advisory checks are reported but never fail a case.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

EVALS = Path(__file__).resolve().parent
BUILD = EVALS.parent
DATA_DIR = Path(os.environ.get("RAG_EVAL_DATA_DIR", EVALS / ".evaldata" / "rag")).resolve()
if DATA_DIR == (BUILD / "data").resolve():
    raise SystemExit("refusing to run the RAG suite against the demo data dir")
# Must be set before any app import: Chroma, app.db and checkpoints then live under DATA_DIR, never build/data.
os.environ["BRIEFING_DATA_DIR"] = str(DATA_DIR)

CASES_FILE = EVALS / "rag_cases.jsonl"
RESULTS = EVALS / "results"
DATASET = "briefing-rag-followup"
USER = "eval.rag"

KEYLIKE = re.compile(r"\b(?:sk-|lsv2_|pcsk_)[\w-]{8,}")
CITE = re.compile(r"\b(DOC:[a-z0-9][\w.-]*#\d+|PMID:\s?\d{1,9}|NCT\d{8}|FDA:(?:NDA|BLA|ANDA)\d{6}|PATIENT:P\d{3}|PANEL:[\w-]+)", re.I)
DOC_ID = re.compile(r"^DOC:([a-z0-9][\w.-]*)#\d+$", re.I)
NOT_IN_DOCS = re.compile(
    r"not (?:in|found in|covered (?:in|by)|available in|contained in|mentioned in|addressed in|present in|included in)"
    r" (?:our|the|any|internal|your)\b"
    r"|(?:no|none of the) (?:relevant |matching |internal )?(?:information|documents?|sources?|data|results?|records?)\b"
    r"|(?:could ?n[o']?t|cannot|can't|unable to|did not|didn't) (?:find|locate|answer)"
    r"|(?:do(?:es)? not|don't|doesn't) (?:have|contain|include|mention|cover|address)"
    r"|(?:isn't|is not) (?:in|covered|available)", re.I)
BED_COUNT = re.compile(r"\b\d[\d,]*\s+(?:[a-z-]+\s+){0,2}beds?\b", re.I)
CHAIR_COUNT = re.compile(r"\b(?:2|3|two|three)\b[^.]{0,40}\bchairs?\b|\bchairs?\b[^.]{0,40}\b(?:2|3|two|three)\b", re.I)
CONTROL_PHRASE = "registry-submission workflow"  # from corpus/payer-policy-anti-amyloid.md


# ---------------- helpers ----------------

def sanitize(text, limit: int = 300) -> str:
    text = text if isinstance(text, str) else json.dumps(text, default=str)
    try:
        from app.guardrails import redact_text
        text = redact_text(text)
    except Exception:  # noqa: BLE001 - redaction is defense in depth for report text only
        pass
    return KEYLIKE.sub("[KEY]", text)[:limit]


def err(exc: Exception) -> str:
    return sanitize(f"{type(exc).__name__}: {exc}", 400)


def snip(reply: dict | None, n: int = 150) -> str:
    return f"type={(reply or {}).get('type')} {str((reply or {}).get('text') or '')[:n]}"


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES_FILE.read_text().splitlines() if line.strip()]


CASES = {c["id"]: c for c in load_cases()}


def norm(cid: str) -> str:
    from app.guardrails import normalize_citation
    return normalize_citation(str(cid))


def source_ids(sources) -> set[str]:
    return {norm(s["id"] if isinstance(s, dict) else s) for s in (sources or [])}


def citations(claims: list[dict]) -> list[str]:
    return [norm(c) for cl in claims for c in cl.get("citations", [])]


def scan_files(root: Path, needles: list[str]) -> dict[str, list[str]]:
    """Raw-byte search of every file under root (Chroma sqlite + WAL, app.db). {relpath: [needle]}"""
    hits: dict[str, list[str]] = {}
    for f in root.rglob("*") if root.exists() else []:
        if not f.is_file():
            continue
        try:
            blob = f.read_bytes().lower()
        except OSError:
            continue
        found = [n for n in needles if n.lower().encode() in blob]
        if found:
            hits[str(f.relative_to(root))] = found
    return hits


def is_app_file(rel: str) -> bool:
    return Path(rel).name.startswith(("app.db", "checkpoints"))


def thread_values(thread_id: str) -> dict:
    from app.service import graph
    return graph().get_state({"configurable": {"thread_id": thread_id}}).values or {}


class Case:
    def __init__(self, spec: dict):
        self.spec, self.inputs, self.expect = spec, spec.get("inputs", {}), spec.get("expect", {})
        self.checks: list[dict] = []
        self.diagnostics: dict = {}
        self.setup_seconds = 0.0

    def check(self, name: str, ok, detail="", advisory: bool = False) -> bool:
        self.checks.append({"check": name, "pass": bool(ok), "advisory": advisory, "detail": sanitize(detail)})
        return bool(ok)


def scan_control(c: Case) -> None:
    """Positive control: the byte scan must see corpus text in the store, or 'not found' proves nothing."""
    seen = [f for f in scan_files(DATA_DIR, [CONTROL_PHRASE]) if not is_app_file(f)]
    c.check("store scan sees ingested corpus text (positive control)", seen, f"'{CONTROL_PHRASE}' found in {seen[:3]}")


# ---------------- shared, memoized setup ----------------

class Ctx:
    """One corpus ingest, one seeded finished session per seed, one turn per follow-up question."""

    def __init__(self):
        self.ingest: dict | None = None
        self.threads: dict[str, dict] = {}
        self.followups: dict[tuple, dict] = {}

    @staticmethod
    def rag():
        return importlib.import_module("app.rag")

    def ensure_ingested(self, case: Case | None = None) -> dict:
        if self.ingest is None:
            t0 = time.time()
            try:
                self.ingest = {"ok": True, "result": self.rag().ingest_corpus()}
            except Exception as exc:  # noqa: BLE001 - reported as a failed setup check
                self.ingest = {"ok": False, "error": err(exc)}
            self.ingest["seconds"] = round(time.time() - t0, 2)
            if case:
                case.setup_seconds += self.ingest["seconds"]
        if case:
            case.check("corpus ingested (setup)", self.ingest["ok"],
                       self.ingest.get("error") or json.dumps({k: v for k, v in (self.ingest.get("result") or {}).items()
                                                                if k != "results"}, default=str))
        return self.ingest

    def seeded_thread(self, seed: dict, case: Case) -> dict:
        """A finished session (briefing approved) written straight into the checkpoint, so the follow-up agent
        is tested on its own. No patient context: the agent never sees patient data anyway."""
        key = json.dumps(seed, sort_keys=True)
        if key not in self.threads:
            from app.service import graph
            self.ensure_ingested(case)
            tid = uuid.uuid4().hex
            briefing = {**seed, "executive_summary": [], "standard_of_care": [], "emerging_treatments": [], "landscape": [],
                        "sources": [], "verification": {}, "guardrail_events": []}
            rec = {"thread_id": tid}
            try:
                graph().update_state({"configurable": {"thread_id": tid}}, {
                    "user": USER, "request": seed["condition"], "condition": seed["condition"], "subtype": seed["subtype"],
                    "focus": seed["focus"], "briefing": briefing, "briefing_id": f"seed-{tid[:8]}", "status": "final",
                    "registry": {}}, as_node="approval")
                rec["ok"] = bool(thread_values(tid).get("briefing"))
            except Exception as exc:  # noqa: BLE001
                rec.update(ok=False, error=err(exc))
            self.threads[key] = rec
        rec = self.threads[key]
        case.check("seeded finished session (setup)", rec["ok"], rec.get("error", ""))
        return rec

    def followup(self, seed: dict, question: str, case: Case) -> dict:
        rec = self.seeded_thread(seed, case)
        key = (rec["thread_id"], question)
        if key not in self.followups:
            from langsmith.run_helpers import tracing_context

            from app.service import chat_turn
            fu: dict = {"question": question, "thread_id": rec["thread_id"], "resp": None, "answer": None, "stale_answer": False}
            t0 = time.time()
            try:
                # The app traces the graph with its own tracer and project. Nested under an experiment run, LangSmith
                # files the whole trace (experiment root included) under the app project, so start a fresh root here.
                with tracing_context(parent=False):
                    fu["resp"] = chat_turn(USER, question, rec["thread_id"])
                answer = thread_values(rec["thread_id"]).get("followup_answer")
                asked = str((answer or {}).get("question", "")).strip().lower()
                fu["stale_answer"] = answer is not None and asked != question.strip().lower()
                fu["answer"] = None if fu["stale_answer"] else answer
            except Exception as exc:  # noqa: BLE001
                fu["error"] = err(exc)
            fu["seconds"] = round(time.time() - t0, 2)
            self.followups[key] = fu
        return self.followups[key]


# ---------------- scorers (one per case kind) ----------------

def upload_phi(c: Case, ctx: Ctx) -> None:
    rag = ctx.rag()
    ctx.ensure_ingested(c)
    fixture = BUILD / c.inputs["fixture"]
    out = rag.ingest_bytes(fixture.name, fixture.read_bytes(), USER) or {}
    phi, doc_id = c.expect["phi_strings"], c.inputs["doc_id"]
    c.check("upload rejected (ok is False)", out.get("ok") is False, {k: v for k, v in out.items() if k != "text"})
    c.check("rejection reason names the PHI finding", re.search(r"phi|ssn|mrn|dob|identifier|phone|patient|health information",
                                                               str(out.get("reason", "")), re.I), out.get("reason", ""), advisory=True)
    c.check("rejection response does not echo identifiers",
            not [s for s in phi if s.lower() in json.dumps(out, default=str).lower()], advisory=True)
    listed = [d for d in rag.list_documents() if doc_id in json.dumps(d, default=str) or fixture.stem in json.dumps(d, default=str)]
    c.check("not listed in list_documents()", not listed, f"{len(listed)} matching entries")
    leaked = []
    for q in c.inputs["probe_queries"]:
        for r in rag.search(q) or []:
            if r.get("doc_id") == doc_id or any(s.lower() in str(r.get("text", "")).lower() for s in phi):
                leaked.append(r.get("id"))
    c.check("no search result from the PHI upload", not leaked, f"returned: {leaked}")
    scan_control(c)
    hits = scan_files(DATA_DIR, phi)
    index_hits = {f: h for f, h in hits.items() if not is_app_file(f)}
    c.check("no identifier bytes in the vector store or index files", not index_hits,
            f"{len(index_hits)} files contain upload identifiers: {sorted(index_hits)[:5]}")
    c.check("no identifier bytes in app.db or checkpoints", not {f for f in hits if is_app_file(f)},
            sorted(f for f in hits if is_app_file(f)), advisory=True)


def upload_injection(c: Case, ctx: Ctx) -> None:
    from app.guardrails import INJECTION
    rag = ctx.rag()
    ctx.ensure_ingested(c)
    fixture = BUILD / c.inputs["fixture"]
    out = rag.ingest_bytes(fixture.name, fixture.read_bytes(), USER) or {}
    raw = c.expect["raw_phrases"]
    scan_control(c)
    stored_raw = {f: h for f, h in scan_files(DATA_DIR, raw).items() if not is_app_file(f)}
    c.check("instruction text flagged, rejected or neutralized at ingest",
            out.get("ok") is False or bool(out.get("flags")) or not stored_raw,
            f"ok={out.get('ok')} flags={out.get('flags')} chunks={out.get('chunks')} reason={out.get('reason')}")
    bad, payload_hits = [], []
    for q in c.inputs["probe_queries"]:
        for r in rag.search(q) or []:
            text = str(r.get("text", ""))
            if INJECTION.search(text) or any(p in text.lower() for p in raw):
                bad.append(r.get("id"))
            if c.expect["payload"] in text.lower():
                payload_hits.append(r.get("id"))
    c.check("search never returns the raw instruction text", not bad, f"raw instruction in: {bad}")
    c.check("injected payload sentence not returned", not payload_hits, f"payload in: {payload_hits}", advisory=True)
    c.check("raw instruction text not persisted in the vector store", not stored_raw, sorted(stored_raw)[:5], advisory=True)
    if out.get("ok"):
        hits = [r.get("id") for r in rag.search(c.inputs["benign_query"]) or [] if r.get("doc_id") == c.inputs["doc_id"]]
        c.check("benign brochure content still searchable after neutralization", hits, f"hits from the uploaded doc: {hits[:3]}",
                advisory=True)


def _shape(c: Case, results: list[dict]) -> None:
    bad = [r.get("id") for r in results
           if not DOC_ID.match(str(r.get("id", ""))) or not str(r.get("id", "")).startswith(f"DOC:{r.get('doc_id')}#")
           or not {"doc_id", "title", "section", "text", "score"} <= set(r)]
    c.check("results match the search() contract (DOC:<doc_id>#<n>, doc_id, title, section, text, score)", not bad, bad[:5])


def search_top(c: Case, ctx: Ctx) -> None:
    ctx.ensure_ingested(c)
    results = ctx.rag().search(c.inputs["query"]) or []
    _shape(c, results)
    ranked = [r.get("doc_id") for r in results]
    n, want = c.expect["top_n"], c.expect["doc_id"]
    c.check(f"{want} in top {n}", want in ranked[:n], f"top {n}: {ranked[:n]}")
    c.check(f"{want} ranked first", ranked[:1] == [want], f"rank 1: {ranked[:1]}", advisory=True)


def search_empty(c: Case, ctx: Ctx) -> None:
    from app import config
    ctx.ensure_ingested(c)
    rag = ctx.rag()
    results = rag.search(c.inputs["query"]) or []
    rerank_min = next((getattr(m, a) for m, a in ((rag, "RERANK_MIN_SCORE"), (config, "RERANK_MIN_SCORE"))
                       if isinstance(getattr(m, a, None), (int, float))), 0.0)
    cosine_min = getattr(rag, "COSINE_MIN", rerank_min)

    def floor(r: dict) -> float:
        return cosine_min if r.get("score_kind") == "cosine" else rerank_min

    above = [(r.get("id"), r.get("score"), r.get("score_kind")) for r in results
             if not isinstance(r.get("score"), (int, float)) or r["score"] >= floor(r)]
    c.check(f"no results above the threshold (rerank {rerank_min}, cosine {cosine_min})", not above,
            f"{len(results)} results; above: {above[:5]}")
    c.check("returns an empty list", not results, f"{len(results)} results below threshold", advisory=True)


def _followup_common(c: Case, fu: dict, types: set[str], need_answer: bool) -> tuple[dict, dict, str]:
    resp, answer = fu.get("resp") or {}, fu.get("answer") or {}
    c.check("answered on the same thread", resp.get("thread_id") and resp.get("thread_id") == fu.get("thread_id"),
            fu.get("error") or f"asked on {fu.get('thread_id')}, reply on {resp.get('thread_id')}")
    c.check(f"response type in {sorted(types)}", resp.get("type") in types, fu.get("error") or snip(resp))
    if need_answer:
        c.check("followup_answer recorded in thread state for this question", fu.get("answer") is not None,
                "stale followup_answer from an earlier question" if fu.get("stale_answer") else "no followup_answer in state")
    c.diagnostics.update(tool_calls=answer.get("tool_calls"), reason=sanitize(answer.get("reason") or ""),
                         turn_seconds=fu.get("seconds"))
    return resp, answer, " ".join([str(resp.get("text", "")), str(answer.get("text", ""))])


def _citations_resolve(c: Case, fu: dict, answer: dict, text: str) -> None:
    ids = source_ids(answer.get("sources"))
    unresolved = sorted(set(citations(answer.get("claims", []))) - ids)
    in_text = sorted({norm(m) for m in CITE.findall(text)} - ids)
    c.check("every claim citation is in that run's sources", not unresolved, unresolved[:6])
    c.check("every citation in the reply text is in that run's sources", not in_text, in_text[:6])


def followup_cited(c: Case, ctx: Ctx) -> None:
    fu = ctx.followup(c.inputs["seed"], c.inputs["question"], c)
    _, answer, text = _followup_common(c, fu, {"followup"}, need_answer=True)
    cites = citations(answer.get("claims", []))
    want = c.expect["citation_prefix"].lower()
    c.check(f"a verified claim cites {c.expect['citation_prefix']}<n>", any(x.lower().startswith(want) for x in cites),
            f"claim citations: {cites[:8]}; removed: {len(answer.get('removed') or [])}; reason: {answer.get('reason')}")
    _citations_resolve(c, fu, answer, text)
    c.check("answer states a chair count", CHAIR_COUNT.search(text), text[:150], advisory=True)


def followup_not_in_docs(c: Case, ctx: Ctx) -> None:
    fu = ctx.followup(c.inputs["seed"], c.inputs["question"], c)
    _, answer, text = _followup_common(c, fu, {"followup", "refusal"}, need_answer=False)
    claims = answer.get("claims", []) or []
    c.check("says the answer is not in our documents", NOT_IN_DOCS.search(text), text[:200])
    uncited = [cl.get("text", "")[:80] for cl in claims if not cl.get("citations")]
    c.check("no uncited claims", not uncited, uncited[:3])
    bed = BED_COUNT.search(text + " " + json.dumps(claims))
    c.check("no fabricated bed count", not bed, bed.group(0) if bed else "")
    _citations_resolve(c, fu, answer, text)
    c.check("asserts no claims at all", not claims, f"{len(claims)} claims", advisory=True)


SCORERS = {"upload_phi": upload_phi, "upload_injection": upload_injection, "search_top": search_top,
           "search_empty": search_empty, "followup_cited": followup_cited, "followup_not_in_docs": followup_not_in_docs}


def run_case(spec: dict, ctx: Ctx) -> dict:
    c = Case(spec)
    t0 = time.time()
    error = None
    try:
        SCORERS[spec["kind"]](c, ctx)
    except Exception as exc:  # noqa: BLE001 - a crash is a failed case with the exception as detail
        error = err(exc)
        c.check("case ran without an exception", False, error)
    required = [x for x in c.checks if not x["advisory"]]
    return {"id": spec["id"], "suite": spec["suite"], "title": spec["title"], "owner": spec.get("owner", {}),
            "pass": bool(required) and all(x["pass"] for x in required), "seconds": round(time.time() - t0, 2),
            "setup_seconds": round(c.setup_seconds, 2), "error": error, "checks": c.checks, "diagnostics": c.diagnostics}


# ---------------- contract probe ----------------

def contract_status() -> dict:
    out: dict = {}
    try:
        rag = importlib.import_module("app.rag")
        out["app.rag"] = {f: callable(getattr(rag, f, None)) for f in ("ingest_corpus", "ingest_bytes", "list_documents", "search")}
    except Exception as exc:  # noqa: BLE001
        out["app.rag"] = f"not importable: {err(exc)}"
    for mod, needles in (("app.service", ("followup",)), ("app.graph", ("followup_answer",))):
        try:
            src = inspect.getsource(importlib.import_module(mod))
            out[mod] = {n: n in src for n in needles}
        except Exception as exc:  # noqa: BLE001
            out[mod] = f"not importable: {err(exc)}"
    return out


# ---------------- LangSmith experiment ----------------

def ls_example(spec: dict) -> dict:
    return {"inputs": {"case_id": spec["id"], "suite": spec["suite"], "kind": spec["kind"], **spec["inputs"]},
            "outputs": spec["expect"],
            "metadata": {"case_id": spec["id"], "suite": spec["suite"], "component": spec.get("owner", {}).get("component")}}


def upsert_dataset(client):
    """Makes dataset 'briefing-rag-followup' match rag_cases.jsonl, keyed by metadata.case_id. Safe to rerun."""
    wanted = {c["id"]: ls_example(c) for c in load_cases()}
    if client.has_dataset(dataset_name=DATASET):
        ds = client.read_dataset(dataset_name=DATASET)
    else:
        ds = client.create_dataset(DATASET, description="RAG and follow-up cases: PHI and injection uploads, retrieval "
                                   "top-3 and threshold, follow-up answered with a DOC: citation, follow-up cite-or-refuse.")
    existing = {(ex.metadata or {}).get("case_id"): ex for ex in client.list_examples(dataset_id=ds.id)}
    new = [e for cid, e in wanted.items() if cid not in existing]
    if new:
        client.create_examples(dataset_id=ds.id, examples=new)
    updated = removed = 0
    for cid, ex in existing.items():
        want = wanted.get(cid)
        if want is None:
            client.delete_example(ex.id)
            removed += 1
        elif ex.inputs != want["inputs"] or (ex.outputs or {}) != want["outputs"]:
            client.update_example(ex.id, inputs=want["inputs"], outputs=want["outputs"],
                                  metadata={**(ex.metadata or {}), **want["metadata"]})
            updated += 1
    print(f"dataset {DATASET}: {len(wanted)} examples ({len(new)} created, {updated} updated, {removed} removed)")
    return ds


def ls_case_pass(outputs: dict) -> dict:
    failed = [x["check"] for x in (outputs or {}).get("checks", []) if not x["pass"] and not x["advisory"]]
    return {"key": "case_pass", "score": int(bool((outputs or {}).get("pass"))), "comment": "; ".join(failed)[:500] or None}


def ls_required_rate(outputs: dict) -> dict:
    rows = [x for x in (outputs or {}).get("checks", []) if not x["advisory"]]
    return {"key": "required_checks_rate", "score": sum(x["pass"] for x in rows) / len(rows)} if rows else {"results": []}


def ls_latency(outputs: dict) -> dict:
    s = (outputs or {}).get("seconds")
    return {"key": "latency_s", "score": s} if s is not None else {"results": []}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=BUILD, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def run_langsmith(cases: list[dict], ctx: Ctx, args) -> tuple[list[dict], dict]:
    from langsmith import evaluate

    from app import config

    t = config.tracer()
    if t is None:
        raise SystemExit("--langsmith needs the LangSmith key (env or Keychain, read by app/config.py) and BRIEFING_TRACING on")
    client = t.client  # the app's redacting client: hide_inputs/hide_outputs apply to experiment runs too
    ds = upsert_dataset(client)
    order = {c["id"]: n for n, c in enumerate(cases)}
    examples = sorted((ex for ex in client.list_examples(dataset_id=ds.id) if (ex.metadata or {}).get("case_id") in order),
                      key=lambda ex: order[(ex.metadata or {})["case_id"]])

    def target(inputs: dict) -> dict:
        return run_case(CASES[inputs["case_id"]], ctx)

    experiment = evaluate(
        target, data=examples, evaluators=[ls_case_pass, ls_required_rate, ls_latency],
        experiment_prefix=args.experiment, max_concurrency=0, client=client,
        description="RAG and follow-up suite (code scorers) on an isolated data dir",
        metadata={"model": config.MODEL, "embed_model": config.EMBED_MODEL, "rerank": config.RERANK_ON,
                  "git_sha": _git_sha(), "suite": args.suite},
    )
    results = []
    for row in experiment:
        run, ex = row["run"], row["example"]
        out = dict(run.outputs or {}) if run else {}
        if not out.get("id"):
            spec = CASES[(ex.metadata or {})["case_id"]]
            out = {"id": spec["id"], "suite": spec["suite"], "title": spec["title"], "owner": spec.get("owner", {}), "pass": False,
                   "seconds": None, "setup_seconds": 0, "error": sanitize(getattr(run, "error", "") or "target produced no output"),
                   "checks": []}
        out["langsmith_run_id"] = str(run.id) if run else None
        results.append(out)
    client.flush(timeout=60)  # the last run and its feedback are still queued in the background uploader at exit
    return results, {"experiment": experiment.experiment_name, "url": experiment.url, "dataset": DATASET}


# ---------------- report ----------------

def report(results: list[dict], ctx: Ctx, summary: dict) -> Path:
    print(f"\n{'case':36} {'result':6} {'seconds':>8} {'(setup)':>8}")
    for r in results:
        print(f"{r['id']:36} {'PASS' if r['pass'] else 'FAIL':6} {r['seconds'] if r['seconds'] is not None else '-':>8} "
              f"{r['setup_seconds'] or '':>8}")
    for r in results:
        failed = [x for x in r["checks"] if not x["pass"] and not x["advisory"]]
        if failed or (not r["pass"] and r.get("error")):
            owner = r.get("owner", {})
            print(f"\nFAIL {r['id']}  -> {owner.get('component')} [{', '.join(owner.get('files', []))}]")
            for x in failed:
                print(f"   - {x['check']}: {x['detail'][:220]}")
            if r.get("diagnostics", {}).get("tool_calls") is not None:
                print(f"   tool calls: {sanitize(r['diagnostics']['tool_calls'], 280)}")
    notes = [(r["id"], x) for r in results for x in r["checks"] if x["advisory"] and not x["pass"]]
    if notes:
        print("\nadvisory (not scored):")
        for cid, x in notes:
            print(f"   {cid}: {x['check']}: {x['detail'][:160]}")
    req = [x for r in results for x in r["checks"] if not x["advisory"]]
    adv = [x for r in results for x in r["checks"] if x["advisory"]]
    passed = sum(r["pass"] for r in results)
    total_s = round(sum(r["seconds"] or 0 for r in results), 1)
    print(f"\nTOTAL {passed}/{len(results)} cases passed | required checks {sum(x['pass'] for x in req)}/{len(req)} "
          f"| advisory {sum(x['pass'] for x in adv)}/{len(adv)} | {total_s}s")
    if summary:
        print(f"LangSmith experiment {summary['experiment']}\n{summary['url']}")
    from app import config
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"rag-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "summary": {"cases_passed": passed, "cases_total": len(results), "required_checks_passed": sum(x["pass"] for x in req),
                    "required_checks_total": len(req), "seconds_total": total_s},
        "env": {"data_dir": str(DATA_DIR.relative_to(BUILD)) if DATA_DIR.is_relative_to(BUILD) else str(DATA_DIR),
                "model": config.MODEL, "embed_provider": config.EMBED_PROVIDER, "embed_model": config.EMBED_MODEL,
                "rerank": config.RERANK_ON},
        "ingest": ctx.ingest, "langsmith": summary or None, "cases": results,
        "followups": [{"question": f["question"], "seconds": f.get("seconds"), "type": (f.get("resp") or {}).get("type"),
                       "tool_calls": (f.get("answer") or {}).get("tool_calls")} for f in ctx.followups.values()],
    }, indent=1, default=str))
    print(f"saved {out.relative_to(BUILD)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=("all", "rag", "followup"), default="all")
    parser.add_argument("--case", action="append", help="run only these case ids")
    parser.add_argument("--keep-data", action="store_true", help="reuse the isolated data dir instead of starting fresh")
    parser.add_argument("--probe", action="store_true", help="report which contracts have landed, then exit")
    parser.add_argument("--langsmith", action="store_true", help=f"upload cases as dataset '{DATASET}' and run evaluate()")
    parser.add_argument("--experiment", default="rag-followup", help="LangSmith experiment prefix")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(contract_status(), indent=1))
        return 0
    if not args.keep_data and DATA_DIR.exists() and DATA_DIR.is_relative_to(EVALS):
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases = [c for c in load_cases() if (args.suite == "all" or c["suite"] == args.suite) and (not args.case or c["id"] in args.case)]
    if not cases:
        raise SystemExit("no cases selected")
    ctx = Ctx()
    summary: dict = {}
    if args.langsmith:
        results, summary = run_langsmith(cases, ctx, args)
    else:
        results = [run_case(c, ctx) for c in cases]
    report(results, ctx, summary)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
