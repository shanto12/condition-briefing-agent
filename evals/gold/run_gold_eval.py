"""Reference-answer (gold) experiment: upload evals/gold/gold_v1.jsonl as 'briefing-gold-v1' and score the real app.

  python -m evals.gold.run_gold_eval                         # upload + full experiment
  python -m evals.gold.run_gold_eval --segment corpus_qa     # one segment (repeatable)
  python -m evals.gold.run_gold_eval --upload-only
  python -m evals.gold.run_gold_eval --rag-only              # corpus_qa scored on app.rag.search hits (no follow-up agent)

Targets: corpus_qa and guardrail_refusals ask the follow-up agent on one finished Alzheimer's session (app.service.chat_turn).
briefing_facts and sponsor_naming score that same briefing. One briefing is shared by every case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

GOLD = Path(__file__).resolve().parent
EVALS = GOLD.parent
BUILD = EVALS.parent
DATA_DIR = (EVALS / ".evaldata" / "gold").resolve()
if DATA_DIR == (BUILD / "data").resolve():
    raise SystemExit("refusing to run against the demo data dir")
os.environ["BRIEFING_DATA_DIR"] = str(DATA_DIR)  # before any app import

from evals.gold import evaluators as ev  # noqa: E402

DATASET = "briefing-gold-v1"
GOLD_FILE = GOLD / "gold_v1.jsonl"
RESULTS = EVALS / "results"
USER = "eval.gold"
BRIEFING = {"request": "Alzheimer's disease", "answer": "Alzheimer's, service-line, no patient"}


def rows() -> list[dict]:
    return [json.loads(line) for line in GOLD_FILE.read_text().splitlines() if line.strip()]


def upsert_dataset(client):
    wanted = {r["id"]: {"inputs": r["inputs"], "outputs": r["outputs"], "split": r["metadata"]["segment"],
                        "metadata": {**r["metadata"], "case_id": r["id"],
                                     "source_of_truth": json.dumps(r["metadata"]["source_of_truth"])[:2000]}}
              for r in rows()}
    if client.has_dataset(dataset_name=DATASET):
        ds = client.read_dataset(dataset_name=DATASET)
    else:
        ds = client.create_dataset(DATASET, description="Gold reference answers v1 (synthetic, unreviewed): corpus Q/A with "
                                   "verbatim key facts, openFDA-verified Alzheimer's products and sponsors, guardrail refusals.")
    existing = {(e.metadata or {}).get("case_id"): e for e in client.list_examples(dataset_id=ds.id)}
    new = [w for cid, w in wanted.items() if cid not in existing]
    if new:
        client.create_examples(dataset_id=ds.id, examples=new)
    updated = removed = 0
    for cid, e in existing.items():
        w = wanted.get(cid)
        if w is None:
            client.delete_example(e.id)
            removed += 1
        elif e.inputs != w["inputs"] or (e.outputs or {}) != w["outputs"] or (e.metadata or {}).get("segment") != w["split"]:
            client.update_example(e.id, inputs=w["inputs"], outputs=w["outputs"], metadata=w["metadata"], split=w["split"])
            updated += 1
    counts: dict[str, int] = {}
    for w in wanted.values():
        counts[w["split"]] = counts.get(w["split"], 0) + 1
    print(f"dataset {DATASET}: {len(wanted)} examples {counts} ({len(new)} created, {updated} updated, {removed} removed)")
    return ds, counts


# ---------------- shared app state: one ingest, one briefing ----------------

class Shared:
    lock = threading.Lock()
    followup_lock = threading.Lock()
    ready = False
    error: str | None = None
    thread_id: str | None = None
    briefing: dict | None = None
    briefing_text = ""
    setup_seconds = 0.0

    @classmethod
    def ensure(cls, need_briefing: bool = True) -> None:
        with cls.lock:
            if cls.ready or cls.error:
                return
            t0 = time.time()
            try:
                from app import rag
                rag.ingest_corpus()
                if need_briefing:
                    from app.service import chat_turn, graph
                    first = chat_turn(USER, BRIEFING["request"])
                    cls.thread_id = first.get("thread_id")
                    if first.get("type") != "questions":
                        raise RuntimeError(f"expected clarifying questions, got {first.get('type')}")
                    second = chat_turn(USER, BRIEFING["answer"], cls.thread_id)
                    if second.get("type") != "approval":
                        raise RuntimeError(f"expected approval gate, got {second.get('type')}")
                    cls.briefing = graph().get_state({"configurable": {"thread_id": cls.thread_id}}).values.get("briefing")
                    cls.briefing_text = second.get("text", "")
                    done = chat_turn(USER, "", cls.thread_id, action="approve")
                    if done.get("type") != "done":
                        raise RuntimeError(f"approval did not finish the session: {done.get('type')}")
                cls.ready = True
            except Exception as exc:  # noqa: BLE001 - every dependent case reports the setup error
                cls.error = f"{type(exc).__name__}: {str(exc)[:300]}"
            cls.setup_seconds = round(time.time() - t0, 1)


def _claim_citations(claims: list[dict]) -> list[str]:
    return [c for cl in claims or [] for c in cl.get("citations", [])]


def followup_target(question: str) -> dict:
    from app.service import chat_turn, graph

    with Shared.followup_lock:  # turns on one session must not interleave
        t0 = time.time()
        reply = chat_turn(USER, question, Shared.thread_id)
        answer = graph().get_state({"configurable": {"thread_id": Shared.thread_id}}).values.get("followup_answer") or {}
    claims = answer.get("claims") or []
    lines = (reply.get("text") or "").splitlines()
    body = "\n".join(lines[1:] if lines and lines[0].startswith("**Follow-up:**") else lines)  # drop the echoed question
    return {"path": "followup", "reply_type": reply.get("type"), "answer_text": body,
            "claims_count": len(claims), "refused": bool(answer.get("refused")),
            "citations": sorted(set(_claim_citations(claims))),
            "tool_calls": [t.get("tool") for t in answer.get("tool_calls", [])], "seconds": round(time.time() - t0, 1)}


def rag_target(question: str) -> dict:
    from app import rag

    t0 = time.time()
    hits = rag.search(question)
    text = "\n".join(h["text"] for h in hits) or "No internal document passage cleared the relevance threshold: not in our documents."
    return {"path": "rag.search", "answer_text": text, "claims_count": len(hits), "refused": False,
            "citations": [h["id"] for h in hits], "seconds": round(time.time() - t0, 1)}


def briefing_target() -> dict:
    b = Shared.briefing or {}
    products = [{k: p.get(k) for k in ("generic_name", "brand_name", "sponsor", "application")} for p in b.get("approved_products", [])]
    return {"path": "briefing", "answer_text": Shared.briefing_text, "approved_products": products,
            "citations": sorted({s["id"] for s in b.get("sources", [])}), "claims_count": len(b.get("standard_of_care", [])),
            "setup_seconds": Shared.setup_seconds}


def make_target(rag_only: bool):
    def target(inputs: dict) -> dict:
        seg = inputs["segment"]
        if rag_only:
            Shared.ensure(need_briefing=False)
            return rag_target(inputs["question"]) if not Shared.error else {"error": Shared.error, "answer_text": ""}
        Shared.ensure()
        if Shared.error:
            return {"error": Shared.error, "answer_text": ""}
        if seg in ("corpus_qa", "guardrail_refusals"):
            return followup_target(inputs["question"])
        return briefing_target()
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="gold-baseline")
    parser.add_argument("--segment", action="append")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--rag-only", action="store_true")
    args = parser.parse_args()

    from langsmith import evaluate

    from app.config import EMBED_MODEL, MODEL, RERANK_ON, tracer

    t = tracer()
    if t is None:
        raise SystemExit("needs LANGSMITH_API_KEY (or the Keychain entry) and BRIEFING_TRACING on")
    client = t.client  # redacting client (hide_inputs/hide_outputs)
    ds, counts = upsert_dataset(client)
    ds_url = f"{client._host_url}/datasets/{ds.id}" if hasattr(client, "_host_url") else str(ds.id)
    print(f"dataset url: {ds_url}")
    if args.upload_only:
        return 0

    spot = ev.spot_check()
    print(f"judge spot-check: {spot['agreement']:.2f} agreement on {spot['pairs']} hand-labeled pairs "
          f"({[(r['label'], r['judge']) for r in spot['rows']]})")

    segments = set(args.segment or []) or ({"corpus_qa"} if args.rag_only else set(counts))
    order = {r["id"]: n for n, r in enumerate(rows())}
    examples = sorted((e for e in client.list_examples(dataset_id=ds.id) if (e.metadata or {}).get("segment") in segments),
                      key=lambda e: order.get((e.metadata or {}).get("case_id"), 999))
    t0 = time.time()
    experiment = evaluate(
        make_target(args.rag_only), data=examples, evaluators=ev.ROW_EVALUATORS,
        summary_evaluators=[ev.segment_means, ev.segment_pass_rate], experiment_prefix=args.experiment,
        max_concurrency=2, client=client,
        description="Gold reference answers v1: key-fact recall, doc citation, sponsor naming, refusals, identifiers, "
                    "OpenAI answer-correctness judge" + (" (rag.search only)" if args.rag_only else " (follow-up agent + briefing)"),
        metadata={"model": MODEL, "embed_model": EMBED_MODEL, "rerank": RERANK_ON, "judge": ev.JUDGE_MODEL,
                  "segments": sorted(segments), "path": "rag.search" if args.rag_only else "followup+briefing"},
    )
    per_case, scores, by_seg = [], {}, {}
    for row in experiment:
        run, ex = row["run"], row["example"]
        seg = (ex.metadata or {}).get("segment")
        res = {r.key: (r.score, r.comment) for r in row["evaluation_results"]["results"] if r.score is not None}
        for k, (s, _) in res.items():
            scores.setdefault(k, []).append(float(s))
            by_seg.setdefault(seg, {}).setdefault(k, []).append(float(s))
        out = run.outputs or {}
        per_case.append({"id": (ex.metadata or {}).get("case_id"), "segment": seg, "scores": {k: v[0] for k, v in res.items()},
                         "comments": {k: (v[1] or "")[:240] for k, v in res.items() if v[0] < 1},
                         "reply_type": out.get("reply_type"), "error": out.get("error") or (run.error and str(run.error)[:200]),
                         "seconds": out.get("seconds"), "tool_calls": out.get("tool_calls")})
    mean = lambda v: round(sum(v) / len(v), 3)  # noqa: E731
    summary = {"experiment": experiment.experiment_name, "url": experiment.url, "dataset": DATASET, "dataset_url": ds_url,
               "counts": counts, "segments_run": sorted(segments), "path": "rag.search" if args.rag_only else "followup+briefing",
               "wall_seconds": round(time.time() - t0, 1), "setup_seconds": Shared.setup_seconds, "setup_error": Shared.error,
               "overall": {k: mean(v) for k, v in sorted(scores.items())},
               "by_segment": {s: {k: mean(v) for k, v in sorted(d.items())} for s, d in sorted(by_seg.items())},
               "judge_spot_check": spot, "cases": per_case}
    print(f"\nexperiment {summary['experiment']}\n{summary['url']}")
    print(json.dumps({"overall": summary["overall"], "by_segment": summary["by_segment"]}, indent=1))
    for c in per_case:
        bad = {k: v for k, v in c["scores"].items() if v < 1}
        if bad or c["error"]:
            print(f"FAIL {c['id']}: {bad} {c['error'] or ''} {json.dumps(c['comments'])[:300]}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"gold-{args.experiment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    print(f"saved {out.relative_to(BUILD)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
