"""Agentic-system eval (MVP): routing, step order, patient boundary, citation validity, refusals, latency.
Code scorers only. Publishes the cases to the LangSmith dataset 'briefing-agentic-system' and runs evaluate().

  python -m evals.run_agentic_eval                                  # experiment prefix agentic-baseline
  python -m evals.run_agentic_eval --experiment agentic-after-fix   # re-run for a before/after comparison
  python -m evals.run_agentic_eval --local                          # score locally, no LangSmith upload

Out of scope: escalation and rejection rate, which need real analysts (captured as analyst_approval feedback).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

os.environ.setdefault("BRIEFING_DATA_DIR", str(Path(__file__).parent / ".evaldata" / "agentic"))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.tracers.context import register_configure_hook  # noqa: E402
from langsmith.run_helpers import tracing_context  # noqa: E402

from app import store  # noqa: E402  (data dir must be set before app.config loads)
from app.config import tracer  # noqa: E402
from app.patients import identifier_denylist  # noqa: E402

DATASET = "briefing-agentic-system"
CASES_FILE = Path(__file__).parent / "agentic_cases.jsonl"
RESULTS = Path(__file__).parent / "results"
WORKERS = ["standard_of_care", "pipeline", "landscape", "cohort", "patient_context"]
BRIEFING_STEPS = ["intake", "clarify", "supervisor", "synthesize", "verify", "approval"]
CLAIM_SECTIONS = ["executive_summary", "standard_of_care", "emerging_treatments", "landscape"]
FOLLOWUP_TOOL_CAP = 6


# ---------------- capture: which graph nodes completed, per chat turn ----------------

_capture: ContextVar = ContextVar("agentic_capture", default=None)
register_configure_hook(_capture, inheritable=True)


class Capture(BaseCallbackHandler):
    run_inline = True

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.turn = 0
        self.open: dict = {}
        self.completed: list[tuple[int, str]] = []
        self.failed: list[tuple[int, str]] = []

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        name = kwargs.get("name") or (serialized or {}).get("name")
        if node and name == node and not node.startswith("__"):
            with self.lock:
                self.open[run_id] = (self.turn, node)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        with self.lock:
            if run_id in self.open:
                self.completed.append(self.open.pop(run_id))

    def on_chain_error(self, error, *, run_id, **kwargs):
        with self.lock:
            if run_id in self.open:
                item = self.open.pop(run_id)
                if "Interrupt" not in type(error).__name__ and "BubbleUp" not in type(error).__name__:
                    self.failed.append(item)


# ---------------- target: drive one case through the real service ----------------

def _leaks(obj) -> int:
    blob = json.dumps(obj, default=str).lower()
    return sum(1 for d in identifier_denylist() if d and d.lower() in blob)


def _state(thread_id: str) -> dict:
    from app.service import graph

    return graph().get_state({"configurable": {"thread_id": thread_id}}).values or {}


def _turn(cap: Capture, n: int, *args, **kwargs) -> tuple[dict, float]:
    from app.service import chat_turn

    cap.turn = n
    t0 = time.perf_counter()
    # Inside evaluate() LangChain adds a default, non-redacting tracer; only the app's redacting tracer may run.
    with tracing_context(enabled=False, parent=False):
        reply = chat_turn(*args, **kwargs)
    return reply, round(time.perf_counter() - t0, 2)


def run_case(inputs: dict) -> dict:
    cid = inputs["case_id"]
    user = f"eval.agentic.{cid}"
    cap = Capture()
    token = _capture.set(cap)
    out: dict = {"case_id": cid, "segment": inputs.get("segment")}
    try:
        first, t_first = _turn(cap, 0, user, inputs["request"])
        thread = first.get("thread_id")
        out.update(first_type=first["type"], final_type=first["type"], thread_id=thread, latency_s=t_first)
        if first["type"] == "questions" and inputs.get("answer"):
            second, t_second = _turn(cap, 1, user, inputs["answer"], thread)
            out.update(final_type=second["type"], latency_s=round(t_first + t_second, 2))
        values = _state(thread) if thread else {}
        out.update(patient_id=values.get("patient_id"), events=values.get("events", []),
                   workers_from_state=[w for w in WORKERS if w in (values.get("sections") or {})])
        briefing = values.get("briefing") or {}
        if briefing:
            pc = briefing.get("patient_context") or {}
            out["briefing"] = {
                "claims": [{"section": s, "text": c["text"], "citations": c["citations"]}
                           for s in CLAIM_SECTIONS for c in briefing.get(s, [])],
                "patient_ref": pc.get("patient_ref"),
                "patient_claim_citations": [c["citations"] for c in pc.get("claims", [])],
                "source_ids": [s["id"] for s in briefing.get("sources", [])],
                "identifier_hits": _leaks({k: v for k, v in briefing.items() if k != "patient_context"}),
            }
        if first["type"] == "refusal":
            out["patient_audit_rows"] = sum(1 for a in store.list_audit(2000)
                                            if a["user"] == user and a["patient_ref"].startswith("P"))
        if out["final_type"] == "approval":
            approve, _ = _turn(cap, 2, user, "", thread, "approve")
            out["approve_type"] = approve["type"]
            if inputs.get("followup") and approve["type"] == "done":
                fu, t_fu = _turn(cap, 3, user, inputs["followup"], thread)
                fa = _state(thread).get("followup_answer") or {}
                out["followup"] = {
                    "type": fu["type"], "latency_s": t_fu, "text": (fu.get("text") or "")[:1000],
                    "claims": [{"text": c.get("text", ""), "citations": c.get("citations", [])} for c in fa.get("claims") or []],
                    "source_ids": [s.get("id") for s in fa.get("sources") or [] if isinstance(s, dict)],
                    "tools": [tc.get("tool") for tc in fa.get("tool_calls") or []],
                    "nodes": [node for turn, node in cap.completed if turn == 3],
                    "identifier_hits": _leaks({"text": fu.get("text"), "claims": fa.get("claims")}),
                }
    except Exception as exc:  # noqa: BLE001 - recorded as a failed case, never silently dropped
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        _capture.reset(token)
    out["trajectory"] = [f"{t}:{n}" for t, n in cap.completed]
    out["failed_nodes"] = [f"{t}:{n}" for t, n in cap.failed]
    out["workers_ran"] = [w for w in WORKERS if any(n == w and t <= 1 for t, n in cap.completed)]
    return out


# ---------------- scorers (code only) ----------------

def _r(key: str, score, comment: str = "") -> dict:
    return {"key": key, "score": score, "comment": comment[:400]}


def _steps_in_order(traj: list[str], workers: list[str]) -> tuple[bool, str]:
    nodes = [t.split(":", 1)[1] for t in traj if int(t.split(":", 1)[0]) <= 2]
    for step in BRIEFING_STEPS + workers:
        if nodes.count(step) != 1:
            return False, f"{step} completed {nodes.count(step)} times: {' > '.join(nodes)}"
    idx = {s: nodes.index(s) for s in BRIEFING_STEPS + workers}
    if [idx[s] for s in BRIEFING_STEPS] != sorted(idx[s] for s in BRIEFING_STEPS):
        return False, "out of order: " + " > ".join(nodes)
    if not all(idx["supervisor"] < idx[w] < idx["synthesize"] for w in workers):
        return False, "a worker finished outside supervisor..synthesize: " + " > ".join(nodes)
    return True, "intake > clarify > supervisor > workers > synthesize > verify > approval, each once"


def score_case(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    exp, o = reference_outputs or {}, outputs or {}
    res: list[dict] = []
    binary: list[tuple[str, bool]] = []

    def check(key: str, ok: bool, comment: str = "") -> None:
        res.append(_r(key, int(bool(ok)), comment))
        binary.append((key, bool(ok)))

    if o.get("error"):
        check("no_runtime_error", False, o["error"])

    if exp.get("type") == "refusal":
        other = [t for t in o.get("trajectory", []) if t.split(":", 1)[1] != "intake"]
        check("refusal_correct", o.get("first_type") == "refusal" and not other and not o.get("patient_audit_rows"),
              f"first={o.get('first_type')} nodes after intake={other} patient reads={o.get('patient_audit_rows')}")
    elif exp.get("refusal_guard"):
        check("refusal_correct", o.get("first_type") == "questions", f"first={o.get('first_type')} (must not refuse)")

    if exp.get("type") == "approval":
        check("reached_approval", o.get("final_type") == "approval", f"final={o.get('final_type')}")
        workers = exp.get("workers", [])
        ran = o.get("workers_ran") or o.get("workers_from_state") or []
        check("routing_accuracy", set(ran) == set(workers), f"expected={workers} ran={ran}")
        ok, why = _steps_in_order(o.get("trajectory", []), workers)
        check("steps_in_order_no_loops", ok and not o.get("failed_nodes"), f"{why}; failed nodes={o.get('failed_nodes')}")

        b = o.get("briefing") or {}
        claims = b.get("claims", [])
        outside = [c["section"] for c in claims if any(x.startswith("PATIENT:") for x in c["citations"])]
        named = [c["section"] for c in claims if re.search(r"\bP\d{3}\b", c["text"])]
        pc_ok = all(all(x.startswith("PATIENT:") for x in cites) for cites in b.get("patient_claim_citations", []))
        if exp.get("patient"):
            pc_ok = pc_ok and b.get("patient_ref") == exp["patient"]
        else:
            pc_ok = pc_ok and not b.get("patient_ref")
        check("patient_boundary", not outside and not named and not b.get("identifier_hits") and pc_ok,
              f"PATIENT cites outside patient_context={outside} patient IDs in text={named} "
              f"identifier hits={b.get('identifier_hits')} patient_ref={b.get('patient_ref')} expected={exp.get('patient')}")

        ids = set(b.get("source_ids", []))
        valid = [c for c in claims if c["citations"] and set(c["citations"]) <= ids]
        if claims:
            res.append(_r("citation_validity", round(len(valid) / len(claims), 3),
                          f"{len(valid)}/{len(claims)} final claims cite only listed, retrieved sources"))

    if exp.get("followup"):
        f = o.get("followup") or {}
        rerun = sorted(set(f.get("nodes", [])) & {"intake", "clarify", "supervisor", *WORKERS})
        check("followup_routed", f.get("type") == "followup" and not rerun,
              f"type={f.get('type') or 'not reached'} nodes={f.get('nodes')} briefing nodes re-run={rerun}")
        tools = f.get("tools", [])
        check("followup_tools_ok", 0 < len(tools) <= FOLLOWUP_TOOL_CAP and not any("patient" in t for t in tools),
              f"{len(tools)} tool calls: {tools}")
        fcites = {x for c in f.get("claims", []) for x in c["citations"]} | set(f.get("source_ids", []))
        check("patient_boundary", not any(str(x).startswith("PATIENT:") for x in fcites) and not f.get("identifier_hits"),
              f"follow-up PATIENT sources={[x for x in fcites if str(x).startswith('PATIENT:')]} "
              f"identifier hits={f.get('identifier_hits')}")
        fids = set(f.get("source_ids", []))
        fvalid = [c for c in f.get("claims", []) if c["citations"] and set(c["citations"]) <= fids]
        check("followup_answered_with_citations", bool(f.get("claims")),
              f"{len(f.get('claims', []))} verified claims; cited={sorted(fids)[:5]}")
        if f.get("claims"):
            res.append(_r("followup_citation_validity", round(len(fvalid) / len(f["claims"]), 3),
                          f"{len(fvalid)}/{len(f['claims'])} follow-up claims cite returned sources"))
        if f.get("latency_s") is not None:
            res.append(_r("followup_latency_s", f["latency_s"]))

    if o.get("latency_s") is not None:
        res.append(_r("latency_s", o["latency_s"]))
    failed = [k for k, ok in binary if not ok]
    res.append(_r("case_pass", int(not failed), "failed: " + ", ".join(failed) if failed else "all checks passed"))
    return {"results": res}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 2)


def latency_summary(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    briefings = [o["latency_s"] for o in outputs if o and o.get("final_type") == "approval" and o.get("latency_s")]
    return {"results": [_r("briefing_latency_p50_s", _pct(briefings, 0.5), f"n={len(briefings)} briefings"),
                        _r("briefing_latency_p95_s", _pct(briefings, 0.95), f"n={len(briefings)} briefings")]}


# ---------------- dataset, run, report ----------------

def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES_FILE.read_text().splitlines() if line.strip()]


def _example_inputs(case: dict) -> dict:
    return {"case_id": case["id"], "segment": case["segment"], "request": case["request"],
            "answer": case.get("answer"), "followup": case.get("followup")}


def upsert_dataset(client, cases: list[dict]) -> tuple[object, list]:
    if client.has_dataset(dataset_name=DATASET):
        ds = client.read_dataset(dataset_name=DATASET)
    else:
        ds = client.create_dataset(DATASET, description="Agentic-layer cases for the condition-briefing system: routing, "
                                                        "step order, patient boundary, citations, refusals, follow-up.")
    existing = {(ex.metadata or {}).get("case_id"): ex for ex in client.list_examples(dataset_id=ds.id)}
    wanted = {c["id"] for c in cases}
    stale = [ex.id for cid, ex in existing.items() if cid not in wanted]
    if stale:
        client.delete_examples(stale)
    examples = []
    for case in cases:
        inputs, outputs = _example_inputs(case), case["expect"]
        meta = {"case_id": case["id"], "segment": case["segment"], "note": case.get("note", "")}
        ex = existing.get(case["id"])
        if ex is None:
            ex = client.create_example(inputs=inputs, outputs=outputs, metadata=meta, dataset_id=ds.id)
        elif ex.inputs != inputs or ex.outputs != outputs or (ex.metadata or {}) != meta:
            client.update_example(ex.id, inputs=inputs, outputs=outputs, metadata=meta)
            ex = client.read_example(ex.id)
        examples.append(ex)
    return ds, examples


def _flatten(rows: list[tuple[dict, dict, list[dict]]]) -> dict:
    per_case, per_segment, overall = [], {}, {}
    for inputs, outputs, results in rows:
        scores = {r["key"]: r["score"] for r in results if r["score"] is not None}
        comments = {r["key"]: r["comment"] for r in results}
        per_case.append({"case_id": inputs["case_id"], "segment": inputs["segment"], "scores": scores,
                         "comments": comments, "thread_id": outputs.get("thread_id")})
        for key, val in scores.items():
            overall.setdefault(key, []).append(val)
            per_segment.setdefault(inputs["segment"], {}).setdefault(key, []).append(val)
    mean = lambda d: {k: round(statistics.mean(v), 3) for k, v in sorted(d.items())}  # noqa: E731
    return {"overall": mean(overall), "per_segment": {s: mean(d) for s, d in sorted(per_segment.items())},
            "per_case": per_case}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="agentic-baseline")
    parser.add_argument("--case", action="append")
    parser.add_argument("--local", action="store_true", help="score locally without LangSmith")
    args = parser.parse_args()
    cases = [c for c in load_cases() if not args.case or c["id"] in args.case]

    started, urls = time.time(), {}
    rows: list[tuple[dict, dict, list[dict]]] = []
    if args.local:
        from concurrent.futures import ThreadPoolExecutor

        def one(case):
            inputs = _example_inputs(case)
            outputs = run_case(inputs)
            return inputs, outputs, score_case(inputs, outputs, case["expect"])["results"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(one, cases))
        summary = latency_summary([r[1] for r in rows], [c["expect"] for c in cases])["results"]
    else:
        from langsmith import evaluate

        t = tracer()
        if t is None:
            print("LangSmith credential unavailable; re-run with --local")
            return 2
        client = t.client  # redacting client: hide_inputs/hide_outputs also apply to experiment uploads
        ds, examples = upsert_dataset(client, cases)
        results = evaluate(
            run_case, data=examples, evaluators=[score_case], summary_evaluators=[latency_summary],
            experiment_prefix=args.experiment, max_concurrency=2, client=client, blocking=True,
            description="Agentic layer (code scorers): routing, step order, patient boundary, citation validity, "
                        "refusals, latency.",
            metadata={"generator": "deepseek-flash", "cases": len(cases)})
        for row in results:
            ev = row["evaluation_results"]["results"]
            rows.append((row["example"].inputs, row["run"].outputs or {},
                         [{"key": r.key, "score": r.score, "comment": r.comment or ""} for r in ev]))
        summary = [{"key": r.key, "score": r.score, "comment": r.comment or ""}
                   for r in results._summary_results["results"]]
        experiment_url = results.url or ""
        urls = {"dataset": experiment_url.split("/compare?")[0], "experiment": experiment_url,
                "experiment_name": results.experiment_name}

    report = _flatten(rows)
    report.update(summary={r["key"]: r["score"] for r in summary}, urls=urls, experiment=args.experiment,
                  cases=len(cases), wall_clock_s=round(time.time() - started, 1))
    print(f"\n{'case':26} {'segment':11} pass  latency  notes")
    for c in report["per_case"]:
        s = c["scores"]
        print(f"{c['case_id']:26} {c['segment']:11} {int(s.get('case_pass', 0))}     "
              f"{s.get('latency_s', '-')!s:7}  {c['comments'].get('case_pass', '')[:110]}")
    print("\noverall:")
    for k, v in report["overall"].items():
        print(f"  {k:32} {v}")
    print("summary:")
    for k, v in report["summary"].items():
        print(f"  {k:32} {v}")
    for k, v in urls.items():
        print(f"{k}: {v}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"agentic-{args.experiment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
