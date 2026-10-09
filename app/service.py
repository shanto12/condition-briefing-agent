"""One chat turn: start a briefing, answer clarifying questions, record the analyst decision, or ask a follow-up.
Shared by the FastAPI app and the eval runner. Each session is one checkpointed thread, so it survives restarts."""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
import uuid
from functools import lru_cache
from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import store
from .config import CHECKPOINT_DB, EMBED_MODEL, LANGSMITH_PROJECT, tracer
from .graph import build_graph
from .guardrails import redact_text
from .render import briefing_markdown, questions_markdown

log = logging.getLogger("briefing")
_graph = None
_lock = threading.Lock()
STATUS = {"questions": "awaiting_answers", "approval": "awaiting_approval", "final": "done",
          "rejected": "rejected", "refused": "refused"}


def graph():
    global _graph
    with _lock:
        if _graph is None:
            CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
            _graph = build_graph(SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)))
    return _graph


def user_hash(user: str) -> str:
    return hashlib.sha256(user.encode()).hexdigest()[:16]


def _config(thread_id: str, user: str, run_id: uuid.UUID, values: dict | None = None) -> dict:
    values = values or {}
    metadata = {"thread_id": thread_id, "user_hash": user_hash(user), "embed_model": EMBED_MODEL,
                "condition": values.get("condition"), "subtype": values.get("subtype"),
                "briefing_id": values.get("briefing_id")}
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30, "run_name": "condition-briefing",
              "run_id": run_id, "metadata": {k: v for k, v in metadata.items() if v}, "tags": ["condition-briefing"]}
    t = tracer()
    if t:
        config["callbacks"] = [RootRunTags(t), t]
    return config


@lru_cache
def _project_id():
    return tracer().client.read_project(project_name=LANGSMITH_PROJECT).id


def trace_url(run_id) -> str | None:
    if not run_id or not tracer():
        return None
    try:
        return tracer().client.get_run_url(run=SimpleNamespace(id=run_id), project_id=_project_id())
    except Exception:  # noqa: BLE001 - a missing link must never fail the chat turn
        log.warning("trace url unavailable", exc_info=True)
        return None


class RootRunTags(BaseCallbackHandler):
    """LangSmith accepts a single final update per run, so tags known only at the end of the run are set on the
    tracer's root run here, before the tracer's own on_chain_end sends that update. Must precede the tracer."""

    def __init__(self, langchain_tracer):
        self.tracer = langchain_tracer

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        run = self.tracer.run_map.get(str(run_id))
        if parent_run_id is not None or run is None or not isinstance(outputs, dict):
            return
        tags = {"has_patient"} if outputs.get("patient_id") else set()
        if any("cached fixture" in e for e in outputs.get("events") or []):
            tags.add("stale_source")
        run.tags = sorted(set(run.tags or []) | tags)
        run.extra.setdefault("metadata", {}).update(
            {k: outputs[k] for k in ("condition", "subtype", "briefing_id") if outputs.get(k)})


def _invoke(payload, thread_id: str, user: str, values: dict | None = None) -> uuid.UUID:
    run_id = uuid.uuid4()
    graph().invoke(payload, _config(thread_id, user, run_id, values))
    return run_id


def _feedback(run_id: str | None, action: str, user: str) -> None:
    t = tracer()
    if not t or not run_id:
        return
    try:
        t.client.create_feedback(run_id=run_id, key="analyst_approval", score=1 if action == "approve" else 0,
                                 comment=f"analyst {action}", source_info={"user_hash": user_hash(user)},
                                 session_id=_project_id(), stop_after_attempt=3)
    except Exception:  # noqa: BLE001 - feedback is best effort; the decision is already recorded in SQLite
        log.warning("feedback failed", exc_info=True)


def _pending(thread_id: str) -> tuple[dict | None, dict]:
    state = graph().get_state({"configurable": {"thread_id": thread_id}})
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value, state.values
    return None, state.values


def _status(pending: dict | None, values: dict) -> str | None:
    return STATUS.get(pending["kind"]) if pending else STATUS.get(values.get("status"), values.get("status"))


def _respond(thread_id: str) -> dict:
    pending, values = _pending(thread_id)
    status = _status(pending, values)
    if pending and pending["kind"] == "questions":
        return {"type": "questions", "thread_id": thread_id, "status": status, "text": questions_markdown(pending)}
    if pending and pending["kind"] == "approval":
        return {"type": "approval", "thread_id": thread_id, "status": status, "briefing_id": values["briefing_id"],
                "text": briefing_markdown(values["briefing"], "draft"), "verification": values["briefing"]["verification"]}
    if values.get("status") == "refused":
        return {"type": "refusal", "thread_id": thread_id, "status": status, "text": values.get("message", "")}
    return {"type": "done", "thread_id": thread_id, "status": status,
            "briefing_id": values.get("briefing_id"), "text": values.get("message", "")}


def _transcript_text(reply: dict, thread_id: str) -> str:
    """Patient context stays in the checkpoint only; the saved transcript gets the population-level briefing."""
    if reply["type"] != "approval":
        return reply["text"]
    _, values = _pending(thread_id)
    return briefing_markdown(store.population_view(values["briefing"]), "draft")


def _record(thread_id: str, user: str, user_type: str, user_text: str, reply: dict, run_id=None) -> dict:
    reply["trace_url"] = trace_url(run_id)
    _, values = _pending(thread_id)
    fields = {"condition": values.get("condition"), "status": reply.get("status"), "briefing_id": values.get("briefing_id")}
    if reply["type"] == "approval" and run_id:
        fields["briefing_run_id"] = str(run_id)
    store.upsert_session(thread_id, user, **fields)
    if user_text:
        store.add_message(thread_id, "user", user_type, re.sub(r"\bP\d{3}\b", "[patient]", user_text))
    store.add_message(thread_id, "assistant", reply["type"], _transcript_text(reply, thread_id), reply["trace_url"])
    return reply


def _followup(thread_id: str, user: str, message: str, values: dict) -> dict:
    if "followup" not in graph().channels:
        reply = {"type": "followup_unavailable", "thread_id": thread_id, "status": _status(None, values),
                 "text": "Follow-up questions are not available yet. Start a **New briefing** to brief on another condition."}
        return _record(thread_id, user, "followup", message, reply)
    run_id = _invoke({"followup": message, "user": user}, thread_id, user, values)
    _, after = _pending(thread_id)
    answer = after.get("followup_answer") or {}
    if not answer.get("text"):
        reply = {"type": "error", "thread_id": thread_id, "status": _status(None, after),
                 "text": "The follow-up run returned no answer. Check the trace and retry."}
    else:
        reply = {"type": "followup", "thread_id": thread_id, "status": _status(None, after), "text": answer["text"],
                 "claims": answer.get("claims", []), "removed": answer.get("removed", []),
                 "sources": answer.get("sources", [])}
    return _record(thread_id, user, "followup", message, reply, run_id)


def chat_turn(user: str, message: str = "", thread_id: str | None = None, action: str | None = None) -> dict:
    message = redact_text((message or "").strip())[:2000]
    if thread_id:
        pending, values = _pending(thread_id)
        owner = (store.get_session(thread_id) or {}).get("user") or values.get("user")
        if owner and owner != user:
            return {"type": "error", "thread_id": thread_id, "text": "Session not found."}
        if pending and pending["kind"] == "questions":
            if not message:
                return {"type": "error", "thread_id": thread_id, "text": "Reply to the questions above to continue."}
            run_id = _invoke(Command(resume={"text": message}), thread_id, user, values)
            return _record(thread_id, user, "answer", message, _respond(thread_id), run_id)
        if pending and pending["kind"] == "approval":
            if action not in ("approve", "reject"):
                reply = {"type": "approval_pending", "thread_id": thread_id, "status": "awaiting_approval",
                         "text": "This briefing is waiting for analyst review. Use **Approve** or **Reject**."}
                return _record(thread_id, user, "message", message, reply)
            run_id = _invoke(Command(resume={"action": action, "user": user}), thread_id, user, values)
            _feedback((store.get_session(thread_id) or {}).get("briefing_run_id"), action, user)
            return _record(thread_id, user, "action", action.capitalize(), _respond(thread_id), run_id)
        if values.get("briefing") and message:
            return _followup(thread_id, user, message, values)
    if not message:
        return {"type": "error", "text": "Type a condition to brief on, for example: dementia."}
    thread_id = uuid.uuid4().hex
    store.upsert_session(thread_id, user, title=message[:80], status="running")
    try:
        run_id = _invoke({"user": user, "request": message, "events": []}, thread_id, user)
    except Exception:
        store.upsert_session(thread_id, user, status="failed")
        store.add_message(thread_id, "user", "message", message)
        raise
    return _record(thread_id, user, "message", message, _respond(thread_id), run_id)


def session_view(thread_id: str, user: str) -> dict | None:
    session = store.get_session(thread_id)
    if not session or session["user"] != user:
        return None
    messages = store.list_messages(thread_id)
    pending, values = _pending(thread_id)
    prompt = None
    if pending:
        live = _respond(thread_id)
        prompt = {"type": live["type"], "text": live["text"], "briefing_id": live.get("briefing_id")}
        for m in reversed(messages):
            if m["role"] == "assistant" and m["type"] == live["type"]:
                m["text"] = live["text"]
                break
    return {"thread_id": thread_id, "title": session["title"], "condition": session["condition"],
            "status": _status(pending, values) or session["status"], "briefing_id": session["briefing_id"],
            "created_at": session["created_at"], "updated_at": session["updated_at"],
            "can_followup": bool(values.get("briefing")) and not pending,
            "pending": prompt, "messages": messages}
