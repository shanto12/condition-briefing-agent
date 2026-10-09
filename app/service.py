"""One chat turn: start a briefing, answer clarifying questions, or record the analyst decision.
Shared by the FastAPI app and the eval runner. Each briefing runs on its own checkpointed thread."""
from __future__ import annotations

import sqlite3
import threading
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import CHECKPOINT_DB, tracer
from .graph import build_graph
from .guardrails import redact_text
from .render import briefing_markdown, questions_markdown

_graph = None
_lock = threading.Lock()


def graph():
    global _graph
    with _lock:
        if _graph is None:
            CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
            _graph = build_graph(SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)))
    return _graph


def _config(thread_id: str, user: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30, "run_name": "condition-briefing",
              "metadata": {"user": user, "thread_id": thread_id}, "tags": ["condition-briefing"]}
    t = tracer()
    if t:
        config["callbacks"] = [t]
    return config


def _pending(thread_id: str) -> tuple[dict | None, dict]:
    state = graph().get_state({"configurable": {"thread_id": thread_id}})
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value, state.values
    return None, state.values


def _respond(thread_id: str) -> dict:
    pending, values = _pending(thread_id)
    if pending and pending["kind"] == "questions":
        return {"type": "questions", "thread_id": thread_id, "text": questions_markdown(pending)}
    if pending and pending["kind"] == "approval":
        return {"type": "approval", "thread_id": thread_id, "briefing_id": values["briefing_id"],
                "text": briefing_markdown(values["briefing"], "draft"), "verification": values["briefing"]["verification"]}
    if values.get("status") == "refused":
        return {"type": "refusal", "thread_id": thread_id, "text": values.get("message", "")}
    return {"type": "done", "thread_id": thread_id, "status": values.get("status"),
            "briefing_id": values.get("briefing_id"), "text": values.get("message", "")}


def chat_turn(user: str, message: str = "", thread_id: str | None = None, action: str | None = None) -> dict:
    message = redact_text((message or "").strip())[:2000]
    if thread_id:
        pending, values = _pending(thread_id)
        if pending and pending["kind"] == "questions":
            if values.get("user") != user:
                return {"type": "error", "thread_id": thread_id, "text": "Only the requester can answer these questions."}
            graph().invoke(Command(resume={"text": message}), _config(thread_id, user))
            return _respond(thread_id)
        if pending and pending["kind"] == "approval":
            if action not in ("approve", "reject"):
                return {"type": "approval_pending", "thread_id": thread_id,
                        "text": "This briefing is waiting for analyst review. Use **Approve** or **Reject**."}
            graph().invoke(Command(resume={"action": action, "user": user}), _config(thread_id, user))
            return _respond(thread_id)
    if not message:
        return {"type": "error", "text": "Type a condition to brief on, for example: dementia."}
    thread_id = uuid.uuid4().hex
    graph().invoke({"user": user, "request": message, "events": []}, _config(thread_id, user))
    return _respond(thread_id)
