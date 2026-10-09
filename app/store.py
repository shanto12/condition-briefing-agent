"""Briefing history (long-term), chat sessions with redacted transcripts, and append-only access audit log, all in SQLite."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone

from .config import APP_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS briefings (
    id TEXT PRIMARY KEY, thread_id TEXT, condition TEXT, subtype TEXT, focus TEXT,
    created_by TEXT, created_at TEXT, status TEXT, approved_by TEXT, approved_at TEXT,
    briefing_json TEXT, sources_json TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, user TEXT NOT NULL,
    patient_ref TEXT NOT NULL, action TEXT NOT NULL, purpose TEXT NOT NULL, thread_id TEXT
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TABLE IF NOT EXISTS sessions (
    thread_id TEXT PRIMARY KEY, user TEXT NOT NULL, title TEXT, condition TEXT, status TEXT,
    briefing_id TEXT, briefing_run_id TEXT, created_at TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions (user, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL, role TEXT NOT NULL, type TEXT,
    text TEXT, trace_url TEXT, ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_thread ON messages (thread_id, id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    APP_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(APP_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def audit(user: str, patient_ref: str, action: str, purpose: str, thread_id: str | None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, user, patient_ref, action, purpose, thread_id) VALUES (?, ?, ?, ?, ?, ?)",
            (now(), user, patient_ref, action, purpose, thread_id),
        )


def list_audit(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def population_view(b: dict) -> dict:
    """Drops every link to the patient (context, PATIENT: sources, patient IDs in events, removed patient claims)."""
    v = b.get("verification", {})
    return {**b, "patient_context": None,
            "sources": [s for s in b.get("sources", []) if s.get("type") != "patient"],
            "guardrail_events": [e for e in b.get("guardrail_events", []) if not re.search(r"\bP\d{3}\b", e)],
            "verification": {**v, "removed": [r for r in v.get("removed", []) if r.get("section") != "patient_context"]}}


def save_draft(briefing: dict, thread_id: str, user: str) -> str:
    """Stores the population-level briefing. Patient-specific context is never persisted here."""
    stored = population_view(briefing)
    briefing_id = uuid.uuid4().hex[:12]
    with connect() as conn:
        conn.execute(
            "INSERT INTO briefings (id, thread_id, condition, subtype, focus, created_by, created_at, status, "
            "briefing_json, sources_json) VALUES (?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)",
            (briefing_id, thread_id, briefing["condition"], briefing["subtype"], briefing["focus"], user, now(),
             json.dumps(stored), json.dumps(stored["sources"])),
        )
    return briefing_id


def set_status(briefing_id: str, status: str, user: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE briefings SET status = ?, approved_by = ?, approved_at = ? WHERE id = ?",
            (status, user, now(), briefing_id),
        )


def list_briefings(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, condition, subtype, focus, created_by, created_at, status, approved_by, approved_at "
            "FROM briefings ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_briefing(briefing_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM briefings WHERE id = ?", (briefing_id,)).fetchone()
    if not row:
        return None
    record = dict(row)
    record["briefing"] = json.loads(record.pop("briefing_json"))
    record.pop("sources_json", None)
    return record


def upsert_session(thread_id: str, user: str, **fields) -> None:
    """Creates the session on first turn; later turns only overwrite the fields that are not None."""
    fields = {k: v for k, v in fields.items() if v is not None and k in
              ("title", "condition", "status", "briefing_id", "briefing_run_id")}
    ts = now()
    with connect() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions (thread_id, user, created_at, updated_at) VALUES (?, ?, ?, ?)",
                     (thread_id, user, ts, ts))
        sets = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(f"UPDATE sessions SET {sets + ', ' if sets else ''}updated_at = ? WHERE thread_id = ?",
                     (*fields.values(), ts, thread_id))


def add_message(thread_id: str, role: str, type_: str, text: str, trace_url: str | None = None) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO messages (thread_id, role, type, text, trace_url, ts) VALUES (?, ?, ?, ?, ?, ?)",
                     (thread_id, role, type_, text, trace_url, now()))


def get_session(thread_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE thread_id = ?", (thread_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(user: str, limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, condition, status, briefing_id, created_at, updated_at FROM sessions "
            "WHERE user = ? ORDER BY updated_at DESC LIMIT ?", (user, limit)).fetchall()
    return [dict(r) for r in rows]


def list_messages(thread_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT role, type, text, trace_url, ts FROM messages WHERE thread_id = ? ORDER BY id",
                            (thread_id,)).fetchall()
    return [dict(r) for r in rows]
