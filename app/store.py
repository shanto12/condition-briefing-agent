"""Briefing history (long-term) and append-only access audit log, both in SQLite."""
from __future__ import annotations

import json
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


def save_draft(briefing: dict, thread_id: str, user: str) -> str:
    """Stores the population-level briefing. Patient-specific context is never persisted here."""
    stored = {**briefing, "patient_context": None}
    briefing_id = uuid.uuid4().hex[:12]
    with connect() as conn:
        conn.execute(
            "INSERT INTO briefings (id, thread_id, condition, subtype, focus, created_by, created_at, status, "
            "briefing_json, sources_json) VALUES (?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)",
            (briefing_id, thread_id, briefing["condition"], briefing["subtype"], briefing["focus"], user, now(),
             json.dumps(stored), json.dumps(briefing.get("sources", []))),
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
