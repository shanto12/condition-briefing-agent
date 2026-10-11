"""SQLite store (audit log, briefings, sessions, messages) and de-identified cohort statistics."""
from __future__ import annotations

import sqlite3
import uuid

import pytest
from _support import NAMES, absent, log_outputs, ssn

from app import patients, store


@pytest.fixture
def fresh_db(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "APP_DB", tmp_path / "app.db")
    return tmp_path / "app.db"


# ---------------- audit log ----------------

@pytest.mark.parametrize("sql", ["UPDATE audit_log SET user = 'x'", "DELETE FROM audit_log",
                                 "UPDATE audit_log SET patient_ref = 'P999' WHERE id = 1"])
def test_audit_log_is_append_only(fresh_db, sql):
    store.audit("u.test", "P001", "read_patient_history", "test", None)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        with store.connect() as conn:
            conn.execute(sql)
    rows = store.list_audit()
    assert len(rows) == 1 and rows[0]["user"] == "u.test" and rows[0]["patient_ref"] == "P001"


def test_audit_insert_order_newest_first(fresh_db):
    for i in range(3):
        store.audit("u.test", f"P00{i + 1}", "read", "test", "t1")
    assert [r["patient_ref"] for r in store.list_audit(2)] == ["P003", "P002"]


# ---------------- briefings ----------------

def _briefing() -> dict:
    return {"condition": "dementia", "subtype": "All subtypes", "focus": "Service-line", "sources": [{"id": "PMID:1"}],
            "patient_context": {"patient_ref": "P001", "claims": [{"text": "Patient detail.", "citations": ["PATIENT:P001"]}]}}


def test_save_draft_never_persists_patient_context(fresh_db):
    briefing_id = store.save_draft(_briefing(), "thread-1", "u.test")
    record = store.get_briefing(briefing_id)
    assert record["briefing"]["patient_context"] is None and record["status"] == "draft"
    with store.connect() as conn:
        raw = conn.execute("SELECT briefing_json, sources_json FROM briefings").fetchone()
    absent("PATIENT:P001", raw[0], "patient citation in briefing_json")
    absent("Patient detail.", raw[0], "patient claim in briefing_json")


def test_set_status_and_listing(fresh_db):
    bid = store.save_draft(_briefing(), "thread-1", "u.test")
    store.set_status(bid, "final", "reviewer.b")
    record = store.get_briefing(bid)
    assert record["status"] == "final" and record["approved_by"] == "reviewer.b" and record["approved_at"]
    assert store.list_briefings()[0]["id"] == bid
    assert store.get_briefing("000000000000") is None


# ---------------- sessions + messages ----------------

def test_session_upsert_keeps_owner_and_unset_fields(fresh_db):
    tid = uuid.uuid4().hex
    store.upsert_session(tid, "alice", title="dementia", status="running")
    store.upsert_session(tid, "mallory", condition="dementia", status=None)
    s = store.get_session(tid)
    assert s["user"] == "alice", "a second upsert must not change the owner"
    assert (s["title"], s["condition"], s["status"]) == ("dementia", "dementia", "running")
    store.upsert_session(tid, "alice", status="awaiting_answers", bogus="ignored")
    assert store.get_session(tid)["status"] == "awaiting_answers"


def test_sessions_listed_per_user_only(fresh_db):
    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    store.upsert_session(a, "alice", title="A")
    store.upsert_session(b, "bob", title="B")
    assert [s["thread_id"] for s in store.list_sessions("alice")] == [a]
    assert [s["thread_id"] for s in store.list_sessions("bob")] == [b]
    assert store.list_sessions("carol") == []
    assert "user" not in store.list_sessions("alice")[0]


def test_messages_ordered_per_thread(fresh_db):
    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    store.add_message(a, "user", "message", "dementia")
    store.add_message(b, "user", "message", "other thread")
    store.add_message(a, "assistant", "questions", "Three quick questions", "https://trace.example/1")
    msgs = store.list_messages(a)
    assert [(m["role"], m["type"]) for m in msgs] == [("user", "message"), ("assistant", "questions")]
    assert msgs[1]["trace_url"] == "https://trace.example/1"
    assert store.get_session(uuid.uuid4().hex) is None


def test_schema_has_expected_tables(fresh_db):
    with store.connect() as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')")}
    assert {"briefings", "audit_log", "sessions", "messages", "audit_no_update", "audit_no_delete"} <= names


# ---------------- cohort ----------------

def test_cohort_small_cells_suppressed_at_default_k():
    stats = patients.cohort_stats("dementia", "All subtypes", "u.test", None)
    assert stats["min_cell"] == 3
    # 3 Alzheimer's, 1 vascular, 1 Lewy in the synthetic test panel; the heart-failure patient is excluded.
    assert stats["matched_patients"] == "5"
    assert stats["by_subtype"]["Alzheimer's disease dementia"] == "3"
    assert stats["by_subtype"]["Vascular dementia"] == "<3"
    assert stats["by_subtype"]["Dementia with Lewy bodies"] == "<3"
    assert stats["by_medication"]["donepezil"] == "3" and stats["by_medication"]["memantine"] == "<3"
    log_outputs({"cohort": stats})


def test_cohort_matched_total_suppressed_when_small():
    stats = patients.cohort_stats("dementia", "Lewy body dementia", "u.test", None)
    assert stats["matched_patients"] == "<3" and set(stats["by_subtype"].values()) == {"<3"}


def test_cohort_zero_match_reports_zero():
    stats = patients.cohort_stats("cystic fibrosis", "All subtypes", "u.test", None)
    assert stats["matched_patients"] == "0" and stats["by_subtype"] == {}


@pytest.mark.parametrize("k,expected", [(1, "3"), (3, "3"), (4, "<4"), (11, "<11")])
def test_cohort_threshold_follows_min_cell(monkeypatch, k, expected):
    monkeypatch.setattr(patients, "COHORT_MIN_CELL", k)
    stats = patients.cohort_stats("dementia", "All subtypes", "u.test", None)
    assert stats["by_subtype"]["Alzheimer's disease dementia"] == expected and stats["min_cell"] == k


def test_cohort_output_has_no_identifiers_and_is_audited():
    before = len(store.list_audit(1000))
    stats = patients.cohort_stats("dementia", "All subtypes", "u.test", "thread-x")
    for pid, name in NAMES.items():
        absent(name, stats, "name")
        absent(ssn(pid), stats, "ssn")
    rows = store.list_audit(1000)
    assert len(rows) == before + 1 and rows[0]["action"] == "cohort_aggregate" and rows[0]["patient_ref"] == "PANEL:*"
