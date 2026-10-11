"""FastAPI integration via TestClient: identity, input validation, session ownership, uploads and security headers."""
from __future__ import annotations

import uuid

import pytest
from _support import absent, log_outputs, ssn

ALICE = {"X-User": "alice.analyst"}
BOB = {"X-User": "bob.analyst"}


def _briefing_session(api, headers=ALICE, answer="all subtypes, service-line, none"):
    first = api.post("/api/chat", json={"message": "dementia"}, headers=headers).json()
    assert first["type"] == "questions", first.get("text")
    second = api.post("/api/chat", json={"message": answer, "thread_id": first["thread_id"]}, headers=headers).json()
    return first["thread_id"], second


# ---------------- identity + validation ----------------

@pytest.mark.parametrize("method,path", [("post", "/api/chat"), ("get", "/api/sessions"), ("get", "/api/briefings"),
                                         ("get", "/api/audit"), ("get", "/api/sessions/" + "a" * 32)])
@pytest.mark.parametrize("header", [None, "", "x", "bad user!", "a" * 65, "<script>"])
def test_missing_or_invalid_user_is_401(api, method, path, header):
    headers = {} if header is None else {"X-User": header}
    kwargs = {"json": {"message": "dementia"}} if method == "post" else {}
    r = getattr(api, method)(path, headers=headers, **kwargs)
    assert r.status_code == 401


@pytest.mark.parametrize("thread_id", ["not-hex", "../../etc/passwd", "A" * 32, "a" * 31, "a" * 33])
def test_thread_id_pattern_validated(api, thread_id):
    r = api.post("/api/chat", json={"message": "hi", "thread_id": thread_id}, headers=ALICE)
    assert r.status_code == 422


def test_message_length_limit(api):
    assert api.post("/api/chat", json={"message": "x" * 2001}, headers=ALICE).status_code == 422
    assert api.post("/api/chat", json={"action": "delete"}, headers=ALICE).status_code == 422


def test_empty_message_asks_for_condition(api):
    r = api.post("/api/chat", json={"message": "   "}, headers=ALICE)
    assert r.status_code == 200 and r.json()["type"] == "error"


# ---------------- end to end + ownership ----------------

def test_briefing_end_to_end_over_http(api):
    tid, second = _briefing_session(api)
    assert second["type"] == "approval" and second["status"] == "awaiting_approval"
    done = api.post("/api/chat", json={"thread_id": tid, "action": "approve"}, headers=ALICE).json()
    assert done["type"] == "done" and done["status"] == "done"
    record = api.get(f"/api/briefings/{second['briefing_id']}", headers=ALICE).json()
    assert record["status"] == "final" and record["approved_by"] == "alice.analyst"
    sessions = api.get("/api/sessions", headers=ALICE).json()
    assert any(s["thread_id"] == tid and s["status"] == "done" for s in sessions)
    log_outputs({"types": [second["type"], done["type"]], "status": record["status"]})


def test_clinical_request_refused_over_http(api):
    r = api.post("/api/chat", json={"message": "Should we start lecanemab for patient P003?"}, headers=ALICE).json()
    assert r["type"] == "refusal"


def test_session_view_is_owner_only(api):
    tid, _ = _briefing_session(api)
    mine = api.get(f"/api/sessions/{tid}", headers=ALICE)
    assert mine.status_code == 200 and mine.json()["pending"]["type"] == "approval"
    assert api.get(f"/api/sessions/{tid}", headers=BOB).status_code == 404
    assert api.get(f"/api/sessions/{uuid.uuid4().hex}", headers=ALICE).status_code == 404
    assert api.get("/api/sessions/not-a-thread", headers=ALICE).status_code == 404
    assert all(s["thread_id"] != tid for s in api.get("/api/sessions", headers=BOB).json())


def test_non_owner_cannot_act_on_session(api):
    tid, second = _briefing_session(api)
    r = api.post("/api/chat", json={"thread_id": tid, "action": "approve"}, headers=BOB)
    assert r.status_code == 404
    assert api.get(f"/api/briefings/{second['briefing_id']}", headers=ALICE).json()["status"] == "draft"


def test_approve_without_pending_approval(api):
    first = api.post("/api/chat", json={"message": "dementia"}, headers=ALICE).json()
    r = api.post("/api/chat", json={"thread_id": first["thread_id"], "action": "approve"}, headers=ALICE).json()
    assert r["type"] == "error" and api.get(f"/api/sessions/{first['thread_id']}", headers=ALICE).json()["status"] == "awaiting_answers"
    r = api.post("/api/chat", json={"thread_id": uuid.uuid4().hex, "action": "approve"}, headers=ALICE).json()
    assert r["type"] == "error"


def test_approval_gate_requires_explicit_action(api):
    tid, second = _briefing_session(api)
    r = api.post("/api/chat", json={"thread_id": tid, "message": "looks good"}, headers=ALICE).json()
    assert r["type"] == "approval_pending"
    assert api.get(f"/api/briefings/{second['briefing_id']}", headers=ALICE).json()["status"] == "draft"


def test_transcript_redacts_phi_and_omits_patient_context(api):
    raw = ssn("P004")
    first = api.post("/api/chat", json={"message": f"dementia briefing, my SSN is {raw}"}, headers=ALICE).json()
    tid = first["thread_id"]
    api.post("/api/chat", json={"message": "Alzheimer's, service-line, P001", "thread_id": tid}, headers=ALICE)
    from app import store

    stored = store.list_messages(tid)
    absent(raw, stored, "SSN in transcript")
    saved_assistant = [m["text"] for m in stored if m["role"] == "assistant" and m["type"] == "approval"]
    assert saved_assistant
    absent("de-identified patient has mild", saved_assistant, "patient claim in saved transcript")
    # The owner's live view re-renders the pending draft from the checkpoint, so it may show patient context.
    view = api.get(f"/api/sessions/{tid}", headers=ALICE).json()
    absent(raw, view, "SSN in session view")


# ---------------- uploads ----------------

def test_upload_type_size_and_empty_limits(api):
    files = lambda name, data: {"file": (name, data, "application/octet-stream")}  # noqa: E731
    assert api.post("/api/documents", files=files("run.exe", b"MZ"), headers=ALICE).status_code == 415
    assert api.post("/api/documents", files=files("big.md", b"a" * (2 * 1024 * 1024 + 1)), headers=ALICE).status_code == 413
    assert api.post("/api/documents", files=files("empty.md", b""), headers=ALICE).status_code == 400
    assert api.post("/api/documents", files=files("x.md", b"hello"), headers={}).status_code == 401


def test_upload_path_traversal_filename_is_flattened(api, monkeypatch):
    from app import rag

    seen = {}

    def capture(name, data, user):
        seen.update(name=name, user=user)
        return {"ok": True, "doc_id": "evil", "chunks": 1, "flags": []}

    monkeypatch.setattr(rag, "ingest_bytes", capture)
    r = api.post("/api/documents", files={"file": ("../../etc/evil.md", b"# Title\nSome text.", "text/markdown")},
                 headers=ALICE)
    assert r.status_code == 200 and seen == {"name": "evil.md", "user": "alice.analyst"}


def test_upload_with_phi_rejected_before_embedding(api):
    from _support import FakeEmbeddings

    before = FakeEmbeddings.calls
    body = f"# Memo\nFollow up with patient, SSN {ssn('P005')}.".encode()
    r = api.post("/api/documents", files={"file": ("memo.md", body, "text/markdown")}, headers=ALICE)
    assert r.status_code == 422 and "PHI" in r.json()["detail"]
    assert FakeEmbeddings.calls == before


@pytest.mark.skip(reason="MVP cut: injection-flag upload needs a Chroma index round-trip; covered by unit test of "
                         "neutralize_injection and by evals/run_eval.py")
def test_upload_injection_flagged(api):
    pass


# ---------------- security headers ----------------

@pytest.mark.parametrize("path,headers", [("/healthz", {}), ("/api/sessions", {}), ("/api/sessions", ALICE)])
def test_security_headers_on_every_response(api, path, headers):
    r = api.get(path, headers=headers)
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "script-src 'self'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer" and r.headers["cache-control"] == "no-store"
