"""Public-hosting guards: demo passphrase gate, per-IP/daily rate limits, CORS. chat_turn is mocked; no LLM."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app import gate

USER = {"X-User": "alice.analyst"}
KEY = "open-sesame"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("DEMO_PASSPHRASE", raising=False)
    gate.reset()
    yield
    gate.reset()


@pytest.fixture
def client(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "chat_turn", lambda *a, **k: {"type": "answer", "text": "ok"})
    with TestClient(main.app) as c:
        yield c


def chat(client, headers=None, **kw):
    return client.post("/api/chat", json={"message": "hi"}, headers={**USER, **(headers or {})}, **kw)


def test_gate_off_by_default(client):
    assert chat(client).status_code == 200


def test_gate_on_requires_key(client, monkeypatch):
    monkeypatch.setenv("DEMO_PASSPHRASE", KEY)
    for headers in (None, {"X-Demo-Key": "wrong"}, {"X-Demo-Key": ""}):
        r = chat(client, headers)
        assert r.status_code == 401 and r.json() == {"detail": "demo_key_required"}
    assert client.get("/api/sessions", headers=USER).status_code == 401
    assert chat(client, {"X-Demo-Key": KEY}).status_code == 200
    assert client.get("/api/sessions", headers={**USER, "X-Demo-Key": KEY}).status_code == 200


def test_gate_leaves_health_and_index_open(client, monkeypatch):
    monkeypatch.setenv("DEMO_PASSPHRASE", KEY)
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code in (200, 404)  # not gated either way


def test_chat_rate_limit_per_ip(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_HOUR", "2")
    a = {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}
    assert chat(client, a).status_code == 200
    assert chat(client, a).status_code == 200
    r = chat(client, a)
    assert r.status_code == 429 and r.json() == {"detail": "rate_limited"}
    assert int(r.headers["Retry-After"]) > 0
    assert chat(client, {"X-Forwarded-For": "2.2.2.2"}).status_code == 200  # other IP unaffected


def test_bad_key_probes_do_not_burn_quota(client, monkeypatch):
    monkeypatch.setenv("DEMO_PASSPHRASE", KEY)
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_HOUR", "1")
    for _ in range(5):
        assert chat(client, {"X-Demo-Key": "bad"}).status_code == 401
    assert chat(client, {"X-Demo-Key": KEY}).status_code == 200
    assert chat(client, {"X-Demo-Key": KEY}).status_code == 429


def test_daily_cap(client, monkeypatch):
    monkeypatch.setenv("DAILY_CHAT_CAP", "2")
    for ip in ("3.3.3.1", "3.3.3.2"):
        assert chat(client, {"X-Forwarded-For": ip}).status_code == 200
    r = chat(client, {"X-Forwarded-For": "3.3.3.3"})
    assert r.status_code == 429 and r.json() == {"detail": "rate_limited"} and "Retry-After" in r.headers


def test_upload_rate_limit(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_UPLOAD_PER_HOUR", "1")
    files = {"file": ("note.exe", b"x", "application/octet-stream")}
    assert client.post("/api/documents", files=files, headers=USER).status_code == 415  # passed limiter, rejected by type
    r = client.post("/api/documents", files=files, headers=USER)
    assert r.status_code == 429 and r.json() == {"detail": "rate_limited"}


@pytest.fixture
def cors_client(monkeypatch):
    """CORS origins are read at import, so rebuild app.main under the env and restore it afterwards."""
    import app.main as main

    monkeypatch.setenv("CORS_ORIGINS", "https://demo.netlify.app/, https://other.example")
    monkeypatch.setenv("DEMO_PASSPHRASE", KEY)
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c
    monkeypatch.undo()
    importlib.reload(main)


def test_cors_preflight_bypasses_gate(cors_client):
    r = cors_client.options("/api/chat", headers={
        "Origin": "https://demo.netlify.app", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-user, x-demo-key, content-type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://demo.netlify.app"
    allowed = r.headers["access-control-allow-headers"].lower()
    assert all(h in allowed for h in ("x-user", "x-demo-key", "content-type"))
    assert "access-control-allow-credentials" not in r.headers


def test_cors_rejects_unlisted_origin_and_decorates_errors(cors_client):
    bad = cors_client.options("/api/chat", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert bad.status_code == 400 and "access-control-allow-origin" not in bad.headers
    r = cors_client.post("/api/chat", json={"message": "hi"}, headers={**USER, "Origin": "https://demo.netlify.app"})
    assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "https://demo.netlify.app"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_no_cors_headers_without_config(client):
    r = client.get("/healthz", headers={"Origin": "https://demo.netlify.app"})
    assert "access-control-allow-origin" not in r.headers
