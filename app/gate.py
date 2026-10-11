"""Public-hosting guards: demo passphrase gate, per-IP sliding-window rate limits, global daily chat cap.

All config is read from the environment per request, so it can change without a restart in tests.
The gate runs first; limits only count requests that passed it, so bad-key probes cannot burn quota.
The passphrase is never logged or echoed.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_lock = threading.Lock()
_hits: dict[tuple[str, str], deque] = defaultdict(deque)  # (bucket, ip) -> request timestamps
_daily = {"day": None, "count": 0}
WINDOW = 3600.0


def reset() -> None:
    with _lock:
        _hits.clear()
        _daily.update(day=None, count=0)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def client_ip(request: Request) -> str:
    """First X-Forwarded-For hop (Render/Netlify proxy); falls back to the socket peer."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        first = fwd.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def _limited(retry_after: int) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "rate_limited"},
                        headers={"Retry-After": str(max(1, retry_after))})


def _check_window(bucket: str, ip: str, limit: int) -> int | None:
    """Record a hit; return seconds until a slot frees if the sliding window is full, else None."""
    now = time.monotonic()
    with _lock:
        q = _hits[(bucket, ip)]
        while q and now - q[0] >= WINDOW:
            q.popleft()
        if len(q) >= limit:
            return int(WINDOW - (now - q[0])) + 1
        q.append(now)
        if len(_hits) > 10000:  # bound memory: drop idle keys
            for key in [k for k, v in _hits.items() if not v or now - v[-1] >= WINDOW]:
                _hits.pop(key, None)
    return None


def _check_daily(cap: int) -> int | None:
    now = datetime.now(timezone.utc)
    today = now.date()
    with _lock:
        if _daily["day"] != today:
            _daily.update(day=today, count=0)
        if _daily["count"] >= cap:
            midnight = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            return int((midnight - now).total_seconds()) + 1
        _daily["count"] += 1
    return None


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def gate_and_limits(request: Request, call_next):
        path, method = request.url.path, request.method
        if not path.startswith("/api/") or method == "OPTIONS":
            return await call_next(request)

        passphrase = os.environ.get("DEMO_PASSPHRASE")
        if passphrase:
            supplied = request.headers.get("x-demo-key", "")
            if not hmac.compare_digest(supplied.encode("utf-8"), passphrase.encode("utf-8")):
                return JSONResponse(status_code=401, content={"detail": "demo_key_required"})

        ip = client_ip(request)
        if method == "POST" and path == "/api/chat":
            wait = _check_window("chat", ip, _int_env("RATE_LIMIT_CHAT_PER_HOUR", 20))
            if wait is None:
                wait = _check_daily(_int_env("DAILY_CHAT_CAP", 200))
            if wait is not None:
                return _limited(wait)
        elif method == "POST" and path == "/api/documents":
            wait = _check_window("upload", ip, _int_env("RATE_LIMIT_UPLOAD_PER_HOUR", 10))
            if wait is not None:
                return _limited(wait)
        return await call_next(request)
