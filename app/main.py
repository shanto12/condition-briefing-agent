"""FastAPI app: chat endpoint, briefing history, audit view, and the static chat page."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import store
from .render import briefing_markdown
from .service import chat_turn

log = logging.getLogger("briefing")
STATIC = Path(__file__).parent / "static"
USER_RE = re.compile(r"^[A-Za-z0-9._@-]{2,64}$")

app = FastAPI(title="Condition Briefing Copilot", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


def require_user(x_user: str | None) -> str:
    """Demo identity header. Production: SSO (OIDC) with role claims, not a client-supplied header."""
    if not x_user or not USER_RE.fullmatch(x_user):
        raise HTTPException(status_code=401, detail="Missing or invalid X-User identity header.")
    return x_user


class ChatIn(BaseModel):
    message: str = Field("", max_length=2000)
    thread_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")
    action: Literal["approve", "reject"] | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatIn, x_user: str | None = Header(None)):
    user = require_user(x_user)
    try:
        return chat_turn(user, body.message, body.thread_id, body.action)
    except Exception:
        log.exception("chat turn failed")
        return JSONResponse(status_code=500, content={"type": "error", "text": "The briefing run failed. Check the trace and retry."})


@app.get("/api/briefings")
def briefings(x_user: str | None = Header(None)):
    require_user(x_user)
    return store.list_briefings()


@app.get("/api/briefings/{briefing_id}")
def briefing(briefing_id: str, x_user: str | None = Header(None)):
    require_user(x_user)
    if not re.fullmatch(r"[a-f0-9]{12}", briefing_id):
        raise HTTPException(status_code=404, detail="Not found")
    record = store.get_briefing(briefing_id)
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    return {**record, "markdown": briefing_markdown(record["briefing"], record["status"])}


@app.get("/api/audit")
def audit(x_user: str | None = Header(None)):
    """Demo convenience. Production: restricted to a compliance role."""
    require_user(x_user)
    return store.list_audit()
