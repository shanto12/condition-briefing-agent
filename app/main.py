"""FastAPI app: chat endpoint, resumable sessions, document upload, briefing history, audit view, and the static chat page."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import store
from .render import briefing_markdown
from .service import chat_turn, session_view

log = logging.getLogger("briefing")
STATIC = Path(__file__).parent / "static"
USER_RE = re.compile(r"^[A-Za-z0-9._@-]{2,64}$")
THREAD_RE = re.compile(r"^[a-f0-9]{32}$")
UPLOAD_TYPES = {".md", ".txt", ".pdf"}
UPLOAD_MAX_BYTES = 2 * 1024 * 1024

app = FastAPI(title="Condition Briefing Copilot", version="0.2.0")
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
    if body.thread_id:
        session = store.get_session(body.thread_id)
        if session and session["user"] != user:
            raise HTTPException(status_code=404, detail="Not found")
    try:
        return chat_turn(user, body.message, body.thread_id, body.action)
    except Exception:
        log.exception("chat turn failed")
        return JSONResponse(status_code=500, content={"type": "error", "text": "The briefing run failed. Check the trace and retry."})


@app.get("/api/sessions")
def sessions(x_user: str | None = Header(None)):
    return store.list_sessions(require_user(x_user))


@app.get("/api/sessions/{thread_id}")
def session(thread_id: str, x_user: str | None = Header(None)):
    user = require_user(x_user)
    view = session_view(thread_id, user) if THREAD_RE.fullmatch(thread_id) else None
    if not view:
        raise HTTPException(status_code=404, detail="Not found")
    return view


@app.get("/api/documents")
def documents(x_user: str | None = Header(None)):
    require_user(x_user)
    try:
        from .rag import list_documents
    except ImportError:
        raise HTTPException(status_code=503, detail="Document store is not available yet.")
    return list_documents()


@app.post("/api/documents")
def upload_document(file: UploadFile = File(...), x_user: str | None = Header(None)):
    user = require_user(x_user)
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in UPLOAD_TYPES:
        raise HTTPException(status_code=415, detail="Only .md, .txt and .pdf files are accepted.")
    data = file.file.read(UPLOAD_MAX_BYTES + 1)
    if len(data) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 2 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")
    try:
        from .rag import ingest_bytes
    except ImportError:
        raise HTTPException(status_code=503, detail="Document store is not available yet.")
    try:
        result = ingest_bytes(name, data, user)
    except Exception:
        log.exception("document ingest failed")
        raise HTTPException(status_code=500, detail="Document ingest failed. Check the server log.")
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("reason") or "Document rejected.")
    return result


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
