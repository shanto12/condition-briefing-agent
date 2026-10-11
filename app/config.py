"""Runtime configuration. Secrets come from the environment (deployment) or macOS Keychain (local demo)."""
from __future__ import annotations

import getpass
import os
import subprocess
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BRIEFING_DATA_DIR", BASE_DIR / "data"))
FIXTURE_DIR = Path(os.environ.get("BRIEFING_FIXTURE_DIR", BASE_DIR / "fixtures"))
PATIENTS_FILE = Path(os.environ.get("BRIEFING_PATIENTS_FILE", BASE_DIR / "data" / "patients.json"))
APP_DB = DATA_DIR / "app.db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"

MODEL = os.environ.get("BRIEFING_MODEL", "deepseek-flash")
# Reasoning budget: off|low|high|max. Default high.
# While thinking is enabled, DeepSeek rejects forced tool_choice ("required" and a named
# function) and rejects response_format json_schema. JSON mode and tool_choice "auto"
# (or omitting tool_choice) are accepted, and those calls still reason.
# Set DEEPSEEK_THINKING=off to disable thinking on every call.
THINKING = os.environ.get("DEEPSEEK_THINKING", "high")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "live-build-interview")
# Small-cell suppression for de-identified cohort counts. Demo panel is tiny; CMS-style reporting uses 11.
COHORT_MIN_CELL = int(os.environ.get("COHORT_MIN_CELL", "3"))
HTTP_TIMEOUT = float(os.environ.get("BRIEFING_HTTP_TIMEOUT", "20"))
MARKET_STATE = os.environ.get("HEALTH_SYSTEM_STATE", "Illinois")

CORPUS_DIR = Path(os.environ.get("BRIEFING_CORPUS_DIR", BASE_DIR / "corpus"))
CHROMA_DIR = DATA_DIR / "chroma"
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "openai")
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "text-embedding-3-large" if EMBED_PROVIDER == "openai" else "llama-text-embed-v2")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "bge-reranker-v2-m3")
RERANK_ON = os.environ.get("RERANK", "on") == "on"
RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "5"))
# Below this rerank score the retriever reports "not in our documents" instead of passing weak chunks on.
RERANK_MIN_SCORE = float(os.environ.get("RERANK_MIN_SCORE", "0.3"))

# Traces go through an explicit, redacting tracer only; the env-driven default tracer would bypass redaction.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


def _secret(env_var: str, keychain_service: str) -> str | None:
    value = os.environ.get(env_var)
    if value:
        return value
    account = os.environ.get("KEYCHAIN_ACCOUNT", getpass.getuser())
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", account, "-s", keychain_service, "-w"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@lru_cache
def openai_key() -> str | None:
    return _secret("OPENAI_API_KEY", "codex-openai-interviews")


@lru_cache
def pinecone_key() -> str | None:
    return _secret("PINECONE_API_KEY", "codex-pinecone-interviews")


@lru_cache
def llm():
    from langchain_deepseek import ChatDeepSeek

    key = _secret("DEEPSEEK_API_KEY", "codex-deepseek-interviews")
    if not key:
        raise RuntimeError("DeepSeek credential unavailable in environment or Keychain")
    thinking = THINKING.strip().lower()
    extra_body = {"thinking": {"type": "disabled" if thinking == "off" else "enabled"}}
    kwargs: dict = {}
    if thinking != "off":
        kwargs["reasoning_effort"] = thinking
    return ChatDeepSeek(
        model=MODEL, api_key=key, temperature=0, timeout=90, max_retries=2,
        extra_body=extra_body, **kwargs,
    )


@lru_cache
def tracer():
    if os.environ.get("BRIEFING_TRACING", "on") == "off":
        return None
    key = _secret("LANGSMITH_API_KEY", "codex-langsmith-personal-key")
    if not key:
        return None
    from langchain_core.tracers.langchain import LangChainTracer
    from langsmith import Client

    from .guardrails import redact_payload

    client = Client(api_key=key, hide_inputs=redact_payload, hide_outputs=redact_payload)
    return LangChainTracer(project_name=LANGSMITH_PROJECT, client=client)
