"""Offline test harness. Config env vars are set BEFORE any app module is imported (app.config reads them at import)."""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

SUITE = "condition-briefing-unit"
_ROOT = Path(tempfile.mkdtemp(prefix="briefing-tests-"))
sys.path.insert(0, str(Path(__file__).parent))

from _support import FakeEmbeddings, FakeStructured, write_corpus, write_panel  # noqa: E402

write_panel(_ROOT / "patients.json")
write_corpus(_ROOT / "corpus")
(_ROOT / "fixtures").mkdir()
os.environ.update({
    "BRIEFING_DATA_DIR": str(_ROOT / "data"),
    "BRIEFING_FIXTURE_DIR": str(_ROOT / "fixtures"),
    "BRIEFING_PATIENTS_FILE": str(_ROOT / "patients.json"),
    "BRIEFING_CORPUS_DIR": str(_ROOT / "corpus"),
    "BRIEFING_TRACING": "off",
    "COHORT_MIN_CELL": "3",
    "HEALTH_SYSTEM_STATE": "Illinois",
    "EMBED_PROVIDER": "openai",
    "EMBED_MODEL": "text-embedding-3-large",
    "RERANK": "on",
    "RERANK_MIN_SCORE": "0.05",
    "RETRIEVE_TOP_K": "5",
    # Placeholders so nothing falls through to the Keychain; every client that would use them is stubbed.
    "OPENAI_API_KEY": "test-offline",
    "PINECONE_API_KEY": "test-offline",
    "DEEPSEEK_API_KEY": "test-offline",
    "KEYCHAIN_ACCOUNT": "pytest-no-keychain",
    "ANONYMIZED_TELEMETRY": "False",
})

import app.config  # noqa: E402,F401  (forces LANGSMITH_TRACING=false; re-enabled below for the pytest plugin only)

if os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGSMITH_TEST_SUITE", SUITE)
    os.environ.setdefault("LANGSMITH_EXPERIMENT", SUITE)
else:
    os.environ["LANGSMITH_TEST_TRACKING"] = "false"

from langsmith import utils as _ls_utils  # noqa: E402

_ls_utils.get_env_var.cache_clear()

# Only explicit test results go to LangSmith: block LangChain auto-tracing of graph/node internals inside test runs.
import langchain_core.callbacks.manager as _lc_manager  # noqa: E402

_lc_manager._tracing_v2_is_enabled = lambda: False


def pytest_collection_modifyitems(config, items):
    for item in items:
        if not item.get_closest_marker("langsmith"):
            item.add_marker(pytest.mark.langsmith(test_suite_name=SUITE))


def pytest_terminal_summary(terminalreporter):
    try:
        from langsmith.testing._internal import _LangSmithTestSuite
    except ImportError:
        return
    for name, suite in (getattr(_LangSmithTestSuite, "_instances", None) or {}).items():
        terminalreporter.write_line(
            f"LangSmith experiment '{suite.experiment.name}' (suite {name}): "
            f"{suite._dataset.url}/compare?selectedSessions={suite.experiment_id}")


# ---------------- autouse safety net ----------------

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No real LLM, embedding, rerank or public-API call can leave the process."""
    import httpx

    from app import config, graph, sources

    def blocked(*_a, **_k):
        raise httpx.ConnectError("network blocked in tests")

    def no_llm(*_a, **_k):
        raise RuntimeError("real LLM blocked in tests")

    monkeypatch.setattr(httpx, "get", blocked)
    monkeypatch.setattr(httpx, "post", blocked)
    monkeypatch.setattr(config, "llm", no_llm)
    monkeypatch.setattr(graph, "llm", no_llm)
    monkeypatch.setattr(sources, "_trial_cache", {})
    try:
        from app import rag
    except ImportError:
        rag = None
    if rag is not None:
        monkeypatch.setattr(rag, "embeddings", lambda provider, model: FakeEmbeddings())
    yield


# ---------------- fixtures ----------------

@pytest.fixture
def fake_llm(monkeypatch):
    from app import graph

    fake = FakeStructured()
    monkeypatch.setattr(graph, "_structured", fake)
    return fake


@pytest.fixture
def stub_sources(monkeypatch):
    """Small public-source fixtures in place of PubMed, openFDA and ClinicalTrials.gov."""
    import copy

    from _support import ARTICLES, PRODUCTS, TRIALS

    from app import sources

    calls = {"pubmed": [], "fda": [], "ctgov": []}

    def pubmed(term, *a, **k):
        calls["pubmed"].append(term)
        return copy.deepcopy(ARTICLES), False

    def fda(terms, *a, **k):
        calls["fda"].append(list(terms))
        return copy.deepcopy(PRODUCTS), False

    def trials(term, *a, **k):
        calls["ctgov"].append(term)
        return copy.deepcopy(TRIALS), False

    monkeypatch.setattr(sources, "pubmed_evidence", pubmed)
    monkeypatch.setattr(sources, "fda_products", fda)
    monkeypatch.setattr(sources, "active_trials", trials)
    return calls


@pytest.fixture
def rag_env(monkeypatch, tmp_path):
    """A fresh Chroma directory and corpus copy per test, with the index cache cleared."""
    rag = pytest.importorskip("app.rag", reason="app/rag.py has not landed")
    from app import config

    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "CORPUS_DIR", corpus)
    rag.index.cache_clear()
    yield rag
    rag.index.cache_clear()


@pytest.fixture
def rag_stub(monkeypatch):
    """Graph-level stub for document retrieval so briefing tests never build an index."""
    try:
        from app import rag
    except ImportError:
        return []
    hits = [{"id": "DOC:payer-policy#1", "doc_id": "payer-policy", "title": "Payer Coverage Summary",
             "section": "Traditional Medicare", "date": "2026-06-20",
             "text": "CMS covers anti-amyloid antibodies under coverage with evidence development.",
             "score": 0.81, "score_kind": "rerank"}]
    monkeypatch.setattr(rag, "search", lambda *a, **k: [dict(h) for h in hits])
    return hits


@pytest.fixture
def tool_model(monkeypatch):
    """Install a scripted tool-calling model for the follow-up agent: tool_model([[("search_documents", {...})], []])."""
    from _support import FakeToolModel

    from app import graph

    def install(script):
        model = FakeToolModel(script)
        monkeypatch.setattr(graph, "llm", lambda: model)
        return model

    return install


@pytest.fixture(scope="session")
def briefing_graph():
    """Compiled once; nodes resolve stubs through module globals at call time, and every test uses a fresh thread."""
    from langgraph.checkpoint.memory import InMemorySaver

    from app.graph import build_graph

    return build_graph(InMemorySaver())


@pytest.fixture
def new_thread():
    return lambda: {"configurable": {"thread_id": uuid.uuid4().hex}}


@pytest.fixture
def api(fake_llm, stub_sources, rag_stub):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client
