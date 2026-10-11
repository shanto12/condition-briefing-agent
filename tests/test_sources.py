"""Public-source client: cached-fixture fallback when the live call fails, and the ClinicalTrials.gov relevance filter."""
from __future__ import annotations

import json

import httpx
import pytest
from _support import SUBTYPES, TRIALS

from app import graph as G
from app import sources


class _Resp:
    def __init__(self, body: dict, status: int = 200):
        self.status_code, self.text = status, json.dumps(body)

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("GET", "https://x"), response=httpx.Response(self.status_code))


STUDY = {"studies": [{"protocolSection": {
    "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Anti-tau study"},
    "statusModule": {"overallStatus": "RECRUITING", "primaryCompletionDateStruct": {"date": "2027-06"}},
    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Eli Lilly and Company", "class": "INDUSTRY"}},
    "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 900}},
    "armsInterventionsModule": {"interventions": [{"name": "Drug X", "type": "DRUG"}]},
    "conditionsModule": {"conditions": ["Alzheimer Disease"]},
    "contactsLocationsModule": {"locations": [{"facility": "Site A", "city": "Chicago", "state": "Illinois",
                                               "country": "United States"}, {"city": "No facility"}]},
}}]}


@pytest.fixture
def fixture_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "FIXTURE_DIR", tmp_path)
    return tmp_path


def test_live_success_writes_fixture_then_outage_serves_it_stale(fixture_dir, monkeypatch):
    monkeypatch.setattr(sources.httpx, "get", lambda *a, **k: _Resp(STUDY))
    trials, stale = sources.active_trials("dementia")
    assert stale is False and trials[0]["nct_id"] == "NCT00000001"
    assert trials[0]["phase"] == "PHASE3" and trials[0]["locations"] == [
        {"facility": "Site A", "city": "Chicago", "state": "Illinois", "country": "United States"}]
    assert len(list(fixture_dir.glob("ctgov-*.cache"))) == 1

    def down(*_a, **_k):
        raise httpx.ConnectError("simulated outage")

    monkeypatch.setattr(sources.httpx, "get", down)
    monkeypatch.setattr(sources, "_trial_cache", {})
    trials, stale = sources.active_trials("dementia")
    assert stale is True and [t["nct_id"] for t in trials] == ["NCT00000001"]


def test_outage_without_fixture_raises_source_unavailable(fixture_dir):
    with pytest.raises(sources.SourceUnavailable, match="no cached copy"):
        sources.active_trials("a condition never cached")


def test_http_error_status_falls_back_to_fixture(fixture_dir, monkeypatch):
    monkeypatch.setattr(sources.httpx, "get", lambda *a, **k: _Resp({"results": [{"x": 1}]}))
    body, stale = sources._get("openfda", sources.OPENFDA, {"search": "q"})
    assert stale is False
    monkeypatch.setattr(sources.httpx, "get", lambda *a, **k: _Resp({}, status=503))
    body, stale = sources._get("openfda", sources.OPENFDA, {"search": "q"})
    assert stale is True and body == {"results": [{"x": 1}]}


def test_openfda_404_means_no_results_not_outage(fixture_dir, monkeypatch):
    monkeypatch.setattr(sources.httpx, "get", lambda *a, **k: _Resp({}, status=404))
    products, stale = sources.fda_products(["a rare condition"])
    assert products == [] and stale is False


def test_trial_cache_shares_one_fetch(fixture_dir, monkeypatch):
    calls = []

    def get(*a, **k):
        calls.append(1)
        return _Resp(STUDY)

    monkeypatch.setattr(sources.httpx, "get", get)
    sources.active_trials("Dementia")
    sources.active_trials("dementia")
    assert len(calls) == 1


def test_search_phrase_strips_possessive_and_quotes():
    assert sources.search_phrase("Alzheimer's disease") == "Alzheimer disease"
    assert sources.search_phrase('"dementia"') == "dementia"


# ---------------- ClinicalTrials.gov relevance filter ----------------

def _state(subtype: str) -> dict:
    return {"condition": "dementia", "subtype": subtype, "subtype_options": SUBTYPES}


@pytest.fixture
def trials_stub(monkeypatch):
    monkeypatch.setattr(sources, "active_trials", lambda term, *a, **k: ([dict(t) for t in TRIALS], False))


def test_relevance_filter_all_subtypes_drops_off_topic(trials_stub):
    kept, returned, stale = G._relevant_trials(_state("All subtypes"))
    assert returned == 4 and stale is False
    assert [t["nct_id"] for t in kept] == ["NCT00000001", "NCT00000002", "NCT00000004"]


def test_relevance_filter_specific_subtype(trials_stub):
    kept, _, _ = G._relevant_trials(_state("Lewy body dementia"))
    assert [t["nct_id"] for t in kept] == ["NCT00000004"]
    kept, _, _ = G._relevant_trials(_state("Alzheimer's disease"))
    assert [t["nct_id"] for t in kept] == ["NCT00000001"]


def test_relevance_filter_nothing_relevant(monkeypatch):
    monkeypatch.setattr(sources, "active_trials",
                        lambda term, *a, **k: ([t for t in TRIALS if t["nct_id"] == "NCT00000003"], False))
    kept, returned, _ = G._relevant_trials(_state("All subtypes"))
    assert kept == [] and returned == 1
