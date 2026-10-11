"""Graph routing with a canned LLM, stubbed public sources and stubbed document retrieval (InMemorySaver checkpoints)."""
from __future__ import annotations

import json

import pytest
from _support import SUBTYPES, absent, log_outputs
from langgraph.types import Command

from app import graph as G
from app import store

USER = "analyst.a"


def pending(g, cfg):
    for task in g.get_state(cfg).tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def start(g, cfg, request):
    g.invoke({"user": USER, "request": request, "events": []}, cfg)
    return g.get_state(cfg).values


def to_approval(g, cfg, answer, request="dementia"):
    start(g, cfg, request)
    assert pending(g, cfg)["kind"] == "questions"
    g.invoke(Command(resume={"text": answer}), cfg)
    p = pending(g, cfg)
    assert p and p["kind"] == "approval", "briefing did not reach the approval gate"
    return g.get_state(cfg).values


def audits_for(thread_id):
    return [a for a in store.list_audit(1000) if a["thread_id"] == thread_id]


# ---------------- intake ----------------

def test_clinical_request_refused_at_intake_before_any_work(briefing_graph, new_thread, fake_llm, stub_sources):
    cfg = new_thread()
    before = len(store.list_audit(1000))
    values = start(briefing_graph, cfg, "What dose of donepezil should I prescribe for patient P002?")
    assert values["status"] == "refused" and "does not diagnose" in values["message"]
    assert pending(briefing_graph, cfg) is None
    assert fake_llm.calls == [] and stub_sources == {"pubmed": [], "fda": [], "ctgov": []}
    assert len(store.list_audit(1000)) == before
    log_outputs({"status": values["status"], "llm_calls": fake_llm.calls})


def test_short_request_skips_condition_extraction(briefing_graph, new_thread, fake_llm):
    cfg = new_thread()
    values = start(briefing_graph, cfg, "dementia")
    assert values["condition"] == "dementia" and fake_llm.calls == ["SubtypeOptions"]
    p = pending(briefing_graph, cfg)
    assert p["kind"] == "questions" and p["subtype_options"] == SUBTYPES + ["All subtypes"]
    assert p["patient_ids"] == ["P001", "P002", "P003", "P004", "P005", "P006"]
    assert len(p["focus_options"]) == 3


def test_long_request_uses_condition_extraction(briefing_graph, new_thread, fake_llm):
    cfg = new_thread()
    values = start(briefing_graph, cfg, "Give me a strategy briefing on dementia for our planning committee")
    assert fake_llm.calls[0] == "ConditionExtract" and values["condition"] == "dementia"


def test_no_condition_named_asks_for_one(briefing_graph, new_thread, fake_llm):
    fake_llm.condition = None
    cfg = new_thread()
    values = start(briefing_graph, cfg, "Can you help me with our planning committee deck?")
    assert values["status"] == "refused" and "Which medical condition" in values["message"]
    assert pending(briefing_graph, cfg) is None


def test_condition_text_is_sanitized_and_capped(briefing_graph, new_thread, fake_llm):
    cfg = new_thread()
    values = start(briefing_graph, cfg, "dementia<script>alert(1)</script>")
    assert "<" not in values["condition"] and "(" not in values["condition"]
    cfg = new_thread()
    values = start(briefing_graph, cfg, "x" * 300)
    assert len(values["condition"]) == 80


def test_subtype_failure_still_asks_questions(briefing_graph, new_thread, fake_llm):
    fake_llm.fail_subtypes = True
    cfg = new_thread()
    start(briefing_graph, cfg, "dementia")
    assert pending(briefing_graph, cfg)["subtype_options"] == ["All subtypes"]


# ---------------- clarify answer parsing ----------------

@pytest.mark.parametrize("text,subtype,focus,patient", [
    ("Alzheimer's, service-line, P003", "Alzheimer's disease", "Service-line", "P003"),
    ("lewy body, trials, none", "Lewy body dementia", "Clinical trial", None),
    ("all subtypes, payer", "All subtypes", "Payer", None),
    ("vascular; formulary and cost exposure; p002", "Vascular dementia", "Payer", "P002"),
    ("research partnerships please", "All subtypes", "Clinical trial", None),
    ("", "All subtypes", "Service-line", None),
    ("Alzheimer's, contract terms, patient P1234", "Alzheimer's disease", "Payer", None),
])
def test_parse_answers(text, subtype, focus, patient):
    got = G.parse_answers(text, "dementia", SUBTYPES)
    assert got["subtype"] == subtype and got["focus"].startswith(focus) and got["patient_id"] == patient


def test_parse_answers_alzheimers_without_apostrophe():
    assert G.parse_answers("alzheimers, service-line", "dementia", SUBTYPES)["subtype"] == "Alzheimer's disease"


def test_plan_runs_patient_context_only_with_patient_id():
    assert "patient_context" not in G.plan({"patient_id": None})
    assert "patient_context" not in G.plan({})
    assert G.plan({"patient_id": "P001"}) == G.WORKERS


# ---------------- full briefing paths ----------------

def test_briefing_with_valid_patient_then_approve(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub):
    cfg = new_thread()
    tid = cfg["configurable"]["thread_id"]
    values = to_approval(briefing_graph, cfg, "Alzheimer's, trials, P001")
    assert values["subtype"] == "Alzheimer's disease" and values["patient_id"] == "P001"
    b = values["briefing"]
    pc = b["patient_context"]
    assert pc["patient_ref"] == "P001" and "Clinician review required" in pc["label"]
    assert [c["text"] for c in pc["claims"]] == [
        "The de-identified patient has mild Alzheimer's disease and attends the memory clinic."]
    assert any(a["action"] == "read_patient_history" and a["patient_ref"] == "P001" for a in audits_for(tid))
    assert any(a["action"] == "cohort_aggregate" for a in audits_for(tid))

    v = b["verification"]
    assert v["claims_verified"] == v["claims_total"] - v["claims_removed"]
    assert v["groundedness"] == round(v["claims_verified"] / v["claims_total"], 3)
    reasons = " ".join(" ".join(r["reasons"]) for r in v["removed"])
    assert "PMID:99999999 not in retrieved sources" in reasons
    assert "NCT00000001 is a clinicaltrials source, not allowed in this section" in reasons
    assert "patient-specific treatment directive" in reasons
    source_ids = {s["id"] for s in b["sources"]}
    cited = {c for sec in ("executive_summary", "standard_of_care", "emerging_treatments", "landscape")
             for claim in b[sec] for c in claim["citations"]}
    assert cited <= source_ids and "DOC:payer-policy#1" in cited
    assert {s["type"] for s in b["sources"] if s["id"] == "DOC:payer-policy#1"} == {"document"}

    stored = store.get_briefing(values["briefing_id"])
    assert stored["status"] == "draft" and stored["briefing"]["patient_context"] is None
    absent("de-identified patient has mild", stored["briefing"], "patient claim in stored briefing")

    briefing_graph.invoke(Command(resume={"action": "approve", "user": "reviewer.b"}), cfg)
    after = briefing_graph.get_state(cfg).values
    assert after["status"] == "final" and pending(briefing_graph, cfg) is None
    record = store.get_briefing(values["briefing_id"])
    assert record["status"] == "final" and record["approved_by"] == "reviewer.b"
    log_outputs({"verification": {k: v[k] for k in ("claims_total", "claims_verified", "claims_removed", "groundedness")},
                 "status": after["status"]})


def test_reject_sets_rejected_status(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub):
    cfg = new_thread()
    values = to_approval(briefing_graph, cfg, "all subtypes, payer, none")
    briefing_graph.invoke(Command(resume={"action": "reject", "user": USER}), cfg)
    assert briefing_graph.get_state(cfg).values["status"] == "rejected"
    assert store.get_briefing(values["briefing_id"])["status"] == "rejected"


def test_unknown_patient_continues_without_patient_context(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub):
    cfg = new_thread()
    tid = cfg["configurable"]["thread_id"]
    values = to_approval(briefing_graph, cfg, "Alzheimer's, service-line, P042")
    assert values["patient_id"] is None and values["briefing"]["patient_context"] is None
    assert any("P042 not in synthetic panel" in e for e in values["briefing"]["guardrail_events"])
    assert not any("PATIENT CONTEXT" in p for p in fake_llm.prompts)
    assert not [a for a in audits_for(tid) if a["action"].startswith("read_patient_history")]


def test_no_patient_requested_skips_patient_step(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub):
    cfg = new_thread()
    tid = cfg["configurable"]["thread_id"]
    values = to_approval(briefing_graph, cfg, "all subtypes, service-line, none")
    assert values["briefing"]["patient_context"] is None
    assert not any("PATIENT CONTEXT" in p for p in fake_llm.prompts)
    assert [a["action"] for a in audits_for(tid)] == ["cohort_aggregate"]
    dispatched = next(e for e in values["events"] if e.startswith("Supervisor dispatched"))
    assert "patient_context" not in dispatched and "cohort" in dispatched


def test_source_injection_removed_before_llm(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub):
    cfg = new_thread()
    values = to_approval(briefing_graph, cfg, "all subtypes, service-line, none")
    assert any("Instruction-like text removed from PMID:11111111" in e for e in values["events"])
    soc_prompt = next(p for p in fake_llm.prompts if "STANDARD OF CARE" in p)
    assert "ignore all previous" not in soc_prompt.lower() and "system prompt" not in soc_prompt.lower()


def test_relevance_filter_drops_off_topic_trials_from_pipeline(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub):
    cfg = new_thread()
    values = to_approval(briefing_graph, cfg, "all subtypes, trials, none")
    ids = {i["nct_id"] for i in values["briefing"]["pipeline"]}
    assert "NCT00000003" not in ids and {"NCT00000001", "NCT00000004"} <= ids


def test_worker_failure_degrades_to_gap(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub, monkeypatch):
    from app import sources

    def boom(*_a, **_k):
        raise RuntimeError("pubmed parser broke")

    monkeypatch.setattr(sources, "pubmed_evidence", boom)
    cfg = new_thread()
    values = to_approval(briefing_graph, cfg, "all subtypes, service-line, none")
    gaps = values["briefing"]["gaps"]
    assert any("standard_of_care step failed (RuntimeError)" in x for x in gaps)
    assert any("No verified claims for standard of care" in x for x in gaps)
    assert values["briefing"]["emerging_treatments"], "other sections still produced"


def test_document_search_failure_keeps_public_sources(briefing_graph, new_thread, fake_llm, stub_sources, monkeypatch):
    from app import rag

    def down(*_a, **_k):
        raise ConnectionError("chroma down")

    monkeypatch.setattr(rag, "search", down)
    cfg = new_thread()
    values = to_approval(briefing_graph, cfg, "all subtypes, service-line, none")
    assert any("Internal document search unavailable (ConnectionError)" in x for x in values["briefing"]["gaps"])
    assert values["briefing"]["standard_of_care"], "public-source claims still verified"


def test_all_sources_down_leaves_section_empty_with_gap(briefing_graph, new_thread, fake_llm, monkeypatch, rag_stub):
    from app import sources

    def unavailable(*_a, **_k):
        raise sources.SourceUnavailable("down and no cache")

    for name in ("pubmed_evidence", "fda_products", "active_trials"):
        monkeypatch.setattr(sources, name, unavailable)
    rag_stub.clear()
    cfg = new_thread()
    values = to_approval(briefing_graph, cfg, "all subtypes, service-line, none")
    b = values["briefing"]
    assert b["standard_of_care"] == [] and b["emerging_treatments"] == []
    assert any("left empty rather than written from model memory" in x for x in b["gaps"])
    assert not any("STANDARD OF CARE" in p for p in fake_llm.prompts)


# ---------------- verify node ----------------

def _verify_state(sections: dict, registry: dict) -> dict:
    return {"user": USER, "condition": "dementia", "subtype": "All subtypes", "focus": "Service-line",
            "sections": sections, "registry": registry, "events": []}


REG = {"PMID:11111111": {"id": "PMID:11111111", "type": "pubmed", "title": "t"},
       "NCT00000001": {"id": "NCT00000001", "type": "clinicaltrials", "title": "t"},
       "DOC:payer-policy#1": {"id": "DOC:payer-policy#1", "type": "document", "title": "Payer", "section": "Medicare"}}


def test_verify_groundedness_and_gap_notes():
    sections = {
        "standard_of_care": {"claims": [{"text": "Good.", "citations": ["PMID:11111111"]},
                                        {"text": "Doc.", "citations": ["DOC:payer-policy#1"]},
                                        {"text": "Fake.", "citations": ["PMID:42"]}], "gaps": []},
        "pipeline": {"claims": [{"text": "Doc in pipeline.", "citations": ["DOC:payer-policy#1"]}],
                     "gaps": ["No active phase 2-3 trials found for 'dementia' on ClinicalTrials.gov."]},
        "landscape": {"claims": [], "gaps": []},
    }
    out = G.verify(_verify_state(sections, REG), {"configurable": {"thread_id": "verify-test"}})
    b = out["briefing"]
    assert [c["text"] for c in b["standard_of_care"]] == ["Good.", "Doc."]
    assert b["emerging_treatments"] == []
    v = b["verification"]
    assert (v["claims_total"], v["claims_verified"], v["claims_removed"], v["groundedness"]) == (4, 2, 2, 0.5)
    assert any("No active phase 2-3 trials" in x for x in b["gaps"])
    assert any("No verified claims for pipeline" in x for x in b["gaps"])
    assert any("No verified claims for landscape" in x for x in b["gaps"])
    assert not any("standard of care" in x for x in b["gaps"])
    assert {s["id"] for s in b["sources"]} == {"PMID:11111111", "DOC:payer-policy#1"}


def test_verify_no_claims_gives_null_groundedness():
    out = G.verify(_verify_state({}, {}), {"configurable": {"thread_id": "verify-empty"}})
    assert out["briefing"]["verification"]["groundedness"] is None and out["status"] == "draft"


def test_verify_doc_citation_not_retrieved_is_dropped():
    sections = {"standard_of_care": {"claims": [{"text": "Memo.", "citations": ["DOC:payer-policy#7"]}], "gaps": []}}
    out = G.verify(_verify_state(sections, REG), {"configurable": {"thread_id": "verify-doc"}})
    assert out["briefing"]["standard_of_care"] == []
    assert "not in retrieved sources" in out["briefing"]["verification"]["removed"][0]["reasons"][0]


# ---------------- follow-up router ----------------

def _finished(briefing_graph, new_thread, patient="P001"):
    cfg = new_thread()
    to_approval(briefing_graph, cfg, f"Alzheimer's, service-line, {patient}")
    briefing_graph.invoke(Command(resume={"action": "approve", "user": USER}), cfg)
    return cfg


def test_entry_router():
    assert G.entry({"followup": "What does CMS require?"}) == "followup"
    assert G.entry({"request": "dementia"}) == "intake"
    assert G.entry({"followup": None, "request": "dementia"}) == "intake"


def test_followup_answers_with_verified_doc_citation(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub, tool_model):
    cfg = _finished(briefing_graph, new_thread)
    model = tool_model([[("search_documents", {"query": "CMS registry requirement"})], []])
    calls_before = len(fake_llm.calls)
    briefing_graph.invoke({"followup": "What does CMS require for anti-amyloid coverage?", "user": USER}, cfg)
    values = briefing_graph.get_state(cfg).values
    answer = values["followup_answer"]
    assert [c["citations"] for c in answer["claims"]] == [["DOC:payer-policy#1"]]
    assert {r["text"] for r in answer["removed"]} == {"An invented memo says otherwise.", "Patient P001 is a good candidate."}
    assert answer["tool_calls"] == [{"tool": "search_documents", "query": "CMS registry requirement", "results": 1}]
    assert [s["id"] for s in answer["sources"]] == ["DOC:payer-policy#1"] and answer["text"]
    assert values["status"] == "final" and values["followup"] is None
    assert fake_llm.calls[calls_before:] == ["FollowupDraft"]
    assert "search_documents" in model.bound and not any("patient" in t for t in model.bound)
    first_prompt = "\n".join(str(m.content) for m in model.seen[0])
    assert "PATIENT:" not in first_prompt and "de-identified patient" not in first_prompt
    log_outputs({"claims": answer["claims"], "removed": len(answer["removed"]), "tool_calls": answer["tool_calls"]})


@pytest.mark.parametrize("question", ["What is P001's history?", "Summarize the patient record for me",
                                      "Should we start lecanemab for patient P003?"])
def test_followup_patient_or_clinical_requests_refused(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub,
                                                       tool_model, question):
    cfg = _finished(briefing_graph, new_thread)
    model = tool_model([[("search_documents", {"query": "x"})]])
    briefing_graph.invoke({"followup": question, "user": USER}, cfg)
    answer = briefing_graph.get_state(cfg).values["followup_answer"]
    assert answer.get("refused") is True and answer["claims"] == [] and model.seen == []


def test_followup_not_in_documents(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub, tool_model):
    cfg = _finished(briefing_graph, new_thread, patient="none")
    rag_stub.clear()
    tool_model([[("search_documents", {"query": "parking garage budget"})], []])
    briefing_graph.invoke({"followup": "What is the parking garage budget?", "user": USER}, cfg)
    answer = briefing_graph.get_state(cfg).values["followup_answer"]
    assert answer["claims"] == [] and "Not in our documents" in answer["reason"]


def test_followup_tool_budget_and_unknown_tools(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub, tool_model):
    cfg = _finished(briefing_graph, new_thread, patient="none")
    script = [[("get_patient_history", {"patient_id": "P001"})]] + [[("search_documents", {"query": f"q{i}"})] for i in range(10)]
    model = tool_model(script)
    briefing_graph.invoke({"followup": "Tell me everything about infusion capacity", "user": USER}, cfg)
    answer = briefing_graph.get_state(cfg).values["followup_answer"]
    assert len(answer["tool_calls"]) == G.MAX_TOOL_CALLS
    assert all(t["tool"] == "search_documents" for t in answer["tool_calls"])
    tool_texts = [str(m.content) for m in model.seen[1] if m.type == "tool"]
    assert tool_texts and "unknown tool" in tool_texts[0]


def test_followup_tool_error_is_reported_not_raised(briefing_graph, new_thread, fake_llm, stub_sources, rag_stub,
                                                    tool_model, monkeypatch):
    from app import sources

    def down(*_a, **_k):
        raise sources.SourceUnavailable("pubmed down")

    monkeypatch.setattr(sources, "pubmed_evidence", down)
    cfg = _finished(briefing_graph, new_thread, patient="none")
    model = tool_model([[("search_guidelines", {"topic": "Alzheimer disease"})], []])
    briefing_graph.invoke({"followup": "What do guidelines say about MRI monitoring?", "user": USER}, cfg)
    answer = briefing_graph.get_state(cfg).values["followup_answer"]
    assert answer["tool_calls"][0]["results"] == 0 and answer["claims"] == []
    assert "failed (SourceUnavailable)" in json.dumps([str(m.content) for m in model.seen[-1]])
