"""Markdown rendering for the chat UI. All cell text is flattened so untrusted titles cannot break tables."""
from __future__ import annotations


def _cell(value) -> str:
    return str(value if value not in (None, "") else "-").replace("|", "/").replace("\n", " ")


def _cite(source_id: str, sources: dict[str, dict]) -> str:
    url = (sources.get(source_id) or {}).get("url")
    return f"[{source_id}]({url})" if url else f"`{source_id}`"


def _claims(claims: list[dict], sources: dict[str, dict]) -> list[str]:
    if not claims:
        return ["- _No verified claims._"]
    return [f"- {c['text']} " + " ".join(_cite(i, sources) for i in c["citations"]) for c in claims]


def questions_markdown(payload: dict) -> str:
    subtypes = ", ".join(payload["subtype_options"])
    focus = "; ".join(payload["focus_options"])
    return "\n".join([
        f"Before I build the **{payload['condition']}** briefing, three quick questions:",
        "",
        f"1. **Subtype**: {subtypes}?",
        f"2. **Planning focus**: {focus}? (default: service-line)",
        f"3. **Optional patient context**: a synthetic patient ID ({payload['patient_ids'][0]}-{payload['patient_ids'][-1]}), or 'none'.",
        "",
        "_Example reply: Alzheimer's, service-line, none_",
    ])


def briefing_markdown(b: dict, status: str = "draft") -> str:
    src = {s["id"]: s for s in b.get("sources", [])}
    v = b.get("verification", {})
    stats = v.get("pipeline_stats", {})
    out = [
        f"# {b['condition'].title()} briefing ({b['subtype']})",
        f"**Focus:** {b['focus']}  ",
        f"**Generated:** {b['generated_at']} | **Status:** {status.upper()}"
        + (" (awaiting analyst approval)" if status == "draft" else ""),
        f"**Verification:** {v.get('claims_verified', 0)}/{v.get('claims_total', 0)} claims passed cite-or-drop"
        f" ({v.get('claims_removed', 0)} removed)",
        "",
        "## Executive summary",
        *_claims(b.get("executive_summary", []), src),
        "",
        "## Current standard of care",
        *_claims(b.get("standard_of_care", []), src),
    ]
    if b.get("approved_products"):
        out += ["", "**FDA-approved products (label indication mentions the condition)**", "",
                "| Product | Brand | Sponsor | Category | Application |", "|---|---|---|---|---|"]
        out += [f"| {_cell(p['generic_name'])} | {_cell(p['brand_name'])} | {_cell(p['sponsor'])} | {_cell(p['category'])} | "
                f"{_cite(p['application'], src)} |" for p in b["approved_products"]]
    out += ["", "## Emerging treatments in development", *_claims(b.get("emerging_treatments", []), src)]
    if b.get("pipeline"):
        out += ["", f"**Late-stage pipeline:** top {len(b['pipeline'])} treatment trials of {stats.get('active_trials', '?')} relevant "
                f"active phase 2-3 trials ({stats.get('phase3', '?')} phase 3, {stats.get('industry_led', '?')} industry-led; "
                f"{stats.get('returned_by_registry', '?')} returned by the registry before the relevance filter)", "",
                "| Trial | Phase | Sponsor | Interventions | Primary completion |", "|---|---|---|---|---|"]
        out += [f"| {_cite(t['nct_id'], src)} | {_cell(t['phase'])} | {_cell(t['sponsor'])} | {_cell(', '.join(t['interventions']))} | "
                f"{_cell(t['primary_completion'])} |" for t in b["pipeline"]]
    out += ["", "## Key companies and institutions", *_claims(b.get("landscape", []), src)]
    if b.get("key_organizations"):
        out += ["", "| Lead sponsor | Type | Active trials | Example trials |", "|---|---|---|---|"]
        out += [f"| {_cell(o['name'])} | {_cell(o['kind'])} | {o['active_trials']} | "
                f"{' '.join(_cite(i, src) for i in o['nct_ids'][:3])} |" for o in b["key_organizations"]]
    for title, rows in (("Most active US trial sites", b.get("key_sites")),
                        (f"Most active sites in our market ({b.get('market_state')})", b.get("market_sites"))):
        if rows:
            out += ["", f"**{title}**", "", "| Site | Location | Active trials | Example trials |", "|---|---|---|---|"]
            out += [f"| {_cell(s['name'])} | {_cell(s['location'])} | {s['active_trials']} | "
                    f"{' '.join(_cite(i, src) for i in s['nct_ids'][:3])} |" for s in rows]
    c = b.get("cohort")
    if c:
        out += ["", "## Our patient panel (de-identified aggregate)", f"_{c['note']}_", "",
                f"- Matched patients: **{c['matched_patients']}** of {c['panel']}"]
        for label, key in (("By subtype", "by_subtype"), ("By age band", "by_age_band"), ("By stage", "by_stage"),
                           ("By current medication", "by_medication")):
            if c.get(key):
                out.append(f"- {label}: " + ", ".join(f"{k} {n}" for k, n in c[key].items()))
    pc = b.get("patient_context")
    if pc:
        out += ["", f"## Patient context: {pc['patient_ref']}", f"> **{pc['label']}**", *_claims(pc.get("claims", []), src)]
    if b.get("gaps"):
        out += ["", "## Gaps and caveats", *[f"- {g}" for g in b["gaps"]]]
    if v.get("removed"):
        out += ["", "## Claims removed by the verifier"]
        out += [f"- ~~{_cell(r['text'])}~~ ({r['section']}: {'; '.join(r['reasons'])})" for r in v["removed"]]
    if b.get("guardrail_events"):
        out += ["", "## Run log and guardrail events", *[f"- {e}" for e in b["guardrail_events"]]]
    out += ["", f"## Sources ({len(b.get('sources', []))})", *_source_lines(b.get("sources", []), src)]
    return "\n".join(out)


def _source_lines(sources: list[dict], src: dict[str, dict]) -> list[str]:
    lines = []
    for s in sources:
        label = _cell(s["title"])
        if s.get("type") == "document":
            label = f"{label}, section \"{_cell(s.get('section'))}\" (internal document, synthetic)"
        lines.append(f"- {_cite(s['id'], src)}: {label}" + (f" ({s['date']})" if s.get("date") else ""))
    return lines


def followup_markdown(answer: dict) -> str:
    src = {s["id"]: s for s in answer.get("sources", [])}
    out = [f"**Follow-up:** {_cell(answer['question'])}", ""]
    if answer.get("refused"):
        return "\n".join(out + [answer["reason"]])
    if answer.get("claims"):
        out += _claims(answer["claims"], src)
    else:
        out.append(f"_{answer.get('reason') or 'Not in our documents or the public sources searched.'}_")
    if answer.get("removed"):
        out += ["", "**Removed by the verifier**"]
        out += [f"- ~~{_cell(r['text'])}~~ ({'; '.join(r['reasons'])})" for r in answer["removed"]]
    if answer.get("tool_calls"):
        out += ["", "_Research steps: " + "; ".join(f"{t['tool']}({_cell(t['query'])}) -> {t['results']}" for t in answer["tool_calls"]) + "_"]
    if answer.get("sources"):
        out += ["", f"**Sources ({len(answer['sources'])})**", *_source_lines(answer["sources"], src)]
    return "\n".join(out)
