"""Guardrails: scope refusal, PHI redaction (text and trace payloads), prompt-injection neutralization,
and citation verification. Enforced in code; prompts only repeat the rules."""
from __future__ import annotations

import re
from functools import lru_cache

SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE = re.compile(r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]\d{4}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MRN = re.compile(r"\bMRN[-:\s]?\d{4,}\b", re.I)
DOB = re.compile(r"\b(DOB|date of birth)\b[:\s]*[\d/.-]{6,10}", re.I)

CLINICAL_REQUEST = [
    re.compile(p, re.I) for p in (
        r"\bprescri(be|bing|ption)\b",
        r"\bwhat (dose|dosage|medication|drug|treatment) should (i|we|he|she|they|the patient|my patient|p\d{3}) "
        r"(give|prescribe|start|take|try|use|receive)\b",
        r"\b(dose|dosage|dosing)\b.*\b(patient|p\d{3})\b",
        r"\bshould (i|we|he|she|they|the patient|my patient|p\d{3}) (start|stop|take|get|give|receive|switch|increase|decrease|discontinue)\b",
        r"\b(diagnose|diagnosis for) (this|my|the|a) (patient|person|mother|father)\b",
        r"\bdoes (this|my|the) patient have\b",
        r"\btreatment (plan|order)s? for\b",
    )
]
CLINICAL_DIRECTIVE = re.compile(
    r"\b(should (start|begin|receive|take|switch)|recommend(ed|s)? (starting|initiating|switching|that)|"
    r"consider (starting|initiating|switching)|prescribe|initiate)\b", re.I)

INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)"
    r"|disregard (all |any |the )?(previous|prior|above|system) (instructions|prompts?|rules)"
    r"|\byou are now\b|\b(system|developer) prompt\b|</?(system|assistant|user|source)>|\bnew instructions?:)",
    re.I)

_ID = r"(?:PMID:\s?\d+|NCT\d{8}|FDA:\s?(?:NDA|BLA|ANDA)\d{6}|PANEL:[\w-]+|PATIENT:P\d{3}|DOC:[\w.-]+#\d+)"
INLINE_IDS = re.compile(rf"\s*[\(\[]\s*(?:sources?:\s*)?{_ID}(?:\s*[,;]\s*{_ID})*\s*[\)\]]", re.I)

CITATION_FORMATS = {
    "pubmed": re.compile(r"^PMID:\d{1,9}$"),
    "clinicaltrials": re.compile(r"^NCT\d{8}$"),
    "fda": re.compile(r"^FDA:(NDA|BLA|ANDA)\d{6}$"),
    "panel": re.compile(r"^PANEL:[\w-]+$"),
    "patient": re.compile(r"^PATIENT:P\d{3}$"),
    "document": re.compile(r"^DOC:[a-z0-9][\w.-]*#\d+$"),
}


# ---------- scope ----------

def clinical_request_reason(text: str) -> str | None:
    for pattern in CLINICAL_REQUEST:
        if pattern.search(text):
            return ("This tool produces strategy and decision-support briefings. It does not diagnose, "
                    "prescribe, or give patient-specific treatment orders. Please route clinical questions "
                    "to the treating clinician.")
    return None


# ---------- PHI redaction ----------

@lru_cache
def _known_identifiers() -> re.Pattern | None:
    from .patients import identifier_denylist

    terms = sorted({t for t in identifier_denylist() if t}, key=len, reverse=True)
    if not terms:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)


def redact_text(text: str) -> str:
    if not text:
        return text
    known = _known_identifiers()
    if known:
        text = known.sub("[REDACTED]", text)
    text = SSN.sub("[SSN]", text)
    text = MRN.sub("[MRN]", text)
    text = DOB.sub("[DOB]", text)
    text = EMAIL.sub("[EMAIL]", text)
    return PHONE.sub("[PHONE]", text)


def phi_findings(text: str) -> list[str]:
    """Names of the PHI patterns present. Used to reject uploads before any text leaves the machine."""
    from .patients import identifier_denylist

    text = text or ""
    found = [name for name, pattern in (("SSN", SSN), ("MRN", MRN), ("DOB", DOB), ("phone", PHONE), ("email", EMAIL))
             if pattern.search(text)]
    lowered = text.lower()
    if any(t and " " in t and t.lower() in lowered for t in identifier_denylist()):
        found.append("known patient name or address")
    return found


def redact_payload(obj):
    """Applied by the LangSmith client to every traced input and output (defense in depth)."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_payload(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return redact_payload(obj.model_dump())
    return obj


# ---------- untrusted content ----------

def neutralize_injection(text: str) -> tuple[str, int]:
    """Drops whole sentences that contain instruction-like text. Returns (clean_text, sentences_removed)."""
    if not text:
        return text, 0
    kept, removed = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if INJECTION.search(sentence):
            removed += 1
        else:
            kept.append(sentence)
    if removed:
        kept.append("[instruction-like text removed by guardrail]")
    return " ".join(kept), removed


def wrap_untrusted(source_id: str, text: str) -> str:
    safe = text.replace("<", "(").replace(">", ")")
    return f'<source id="{source_id}">\n{safe}\n</source>'


# ---------- citation verification ----------

def normalize_citation(raw: str) -> str:
    c = raw.strip().strip("[]()").strip()
    for pattern, fmt in (
        (r"(?i)^pmid[:\s]*(\d{1,9})$", "PMID:{}"),
        (r"(?i)^(nct\d{8})$", "{}"),
        (r"(?i)^(?:fda[:\s]*)?((?:nda|bla|anda)\d{6})$", "FDA:{}"),
        (r"(?i)^patient[:\s]*(p\d{3})$", "PATIENT:{}"),
        (r"(?i)^panel[:\s]*([\w-]+)$", "PANEL:{}"),
        (r"(?i)^doc[:\s]*([\w.-]+#\d+)$", "DOC:{}"),
    ):
        m = re.match(pattern, c)
        if m:
            value = m.group(1)
            if fmt.startswith("DOC"):
                return fmt.format(value.lower())
            return fmt.format(value.upper() if not fmt.startswith("PANEL") else value)
    return c


def verify_claims(claims: list[dict], registry: dict[str, dict], allowed_types: set[str],
                  clinical_check: bool = False) -> tuple[list[dict], list[dict]]:
    """Cite-or-drop. A claim survives only if every citation exists in the retrieved-source registry,
    matches its ID format, and is a source type allowed for that section."""
    kept, removed = [], []
    for claim in claims:
        citations = [normalize_citation(c) for c in claim.get("citations", [])]
        problems = []
        if not citations:
            problems.append("no citation")
        for c in citations:
            source = registry.get(c)
            if source is None:
                problems.append(f"{c} not in retrieved sources")
            elif source["type"] not in allowed_types:
                problems.append(f"{c} is a {source['type']} source, not allowed in this section")
            elif not CITATION_FORMATS[source["type"]].match(c):
                problems.append(f"{c} malformed")
        if clinical_check and CLINICAL_DIRECTIVE.search(claim.get("text", "")):
            problems.append("patient-specific treatment directive")
        if INJECTION.search(claim.get("text", "")):
            problems.append("instruction-like text")
        if problems:
            removed.append({"text": claim.get("text", ""), "reasons": problems})
        else:
            text = INLINE_IDS.sub("", claim["text"]).strip()
            kept.append({"text": text, "citations": list(dict.fromkeys(citations))})
    return kept, removed
