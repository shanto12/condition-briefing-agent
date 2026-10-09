"""Synthetic patient panel, the get_patient_history tool, and de-identified cohort statistics.
Direct identifiers never leave this module: records are de-identified before they reach state, the LLM, or traces."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from functools import lru_cache
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool

from . import store
from .config import COHORT_MIN_CELL, PATIENTS_FILE
from .guardrails import neutralize_injection, redact_text
from .sources import search_phrase

PANEL_SOURCE_ID = "PANEL:synthetic-v1"
GENERIC_WORDS = {"disease", "disorder", "syndrome", "the", "of", "and", "with", "type", "due", "to", "all", "subtypes"}


@lru_cache
def _panel() -> dict[str, dict]:
    data = json.loads(PATIENTS_FILE.read_text())
    return {p["id"]: p for p in data["patients"]}


def patient_ids() -> list[str]:
    return sorted(_panel())


def identifier_denylist() -> list[str]:
    out = []
    for p in _panel().values():
        out += [p["name"], p["name"].split()[-1], p["address"], p["mrn"], p["phone"], p["ssn"]]
    return out


def age_band(dob: str, today: date | None = None) -> str:
    today = today or date.today()
    born = date.fromisoformat(dob)
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return "<65" if age < 65 else "65-74" if age < 75 else "75-84" if age < 85 else "85+"


def deidentify(record: dict) -> tuple[dict, list[str]]:
    """Safe-Harbor style: drop name/MRN/SSN/phone/address, DOB -> age band, dates -> year, scrub free text."""
    events = []
    encounters = []
    for enc in record.get("encounters", []):
        note, removed = neutralize_injection(redact_text(enc["note"]))
        if removed:
            events.append(f"Prompt-injection text removed from {record['id']} encounter note ({enc['date'][:4]}).")
        encounters.append({"year": enc["date"][:4], "type": enc["type"], "note": note})
    return {
        "patient_ref": record["id"],
        "age_band": age_band(record["dob"]),
        "sex": record["sex"],
        "conditions": [{"name": c["name"], "icd10": c["icd10"], "onset_year": c["onset"][:4]} for c in record["conditions"]],
        "stage": record.get("stage"),
        "moca": record.get("moca"),
        "medications": record.get("medications", []),
        "allergies": record.get("allergies", []),
        "biomarkers": record.get("biomarkers", {}),
        "encounters": encounters,
    }, events


@tool
def get_patient_history(
    patient_id: str,
    user: Annotated[str, InjectedToolArg],
    purpose: Annotated[str, InjectedToolArg],
    thread_id: Annotated[str | None, InjectedToolArg] = None,
) -> dict:
    """Return the de-identified medical history for one synthetic patient (identifiers removed, age banded)."""
    pid = patient_id.strip().upper()
    if not re.fullmatch(r"P\d{3}", pid):
        return {"error": "Patient IDs look like P001."}
    record = _panel().get(pid)
    store.audit(user, pid, "read_patient_history" if record else "read_patient_history_not_found", purpose, thread_id)
    if not record:
        return {"error": f"No patient {pid} in the synthetic panel."}
    data, events = deidentify(record)
    return {"record": data, "guardrail_events": events}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", search_phrase(text).lower()) if w not in GENERIC_WORDS}


def _suppress(counter: Counter, k: int) -> dict[str, str]:
    return {name: (str(n) if n >= k else f"<{k}") for name, n in counter.most_common() if n > 0}


def cohort_stats(condition: str, subtype: str, user: str, thread_id: str | None) -> dict:
    """Aggregate, de-identified counts across the panel with small-cell suppression."""
    store.audit(user, "PANEL:*", "cohort_aggregate", f"cohort stats for {condition} / {subtype}", thread_id)
    condition_words = _words(condition)
    subtype_words = _words(subtype) - condition_words
    by_subtype, by_age, by_stage, by_med = Counter(), Counter(), Counter(), Counter()
    matched = 0
    for p in _panel().values():
        hits = [c["name"] for c in p["conditions"]
                if condition_words and condition_words <= _words(c["name"])
                and (not subtype_words or subtype_words & _words(c["name"]))]
        if not hits:
            continue
        matched += 1
        by_subtype[hits[0]] += 1
        by_age[age_band(p["dob"])] += 1
        by_stage[p.get("stage") or "unknown"] += 1
        for med in p.get("medications", []):
            by_med[med["name"]] += 1
    k = COHORT_MIN_CELL
    return {
        "panel": f"{PANEL_SOURCE_ID} ({len(_panel())} synthetic patients)",
        "matched_patients": str(matched) if matched >= k or matched == 0 else f"<{k}",
        "by_subtype": _suppress(by_subtype, k),
        "by_age_band": _suppress(by_age, k),
        "by_stage": _suppress(by_stage, k),
        "by_medication": _suppress(by_med, k),
        "min_cell": k,
        "note": f"De-identified aggregate. Counts below {k} are suppressed. Synthetic panel, not real patients.",
    }
