"""Public evidence sources, no API keys: PubMed E-utilities, openFDA drug labels, ClinicalTrials.gov v2.

Each successful response is cached under fixtures/. If a source is down, the cached copy is served and
flagged stale so the briefing says so instead of silently going thin.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import date

import httpx
from defusedxml import ElementTree as ET

from .config import FIXTURE_DIR, HTTP_TIMEOUT

PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"
OPENFDA = "https://api.fda.gov/drug/label.json"
ACTIVE_STATUSES = "RECRUITING,ACTIVE_NOT_RECRUITING,NOT_YET_RECRUITING,ENROLLING_BY_INVITATION"
TRIAL_FIELDS = ",".join([
    "NCTId", "BriefTitle", "OverallStatus", "Phase", "LeadSponsorName", "LeadSponsorClass",
    "InterventionName", "InterventionType", "EnrollmentCount", "PrimaryCompletionDate", "Condition",
    "LocationFacility", "LocationCity", "LocationState", "LocationCountry",
])
HEADERS = {"User-Agent": "condition-briefing-demo/0.1"}
REPACKAGER = re.compile(
    r"repack|aphena|bryant ranch|proficient rx|quality care|a-s medication|direct[_ ]rx|preferred pharm|"
    r"nucare|henry schein|lake erie|golden state|medsource|unit dose|asclemed|denton|northwind|pd-rx|"
    r"rebel distrib|stat rx|cardinal health|major pharm|avkare|american health packaging", re.I)


class SourceUnavailable(RuntimeError):
    pass


def _fixture_path(name: str, url: str, params: dict):
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:16]
    return FIXTURE_DIR / f"{name}-{key}.cache"


def _get(name: str, url: str, params: dict, *, as_text: bool = False, empty_on_404=None):
    """Returns (body, stale). stale=True means the live call failed and a cached fixture was used."""
    path = _fixture_path(name, url, params)
    try:
        response = httpx.get(url, params=params, timeout=HTTP_TIMEOUT, headers=HEADERS)
        if response.status_code == 404 and empty_on_404 is not None:
            return empty_on_404, False
        response.raise_for_status()
        body = response.text if as_text else response.json()
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text)
        return body, False
    except (httpx.HTTPError, ValueError) as exc:
        if path.exists():
            text = path.read_text()
            return (text if as_text else json.loads(text)), True
        raise SourceUnavailable(f"{name} unreachable and no cached copy ({type(exc).__name__})") from exc


def search_phrase(term: str) -> str:
    return re.sub(r"'s\b", "", term).replace('"', "").strip()


# ---------- PubMed: standard of care evidence (guidelines first, then systematic reviews) ----------

def pubmed_evidence(term: str, n_guidelines: int = 5, n_reviews: int = 4) -> tuple[list[dict], bool]:
    phrase = search_phrase(term)
    base = f'("{phrase}"[ti] OR "{phrase}"[majr])'
    queries = [
        (f"{base} AND (guideline[pt] OR practice guideline[pt] OR consensus development conference[pt])", n_guidelines),
        (f"{base} AND (systematic review[pt] OR meta-analysis[pt])", n_reviews),
    ]
    ids: list[str] = []
    stale = False
    for query, n in queries:
        data, was_stale = _get("pubmed-search", f"{PUBMED}/esearch.fcgi", {
            "db": "pubmed", "term": query, "retmode": "json", "retmax": n, "sort": "relevance",
            "datetype": "pdat", "mindate": "2020", "maxdate": str(date.today().year),
        })
        stale |= was_stale
        ids += [i for i in data.get("esearchresult", {}).get("idlist", []) if i not in ids]
    if not ids:
        return [], stale
    xml, was_stale = _get("pubmed-fetch", f"{PUBMED}/efetch.fcgi",
                          {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, as_text=True)
    stale |= was_stale
    articles = []
    for node in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = node.findtext(".//MedlineCitation/PMID")
        title_node = node.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        year = node.findtext(".//JournalIssue/PubDate/Year") or (node.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        abstract = " ".join("".join(t.itertext()) for t in node.findall(".//Abstract/AbstractText"))
        articles.append({
            "pmid": pmid, "title": title, "year": year,
            "journal": node.findtext(".//Journal/ISOAbbreviation") or node.findtext(".//Journal/Title") or "",
            "pub_types": [p.text for p in node.findall(".//PublicationType") if p.text],
            "abstract": abstract[:1400],
        })
    return articles, stale


# ---------- openFDA: approved products whose label indication mentions the condition ----------

def fda_products(terms: list[str]) -> tuple[list[dict], bool]:
    phrases = []
    for term in terms:
        for phrase in (term, term.split()[0] if term.split() and term.split()[0].endswith("'s") else None):
            if phrase and phrase.lower() not in phrases:
                phrases.append(phrase.lower())
    quoted = " ".join(f'"{p.replace(chr(34), "")}"' for p in phrases)
    params = {"search": f"indications_and_usage:({quoted}) AND openfda.application_number:(NDA* BLA*)", "limit": 100}
    data, stale = _get("openfda", OPENFDA, params, empty_on_404={"results": []})
    by_app: dict[str, dict] = {}
    for label in data.get("results", []):
        fda = label.get("openfda", {})
        app = (fda.get("application_number") or [None])[0]
        if not app:
            continue
        manufacturer = (fda.get("manufacturer_name") or [""])[0]
        current = by_app.get(app)
        if current and not REPACKAGER.search(current["sponsor"] or ""):
            continue
        indication = (label.get("indications_and_usage") or [""])[0]
        category = ("diagnostic imaging agent" if re.search(r"\b(PET|SPECT|imaging|radioactive diagnostic)\b", indication)
                    else "biologic" if app.startswith("BLA") else "drug")
        by_app[app] = {
            "application": app,
            "generic_name": (fda.get("generic_name") or ["?"])[0].lower(),
            "brand_name": (fda.get("brand_name") or [None])[0],
            "sponsor": manufacturer,
            "category": category,
            "indication": indication[:700],
            "boxed_warning": (label.get("boxed_warning") or [""])[0][:500],
        }
    return list(by_app.values()), stale


# ---------- ClinicalTrials.gov: active phase 2/3 pipeline, sponsors, sites ----------

_trial_cache: dict[str, tuple[float, list[dict], bool]] = {}
_trial_lock = threading.Lock()


def active_trials(term: str, ttl_seconds: int = 3600) -> tuple[list[dict], bool]:
    """Pipeline and landscape workers run in parallel on the same query; one fetch serves both."""
    key = search_phrase(term).lower()
    with _trial_lock:
        hit = _trial_cache.get(key)
        if hit and time.time() - hit[0] < ttl_seconds:
            return hit[1], hit[2]
        data, stale = _get("ctgov", CTGOV, {
            "query.cond": search_phrase(term),
            "filter.overallStatus": ACTIVE_STATUSES,
            "query.term": "AREA[Phase](PHASE2 OR PHASE3)",
            "fields": TRIAL_FIELDS,
            "pageSize": 500,
        })
        studies = [_flatten_trial(s) for s in data.get("studies", [])]
        _trial_cache[key] = (time.time(), studies, stale)
        return studies, stale


def _flatten_trial(study: dict) -> dict:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    lead = p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    design = p.get("designModule", {})
    interventions = p.get("armsInterventionsModule", {}).get("interventions", [])
    locations = p.get("contactsLocationsModule", {}).get("locations", [])
    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle", ""),
        "status": status.get("overallStatus", ""),
        "phase": "/".join(design.get("phases", [])),
        "sponsor": lead.get("name", "Unknown"),
        "sponsor_class": lead.get("class", "OTHER"),
        "interventions": [i.get("name") for i in interventions if i.get("name")][:4],
        "intervention_types": sorted({i.get("type") for i in interventions if i.get("type")}),
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "primary_completion": status.get("primaryCompletionDateStruct", {}).get("date"),
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "locations": [
            {"facility": loc.get("facility"), "city": loc.get("city"), "state": loc.get("state"), "country": loc.get("country")}
            for loc in locations if loc.get("facility")
        ],
    }
