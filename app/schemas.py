"""Schema-locked briefing contract. The LLM only ever fills Claim lists; everything else is assembled by code."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["pubmed", "clinicaltrials", "fda", "panel", "patient", "document"]


class Source(BaseModel):
    id: str
    type: SourceType
    title: str
    url: str | None = None
    date: str | None = None
    section: str | None = None


class Claim(BaseModel):
    text: str = Field(description="One factual sentence, written for a health system strategy audience.")
    citations: list[str] = Field(
        description="Source IDs copied exactly from the provided sources, e.g. PMID:38772068, NCT06964230, FDA:BLA761269."
    )


class SectionDraft(BaseModel):
    """What a worker LLM is allowed to return."""
    claims: list[Claim] = Field(description="3-7 cited claims. Return fewer if the sources do not support more.")


class ApprovedProduct(BaseModel):
    generic_name: str
    brand_name: str | None = None
    sponsor: str | None = None
    application: str
    category: str


class PipelineItem(BaseModel):
    nct_id: str
    title: str
    phase: str
    status: str
    sponsor: str
    sponsor_class: str
    interventions: list[str]
    enrollment: int | None = None
    primary_completion: str | None = None


class Organization(BaseModel):
    name: str
    kind: str
    active_trials: int
    nct_ids: list[str]


class Site(BaseModel):
    name: str
    location: str
    active_trials: int
    nct_ids: list[str]


class CohortStats(BaseModel):
    panel: str
    matched_patients: str
    by_subtype: dict[str, str]
    by_age_band: dict[str, str]
    by_stage: dict[str, str]
    by_medication: dict[str, str]
    min_cell: int
    note: str


class PatientContext(BaseModel):
    patient_ref: str
    label: str = "Decision support only. Clinician review required."
    claims: list[Claim]


class Briefing(BaseModel):
    condition: str
    subtype: str
    focus: str
    generated_at: str
    executive_summary: list[Claim] = []
    standard_of_care: list[Claim] = []
    approved_products: list[ApprovedProduct] = []
    emerging_treatments: list[Claim] = []
    pipeline: list[PipelineItem] = []
    key_organizations: list[Organization] = []
    key_sites: list[Site] = []
    market_state: str | None = None
    market_sites: list[Site] = []
    landscape: list[Claim] = []
    cohort: CohortStats | None = None
    patient_context: PatientContext | None = None
    gaps: list[str] = []
    sources: list[Source] = []
    verification: dict = {}
    guardrail_events: list[str] = []
