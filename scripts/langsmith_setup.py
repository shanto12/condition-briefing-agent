"""Idempotent LangSmith setup for the human-review + monitoring layer.

Creates (or reuses): the 'analyst-review' annotation queue with rubric, reviewer feedback configs,
automation rules that sample root briefing runs into the queue, then seeds the queue with recent runs.

Run from build/:
    LANGSMITH_API_KEY=$(security find-generic-password -a shanto -s codex-langsmith-personal-key -w) \
        ../../.venv/bin/python scripts/langsmith_setup.py
"""
from __future__ import annotations

import os
import sys
import warnings

from langsmith import Client
from langsmith import schemas as ls_schemas

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT = os.environ.get("LANGSMITH_PROJECT", "live-build-interview")
ROOT_RUN_NAME = "condition-briefing"
QUEUE_NAME = "analyst-review"
SEED_LIMIT = 10
DATASETS = ["condition-briefing-golden", "briefing-agentic-system", "condition-briefing-unit",
            "briefing-gold-v1", "briefing-rag-followup"]

RUBRIC = {
    "claim_supported": "Every claim carries a citation and the cited source actually supports it.",
    "trial_status_current": "Trial statuses (recruiting/active/completed) match the current registry record.",
    "sponsor_correct": "Trial sponsors and organizations are named correctly.",
    "phi_clean": "No patient identifiers (name, MRN, SSN, DOB, phone, email) appear in the briefing.",
}
QUEUE_DESCRIPTION = (
    "Analyst review of sampled condition briefings. Score 1 = pass, 0 = fail: "
    "(a) every claim cited and supported; (b) trial statuses current; "
    "(c) sponsors named correctly; (d) no patient identifiers. "
    "Any 0 -> add the run to briefing-gold-v1 as a golden case."
)

RULES = [
    {
        "display_name": "analyst-review: 10% of root briefings",
        "sampling_rate": 0.10,
        "filter": f'and(eq(is_root, true), eq(name, "{ROOT_RUN_NAME}"))',
    },
    {
        "display_name": "analyst-review: all has_patient briefings",
        "sampling_rate": 1.0,
        "filter": f'and(eq(is_root, true), eq(name, "{ROOT_RUN_NAME}"), has(tags, "has_patient"))',
    },
]


def ensure_queue(client: Client):
    rubric_items = [ls_schemas.AnnotationQueueRubricItem(
        feedback_key=k, description=v, score_descriptions={"0": "fail", "1": "pass"}) for k, v in RUBRIC.items()]
    instructions = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    existing = next(iter(client.list_annotation_queues(name=QUEUE_NAME)), None)
    if existing:
        client.update_annotation_queue(existing.id, description=QUEUE_DESCRIPTION,
                                       rubric_instructions=instructions, rubric_items=rubric_items)
        return existing, "reused"
    queue = client.create_annotation_queue(name=QUEUE_NAME, description=QUEUE_DESCRIPTION,
                                           rubric_instructions=instructions, rubric_items=rubric_items)
    return queue, "created"


def ensure_feedback_configs(client: Client) -> dict[str, str]:
    have = {c.feedback_key for c in client.list_feedback_configs(feedback_key=list(RUBRIC))}
    status = {}
    config = ls_schemas.FeedbackConfig(type="continuous", min=0, max=1)
    for key in RUBRIC:
        if key in have:
            status[key] = "reused"
            continue
        client.create_feedback_config(key, feedback_config=config)
        status[key] = "created"
    return status


def seed_queue(client: Client, queue_id) -> tuple[int, int]:
    runs = list(client.list_runs(project_name=PROJECT, is_root=True, filter=f'eq(name, "{ROOT_RUN_NAME}")',
                                 limit=SEED_LIMIT, select=["id"]))
    already = {str(r.id) for r in client.list_runs_from_annotation_queue(queue_id, limit=500)}
    new_ids = [r.id for r in runs if str(r.id) not in already]
    if new_ids:
        client.add_runs_to_annotation_queue(queue_id, run_ids=new_ids)
    return len(new_ids), len(runs)


def ensure_rules(client: Client, project_id, queue_id) -> dict[str, str]:
    """The SDK has no rule helper; the REST endpoint /runs/rules backs the UI 'Automations' tab."""
    url = f"{client.api_url}/runs/rules"
    resp = client.session.get(url, headers=client._headers, params={"session_id": str(project_id)})
    if resp.status_code != 200:
        return {r["display_name"]: f"NOT CREATED (list rules HTTP {resp.status_code})" for r in RULES}
    existing = {r.get("display_name") for r in resp.json()}
    status = {}
    for rule in RULES:
        if rule["display_name"] in existing:
            status[rule["display_name"]] = "reused"
            continue
        body = {**rule, "session_id": str(project_id), "is_enabled": True,
                "add_to_annotation_queue_id": str(queue_id)}
        r = client.session.post(url, headers=client._headers, json=body)
        status[rule["display_name"]] = "created" if r.status_code in (200, 201) else (
            f"NOT CREATED (HTTP {r.status_code}: {r.text[:200]})")
    return status


def main() -> int:
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY not set", file=sys.stderr)
        return 1
    client = Client()
    web = os.environ.get("LANGSMITH_WEB_URL", "https://smith.langchain.com")
    project = client.read_project(project_name=PROJECT)
    org = project.tenant_id

    queue, queue_status = ensure_queue(client)
    fb_status = ensure_feedback_configs(client)
    added, seen = seed_queue(client, queue.id)
    rule_status = ensure_rules(client, project.id, queue.id)
    in_queue = len(list(client.list_runs_from_annotation_queue(queue.id, limit=500)))

    print(f"Project   {PROJECT}: {web}/o/{org}/projects/p/{project.id}")
    print(f"Queue     {QUEUE_NAME} ({queue_status}): {web}/o/{org}/annotation-queues/{queue.id}")
    print(f"          seeded {added} new of {seen} recent '{ROOT_RUN_NAME}' root runs; {in_queue} runs in queue")
    for key, s in fb_status.items():
        print(f"Feedback  {key}: {s}")
    for name, s in rule_status.items():
        print(f"Rule      {name}: {s}")
    for name in DATASETS:
        if client.has_dataset(dataset_name=name):
            ds = client.read_dataset(dataset_name=name)
            print(f"Dataset   {name}: {web}/o/{org}/datasets/{ds.id}")
        else:
            print(f"Dataset   {name}: not present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
