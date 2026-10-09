# Condition Briefing Copilot (demo)

Chat tool for a health system strategy team: enter a condition, answer three clarifying questions, get a
schema-locked briefing (standard of care, emerging treatments, key companies and institutions, de-identified
panel stats, optional patient context) where every claim cites a PMID, NCT ID, or FDA application number.
An analyst approves the draft before it is marked final.

## Run

```bash
cd build
../../.venv/bin/python -m uvicorn app.main:app --port 8000   # open http://127.0.0.1:8000
../../.venv/bin/python -m evals.run_eval                     # golden set (add --offline for no LLM/network)
```

Secrets: `DEEPSEEK_API_KEY` and `LANGSMITH_API_KEY` from the environment, else macOS Keychain
(`codex-deepseek-interviews`, `codex-langsmith-personal-key`). Traces: LangSmith project `live-build-interview`.
Config: `HEALTH_SYSTEM_STATE` (default Illinois), `COHORT_MIN_CELL` (default 3), `BRIEFING_DATA_DIR`.

## Flow

`intake -> clarify (interrupt) -> supervisor -> [standard_of_care | pipeline | landscape | cohort | patient_context] -> synthesize -> verify -> approval (interrupt)`

- Sources: PubMed E-utilities, openFDA drug labels, ClinicalTrials.gov v2 (no keys). Responses cached in `fixtures/`; on outage the cache is served and flagged.
- Guardrails: cite-or-drop verifier, clinical-request refusal, untrusted-text injection filter, PHI redaction before LLM and traces, append-only audit log, small-cell suppression.
- Storage: `data/app.db` (briefings, audit_log), `data/checkpoints.db` (LangGraph sessions). Patient context is shown in chat but never persisted with the briefing.

## Not production

Synthetic patients only. DeepSeek is not a BAA-covered endpoint: real PHI requires Azure OpenAI, Bedrock, or a
self-hosted model under a BAA. `X-User` header stands in for SSO. Not HIPAA compliant as built.
