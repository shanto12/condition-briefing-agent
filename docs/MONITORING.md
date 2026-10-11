# Monitoring and human review (LangSmith)

Project: `live-build-interview`. Setup is scripted and idempotent:

```bash
cd build
LANGSMITH_API_KEY=$(security find-generic-password -a shanto -s codex-langsmith-personal-key -w) \
  ../../.venv/bin/python scripts/langsmith_setup.py
```

It creates or reuses the `analyst-review` annotation queue (with rubric), the four reviewer feedback keys, and two
automation rules. It also seeds the queue with the 10 most recent `condition-briefing` root runs and prints the
project, queue, and dataset URLs.

## What's traced, what's redacted

- **Traced:** every briefing graph run (root run `condition-briefing`) and its child nodes: retrieval, LLM calls,
  guardrails, tools. Tracing goes only through the explicit tracer in `app/config.py`. The environment-driven
  default tracer is turned off so nothing bypasses redaction.
- **Metadata:** hashed user ID, condition, subtype, `briefing_id`, embedding model. **Tags:** `has_patient`,
  `stale_source`.
- **Redacted before upload:** inputs and outputs pass through `guardrails.redact_payload` (the `hide_inputs` and
  `hide_outputs` hooks). SSN, MRN, DOB, phone, and email are replaced with placeholders. User IDs are hashed, never
  sent raw.
- **Feedback:** approve or reject in the app writes `analyst_approval` (1.0 or 0.0). Reviewers in the queue write
  `claim_supported`, `trial_status_current`, `sponsor_correct`, and `phi_clean` (0 or 1).

## Metrics and alert thresholds

| Metric | Source | Alert |
|---|---|---|
| Rejection rate | mean of `analyst_approval` = 0, 7-day rolling | > 15%, or +5 pts week over week |
| Groundedness | `claim_supported` mean (queue) + LLM-judge online eval | < 0.95 |
| Citation-drop rate | share of briefings where guardrail stripped uncited claims | > 10% |
| Stale-source rate | share of runs tagged `stale_source` | > 20%, or any jump after a corpus refresh |
| PHI leak | `phi_clean` = 0 | **any single case, page immediately** |
| Latency | root run p50 / p95 | p95 > 60 s (p50 > 25 s warn) |
| Cost per briefing | LangSmith token cost per root run | > 2x the trailing 7-day median |
| Error rate | root runs with `error` | > 2% over 1 h |

Set these as LangSmith monitoring alerts on the project's root-run filter `eq(name, "condition-briefing")`. Wire
the PHI alert to on-call, not just a dashboard.

## Review-queue workflow

1. Runs reach `analyst-review` through the automation rules (below) or when added by hand from a trace.
2. The analyst opens the next run and scores the rubric. Each item is 1 = pass, 0 = fail:
   (a) `claim_supported`: every claim is cited and the citation supports it;
   (b) `trial_status_current`: trial statuses match the registry;
   (c) `sponsor_correct`: sponsors are named correctly;
   (d) `phi_clean`: no patient identifiers.
3. **Any 0 becomes a golden case:** use "Add to dataset" to send the run to `briefing-gold-v1`. Write the
   corrected expected output (the right status, sponsor, or citation) and a `failure_mode` tag in metadata.
4. The next eval run on `condition-briefing-golden` + `briefing-gold-v1` must pass that case before the fix ships.
   The golden set only grows; cases are never deleted silently.

## Online-eval sampling plan

These two rules were **created via the `/runs/rules` API** and are visible under Project > Automations:

- `analyst-review: 10% of root briefings`: samples 10% of `condition-briefing` root runs into the queue.
- `analyst-review: all has_patient briefings`: sends 100% of `has_patient` runs to the queue. Patient context is
  the highest-risk path for PHI and personalization errors.

Next step, **not yet created**: an LLM-as-judge online evaluator for groundedness on the same 10% sample. In the
UI, go to Project > Automations > New rule. Use the same filter, sampling 0.10, action "Online evaluation", with a
judge prompt that checks claims against the retrieved context and writes `groundedness` (0 to 1). Calibrate the
judge against ~30 analyst-scored runs before alerting on it. Raise the sampling rate to 25% for the first two weeks
after any change.

## Re-evaluating on drift

Run the full eval suite (`evals/`) against all golden datasets and compare to the last baseline experiment before
promoting any of these changes:

- **Model change:** a new DeepSeek version, a provider switch, or a new embedding or reranker model. Experiments
  are already tagged by embedding model.
- **Policy or prompt change:** guardrail rules, citation policy, the rerank threshold (`RERANK_MIN_SCORE`), or
  cohort minimum-cell size.
- **Corpus or registry refresh:** a new corpus ingest, or trial-registry status churn (watch `stale_source`).
- **Live signal:** the rejection rate or groundedness alert fires, or reviewer scores trend down for 2 weeks.

Promotion gate: no regression on PHI (must stay 100%), groundedness within 2 points of baseline, and p95 latency
within 20%. Otherwise roll back, or ship in shadow mode with a canary.
