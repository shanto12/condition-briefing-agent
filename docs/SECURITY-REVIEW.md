# Security and PHI review (manual, first pass)

- **Reviewed:** Friday Oct 9 2026, 12:21 to 12:45 CDT. Re-check planned for about 1:15 PM.
- **Scope:** the working tree of `build/`: commit `a420f9a` plus uncommitted changes as of 12:35. That includes
  the new `app/rag.py`, the RAG wiring and follow-up agent in `app/graph.py`, sessions and uploads in `app/main.py`,
  `app/service.py` and `app/store.py`, `app/render.py`, `app/static/*`, `evals/run_eval.py`,
  `evals/run_retrieval_eval.py`, `corpus/` and `README.md`. Files that landed after 12:35 (`tests/`,
  `evals/run_rag_eval.py`, `evals/run_agentic_eval.py`, `evals/judges.py`, `.dockerignore`, the `Dockerfile`
  changes) are left for the re-check.
- **Locations** name the function, because line numbers shift while other agents edit. Where line numbers are
  given, they come from the 12:30 snapshot.
- **Method:** manual code review, secret and PHI pattern scans, read-only queries of the local SQLite stores,
  offline Python checks, and HTTP requests to the local server on port 8000. The Aikido scanner was **not** run;
  this review stands in for it. It is not a penetration test and makes no compliance claim. The build is
  **not HIPAA compliant** (see README, "Known gaps").

## Verdict

**Ready to publish to a public repo: YES, on condition that H1 is fixed (or the README claim corrected) before the
push and the 1:15 re-check passes.**

- No secrets are in tracked files, untracked publishable files or git history. Keys are not printed, logged,
  returned by the API or serialized into traces.
- No real PHI. All patient and document data is synthetic and labeled. Databases, Chroma, eval data and `.env` are
  git-ignored.
- One High finding (H1): patient references persist in shared briefing records and transcripts, which every user
  can read. This contradicts a control stated in the README. It is a runtime design gap, not a leak through the
  repo, but the README should not state a control that isn't true when the repo goes public.
- The follow-up agent passes the first look. Its prompt is built only from population-level briefing sections and
  non-patient sources, it has no patient tools, it refuses patient-specific questions, and its answers go through
  the citation verifier. Remaining gaps are M2, M3 and M7.

## Findings

Severity reflects this build as a public demo on synthetic data. Severity tags such as "(deploy)" mean the risk
only applies if the service is reachable beyond localhost.

| ID | Severity | Issue | Location | Fix | Owner file |
|---|---|---|---|---|---|
| H1 | High | Patient linkage is persisted in shared records. `save_draft` stores everything except `patient_context`, so `sources` keeps `PATIENT:Pxxx`, `guardrail_events` keeps `patient=Pxxx` and injection events name the patient, and `verification.removed` can hold removed patient-context claim text. The saved transcript has the same content. Any user can read briefings through `GET /api/briefings/{id}`. This contradicts the README PHI bullet ("never stored with the briefing") and the `save_draft` docstring. | `app/store.py` `save_draft`; `app/service.py` `_transcript_text`; data comes from `app/graph.py` `verify`; exposed by `app/main.py` `briefing()` | Add one helper and use it in both `save_draft` and `_transcript_text` (code below). | `app/store.py`, `app/service.py` (sessions/UI agent) |
| M1 | Medium | The audit log is readable by every user. `GET /api/audit` returns who read which patient, the purpose, and `thread_id`, to any `X-User` value. | `app/main.py` `audit()`; `app/store.py` `list_audit` | Add an `AUDIT_VIEWERS` env allowlist (for example `compliance.demo`). Everyone else gets only their own rows: `list_audit(user=None if user in AUDIT_VIEWERS else user)` with `WHERE user = ?`. | `app/main.py`, `app/store.py` |
| M2 | Medium | Uploaded documents can poison retrieval. Uploads are global: `graph.internal_documents` and `search_documents` search every chunk with no source filter, so an upload can be cited in every user's briefing at once. The briefing then labels it "internal document, synthetic" (`render._source_lines`). Any user can overwrite another user's upload by reusing its `doc_id` in front matter; only corpus documents are protected. The injection filter is regex-only: 3 of 4 paraphrased injections passed in a test (evidence below). | `app/rag.py` `Index.add`, `parse`; `app/render.py` `_source_lines` | In `Index.add`, reject when `existing["source"] == "upload" and existing["uploaded_by"] != doc["uploaded_by"]`. Namespace uploads (`doc_id = f"upload-{slug}"`). Put `source` in `document_source` and render "uploaded by X, unreviewed" instead of "synthetic" for uploads. Best option: the briefing retriever filters `where={"source": "corpus"}` until an upload is reviewed. | `app/rag.py`, `app/graph.py`, `app/render.py` (RAG agent) |
| M3 | Medium | The PHI screen is pattern-only. It misses unlabeled names, street addresses, emails (the `EMAIL` regex exists but `phi_findings` doesn't use it) and dates without a "DOB" label. Uploaded text that passes is sent to OpenAI (embeddings) and Pinecone (rerank). Chat messages are masked with the same patterns plus the synthetic panel's own identifiers, so an unknown patient's name in a question passes through unchanged. | `app/guardrails.py` `phi_findings`, `redact_text` | Add `("email", EMAIL)`. Add a label rule `\b(patient\|pt\|name\|dob\|mrn\|member id)\s*[:#]` and a street-address rule `\b\d{1,5}\s+[A-Z][a-z]+(\s[A-Z][a-z]+)*\s(St\|Ave\|Rd\|Dr\|Ln\|Ct\|Blvd)\b`. In the README, call it a "pattern-based PHI screen", not a guarantee. | `app/guardrails.py` (RAG agent), `README.md` |
| M7 | Medium | Follow-up tool arguments are unfiltered free text. The model writes `topic`, `condition` and `query` from the user's question, and they go to PubMed, ClinicalTrials.gov and openFDA (public APIs) and to OpenAI and Pinecone (document search). Together with M3, a non-panel patient's name in a follow-up question could end up in a public API query. The intake path already strips the condition to `[\w\s'\-,]` and 80 characters; the tools don't. | `app/graph.py` `search_guidelines`, `search_trials`, `search_fda_approvals`, `search_documents`, `followup` | At the top of each tool: `q = re.sub(r"[^\w\s'\-,]", "", redact_text(arg))[:80]`. Refuse the follow-up when `phi_findings(question)` is non-empty, next to the `PATIENT_REQUEST` check. | `app/graph.py` (RAG/follow-up agent) |
| M4 | Medium (deploy) | Upload size is checked only after parsing, and PDFs have no page cap. Starlette spools the whole multipart body to a temp file before the 2 MB `read()` check runs, and `_decode` extracts text from every PDF page. | `app/main.py` `upload_document`; `app/rag.py` `_decode` | Add middleware: for `POST /api/documents`, return 413 when `Content-Length` is over 2.5 MB and 411 when it is missing. In `_decode`, reject PDFs over 50 pages and truncate extracted text to about 400k characters. | `app/main.py`, `app/rag.py` |
| M5 | Medium | Approval is self-approval. After the ownership fix, only the requester can approve their own draft, so the approval pause records the requester's sign-off rather than a second person's review. | `app/service.py` `chat_turn` (approval branch); `app/graph.py` `approval` | Demo: state in the README that approval is the requester's sign-off. Production: a reviewer role from SSO claims, and block approver == requester. | `README.md` (deck/README agent) |
| M6 | Medium (deploy) | Identity is a client-supplied `X-User` header, and the Dockerfile binds `0.0.0.0`. Anyone who can reach the port can act as any user. This is documented as a demo limitation. | `app/main.py` `require_user`; `Dockerfile` CMD | Keep the demo on `127.0.0.1`. The runbook should require an authenticating proxy (OIDC) that sets the identity header and strips any client-supplied `X-User`. | `Dockerfile`, `docs/RUNBOOK.md` (deploy agent) |
| L1 | Low | The retrieval eval uses an unredacted LangSmith client: `Client(api_key=...)` has no `hide_inputs` or `hide_outputs`. Today it only carries synthetic questions and corpus chunks. | `evals/run_retrieval_eval.py` `langsmith_experiments` | `client = config.tracer().client`, the same redacting client the golden-set eval already uses. | `evals/run_retrieval_eval.py` (RAG agent) |
| L2 | Low | LangSmith metadata and errors are not redacted. `hide_metadata` is available in langsmith 0.12.6 but unused, and run errors and tracebacks are not covered by `hide_outputs`. Current metadata (`thread_id`, `user_hash`, `condition`, `subtype`, `briefing_id`) contains no PHI. | `app/config.py` `tracer` | `Client(api_key=key, hide_inputs=redact_payload, hide_outputs=redact_payload, hide_metadata=redact_payload)`. Keep metadata PHI-free. | `app/config.py` (coordinator) |
| L3 | Low | `redact_payload` passes dataclasses through unchanged (for example LangGraph `Command` and `Interrupt`). Today they only carry already-redacted text and IDs. | `app/guardrails.py` `redact_payload` | Add `if dataclasses.is_dataclass(obj) and not isinstance(obj, type): return redact_payload(dataclasses.asdict(obj))`. | `app/guardrails.py` |
| L4 | Low | `GET /api/briefings/{id}` returns `thread_id` to any user. The new session ownership checks mean this no longer grants control of the session, but the exposure is unnecessary. | `app/store.py` `get_briefing` | `record.pop("thread_id", None)` in `get_briefing`. | `app/store.py` |
| L5 | Low | The markdown renderer turns any https link inside untrusted text into a clickable anchor (LLM claims, trial titles, upload titles). That allows phishing links. It cannot run scripts: the renderer escapes first and the CSP blocks inline script. | `app/static/app.js` `inline()` | Only linkify allowlisted hosts (`pubmed.ncbi.nlm.nih.gov`, `clinicaltrials.gov`, `www.accessdata.fda.gov`, `smith.langchain.com`); render any other link as plain text. | `app/static/app.js` (UI agent) |
| L6 | Low | Chroma's anonymized telemetry is on by default (`chromadb` 1.5.9 `Settings().anonymized_telemetry == True`). It sends usage events, not document text. | `app/rag.py` `Index.__init__` | Pass `client_settings=chromadb.config.Settings(anonymized_telemetry=False)` to `Chroma(...)`, or set `ANONYMIZED_TELEMETRY=False`. | `app/rag.py` |
| L7 | Low | The audit log is append-only (triggers block UPDATE and DELETE) but not tamper-evident: anyone with file access can drop the triggers or the table. Checkpoints and transcripts have no retention limit. | `app/store.py` `SCHEMA` | Production: send audit events to WORM storage or a SIEM (or hash-chain the rows), and add a retention and purge job. Document this in the README and deck. | `README.md`, `docs/PRESENTATION.md` |
| L8 | Low | Synthetic data could resemble real records. 4 of 10 fake SSNs (`900-55-1903`, `900-60-2217`, `900-71-3340`, `900-82-0456`) match the IRS ITIN number format, and the addresses are real streets in real towns. All are labeled synthetic. | `data/patients.json` | Optional: switch to `000-xx-xxxx` (never issued) and fictional addresses. Keep `900-12-4481` or update the redaction check in `evals/run_eval.py` to match. | `data/patients.json` (coordinator) |
| L9 | Low | Transitive dependencies are unpinned: no lock file or hashes, and no CVE scan was run because `pip-audit` isn't installed. | `requirements.txt` | Generate `requirements.lock` with hashes (`uv pip compile --generate-hashes`) and run `pip-audit` in CI. | coordinator |
| L10 | Low (hygiene) | The `.gitignore` rule `fixtures/` also ignores `evals/fixtures/`, so `upload_phi.md` and `upload_injection.md` won't be published and the RAG tests will fail on a fresh clone. The offline API cache (`fixtures/`) isn't published either, although the plan says it should be. Both sets are synthetic or public data and safe to publish. | `.gitignore` | Change `fixtures/` to `/fixtures/` (root only) or add `!evals/fixtures/`. Decide whether to commit `/fixtures/` (about 11 MB of public API responses). | `.gitignore` (coordinator) |
| I1 | Info | `/docs`, `/redoc` and `/openapi.json` are exposed. FastAPI 422 responses echo rejected input back to the caller (the same caller only; not logged). `user_hash` is an unsalted SHA-256 of the username. | `app/main.py` `FastAPI(...)`; `app/service.py` `user_hash` | Deploy: `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)` behind an env flag. Salt the user hash with a secret. | `app/main.py`, `app/service.py` |

### Fix for H1

Put this in `app/store.py` (add `import re`), call it in `save_draft` instead of
`{**briefing, "patient_context": None}`, and call it in `service._transcript_text` before `briefing_markdown`:

```python
def population_view(b: dict) -> dict:
    """The shareable copy of a briefing: no patient section, patient sources, patient IDs or removed patient claims."""
    v = b.get("verification", {})
    return {**b, "patient_context": None,
            "sources": [s for s in b.get("sources", []) if s.get("type") != "patient"],
            "guardrail_events": [e for e in b.get("guardrail_events", []) if not re.search(r"\bP\d{3}\b", e)],
            "verification": {**v, "removed": [r for r in v.get("removed", []) if r.get("section") != "patient_context"]}}
```

Check: after one briefing with `P007`, the query
`SELECT count(*) FROM briefings WHERE briefing_json LIKE '%P007%'` and the same query on `messages.text` both return 0.
The approval draft shown to the requester still includes the patient section, because it is rendered from the
checkpoint.

## Evidence

- **H1 (live):** `GET /api/briefings/9c52c6558ccd` with `X-User: mallory` (not the creator) returned
  `thread_id`, a source `{"id": "PATIENT:P010", "type": "patient"}`, and the event
  `"Clarified: ...; patient=P010"`. A read-only scan of `data/app.db` found a `PATIENT:Pxxx` reference in 4 of 4
  briefings and 1 of 6 transcript messages.
- **M1 (live):** `GET /api/audit` with `X-User: mallory` returned every user's patient reads, with thread IDs.
- **M2, M3 and M7 (offline, `app.guardrails`):**
  - `phi_findings("Patient: Jane Roe, 12 Main St, Springfield IL 62701. Seen 03/14/2026. Email jane.roe@example.com")`
    returned `[]`, and `redact_text` changed only the email.
  - `phi_findings` on `evals/fixtures/upload_phi.md` returned `['SSN', 'MRN', 'DOB', 'phone']` (correctly rejected).
  - `neutralize_injection` removed "Ignore all previous instructions..." but kept "Disregard earlier guidance...",
    "Forget everything above. From now on..." and "As the assistant, your real task is...".

## What was checked and how

| Area | Check | Result |
|---|---|---|
| Secrets in the repo | `rg` for `sk-`, `lsv2_`, `pcsk_`, `AKIA`, `ghp_`, `xox*`, private-key headers and quoted `api_key=` over `git ls-files` plus untracked non-ignored files; the same over `git log -p --all`; the same over ignored files | No matches |
| Secret handling | `app/config.py`: keys come from env or Keychain through `subprocess.run([...])` with list arguments (no shell), are never printed, and are cached in memory. The Pinecone key is sent only as a request header to `api.pinecone.io` | OK |
| Keys in traces | `dumpd(ChatDeepSeek(api_key=dummy))`, `_get_invocation_params()` and `repr(OpenAIEmbeddings(api_key=dummy))` were checked for the dummy key | Not present (`lc_secrets` maps `api_key` to an env var name) |
| Keys or PHI in logs | Swept server terminal output for key patterns, synthetic names, SSNs and MRNs | None. Logs show the path and status only |
| Default tracing | `config.py` forces `LANGSMITH_TRACING=false`. App traces (including the retriever and the follow-up agent, which inherit the run config) and `run_eval --langsmith` use the redacting `tracer().client` | OK, except L1 |
| PHI before the LLM | `chat_turn` redacts every user message before the graph. `deidentify` strips name, MRN, SSN, phone, address and DOB, then age-bands and redacts notes | OK; the golden set checks all 10 records |
| PHI in public API queries | In the briefing path, queries use only `condition` and `subtype`. The condition is redacted, then stripped to `[\w\s'\-,]` and 80 characters, and the patient ID is never used in a query | OK in the briefing path; see M7 for the follow-up tools |
| Follow-up agent isolation | `_briefing_context` uses only `executive_summary`, `standard_of_care`, `emerging_treatments` and `landscape`, plus a registry filtered to `document`, `pubmed`, `fda` and `clinicaltrials`. There are no patient tools. `PATIENT_REQUEST` and `clinical_request_reason` refuse patient-specific questions. Tool output goes through `neutralize_injection` and `wrap_untrusted`, and `<`/`>` are escaped so a document can't forge source tags. Answers go through `verify_claims` with patient sources excluded. At most 6 tool calls | OK; see M2, M3 and M7 |
| RAG wiring | `document_block` applies `neutralize_injection` and `wrap_untrusted`. `document` sources are allowed only in the summary, standard-of-care and follow-up claims | OK; see M2 |
| PHI in stores | Read-only scan of `app.db` (briefings, messages, audit log) and the checkpoint DB plus WAL for synthetic names, SSNs and MRNs | 0 direct identifiers. Patient references are present (H1) |
| Audit | `get_patient_history` writes the audit row before returning data, and the tool raises if the write fails, so nothing is disclosed without an audit row. Cohort reads are audited as `PANEL:*`. Triggers block UPDATE and DELETE (golden-set check) | OK; see L7 |
| AuthZ | Missing or invalid `X-User` returns 401 (live). `/api/chat` returns 404 for a non-owner of a session, with a second owner check in `chat_turn` that covers older threads without a session row. `/api/sessions/{id}` is owner-only and returns 404. `/api/sessions` lists only the caller's sessions | OK; see M1, M5, L4 |
| Input validation | Sent a 3,000-character message and `thread_id="../../etc"` | Both return 422 (live). `action` is restricted to approve or reject |
| Upload | The filename is reduced to its base name; extension allowlist; 2 MB read cap; the file is never written to disk (no path traversal); `doc_id` is slugged; uploads cannot replace corpus documents; the PHI gate runs before embedding | OK; see M2 to M4 |
| SQL | All queries use parameters. The dynamic column names in `upsert_session` come from an allowlist | OK |
| XML | PubMed XML is parsed with `defusedxml` | OK |
| XSS / CSP | `app.js` escapes all HTML before applying markdown, links must be https and get `rel="noopener noreferrer"`, and user text, history, sessions, documents, upload status and the audit view use `textContent`. The trace link is built in the DOM with an https check. Live headers on `/` and `/static/*`: CSP `script-src 'self'` with no inline script, `frame-ancestors 'none'`, `nosniff`, `no-referrer`, `no-store` | OK; see L5 |
| Dependencies | `requirements.txt` pins every direct dependency with `==` (including `python-multipart`, `pypdf` and `chromadb`). `pip list --outdated` returned none of them | OK; see L9 |
| Repo hygiene | `git ls-files` plus untracked and ignored listings | Would publish: the app code, `data/patients.json` (notice: "SYNTHETIC DATA ONLY"), `corpus/*.md` (11 files, each with `synthetic: true` and a body notice), `evals/*.py`, `evals/*.jsonl`, `Dockerfile`, `README.md`, `docs/`. Ignored: `data/*.db*`, `data/chroma/`, `evals/.evaldata/`, `evals/results/`, `fixtures/`, `evals/fixtures/`, `__pycache__/`, `.env`. The commit author uses a GitHub noreply address |

## Not checked

- No SAST or CVE scan: Aikido wasn't run and `pip-audit` isn't installed.
- I did not pull stored runs from LangSmith to confirm redaction on the server side. Suggested spot check: open the
  trace of the `P007` run and confirm the patient-context inputs show `[REDACTED]` and no SSN, MRN or phone number.
  The `followup` node's traced input is the whole thread state, including the de-identified patient section, with
  identifiers masked by `hide_inputs`.
- Files that landed after 12:35 (`tests/`, `evals/run_rag_eval.py`, `evals/run_agentic_eval.py`,
  `evals/judges.py`, `.dockerignore`, the `Dockerfile` changes, `requirements-dev.txt`) and the final README and
  deck.
- I ran no live briefings, follow-ups or uploads against port 8000, to avoid changing demo state. The approval,
  ownership and follow-up behavior was confirmed from the code.
- The Docker image wasn't built (Docker isn't installed).
- I didn't verify the DeepSeek, OpenAI or Pinecone data-retention terms. The README already says DeepSeek has no BAA.
- No load or denial-of-service testing.

## Must verify at re-check (about 1:15 PM)

1. **H1** is fixed (re-run the `LIKE '%P0%'` scan after a fresh patient briefing), or the README PHI bullet is
   corrected.
2. Status of **M1, M2, M3 and M7**: fixed, or listed under "Known gaps" in the README.
3. **New files:** tests and evals don't create unredacted LangSmith clients or write PHI-like fixtures outside
   ignored paths. `.dockerignore` excludes `data/*.db*`, `data/chroma/`, `.env` and `evals/.evaldata/`.
4. **Pre-push:** `git diff --cached --name-only` contains no `data/*.db*`, `data/chroma/`, `.env`, `evals/.evaldata/`
   or `evals/results/`. Re-run the secret grep on the staged tree.
