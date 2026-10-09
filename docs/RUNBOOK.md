# Runbook: Condition Briefing Copilot (MVP)

One FastAPI process (`app.main:app`) running the LangGraph pipeline. All state lives in one data directory:
`app.db` (briefings, sessions, audit log), `checkpoints.db` (LangGraph) and `chroma/` (document index).
Demo build on synthetic patients; not cleared for real PHI.

## Before any deployment: SSO

The `X-User` header is a demo identity that the client supplies, and the server trusts it as-is. It **must be
replaced by SSO** before this is deployed anywhere reachable. Use OIDC (Entra ID or Okta) with validated tokens and
role claims, at an auth proxy or in app middleware, and restrict `/api/audit` to a compliance role.

## Local run

```bash
cd build
../../.venv/bin/python -m app.rag ingest                   # first time only: builds data/chroma (about 1 minute)
../../.venv/bin/python -m uvicorn app.main:app --port 8000  # binds 127.0.0.1 only; open http://127.0.0.1:8000
```

Secrets come from environment variables first. On the demo laptop, missing variables fall back to macOS Keychain.

## Container run

The image has **not been built yet** (no Docker on the build machine). Build it with BuildKit (the default since
Docker 23):

```bash
cd build
docker build -t condition-briefing:0.1.0 .
docker volume create briefing-data
docker run -d --name briefing -p 127.0.0.1:8000:8000 -v briefing-data:/data \
  -e DEEPSEEK_API_KEY -e OPENAI_API_KEY -e PINECONE_API_KEY -e LANGSMITH_API_KEY \
  condition-briefing:0.1.0
```

- **Network binding:** uvicorn binds `0.0.0.0` **only inside the container**, which it needs so the published port can reach it. On the host, publish to `127.0.0.1` as above, or put it behind the ingress or load balancer. Never publish the port directly on a public interface.
- **Data:** `/data` holds both SQLite files, `chroma/` and the public API cache. Run one replica on block storage; SQLite breaks on NFS or SMB.
- **First start:** if the index is missing, the entrypoint runs `python -m app.rag ingest`, then starts uvicorn. If ingest fails, for example because there is no embedding key, the container exits instead of serving.
- **Image contents:** runs as UID 10001 (not root); no secrets or databases are baked in.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | (secret) | LLM; required |
| `OPENAI_API_KEY` | (secret) | Embeddings; required for ingest and search |
| `PINECONE_API_KEY` | (secret) | Reranker; without it, retrieval falls back to a cosine threshold |
| `LANGSMITH_API_KEY` | (secret) | Traces and feedback; optional |
| `BRIEFING_DATA_DIR` | `data/`, or `/data` in the container | SQLite files and `chroma/` |
| `BRIEFING_FIXTURE_DIR` | `fixtures/`, or `/data/fixtures` in the container | Cached public API responses |
| `HEALTH_SYSTEM_STATE` | `Illinois` | Market for the landscape section |
| `EMBED_PROVIDER` / `EMBED_MODEL` | `openai` / `text-embedding-3-large` | Embeddings; each pair has its own collection |
| `RERANK` / `RERANK_MODEL` | `on` / `bge-reranker-v2-m3` | Pinecone-hosted reranking |
| `RERANK_MIN_SCORE` | `0.3` | Below this, a chunk counts as "not in our documents" |
| `RETRIEVE_TOP_K` | `5` | Chunks passed to the model |
| `BRIEFING_TRACING` / `LANGSMITH_PROJECT` | `on` / `live-build-interview` | LangSmith tracing |
| `PORT` | `8000` | Container listen port |
| `INGEST_ON_START` | `missing` | Container only: `missing`, `always` or `off` |

## Secrets via a vault

In a container the app reads secrets **only** from environment variables; the Keychain fallback is local-only.
Store the four keys in AWS Secrets Manager or Azure Key Vault and let the orchestrator inject them at start:
- **ECS:** the task definition's `secrets` / `valueFrom`.
- **EKS or AKS:** the Secrets Store CSI driver.
- **Azure Container Apps:** `keyvaultref:` plus `secretref:`.

Never put them in the image, the repo or a committed env file. Rotate in the vault, then restart.

## Health check

`GET /healthz` returns `{"ok": true}`. It is liveness only: it doesn't check keys, the databases or the index. The
image's `HEALTHCHECK` calls it every 30 s with a 120 s start period, because the first start ingests the corpus
before uvicorn binds.

## Scaling path

1. **Postgres:** move checkpoints to `langgraph-checkpoint-postgres` (`PostgresSaver`) and the app tables to the same database. This is the step that allows more than one replica.
2. **Managed vector store:** Pinecone serverless, pgvector or Azure AI Search, behind the existing `DocumentRetriever` interface.
3. **SSO instead of the `X-User` header:** required before deployment (see the top of this runbook).
4. **A BAA-covered model endpoint before any real PHI:** Azure OpenAI, Bedrock, or OpenAI under a BAA. DeepSeek is not covered, and every vendor that sees document or patient text needs a BAA.

## Verified so far

The entrypoint script was extracted from the Dockerfile and run as-is on port 8002, against an empty temp directory
standing in for `/data`:
- **First start:** it ingested 11 documents and served `/healthz`, and the `HEALTHCHECK` command passed.
- **Restart:** it skipped ingest.
- **No OpenAI key:** it exited before serving.

The pinned requirements resolve with Linux wheels only (amd64 and arm64). Not verified: `docker build` and `docker run`.
