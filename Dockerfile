# syntax=docker/dockerfile:1.7
# Condition Briefing Copilot: one uvicorn process; SQLite (app.db, checkpoints.db) and Chroma live on the /data volume.
# Bump the tag and the digest together.
FROM python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

# The fixture cache must sit on the volume: /srv is read-only for the app user.
# ANONYMIZED_TELEMETRY turns off Chroma's product telemetry.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    ANONYMIZED_TELEMETRY=False \
    BRIEFING_DATA_DIR=/data \
    BRIEFING_FIXTURE_DIR=/data/fixtures \
    PORT=8000 \
    INGEST_ON_START=missing

WORKDIR /srv

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
 && install -d -o app -g app -m 0750 /data

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY corpus ./corpus
COPY data/patients.json ./data/patients.json

COPY --chmod=0755 <<"EOF" /usr/local/bin/briefing-entrypoint
#!/bin/sh
# Builds the document index on a fresh volume, then execs uvicorn as PID 1 so SIGTERM reaches it.
set -eu
case "${INGEST_ON_START:-missing}" in
  always) ingest=1 ;;
  missing) if [ -f "${BRIEFING_DATA_DIR:-/data}/chroma/chroma.sqlite3" ]; then ingest=0; else ingest=1; fi ;;
  off) ingest=0 ;;
  *) echo "briefing-entrypoint: INGEST_ON_START must be missing, always or off" >&2; exit 64 ;;
esac
if [ "$ingest" = 1 ]; then
  echo "briefing-entrypoint: ingesting corpus into ${BRIEFING_DATA_DIR:-/data}/chroma"
  python -m app.rag ingest
fi
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
EOF

# Numeric UID so Kubernetes runAsNonRoot can verify it.
USER 10001:10001
VOLUME ["/data"]
EXPOSE 8000

# Secrets arrive as env vars at run time (DEEPSEEK_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, LANGSMITH_API_KEY).
# None are baked in. DeepSeek is demo-only: real PHI needs a BAA-covered model endpoint.

# First start on an empty volume embeds the corpus before uvicorn binds (about 60 s for 11 documents).
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8000'), timeout=4)"]

ENTRYPOINT ["/usr/local/bin/briefing-entrypoint"]
