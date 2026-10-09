FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BRIEFING_DATA_DIR=/data
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data/patients.json ./data/patients.json
RUN useradd --create-home appuser && mkdir -p /data && chown appuser /data
USER appuser
# Secrets come from the orchestrator's secret store: DEEPSEEK_API_KEY (demo only; real PHI needs a
# BAA-covered endpoint), LANGSMITH_API_KEY. Never bake them into the image.
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
