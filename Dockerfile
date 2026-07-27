FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEMO_DATABASE_PATH=/data/chroma \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

WORKDIR /app

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src src
RUN pip install --no-cache-dir ".[web]"

COPY docs docs

RUN useradd --create-home app \
    && mkdir -p /data/chroma /home/app/.cache/huggingface \
    && chown -R app:app /data /home/app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["sh", "-c", "python -m mcp_rag_assistant.web.bootstrap && exec uvicorn mcp_rag_assistant.web.app:app --host 0.0.0.0 --port 8000"]
