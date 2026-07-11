# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

# Optional extras, comma-separated: rerank, otel, postgres.
ARG EXTRAS=""

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".${EXTRAS:+[$EXTRAS]}"
# Fetch the tokenizer at build time so the container needs no network for it.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"


FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000 \
    RAG_INDEX_DIR=/app/var/index \
    RAG_INGEST_ROOT=/app/data \
    RAG_INGEST_ON_STARTUP=/app/data/docs

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken
COPY --chown=app:app data ./data
RUN mkdir -p /app/var && chown app:app /app/var

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["rag-service"]
