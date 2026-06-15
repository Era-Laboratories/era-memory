# era-memory HTTP service (Tier 1 default). Deployable to era-labs-tools (Cloud Run / GKE).
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Tier-1 stack: Postgres+pgvector, OpenAI-compatible embeddings, HTTP server.
RUN pip install --no-cache-dir ".[postgres,openai,server]"

ENV MEMORY_TIER=1
EXPOSE 8080

# Build the app from env at startup (MEMORY_PG_DSN, MEMORY_BEARER_TOKEN, MEMORY_EMBEDDING_URL...).
CMD ["uvicorn", "era_memory.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
