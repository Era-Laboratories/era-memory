# Installing and using era-memory

A practical, vendor-neutral guide for using era-memory in **your own** AI application — no Era
account, cloud project, or private registry required. For the architecture overview see
[`README.md`](README.md); for Era-internal deployment see
[`docs/HANDOVER-era-labs-tools.md`](docs/HANDOVER-era-labs-tools.md).

## 1. Requirements

- **Python 3.10+**
- That's it for the base library — it has **zero third-party dependencies**. Each backend
  (SQLite vectors, Postgres, an embedding endpoint, the HTTP service) is an optional extra you
  opt into.

## 2. Install

> **Not yet published to PyPI.** Install from source for now. (`pip install era-memory` will
> work once the first release is cut.)

```bash
git clone https://github.com/Era-Laboratories/era-memory.git
cd era-memory
python -m venv .venv && . .venv/bin/activate
pip install -e ".[tier0]"
```

### Extras

| Extra | Pulls in | Use it for |
|-------|----------|------------|
| `sqlite` | `sqlite-vec` | On-disk vector + lexical search in one SQLite file (Tier 0) |
| `openai` | `httpx` | The OpenAI-compatible embedder (any `/embeddings` endpoint) |
| `postgres` | `asyncpg`, `pgvector` | Postgres + pgvector store (Tier 1) |
| `server` | `fastapi`, `uvicorn`, `pydantic`, `orjson` | The HTTP service |
| `encryption` | `cryptography` | Envelope encryption of stored content |
| `redis` | `redis` | Redis-backed queue |
| `tier0` | `sqlite` + `localembed` | Convenience bundle for laptop/offline |
| `tier1` | `postgres` + `openai` + `redis` + `server` | Convenience bundle for the service |

Mix freely, e.g. `pip install -e ".[sqlite,openai]"` for an on-disk store with a real embedder.

> **The Tier 2 `[milvus]`, `[vllm]`, `[gcp]` extras are not yet shipped** (adapters land in a
> later release). The private Era observability stack is intentionally **not** in the public
> package. Everything in the table above is fully public and installable.

## 3. Verify the install

```bash
python -c "from era_memory import build_memory, MemoryRecord, SearchRequest; print('ok')"
```

## 4. Use it as a library (Tier 0, in-process)

```python
import asyncio
from era_memory import build_memory, MemoryRecord, SearchRequest

async def main():
    mem = build_memory(tier=0, db_path="memory.db")   # one SQLite file; drop db_path for in-RAM
    await mem.store(MemoryRecord(user_id="u1", content="Ada prefers dark roast coffee"))
    res = await mem.search(SearchRequest(user_id="u1", query="coffee"))
    print([r.content for r in res.results])

asyncio.run(main())
```

## 5. Set up a real embedder (required for semantic search)

**This is the step most likely to trip you up.** With no embedder supplied and nothing set up,
era-memory uses a **non-semantic** hashed-token stand-in so the library runs with zero setup — it
only matches on shared literal words. You have two ways to get real semantic retrieval; pick one.

### Option A — local ONNX model (offline, no endpoint, no API key)

```bash
pip install "era-memory[localembed]"   # fastembed + onnxruntime, CPU-only
era-memory setup                        # interactive: pick a model, download it from Hugging Face
era-memory status                       # shows what's configured / cached
```

`setup` offers a choice of model and caches it under `~/.cache/era-memory/models`
(override with `ERA_MEMORY_MODEL_DIR`):

| Model key | Model | Dim | Download | License |
|-----------|-------|-----|----------|---------|
| `bge-small` (default) | `BAAI/bge-small-en-v1.5` | 384 | ~90 MB | MIT |
| `mxbai-large` | `mixedbread-ai/mxbai-embed-large-v1` | 1024 | ~640 MB | Apache-2.0 |

Once a model is cached, **`build_memory(...)` uses it automatically** — no arguments needed:

```python
from era_memory import build_memory
mem = build_memory(tier=0, db_path="memory.db")   # auto-resolves the cached local model
```

Or skip the CLI and let the library download-and-serve on first use:

```python
mem = build_memory(tier=0, db_path="memory.db", embedder="auto")
```

> No surprises: a bare `build_memory()` only uses an **already-cached** model and never downloads
> on its own. The download happens only via `era-memory setup` or `embedder="auto"` (both explicit).
> Non-interactive (CI): `era-memory setup --yes --model bge-small`.

### Option B — an OpenAI-compatible endpoint (hosted or local)

Talks to *any* server exposing `POST /embeddings` — so still not cloud-locked:

```python
from era_memory import build_memory
from era_memory.adapters.openai import OpenAICompatibleEmbedder

embedder = OpenAICompatibleEmbedder(
    base_url="http://localhost:11434/v1",  # endpoint base; `/embeddings` is appended
    model="nomic-embed-text",
    dimensions=768,                        # MUST match the model's output (see §6)
    api_key=None,                          # set if your endpoint requires a bearer key
)
mem = build_memory(tier=0, db_path="memory.db", embedder=embedder)
```

| Provider | `base_url` | Notes |
|----------|-----------|-------|
| **Ollama** (local) | `http://localhost:11434/v1` | `ollama pull nomic-embed-text`; no API key |
| **vLLM** (self-hosted) | `http://<host>:8000/v1` | Serve any HF embedding model |
| **LiteLLM** proxy | `http://<host>:4000/v1` | Fan out to many providers behind one URL |
| **OpenAI** | `https://api.openai.com/v1` | `model="text-embedding-3-small"`, `dimensions=1536`, set `api_key` |

Set `MEMORY_EMBEDDING_URL` (+ `MEMORY_EMBEDDING_MODEL`/`_DIMENSIONS`/`_API_KEY`) to use this path
from the HTTP service or without constructing the embedder yourself.

## 6. Choosing a dimension

The vector column is created at your embedder's dimension and a store is **pinned to one
`(model, dim)` for its lifetime** — changing it later means re-embedding into a fresh store. Pick
once, up front. See the
[pairings table](README.md#choosing-a-store-embedder-and-dimension) and
[`docs/adr/0001-dimension-is-a-per-deployment-contract.md`](docs/adr/0001-dimension-is-a-per-deployment-contract.md).
If you set `MEMORY_EMBEDDING_DIMENSIONS` **and** pass an embedder with a different
`.dimensions`, wiring fails fast with a clear error rather than corrupting the store.

## 7. Run it as a service (Tier 1, Postgres + HTTP)

```bash
pip install -e ".[tier1]"
docker compose up --build      # Postgres+pgvector + the HTTP API on :8080
```

Configure via environment variables:

| Variable | Required | Meaning |
|----------|----------|---------|
| `MEMORY_TIER` | yes (`1`) | Selects the Postgres+HTTP adapter set |
| `MEMORY_PG_DSN` | yes | `postgresql://user:pass@host:5432/db` (needs pgvector ≥ 0.7 for `halfvec`) |
| `MEMORY_BEARER_TOKEN` | yes | Static bearer token clients must send as `Authorization: Bearer …` |
| `MEMORY_EMBEDDING_URL` | for real search | OpenAI-compatible `/embeddings` base URL; omit → non-semantic dev embedder |
| `MEMORY_EMBEDDING_MODEL` | with URL | Model name sent to the endpoint |
| `MEMORY_EMBEDDING_DIMENSIONS` | with URL | Output dim; must match the model and stay constant for the DB's life |
| `MEMORY_EMBEDDING_API_KEY` | if endpoint needs it | Bearer key forwarded to the embedding endpoint |
| `MEMORY_DB_PATH` | Tier 0 only | SQLite file path when running the app at Tier 0 |

Routes: `GET /health`, `GET /ready`, `POST /api/memories`, `POST /api/memories/search`.

```bash
TOKEN="change-me"
curl -s -H "Authorization: Bearer $TOKEN" -H "X-User-Id: u1" -H "Content-Type: application/json" \
  -X POST localhost:8080/api/memories -d '{"content":"Ada prefers dark roast coffee"}'
curl -s -H "Authorization: Bearer $TOKEN" -H "X-User-Id: u1" -H "Content-Type: application/json" \
  -X POST localhost:8080/api/memories/search -d '{"query":"coffee"}'
```

> `user_id` comes from the `X-User-Id` header and scopes/isolates all data — your calling app
> sets it. The bearer token is a single shared secret, not per-user identity; put the service
> behind your own auth/perimeter for multi-tenant production use.

## 8. Troubleshooting

- **Search returns junk / nothing relevant** → you're on the default non-semantic embedder.
  Set up a real one (§5).
- **`pip install` fails on `era-telemetry` / a git URL** → you're installing the Era-internal
  `requirements-internal.txt` without access to the private repos. External users don't need it;
  use the public extras (§2).
- **`halfvec` type errors on Postgres** → your pgvector is < 0.7. Upgrade, or use the
  `pgvector/pgvector:pg17` image as in `docker-compose.yml`.
- **`ConfigurationError: Embedder dimension … != MEMORY_EMBEDDING_DIMENSIONS …`** → your env
  and your embedder disagree on dimension; set one source of truth (§6).
- **Run the tests** to confirm a healthy install:
  ```bash
  pip install -e ".[dev,sqlite,postgres,openai,server,encryption]"
  pytest -q            # 81 pass + 4 skipped without a local Postgres
  ```
