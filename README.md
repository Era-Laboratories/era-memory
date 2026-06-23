# era-memory

A portable, tier-selected memory system for agents and apps — record storage, vector
storage, and hybrid (vector + lexical) search with RRF fusion, recency decay, and
deduplication. **The retrieval logic is identical at every scale; only the adapters change.**

It is **ports & adapters**: all logic depends on nine small interfaces (`RecordStore`,
`VectorStore`, `Embedder`, `Queue`, `Extractor`, `BlobStore`, `KMS`, `Auth`, `Telemetry`),
and one `MEMORY_TIER` knob selects an adapter set.

| Tier | Stack | Extras |
|------|-------|---------|
| **0 — laptop / offline** | SQLite + sqlite-vec, in-process | `era-memory[tier0]` |
| **1 — team / VM / Cloud SQL** | Postgres + pgvector, hosted embeddings, HTTP | `era-memory[tier1]` |
| **2 — enterprise** | Milvus + vLLM + Redis Sentinel + GCP KMS | `era-memory[milvus,vllm,redis,gcp]` |

The base package has **zero third-party dependencies** — no mandatory cloud account, GPU, or
private registry to run Tier 0/1. Every backend is an optional extra.

## Quick start

```bash
pip install "era-memory[tier0,localembed]"   # library + on-disk store + local embedder
era-memory setup                             # one-time: download a small embedding model from HF
```

```python
import asyncio
from era_memory import build_memory, MemoryRecord, SearchRequest

async def main():
    mem = build_memory(tier=0, db_path="memory.db")        # records + vectors in one SQLite file
    await mem.store(MemoryRecord(user_id="u1", content="Ada prefers dark roast coffee"))
    res = await mem.search(SearchRequest(user_id="u1", query="what does Ada drink?"))
    print(res.results[0].content)                          # -> Ada prefers dark roast coffee

asyncio.run(main())
```

Local, offline, no API key. For a hosted service, embedder options, and deploys, see
[`INSTALL.md`](INSTALL.md).

## Install

```bash
pip install "era-memory[tier0]"   # laptop/offline; use [tier1] for the Postgres + HTTP service
```

The Tier 2 observability stack uses **private** Era packages and is **not** part of the public
package — Era-internal deploys install it separately via `requirements-internal.txt`. Every
extra listed above is fully public.

## Status

**M0, M1, and M2 are done.** The single conformance suite passes against the in-memory,
SQLite, and Postgres/pgvector backends; Tier 1 has been validated end-to-end against a live
`docker compose` stack. **104 tests pass with Postgres (81 + 4 skipped without it).**
See [`docs/PROGRESS.md`](docs/PROGRESS.md) for the milestone detail and
[`docs/era-memory-light-spec.md`](docs/era-memory-light-spec.md) for the full spec.

**Published:** `era-memory` is on [PyPI](https://pypi.org/project/era-memory/) (`pip install
era-memory`). The offline ONNX embedder is shipped (`era-memory setup`).
**Not yet available:** Tier 2 (Milvus/vLLM/Redis) adapters.

> **Deploying inside Era Labs Tools?** Start with
> [`docs/HANDOVER-era-labs-tools.md`](docs/HANDOVER-era-labs-tools.md). Everyone else: see
> [`INSTALL.md`](INSTALL.md).

> **⚠️ Read this before judging retrieval quality.** The **default** embedder is a
> deterministic hashed bag-of-words stand-in. It exercises the storage/search plumbing so the
> library runs with zero setup, but it is **not semantic** — a query only matches on shared
> literal tokens, so "what does Ada drink?" will *not* find "Ada prefers dark roast coffee".
> **For real retrieval, set up a real embedder** (one-time):
>
> ```bash
> pip install "era-memory[localembed]"   # ONNX, CPU-only
> era-memory setup                        # downloads a small model from Hugging Face, caches it
> ```
>
> After that, a plain `build_memory(...)` auto-uses the cached model — no endpoint, GPU, or API
> key. Prefer a one-liner? `build_memory(embedder="auto")` downloads-if-needed and serves it.
> Prefer a hosted/local endpoint instead? Point the OpenAI-compatible embedder
> (`era_memory.adapters.openai.OpenAICompatibleEmbedder`, or `MEMORY_EMBEDDING_URL`) at any
> `/embeddings` server — OpenAI, Ollama, vLLM, LiteLLM. See
> [Choosing a store, embedder, and dimension](#choosing-a-store-embedder-and-dimension) and
> [`INSTALL.md`](INSTALL.md).

## Use it as a library (in-process, no infra)

```python
import asyncio
from era_memory import build_memory, MemoryRecord, SearchRequest, SessionPayload

async def main():
    # Tier 0, persisted to a single SQLite file (records + vectors + FTS together).
    #
    # With no `embedder=` this auto-uses a local model if you've run `era-memory setup`,
    # otherwise it falls back to the non-semantic dev embedder (see the warning above).
    # `embedder="auto"` downloads-and-serves a local model on the spot.
    mem = build_memory(tier=0, db_path="memory.db")

    await mem.store(MemoryRecord(user_id="u1", content="Ada prefers dark roast coffee"))

    # Or extract memories from a raw conversation (heuristic extractor by default):
    await mem.encode(SessionPayload(
        user_id="u1", session_id="s1",
        conversation="User: I just adopted a cat named Mochi.\nUser: She loves tuna.",
    ))

    res = await mem.search(SearchRequest(user_id="u1", query="what does Ada drink?"))
    print(res.results[0].content if res.results else "no results")

asyncio.run(main())
```

Drop `db_path` for a pure in-memory store (tests, ephemeral use).

## Run it as a service (Tier 1)

```bash
docker compose up --build          # Postgres+pgvector + the HTTP API on :8080
```

```bash
TOKEN="change-me"
curl -s -H "Authorization: Bearer $TOKEN" -H "X-User-Id: u1" -H "Content-Type: application/json" \
  -X POST localhost:8080/api/memories -d '{"content":"Ada prefers dark roast coffee"}'

curl -s -H "Authorization: Bearer $TOKEN" -H "X-User-Id: u1" -H "Content-Type: application/json" \
  -X POST localhost:8080/api/memories/search -d '{"query":"coffee"}'
```

HTTP routes: `GET /health`, `GET /ready`, `POST /api/memories`, `POST /api/memories/search`.
Configure via env — see [`docs/HANDOVER-era-labs-tools.md`](docs/HANDOVER-era-labs-tools.md).

## Architecture in one paragraph

Records and vectors live behind the `RecordStore` and `VectorStore` ports. At Tiers 0/1 they
share one backend (SQLite file / one Postgres DB), so the dual-write **collapses into a single
transaction** — there is no "record saved but vector failed" orphan state. At Tier 2 they are
separate systems (Postgres + Milvus) and the orchestrator preserves era-core's fail-fast
semantics. Hybrid search fuses a vector-ANN leg and a lexical leg via Reciprocal Rank Fusion
(k=60, 0.6/0.4), then scales by importance and a 30-day recency half-life.

## Choosing a store, embedder, and dimension

The **embedding dimension is not fixed** — the vector column is created at whatever dimension
your embedder emits (`vec0(dim)` on SQLite, `halfvec(dim)` on Postgres, where `halfvec` indexes
up to 4000 dims). Pick a store + embedder + dimension to match your deployment; nothing in the
retrieval logic changes.

| Deployment | Store | Embedder (license) | Dim |
|---|---|---|---|
| Laptop / offline / tests | SQLite + sqlite-vec | `all-MiniLM-L6-v2` (Apache-2.0) | 384 |
| Small team / single VM | SQLite **or** Postgres + pgvector | `bge-base-en-v1.5` (MIT) / `nomic-embed-text-v1.5` (Apache-2.0) | 768 |
| Production / Cloud SQL | Postgres + pgvector (`halfvec`) | `mxbai-embed-large-v1` (Apache-2.0) **or** OpenAI `text-embedding-3-*` (MRL→1024) | 1024 |
| Enterprise / era-core parity | Postgres + Milvus | Qwen3 family via vLLM, MRL→2048 | 2048 |

These are starting points, not the only valid choices — any model your endpoint serves works.
There is **one hard rule** (enforced by a `(model, dim)` guard and a startup check): a store is
**pinned to a single `(model, dim)` for its whole life**. The contract is *same vector space on
both write and query* — which means same model, same dimension, same MRL-truncation length, and
same normalization. Changing the model or dimension later means **re-embedding into a new
store**, not a config flip. Lightweight models cap at ~1024 native dims; reaching 2048+ requires
a larger model (e.g. Qwen3) MRL-truncated down — see
[`docs/adr/0001-dimension-is-a-per-deployment-contract.md`](docs/adr/0001-dimension-is-a-per-deployment-contract.md)
for the full reasoning.

Set the dimension with `MEMORY_EMBEDDING_DIMENSIONS` (or pass an `Embedder` whose `.dimensions`
is the source of truth — if you do both and they disagree, wiring fails fast).

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,sqlite,postgres,openai,server,encryption]"
pytest -q                                              # 81 + 4 skipped (no Postgres)
MEMORY_TEST_PG_DSN=postgresql://postgres:era@localhost:55433/era pytest -q   # 104 (with Postgres)
ruff check src tests
```

A new backend is added by implementing a port and making it pass the existing
`tests/conformance/` suite — that is the contract.

## License

Apache-2.0.
