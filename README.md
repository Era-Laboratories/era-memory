# era-memory

A portable, tier-selected memory system for agents and apps — record storage, vector
storage, and hybrid (vector + lexical) search with RRF fusion, recency decay, and
deduplication. **The retrieval logic is identical at every scale; only the adapters change.**

It is **ports & adapters**: all logic depends on nine small interfaces (`RecordStore`,
`VectorStore`, `Embedder`, `Queue`, `Extractor`, `BlobStore`, `KMS`, `Auth`, `Telemetry`),
and one `MEMORY_TIER` knob selects an adapter set.

| Tier | Stack | Install |
|------|-------|---------|
| **0 — laptop / offline** | SQLite + sqlite-vec, local embeddings, in-process | `pip install "era-memory[tier0]"` |
| **1 — team / VM / Cloud SQL** | Postgres + pgvector, hosted embeddings, HTTP | `pip install "era-memory[tier1]"` |
| **2 — enterprise** | Milvus + vLLM + Redis Sentinel + GCP KMS | `pip install "era-memory[milvus,vllm,redis,gcp,era]"` |

The base package has **zero third-party dependencies** — no mandatory cloud account, GPU, or
private registry to run Tier 0/1. Every backend is an optional extra.

## Status

**M0, M1, and M2 are done.** The single conformance suite passes against the in-memory,
SQLite, and Postgres/pgvector backends; Tier 1 has been validated end-to-end against a live
`docker compose` stack. **104 tests pass with Postgres (81 + 4 skipped without it).**
See [`docs/PROGRESS.md`](docs/PROGRESS.md) for the milestone detail and
[`docs/era-memory-light-spec.md`](docs/era-memory-light-spec.md) for the full spec.

> **Deploying inside Era Labs Tools?** Start with
> [`docs/HANDOVER-era-labs-tools.md`](docs/HANDOVER-era-labs-tools.md).

> **Embedder note:** the default dev embedder is a deterministic hashed bag-of-words
> stand-in — it exercises the storage/search plumbing but is **not semantic**. For real
> retrieval quality, point the OpenAI-compatible embedder at a real `/embeddings` endpoint
> (Tier 1) — see the handover doc. A local ONNX embedder for true-offline use is on the roadmap.

## Use it as a library (in-process, no infra)

```python
import asyncio
from era_memory.wiring import build_memory
from era_memory.models import MemoryRecord, SearchRequest, SessionPayload

async def main():
    # Tier 0, persisted to a single SQLite file (records + vectors + FTS together).
    mem = build_memory(tier=0, db_path="memory.db")

    await mem.store(MemoryRecord(user_id="u1", content="Ada prefers dark roast coffee"))

    # Or extract memories from a raw conversation (heuristic extractor by default):
    await mem.encode(SessionPayload(
        user_id="u1", session_id="s1",
        conversation="User: I just adopted a cat named Mochi.\nUser: She loves tuna.",
    ))

    res = await mem.search(SearchRequest(user_id="u1", query="what does Ada drink?"))
    print(res.results[0].content)

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
