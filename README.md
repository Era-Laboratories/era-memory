# era-memory

A portable, tier-selected memory system for agents and apps — record storage, vector
storage, and hybrid (vector + lexical) search with RRF fusion, recency decay, and
deduplication. The retrieval logic is identical at every scale; only the adapters change.

It is **ports & adapters**: all logic depends on small interfaces (`RecordStore`,
`VectorStore`, `Embedder`, `Queue`, `Extractor`, `BlobStore`, `KMS`, `Auth`, `Telemetry`),
and one `MEMORY_TIER` knob selects an adapter set.

| Tier | Stack | Install |
|------|-------|---------|
| **0 — laptop / offline** | SQLite + sqlite-vec, CPU ONNX embeddings, in-process | `pip install era-memory[tier0]` |
| **1 — team / VM / Cloud SQL** | Postgres + pgvector, hosted embeddings | `pip install era-memory[tier1]` |
| **2 — enterprise** | Milvus + vLLM + Redis Sentinel + GCP KMS | `pip install era-memory[milvus,vllm,redis,gcp,era]` |

The base package has **zero third-party dependencies** — no mandatory cloud account, GPU,
or private registry to run Tier 0/1.

See [`docs/era-memory-light-spec.md`](docs/era-memory-light-spec.md) for the full development
spec, milestones, and success criteria, and [`docs/PROGRESS.md`](docs/PROGRESS.md) for status.

## Status

**M0** (ports + in-memory adapters + dual-write orchestrator + conformance suite) and
**M1** (Tier-0 SQLite + sqlite-vec single-file collapse) are **done** — 72 tests pass, the
conformance suite is green against both the in-memory and SQLite backends. Next: the ONNX
embedder, then Tier 1 (Postgres + pgvector). See `docs/PROGRESS.md`.

## Quick taste (in-process, no infra)

```python
import asyncio
from era_memory.wiring import build_memory
from era_memory.models import MemoryRecord, SearchRequest

async def main():
    mem = build_memory(tier=0)  # in-memory adapters until Tier-0 SQLite lands
    await mem.store(MemoryRecord(user_id="u1", content="Ada prefers dark roast coffee."))
    res = await mem.search(SearchRequest(user_id="u1", query="what coffee does Ada like?"))
    print(res.results[0].content)

asyncio.run(main())
```
