# Handover — using `era-memory` for a memory app in Era Labs Tools

**Audience:** the engineer building a memory system app on the `era-labs-tools` GCP project.
**TL;DR:** Don't build record-store + vector-store + hybrid-search + dedup from scratch —
`era-memory` already is that, as a portable library/service. Run it as **Tier 1**
(Postgres + pgvector) on `era-labs-tools`. You integrate either **in-process as a Python
library** or by **deploying it as an HTTP service** and calling it. This doc gets you from
zero to a working deployment and tells you the three things that will bite you if you skip them.

---

## 1. What it is (and what you get for free)

A tier-selected memory system built on **ports & adapters**. The retrieval logic is identical
at every scale; only the backing adapters change. You get, already implemented and tested:

- **Store / search / delete / encode** over a clean async API.
- **Hybrid search**: vector ANN + lexical (BM25/`ts_rank`) fused by Reciprocal Rank Fusion
  (k=60, semantic 0.6 / lexical 0.4), then scaled by `importance_score` and a 30-day recency
  half-life.
- **Deduplication** at cosine ≥ 0.85.
- **An extraction pipeline** (`encode`) that turns a raw conversation into typed memories
  (entropy gate → extract → embed → dedup → write).
- **Single-transaction writes** at Tier 1 — records and vectors live in one Postgres DB, so
  there is no "record saved but vector failed" orphan state.
- **Per-user isolation**, bearer-token auth, and a small HTTP surface.

The same conformance suite passes against in-memory, SQLite, and Postgres, and Tier 1 has been
validated end-to-end against a live `docker compose` stack.

## 2. Pick your integration mode

**(A) In-process library** — your app is Python and you want the lowest latency / no extra
service:

```python
from era_memory.wiring import build_memory_async
from era_memory.adapters.openai import OpenAICompatibleEmbedder
from era_memory.models import MemoryRecord, SearchRequest

embedder = OpenAICompatibleEmbedder(
    base_url="https://<your-embedding-endpoint>/v1",
    model="<model>", dimensions=1024, api_key="<key-if-needed>",
)
mem = await build_memory_async(tier=1, dsn="postgresql://...", embedder=embedder)

await mem.store(MemoryRecord(user_id="u1", content="Ada prefers dark roast coffee"))
res = await mem.search(SearchRequest(user_id="u1", query="what does Ada drink?"))
```

**(B) Standalone HTTP service** — your app isn't Python, or you want memory as its own
deployable. Deploy the included `Dockerfile` and call the REST API (section 4). **Recommended
for a shared internal tool.**

## 3. Deploy to Era Labs Tools (Tier 1)

Target shape: **Cloud Run** (the API) + **Cloud SQL for PostgreSQL** (with the `pgvector`
extension) + **an embedding endpoint**.

1. **Cloud SQL**: create a Postgres instance; enable the `vector` extension
   (`CREATE EXTENSION vector;`). The app creates its own tables/indexes on first start.
   - **Requires pgvector ≥ 0.7** for the `halfvec` type. Confirm the instance's pgvector
     version; if it's older, either upgrade or use a ≤2000-dim embedding model (see §6).
2. **Embedding endpoint**: stand up (or point at) any **OpenAI-compatible `/embeddings`**
   service — a self-hosted vLLM/GPU embedder on `era-labs-tools`, OpenAI, or LiteLLM. This is
   what gives real semantic quality (see §6, caveat #1).
3. **Build & deploy the image** (`Dockerfile` in the repo root) to Cloud Run.
4. **Set env** (section 5). At minimum: `MEMORY_TIER=1`, `MEMORY_PG_DSN`,
   `MEMORY_BEARER_TOKEN`, `MEMORY_EMBEDDING_URL`, `MEMORY_EMBEDDING_DIMENSIONS`.
5. **Connectivity**: use Cloud SQL private IP or the Cloud SQL connector; put the resulting
   DSN in `MEMORY_PG_DSN` (asyncpg format, e.g. `postgresql://user:pass@HOST:5432/db`).

To try the whole stack locally first: `docker compose up --build` (Postgres + API on `:8080`).

## 4. HTTP API contract

All routes except `/health` and `/ready` require:
`Authorization: Bearer <MEMORY_BEARER_TOKEN>` and `X-User-Id: <user>`.

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | — | `{"status":"ok"}` |
| `GET /ready` | — | `{"status":"ready","vector_store_connected":true}` |
| `POST /api/memories` | `{"content","memory_type?","importance_score?","confidence?","experience_id?","metadata?"}` | `{"id","memory_type","source_type":"memory"}` |
| `POST /api/memories/search` | `{"query","strategy?","limit?","memory_type?","experience_id?"}` | `{"results":[{id,content,score,memory_type,source_type}],"strategy","total_candidates","latency_ms"}` |

`strategy` ∈ `hybrid` (default) `vector_only` `bm25_only` `deep`. Response shapes deliberately
mirror era-core's so a future Tier-2 deployment can be API-compatible.

## 5. Configuration (environment variables)

| Var | Tier | Meaning |
|---|---|---|
| `MEMORY_TIER` | all | `0` (SQLite) or `1` (Postgres). The HTTP app supports 0/1. |
| `MEMORY_PG_DSN` | 1 | asyncpg DSN for Cloud SQL. **Required at Tier 1.** |
| `MEMORY_DB_PATH` | 0 | SQLite file path (Tier 0 only). |
| `MEMORY_BEARER_TOKEN` | all | Static token clients must present. Set a real secret. |
| `MEMORY_EMBEDDING_URL` | 1 | OpenAI-compatible `/embeddings` base URL. Omit → dev hash embedder (NOT semantic). |
| `MEMORY_EMBEDDING_MODEL` | 1 | Embedding model name sent to the endpoint. |
| `MEMORY_EMBEDDING_DIMENSIONS` | all | Stored/indexed vector dim. **Must stay constant for a DB's life** (see §6). |
| `MEMORY_EMBEDDING_API_KEY` | 1 | Bearer key for the embedding endpoint, if required. |
| `MEMORY_RRF_K`, `MEMORY_RRF_SEMANTIC_WEIGHT`, `MEMORY_RRF_LEXICAL_WEIGHT`, `MEMORY_RECENCY_HALF_LIFE_DAYS`, `MEMORY_DEDUP_THRESHOLD` | all | Ranking/dedup tuning. Defaults match era-core. |

## 6. The three things that will bite you — read this

1. **The default embedder is a non-semantic stand-in.** With no `MEMORY_EMBEDDING_URL`, the
   service uses a deterministic hashed-bag-of-words embedder that only matches on shared
   literal tokens. It proves the plumbing; it is **not** real retrieval. **You must point
   `MEMORY_EMBEDDING_URL` at a real embedding endpoint** for a usable app.

2. **Embedding dimension is locked per database (single-tier-for-life).** The vector column
   and index are created at the chosen dimension on first start, and the store refuses to open
   with a different `(model, dim)`. Changing the embedding model later = **re-embed into a new
   database**, not a config flip. Pick the model/dim once.
   - **pgvector `halfvec` indexes up to 4000 dims** (the plain `vector` type caps at 2000).
     The locked production dimension is `halfvec(2048)`. If you want a different model, keep
     the indexed dimension ≤ 4000.

3. **Auth is a static bearer token, not full JWT/Oathkeeper.** Fine for an internal tool
   behind the org perimeter; do not treat it as end-user identity. `user_id` comes from the
   `X-User-Id` header — your calling app is responsible for setting it correctly, since all
   data is scoped (and isolated) by it.

## 7. Status & limitations (be honest with your own roadmap)

**Done & tested:** Tiers 0 and 1, hybrid search, dedup, encode pipeline, single-transaction
collapse, bearer auth, HTTP API, Docker/compose. 104 tests pass with Postgres.

**Not yet / opt-in:**
- **Local ONNX embedder** (true-offline laptop tier) — not built; use a hosted endpoint.
- **Encryption** — envelope encryption + local KMS exist and are tested, but encryption is
  off by default and not yet wired through the HTTP write path. Don't assume at-rest field
  encryption is on.
- **Tier 2** (Milvus/vLLM/Redis, GCP KMS, OTel) and the **era-core API-compatibility golden
  test** are designed but not implemented.
- **The extraction `Extractor` defaults to a heuristic** (line-per-memory). For higher-quality
  extraction, implement an LLM-backed `Extractor` adapter (the port is ready).

## 8. Extending it

Add a backend by implementing the relevant port (`src/era_memory/ports/`) and making it pass
`tests/conformance/` — that suite *is* the contract. The most likely things you'll want:
an LLM `Extractor`, or a different `Embedder`. Both are single-file adapters; nothing else
changes because the core logic only depends on the ports.

Full design rationale, milestones, and success criteria: `docs/era-memory-light-spec.md`.
Milestone status: `docs/PROGRESS.md`.

## 9. Where the code is

**Repository:** https://github.com/Era-Laboratories/era-memory (private — request access from
Alexander Ollman).

```bash
git clone https://github.com/Era-Laboratories/era-memory.git
```

Naming note: the **public-facing portable** memory system is `Era-Laboratories/era-memory`
(this repo). The **internal era-core service** was renamed to `Era-Laboratories/era-mneme`;
its runtime namespace/service in the cluster remain `era-memory*` and are unchanged.
