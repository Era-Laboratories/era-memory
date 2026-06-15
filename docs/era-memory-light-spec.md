# era-memory-light — Standalone Portable Memory Library

**Status:** Draft spec for a new standalone repository
**Source of truth:** `era-core/era-memory` (`era-memory-service` + `era-memory-encoder`)
**Audience:** the engineer building the new repo
**Validation:** Each tier in this spec was independently feasibility-checked against the real production code and current library capabilities (see §12 for the per-tier findings that shaped it).

---

## 1. Goal

Turn `era-memory` — today a two-service, cluster-bound system welded to Cloud SQL + Milvus + a GPU vLLM service + Redis Sentinel + GCP KMS + Oathkeeper + TensorZero + private `era-*` git packages — into a single library that **anyone can `pip install` and run inside their own agent framework or local app**, with the *same* retrieval logic at every scale.

The retrieval logic is sound and portable. The weight is entirely in the infrastructure bindings. The design move is **ports & adapters, tier-selected**: lift the pure logic into a backend-agnostic core, put every infrastructure touchpoint behind a small interface, ship multiple adapters, and select an adapter set with one `MEMORY_TIER` knob.

`era-memory` already proves the pattern works — it has two real seams today (`embedding_provider`, and a clean `KmsProvider` ABC with `gcp`/`local`). This generalizes that to **nine ports**.

### What "done" means
A user runs `pipx install era-memory-light` on a laptop, with no cloud account, no GPU, and no private package registry, and gets working store + hybrid search + dedup offline. The *same* library, configured for Tier 2, is byte-for-byte API-compatible with today's `era-memory`.

---

## 2. Non-goals (hard lines)

- **No mandatory cloud account, GPU, or private registry to run Tier 0/1.** This is the defining constraint. It specifically requires removing the in-house `era-telemetry` / `era-version-schema` git dependencies from the base install (see §7).
- **No cross-tier embedding interoperability.** A 384-d Tier-0 index cannot be queried by a 2048-d Tier-2 vector; truncation does not bridge model families. The design is **single-tier-for-life with a documented re-embed migration** (see §6).
- **No new object store.** Production has *no* BlobStore. Do not invent a GCS/S3 dependency "for parity" (see §5, BlobStore).
- **No behavior change at Tier 2.** Tier 2 reproduces today's `era-memory` exactly — including its warts (sync pymilvus inside `async def`, fail-fast-keep-Postgres dual write, degraded-mode Milvus). Faithful re-expression, not improvement.

---

## 3. Architecture: the nine ports

All ports are `Protocol`/`ABC` interfaces in `core/ports/`. The pure logic in `core/` (RRF fusion, recency decay, cosine dedup, Matryoshka truncate, the extraction pipeline orchestration, envelope encryption) imports *only* ports — never a concrete backend.

| Port | Responsibility | Already a seam today? |
|---|---|---|
| **RecordStore** | Durable record rows; lexical (BM25/tsvector) search; fetch-by-ids; access-count updates; unit-of-work | No — raw asyncpg SQL inline in route handlers |
| **VectorStore** | Vector insert/search/delete; `is_connected()` | No — direct pymilvus |
| **Embedder** | Text → embedding; Matryoshka truncate + L2-normalize | **Partial** — provider protocol exists but factory hard-rejects non-`vllm` |
| **Queue** | Hand a session to the encoder pipeline (durable or in-process) | No — direct `redis.asyncio` |
| **Extractor** | Conversation → typed memories (LLM or heuristic) | No — direct TensorZero HTTP |
| **BlobStore** | Large-object persistence | **N/A — phantom port** (no object store in prod; DEK lives in Postgres) |
| **KMS** | Wrap/unwrap DEKs for envelope encryption | **Yes — best existing seam** (`KmsProvider` ABC, `gcp`/`local` factory, lazy import) |
| **Auth** | Request → `user_id` (JWT or bearer or none) | No — hardwired Oathkeeper JWKS middleware |
| **Telemetry** | Logging, tracing, `/health` `/ready` `/version` contract | No — hardwired private `era-telemetry`/`era-version-schema` |

> **The KMS port is the reference pattern.** It is already an `ABC` with a factory and a lazy `google-cloud-kms` import behind the `"gcp"` branch, so `[gcp]` is already install-optional. Every other port should imitate its shape.

### Pure logic to lift verbatim into `core/` (backend-agnostic, no rewrite)
- `_rrf_fuse` — RRF, `k=60`, semantic `0.6` / lexical `0.4` (`era-memory-service/src/services/search.py`)
- `_recency_decay` — `exp(-0.693 * age_days / 30)` (30-day half-life)
- Final scoring — `base_rrf * importance * (1 - recency_weight + recency_weight*recency)`
- Deep-search weak-trigger — `len(rrf) < limit OR top_cosine < 0.5`; two **separate** RRF dicts, never cross-fused; `RAW_TRANSCRIPT_IMPORTANCE_PROXY = 0.5`
- `_truncate_and_normalize` — Matryoshka (`era-memory-service/src/services/embedding.py`)
- `_entropy_score`, `_cosine_similarity`, dedup @ `0.85` (`era-memory-encoder/src/services/encoder.py`)
- Envelope encryption format `version||nonce||ct||tag`, AAD `{userId}:{fieldName}:{dekVersion}` (`era-memory-service/src/encryption/`)

---

## 4. The dual-write orchestrator (the crux)

This is the single most important design artifact. Get it wrong and Tier 2 silently diverges from prod; get it right and Tiers 0/1 simplify for free.

### Today's exact semantics (must be preserved at Tier 2)
1. Postgres insert commits **first**, inside `async with db_pool.get_transaction()`.
2. Embedding is best-effort *before* the write (failure → `embedding=None`, row still written, vector skipped).
3. Vector insert runs **after** the Postgres commit, **only if** an embedding exists.
4. Vector failure → **HTTP 503** with a literal detail string; **the Postgres row is NOT rolled back** (intentional fail-fast-keep-Postgres asymmetry). There are **three distinct 503 strings** (single memory / batch / session_content) and they are part of the API contract.
5. `delete` is the inverse: Postgres soft-delete is authoritative; vector delete failure is logged WARNING and **swallowed**.
6. `session_content` uses `ON CONFLICT (user_id, session_id) DO NOTHING`; on conflict the vector insert is **skipped entirely**.

### The contract — an orchestrator above two ports, NOT a shared/2-phase transaction

```python
# core/orchestration/dual_write.py  (tier-agnostic; the ONLY place that branches by tier)

async def dual_write(record_store, vector_store, row, vector_record):
    async with record_store.unit_of_work() as uow:
        stored, was_inserted = await record_store.insert_memory(uow, row)  # commits on ctx exit
    # Postgres has committed here — matches prod ordering exactly.
    if was_inserted and vector_record.embedding:
        try:
            await vector_store.insert([vector_record])
        except VectorStoreWriteError:
            raise DualWriteVectorError(stored)   # orchestrator maps to the exact 503 + detail string
    return stored
```

Fidelity rules:
- The vector write happens **outside** the unit-of-work, **after** commit. Never wrap it inside the uow.
- **Single-store tiers (0/1) collapse trivially:** when RecordStore and VectorStore are the same backend, the "vector" is just a column written *inside* the same uow, and `VectorStore.insert` is a no-op. Tiers 0/1 therefore get one atomic transaction and **can never hit the 503 path** — the fail-fast-keep-Postgres orphan-row failure mode disappears. This is the biggest simplification the port design buys.
- `RecordStore.insert_memory` **must return `(row, was_inserted: bool)`** so the orchestrator can skip the vector insert on the `ON CONFLICT` dedup path. Without this signal, idempotent retries fire spurious vector writes — a behavior change.
- The orchestrator owns the exception→HTTP mapping so the **three 503 detail strings stay byte-identical**.
- `soft_delete` is a separate orchestrator method (Postgres authoritative + vector best-effort/swallow). Do not unify with the write path.

---

## 5. The three tiers (validated adapter sets)

One knob: `MEMORY_TIER ∈ {0,1,2}` (or per-port overrides). Each tier was independently validated; the verdicts and the changes they forced are folded in below.

### Tier 0 — Laptop / fully local / offline  · `pipx install` · **Verdict: feasible with changes**

| Port | Adapter |
|---|---|
| RecordStore | SQLite sidecar table (`memories`, `session_content`) in the same db file |
| VectorStore | `sqlite-vec` `vec0` virtual table **in the same file** → single-transaction dual-write |
| Embedder | `fastembed` / ONNX Runtime, CPU, **384-d default** (`bge-small`), 768-d Matryoshka (`nomic-embed-text-v1.5`) opt-in |
| Queue | In-process direct call (optionally `asyncio` task / threadpool); no Redis |
| Extractor | **Heuristic / no-LLM default**; opt-in Ollama; opt-in hosted API |
| BlobStore | Local filesystem under `~/.era-memory/` (or none) |
| KMS | `LocalKmsProvider` with a **persisted** master key; encryption opt-in |
| Auth | No-op (single local user) |
| Telemetry | No-op (strip Prometheus calls) |

**Changes forced by validation:**
1. Keep a **SQLite sidecar table for the full record**; `vec0` holds only the vector + filter axes (`user_id` as a **partition key**; `memory_type`/`experience_id`/`created_at`/`importance_score` as **metadata columns**). Reason: `vec0` *auxiliary* columns are stored but **not filterable**, and large blobs bloat the vector index. Still one file, still one transaction.
2. **Persist the LocalKms master key** (e.g. `~/.era-memory/`, mode `0600`, or OS keychain). The spec's original "ephemeral key" loses all encrypted data on restart. Or default encryption *off* at Tier 0.
3. **Default Extractor = heuristic/no-LLM** (turn/entropy-window → candidate memory; rules for entities/topics; entropy → importance). Anything requiring Ollama or an API key breaks the "offline, zero accounts" contract. Ollama is the blessed opt-in.
4. **Freeze a small CPU dim (384 default)**, write `(model, dim)` into a db `meta` row, and **refuse to open** a db whose recorded `(model, dim)` ≠ the configured embedder (mirrors the existing dimension guard).

**Lexical leg:** SQLite **FTS5** (`bm25()` is native — cleaner than prod's pg_search+ILIKE; no ILIKE fallback needed). FTS5 ships in stdlib `sqlite3` on CPython ≥3.10; add a one-line startup capability assert.

**Risks (ranked):** cross-tier dim lock-in (by design, accept it) · `sqlite-vec` is brute-force, pre-1.0 alpha → pin version, per-user partition keeps scans small · no durability/retry without Redis → idempotent encode + single-tx write · **accidental `torch` bloat** → use `fastembed`/ONNX only, clean-venv test asserts torch absent · ephemeral-key data loss (change #2).

### Tier 1 — Small team / single VM · `docker compose up` · **Verdict: feasible with changes**

| Port | Adapter |
|---|---|
| RecordStore | Postgres (asyncpg, existing raw-SQL) |
| VectorStore | **Same Postgres via pgvector** (`halfvec`/`vector` column, HNSW cosine) → collapses the dual-write |
| Embedder | `OpenAICompatibleProvider` → any OpenAI-compatible `/v1/embeddings` (OpenAI, vLLM, LiteLLM, Ollama shim) |
| Queue | single-node Redis Streams (**recommended default**) or in-process asyncio |
| Extractor | OpenAI-compatible chat-completions + JSON schema (default); TensorZero opt-in |
| BlobStore | Local filesystem (session content lives in Postgres) |
| KMS | `local` provider (existing default) |
| Auth | static bearer token + `X-User-Id` |
| Telemetry | OTel/Pyroscope off by default (empty endpoint = disabled) |

**The one hard blocker — embedding dimension vs pgvector index cap:**
- Production embedding is **2048-d** (Qwen3-VL truncated from 4096). pgvector HNSW/IVFFlat index the `vector` type to a **max of 2000 dims** — the prod Postgres column is *already* `vector(2048)` and *deliberately unindexed* for this reason.
- **Resolve before anything else (pick one):** (a) column type `halfvec(2048)` + `halfvec_cosine_ops` HNSW (indexable to 4000 dims, half storage, minor fp16 recall loss) — keeps the prod model; or (b) a ≤2000-d Tier-1 embedding model (e.g. a 1536-d hosted model) — cleanest since Tier 1 is OpenAI-compatible-endpoint-based anyway. **Update the schema migration to actually create the HNSW index** (today it intentionally omits it).

**Other changes forced by validation:**
- **Relax `create_embedding_provider()`** to accept any OpenAI-compatible endpoint, not just `provider=="vllm"`.
- **Lexical default = native `tsvector`/`ts_rank`** (no extension → runs on the stock `pgvector/pgvector` image). `ts_rank` ≠ BM25 (no IDF), but **RRF consumes rank order, not raw scores**, so the gap is largely laundered at 0.4 weight. Offer **ParadeDB (`paradedb/paradedb` image)** as an opt-in prod-identical-BM25 lexical adapter.
- **Delete the Milvus insert block**; fold the embedding into the single-transaction INSERT.
- Add an **in-process Queue adapter** (wraps `process_session`); keep Redis as the recommended default.

**Risks (ranked):** the 2048>2000 cap (blocker, above) · dim lock-in (single model for the deployment's life) · `ts_rank`≠BM25 relevance gap (RRF mitigates; ParadeDB for parity) · HNSW insert write-amplification (fine at team rates; bulk-load then index) · in-process queue durability loss (prefer Redis) · `halfvec` fp16 recall loss (tune `ef_search`) · **pre-existing Py2 syntax bug** `except json.JSONDecodeError, ValueError:` in `era-memory-encoder/src/services/api_clients.py` — clean up during extraction.

### Tier 2 — Enterprise / K8s · today's stack exactly · **Verdict: feasible with changes**

| Port | Adapter | Clean? |
|---|---|---|
| RecordStore | Postgres / asyncpg | leak → owns lexical BM25 + unit-of-work (see §4) |
| VectorStore | Milvus / pymilvus (2 collections, HNSW M16/efC256, COSINE, ef128, dim 2048) | leak → degraded-mode + sync-in-async wart kept verbatim; expose `is_connected()` |
| Embedder | vLLM OpenAI-compatible | clean → wrapping removes the `!= "vllm"` hard-check |
| Queue | Redis-Sentinel | clean |
| Extractor | TensorZero (`POST /inference function_name=extract_memories`) | clean |
| BlobStore | **none — no-op / not-applicable** | phantom; do not add a cloud blob dep |
| KMS | GCP-KMS | clean (reference seam) |
| Auth | Oathkeeper-JWT **and** `X-Service-Token`+`X-User-Id` (both load-bearing) | clean |
| Telemetry | OTel / `era-telemetry` behind `[era]` extra | clean once optional |

**Changes forced by validation:**
1. Land the **dual-write orchestrator** (§4) as a first-class tier-agnostic component; `insert_memory` returns `(row, was_inserted)`; pin the 3 literal 503 strings.
2. **Drop `era-memory-test-suite` as the API-compat proof** — it is an **LLM-judge behavioral quality harness** that talks to `era-ingress`, *not* a deterministic contract test. Build a new golden/contract test instead (see §8). Keep the LLM-judge suite only as a *secondary, non-gating* quality regression.
3. **BlobStore is a no-op** at Tier 2 (DEK lives in Postgres `user_encryption_keys`).
4. **Telemetry + the `/version` `/health` `/ready` contract become optional via `[era]`** with no-op/in-tree defaults (see §7). The `/ready` gating policy is load-bearing and must match: service **gates** `postgresql`, **reports** `milvus`; encoder **gates** `redis`, conditionally gates `consumer`/`detector`.
5. Keep lexical BM25 (pg_search + **ILIKE fallback**) on the RecordStore port — the fallback is *behavior*, not infra.
6. Carry forward the **`encrypted_only` search auto-downgrade** (rewrites `deep`/`hybrid`/`bm25_only` → `vector_only` when plaintext `content` is NULL) — the search core must be able to read the KMS/encryption write-mode (a deliberate cross-port read).
7. **Collapse the two duplicate vLLM embedding clients** (service + encoder) into one Embedder adapter.

**Risks (ranked):** dual-write ordering + dedup signal (highest fidelity risk) · API-compat "proof" is a mirage until the new test exists · sync pymilvus inside `async def` (keep the wart — "fixing" it changes latency/parity) · Milvus degraded-mode `/ready` reports-not-gates (don't "improve" to gating) · private deps imported at module load (port must be *truly* import-optional) · encryption AAD field-name coupling (`content` vs `session_content`) · pre-existing Py2 `except X, Y:` syntax in several encoder files — **hard SyntaxErrors on Py3**, must resolve so the library imports cleanly.

---

## 6. Cross-tier embedding dimensions (the accepted constraint)

Different model families occupy different vector spaces, and Matryoshka only truncates *down within one family*. A 384-d Tier-0 vector is not comparable to a 2048-d Tier-2 vector. The code already hard-guards this (`embedding_dimension_mismatch_discarded`).

**Decision: single-tier-for-life.** Store `(model_id, dim)` as a property of the deployment (db `meta` row at Tier 0; schema/config at Tiers 1/2) and **refuse to open/serve** a store whose recorded `(model, dim)` ≠ the configured embedder. **Tier migration = full re-embed into a fresh store + index rebuild** — never an in-place dim change. Document this re-embed runbook as the only supported migration path.

---

## 7. Removing the private-dep install blocker

Both `pyproject.toml`s import `era-telemetry` and `era-version-schema` from `git+https://github.com/Era-Laboratories/...` **at module load** — so today you cannot even import the app without private GitHub auth. This is *the* blocker to "anyone can install."

**Fix:**
- **Telemetry port** with a no-op default: `create_logger` → stdlib structlog, `setup_observability` → no-op, propagation → identity. Tiers 0/1 use it; base install pulls zero private deps from PyPI.
- **`pip install era-memory-light[era]`** pulls the two git deps and wires real OTel — prod observability stays identical (same `setup_observability(...)`, same `create_logger(...)`).
- **Service-contract endpoints** (`/version` `/health` `/ready`): a small in-tree renderer reproduces the exact JSON shape + gating policy at Tiers 0/1; `[era]` swaps in the real `era-version-schema` routers. Keep the OTel transitive version ceilings *inside* the `[era]` extra only.

**Packaging discipline:** base package ≈ zero deps; each backend is an optional extra. `pip install era-memory-light[sqlite,localembed]` stays small (no `torch`, no `pymilvus`, no `asyncpg`, no `redis`). `[postgres,pgvector,openai]` = Tier 1; `[milvus,vllm,redis,gcp,era]` = Tier 2. CI asserts `torch` is absent from the Tier-0 install.

---

## 8. Conformance & API-compatibility testing

### 8.1 One conformance suite, every adapter passes it
A single backend-agnostic test suite runs against **every** adapter for a port (in-memory, SQLite, Postgres, Milvus, …). "Works on a laptop" and "works in prod" must mean the same thing. Covers: insert/fetch round-trip, vector KNN ordering by cosine, metadata filtering, user-scoping/IDOR isolation, dedup @0.85, the unit-of-work rollback property, lexical ranking.

### 8.2 Tier-2 API-compatibility = a NEW deterministic golden test (net-new scope)
The existing `era-memory-test-suite` **cannot** serve this (LLM-judge, non-deterministic, hits era-ingress). Build:
1. **OpenAPI-diff gate** — Tier-2 `openapi.json` identical to current era-memory modulo `version`.
2. **Golden-response replay** — recorded requests (`POST /api/memories`, `/batch`, `/api/session-content`, `POST /api/memories/search` for all 4 strategies incl. `deep`, `GET/PATCH/DELETE`) replayed against current era-memory and Tier-2-light over **shared Postgres+Milvus fixtures + frozen embeddings**; assert byte-identical JSON except `{id, timestamps, latency_ms}`.
3. **Failure-semantics tests** — kill Milvus mid-write → assert exact 503 + each of the 3 detail strings + Postgres row survives; delete swallows Milvus failure; session `ON CONFLICT` skips the vector insert.

**Shapes that will drift — pin them:** `MemoryResponse.source_type` default `"memory"` / `"raw_conversation"`; session-row-as-memory defaults (`importance_score=0.0`, `confidence=1.0`, `memory_type=EPISODE`); `SearchResponse.latency_ms` 1-dp rounding; `total_candidates` (sum of two RRF dicts in deep); 403 self-attribution messages; `min_length=1` 422 on empty session content.

---

## 9. Repo structure

```
era-memory-light/
  core/
    logic/            # rrf, recency, dedup, matryoshka, entropy, envelope — pure, no infra imports
    pipeline/         # extraction pipeline orchestration (entropy → extract → embed → dedup → write)
    orchestration/    # dual_write.py, soft_delete.py  (tier-agnostic; only place that branches by tier)
    ports/            # RecordStore, VectorStore, Embedder, Queue, Extractor, BlobStore, KMS, Auth, Telemetry
    models/           # MemoryCreate/Response, SearchRequest/Response, ExtractionResult, ...
  adapters/
    record_store/     # memory.py, sqlite.py, postgres.py
    vector_store/     # memory.py, sqlite_vec.py, pgvector.py, milvus.py
    embedder/         # memory.py, onnx_fastembed.py, openai_compat.py, vllm.py
    queue/            # inprocess.py, redis.py, redis_sentinel.py
    extractor/        # heuristic.py, openai_chat.py, ollama.py, tensorzero.py
    kms/              # local.py, gcp.py
    auth/             # noop.py, bearer.py, oathkeeper_jwt.py
    telemetry/        # noop.py, era.py   (era.py only under [era] extra)
  app/                # FastAPI app — same routes, now calling ports via the orchestrator
  wiring.py           # composition root: MEMORY_TIER + env → concrete adapter set
  tests/
    conformance/      # one suite per port, run against every adapter
    golden/           # OpenAPI-diff + replay + failure-semantics (Tier-2 API-compat gate)
  pyproject.toml      # extras: sqlite, localembed, postgres, pgvector, milvus, vllm, openai, redis, gcp, era
  docker-compose.yml  # Tier-1 stack
```

---

## 10. Milestones, validation & success criteria

Each milestone has an **exit gate** that must be green before the next begins. Gates are automated tests, not reviews.

### M0 — Interfaces & pure core (no infra)
**Build:** the 9 ports; lift the pure logic into `core/logic/`; the dual-write orchestrator (§4); **in-memory adapters for every port**; the conformance suite skeleton.
**Validate:** unit tests for `_rrf_fuse` (k=60, 0.6/0.4), `_recency_decay` (0d→1.0, 30d→~0.5), `_cosine_similarity`, `_entropy_score`, Matryoshka truncate+normalize, envelope round-trip + AAD-mismatch raises; orchestrator commit-ordering with mocked ports (vector write happens after commit; `was_inserted=False` skips vector; vector failure raises the mapped error without rolling back the record).
**Exit gate / success criteria:**
- Conformance suite passes against the in-memory adapter set for all 9 ports.
- `core/` has **zero imports** of any concrete backend (enforced by an import-linter rule).
- Ranking parity: `_rrf_fuse`/scoring reproduce current era-memory output on a fixed fixture to ≥1e-9.

### M1 — Tier 0 vertical slice (laptop, offline)
**Build:** SQLite RecordStore + `sqlite-vec` VectorStore (single-file, single-transaction), `fastembed`/ONNX Embedder (384-d), FTS5 lexical, in-process Queue, heuristic Extractor, persisted LocalKms, no-op Auth/Telemetry, `pipx`-installable entrypoint.
**Validate:** conformance suite passes on the SQLite/sqlite-vec adapters; integration tests (vec0 KNN cosine ordering; metadata `WHERE` filters inside KNN; `user_id` partition isolation = no cross-user leak; FTS5 `bm25()` ranking; hybrid fuse; single-transaction rollback leaves no orphan; deep-search fires below 0.5).
**End-to-end acceptance (offline, the gate):**
1. Clean-venv `pip install era-memory-light[sqlite,localembed]`; assert `torch` absent.
2. Network disabled → store **200** memories across 2 users (heuristic extractor, real ONNX embedder).
3. 10 queries hit top-k relevance on a labeled set; user B never sees user A's data.
4. Near-duplicate (cosine ≥0.85) is dropped (count unchanged).
5. **Zero outbound network connections** during the whole run.
6. Restart → re-open the same db with the persisted key → all 200 still searchable and decryptable.
**Success criteria (measurable):**
- Cold install → first search **< 60s** (incl. one-time model download); subsequent first-search **< 5s**.
- `[sqlite,localembed]` site-packages **< 300 MB**, no `torch`, no CUDA.
- Embed latency (CPU, 384-d): p50 **< 15 ms**, p95 **< 50 ms**.
- Hybrid search @10k records/user: p50 **< 50 ms**, p95 **< 150 ms**. @100k: p95 **< 300 ms** (brute-force ceiling; flag beyond).
- Near-dup rejection = 100% on the fixture; false-dedup of distinct memories = 0.
- Crash injected mid-encode → **0 orphaned** records/vectors.

### M2 — Tier 1 vertical slice (team, `docker compose up`)
**Build:** Postgres RecordStore + pgvector VectorStore (single-transaction; **`halfvec(2048)` or ≤2000-d model decided** — §5/§11); OpenAI-compatible Embedder (factory relaxed); `tsvector`/`ts_rank` lexical (ParadeDB opt-in); Redis + in-process Queue adapters; OpenAI-chat Extractor; bearer Auth; `docker-compose.yml`; schema migration that **creates the HNSW index**.
**Validate:** conformance suite on the Postgres/pgvector adapters via testcontainers; `CREATE EXTENSION vector` + HNSW index succeed at the chosen dim; **single-transaction write** — force an in-tx error → full rollback, no orphan (the property the prod 503 path violates); pgvector `<=>` ordering; `ts_rank` lexical; full hybrid; deep fallback; IDOR scoping.
**End-to-end acceptance (through `docker compose up`):** stack up with a local embedding+extractor → POST a session transcript → encoder extracts → `POST /api/memories/search` returns it in top-k → the written row carries a non-null **indexed** vector → recall@k vs an in-test brute-force exact-NN baseline.
**Success criteria (measurable):**
- Hybrid search p95 **< 50 ms @ 100k** rows, **< 150 ms @ 1M** rows (HNSW `m=16`, tuned `ef_search`).
- `docker compose up` → first healthy `/health` accepting writes **< 60s** (image pull excluded).
- Hybrid **recall@10 ≥ 0.95** vs brute-force @100k (≥0.93 @1M with `halfvec`).
- **0 orphaned** (vector-missing) rows under fault injection — the defining Tier-1 win.
- Encoder sustains **≥10 sessions/min** end-to-end on one VM without queue backlog growth.

### M3 — Tier 2 adapters + API-compatibility gate
**Build:** Milvus VectorStore (2 collections, exact prod index params, degraded-mode, `is_connected()`), vLLM Embedder (hard-check removed), Redis-Sentinel Queue, TensorZero Extractor, GCP-KMS, Oathkeeper-JWT + service-token Auth, `era-telemetry` adapter under `[era]`, in-tree service-contract renderer; resolve the encoder Py2-syntax + duplicate-embedding-client debt.
**Validate:** conformance suite on the Milvus/Postgres/Redis adapters; the §8.2 golden test (OpenAPI diff + replay + failure-semantics) against current era-memory on shared fixtures.
**Success criteria (measurable):**
- **API-compat golden test: 100% pass** (OpenAPI identical modulo version; replay byte-identical modulo `{id, timestamps, latency_ms}`).
- Search ranking **identical** to current era-memory on a fixed corpus + fixed embeddings: id-order and `score` match to ≥1e-9 for all 4 strategies (deep included).
- Dual-write failure semantics preserved: Milvus-down → 503 with each exact detail string; Postgres row persists 100% of injected-failure runs; delete swallows Milvus failure; `ON CONFLICT` skips vector insert 100% of retry runs.
- Latency parity: search & create/batch p50/p95 within **±10%** of current era-memory at equal QPS on identical hardware/fixtures.
- `pip install era-memory-light` (no `[era]`) succeeds on a clean machine **with no GitHub auth**; `[era]` reproduces current observability.

### M4 — Hardening, docs, release
**Build:** the cross-tier re-embed migration runbook (§6); per-tier quickstarts; the published OSS package; the secondary LLM-judge quality-regression harness wired as non-gating CI.
**Success criteria:** all three tier quickstarts reproduced from scratch by someone outside the team following only the docs; conformance + golden suites green in CI on every adapter; semver + changelog established.

### (Optional) M5 — era-core collapse
Retire era-core's in-tree `era-memory` onto light Tier 2. **Blockers (must all clear first):** the M3 API-compat gate green; the dual-write orchestrator + failure-semantics gate green; encoder imports cleanly on the target Python; `[era]` extra guaranteed in-cluster; **schema/migration ownership** (per the manual-migration runbook — `migrate_schema.py` is not run by CD); **edge routing** repointed via Traefik IngressRoute with identical path prefixes; Auth issuer/JWKS/service-token match Oathkeeper exactly; **Milvus collections bound to the same live collections** (`era_memories`, `era_session_content`, dim 2048) — a fresh bootstrap against empty collections loses all production vectors.

---

## 11. Decisions to settle with the engineer before M2/M3

1. **Tier-1 dimension strategy (blocks M2):** `halfvec(2048)` to keep the prod model, **or** a ≤2000-d Tier-1 model. Recommendation: `halfvec(2048)` if model parity with prod matters; ≤2000-d hosted model if simplicity wins.
2. **Cross-tier migration:** confirm **single-tier-for-life + re-embed** (recommended) vs attempting interoperable dims (not recommended; physically blocked across model families).
3. **Public OSS vs private repo:** gates whether the `era-*` git deps may even be referenced. If public OSS, the Telemetry/version ports (§7) are mandatory, not optional.
4. **Tier-1 lexical default:** `tsvector`/`ts_rank` on the stock image (recommended) vs ParadeDB BM25 for prod-identical relevance (needs custom image).
5. **Tier-0 default dim:** 384 (`bge-small`, smallest/fastest — recommended) vs 768 (`nomic`, Matryoshka, reuses `_truncate_and_normalize`).
6. **Tier-0/1 encryption default:** off (simplest) vs on with a persisted local key.

---

## 13. Repository identity, naming & first deployment

**The library is independent of era-core.** It is its own GitHub repo, with its own CI/CD, and deploys to its own GCP project — it does not live in the era-core workspace, the era-core cluster, or the `era-core-platform` GitOps repo. era-core's in-tree `era-memory` becomes *one eventual Tier-2 consumer*, not the center of gravity.

**First consumer / reference deployment: internal tools on `era-labs-tools` GCP.** This is a **Tier-1** deployment and the dogfood target that proves M2:
- Cloud Run (or small GKE) for the API + encoder
- **Cloud SQL for Postgres + `pgvector`** as the single record+vector store (`halfvec(2048)` per the locked decision)
- An OpenAI-compatible embedding endpoint
- Static bearer-token auth; Telemetry off or Cloud-native
- Synced by its **own** Argo/Cloud Run config — **not** era-core-platform's ApplicationSet.

**The two `era-memory`s (rename discipline).** The string `era-memory` means two unrelated things in era-core; only one is ever renamed:
1. **Build/repo name** `era-memory` — may be renamed to `era-memory-internal` to free the GitHub-org name. Touches: GitHub repo (auto-redirects), GCP Artifact Registry path, the repo's own `cd.yaml`, `clone-era-core.sh`. **Argo CD is insulated** — its `repoURL` is `era-core-platform`, not the app repo — so a repo rename needs **no Argo change** unless the AR registry path is also renamed (avoid that; keep the legacy AR path → zero kustomize/image edits).
2. **Runtime namespace + service identity** `era-memory` / `era-memory-service` / `era-memory-encoder` — **never rename.** Traefik IngressRoutes, NetworkPolicies, Milvus/Redis/Qdrant namespace selectors, certs (`era-memory-cert`), Pyrra SLOs, Grafana/Prometheus all key off these. Renaming them breaks prod.

Because the new library deploys to `era-labs-tools` (its own project + namespace), it touches **neither** identity. The incumbent GitHub rename is therefore an **optional, decoupled, do-it-later step** with no cluster/Argo impact — captured here, not blocking development.

---

## 12. Provenance

Built from a direct read of `era-core/era-memory` (`era-memory-service`, `era-memory-encoder`, `docs/`, `pyproject.toml`, `scripts/migrate_schema.py`). Each tier was independently feasibility-validated against the real code and current library capabilities (sqlite-vec metadata/partition columns, FTS5 `bm25()`, fastembed/ONNX, pgvector `halfvec` + HNSW 2000-dim cap, ParadeDB vs `ts_rank`, Milvus index params). All three verdicts: **feasible with changes** — every required change is folded into §5 and the milestone gates above.
