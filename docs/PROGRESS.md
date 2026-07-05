# Milestone progress

Spec: [era-memory-light-spec.md](era-memory-light-spec.md). Gates are automated tests.

## ✅ M0 — Interfaces & pure core (no infra) — DONE
- 9 ports (`src/era_memory/ports/`) as ABCs; KMS-style seam generalized.
- Pure core logic lifted from era-core (`core/logic/`): RRF fusion (k=60, 0.6/0.4),
  recency decay (30-day half-life), cosine dedup (0.85), Matryoshka truncate, entropy gate,
  envelope encryption (AES-GCM + AAD).
- Dual-write **orchestrator** (`core/orchestration.py`) with the co-transactional collapse;
  3 literal 503 detail strings pinned; `(row, was_inserted)` dedup signal.
- In-memory reference adapters for all 9 ports (`adapters/memory/`).
- Conformance suite (`tests/conformance/`) + orchestration tests + import-isolation gate.

**Gates green:** core logic ranking parity; orchestrator commit-ordering / skip / rollback;
`core/` imports no backend; base install pulls **zero** third-party deps (subprocess-checked).

## ✅ M1 — Tier-0 vertical slice (laptop, offline) — DONE
- SQLite adapters (`adapters/sqlite/`): records + FTS5 + sqlite-vec `vec0` in **one file**,
  one shared connection → dual-write **collapses to a single transaction**
  (`co_transactional = True`); the 503 orphan-row path is structurally impossible.
- vec0 with cosine distance, `user_id` partition key, metadata filtering (memory_type,
  experience_id) inside the KNN query. FTS5 `bm25()` lexical leg.
- `(model, dim)` lock-in guard (refuse mismatched reopen); persisted local KMS key.
- The **same conformance suite** runs against the SQLite adapters (parametrized fixtures).

**Gates green:** SQLite passes the full conformance suite; collapse atomicity (vector
failure rolls back the record — no orphan); model/dim guard; persistence across reopen;
persisted KMS key round-trips. **72 tests pass, ruff clean.**

## ✅ M2 — Tier-1 Postgres + pgvector (era-labs-tools shape) — DONE
- Postgres adapters (`adapters/postgres/`): `memories` (+ tsvector/`ts_rank` lexical) and
  `memory_vectors` (`halfvec` + HNSW cosine) in ONE database, sharing one asyncpg
  connection per unit-of-work → dual-write **collapses to a single transaction**
  (`co_transactional = True`). The column is `halfvec(dim)` — `dim` comes from the embedder
  at `connect()`, not a fixed number — so any model's dim works; `halfvec` indexes up to 4000
  dims, side-stepping pgvector's 2000-dim cap (so 2048+ models fit). A deployment is pinned to
  one `(model, dim)` for the store's life (`(model, dim)` guard; see `docs/adr/0001`).
  Embeddings bound as string literals cast to `halfvec` (no codec dep). ON CONFLICT upsert.
- **OpenAI-compatible embedder** (`adapters/openai/`) — the production embedding path;
  points at any `/embeddings` endpoint (OpenAI, a vLLM/GPU service on era-labs-tools,
  LiteLLM, Ollama). Matryoshka truncate+normalize client-side.
- Static **bearer-token auth** (`adapters/auth.py`); async wiring (`build_memory_async`).
- Lean **HTTP surface** (`app.py`, `[server]` extra): `/health`, `/ready`,
  `POST /api/memories`, `POST /api/memories/search` (era-core-shaped responses).
- **`Dockerfile` + `docker-compose.yml`** — the era-labs-tools deployment shape
  (swap Postgres for Cloud SQL there).

**Gates green:** the **same conformance suite** passes against pgvector; Tier-1 collapse
atomicity (vector failure rolls back the record — no orphan); user isolation; embedder
truncation; bearer auth; HTTP smoke. **Validated against a live `docker compose` stack**:
create → both `memories` and `memory_vectors` get the row in one transaction, hybrid search
ranks correctly, 401 on bad token, cross-user isolation holds.
**104 tests pass with Postgres (81 + 4 skipped without it), ruff clean.**

### Note on the default embedder
The default dev embedder is a **deterministic hashed bag-of-words stand-in** (offline,
non-semantic) — it keeps `build_memory()` runnable with zero setup. For real retrieval use a
real embedder (now shipped, see M1.1); a bare `build_memory()` emits a one-time warning when it
falls back to the stand-in.

## ✅ M1.1 — Offline ONNX embedder (`fastembed`) + zero-config setup — DONE
- `FastEmbedEmbedder` (`adapters/fastembed/`): ONNX/CPU, lazy model load, Matryoshka
  truncate+normalize, behind the `Embedder` port with no other change. Curated model registry
  (`bge-small` 384 / `mxbai-large` 1024), readiness-sentinel cache detection.
- **Embedder resolver** (`embedders.py`): endpoint env → already-cached local model →
  opt-in download → dev stand-in. Cache detection is file-only and construction is lazy, so a
  bare `build_memory()` stays backend-free until search runs (import-isolation gate still green).
- **`era-memory` CLI** (`__main__.py`): `setup` (interactive HF download w/ model choice,
  `--yes`/`--model`/`--force` for CI) and `status`. `build_memory(embedder="auto")` downloads
  on demand; default `build_memory()` only uses an already-cached model (no surprise downloads).

**Gates green:** resolver precedence; invalid-arg rejection; dev-fallback warning; import
isolation holds with `fastembed` installed + a model cached; gated network test downloads
`bge-small` and confirms a real semantic signal (`ERA_MEMORY_TEST_FASTEMBED=1`).

## ✅ 0.1.2 — production hardening (three live-incident fixes) — DONE
Discovered operating era-memory behind era-argus (2026-07-02):

- **Bounded ONNX Runtime thread pool.** The offline `fastembed`/ORT embedder spawned an
  unbounded intra-/inter-op thread pool; under a 1-CPU GKE CFS quota a large-model
  (`mxbai-large`) inference starved the process — including FastAPI probe responses — and
  kubelet SIGTERMed a healthy pod. `FastEmbedEmbedder` (and the `download_model` warmup) now
  pass a bounded `threads` to fastembed's `TextEmbedding` (→ `SessionOptions.intra_op_num_threads`
  / `inter_op_num_threads`). New env `MEMORY_EMBED_THREADS`, default `2` — small enough that
  probe responses stay schedulable under the quota. The k8s cpu:2 + relaxed-probe change was
  the mitigation; this is the durable fix.
- **`content_hash` is now populated.** The `memories.content_hash` column + `ix_mem_user_hash`
  index existed in both backends but no writer set them — every row was NULL, and an external
  dedup pass grouping on the column mass-collapsed distinct memories. `MemoryRecord.__post_init__`
  now stamps `content_hash = SHA-256 hex of the exact content` whenever the caller leaves it
  None, so every writer (store path + encoder pipeline) persists it in both the sqlite and
  postgres adapters. **Backfill:** no migration rewrites existing rows — **pre-0.1.2 rows keep
  `content_hash = NULL`**; only rows written at ≥0.1.2 carry the hash.
- **Write-time idempotency on `POST /api/memories`.** A client retry after a timeout (server
  completed after the client aborted) previously created an exact-duplicate episode. Now that
  `content_hash` is populated, both adapters' existing `(user_id, content_hash)` dedup on
  ACTIVE rows engages: a duplicate create returns the existing first record instead of
  inserting. The HTTP response keeps the same `{id, memory_type, source_type}` shape and adds
  `deduplicated: true` **only** when it hit (absent on a fresh insert). Dedup keys on
  `(user_id, content_hash)` **only** — a byte-identical body with a **different `experience_id`**
  still collapses onto the first record, which is preserved untouched (the recall value of a
  byte-identical body twice is nil regardless of the experience tag).

**Gates green:** ruff clean; full pytest suite (thread param plumbed to the constructor +
env override; `content_hash` populated and persisted across backends; idempotent create
returns the existing id + `deduplicated`; same-content-different-user is **not** deduped).

## ⏭ Next
- **Deploy:** push the image to `era-labs-tools` (Cloud Run + Cloud SQL pgvector), point at a
  real embedding endpoint, set a real `MEMORY_BEARER_TOKEN`.
- **M3:** Tier 2 — Milvus/vLLM/Redis adapters + the API-compatibility golden test vs era-core.
