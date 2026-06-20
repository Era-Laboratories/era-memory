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

## ⏭ Next
- **Deploy:** push the image to `era-labs-tools` (Cloud Run + Cloud SQL pgvector), point at a
  real embedding endpoint, set a real `MEMORY_BEARER_TOKEN`.
- **M3:** Tier 2 — Milvus/vLLM/Redis adapters + the API-compatibility golden test vs era-core.
