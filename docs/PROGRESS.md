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
  (`co_transactional = True`). `halfvec` side-steps pgvector's 2000-dim cap (locked at
  `halfvec(2048)` for prod parity; tests use small dims). Embeddings bound as string
  literals cast to `halfvec` (no codec dep). `(model, dim)` guard + ON CONFLICT upsert.
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

### Known limitation
The default dev embedder is a **deterministic hashed bag-of-words stand-in** (offline,
non-semantic). Production uses the OpenAI-compatible embedder against a real endpoint; the
true-offline-laptop ONNX embedder (`fastembed`, 384-d) is a later follow-up (M1.1) and slots
behind the `Embedder` port with no other change.

## ⏭ Next
- **M1.1:** ONNX (`fastembed`) embedder for the true-offline-laptop tier.
- **Deploy:** push the image to `era-labs-tools` (Cloud Run + Cloud SQL pgvector), point at a
  real embedding endpoint, set a real `MEMORY_BEARER_TOKEN`.
- **M3:** Tier 2 — Milvus/vLLM/Redis adapters + the API-compatibility golden test vs era-core.
