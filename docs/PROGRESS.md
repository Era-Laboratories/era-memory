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

### Known limitation (next up)
The default Tier-0 embedder is a **deterministic hashed bag-of-words stand-in** — it proves
the storage/search plumbing but is **not semantic**. The real Tier-0 ONNX embedder
(`fastembed`, 384-d) is the immediate next task; it slots behind the `Embedder` port with no
other changes. Wire via `build_memory(tier=0, db_path=..., embedder=OnnxEmbedder())`.

## ⏭ Next
- **M1.1:** ONNX (`fastembed`) embedder adapter + offline acceptance test with a real model.
- **M2:** Tier 1 — Postgres + pgvector (`halfvec(2048)`), OpenAI-compatible embedder,
  bearer auth, docker-compose; deploy target = internal tools on `era-labs-tools` GCP.
- **M3:** Tier 2 — Milvus/vLLM/Redis adapters + the API-compatibility golden test.
