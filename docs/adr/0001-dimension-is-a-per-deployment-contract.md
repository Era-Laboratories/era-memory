# ADR 0001 — Embedding dimension is a per-deployment contract, not a fixed value

- **Status:** Accepted
- **Date:** 2026-06-18
- **Context tier:** all (Tier 0 SQLite, Tier 1 Postgres/pgvector, Tier 2 Milvus)

## Context

Early documentation described the vector dimension as "locked at `halfvec(2048)` for
production parity," which read as if 2048 were a fixed property of the system. In practice
the code was already dimension-agnostic: the SQLite (`vec0(dim)`) and Postgres
(`halfvec(dim)`) vector columns are created from the embedder's output dimension at startup
(`wiring.py` passes `embedder.dimensions` into the store), guarded by a `(model, dim)`
signature check that refuses to reopen a store with a mismatched pairing.

Two facts make the "fixed 2048" framing actively misleading and worth correcting:

1. **Dimension is a free parameter; the *vector space* is what must be consistent.** Store
   and retrieval work at any dimension as long as every vector — index-time and query-time —
   comes from the *same model, same dimension, same MRL-truncation length, and same
   normalization*. The specific number (384, 768, 1024, 2048) does not affect correctness,
   only retrieval quality, storage size, and index build cost.

2. **Lightweight, redistributable embedders cap at ~1024 native dimensions.** The CPU/ONNX
   tier (all-MiniLM 384, bge/nomic 768, bge-large/mxbai/arctic 1024) tops out at 1024.
   Matryoshka (MRL) only truncates *down*, never up, so no lightweight model can produce a
   meaningful 2048-d vector. Reaching 2048+ requires a large model (Qwen3-4B @2560,
   Qwen3-8B @4096) MRL-truncated to 2048 — GPU-class, not offline-laptop. Forcing 2048
   everywhere either rules out the offline tier or wastes ~50% storage zero-padding a 1024
   model into a 2048 column.

## Decision

- Treat embedding dimension as a **per-deployment choice**, selected to match the embedder,
  and document it as such (README "Choosing a store, embedder, and dimension").
- Keep the **one hard invariant**: a store is pinned to a single `(model, dim)` for its
  lifetime. Changing model or dimension means re-embedding into a new store, not a config
  flip. This is enforced by the existing `(model, dim)` store guard and, additionally, a
  startup reconciliation (`_reconcile_embedding_dim`) that fails fast when an explicit
  embedder's `.dimensions` conflicts with a pinned `MEMORY_EMBEDDING_DIMENSIONS`.
- Provide deployment-size **pairings** (store + embedder + dimension) as guidance, with no
  single mandated default — deployers choose by scale and licensing needs.
- Do **not** standardize on 2048. It remains a valid choice for era-core parity (large model,
  MRL→2048), documented as one row in the pairings table rather than "the" dimension.

## Consequences

- **Positive:** A single Apache-2.0 embedder (e.g. `mxbai-embed-large-v1` @1024) can serve
  laptop SQLite through Cloud SQL Postgres identically — true offline↔hosted parity, which a
  fixed 2048 made impossible. Smaller dims cut storage and HNSW index cost. The offline tier
  (M1.1 fastembed/ONNX) is unblocked.
- **Negative / trade-off:** No one "blessed" dimension means deployers must make a choice;
  the pairings table and fail-fast check mitigate the footgun. Cross-deployment vector reuse
  is only valid when `(model, dim, truncation, normalization)` all match — called out
  explicitly.
- **Unchanged:** Retrieval logic, RRF/recency constants, and the co-transactional dual-write
  collapse are all dimension-independent and untouched.

## References

- `src/era_memory/wiring.py` — `_reconcile_embedding_dim`, embedder→store dim flow
- `src/era_memory/adapters/postgres/__init__.py` — `halfvec(dim)` column + `(model, dim)` guard
- `src/era_memory/adapters/sqlite/__init__.py` — `vec0(dim)` column + guard
- README — "Choosing a store, embedder, and dimension"
