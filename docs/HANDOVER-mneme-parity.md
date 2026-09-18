# Handover — parity with Mneme

**Audience:** the engineer or coding agent that starts the parity work.
**Written:** 2026-09-18. **Baseline:** `era-memory@1cf1697` (0.1.2 line).
**Status:** ON HOLD. Read "Start conditions" before you change any code.

## TL;DR

Mneme is Era's internal memory service. It shares an origin with `era-memory` and has had far
more development. The goal is parity where a capability is portable. Mneme is the source of
truth. `era-memory` receives versioned cuts of the portable parts and publishes them. Mneme
never depends on `era-memory`.

Nothing is built yet. This document holds the analysis, the owner decisions, the five
requirements that came from a consumer of this package, and the order of work. The current
M3 plan (Milvus/vLLM/Redis adapters) is obsolete, because Mneme no longer uses Milvus.

Era engineers: the file-level map of Mneme (routes, modules, line references) is in the
private `era-core` repo at `docs/plans/2026-09-18-era-memory-parity/manifest.md`. This public
document is complete for the `era-memory` side of the work.

---

## 1. Start conditions

Do not start the port until both conditions are true.

1. **The Mneme description fix has merged.** Mneme's generated OpenAPI descriptions and two of
   its docs still described behavior that the code had removed: importance as a ranking
   factor, soft delete, an "archived" memory state, and a Milvus connection. The Mneme owner
   tracks the fix as ERA-12223. The generated spec is the parity fixture, so a stale
   description becomes a wrong public contract. Pin the first cut to the Mneme commit that
   merges that fix. On 2026-09-18 Mneme `main` was at `22502b8f`, without the fix.
2. **The repository owner lifts the hold.** The Mneme owner asked on 2026-09-18 to wait
   before `era-memory` changes. Until the hold lifts, open no feature branch, PR or release
   here. Two tasks touch nothing public and may run before that: the scale spike (section 6)
   and the golden fixtures (section 7, step 1).

Before the first change, also decide what to do with `uv.lock`. It is untracked in the
working copy that produced this analysis and is not part of `main`.

## 2. Owner decisions (2026-09-18)

| # | Decision |
|---|---|
| 1 | **Mneme is the source of truth.** `era-memory` receives versioned cuts. An earlier recommendation made `era-memory` own the retrieval modules with Mneme as a consumer. The owner rejected it. Do not build it. |
| 2 | **No separate tracker.** This document and the private manifest track the work. |
| 3 | **Classify first, then fix the Mneme descriptions, then port.** The classification is section 4. |
| 4 | **Importance leaves the ranking**, as it did in Mneme on 2026-09-15. See section 5. |

Keep the general public API. Parity arrives as a Mneme compatibility profile beside it, not
as a replacement.

## 3. Where the two differ today

| Area | `era-memory@1cf1697` | Mneme |
|---|---|---|
| HTTP surface | 5 routes: `/health`, `/ready`, create, delete, search (`src/era_memory/app.py`) | 24 service operations and 5 encoder operations |
| Search request | query, filters, strategy, ranking knobs | two fields: a probe and a limit. Unknown fields are a validation error. |
| Search legs | configurable strategies, bounded candidate lists | both legs always run. No strategy. |
| Lexical leg | SQLite FTS5 or Postgres `ts_rank` | real BM25 over its own analyzer |
| Vector leg | `vec0` (SQLite), pgvector (Postgres) | exact in-memory scan over a sealed, quantized per-user index |
| Score | RRF x importance x recency, recency weight `0.3` | RRF x recency only, recency weight `0.4` |
| Recency input | `created_at` | the event time of the source item |
| Response | memory records | episodes with ranking diagnostics |
| Domain model | general memory types, portable tiers | immutable episodes, biographies, per-owner encryption keys, rebuild audit records |
| Delete | `delete` (soft) and `purge` (erasing) | hard delete only |
| Vector backend plan | M3 targets Milvus | Milvus removed |

## 4. Classification

**Port** means: cut the Mneme module with the smallest change that removes Era bindings, and
pin the cut to a Mneme commit. **Adapter** means: the behavior is portable, and it arrives
behind an existing `era-memory` port with a local adapter. **Internal** stays private.

| Capability | Class | Note |
|---|---|---|
| Fusion and re-scoring constants | port | Replaces `core/logic/rrf.py`. See section 5. |
| Two-leg episode search | port | Removes the strategy switch in the compatibility profile. |
| Text analyzer | port | Pure function. Golden-fixture target. |
| BM25 lexical leg | port | |
| Exact vector leg with quantizer | port | A candidate replacement for `vec0` at Tier 0. See section 6. |
| Sealed index codec, write path, mutation | port | Golden-fixture target. |
| Sealed index storage, vault, blob cache | adapter | Behind the `BlobStore` port. Local disk first. |
| Index rebuild | port | As a CLI command, not an HTTP route. |
| Search route, compatibility profile | port | Two-field request, episodes response. |
| Batch store route | port | Requirement 4. |
| List, get one, delete one | port | The profile delete maps to `purge`. |
| Erase all data for the caller | port | A privacy requirement for public users too. |
| Biography read and update | adapter | A second domain object. After retrieval and the record routes. |
| Vault prewarm | adapter | Only meaningful with the sealed index. |
| `/version` | adapter | Package version and git SHA only. |
| Envelope encryption core, DEK cache, local KMS | port | Diff against `core/logic/envelope.py` and the `kms` port first. |
| Cloud KMS provider | adapter | Optional extra. |
| Encoder `/encode` | adapter | `Memory.encode` and `submit_session` exist. Compare prompts and the refusal rule. Do not port the queue and gateway clients. |
| Metrics | adapter | Behind the `Telemetry` port. |
| Embedding client | adapter | Text-only in Mneme too. Requirement 1 goes past Mneme. |
| Admin routes, platform-admin gate, admin audit | internal | Bind to Era identity. |
| Encryption operations routes | internal | Operate Era's key migration. |
| Owner and org resolution | internal | The public profile uses `user_id` from the `Auth` port as the single owner. |
| Service auth, feature flags, sandbox, profiling, deployment manifests, CI | internal | |

## 5. Ranking change (breaking)

| | Today | Target |
|---|---|---|
| Score | `final_score(base, importance, recency)` in `core/search.py` | `rrf_score x (1 - 0.4 + 0.4 x recency)` |
| Recency weight | `0.3`, from `MEMORY_RECENCY_WEIGHT` (`config.py`) | `0.4` |
| Half-life | 30 days | 30 days |
| Recency input | `created_at` | event time (requirement 5) |
| Importance | multiplier | stored and returned. Not a ranking input. |

Recency moves the score inside `[0.6, 1]` only. It separates comparably relevant results. It
never puts a weak new result above a strong old one.

`importance_score` stays on `MemoryRecord` and in responses. Mneme keeps the column and
reports it as a diagnostic, and its owner confirmed it stays. `min_importance` leaves the
ranking path. Record the change as breaking in the release notes.

Open point: decide whether the general API keeps `MEMORY_RECENCY_WEIGHT` as a setting. The
compatibility profile must use the constant.

## 6. The five requirements

These came from a consumer of `era-memory` that imports large personal archives and uses a
multimodal embedder.

| # | Requirement | Class | Evidence at `1cf1697` | Change |
|---|---|---|---|---|
| 1 | Multimodal Embedder input | public-only | `ports/embedder.py`: `embed(self, texts: list[str])` | Widen the port to typed input items (text, image). A text-only embedder rejects an image item with a typed error. The Qwen3-VL adapter is the first consumer. |
| 2 | Replace by external key | public-only | Idempotency keys on `(user_id, content_hash)` only (`models.py`, `memory.py`). An edited note gets a new hash, so a second ACTIVE record appears. Mneme episodes are immutable and carry no such key. | Add an optional `external_key`. A store with a known `(user_id, external_key)` replaces the active record and its vector in one unit of work. Hash idempotency stays for records without a key. |
| 3 | Source and person filters | public-only | `vec0` metadata columns are `memory_type`, `experience_id`, `created_at` (`adapters/sqlite/__init__.py`). `SearchFilters` has `memory_type` and `experience_id`. | Add `source` and `person` to the record, to `SearchFilters`, and to the vector metadata in every adapter. A `vec0` column change needs a table rebuild, so ship a migration. These filters belong to the general API. The compatibility profile forbids extra fields. |
| 4 | Batch store | port | `dual_write_batch` exists in `core/orchestration.py`. No facade method and no route call it. | Add `Memory.store_many` and `POST /api/memories/batch`. Requirement 2 applies per item. Match Mneme's session guard: a session whose episodes were all deleted has no episodes, so the guard accepts a new batch. |
| 5 | Recency from the source item time | port | `core/search.py` computes age from `rec.created_at`. A bulk import gives a 2019 message the same recency as a message from today. `temporal_anchor` exists on the record as `Optional[str]`, and search ignores it. | Make the event time a typed epoch field. Importers set it. A live store defaults it to the store time. Search scores from it. Carry it in the vector metadata. Mneme requires the field and has no `created_at` fallback. |

Requirements 4 and 5 are parity items. Mneme already has both. Requirements 1, 2 and 3 have
no Mneme equivalent and are public-side design.

### Open measurement — Tier 0 at one million rows (spike S4)

`vec0` stores `float[dim]`. One million chunks at 1024 dimensions need about 4.1 GB
(1,000,000 x 1024 x 4 bytes), before metadata and the FTS5 table.

`vec0` search is believed to be an exact scan. Nobody verified that against the current
sqlite-vec release. Verify it in the spike.

Spike S4 measures Tier 0 search at one million rows: p50 and p95 latency, resident memory and
file size. Run Mneme's exact vector leg with its quantizer on the same data, because that
module is a port item and may replace `vec0` at this scale. The spike needs no change to this
repository. It was proposed on 2026-09-18 and deferred by the owner.

## 7. Order of work

1. Golden fixtures from the pinned Mneme commit: analyzer, sealed codec, BM25, vector
   ranking, fusion, re-scoring. Sanitized data only. No production content.
2. Retrieval cut: fusion, two-leg search, the search index modules. Includes section 5 and
   requirement 5.
3. Requirements 4, 2, 3, 1, in that order. Batch is first because importers need it.
4. Record routes and the Mneme compatibility profile.
5. Encryption cut, then biography, then the encoder comparison.
6. Replace the M3 entry in `docs/PROGRESS.md` and the "Not yet available" line in `README.md`
   with the sealed-retrieval milestone.

Each cut states the Mneme commit it came from, in the PR description and in the changelog.

## 8. Rules for the port

- Never copy Era identity, org resolution, admin gates, deployment manifests, internal
  hostnames or production data into this repository. It is public.
- Keep ports and adapters. A ported module depends on a port, never on a concrete backend.
- Keep both CONTRIBUTING invariants. Run the full conformance suite against the in-memory,
  SQLite and Postgres backends.
- Ground every new dependency version against its current release before you add it.
- Validation at the baseline: `uv run pytest -q` gave 113 passed and 4 skipped, and
  `uv run ruff check src tests` was clean, without Postgres.
