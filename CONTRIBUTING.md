# Contributing to era-memory

Thanks for your interest! era-memory is a **ports & adapters** memory system: all logic depends
on nine small interfaces, and a backend is "just" an adapter that satisfies a port. That shape
drives how contributions work.

## Development setup

```bash
git clone https://github.com/Era-Laboratories/era-memory.git
cd era-memory
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,sqlite,postgres,openai,server,encryption]"
```

## Run the checks

```bash
ruff check src tests                  # lint (line length 100, target py310)
pytest -q                             # unit + conformance (Postgres tests skip if no DSN)

# Optional, fuller coverage:
MEMORY_TEST_PG_DSN=postgresql://postgres:era@localhost:55433/era pytest -q   # Postgres-backed
ERA_MEMORY_TEST_FASTEMBED=1 pytest -q tests/test_embedder_resolution.py      # downloads a model
```

A quick local Postgres for the DSN above: `docker compose up -d postgres` (pgvector image).

## Two invariants that PRs must preserve

These are enforced by tests — please don't work around them:

1. **The base import pulls zero third-party dependencies.** `import era_memory` and a bare
   `build_memory(tier=0)` must not import any backend (`tests/test_import_isolation.py`). Keep
   backend imports *inside* functions/adapters, never at module top level.
2. **The conformance suite is the contract.** Adding a backend means implementing a port under
   `src/era_memory/adapters/<name>/` and making the existing `tests/conformance/` suite pass for
   it — unchanged. Don't fork the suite per backend; wire your adapter into the fixtures.

## Submitting changes

1. Branch off `main` (the repo requires PRs — direct pushes to `main` are blocked).
2. Make your change with a test; keep `ruff` and `pytest` green.
3. Open a PR. CI (lint + tests incl. Postgres-backed conformance) must pass before merge.
4. Write a clear description: what changed and why. For a notable design decision, add an ADR
   under `docs/adr/` (see [`docs/adr/0001-...`](docs/adr/0001-dimension-is-a-per-deployment-contract.md)).

## Scope tips

- **New embedder / vector store / record store** → implement the relevant port, pass conformance.
- **Retrieval tuning** (RRF, recency, dedup) lives in `src/era_memory/core/logic/` and is pure;
  changing constants there affects ranking parity across all tiers — call it out explicitly.
- **Docs-only / typo fixes** are very welcome and don't need an ADR.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to
uphold it.
