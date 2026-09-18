# Instructions for coding agents

Read these before you change anything in this repository.

1. **Parity work is ON HOLD.** Read [`docs/HANDOVER-mneme-parity.md`](docs/HANDOVER-mneme-parity.md)
   first. It holds the start conditions, the owner decisions, the classification of what to
   port, the five open requirements, and the order of work. Do not start the port, and do not
   build the old M3 plan (Milvus/vLLM/Redis adapters), until its start conditions are true.
2. This repository is public. Never add Era-internal identity code, hostnames, deployment
   manifests, ticket contents or production data.
3. Follow [`CONTRIBUTING.md`](CONTRIBUTING.md): branch off `main`, open a PR, keep both
   invariants, and keep CI green. Never push to `main`.
4. Milestone state is in [`docs/PROGRESS.md`](docs/PROGRESS.md). The spec is
   [`docs/era-memory-light-spec.md`](docs/era-memory-light-spec.md).
