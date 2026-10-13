# Security Policy

## Supported versions

era-memory is pre-1.0; security fixes are made on the latest released `0.x` line.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's **["Report a vulnerability"](https://github.com/Era-Laboratories/era-memory/security/advisories/new)**
button (repository → **Security** tab → **Report a vulnerability**). This opens a private
advisory visible only to you and the maintainers. Private vulnerability reporting is enabled on
this repository.

If you cannot use GitHub, email **hello@era.world**.

Please include:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept),
- affected version(s) and configuration (tier, backends),
- any suggested remediation.

## What to expect

- **Acknowledgement** within a few business days.
- An assessment and, if confirmed, a fix on a private branch, coordinated disclosure, and a
  patched release.
- Credit in the advisory if you'd like it.

## Scope notes

A few things are deployment responsibilities rather than library vulnerabilities — useful to
know before reporting:

- **Tier 1 bearer-token auth** is a single shared secret bound to one owner, `MEMORY_PRINCIPAL`.
  Every request with the token acts as that owner; `X-User-Id` is ignored. The HTTP server refuses
  to start without both the token and the principal. It is not per-user identity — for
  multi-tenant use, run one token/principal pair per owner or add a real per-user `Auth` adapter.
- **Embedding/HTTP endpoints** you configure (`MEMORY_EMBEDDING_URL`, DSNs, tokens) and the
  secrets for them are managed by you — keep them out of source control.

Genuine issues in era-memory's own code (e.g. cross-`user_id` data leakage, injection in the
storage/search layer, encryption flaws) are in scope and very much want to hear about.
