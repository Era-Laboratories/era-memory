# Changelog

## 0.1.4

### Breaking: the HTTP server binds its bearer token to one configured owner

- **New required setting `MEMORY_PRINCIPAL`.** It names the single owner (`user_id`) that
  `MEMORY_BEARER_TOKEN` acts as. Every request that presents the token reads and writes as
  this owner.
- **`X-User-Id` is ignored.** Before 0.1.4, a request with a valid token could name any
  owner in the `X-User-Id` header, so one shared secret gave access to every owner's data.
  The header no longer has any effect. Clients can stop sending it.
- **The HTTP server never runs unauthenticated.** Startup now fails when
  `MEMORY_BEARER_TOKEN` or `MEMORY_PRINCIPAL` is unset or empty. Before 0.1.4, an empty token
  silently left the server open with no auth. `create_app(memory)` also refuses a `Memory`
  that still carries the in-process `NoopAuth` default.
- `BearerAuth(token, principal)`: the constructor now takes the principal and rejects an
  empty token or principal. The `user_header` keyword is removed.
- `NoopAuth` (the in-process / library default) no longer reads `X-User-Id`. It always
  returns its `default_user`.

**Upgrade:** set `MEMORY_PRINCIPAL` to the `user_id` your existing rows are stored under
before you roll out 0.1.4. The variable is harmless to 0.1.3, so set it first, then move the
image. A different value orphans the existing rows (they stay in the store, but no request
can reach them).
