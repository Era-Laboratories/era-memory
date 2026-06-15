"""Bearer-token Auth adapter (Tier 1)."""

from __future__ import annotations

from era_memory.adapters.auth import BearerAuth


async def test_valid_token_returns_user():
    auth = BearerAuth("s3cret")
    assert await auth.authenticate({"Authorization": "Bearer s3cret", "X-User-Id": "u1"}) == "u1"


async def test_wrong_token_rejected():
    auth = BearerAuth("s3cret")
    assert await auth.authenticate({"Authorization": "Bearer nope", "X-User-Id": "u1"}) is None


async def test_missing_bearer_rejected():
    auth = BearerAuth("s3cret")
    assert await auth.authenticate({"X-User-Id": "u1"}) is None
