"""Bearer-token Auth adapter (Tier 1): the token stands for one configured principal."""

from __future__ import annotations

import pytest

from era_memory.adapters.auth import BearerAuth
from era_memory.adapters.memory import NoopAuth


async def test_valid_token_returns_configured_principal():
    auth = BearerAuth("s3cret", "owner")
    assert await auth.authenticate({"Authorization": "Bearer s3cret"}) == "owner"


async def test_client_supplied_user_header_is_ignored():
    auth = BearerAuth("s3cret", "owner")
    headers = {"Authorization": "Bearer s3cret", "X-User-Id": "someone-else"}
    assert await auth.authenticate(headers) == "owner"


async def test_wrong_token_rejected():
    auth = BearerAuth("s3cret", "owner")
    assert await auth.authenticate({"Authorization": "Bearer nope", "X-User-Id": "u1"}) is None


async def test_missing_bearer_rejected():
    auth = BearerAuth("s3cret", "owner")
    assert await auth.authenticate({"X-User-Id": "u1"}) is None


@pytest.mark.parametrize(("token", "principal"), [("", "owner"), ("s3cret", "")])
def test_empty_token_or_principal_refused(token, principal):
    with pytest.raises(ValueError):
        BearerAuth(token, principal)


async def test_noop_auth_ignores_user_header():
    assert await NoopAuth().authenticate({"X-User-Id": "someone-else"}) == "local"
