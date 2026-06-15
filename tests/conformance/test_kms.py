"""KMS conformance — every adapter must pass this unchanged."""

from __future__ import annotations

import pytest


async def test_generate_dek_length(kms):
    dek = await kms.generate_dek()
    assert len(dek) == 32


async def test_wrap_unwrap_round_trip(kms):
    dek = await kms.generate_dek()
    aad = "user1:content:1"
    wrapped = await kms.wrap_dek(dek, aad)
    assert wrapped != dek
    assert await kms.unwrap_dek(wrapped, aad) == dek


async def test_aad_mismatch_fails(kms):
    dek = await kms.generate_dek()
    wrapped = await kms.wrap_dek(dek, "user1:content:1")
    with pytest.raises(Exception):
        await kms.unwrap_dek(wrapped, "user1:OTHER_FIELD:1")


async def test_provider_name(kms):
    assert isinstance(kms.provider_name, str) and kms.provider_name
