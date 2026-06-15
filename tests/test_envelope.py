"""Envelope encryption round-trip + DEK lifecycle. Needs the [encryption] extra."""

from __future__ import annotations

import pytest

cryptography = pytest.importorskip("cryptography")

from era_memory.core.logic import envelope  # noqa: E402


def test_encrypt_decrypt_round_trip():
    dek = b"\x00" * 32
    blob = envelope.encrypt("secret content", dek, user_id="u1", field_name="content")
    assert blob != b"secret content"
    assert envelope.decrypt(blob, dek, user_id="u1", field_name="content") == "secret content"


def test_aad_mismatch_raises():
    dek = b"\x01" * 32
    blob = envelope.encrypt("x", dek, user_id="u1", field_name="content")
    with pytest.raises(Exception):
        envelope.decrypt(blob, dek, user_id="u1", field_name="session_content")  # wrong field


def test_wrong_key_raises():
    blob = envelope.encrypt("x", b"\x02" * 32, user_id="u1", field_name="content")
    with pytest.raises(Exception):
        envelope.decrypt(blob, b"\x03" * 32, user_id="u1", field_name="content")


async def test_kms_dek_with_envelope(kms):
    """Generate a DEK, wrap it, unwrap it, and use it for content encryption."""
    dek = await kms.generate_dek()
    aad = "u1:content:1"
    wrapped = await kms.wrap_dek(dek, aad)
    recovered = await kms.unwrap_dek(wrapped, aad)
    blob = envelope.encrypt("hello", recovered, user_id="u1", field_name="content")
    assert envelope.decrypt(blob, dek, user_id="u1", field_name="content") == "hello"
