"""
Envelope encryption format. AES-256-GCM, lazy ``cryptography`` import (opt-in extra).

Format:  version(1) || nonce(12) || ciphertext || tag(16)   (tag appended by AESGCM)
AAD:     "{user_id}:{field_name}:{dek_version}"  — binds ciphertext to its field/user so a
         blob can't be replayed into another field. Mirrors era-core's envelope + AAD.

The DEK is a 32-byte data key; KMS adapters wrap/unwrap it. This module only does the
content-layer encryption given a DEK.
"""

from __future__ import annotations

import os

_VERSION = b"\x01"
_NONCE_LEN = 12


def _aesgcm(dek: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Encryption requires the 'cryptography' extra: pip install era-memory[encryption]"
        ) from e
    return AESGCM(dek)


def make_aad(user_id: str, field_name: str, dek_version: int) -> bytes:
    return f"{user_id}:{field_name}:{dek_version}".encode("utf-8")


def encrypt(plaintext: str, dek: bytes, *, user_id: str, field_name: str, dek_version: int = 1) -> bytes:
    aes = _aesgcm(dek)
    nonce = os.urandom(_NONCE_LEN)
    aad = make_aad(user_id, field_name, dek_version)
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), aad)
    return _VERSION + nonce + ct


def decrypt(blob: bytes, dek: bytes, *, user_id: str, field_name: str, dek_version: int = 1) -> str:
    if not blob or blob[:1] != _VERSION:
        raise ValueError("unrecognized envelope version")
    nonce = blob[1 : 1 + _NONCE_LEN]
    ct = blob[1 + _NONCE_LEN :]
    aes = _aesgcm(dek)
    aad = make_aad(user_id, field_name, dek_version)
    pt = aes.decrypt(nonce, ct, aad)  # raises InvalidTag on AAD/key mismatch
    return pt.decode("utf-8")
