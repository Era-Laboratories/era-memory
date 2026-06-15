from __future__ import annotations

import abc


class KMS(abc.ABC):
    """
    Wraps/unwraps Data Encryption Keys for envelope encryption. Tier 0/1 = local master
    key; Tier 2 = GCP KMS. Mirrors era-core's existing KmsProvider ABC (the reference seam).
    """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str: ...

    @abc.abstractmethod
    async def generate_dek(self) -> bytes:
        """Return a fresh 32-byte plaintext data key."""

    @abc.abstractmethod
    async def wrap_dek(self, dek: bytes, aad: str) -> bytes:
        """Encrypt a DEK under the master key, binding ``aad``."""

    @abc.abstractmethod
    async def unwrap_dek(self, wrapped: bytes, aad: str) -> bytes:
        """Decrypt a wrapped DEK; raise on aad/key mismatch."""
