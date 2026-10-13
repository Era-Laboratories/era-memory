"""Static bearer-token Auth adapter (Tier 1). Drop-in for the Oathkeeper-JWT Auth port.

The token is a credential for exactly one owner: ``principal``. A request that presents
the token acts as that principal and nothing else. Client-supplied identity headers
(``X-User-Id`` and friends) are ignored — a shared secret cannot vouch for a caller-chosen
user id, so honouring one would let any token holder read and write any owner's data.
"""

from __future__ import annotations

import hmac
from typing import Optional

from ..ports import Auth


class BearerAuth(Auth):
    def __init__(self, token: str, principal: str) -> None:
        if not token:
            raise ValueError("BearerAuth needs a non-empty token")
        if not principal:
            raise ValueError("BearerAuth needs a non-empty principal")
        self._token = token
        self._principal = principal

    async def authenticate(self, headers: dict[str, str]) -> Optional[str]:
        # HTTP header names are case-insensitive (Starlette lower-cases them).
        lower = {k.lower(): v for k, v in headers.items()}
        auth = lower.get("authorization", "")
        if not auth.startswith("Bearer "):
            return None
        if not hmac.compare_digest(auth[7:], self._token):
            return None
        return self._principal
