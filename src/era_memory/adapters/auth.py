"""Static bearer-token Auth adapter (Tier 1). Drop-in for the Oathkeeper-JWT Auth port."""

from __future__ import annotations

import hmac
from typing import Optional

from ..ports import Auth


class BearerAuth(Auth):
    def __init__(self, token: str, *, user_header: str = "X-User-Id") -> None:
        self._token = token
        self._user_header = user_header

    async def authenticate(self, headers: dict[str, str]) -> Optional[str]:
        # HTTP header names are case-insensitive (Starlette lower-cases them).
        lower = {k.lower(): v for k, v in headers.items()}
        auth = lower.get("authorization", "")
        if not auth.startswith("Bearer "):
            return None
        if not hmac.compare_digest(auth[7:], self._token):
            return None
        return lower.get(self._user_header.lower())
