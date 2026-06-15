from __future__ import annotations

import abc
from typing import Optional


class Auth(abc.ABC):
    """Request headers -> ``user_id`` (or None to reject). Tier 0 = no-op single user."""

    @abc.abstractmethod
    async def authenticate(self, headers: dict[str, str]) -> Optional[str]: ...
