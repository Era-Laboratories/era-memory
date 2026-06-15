from __future__ import annotations

import abc
from typing import Optional


class BlobStore(abc.ABC):
    """
    Large-object storage. NOTE: era-core production has no object store — this port exists
    for completeness and is no-op at Tier 2. Tier 0/1 may use local filesystem.
    """

    @abc.abstractmethod
    async def put(self, key: str, data: bytes) -> None: ...

    @abc.abstractmethod
    async def get(self, key: str) -> Optional[bytes]: ...
