from __future__ import annotations

import abc
import logging
from typing import Any


class Telemetry(abc.ABC):
    """
    Observability seam. No-op default keeps the base install free of the private
    era-telemetry git dep; the ``[era]`` extra provides the real OTel adapter.
    """

    @abc.abstractmethod
    def event(self, name: str, **fields: Any) -> None: ...

    def logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)
