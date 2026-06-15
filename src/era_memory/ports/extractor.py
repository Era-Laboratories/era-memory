from __future__ import annotations

import abc

from ..models import ExtractionResult, SessionPayload


class Extractor(abc.ABC):
    """
    Conversation -> typed memories. Tier 0 default is heuristic/no-LLM (offline);
    Tier 1 = OpenAI-compatible chat; Tier 2 = TensorZero.
    """

    @abc.abstractmethod
    async def extract(self, payload: SessionPayload) -> ExtractionResult: ...
