"""
Entropy gate for the extraction pipeline. Pure Python.

A cheap information-content heuristic (normalized Shannon entropy over word frequencies)
used to drop low-signal turns before the (expensive) extractor runs. Mirrors the role of
era-core's ``_entropy_score`` filter.
"""

from __future__ import annotations

import math
import re

_WORD = re.compile(r"\w+")


def entropy_score(text: str) -> float:
    """
    Normalized Shannon entropy in [0, 1] over word frequencies.

    0.0 for empty/single-token text; approaches 1.0 as the vocabulary becomes large and
    uniformly distributed (high information content).
    """
    words = _WORD.findall(text.lower())
    if len(words) < 2:
        return 0.0
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    total = len(words)
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_h = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return h / max_h if max_h > 0 else 0.0


def passes_entropy(text: str, *, threshold: float = 0.35) -> bool:
    return entropy_score(text) >= threshold
