"""Model adapter contract.

An adapter answers all queries that share one image in a single call, so
backends that can reuse work across those queries (the HF adapter's prefix
cache) get to do so. Closed-API adapters simply loop.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol

from PIL import Image

from ..schema import Prediction, Query
from .prompts import LETTERS


class ModelAdapter(Protocol):
    name: str
    reproducible: bool

    def run_group(self, image: Optional[Image.Image], queries: List[Query]) -> List[Prediction]:
        ...

    def meta(self) -> Dict[str, Any]:
        ...


def parse_letter(text: str, n_options: int) -> Optional[int]:
    """Option index from a text answer like 'B', 'B.', '(B) cat' -- for models without logprobs."""
    m = re.search(r"\b([A-E])\b", (text or "").strip().upper()[:12])
    if m and LETTERS.index(m.group(1)) < n_options:
        return LETTERS.index(m.group(1))
    return None


def one_hot(i: Optional[int], n: int) -> Optional[List[float]]:
    if i is None:
        return None
    return [1.0 if j == i else 0.0 for j in range(n)]


def parse_yes_no(text: str) -> Optional[List[float]]:
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return [1.0, 0.0]
    if t.startswith("no"):
        return [0.0, 1.0]
    return None
