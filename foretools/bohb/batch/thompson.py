from __future__ import annotations

from typing import Any

import numpy as np

from .base import BatchSelector


class ThompsonSamplingSelector(BatchSelector):
    def select(
        self, candidates: list[dict[str, Any]], scores: np.ndarray, n: int
    ) -> list[int]:
        if n <= 0:
            return []
        return list(np.argsort(scores)[-n:][::-1])
