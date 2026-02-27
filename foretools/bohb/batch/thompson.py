from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import BatchSelector


class ThompsonSamplingSelector(BatchSelector):
    def select(
        self, candidates: List[Dict[str, Any]], scores: np.ndarray, n: int
    ) -> List[int]:
        if n <= 0:
            return []
        return list(np.argsort(scores)[-n:][::-1])

