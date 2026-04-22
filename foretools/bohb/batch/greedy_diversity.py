from __future__ import annotations

from typing import Any
from collections.abc import Callable

import numpy as np

from .base import BatchSelector


class GreedyDiversitySelector(BatchSelector):
    def __init__(self, distance_fn: Callable[[dict[str, Any], dict[str, Any]], float]):
        self.distance_fn = distance_fn

    def select(
        self, candidates: list[dict[str, Any]], scores: np.ndarray, n: int
    ) -> list[int]:
        if n <= 0:
            return []
        sorted_idx = np.argsort(scores)[::-1]
        selected = [int(sorted_idx[0])]
        for i in range(1, len(sorted_idx)):
            cand_idx = int(sorted_idx[i])
            min_dist = float("inf")
            for sel in selected:
                dist = self.distance_fn(candidates[cand_idx], candidates[sel])
                min_dist = min(min_dist, dist)
            # Small diversity bonus.
            scores[cand_idx] += 0.08 * min_dist
            if len(selected) >= n:
                break
            selected.append(cand_idx)
        return selected[:n]

