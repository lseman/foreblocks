from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from .base import BatchSelector


class LocalPenalizationSelector(BatchSelector):
    def __init__(
        self,
        distance_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
        penalization_power: float,
    ):
        self.distance_fn = distance_fn
        self.power = float(penalization_power)

    def select(
        self, candidates: List[Dict[str, Any]], scores: np.ndarray, n: int
    ) -> List[int]:
        if n <= 0:
            return []
        base_scores = scores.copy()
        sorted_idx = np.argsort(base_scores)[::-1]
        selected = [int(sorted_idx[0])]
        for _ in range(1, len(sorted_idx)):
            best_idx = None
            best_score = -float("inf")
            for cand_idx in sorted_idx:
                if cand_idx in selected:
                    continue
                penalty = 1.0
                for sel in selected:
                    dist = self.distance_fn(candidates[cand_idx], candidates[sel])
                    penalty *= 1.0 / (1.0 + dist**self.power)
                score = float(base_scores[cand_idx]) * float(penalty)
                if score > best_score:
                    best_score = score
                    best_idx = int(cand_idx)
            if best_idx is None:
                break
            selected.append(best_idx)
            if len(selected) >= n:
                break
        return selected[:n]

