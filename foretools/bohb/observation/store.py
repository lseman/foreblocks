from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


Observation = Tuple[Dict[str, Any], float, Optional[float]]


@dataclass
class ObservationStore:
    max_budget: Optional[float] = None
    split_budget_correction: float = 0.25

    def sort_for_split(
        self,
        observations: List[Observation],
        loss_scale: float,
        atpe_filter: Optional[Callable[[List[Observation]], List[Observation]]] = None,
    ) -> List[Observation]:
        if not observations:
            return []
        obs = observations
        if atpe_filter is not None:
            obs = atpe_filter(obs)

        def split_score(o: Observation) -> float:
            loss, b = float(o[1]), o[2]
            if (
                b is None
                or self.max_budget is None
                or self.max_budget <= 0
                or self.split_budget_correction <= 0
            ):
                return loss
            budget_frac = float(max(0.0, min(1.0, b / self.max_budget)))
            corr = self.split_budget_correction * float(loss_scale) * (1.0 - budget_frac)
            return float(loss + corr)

        return sorted(obs, key=split_score)

    def split_good_bad(
        self, sorted_observations: List[Observation], n_good: int
    ) -> Tuple[List[Observation], List[Observation]]:
        if not sorted_observations:
            return [], []
        n = len(sorted_observations)
        k = int(max(1, min(n_good, n)))
        good = sorted_observations[:k]
        bad = sorted_observations[k:] if k < n else []
        return good, bad
