from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Optional


Observation = tuple[dict[str, Any], float, Optional[float]]


@dataclass
class ObservationStore:
    max_budget: float | None = None
    split_budget_correction: float = 0.25

    def sort_for_split(
        self,
        observations: list[Observation],
        loss_scale: float,
        atpe_filter: Callable[[list[Observation]], list[Observation]] | None = None,
    ) -> list[Observation]:
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
            corr = (
                self.split_budget_correction * float(loss_scale) * (1.0 - budget_frac)
            )
            return float(loss + corr)

        return sorted(obs, key=split_score)

    def split_good_bad(
        self, sorted_observations: list[Observation], n_good: int
    ) -> tuple[list[Observation], list[Observation]]:
        if not sorted_observations:
            return [], []
        n = len(sorted_observations)
        k = int(max(1, min(n_good, n)))
        good = sorted_observations[:k]
        bad = sorted_observations[k:] if k < n else []
        return good, bad
