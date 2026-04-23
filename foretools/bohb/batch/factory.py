from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BatchSelector
from .greedy_diversity import GreedyDiversitySelector
from .local_penalization import LocalPenalizationSelector
from .thompson import ThompsonSamplingSelector


def build_batch_selector(
    strategy: str,
    *,
    distance_fn: Callable[[dict[str, Any], dict[str, Any]], float] | None = None,
    penalization_power: float = 2.0,
) -> BatchSelector:
    key = str(strategy or "diversity").lower()
    if key == "ts":
        return ThompsonSamplingSelector()
    if key in {"local_penalization", "lp"}:
        if distance_fn is None:
            raise ValueError(
                "distance_fn is required for local_penalization batch strategy"
            )
        return LocalPenalizationSelector(distance_fn, penalization_power)
    if distance_fn is None:
        raise ValueError("distance_fn is required for diversity batch strategy")
    return GreedyDiversitySelector(distance_fn)
