from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import BatchSelector
from .greedy_diversity import GreedyDiversitySelector
from .local_penalization import LocalPenalizationSelector
from .thompson import ThompsonSamplingSelector


def build_batch_selector(
    strategy: str,
    *,
    distance_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
    penalization_power: float = 2.0,
) -> BatchSelector:
    key = str(strategy or "diversity").lower()
    if key == "ts":
        return ThompsonSamplingSelector()
    if key in {"local_penalization", "lp"}:
        if distance_fn is None:
            raise ValueError("distance_fn is required for local_penalization batch strategy")
        return LocalPenalizationSelector(distance_fn, penalization_power)
    if distance_fn is None:
        raise ValueError("distance_fn is required for diversity batch strategy")
    return GreedyDiversitySelector(distance_fn)

