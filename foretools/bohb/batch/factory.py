from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BatchSelector
from .greedy_diversity import GreedyDiversitySelector
from .local_penalization import LocalPenalizationSelector
from .thompson import ThompsonSamplingSelector

try:  # optional
    from .qnei import qNoisyEISelector
except Exception:
    qNoisyEISelector = None

# Strategy registry: maps strategy names to (factory, requires_distance_fn)
_REGISTRY = {
    "diversity": (GreedyDiversitySelector, True),
    "greedy_diversity": (GreedyDiversitySelector, True),
    "lp": (LocalPenalizationSelector, True),
    "local_penalization": (LocalPenalizationSelector, True),
    "ts": (ThompsonSamplingSelector, False),
    "thompson": (ThompsonSamplingSelector, False),
}

if qNoisyEISelector is not None:
    _REGISTRY["qnei"] = (qNoisyEISelector, False)
    _REGISTRY["qnei_batch"] = (qNoisyEISelector, False)
    _REGISTRY["qnei_greedy"] = (qNoisyEISelector, False)


def build_batch_selector(
    strategy: str,
    *,
    distance_fn: Callable[[dict[str, Any], dict[str, Any]], float] | None = None,
    penalization_power: float = 2.0,
    **extra_params: Any,
) -> BatchSelector | qNoisyEISelector:
    """
    Build a batch selector by name.

    Supported strategies:
    - "diversity" / "greedy_diversity": Greedy selection with diversity bonus
    - "lp" / "local_penalization": Local penalization by distance
    - "ts" / "thompson": Thompson sampling with uncertainty
    - "qnei" / "qnei_batch" / "qnei_greedy": qNEI greedy batch selection
      (requires sklearn.gaussian_process)

    Args:
        strategy: Strategy name (case-insensitive).
        distance_fn: Function for distance-based strategies.
        penalization_power: Power for LP strategy (default: 2.0).
        extra_params: Additional parameters passed to the selector.
            For Thompson: alpha, n_samples, uncertainty_fn
            For qNEI: n_fantasies, noise_level, exploration_weight

    Returns:
        Batch selector instance.
    """
    key = str(strategy or "diversity").lower().replace("-", "_").replace(" ", "_")

    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown batch strategy: '{strategy}'. "
            f"Available: {available}"
        )

    cls, needs_distance = _REGISTRY[key]

    # Build distance_fn if required but not provided
    if needs_distance and distance_fn is None:
        raise ValueError(
            f"distance_fn is required for '{strategy}' batch strategy"
        )

    if key in {"ts", "thompson"}:
        return ThompsonSamplingSelector(
            uncertainty_fn=extra_params.get("uncertainty_fn"),
            alpha=extra_params.get("alpha", 1.0),
            n_samples=extra_params.get("n_samples", 10),
        )

    if key in {"lp", "local_penalization"}:
        return LocalPenalizationSelector(distance_fn, penalization_power)

    if key in {"diversity", "greedy_diversity"}:
        return GreedyDiversitySelector(distance_fn)

    if key in {"qnei", "qnei_batch", "qnei_greedy"}:
        if qNoisyEISelector is None:
            raise ImportError(
                "qNEI requires sklearn.gaussian_process. "
                "Install scikit-learn to use this strategy."
            )
        return qNoisyEISelector(
            n_fantasies=extra_params.get("n_fantasies", 16),
            noise_level=extra_params.get("noise_level", 1e-4),
            constraint_penalty=extra_params.get("constraint_penalty", 10.0),
            exploration_weight=extra_params.get("exploration_weight", 0.0),
        )

    # Should not reach here, but be safe
    if needs_distance:
        return cls(distance_fn, **extra_params)
    return cls(**extra_params)
