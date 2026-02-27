from __future__ import annotations

from typing import Any, Callable

from .base import GammaStrategy
from .default import DefaultGammaStrategy


def build_gamma_strategy(
    gamma: Any,
    gamma_strategy: str,
    adaptive_gamma_fn: Callable[[int], int],
) -> GammaStrategy:
    return DefaultGammaStrategy(gamma, gamma_strategy, adaptive_gamma_fn)

