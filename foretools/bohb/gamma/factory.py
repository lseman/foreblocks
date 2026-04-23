from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import GammaStrategy
from .default import DefaultGammaStrategy


def build_gamma_strategy(
    gamma: Any,
    gamma_strategy: str,
    adaptive_gamma_fn: Callable[[int], int],
) -> GammaStrategy:
    return DefaultGammaStrategy(gamma, gamma_strategy, adaptive_gamma_fn)

