from __future__ import annotations

import math
from typing import Any, Callable

from .base import GammaStrategy


class DefaultGammaStrategy(GammaStrategy):
    def __init__(
        self,
        gamma: Any,
        gamma_strategy: str,
        adaptive_gamma_fn: Callable[[int], int],
    ):
        self.gamma = gamma
        self.gamma_strategy = str(gamma_strategy).lower()
        self.adaptive_gamma_fn = adaptive_gamma_fn

    def n_good(self, n_obs: int) -> int:
        if callable(self.gamma):
            try:
                return int(self.gamma(n_obs))
            except Exception:
                return max(1, int(0.15 * n_obs))
        if self.gamma_strategy in {"adaptive", "watanabe2023"}:
            return int(self.adaptive_gamma_fn(n_obs))
        if self.gamma_strategy == "linear":
            # Watanabe-style lifecycle schedule:
            # start conservative, gradually widen the good set as data grows.
            gamma_frac = min(0.25, max(0.08, 0.08 + 0.0015 * n_obs))
            return max(1, int(gamma_frac * n_obs))
        if self.gamma_strategy == "sqrt":
            return max(1, int(math.sqrt(n_obs)))
        if self.gamma_strategy == "decay":
            initial = 0.25
            minimum = 0.10
            decay = 0.0008
            curr_gamma = max(minimum, initial - decay * n_obs)
            return max(1, int(curr_gamma * n_obs))
        return max(1, int(float(self.gamma) * n_obs))
