from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HyperbandScheduler:
    min_budget: float
    max_budget: float
    eta: int

    def __post_init__(self) -> None:
        self.min_budget = float(self.min_budget)
        self.max_budget = float(self.max_budget)
        self.eta = int(self.eta)
        if self.min_budget <= 0 or self.max_budget <= 0:
            raise ValueError("Hyperband budgets must be positive")
        if self.eta < 2:
            raise ValueError("Hyperband eta must be >= 2")

    @property
    def s_max(self) -> int:
        return int(math.log(self.max_budget / self.min_budget, self.eta))

    @property
    def B(self) -> float:
        return float((self.s_max + 1) * self.max_budget)

    def brackets(self) -> list[tuple[int, int, float]]:
        return [self.bracket(s) for s in reversed(range(self.s_max + 1))]

    def bracket(self, s: int) -> tuple[int, int, float]:
        n = int(math.ceil((self.B / self.max_budget) * (self.eta**s) / (s + 1)))
        r = float(self.max_budget * (self.eta ** (-s)))
        return s, n, r

    def successive_halving(
        self, s: int, n: int, r: float
    ) -> list[tuple[int, int, float]]:
        return [
            (i, max(1, int(n * (self.eta ** (-i)))), float(r * (self.eta**i)))
            for i in range(s + 1)
        ]
