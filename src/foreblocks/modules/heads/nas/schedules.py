"""Temperature schedules for differentiable head architecture search."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CosineTemperatureSchedule:
    initial: float = 1.0
    final: float = 0.1
    steps: int = 10_000

    def __call__(self, step: int) -> float:
        progress = min(1.0, max(0.0, step / max(1, self.steps)))
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final + (self.initial - self.final) * weight


__all__ = ["CosineTemperatureSchedule"]
