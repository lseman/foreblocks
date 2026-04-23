from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any


@dataclass(frozen=True)
class PruningConfig:
    final_min_history: int = 8
    final_budget_gate_multiplier: float = 2.0
    final_quantile_base: float = 0.75
    final_quantile_growth: float = 0.20
    final_quantile_min: float = 0.70
    final_quantile_max: float = 0.92
    final_prob_base_conservative: float = 0.40
    final_prob_base_balanced: float = 0.65
    final_prob_base_aggressive: float = 0.85
    final_prob_growth: float = 0.25
    final_relative_worst_offset: float = 1.0
    final_prob_cap: float = 0.95

    step_min_progress: float = 0.15
    step_min_history: int = 8
    step_exact_match_min: int = 6
    step_progress_tolerance: float = 0.15
    step_budget_ratio_limit: float | None = None
    step_quantile_conservative: float = 0.99
    step_quantile_balanced: float = 0.97
    step_quantile_aggressive: float = 0.95
    step_sigma_conservative: float = 3.5
    step_sigma_balanced: float = 3.0
    step_sigma_aggressive: float = 2.5
    step_quantile_progress_slope: float = 0.10
    step_quantile_floor: float = 0.80
    step_sigma_progress_slope: float = 1.0
    step_sigma_floor: float = 1.5

    def copy_with(self, **overrides: Any) -> PruningConfig:
        return replace(self, **overrides)
