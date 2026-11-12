"""Shared LOWESS execution; adapters own masking and output containers."""

from __future__ import annotations

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


def _lowess_smooth(
    values: np.ndarray,
    positions: np.ndarray,
    *,
    frac: float,
    it: int,
    delta: float = 0.0,
) -> np.ndarray:
    return lowess(values, positions, frac=frac, it=it, delta=delta, return_sorted=False)
