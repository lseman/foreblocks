"""foreblocks.processing.filters.kalman.

Kalman filter and smoother for time-series denoising (numpy implementation).

"""

from __future__ import annotations

import numpy as np

from foreblocks.processing.filters._kalman import (
    _estimate_noise_variance as _estimate_noise_variance,
    _kalman_filter_1d as _kalman_filter_1d,
    _kalman_smoother_1d as _kalman_smoother_1d,
)
from foreblocks.processing.filters.utils import _as_2d


def kalman_filter(
    data: np.ndarray,
    *,
    n_iter: int = 5,
    min_points: int = 10,
    em_on_valid_only: bool = True,
) -> np.ndarray:
    x = _as_2d(data)
    T, F = x.shape
    out = x.copy()

    for i in range(F):
        col = x[:, i]
        mask = ~np.isnan(col)
        if mask.sum() < min_points:
            continue

        obs = col[mask]

        # Iterative noise estimation and smoothing
        for _ in range(n_iter):
            r, q = _estimate_noise_variance(obs)

            # Forward filter
            filtered_states, filtered_covariances = _kalman_filter_1d(obs, r, q)

            # Backward smoother
            smoothed = _kalman_smoother_1d(filtered_states, filtered_covariances, r, q)

            # Update obs with smoothed values for next iteration
            if em_on_valid_only:
                obs = smoothed.copy()
            else:
                # Full length smoothing
                pass

        out[mask, i] = smoothed.reshape(-1)

    return out
