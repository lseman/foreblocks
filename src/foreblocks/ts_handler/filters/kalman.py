"""foreblocks.ts_handler.filters.kalman.

Kalman filter and smoother for time-series denoising (numpy implementation).

"""

from __future__ import annotations

import numpy as np

from foreblocks.ts_handler.filters.utils import _as_2d, _nan_interp_1d


def _estimate_noise_variance(col: np.ndarray) -> tuple[float, float]:
    """Estimate measurement noise (R) and process noise (Q) from data."""
    valid = col[~np.isnan(col)]
    if len(valid) < 3:
        return 1.0, 0.01

    # R: measurement noise from median absolute deviation of differences
    diffs = np.diff(valid)
    if len(diffs) < 2:
        r = float(np.var(valid)) + 1e-8
        q = 0.01
    else:
        mad_diffs = np.median(np.abs(diffs - np.median(diffs)))
        r = max(1e-8, (mad_diffs / 0.6745) ** 2)
        # Q: process noise, small for smooth signals
        q = max(1e-6, r * 0.01)

    return float(r), float(q)


def _kalman_filter_1d(obs: np.ndarray, r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward Kalman filter pass.
    Returns: filtered_states, filtered_covariances
    """
    n = len(obs)
    if n == 0:
        return np.array([]), np.array([])

    # State: x (scalar)
    # Initial state
    x_est = obs[0]
    p_est = r  # Initial covariance

    filtered_states = np.zeros(n)
    filtered_covariances = np.zeros(n)

    for t in range(n):
        # Prediction
        x_pred = x_est
        p_pred = p_est + q

        # Update
        if t < len(obs) and not np.isnan(obs[t]):
            y = obs[t]
            # Kalman gain
            k = p_pred / (p_pred + r)
            x_est = x_pred + k * (y - x_pred)
            p_est = (1.0 - k) * p_pred
        else:
            # If NaN, just use prediction
            x_est = x_pred
            p_est = p_pred

        filtered_states[t] = x_est
        filtered_covariances[t] = p_est

    return filtered_states, filtered_covariances


def _kalman_smoother_1d(filtered_states: np.ndarray, filtered_covariances: np.ndarray, r: float, q: float) -> np.ndarray:
    """
    RTS (Rauch-Tung-Striebel) smoother backward pass.
    """
    n = len(filtered_states)
    if n == 0:
        return np.array([])

    smoothed_states = np.zeros(n)
    smoothed_covariances = np.zeros(n)

    # Backward pass
    smoothed_states[-1] = filtered_states[-1]
    smoothed_covariances[-1] = filtered_covariances[-1]

    for t in range(n - 2, -1, -1):
        # Smoother gain
        p_pred_next = filtered_covariances[t + 1] + q
        k_s = filtered_covariances[t] / p_pred_next

        # Smoothed state and covariance
        smoothed_states[t] = filtered_states[t] + k_s * (smoothed_states[t + 1] - filtered_states[t])
        smoothed_covariances[t] = filtered_covariances[t] + k_s * (smoothed_covariances[t + 1] - filtered_covariances[t + 1])

    return smoothed_states


def kalman_filter(
    data: np.ndarray,
    *,
    n_iter: int = 5,
    min_points: int = 10,
    em_on_valid_only: bool = True,
) -> np.ndarray:
    """
    Apply a per-feature Kalman smoother (numpy implementation).

    Improvements:
      - preserves NaNs
      - estimates noise parameters from data
      - configurable iterations and min points
    """
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
