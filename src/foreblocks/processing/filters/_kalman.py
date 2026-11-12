"""NumPy Kalman algorithms shared by the filtering adapters.

The scalar local-level model and two-state local-linear-trend model retain
separate equations, initial conditions, and noise estimators. Adapters own
missing-data policies, iteration, parameter defaults, and output containers.
"""

from __future__ import annotations

import numpy as np


def _estimate_noise_variance(col: np.ndarray) -> tuple[float, float]:
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


def _kalman_filter_1d(
    obs: np.ndarray, r: float, q: float
) -> tuple[np.ndarray, np.ndarray]:
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


def _kalman_smoother_1d(
    filtered_states: np.ndarray, filtered_covariances: np.ndarray, r: float, q: float
) -> np.ndarray:
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
        smoothed_states[t] = filtered_states[t] + k_s * (
            smoothed_states[t + 1] - filtered_states[t]
        )
        smoothed_covariances[t] = filtered_covariances[t] + k_s * (
            smoothed_covariances[t + 1] - filtered_covariances[t + 1]
        )

    return smoothed_states


def _estimate_noise_variances(x: np.ndarray) -> tuple[float, float]:
    d2 = np.diff(np.diff(x))
    q = float(np.var(d2) / 4.0) + 1e-8
    r = float(np.var(x - np.convolve(x, np.ones(5) / 5, mode="same"))) + 1e-8
    return q, r


def _local_linear_trend_smoother(y: np.ndarray, q: float, r: float) -> np.ndarray:
    """Run the two-state local-linear-trend Kalman filter and RTS smoother."""
    N = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([q, q * 0.1])
    R = np.array([[r]])

    # --- Forward Kalman filter ---
    x_filt = np.zeros((N, 2))
    P_filt = np.zeros((N, 2, 2))
    x_pred = np.array([y[0], 0.0])
    P_pred = np.eye(2) * r * 10

    for t in range(N):
        innov = y[t] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K_gain = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + (K_gain @ innov).ravel()
        P_upd = (np.eye(2) - K_gain @ H) @ P_pred
        x_filt[t] = x_upd
        P_filt[t] = P_upd
        if t < N - 1:
            x_pred = F @ x_upd
            P_pred = F @ P_upd @ F.T + Q

    # --- Backward RTS smoother ---
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(N - 2, -1, -1):
        P_pred_t = F @ P_filt[t] @ F.T + Q
        G = P_filt[t] @ F.T @ np.linalg.inv(P_pred_t)
        x_smooth[t] += G @ (x_smooth[t + 1] - F @ x_filt[t])
        P_smooth[t] += G @ (P_smooth[t + 1] - P_pred_t) @ G.T

    return x_smooth[:, 0]
