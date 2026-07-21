"""foreblocks.ts_handler.auto_filter.filters.kalman_rts.

Kalman RTS smoother (1-D local-linear-trend model, pure NumPy).

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foreblocks.ts_handler.auto_filter.filters.utils import _as_series
from foreblocks.ts_handler.auto_filter.registry import register_filter


def _estimate_noise_variances(x: np.ndarray) -> tuple[float, float]:
    d2 = np.diff(np.diff(x))
    q = float(np.var(d2) / 4.0) + 1e-8
    r = float(np.var(x - np.convolve(x, np.ones(5) / 5, mode="same"))) + 1e-8
    return q, r


@register_filter("Kalman RTS")
def kalman_rts_smoother(
    ts: pd.Series,
    q: float | None = None,
    r: float | None = None,
) -> pd.Series:
    y = ts.values.astype(float)
    N = len(y)
    if N < 4:
        return ts.copy()

    q_auto, r_auto = _estimate_noise_variances(y)
    q = float(q) if q is not None else q_auto
    r = float(r) if r is not None else r_auto

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

    return _as_series(x_smooth[:, 0], ts.index, name="kalman_rts")
