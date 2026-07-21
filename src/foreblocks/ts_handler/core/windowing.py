"""foreblocks.ts_handler.windowing.

Sequence windowing and creation for time-series preprocessing.

"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _create_sequences(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    feats: list[int] | None = None,
    time_feats: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"data must be 2D [T,D], got {x.shape}")

    T, D = x.shape
    feats_idx = list(range(D)) if feats is None else list(feats)

    max_idx = T - window_size - horizon + 1
    if max_idx <= 0:
        raise ValueError(
            f"Not enough data for window_size={window_size} and horizon={horizon} (len={T})."
        )

    try:
        X_all = sliding_window_view(x, window_shape=window_size, axis=0)
        X = np.ascontiguousarray(np.transpose(X_all[:max_idx, :, :], (0, 2, 1)))

        y_src = x[window_size : window_size + max_idx + horizon - 1]
        y_all = sliding_window_view(y_src, window_shape=horizon, axis=0)
        y = np.ascontiguousarray(np.transpose(y_all[:, feats_idx, :], (0, 2, 1)))

        tf = None
        if time_feats is not None:
            tf2 = np.asarray(time_feats, dtype=float)
            if tf2.ndim != 2 or tf2.shape[0] != T:
                raise ValueError(f"time_feats must be [T,F], got {tf2.shape}")
            tf_all = sliding_window_view(tf2, window_shape=window_size, axis=0)
            tf = np.ascontiguousarray(np.transpose(tf_all[:max_idx, :, :], (0, 2, 1)))

        return (
            np.asarray(X),
            np.asarray(y),
            (np.asarray(tf) if tf is not None else None),
        )

    except Exception:
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        tf_list: list[np.ndarray] = []

        for i in range(max_idx):
            X_list.append(x[i : i + window_size])
            y_list.append(x[i + window_size : i + window_size + horizon][:, feats_idx])
            if time_feats is not None:
                tf_list.append(time_feats[i : i + window_size])

        Xn = np.asarray(X_list)
        yn = np.asarray(y_list)
        tf = np.asarray(tf_list) if time_feats is not None else None
        return Xn, yn, tf
