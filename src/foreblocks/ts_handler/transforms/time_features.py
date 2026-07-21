"""foreblocks.ts_handler.time_features.

Time feature generation and inference for time-series preprocessing.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foreblocks.ts_handler.utils import _cyclical_encode


def _infer_timestamp_frequency(timestamps: np.ndarray) -> str:
    ts = pd.to_datetime(timestamps)
    if len(ts) < 2:
        return "h"

    deltas = pd.Series(ts).diff().dropna().dt.total_seconds().to_numpy()
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return "h"

    median_seconds = float(np.median(np.abs(deltas)))
    if median_seconds <= 3 * 3600:
        return "h"
    if median_seconds <= 2 * 86400:
        return "d"
    if median_seconds <= 14 * 86400:
        return "w"
    return "m"


def _generate_time_features(
    timestamps: np.ndarray, freq: str = "auto", time_feature_mode: str = "cyclical"
) -> np.ndarray:
    ts = pd.to_datetime(timestamps)
    resolved_freq = (
        _infer_timestamp_frequency(ts)
        if (freq or "").lower() == "auto"
        else freq.lower()
    )
    mode = (time_feature_mode or "cyclical").lower()

    if mode == "legacy":
        return np.column_stack(
            [
                ts.month.to_numpy(dtype=np.float32) / 12.0,
                ts.day.to_numpy(dtype=np.float32) / 31.0,
                ts.weekday.to_numpy(dtype=np.float32) / 6.0,
                (
                    ts.hour.to_numpy(dtype=np.float32) / 23.0
                    if resolved_freq == "h"
                    else np.zeros(len(ts), dtype=np.float32)
                ),
            ]
        ).astype(np.float32, copy=False)

    features = [
        _cyclical_encode(ts.month.to_numpy(dtype=np.float32) - 1.0, 12.0),
        _cyclical_encode(ts.day.to_numpy(dtype=np.float32) - 1.0, 31.0),
        _cyclical_encode(ts.weekday.to_numpy(dtype=np.float32), 7.0),
    ]
    if resolved_freq == "h":
        features.append(_cyclical_encode(ts.hour.to_numpy(dtype=np.float32), 24.0))
    else:
        features.append(np.zeros((len(ts), 2), dtype=np.float32))

    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _maybe_make_time_features(
    time_stamps: np.ndarray | None,
    generate_time_features: bool,
    infer_timestamp_frequency: callable,
    generate_time_features_fn: callable,
    T: int,
) -> np.ndarray | None:
    if time_stamps is None or not generate_time_features:
        return None
    tf = generate_time_features_fn(
        time_stamps, freq=infer_timestamp_frequency(time_stamps)
    )
    if tf.shape[0] != T:
        raise ValueError(f"time_stamps length {len(time_stamps)} != T {T}")
    return tf
