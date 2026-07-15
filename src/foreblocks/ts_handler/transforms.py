"""foreblocks.ts_handler.transforms.

Transform and scaling stages for time-series preprocessing.

"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from foreblocks.ts_handler.utils import apply_log_transform, compute_basic_stats


class _TransformState:
    """Container for transform state (log flags, scaler, log offset, diff values, trend component)."""

    log_transform_flags: list[bool] | None = None
    scaler: any | None = None
    log_offset: np.ndarray | None = None
    diff_values: np.ndarray | None = None
    trend_component: np.ndarray | None = None


def _should_log_transform(sk: float, ku: float) -> bool:
    return (abs(sk) > 1.0) or (ku > 5.0)


def _centered(data: np.ndarray, means: np.ndarray) -> np.ndarray:
    return data - means[np.newaxis, :]


def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 8:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    # Calculate base threshold scale relative to the raw absolute deviations.
    # Modified Z-score uses 0.6745. The conversion from Z-threshold to raw value is therefore / 0.6745
    return (mad / 0.6745) + 1e-12


def _ensure_log_flags(state: _TransformState, data: np.ndarray) -> None:
    if state.log_transform_flags is not None:
        return
    _, _, _, skews, kurts = compute_basic_stats(data)
    flags = [
        _should_log_transform(float(sk), float(ku))
        for sk, ku in zip(skews, kurts)
    ]
    state.log_transform_flags = flags


def _build_scaler(method: str | None, n_samples: int) -> any | None:
    scaling = (method or "standard").lower()
    if scaling in {"none", "off", "false", "log_only"}:
        return None
    if scaling == "robust":
        return RobustScaler(quantile_range=(5.0, 95.0))
    if scaling == "quantile":
        return QuantileTransformer(
            output_distribution="normal",
            random_state=42,
            n_quantiles=min(1000, max(10, n_samples)),
        )
    if scaling == "box_cox":
        return PowerTransformer(method="yeo-johnson")
    return StandardScaler()


def _apply_scaling_stage(
    handler: any,
    data: np.ndarray,
    mode: str,
    normalize: bool,
    scaling_method: str,
    vprint: callable,
) -> np.ndarray:
    if not normalize:
        return data

    vprint(f"Applying normalization ({scaling_method})")
    if mode == "fit":
        handler.scaler = _build_scaler(scaling_method, data.shape[0])
        return data if handler.scaler is None else handler.scaler.fit_transform(data)

    if handler.scaler is None:
        if (scaling_method or "").lower() in {"none", "off", "false", "log_only"}:
            return data
        raise RuntimeError("scaler is not fitted.")
    return handler.scaler.transform(data)


def _apply_log_stage(
    handler: any,
    data: np.ndarray,
    mode: str,
    log_transform: bool,
    log_transform_flags: list[bool] | None,
    vprint: callable,
) -> np.ndarray:
    _ensure_log_flags(handler, data)
    flags = log_transform_flags or handler.log_transform_flags or []
    if not any(flags):
        return data

    vprint("Applying selective log transformation")
    if mode == "fit":
        transformed, handler.log_offset = apply_log_transform(data, flags)
        return transformed

    if handler.log_offset is None:
        raise RuntimeError("log_offset is not fitted.")
    transformed, _ = apply_log_transform(data, flags, offsets=handler.log_offset)
    return transformed
