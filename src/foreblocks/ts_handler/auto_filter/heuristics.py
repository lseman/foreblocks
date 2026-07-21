"""foreblocks.ts_handler.auto_filter.heuristics.

Heuristics for suggesting ScoringWeights from signal characteristics.

"""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from scipy import signal

from foreblocks.ts_handler.auto_filter.filters import (
    _autocorr,
)
from foreblocks.ts_handler.auto_filter.filters.utils import _valid_odd_window
from foreblocks.ts_handler.auto_filter.metrics import ScoringWeights


def _safe_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return _clip01(1.0 - ss_res / max(ss_tot, 1e-12))


def _softmax_simplex(
    logits: np.ndarray,
    *,
    floor: float = 0.05,
    cap: float = 0.60,
    temperature: float = 1.0,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - float(np.max(logits))
    raw = np.exp(logits)
    raw /= max(float(raw.sum()), 1e-12)

    n = raw.size
    floor = float(np.clip(floor, 0.0, 1.0 / max(n, 1) - 1e-9))
    weights = floor + (1.0 - floor * n) * raw

    for _ in range(3):
        over = weights > cap
        if not np.any(over):
            break
        excess = float(np.sum(weights[over] - cap))
        weights[over] = cap
        under = ~over
        if np.any(under):
            weights[under] += (
                excess * weights[under] / max(float(weights[under].sum()), 1e-12)
            )
    weights /= max(float(weights.sum()), 1e-12)
    return weights


def _enforce_smoothing_share(
    raw: np.ndarray,
    *,
    smoothing_pressure: float,
    outliers: float,
    roughness: float,
    trend: float,
    periodicity: float,
) -> np.ndarray:
    weights = np.asarray(raw, dtype=float).copy()
    smoothing_bias = _clip01(
        0.55 * smoothing_pressure
        + 0.15 * outliers
        + 0.15 * roughness
        + 0.10 * (1.0 - trend)
        - 0.10 * periodicity
    )

    if smoothing_bias < 0.50:
        return weights / max(float(weights.sum()), 1e-12)

    target_smoothing_share = 0.46 + 0.12 * _clip01((smoothing_bias - 0.50) / 0.50)
    current_smoothing_share = float(weights[2] + weights[3])
    if current_smoothing_share >= target_smoothing_share:
        return weights / max(float(weights.sum()), 1e-12)

    min_fidelity = 0.08
    min_derivative = 0.12 if trend < 0.20 else 0.16
    min_spectral = 0.04
    min_gcv = 0.05
    min_iid = 0.04
    available_from_fidelity = max(float(weights[0]) - min_fidelity, 0.0)
    available_from_gcv = max(float(weights[1]) - min_gcv, 0.0)
    available_from_spectral = max(float(weights[4]) - min_spectral, 0.0)
    available_from_iid = max(float(weights[5]) - min_iid, 0.0)
    available_from_derivative = max(float(weights[6]) - min_derivative, 0.0)
    available = (
        available_from_fidelity
        + available_from_gcv
        + available_from_spectral
        + available_from_iid
        + available_from_derivative
    )
    needed = target_smoothing_share - current_smoothing_share
    transfer = min(needed, available)

    if transfer <= 0.0:
        return weights / max(float(weights.sum()), 1e-12)

    if available > 0.0:
        weights[0] -= transfer * available_from_fidelity / available
        weights[1] -= transfer * available_from_gcv / available
        weights[4] -= transfer * available_from_spectral / available
        weights[5] -= transfer * available_from_iid / available
        weights[6] -= transfer * available_from_derivative / available
    weights[2] += 0.45 * transfer
    weights[3] += 0.55 * transfer
    weights /= max(float(weights.sum()), 1e-12)
    return weights


def _strength_label(value: float) -> str:
    if value >= 0.70:
        return "high"
    if value >= 0.35:
        return "moderate"
    return "low"


def _round_mapping(values: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {key: round(float(value), digits) for key, value in values.items()}


def _build_weight_explanation(
    weights: ScoringWeights,
    diagnostics: dict[str, float],
    derived: dict[str, float],
    logits: np.ndarray,
) -> dict[str, Any]:
    weight_map = {
        "fidelity_mse": weights.fidelity_mse,
        "gcv": weights.gcv,
        "roughness": weights.roughness,
        "residual_autocorr": weights.residual_autocorr,
        "spectral_distance": weights.spectral_distance,
        "residual_iid": weights.residual_iid,
        "derivative_corr": weights.derivative_corr,
    }
    logit_map = dict(zip(weight_map.keys(), logits, strict=True))

    reasons: list[str] = []
    if derived["smoothing_pressure"] >= 0.35:
        reasons.append(
            "Smoothing pressure is "
            f"{_strength_label(derived['smoothing_pressure'])}; this lowers "
            "fidelity pressure and raises roughness/residual-whiteness pressure."
        )
    else:
        reasons.append(
            "Smoothing pressure is low; the raw series appears informative enough "
            "to keep fidelity relatively important."
        )

    if diagnostics["periodicity"] >= 0.35 or diagnostics["memory"] >= 0.35:
        reasons.append(
            "Periodic or autocorrelated structure is visible, so residual "
            "autocorrelation and derivative preservation get more weight."
        )

    if diagnostics["trend"] >= 0.35:
        reasons.append(
            "Trend strength is "
            f"{_strength_label(diagnostics['trend'])}; derivative correlation "
            "is favored to preserve the signal shape."
        )

    if diagnostics["jumps"] >= 0.05:
        reasons.append(
            "Potential level shifts are present, so the heuristic avoids putting "
            "too much emphasis on generic roughness minimization."
        )

    if diagnostics["outliers"] >= 0.10:
        reasons.append(
            "Outlier pressure is noticeable, which shifts weight away from exact "
            "fidelity and toward smoothing criteria."
        )

    if not reasons:
        reasons.append(
            "No dominant structure or noise pressure was detected, so the weights "
            "stay near the balanced default prior."
        )

    rationale = {
        "fidelity_mse": [
            "increases when estimated noise is low",
            "decreases when smoothing pressure or outlier pressure is high",
        ],
        "gcv": [
            "increases when there is genuine noise to remove so the bias/variance "
            "tradeoff is informative",
            "decreases for spike-heavy or jump-dominated inputs where the noise "
            "model and the df estimate are less reliable",
        ],
        "roughness": [
            "increases for high-frequency noise, outliers, or rough local jitter",
            "decreases when jumps or periodic structure should be preserved",
        ],
        "residual_autocorr": [
            "Ljung-Box Q on the residual — increases when memory or periodicity "
            "suggest residual whiteness matters",
            "slightly decreases when simple trend dominates",
        ],
        "spectral_distance": [
            "increases when periodicity/memory make PSD-shape preservation important",
            "decreases for spike-heavy or jump-dominated inputs (PSD is noisy there)",
        ],
        "residual_iid": [
            "increases when there is genuine noise to remove, so an iid residual is meaningful",
            "decreases for spike/jump-heavy inputs where the residual is not expected to be Gaussian",
        ],
        "derivative_corr": [
            "increases for trend, periodicity, memory, and level-shift structure",
            "decreases when the series looks mostly noisy",
        ],
    }

    return {
        "weights": _round_mapping(weight_map),
        "diagnostics": _round_mapping(diagnostics),
        "derived": _round_mapping(derived),
        "logits": _round_mapping(logit_map),
        "reasons": reasons,
        "rationale": rationale,
    }


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mad = float(np.median(np.abs(x - np.median(x))))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(x))
    return max(scale, 1e-12)


def _clean_signal_values(ts: pd.Series) -> np.ndarray:
    values = pd.Series(ts, copy=False).astype(float).replace([np.inf, -np.inf], np.nan)
    if values.notna().any():
        values = values.interpolate(limit_direction="both").ffill().bfill()
    else:
        values = values.fillna(0.0)
    return values.to_numpy(dtype=float)


def _signal_characteristics(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 4 or _robust_scale(x) <= 1e-12:
        return {
            "noise": 0.0,
            "periodicity": 0.0,
            "trend": 0.0,
            "high_freq": 0.0,
            "outliers": 0.0,
            "jumps": 0.0,
            "memory": 0.0,
            "roughness": 0.0,
        }

    scale = _robust_scale(x)
    centered = x - np.median(x)

    diff1 = np.diff(centered)
    diff2 = np.diff(centered, n=2)
    noise_ratio_diff = _safe_ratio(_robust_scale(diff2), np.sqrt(6.0) * scale)
    sg_window = _valid_odd_window(n, preferred=max(7, n // 12), minimum=5)
    if sg_window >= 5:
        polyorder = min(2, sg_window - 1)
        try:
            baseline = signal.savgol_filter(
                centered, sg_window, polyorder, mode="interp"
            )
        except Exception:
            baseline = (
                pd.Series(centered)
                .rolling(window=sg_window, center=True, min_periods=1)
                .median()
                .to_numpy()
            )
    else:
        baseline = np.full_like(centered, np.median(centered))
    noise_ratio_hp = _safe_ratio(_robust_scale(centered - baseline), scale)
    noise = _clip01(
        0.55 * noise_ratio_diff / (noise_ratio_diff + 0.55)
        + 0.45 * noise_ratio_hp / (noise_ratio_hp + 0.70)
    )

    t = np.arange(n, dtype=float)
    if n > 1:
        t = (t - t.mean()) / max(float(t.std()), 1e-12)
    try:
        linear_fit = np.polyval(np.polyfit(t, x, 1), t)
        quad_fit = np.polyval(np.polyfit(t, x, min(2, n - 1)), t)
        trend_fit = (
            quad_fit if _safe_r2(x, quad_fit) > _safe_r2(x, linear_fit) else linear_fit
        )
        trend = max(_safe_r2(x, linear_fit), _safe_r2(x, quad_fit))
    except Exception:
        trend = 0.0
        trend_fit = np.full_like(x, x.mean())

    detrended = x - trend_fit
    detrended -= detrended.mean()
    if np.allclose(detrended, 0.0):
        periodicity = 0.0
        high_freq = 0.0
    else:
        freqs, power = signal.periodogram(
            detrended,
            scaling="spectrum",
            detrend=False,
        )
        freqs = freqs[1:]
        power = np.maximum(power[1:], 0.0)
        total_power = float(power.sum())
        if len(power) < 2 or total_power <= 1e-20:
            periodicity = 0.0
            high_freq = 0.0
        else:
            probs = power / total_power
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            entropy /= max(float(np.log(len(probs))), 1e-12)
            peak_share = float(probs.max())
            top_k = min(3, len(probs))
            top_share = float(np.partition(probs, -top_k)[-top_k:].sum())
            periodicity = _clip01(
                0.55 * (1.0 - entropy) + 0.30 * top_share + 0.15 * peak_share
            )
            high_freq = _clip01(float(power[freqs >= 0.25].sum()) / total_power)

    z = np.abs(centered) / scale
    outliers = _clip01(float(np.mean(z > 3.5)) * 8.0)

    diff_scale = _robust_scale(diff1) if diff1.size else 0.0
    if diff1.size and diff_scale > 1e-12:
        dz = np.abs(diff1 - np.median(diff1)) / diff_scale
        jumps = _clip01(float(np.mean(dz > 4.5)) * 10.0)
        roughness = _clip01(_safe_ratio(diff_scale, scale) / 1.35)
    else:
        jumps = 0.0
        roughness = 0.0

    max_lag = min(24, max(1, n // 5))
    ac_vals = [abs(_autocorr(centered, lag)) for lag in range(1, max_lag + 1)]
    memory = _clip01(float(np.nanmean(ac_vals)) if ac_vals else 0.0)

    return {
        "noise": noise,
        "periodicity": periodicity,
        "trend": trend,
        "high_freq": high_freq,
        "outliers": outliers,
        "jumps": jumps,
        "memory": memory,
        "roughness": roughness,
    }


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(den, 1e-12))


@overload
def suggest_weights(
    ts: pd.Series, *, explain: Literal[False] = False
) -> ScoringWeights: ...


@overload
def suggest_weights(
    ts: pd.Series, *, explain: Literal[True]
) -> tuple[ScoringWeights, dict[str, Any]]: ...


def suggest_weights(
    ts: pd.Series, *, explain: bool = False
) -> ScoringWeights | tuple[ScoringWeights, dict[str, Any]]:
    x = _clean_signal_values(ts)
    c = _signal_characteristics(x)
    noise = c["noise"]
    clean = 1.0 - noise
    periodicity = c["periodicity"]
    trend = c["trend"]
    high_freq = c["high_freq"]
    outliers = c["outliers"]
    jumps = c["jumps"]
    memory = c["memory"]
    roughness = c["roughness"]
    structure = _clip01(
        0.35 * trend + 0.35 * periodicity + 0.20 * memory + 0.10 * jumps
    )
    smoothing_pressure = _clip01(
        0.42 * noise + 0.24 * high_freq + 0.18 * outliers + 0.16 * roughness
    )
    shape_pressure = _clip01(
        0.35 * structure + 0.30 * trend + 0.20 * periodicity + 0.15 * jumps
    )

    logits = np.array(
        [
            0.45
            + 0.85 * clean
            + 0.30 * structure
            - 1.35 * smoothing_pressure
            - 0.55 * outliers
            - 0.30 * roughness,
            -0.10 + 1.00 * noise + 0.30 * roughness - 0.50 * outliers - 0.25 * jumps,
            -0.25
            + 1.25 * smoothing_pressure
            + 0.45 * high_freq
            + 0.45 * outliers
            + 0.35 * roughness
            - 0.15 * jumps
            - 0.15 * periodicity,
            0.30
            + 0.95 * noise
            + 0.75 * memory
            + 0.50 * periodicity
            + 0.35 * high_freq
            + 0.20 * smoothing_pressure
            - 0.10 * trend
            - 0.08 * jumps,
            -0.15
            + 1.20 * periodicity
            + 0.45 * memory
            + 0.20 * trend
            - 0.30 * outliers
            - 0.20 * jumps,
            -0.10
            + 0.85 * noise
            + 0.30 * high_freq
            + 0.20 * memory
            - 0.55 * outliers
            - 0.40 * jumps,
            0.30
            + 0.70 * shape_pressure
            + 0.50 * structure
            + 0.20 * clean
            + 0.22 * jumps
            - 0.40 * noise
            - 0.20 * roughness,
        ],
        dtype=float,
    )

    raw = _softmax_simplex(logits, floor=0.03, cap=0.40, temperature=1.35)
    raw = _enforce_smoothing_share(
        raw,
        smoothing_pressure=smoothing_pressure,
        outliers=outliers,
        roughness=roughness,
        trend=trend,
        periodicity=periodicity,
    )
    if structure >= 0.45 and smoothing_pressure < 0.25 and raw[6] <= raw[3]:
        transfer = min((raw[3] - raw[6]) + 0.02, max(raw[3] - 0.12, 0.0))
        if transfer > 0.0:
            raw[3] -= transfer
            raw[6] += transfer
            raw /= max(float(raw.sum()), 1e-12)
    weights = ScoringWeights(
        fidelity_mse=float(raw[0]),
        gcv=float(raw[1]),
        roughness=float(raw[2]),
        residual_autocorr=float(raw[3]),
        spectral_distance=float(raw[4]),
        residual_iid=float(raw[5]),
        derivative_corr=float(raw[6]),
    )
    if not explain:
        return weights

    explanation = _build_weight_explanation(
        weights=weights,
        diagnostics=c,
        derived={
            "clean": clean,
            "structure": structure,
            "smoothing_pressure": smoothing_pressure,
            "shape_pressure": shape_pressure,
        },
        logits=logits,
    )
    return weights, explanation
