"""foreblocks.ts_handler.auto_filter.metrics.

Scoring metrics and functions for filter evaluation.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from foreblocks.ts_handler.auto_filter.filters import (
    _autocorr,
)


@dataclass
class ScoringWeights:
    fidelity_mse: float = 0.16
    gcv: float = 0.18
    roughness: float = 0.10
    residual_autocorr: float = 0.16
    spectral_distance: float = 0.14
    residual_iid: float = 0.10
    derivative_corr: float = 0.16

    def __post_init__(self) -> None:
        total = (
            self.fidelity_mse
            + self.gcv
            + self.roughness
            + self.residual_autocorr
            + self.spectral_distance
            + self.residual_iid
            + self.derivative_corr
        )
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"ScoringWeights must sum to 1, got {total:.4f}")


def _ljung_box_stat(residual: np.ndarray, lags: int) -> float:
    if len(residual) < lags + 2:
        return 0.0
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(residual, lags=lags, return_df=True)
        return float(lb["lb_stat"].iloc[-1])
    except Exception:
        return float(
            np.mean([abs(_autocorr(residual, lag)) for lag in range(1, lags + 1)])
        )


def _noise_var(original: np.ndarray) -> float:
    if len(original) < 3:
        return float(np.var(original)) + 1e-12
    return float(np.median(np.abs(np.diff(original))) / 0.6745) ** 2 + 1e-12


def _mc_divergence(
    denoised: np.ndarray,
    original: np.ndarray,
    filter_fn: Callable[[pd.Series], pd.Series],
    index: pd.Index,
    sigma: float,
    *,
    seed: int = 42,
    n_probes: int = 1,
) -> float | None:
    n = len(original)
    eps = max(0.1 * sigma, 1e-8)
    rng = np.random.default_rng(seed)
    divs: list[float] = []
    for _ in range(max(1, n_probes)):
        b = rng.standard_normal(n)
        try:
            perturbed = filter_fn(pd.Series(original + eps * b, index=index))
            perturbed = np.asarray(perturbed, dtype=float).reshape(-1)[:n]
        except Exception:
            return None
        divs.append(max(float(np.dot(b, perturbed - denoised) / eps), 0.0))
    return float(np.mean(divs))


def _gcv_estimate(
    denoised: np.ndarray,
    original: np.ndarray,
    filter_fn: Callable[[pd.Series], pd.Series] | None = None,
    index: pd.Index | None = None,
    *,
    seed: int = 42,
    n_probes: int = 1,
) -> float:
    n = len(original)
    if n < 8:
        return float(np.mean((original - denoised) ** 2))

    residual = original - denoised
    mse = float(np.mean(residual**2))
    sigma = np.sqrt(_noise_var(original))

    df: float | None = None
    if filter_fn is not None and index is not None:
        df = _mc_divergence(
            denoised, original, filter_fn, index, sigma, seed=seed, n_probes=n_probes
        )

    if df is None:
        var_in = float(np.var(original)) + 1e-12
        df = n * _clip01(1.0 - mse / var_in)

    df = float(np.clip(df, 0.0, 0.999 * n))
    denom = (1.0 - df / n) ** 2
    return mse / max(denom, 1e-9)


def _relative_curvature(denoised: np.ndarray, original: np.ndarray) -> float:
    if len(denoised) < 3 or len(original) < 3:
        return 1.0
    d2_out = float(np.std(np.diff(denoised, n=2)))
    d2_in = float(np.std(np.diff(original, n=2)))
    return _safe_ratio(d2_out, d2_in)


def _residual_iid_defect(residual: np.ndarray) -> float:
    n = len(residual)
    if n < 16:
        return 0.0
    r = residual - np.mean(residual)
    scale = _robust_scale(r)
    if scale <= 1e-12:
        return 0.0

    half = n // 2
    v1 = float(np.var(r[:half])) + 1e-12
    v2 = float(np.var(r[half:])) + 1e-12
    hetero = abs(np.log(v2 / v1))

    z = r / scale
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4)) - 3.0
    non_gauss = (skew**2) / 6.0 + (kurt**2) / 24.0

    return float(hetero + non_gauss)


def _spectral_distance(denoised: np.ndarray, original: np.ndarray) -> float:
    n = min(len(denoised), len(original))
    if n < 32:
        return 0.0
    nperseg = min(256, max(32, n // 8))
    _, p_o = signal.welch(original, nperseg=nperseg, detrend=False)
    _, p_d = signal.welch(denoised, nperseg=nperseg, detrend=False)
    p_o = np.maximum(p_o, 1e-12)
    p_d = np.maximum(p_d, 1e-12)
    p_o /= p_o.sum()
    p_d /= p_d.sum()
    kl_od = float(np.sum(p_o * np.log(p_o / p_d)))
    kl_do = float(np.sum(p_d * np.log(p_d / p_o)))
    return 0.5 * (kl_od + kl_do)


def filter_metrics(
    denoised: pd.Series,
    original: pd.Series,
    max_lag: int = 20,
    filter_fn: Callable[[pd.Series], pd.Series] | None = None,
    *,
    use_mc_gcv: bool = True,
) -> dict[str, float]:
    d = denoised.values.astype(float)
    o = original.values.astype(float)
    residual = o - d

    upper = min(max_lag, len(residual) - 1)
    lb_q = _ljung_box_stat(residual, lags=upper) if upper >= 1 else 0.0

    gcv_fn = filter_fn if use_mc_gcv else None
    return {
        "fidelity_mse": float(np.mean((d - o) ** 2)),
        "gcv": _gcv_estimate(d, o, filter_fn=gcv_fn, index=original.index),
        "roughness": _relative_curvature(d, o),
        "residual_autocorr": lb_q,
        "spectral_distance": _spectral_distance(d, o),
        "residual_iid": _residual_iid_defect(residual),
        "derivative_corr": abs(_safe_corr(np.diff(d), np.diff(o))),
    }


def _rank_normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
    n = len(values)
    if n == 1:
        return np.array([0.5])
    order = np.argsort(values)
    ranks = np.empty(n)
    ranks[order] = np.arange(n) / (n - 1)
    return 1.0 - ranks if invert else ranks


def _compute_scores(
    mdf: pd.DataFrame,
    weights: ScoringWeights,
    *,
    band_penalty: pd.Series | None = None,
) -> pd.Series:
    fid_n = _rank_normalize(mdf["fidelity_mse"].values)
    gcv_n = _rank_normalize(mdf["gcv"].values)
    rough_n = _rank_normalize(mdf["roughness"].values)
    resid_n = _rank_normalize(mdf["residual_autocorr"].values)
    spec_n = _rank_normalize(mdf["spectral_distance"].values)
    iid_n = _rank_normalize(mdf["residual_iid"].values)
    deriv_n = _rank_normalize(mdf["derivative_corr"].values, invert=True)

    score = (
        weights.fidelity_mse * fid_n
        + weights.gcv * gcv_n
        + weights.roughness * rough_n
        + weights.residual_autocorr * resid_n
        + weights.spectral_distance * spec_n
        + weights.residual_iid * iid_n
        + weights.derivative_corr * deriv_n
    )
    if band_penalty is not None:
        score = score + band_penalty.reindex(mdf.index).fillna(0.0).to_numpy()
    return pd.Series(score, index=mdf.index, name="score")


def _candidate_band_penalties(
    original: pd.Series,
    candidates: dict[str, pd.Series],
    target: dict[str, float],
) -> pd.Series:
    penalties: dict[str, float] = {}
    for name, series in candidates.items():
        penalty, _ = _band_penalty(series, original, target=target)
        penalties[name] = penalty
    return pd.Series(penalties)


def _band_penalty(
    winner: pd.Series,
    original: pd.Series,
    *,
    target: dict[str, float],
) -> tuple[float, dict[str, float]]:
    o = original.to_numpy(dtype=float)
    d = winner.to_numpy(dtype=float)
    residual = o - d
    rel_mae = float(np.mean(np.abs(residual))) / max(float(np.mean(np.abs(o))), 1e-12)
    roughness_ratio = float(np.std(np.diff(d))) / max(float(np.std(np.diff(o))), 1e-12)
    derivative_corr = abs(_safe_corr(np.diff(d), np.diff(o)))
    penalty = (
        4.0 * max(target["rel_mae_min"] - rel_mae, 0.0)
        + 4.0 * max(rel_mae - target["rel_mae_max"], 0.0)
        + 3.0 * max(target["roughness_ratio_min"] - roughness_ratio, 0.0)
        + 3.0 * max(roughness_ratio - target["roughness_ratio_max"], 0.0)
        + 4.0 * max(target["derivative_corr_min"] - derivative_corr, 0.0)
    )
    diagnostics = {
        "rel_mae": rel_mae,
        "roughness_ratio": roughness_ratio,
        "derivative_corr": derivative_corr,
    }
    return penalty, diagnostics


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(den, 1e-12))


def _robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mad = float(np.median(np.abs(x - np.median(x))))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(x))
    return max(scale, 1e-12)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(c) else float(c)
