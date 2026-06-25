"""Time-series denoising toolkit with automatic filter selection."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy import signal

from foreblocks.ts_handler.auto_filter.filters import (
    _autocorr,
    _safe_corr,
    _valid_odd_window,
    bilateral_filter,
    butter_lowpass,
    ceemdan_vmd_filter,
    gaussian_filter,
    gaussian_process_smoother,
    hp_filter,
    kalman_rts_smoother,
    l1_trend_filter,
    lowess_filter,
    non_local_means_filter,
    robust_loess_filter,
    savgol_filter,
    ssa_filter,
    stl_residual_denoise,
    train_dae,
    train_vae,
    tv_denoise,
    vmd_filter,
    wavelet_denoise,
    whittaker_smoother,
)
from foreblocks.ts_handler.auto_filter.registry import (
    _FILTER_REGISTRY,
    _SLOW_FILTERS,
    register_filter,
)


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


__all__ = [
    "auto_filter",
    "register_filter",
    "gaussian_filter",
    "savgol_filter",
    "butter_lowpass",
    "wavelet_denoise",
    "tv_denoise",
    "lowess_filter",
    "robust_loess_filter",
    "whittaker_smoother",
    "hp_filter",
    "l1_trend_filter",
    "ssa_filter",
    "gaussian_process_smoother",
    "non_local_means_filter",
    "kalman_rts_smoother",
    "bilateral_filter",
    "stl_residual_denoise",
    "vmd_filter",
    "ceemdan_vmd_filter",
    "train_dae",
    "train_vae",
    "filter_metrics",
    "tune_weights",
    "tune_filter",
    "TuneFilterResult",
    "suggest_weights",
    "plot_results",
]

_SEED = 42

# ---------------------------------------------------------------------------
# Metrics and scoring
# ---------------------------------------------------------------------------


@dataclass
class ScoringWeights:
    """Weights for the composite filter score (must sum to 1).

    Seven axes covering complementary aspects of denoising quality:

    fidelity_mse
        Raw MSE between output and input. Cheap anchor, but rewards filters
        that barely touch the signal.
    gcv
        Generalized Cross-Validation score (Craven & Wahba 1979):
        MSE / (1 − df/n)², with effective degrees of freedom df estimated by
        Monte-Carlo divergence probing (Ramani et al. 2008). The standard
        automatic smoothing-parameter selector — bias-aware, so it penalises
        both over-smoothing (high residual energy) and under-smoothing
        (high df) without needing the clean signal.
    roughness
        Relative curvature: std(diff²(out)) / std(diff²(in)). Scale-free,
        outlier-robust smoothness measure. Lower → smoother relative to input.
    residual_autocorr
        Ljung-Box Q-statistic at lag 20 on the residual. Lower Q → whiter
        residual → less signal left behind.
    spectral_distance
        Symmetric KL divergence between Welch-PSD of output and input. Catches
        filters with acceptable MSE that kill a seasonal peak.
    residual_iid
        Heteroskedasticity + non-Gaussianity of the residual. Lower → the
        removed component is closer to iid noise (not structured signal).
    derivative_corr
        |corr| of first-differences input vs output. Higher → better local
        shape preservation.
    """

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
    """Ljung-Box Q-statistic. Lower Q → whiter residual."""
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
    """Robust additive-noise variance estimate (MAD of first differences)."""
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
    seed: int = _SEED,
    n_probes: int = 1,
) -> float | None:
    """Monte-Carlo estimate of the effective degrees of freedom div_x(f).

    Uses the finite-difference probe of Ramani, Blu & Unser (2008):
        df ≈ ⟨b, f(x + εb) − f(x)⟩/ε,   b ~ N(0, I),
    averaged over ``n_probes`` independent probes. Each probe re-runs the
    filter once, so this is the dominant cost of GCV; ``n_probes=1`` is the
    default because the divergence enters GCV through a forgiving
    ``(1 − df/n)²`` denominator and does not need to be precise. Returns
    ``None`` if the filter cannot be evaluated on a perturbed input.
    """
    n = len(original)
    # ε on the order of the noise std (Ramani et al. recommend a fraction of σ).
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
        # Per-sample df is non-negative in expectation for a smoother.
        divs.append(max(float(np.dot(b, perturbed - denoised) / eps), 0.0))
    return float(np.mean(divs))


def _gcv_estimate(
    denoised: np.ndarray,
    original: np.ndarray,
    filter_fn: Callable[[pd.Series], pd.Series] | None = None,
    index: pd.Index | None = None,
    *,
    seed: int = _SEED,
    n_probes: int = 1,
) -> float:
    """Generalized Cross-Validation score (Craven & Wahba 1979). Lower is better.

        GCV = (‖x − x̂‖²/n) / (1 − df/n)²,

    where df is the effective degrees of freedom of the estimator, estimated by
    Monte-Carlo divergence probing (Ramani et al. 2008). GCV is the standard
    automatic smoothing-parameter selector and is **bias-aware** where SURE is
    not: the numerator grows when a filter over-smooths (removes real signal,
    raising residual energy), while the ``(1 − df/n)²`` denominator grows the
    score for under-smoothing (high df). The minimiser trades the two off
    without needing the clean signal.

    If ``filter_fn`` is not supplied we cannot probe df, so we fall back to a
    df estimate from the residual energy ratio (a coarse but monotone proxy).
    """
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
        # Proxy: variance explained ≈ effective df fraction. Coarse fallback.
        var_in = float(np.var(original)) + 1e-12
        df = n * _clip01(1.0 - mse / var_in)

    # Clamp df to (0, n) so the denominator stays well-defined.
    df = float(np.clip(df, 0.0, 0.999 * n))
    denom = (1.0 - df / n) ** 2
    return mse / max(denom, 1e-9)


def _relative_curvature(denoised: np.ndarray, original: np.ndarray) -> float:
    """Output curvature relative to input curvature (scale-free smoothness).

    std(diff²(output)) / std(diff²(input)). Second differences measure
    curvature; the ratio is dimensionless and far less outlier-dominated than
    std of first differences. Lower → smoother output relative to input.
    Complements ``derivative_corr`` (which measures shape *correlation*) rather
    than re-scoring the first-difference magnitude.
    """
    if len(denoised) < 3 or len(original) < 3:
        return 1.0
    d2_out = float(np.std(np.diff(denoised, n=2)))
    d2_in = float(np.std(np.diff(original, n=2)))
    return _safe_ratio(d2_out, d2_in)


def _residual_iid_defect(residual: np.ndarray) -> float:
    """How far the residual is from iid Gaussian noise (lower is better).

    A good denoiser leaves structureless, homoskedastic, near-Gaussian
    residuals. We combine two cheap, complementary defects:

    * **Heteroskedasticity** — |log(var(second half) / var(first half))|.
      Catches filters that denoise some regions but leave noise elsewhere.
    * **Non-Gaussianity** — normalised excess of robust skew + kurtosis
      (Jarque-Bera-style), which spikes when the residual still contains
      structured signal (peaks, ramps) rather than noise.
    """
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
    non_gauss = (skew**2) / 6.0 + (kurt**2) / 24.0  # JB per-sample form

    return float(hetero + non_gauss)


def _spectral_distance(denoised: np.ndarray, original: np.ndarray) -> float:
    """Symmetric KL divergence between Welch power spectra.

    Both spectra are normalised to sum to 1 (so they're proper distributions),
    so the metric is sensitive to *shape* of the spectrum, not absolute power.
    A filter that wipes out a seasonal peak gets a large value; a filter that
    only attenuates broadband noise gets a small one.
    """
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
    """Compute seven quality metrics for a denoised series.

    Returns a dict with keys:
        fidelity_mse        — lower is better (MSE vs input)
        gcv                 — lower is better (Generalized Cross-Validation)
        roughness           — lower is better (relative curvature ratio)
        residual_autocorr   — lower is better (Ljung-Box Q-stat, lags=max_lag)
        spectral_distance   — lower is better (symmetric KL of Welch PSDs)
        residual_iid        — lower is better (heteroskedasticity + non-Gaussianity)
        derivative_corr     — higher is better (shape preservation)

    Parameters
    ----------
    filter_fn:
        The filter callable that produced ``denoised``. When supplied *and*
        ``use_mc_gcv`` is True, ``gcv`` estimates the effective degrees of
        freedom via one Monte-Carlo divergence probe (one extra filter
        evaluation); otherwise it falls back to a cheap residual-energy proxy
        for df.
    use_mc_gcv:
        Set False to skip the Monte-Carlo probe entirely. This is the hot path
        for the Optuna tuners (``tune_weights`` / ``tune_filter``), which call
        ``filter_metrics`` once per trial — probing there would re-run every
        filter on every trial. The cheap df proxy keeps GCV monotone enough for
        ranking while removing the per-trial filter re-evaluation.
    """
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
    """Rank-based normalization to [0, 1] — robust to outliers."""
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
    """Per-candidate hinge penalty for being outside the fidelity/smoothness band.

    Without this term, rank-normalised scoring can elect a filter that wins on
    every metric *rank* while sitting far outside the band (e.g. a Gaussian
    default whose ``rel_mae`` is 0.9 still has the best roughness rank).
    """
    penalties: dict[str, float] = {}
    for name, series in candidates.items():
        penalty, _ = _band_penalty(series, original, target=target)
        penalties[name] = penalty
    return pd.Series(penalties)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None or n_jobs == -1:
        return max(os.cpu_count() or 1, 1)
    return max(int(n_jobs), 1)


def _run_filter_candidate(
    name: str,
    fn: Callable[[pd.Series], pd.Series],
    ts: pd.Series,
) -> tuple[str, pd.Series]:
    return name, fn(ts)


def _metric_row(
    name: str,
    series: pd.Series,
    ts: pd.Series,
    filter_fn: Callable[[pd.Series], pd.Series] | None,
    use_mc_gcv: bool,
) -> tuple[str, dict[str, float]]:
    return name, filter_metrics(
        series,
        ts,
        filter_fn=filter_fn,
        use_mc_gcv=use_mc_gcv,
    )


def _executor_cls(parallel_backend: str):
    if parallel_backend == "thread":
        return ThreadPoolExecutor
    if parallel_backend == "process":
        return ProcessPoolExecutor
    raise ValueError(
        "parallel_backend must be one of {'thread', 'process'}, "
        f"got {parallel_backend!r}."
    )


def _progress_iter(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def auto_filter(
    ts: pd.Series,
    fast: bool = False,
    weights: ScoringWeights | None = None,
    target_band: dict[str, float] | None = None,
    n_jobs: int | None = 1,
    parallel_backend: Literal["thread", "process"] = "thread",
    use_mc_gcv: bool = True,
    progress: bool = False,
) -> tuple[str, pd.Series, pd.DataFrame]:
    """Run all registered filters, score them, and return the best.

    Parameters
    ----------
    ts:
        Input (noisy) time series.
    fast:
        If True, skip slow filters (e.g. DAE, CEEMDAN+VMD).
    weights:
        Custom ``ScoringWeights`` instance. Defaults to ``ScoringWeights()``.
    target_band:
        Optional fidelity/smoothness band added as a hinge penalty to the
        composite score.  Defaults to ``_DEFAULT_TARGET_BAND`` (the same band
        used by :func:`tune_filter` and the notebook search), which prevents
        rank-normalised scoring from electing a candidate whose absolute
        ``rel_mae`` or ``roughness_ratio`` is far outside what a useful
        denoiser should produce.  Pass an empty dict ``{}`` to disable.
    n_jobs:
        Number of workers used for independent filter and metric evaluations.
        Use ``-1`` or ``None`` for all CPUs.  The default ``1`` keeps
        historical sequential behavior.
    parallel_backend:
        ``"thread"`` has low overhead and works with monkeypatched filters.
        ``"process"`` is usually faster for CPU-bound Python filters, but
        registered filter callables must be pickleable.
    use_mc_gcv:
        If True, estimate GCV degrees of freedom with a Monte-Carlo divergence
        probe, which may re-run each filter during metric scoring.  Set False
        for much faster ranking when trying many filters.
    progress:
        If True, show tqdm progress bars for candidate generation and metric
        scoring when ``tqdm`` is installed. Falls back silently when the
        dependency is unavailable.

    Returns
    -------
    best_name:
        Name of the top-ranked filter.
    best_series:
        Denoised series produced by the best filter.
    score_table:
        DataFrame with per-filter metrics, optional band penalty, and the
        composite score, sorted ascending.
    """
    if weights is None:
        weights = ScoringWeights()
    if target_band is None:
        target_band = _DEFAULT_TARGET_BAND

    active = {
        name: fn
        for name, fn in _FILTER_REGISTRY.items()
        if not (fast and name in _SLOW_FILTERS)
    }

    resolved_n_jobs = _resolve_n_jobs(n_jobs)
    candidates_unordered: dict[str, pd.Series] = {}
    if resolved_n_jobs == 1 or len(active) <= 1:
        for name, fn in _progress_iter(
            active.items(),
            total=len(active),
            desc="auto_filter candidates",
            enabled=progress,
        ):
            try:
                candidates_unordered[name] = fn(ts)
            except Exception as exc:
                warnings.warn(
                    f"Filter '{name}' raised an exception and was skipped: {exc}",
                    stacklevel=2,
                )
    else:
        max_workers = min(resolved_n_jobs, len(active))
        executor_cls = _executor_cls(parallel_backend)
        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_filter_candidate, name, fn, ts): name
                for name, fn in active.items()
            }
            for future in _progress_iter(
                as_completed(futures),
                total=len(futures),
                desc="auto_filter candidates",
                enabled=progress,
            ):
                name = futures[future]
                try:
                    candidate_name, candidate = future.result()
                    candidates_unordered[candidate_name] = candidate
                except Exception as exc:
                    warnings.warn(
                        f"Filter '{name}' raised an exception and was skipped: {exc}",
                        stacklevel=2,
                    )

    candidates = {
        name: candidates_unordered[name]
        for name in active
        if name in candidates_unordered
    }

    if not candidates:
        raise RuntimeError("All filters failed. Cannot rank.")

    if resolved_n_jobs == 1 or len(candidates) <= 1:
        metrics_rows = {
            name: filter_metrics(
                series,
                ts,
                filter_fn=active.get(name),
                use_mc_gcv=use_mc_gcv,
            )
            for name, series in _progress_iter(
                candidates.items(),
                total=len(candidates),
                desc="auto_filter metrics",
                enabled=progress,
            )
        }
    else:
        metrics_unordered: dict[str, dict[str, float]] = {}
        max_workers = min(resolved_n_jobs, len(candidates))
        executor_cls = _executor_cls(parallel_backend)
        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _metric_row,
                    name,
                    series,
                    ts,
                    active.get(name),
                    use_mc_gcv,
                ): name
                for name, series in candidates.items()
            }
            for future in _progress_iter(
                as_completed(futures),
                total=len(futures),
                desc="auto_filter metrics",
                enabled=progress,
            ):
                name = futures[future]
                metrics_name, metrics = future.result()
                metrics_unordered[metrics_name] = metrics
        metrics_rows = {
            name: metrics_unordered[name]
            for name in candidates
            if name in metrics_unordered
        }
    mdf = pd.DataFrame(metrics_rows).T
    band_pen = (
        _candidate_band_penalties(ts, candidates, target_band) if target_band else None
    )
    mdf["score"] = _compute_scores(mdf, weights, band_penalty=band_pen)
    if band_pen is not None:
        mdf["band_penalty"] = band_pen.reindex(mdf.index).fillna(0.0)
    mdf = mdf.sort_values("score")

    best_name = mdf.index[0]
    return best_name, candidates[best_name], mdf


# ---------------------------------------------------------------------------
# Unsupervised weight tuning via Optuna (Bayesian TPE)
# ---------------------------------------------------------------------------


_DEFAULT_TARGET_BAND: dict[str, float] = {
    "rel_mae_min": 0.02,
    "rel_mae_max": 0.12,
    "roughness_ratio_min": 0.35,
    "roughness_ratio_max": 0.92,
    "derivative_corr_min": 0.90,
}

_TUNE_FILTER_BAND_PENALTY_WEIGHT = 10.0


def _band_penalty(
    winner: pd.Series,
    original: pd.Series,
    *,
    target: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Notebook-style band penalty: keep the filter inside a fidelity/smoothness window.

    Returns the scalar penalty plus the three diagnostics (rel_mae,
    roughness_ratio, derivative_corr) that drove it.
    """
    o = original.values.astype(float)
    d = winner.values.astype(float)
    residual = o - d

    abs_mean = max(float(np.mean(np.abs(o))), 1e-12)
    rel_mae = float(np.mean(np.abs(residual))) / abs_mean

    orig_rough = max(float(np.std(np.diff(o))), 1e-12)
    win_rough = float(np.std(np.diff(d)))
    roughness_ratio = win_rough / orig_rough

    derivative_corr = abs(_safe_corr(np.diff(d), np.diff(o)))

    penalty = _target_band_penalty_from_diagnostics(
        rel_mae=rel_mae,
        roughness_ratio=roughness_ratio,
        derivative_corr=derivative_corr,
        rel_mae_band=(target["rel_mae_min"], target["rel_mae_max"]),
        roughness_ratio_band=(
            target["roughness_ratio_min"],
            target["roughness_ratio_max"],
        ),
        min_derivative_corr=target["derivative_corr_min"],
    )

    diagnostics = {
        "rel_mae": rel_mae,
        "roughness_ratio": roughness_ratio,
        "derivative_corr": derivative_corr,
    }
    return penalty, diagnostics


def _bounded_unit(value: float) -> float:
    """Compress non-negative diagnostics to [0, 1) without rank context."""
    value = max(float(value), 0.0)
    return value / (1.0 + value)


def _target_band_penalty_from_diagnostics(
    *,
    rel_mae: float,
    roughness_ratio: float,
    derivative_corr: float,
    rel_mae_band: tuple[float, float],
    roughness_ratio_band: tuple[float, float],
    min_derivative_corr: float,
) -> float:
    penalty = 0.0
    penalty += 4.0 * max(rel_mae_band[0] - rel_mae, 0.0)
    penalty += 4.0 * max(rel_mae - rel_mae_band[1], 0.0)
    penalty += 3.0 * max(roughness_ratio_band[0] - roughness_ratio, 0.0)
    penalty += 3.0 * max(roughness_ratio - roughness_ratio_band[1], 0.0)
    penalty += 4.0 * max(min_derivative_corr - derivative_corr, 0.0)
    return penalty


def _unsupervised_proxy(
    winner: pd.Series,
    original: pd.Series,
    candidates: dict[str, pd.Series],
) -> float:
    """Target-band quality proxy for the winning filter (lower is better).

    The previous version mixed Ljung-Box whiteness with raw output roughness,
    which rewarded *flatter* winners and biased the tuner toward
    over-smoothing.  We now combine:

    1. **Residual whiteness** — rank-normalised Ljung-Box Q-statistic.
       Whiter residual is better, but only as a tie-breaker.

    2. **Target-band penalty** — hinge loss that pushes the winner into the
       same band the notebook search uses (rel_mae 0.02-0.12,
       roughness_ratio 0.35-0.92, derivative_corr >= 0.90).

    The band term dominates whenever the winner sits outside the band, so a
    filter that simply flattens the signal cannot win on whiteness alone.
    """
    lags = min(20, len(original) // 5)
    lb_stats: list[float] = []

    for s in candidates.values():
        resid = (original - s).values.astype(float)
        try:
            lb_stats.append(_ljung_box_stat(resid, lags=lags))
        except Exception:
            lb_stats.append(float("nan"))

    names = list(candidates.keys())
    winner_name = winner.name
    if winner_name not in names:
        for i, s in enumerate(candidates.values()):
            if np.allclose(s.values, winner.values, atol=1e-10):
                winner_idx = i
                break
        else:
            winner_idx = 0
    else:
        winner_idx = names.index(winner_name)

    def _rank_norm_list(vals: list[float]) -> list[float]:
        arr = np.array(vals, dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() < 2:
            return [0.5] * len(vals)
        ranks = np.empty(len(arr))
        order = np.argsort(arr[finite])
        finite_indices = np.where(finite)[0]
        for rank, fi in enumerate(order):
            ranks[finite_indices[fi]] = rank / (finite.sum() - 1)
        ranks[~finite] = 1.0
        return ranks.tolist()

    lb_norm = _rank_norm_list(lb_stats)
    band_pen, _ = _band_penalty(winner, original, target=_DEFAULT_TARGET_BAND)
    return 0.5 * lb_norm[winner_idx] + band_pen


def _oversmoothing_penalty(
    winner: pd.Series,
    original: pd.Series,
    metrics: pd.Series,
    *,
    min_derivative_corr: float | None = None,
    min_rel_mae: float | None = None,
    max_rel_mae: float | None = None,
    min_roughness_ratio: float | None = None,
    max_roughness_ratio: float | None = None,
) -> float:
    """Penalty term that keeps the winner inside a fidelity/smoothness band.

    Both sides of the band matter: ``max_rel_mae`` / ``min_roughness_ratio``
    discourage flattening, while ``min_rel_mae`` / ``max_roughness_ratio``
    discourage filters that barely touch the signal.
    """
    penalty = 0.0

    if min_derivative_corr is not None:
        penalty += 4.0 * max(
            min_derivative_corr - float(metrics["derivative_corr"]), 0.0
        )

    rel_mae = float(np.mean(np.abs((original - winner).values.astype(float)))) / max(
        float(np.mean(np.abs(original.values.astype(float)))), 1e-12
    )
    if max_rel_mae is not None:
        penalty += 3.0 * max(rel_mae - max_rel_mae, 0.0)
    if min_rel_mae is not None:
        penalty += 3.0 * max(min_rel_mae - rel_mae, 0.0)

    original_roughness = float(np.std(np.diff(original.values.astype(float))))
    winner_roughness = float(np.std(np.diff(winner.values.astype(float))))
    roughness_ratio = winner_roughness / max(original_roughness, 1e-12)
    if min_roughness_ratio is not None:
        penalty += 2.5 * max(min_roughness_ratio - roughness_ratio, 0.0)
    if max_roughness_ratio is not None:
        penalty += 2.5 * max(roughness_ratio - max_roughness_ratio, 0.0)

    return penalty


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(den, 1e-12))


def _robust_scale(x: np.ndarray) -> float:
    """MAD-based scale with a std fallback for near-constant signals."""
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
    """Map diagnostic logits to a stable simplex with floors and soft caps."""
    logits = np.asarray(logits, dtype=float)
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - float(np.max(logits))
    raw = np.exp(logits)
    raw /= max(float(raw.sum()), 1e-12)

    n = raw.size
    floor = float(np.clip(floor, 0.0, 1.0 / max(n, 1) - 1e-9))
    weights = floor + (1.0 - floor * n) * raw

    # A single metric should rarely dominate an unsupervised denoising choice.
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
    """Shift weight mass toward denoising criteria when noise pressure is high.

    The heuristic logits can still leave too much mass on fidelity/shape for
    differenced or spike-heavy series where actual smoothing should dominate.
    This post-pass enforces a minimum combined share for roughness and residual
    whiteness when the signal diagnostics say the input is mostly noise.
    """
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

    # Indices match ScoringWeights field order:
    # 0=fidelity_mse, 1=gcv, 2=roughness, 3=residual_autocorr,
    # 4=spectral_distance, 5=residual_iid, 6=derivative_corr.
    # "Smoothing criteria" = roughness + residual_autocorr.
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


def _signal_characteristics(x: np.ndarray) -> dict[str, float]:
    """Robust, cheap signal diagnostics used by ``suggest_weights``."""
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

    # Robust white-noise proxies. The second-difference estimator catches
    # sample-to-sample jitter, while the high-pass residual avoids calling a
    # strong but smooth trend "noise".
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
                pd
                .Series(centered)
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
    """Heuristically suggest :class:`ScoringWeights` from signal characteristics.

    The heuristic is intentionally cheap and deterministic, but it follows a
    signal-aware prior similar to modern unsupervised denoising objectives:
    separate structure from high-frequency pressure, detect whether the series
    has memory/seasonality/trend/jumps, then project bounded evidence logits to
    the weight simplex.

    Signal properties measured
    --------------------------
    noise
        Robust MAD estimate from second differences. Noisier signals reduce
        fidelity pressure because matching the raw series too closely preserves
        noise.
    periodicity
        Low spectral entropy plus concentrated periodogram energy. Periodic
        structure increases residual-whiteness and derivative-preservation
        pressure.
    trend
        R2 of a linear fit. Trend-dominated signals increase shape preservation.
    high_freq / outliers
        Extra pressure to smooth when high-frequency energy or spikes dominate.
    jumps / memory
        Jump pressure protects level shifts from being over-smoothed. Memory
        and periodic structure increase residual-whiteness and
        derivative-preservation pressure because correlated residuals often
        mean the filter removed real signal.

    Examples
    --------
    >>> w = suggest_weights(ts_noisy)
    >>> best_name, best_series, score_table = auto_filter(ts_noisy, weights=w)
    >>> w, why = suggest_weights(ts_noisy, explain=True)
    >>> why["reasons"]
    """
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

    # Logit order matches ScoringWeights field order:
    # [fidelity_mse, gcv, roughness, residual_autocorr, spectral_distance,
    #  residual_iid, derivative_corr]
    logits = np.array(
        [
            # fidelity_mse: prefer when clean & structured, avoid when noisy
            0.45
            + 0.85 * clean
            + 0.30 * structure
            - 1.35 * smoothing_pressure
            - 0.55 * outliers
            - 0.30 * roughness,
            # gcv: prefer when there is genuine noise to remove so the
            # bias/variance tradeoff is informative. Down-weight for
            # spike/jump-heavy inputs where the noise model and the df estimate
            # are less reliable.
            -0.10 + 1.00 * noise + 0.30 * roughness - 0.50 * outliers - 0.25 * jumps,
            # roughness (relative curvature): prefer when noise/HF dominate
            -0.25
            + 1.25 * smoothing_pressure
            + 0.45 * high_freq
            + 0.45 * outliers
            + 0.35 * roughness
            - 0.15 * jumps
            - 0.15 * periodicity,
            # residual_autocorr (Ljung-Box): prefer when memory/periodicity
            # suggest residual whiteness is informative
            0.30
            + 0.95 * noise
            + 0.75 * memory
            + 0.50 * periodicity
            + 0.35 * high_freq
            + 0.20 * smoothing_pressure
            - 0.10 * trend
            - 0.08 * jumps,
            # spectral_distance: prefer when periodicity is high (PSD shape
            # matters) or when memory is high (lots of in-band energy)
            -0.15
            + 1.20 * periodicity
            + 0.45 * memory
            + 0.20 * trend
            - 0.30 * outliers
            - 0.20 * jumps,
            # residual_iid: prefer when there is genuine noise to remove, so
            # an iid residual is meaningful. Down-weight for spike/jump-heavy
            # series where the residual is not expected to be Gaussian.
            -0.10
            + 0.85 * noise
            + 0.30 * high_freq
            + 0.20 * memory
            - 0.55 * outliers
            - 0.40 * jumps,
            # derivative_corr: prefer when there is shape/trend/jumps to preserve
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
    # Indices 0=fid, 1=gcv, 2=rough, 3=resid_ac, 4=spec, 5=iid, 6=deriv.
    # Bump derivative_corr above residual_autocorr when structure is clear.
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


def tune_weights(
    ts: pd.Series,
    n_trials: int = 100,
    fast: bool = True,
    seed: int = _SEED,
    verbose: bool = False,
    warm_start: bool = True,
    min_derivative_corr: float | None = 0.90,
    min_rel_mae: float | None = 0.02,
    max_rel_mae: float | None = 0.12,
    min_roughness_ratio: float | None = 0.35,
    max_roughness_ratio: float | None = 0.92,
) -> ScoringWeights:
    """Auto-tune :class:`ScoringWeights` using Optuna (Bayesian TPE, unsupervised).

    All filters are run **once** before the optimisation loop; Optuna only
    searches the 7-dimensional weight simplex (fidelity_mse, gcv, roughness,
    residual_autocorr, spectral_distance, residual_iid, derivative_corr).

    Objective (minimised)
    ---------------------
    Two-part proxy evaluated on whichever filter the trial weights select:

    * **Residual whiteness** — rank-normalised Ljung-Box Q-statistic on
      (original − denoised). Acts as a tie-breaker.
    * **Target-band penalty** — hinge loss that keeps the winner inside a
      fidelity/smoothness window (``rel_mae`` ∈ [min, max],
      ``roughness_ratio`` ∈ [min, max], ``derivative_corr`` ≥ min).
      Without the band, the optimiser would converge on whichever filter
      flattens the signal hardest, since that maximises residual whiteness.

    Parameters
    ----------
    ts:
        Input (noisy) time series.
    n_trials:
        Number of Optuna trials (default 100).
    fast:
        If True, exclude slow filters (DAE, CEEMDAN+VMD) — strongly recommended.
    seed:
        Random seed for reproducibility.
    verbose:
        If False (default), Optuna logging is suppressed.
    warm_start:
        If True (default), enqueue the :func:`suggest_weights` heuristic as
        the first Optuna trial so the TPE sampler starts from a signal-aware
        point rather than a random one.
    min_derivative_corr:
        Optional lower bound for acceptable derivative correlation of the
        selected winner. Lower values receive an optimization penalty.
    max_rel_mae:
        Optional upper bound on relative mean absolute deviation between the
        winner and the original signal. Higher values receive a penalty.
    min_roughness_ratio:
        Optional lower bound on output roughness divided by input roughness.
        Very small ratios indicate aggressive flattening and receive a penalty.

    Returns
    -------
    ScoringWeights
        Tuned weights passable directly to :func:`auto_filter`.

    Examples
    --------
    >>> weights = tune_weights(my_series, n_trials=200)
    >>> best_name, best_series, score_table = auto_filter(my_series, weights=weights)
    """
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    active = {
        name: fn
        for name, fn in _FILTER_REGISTRY.items()
        if not (fast and name in _SLOW_FILTERS)
    }

    candidates: dict[str, pd.Series] = {}
    for name, fn in active.items():
        try:
            out = fn(ts).rename(name)
            candidates[name] = out
        except Exception as exc:
            warnings.warn(f"tune_weights: filter '{name}' failed: {exc}", stacklevel=2)

    if len(candidates) < 2:
        raise RuntimeError("Fewer than 2 filters succeeded; cannot tune weights.")

    metrics_rows = {
        name: filter_metrics(series, ts, filter_fn=active.get(name))
        for name, series in candidates.items()
    }
    mdf_base = pd.DataFrame(metrics_rows).T

    # Order matches ScoringWeights fields:
    # [fidelity_mse, gcv, roughness, residual_autocorr, spectral_distance, residual_iid, derivative_corr]
    n_weights = 7

    def _weights_from_raw(raw: np.ndarray) -> ScoringWeights:
        return ScoringWeights(
            fidelity_mse=float(raw[0]),
            gcv=float(raw[1]),
            roughness=float(raw[2]),
            residual_autocorr=float(raw[3]),
            spectral_distance=float(raw[4]),
            residual_iid=float(raw[5]),
            derivative_corr=float(raw[6]),
        )

    def objective(trial: optuna.Trial) -> float:
        logits = np.array([
            trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(n_weights)
        ])
        logits /= logits.sum()
        w = _weights_from_raw(logits)
        scores = _compute_scores(mdf_base, w)
        best_name = str(scores.idxmin())
        winner = candidates[best_name]
        base_value = _unsupervised_proxy(winner, ts, candidates)
        penalty = _oversmoothing_penalty(
            winner,
            ts,
            mdf_base.loc[best_name],
            min_derivative_corr=min_derivative_corr,
            min_rel_mae=min_rel_mae,
            max_rel_mae=max_rel_mae,
            min_roughness_ratio=min_roughness_ratio,
            max_roughness_ratio=max_roughness_ratio,
        )
        return base_value + penalty

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    if warm_start:
        sw = suggest_weights(ts)
        study.enqueue_trial({
            "w0": sw.fidelity_mse,
            "w1": sw.gcv,
            "w2": sw.roughness,
            "w3": sw.residual_autocorr,
            "w4": sw.spectral_distance,
            "w5": sw.residual_iid,
            "w6": sw.derivative_corr,
        })

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best = study.best_params
    raw = np.array([best[f"w{i}"] for i in range(n_weights)], dtype=float)
    raw /= raw.sum()
    tuned = _weights_from_raw(raw)

    if verbose:
        print(f"\nTuned ScoringWeights (n_trials={n_trials}):")
        print(f"  fidelity_mse      = {tuned.fidelity_mse:.4f}")
        print(f"  gcv               = {tuned.gcv:.4f}")
        print(f"  roughness         = {tuned.roughness:.4f}")
        print(f"  residual_autocorr = {tuned.residual_autocorr:.4f}")
        print(f"  spectral_distance = {tuned.spectral_distance:.4f}")
        print(f"  residual_iid      = {tuned.residual_iid:.4f}")
        print(f"  derivative_corr   = {tuned.derivative_corr:.4f}")
        print(f"  best trial value  = {study.best_value:.6f}")

    return tuned


# ---------------------------------------------------------------------------
# Direct filter+parameter search via Optuna
# ---------------------------------------------------------------------------


@dataclass
class TuneFilterResult:
    """Result of :func:`tune_filter`."""

    name: str
    params: dict[str, float | int]
    series: pd.Series
    metrics: dict[str, float]
    rel_mae: float
    roughness_ratio: float
    objective_value: float
    band_penalty: float = 0.0


_TUNE_FILTER_FAMILIES = (
    "gaussian",
    "tv",
    "bilateral",
    "savgol",
    "butter",
    "wavelet",
    "lowess",
    "gp",
    "nlm",
    "hp",
    "kalman",
    "stl",
    "vmd",
    "ssa",
    "whittaker",
    "l1_trend",
)

_TUNE_FILTER_SLOW_FAMILIES = ("ceemdan_vmd", "vae")


def _run_parametrized_filter(
    name: str, params: dict[str, float | int], ts: pd.Series
) -> pd.Series:
    if name == "gaussian":
        return gaussian_filter(ts, sigma=float(params["sigma"]))
    if name == "tv":
        return tv_denoise(ts, weight=float(params["weight"]))
    if name == "bilateral":
        return bilateral_filter(
            ts,
            sigma_t=float(params["sigma_t"]),
            sigma_v=float(params["sigma_v"]),
        )
    if name == "savgol":
        return savgol_filter(
            ts,
            window=int(params["window"]),
            polyorder=int(params["polyorder"]),
        )
    if name == "butter":
        return butter_lowpass(
            ts,
            cutoff=float(params["cutoff"]),
            order=int(params["order"]),
        )
    if name == "wavelet":
        return wavelet_denoise(
            ts,
            levels=int(params["levels"]),
            cycle_spins=int(params["cycle_spins"]),
            wavelet=str(params["wavelet"]),
        )
    if name == "lowess":
        return lowess_filter(ts, frac=float(params["frac"]), it=int(params["it"]))
    if name == "gp":
        return gaussian_process_smoother(
            ts,
            length_scale=float(params["length_scale"]),
            noise=float(params["noise"]),
            max_inducing=int(params["max_inducing"]),
        )
    if name == "nlm":
        return non_local_means_filter(
            ts,
            patch_radius=int(params["patch_radius"]),
            search_radius=int(params["search_radius"]),
            h=float(params["h"]),
        )
    if name == "hp":
        return hp_filter(ts, lamb=float(params["lamb"]))
    if name == "kalman":
        return kalman_rts_smoother(
            ts,
            q=float(params["q"]),
            r=float(params["r"]),
        )
    if name == "stl":
        return stl_residual_denoise(
            ts,
            period=int(params["period"]),
            seasonal=int(params["seasonal"]),
            resid_levels=int(params["resid_levels"]),
            cycle_spins=int(params["cycle_spins"]),
        )
    if name == "vmd":
        return vmd_filter(
            ts,
            K=int(params["K"]),
            alpha=float(params["alpha"]),
            tau=float(params["tau"]),
            drop_modes=int(params["drop_modes"]),
        )
    if name == "ceemdan_vmd":
        return ceemdan_vmd_filter(
            ts,
            trials=int(params["trials"]),
            epsilon=float(params["epsilon"]),
            K=int(params["K"]),
            alpha=float(params["alpha"]),
            tau=float(params["tau"]),
        )
    if name == "vae":
        return train_vae(
            ts,
            window=int(params["window"]),
            epochs=int(params["epochs"]),
            noise_std=float(params["noise_std"]),
            beta=float(params["beta"]),
            latent_size=int(params["latent_size"]),
        )
    if name == "ssa":
        return ssa_filter(
            ts,
            window=int(params["window"]),
            n_components=int(params["n_components"]),
        )
    if name == "whittaker":
        return whittaker_smoother(
            ts, lam=float(params["lam"]), order=int(params["order"])
        )
    if name == "l1_trend":
        return l1_trend_filter(ts, lam=float(params["lam"]))
    raise ValueError(f"Unknown filter family: {name}")


def _suggest_filter_and_params(
    trial: optuna.Trial,
    families: tuple[str, ...],
    value_scale: float,
) -> tuple[str, dict[str, float | int]]:
    name = trial.suggest_categorical("filter_name", list(families))

    if name == "gaussian":
        return name, {"sigma": trial.suggest_float("sigma", 0.6, 6.0, log=True)}
    if name == "tv":
        return name, {"weight": trial.suggest_float("weight", 0.03, 1.2, log=True)}
    if name == "bilateral":
        return name, {
            "sigma_t": trial.suggest_float("sigma_t", 1.0, 14.0),
            "sigma_v": trial.suggest_float(
                "sigma_v", 0.2 * value_scale, 2.5 * value_scale
            ),
        }
    if name == "savgol":
        window = trial.suggest_int("window", 5, 31, step=2)
        polyorder = trial.suggest_int("polyorder", 2, min(4, window - 1))
        return name, {"window": window, "polyorder": polyorder}
    if name == "butter":
        return name, {
            "cutoff": trial.suggest_float("cutoff", 0.02, 0.25),
            "order": trial.suggest_int("order", 2, 5),
        }
    if name == "wavelet":
        return name, {
            "levels": trial.suggest_int("levels", 1, 4),
            "cycle_spins": trial.suggest_int("cycle_spins", 1, 6),
            "wavelet": trial.suggest_categorical("wavelet", ["db4", "sym8", "coif3"]),
        }
    if name == "lowess":
        return name, {
            "frac": trial.suggest_float("frac", 0.02, 0.20),
            "it": trial.suggest_int("it", 0, 3),
        }
    if name == "gp":
        return name, {
            "length_scale": trial.suggest_float("gp_length_scale", 2.0, 96.0, log=True),
            "noise": trial.suggest_float("gp_noise", 0.01, 0.35, log=True),
            "max_inducing": trial.suggest_categorical(
                "gp_max_inducing", [96, 160, 256]
            ),
        }
    if name == "nlm":
        return name, {
            "patch_radius": trial.suggest_int("nlm_patch_radius", 1, 5),
            "search_radius": trial.suggest_int("nlm_search_radius", 12, 96, step=12),
            "h": trial.suggest_float("nlm_h", 0.05 * value_scale, 2.0 * value_scale),
        }
    if name == "hp":
        return name, {"lamb": trial.suggest_float("hp_lamb", 10.0, 1e5, log=True)}
    if name == "kalman":
        return name, {
            "q": trial.suggest_float("kalman_q", 1e-8, 1.0, log=True),
            "r": trial.suggest_float("kalman_r", 1e-6, 5.0, log=True),
        }
    if name == "stl":
        period = trial.suggest_categorical("stl_period", [12, 24, 48, 168])
        return name, {
            "period": int(period),
            "seasonal": trial.suggest_int("stl_seasonal", 7, 31, step=2),
            "resid_levels": trial.suggest_int("stl_resid_levels", 1, 3),
            "cycle_spins": trial.suggest_int("stl_cycle_spins", 1, 4),
        }
    if name == "vmd":
        return name, {
            "K": trial.suggest_int("vmd_K", 3, 7),
            "alpha": trial.suggest_float("vmd_alpha", 200.0, 8000.0, log=True),
            "tau": trial.suggest_float("vmd_tau", 0.0, 0.2),
            "drop_modes": trial.suggest_int("vmd_drop_modes", 1, 2),
        }
    if name == "ceemdan_vmd":
        return name, {
            "trials": trial.suggest_int("ceemdan_trials", 8, 32),
            "epsilon": trial.suggest_float("ceemdan_epsilon", 0.001, 0.02, log=True),
            "K": trial.suggest_int("ceemdan_vmd_K", 3, 6),
            "alpha": trial.suggest_float("ceemdan_vmd_alpha", 500.0, 8000.0, log=True),
            "tau": trial.suggest_float("ceemdan_vmd_tau", 0.0, 0.2),
        }
    if name == "vae":
        return name, {
            "window": trial.suggest_int("vae_window", 9, 41, step=2),
            "epochs": trial.suggest_int("vae_epochs", 8, 24),
            "noise_std": trial.suggest_float("vae_noise_std", 0.04, 0.25),
            "beta": trial.suggest_float("vae_beta", 0.002, 0.08, log=True),
            "latent_size": trial.suggest_int("vae_latent_size", 3, 12),
        }
    if name == "whittaker":
        return name, {
            "lam": trial.suggest_float("lam", 1.0, 1e5, log=True),
            "order": trial.suggest_int("order", 1, 3),
        }
    if name == "l1_trend":
        return name, {"lam": trial.suggest_float("l1_lam", 0.1, 50.0, log=True)}
    if name == "ssa":
        window = trial.suggest_int("ssa_window", 12, 120, step=4)
        n_components = trial.suggest_int("ssa_components", 1, 8)
        return name, {"window": window, "n_components": n_components}
    raise ValueError(f"Unknown filter family: {name}")


def tune_filter(
    ts: pd.Series,
    n_trials: int = 60,
    seed: int = _SEED,
    verbose: bool = False,
    progress: bool = False,
    families: tuple[str, ...] = _TUNE_FILTER_FAMILIES,
    rel_mae_band: tuple[float, float] = (0.02, 0.12),
    roughness_ratio_band: tuple[float, float] = (0.35, 0.92),
    min_derivative_corr: float = 0.90,
) -> TuneFilterResult:
    """Search filter family and per-filter hyperparameters jointly with Optuna.

    Unlike :func:`auto_filter` (which picks among fixed-default filters) and
    :func:`tune_weights` (which only tunes scoring weights), this function
    optimises the actual filter parameters subject to a fidelity/smoothness
    band.  This is the closest analogue to the notebook's Bayesian search and
    is the right tool when the default-configured filters all over- or
    under-smooth the input.

    Objective (minimised)
    ---------------------
    A target-band hinge penalty, weighted heavily enough to act as the primary
    guardrail, plus a bounded weighted sum of:

    * residual autocorrelation (whiteness)        — weight 0.75
    * fidelity MSE                                 — weight 0.10
    * output roughness                             — weight 0.10
    * negative derivative correlation              — weight -0.15

    * ``rel_mae`` ∈ ``rel_mae_band``
    * ``roughness_ratio`` ∈ ``roughness_ratio_band``
    * ``derivative_corr`` ≥ ``min_derivative_corr``

    Parameters
    ----------
    ts:
        Input (noisy) time series.  Pre-normalise (z-score) for stable
        ``sigma_v`` bounds in the bilateral search.
    n_trials:
        Number of Optuna trials.
    progress:
        If True, show Optuna's tqdm-backed trial progress bar when `tqdm` is
        installed. This is independent from `verbose`, so you can keep Optuna
        logs quiet while still seeing search progress.
    families:
        Filter families to include in the search.  Defaults to all supported
        parametric filters.
    rel_mae_band, roughness_ratio_band, min_derivative_corr:
        Target band for the tradeoff between fidelity and smoothing.  The
        defaults match the bands used in ``estudo_rebeca/dataset_run.ipynb``.

    Returns
    -------
    TuneFilterResult
        Best filter name, parameters, denoised series, metrics, and target
        diagnostics.
    """
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    x = np.asarray(ts.to_numpy(), dtype=float)
    abs_mean = max(float(np.mean(np.abs(x))), 1e-12)
    orig_roughness = max(float(np.std(np.diff(x))), 1e-12)
    value_scale = float(np.median(np.abs(x - np.median(x)))) + 1e-6

    def _run(name: str, params: dict[str, float | int]) -> pd.Series:
        return _run_parametrized_filter(name, params, ts)

    def objective(trial: optuna.Trial) -> float:
        name, params = _suggest_filter_and_params(trial, families, value_scale)
        filtered = _run(name, params)
        # Skip the Monte-Carlo GCV probe inside the per-trial objective: it
        # would re-run the filter on every trial. The cheap df proxy is enough
        # for ranking, and the band penalty already governs the tradeoff.
        metrics = filter_metrics(filtered, ts, use_mc_gcv=False)
        residual = ts - filtered
        rel_mae = float(residual.abs().mean()) / abs_mean
        roughness_ratio = float(np.std(np.diff(filtered.to_numpy()))) / orig_roughness

        derivative_corr = float(metrics["derivative_corr"])
        band_penalty = _target_band_penalty_from_diagnostics(
            rel_mae=rel_mae,
            roughness_ratio=roughness_ratio,
            derivative_corr=derivative_corr,
            rel_mae_band=rel_mae_band,
            roughness_ratio_band=roughness_ratio_band,
            min_derivative_corr=min_derivative_corr,
        )

        # Whiteness: Ljung-Box Q normalised by its dof (≈ lags) so it stays O(1)
        # and is comparable in scale to the other O(1) terms.
        lags = max(min(20, len(ts) // 5), 1)
        whiteness = float(metrics["residual_autocorr"]) / lags

        base_loss = (
            0.45 * _bounded_unit(whiteness)
            + 0.20 * _bounded_unit(float(metrics["residual_iid"]))
            + 0.15 * _bounded_unit(float(metrics["roughness"]))
            + 0.10 * _bounded_unit(float(metrics["spectral_distance"]))
            - 0.15 * derivative_corr
        )
        loss = _TUNE_FILTER_BAND_PENALTY_WEIGHT * band_penalty + base_loss

        trial.set_user_attr("filter_name", name)
        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", dict(metrics))
        trial.set_user_attr("rel_mae", rel_mae)
        trial.set_user_attr("roughness_ratio", roughness_ratio)
        trial.set_user_attr("band_penalty", band_penalty)
        trial.set_user_attr("base_loss", base_loss)
        return float(loss)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=(progress and tqdm is not None),
    )

    best = study.best_trial
    best_name = best.user_attrs["filter_name"]
    best_params = best.user_attrs["params"]
    best_series = _run_parametrized_filter(best_name, best_params, ts).rename(
        ts.name if ts.name is not None else best_name
    )

    return TuneFilterResult(
        name=best_name,
        params=best_params,
        series=best_series,
        metrics=best.user_attrs["metrics"],
        rel_mae=float(best.user_attrs["rel_mae"]),
        roughness_ratio=float(best.user_attrs["roughness_ratio"]),
        objective_value=float(study.best_value),
        band_penalty=float(best.user_attrs["band_penalty"]),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    noisy: pd.Series,
    score_table: pd.DataFrame,
    candidates: dict[str, pd.Series],
    best_name: str,
    clean: pd.Series | None = None,
    top_k: int = 6,
) -> plt.Figure:
    """Three-panel plot: best filter / top-k comparison / ranking bar chart."""
    ranked = list(score_table.index)
    top_names = ranked[: min(top_k, len(ranked))]

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        gridspec_kw={"height_ratios": [2.2, 2.0, 1.3]},
    )

    ax = axs[0]
    ax.plot(noisy.index, noisy.values, label="Noisy", alpha=0.35, color="tab:gray")
    ax.plot(
        noisy.index,
        candidates[best_name].values,
        label=f"Best: {best_name}",
        linewidth=2.4,
        color="tab:blue",
    )
    if clean is not None:
        ax.plot(
            noisy.index,
            clean.values,
            label="Clean (reference)",
            linestyle="--",
            alpha=0.8,
            color="tab:green",
        )
    ax.set_title("Best Filter vs Noisy Input")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    ax = axs[1]
    ax.plot(noisy.index, noisy.values, label="Noisy", alpha=0.20, color="tab:gray")
    for name in top_names:
        ax.plot(noisy.index, candidates[name].values, label=name, linewidth=1.4)
    ax.set_title(f"Top {len(top_names)} Filters by Score")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(alpha=0.2)

    ax = axs[2]
    score_series = score_table["score"]
    bar_colors = [
        "tab:blue" if n == best_name else "tab:orange" for n in score_series.index
    ]
    ax.bar(score_series.index, score_series.values, color=bar_colors)
    ax.set_ylabel("Score (lower is better)")
    ax.set_title("Filter Ranking")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2, axis="y")

    offset = max(1e-4, float(score_series.max()) * 0.03)
    for i, (name, val) in enumerate(score_series.items()):
        ax.text(
            i,
            val + offset,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo — only runs when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(_SEED)

    t = np.linspace(0, 10, 1000)
    clean = np.sin(2 * np.pi * t) + 0.5 * t
    noisy = clean + rng.normal(0, 0.5, len(t))
    index = pd.date_range("2020-01-01", periods=len(t), freq="D")

    noisy_s = pd.Series(noisy, index=index, name="noisy")
    clean_s = pd.Series(clean, index=index, name="clean")

    print("Tuning weights with Optuna (n_trials=150, fast=True)...")
    tuned_weights = tune_weights(noisy_s, n_trials=150, fast=True, verbose=True)

    best_name, best_series, score_table = auto_filter(
        noisy_s, fast=False, weights=tuned_weights
    )

    print(f"\nBest filter: {best_name}\n")
    print("Scores and metrics (lower score is better):")
    print(score_table.round(4).to_string())

    all_candidates: dict[str, pd.Series] = {}
    for name in score_table.index:
        fn = _FILTER_REGISTRY[name]
        all_candidates[name] = fn(noisy_s) if name != best_name else best_series

    fig = plot_results(noisy_s, score_table, all_candidates, best_name, clean=clean_s)
    plt.show()
