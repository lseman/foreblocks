"""
denoise.py — Time-series denoising toolkit
==========================================
Provides classical, signal-processing, and learning-based filters with
automatic ranking via a composite metric score.

Filters (16 total)
------------------
Classical:
  Moving Average, Gaussian, Savitzky-Golay, Butterworth Lowpass,
  Exponential Smoothing, FFT Denoising

Nonparametric / statistical:
  LOWESS, Hodrick-Prescott, TV Denoising

Subspace / optimal:
  SSA (Singular Spectrum Analysis), Kalman RTS Smoother,
  Wavelet (Bayes+Garrote)

Edge-preserving:
  Bilateral

Adaptive decomposition:
  EMD+VMD Baseline, CEEMDAN+VMD

Learning-based (slow):
  Denoising Autoencoder

Usage
-----
    from denoise import auto_filter, tune_weights, register_filter

    weights = tune_weights(my_series, n_trials=150)
    best_name, best_series, score_table = auto_filter(my_series, weights=weights)

    # Register a custom filter
    @register_filter("My Filter")
    def my_filter(ts: pd.Series) -> pd.Series:
        ...
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PyEMD import CEEMDAN, EMD
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from vmdpy import VMD

__all__ = [
    "auto_filter",
    "register_filter",
    "moving_average",
    "gaussian_filter",
    "savgol_filter",
    "butter_lowpass",
    "exponential_smoothing",
    "fft_denoise",
    "wavelet_denoise",
    "tv_denoise",
    "lowess_filter",
    "hp_filter",
    "ssa_filter",
    "kalman_rts_smoother",
    "bilateral_filter",
    "ceemdan_vmd_filter",
    "emd_vmd_baseline",
    "train_dae",
    "filter_metrics",
    "tune_weights",
    "plot_results",
]

_SEED = 42

# ---------------------------------------------------------------------------
# Filter registry
# ---------------------------------------------------------------------------

_FILTER_REGISTRY: dict[str, Callable[[pd.Series], pd.Series]] = {}
_SLOW_FILTERS: set[str] = set()


def register_filter(name: str, *, slow: bool = False):
    """Decorator that adds a filter function to the auto-selection registry.

    Parameters
    ----------
    name:
        Display name used in scoring tables and plots.
    slow:
        If True the filter is skipped when ``auto_filter(..., fast=True)``.
    """

    def decorator(fn: Callable[[pd.Series], pd.Series]):
        _FILTER_REGISTRY[name] = fn
        if slow:
            _SLOW_FILTERS.add(name)
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_series(
    values: np.ndarray, index: pd.Index, name: str | None = None
) -> pd.Series:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) != len(index):
        raise ValueError(
            f"Filtered output length {len(arr)} != input length {len(index)}."
        )
    return pd.Series(arr, index=index, name=name)


def _valid_odd_window(n: int, preferred: int, minimum: int = 3) -> int:
    if n <= 1:
        return 1
    w = max(int(preferred), minimum)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return max(w, 1)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def _autocorr(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return 0.0
    return _safe_corr(x[:-lag], x[lag:])


# ---------------------------------------------------------------------------
# Classical filters
# ---------------------------------------------------------------------------


@register_filter("Moving Average")
def moving_average(ts: pd.Series, window: int = 7) -> pd.Series:
    window = _valid_odd_window(len(ts), window, minimum=3)
    return ts.rolling(window=window, center=True, min_periods=1).mean()


@register_filter("Gaussian")
def gaussian_filter(ts: pd.Series, sigma: float = 2.0) -> pd.Series:
    """Gaussian kernel smoother via scipy.ndimage.

    Parameters
    ----------
    sigma:
        Standard deviation of the Gaussian kernel in samples.
        Larger values → more smoothing.  Default 2.0 is intentionally
        light-handed to preserve local structure.
    """
    sigma = max(float(sigma), 0.1)
    values = gaussian_filter1d(ts.values.astype(float), sigma=sigma)
    return _as_series(values, ts.index, name="gaussian")


@register_filter("Savitzky-Golay")
def savgol_filter(ts: pd.Series, window: int = 11, polyorder: int = 2) -> pd.Series:
    n = len(ts)
    if n < 5:
        return ts.copy()
    window = _valid_odd_window(n, window, minimum=max(3, polyorder + 2))
    polyorder = min(polyorder, window - 1)
    values = signal.savgol_filter(ts.values, window_length=window, polyorder=polyorder)
    return _as_series(values, ts.index, name="savgol")


@register_filter("Butterworth Lowpass")
def butter_lowpass(ts: pd.Series, cutoff: float = 0.15, order: int = 3) -> pd.Series:
    cutoff = float(np.clip(cutoff, 1e-4, 0.99))
    b, a = signal.butter(order, cutoff, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))
    if len(ts) <= padlen:
        warnings.warn(
            f"butter_lowpass: series too short for filtfilt (need >{padlen}), "
            "falling back to moving average.",
            stacklevel=2,
        )
        return moving_average(ts, window=min(9, len(ts)))
    values = signal.filtfilt(b, a, ts.values)
    return _as_series(values, ts.index, name="butter")


@register_filter("Exponential Smoothing")
def exponential_smoothing(ts: pd.Series, alpha: float = 0.2) -> pd.Series:
    alpha = float(np.clip(alpha, 1e-3, 0.999))
    fit = SimpleExpSmoothing(ts.astype(float)).fit(
        smoothing_level=alpha, optimized=False
    )
    return _as_series(fit.fittedvalues.values, ts.index, name="exp_smooth")


@register_filter("FFT Denoising")
def fft_denoise(ts: pd.Series, cutoff: float = 0.15, soft: bool = False) -> pd.Series:
    """Frequency-domain denoising.

    Parameters
    ----------
    soft:
        If True, apply soft (gradual cosine) roll-off instead of a hard cutoff,
        which reduces Gibbs ringing artefacts.
    """
    cutoff = float(np.clip(cutoff, 1e-4, 0.49))
    fft = np.fft.fft(ts.values)
    freq = np.abs(np.fft.fftfreq(len(ts)))
    if soft:
        transition = cutoff * 0.2
        mask = np.where(
            freq <= cutoff - transition,
            1.0,
            np.where(
                freq >= cutoff + transition,
                0.0,
                0.5
                * (
                    1
                    + np.cos(np.pi * (freq - (cutoff - transition)) / (2 * transition))
                ),
            ),
        )
        fft *= mask
    else:
        fft[freq > cutoff] = 0
    return _as_series(np.real(np.fft.ifft(fft)), ts.index, name="fft")


# ---------------------------------------------------------------------------
# Wavelet denoising (pure-NumPy Haar, BayesShrink + garrote + cycle spinning)
# ---------------------------------------------------------------------------


def _garrote_threshold(c: np.ndarray, thr: float) -> np.ndarray:
    a = np.abs(c)
    return np.where(a <= thr, 0.0, c * (1.0 - (thr * thr) / (a * a + 1e-12)))


def _haar_dwt_multilevel(
    x: np.ndarray, levels: int
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    approx = np.asarray(x, dtype=float).copy()
    details, lengths = [], []
    for _ in range(levels):
        lengths.append(len(approx))
        if len(approx) < 2:
            break
        if len(approx) % 2 == 1:
            approx = np.pad(approx, (0, 1), mode="edge")
        a = (approx[0::2] + approx[1::2]) / np.sqrt(2.0)
        d = (approx[0::2] - approx[1::2]) / np.sqrt(2.0)
        details.append(d)
        approx = a
    return approx, details, lengths


def _haar_idwt_multilevel(
    approx: np.ndarray,
    details: list[np.ndarray],
    lengths: list[int],
) -> np.ndarray:
    rec = np.asarray(approx, dtype=float).copy()
    for d, orig_len in zip(reversed(details), reversed(lengths)):
        up = np.empty(2 * len(d), dtype=float)
        up[0::2] = (rec + d) / np.sqrt(2.0)
        up[1::2] = (rec - d) / np.sqrt(2.0)
        rec = up[:orig_len]
    return rec


def _haar_wavelet_denoise_once(x: np.ndarray, levels: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return x.copy()
    max_levels = max(1, int(np.floor(np.log2(len(x)))) - 1)
    levels = int(np.clip(levels, 1, max_levels))

    approx, details, lengths = _haar_dwt_multilevel(x, levels=levels)
    if not details:
        return x.copy()

    sigma = np.median(np.abs(details[0])) / 0.6745 + 1e-12
    den_details = []
    for j, d in enumerate(details):
        sigma_y = np.std(d) + 1e-12
        sigma_x = np.sqrt(max(sigma_y**2 - sigma**2, 0.0))
        bayes_thr = np.inf if sigma_x < 1e-12 else (sigma**2) / sigma_x
        universal_thr = sigma * np.sqrt(2.0 * np.log(d.size + 1.0))
        thr = min(bayes_thr, universal_thr) * (0.92**j)
        den_details.append(_garrote_threshold(d, thr))

    return _haar_idwt_multilevel(approx, den_details, lengths)


@register_filter("Wavelet (Bayes+Garrote)")
def wavelet_denoise(ts: pd.Series, levels: int = 4, cycle_spins: int = 4) -> pd.Series:
    x = ts.values.astype(float)
    acc = np.zeros_like(x)
    for shift in range(max(1, int(cycle_spins))):
        xs = np.roll(x, shift)
        acc += np.roll(_haar_wavelet_denoise_once(xs, levels=levels), -shift)
    return _as_series(acc / cycle_spins, ts.index, name="wavelet")


# ---------------------------------------------------------------------------
# Total-variation denoising (1-D Chambolle projection)
# ---------------------------------------------------------------------------


@register_filter("TV Denoising")
def tv_denoise(
    ts: pd.Series, weight: float = 0.35, max_iter: int = 300, eps: float = 1e-5
) -> pd.Series:
    y = ts.values.astype(float)
    n = len(y)
    if n < 3:
        return ts.copy()

    p = np.zeros(n - 1, dtype=float)
    out = y.copy()
    tau = 0.25

    for _ in range(max_iter):
        out_prev = out.copy()
        p = p + (tau / max(weight, 1e-6)) * np.diff(out)
        p /= np.maximum(1.0, np.abs(p))

        div = np.empty(n, dtype=float)
        div[0], div[1:-1], div[-1] = p[0], p[1:] - p[:-1], -p[-1]

        out = y + weight * div
        if np.linalg.norm(out - out_prev) / np.sqrt(n) < eps:
            break

    return _as_series(out, ts.index, name="tv")


# ---------------------------------------------------------------------------
# LOWESS  (locally weighted scatterplot smoothing)
# ---------------------------------------------------------------------------


@register_filter("LOWESS")
def lowess_filter(ts: pd.Series, frac: float = 0.08, it: int = 2) -> pd.Series:
    """Locally weighted regression smoother (Cleveland 1979).

    Parameters
    ----------
    frac:
        Fraction of data used in each local regression window.
        Smaller → less smoothing, more local detail.
    it:
        Number of robustifying iterations (down-weights outliers).
    """
    x = np.arange(len(ts), dtype=float)
    y = ts.values.astype(float)
    smoothed = sm_lowess(y, x, frac=frac, it=it, return_sorted=False)
    return _as_series(smoothed, ts.index, name="lowess")


# ---------------------------------------------------------------------------
# Hodrick-Prescott filter
# ---------------------------------------------------------------------------


@register_filter("Hodrick-Prescott")
def hp_filter(ts: pd.Series, lamb: float = 1600.0) -> pd.Series:
    """Hodrick-Prescott trend extraction.

    Parameters
    ----------
    lamb:
        Smoothing parameter λ.  Standard values: 1600 (quarterly), 100 (annual),
        129600 (monthly).  Higher → smoother trend.
    """
    try:
        cycle, trend = hpfilter(ts.astype(float), lamb=lamb)
        return _as_series(trend.values, ts.index, name="hp")
    except Exception as exc:
        warnings.warn(
            f"HP filter failed ({exc}); falling back to moving average.", stacklevel=2
        )
        return moving_average(ts)


# ---------------------------------------------------------------------------
# SSA — Singular Spectrum Analysis  (pure NumPy)
# ---------------------------------------------------------------------------


def _ssa_reconstruct(x: np.ndarray, window: int, n_components: int) -> np.ndarray:
    """Reconstruct signal from the leading SSA eigentriples."""
    N = len(x)
    L = window
    K = N - L + 1

    # Trajectory (Hankel) matrix  (L × K)
    X_emb = np.stack([x[i : i + L] for i in range(K)], axis=1)

    U, s, Vt = np.linalg.svd(X_emb, full_matrices=False)

    n = min(n_components, len(s))
    X_rec = sum(s[i] * np.outer(U[:, i], Vt[i]) for i in range(n))

    # Anti-diagonal averaging (Hankelization → 1-D)
    result = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            result[i + j] += X_rec[i, j]
            counts[i + j] += 1
    return result / np.maximum(counts, 1)


@register_filter("SSA")
def ssa_filter(
    ts: pd.Series, window: int | None = None, n_components: int = 3
) -> pd.Series:
    """Singular Spectrum Analysis denoising.

    Embeds the series in a trajectory (Hankel) matrix, performs SVD, keeps
    the leading ``n_components`` eigentriples (trend + dominant oscillations),
    and reconstructs via anti-diagonal averaging.

    Parameters
    ----------
    window:
        Embedding window length L.  Defaults to ``len(ts) // 4``, clamped to
        [8, 200].
    n_components:
        Number of leading eigentriples to retain.
    """
    x = ts.values.astype(float)
    N = len(x)
    if N < 16:
        return ts.copy()
    if window is None:
        window = int(np.clip(N // 4, 8, 200))
    window = int(np.clip(window, 4, N // 2))
    n_components = max(1, int(n_components))
    reconstructed = _ssa_reconstruct(x, window=window, n_components=n_components)
    return _as_series(reconstructed, ts.index, name="ssa")


# ---------------------------------------------------------------------------
# Kalman RTS smoother  (1-D local-linear-trend model, pure NumPy)
# ---------------------------------------------------------------------------


def _estimate_noise_variances(x: np.ndarray) -> tuple[float, float]:
    """Heuristic variance init from data statistics."""
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
    """Rauch-Tung-Striebel (RTS) optimal smoother — local linear trend model.

    State: [level, slope]ᵀ.  Transition: level_{t+1} = level_t + slope_t + w,
    slope_{t+1} = slope_t + v.  Observation: y_t = level_t + e.

    Parameters
    ----------
    q:
        Process noise variance.  Auto-estimated from the series if None.
    r:
        Observation noise variance.  Auto-estimated if None.
    """
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


# ---------------------------------------------------------------------------
# Bilateral filter  (1-D, edge-preserving)
# ---------------------------------------------------------------------------


@register_filter("Bilateral")
def bilateral_filter(
    ts: pd.Series,
    sigma_t: float = 5.0,
    sigma_v: float | None = None,
) -> pd.Series:
    """Edge-preserving 1-D bilateral filter.

    Combines a Gaussian spatial (time-domain) kernel with a Gaussian range
    (value-domain) kernel.  Unlike a pure Gaussian, it suppresses noise while
    preserving abrupt level shifts and local extrema.

    Parameters
    ----------
    sigma_t:
        Temporal kernel width in samples.
    sigma_v:
        Value-range kernel width.  Defaults to the MAD-based noise estimate
        of the series (auto-scaled per signal).
    """
    y = ts.values.astype(float)
    N = len(y)

    if sigma_v is None:
        sigma_v = float(np.median(np.abs(np.diff(y))) / 0.6745) + 1e-6
    sigma_v = max(float(sigma_v), 1e-6)
    sigma_t = max(float(sigma_t), 0.5)

    half_win = int(np.ceil(3 * sigma_t))
    result = np.empty(N, dtype=float)

    for i in range(N):
        lo = max(0, i - half_win)
        hi = min(N, i + half_win + 1)
        idx = np.arange(lo, hi)
        w_t = np.exp(-0.5 * ((idx - i) / sigma_t) ** 2)
        w_v = np.exp(-0.5 * ((y[lo:hi] - y[i]) / sigma_v) ** 2)
        w = w_t * w_v
        result[i] = np.dot(w, y[lo:hi]) / (w.sum() + 1e-12)

    return _as_series(result, ts.index, name="bilateral")


# ---------------------------------------------------------------------------
# EMD + VMD baseline  (PyEMD + vmdpy)
# ---------------------------------------------------------------------------


@register_filter("EMD+VMD Baseline")
def emd_vmd_baseline(
    ts: pd.Series,
    max_imfs: int = 4,
    K: int = 3,
    alpha: float = 1500.0,
    tau: float = 0.0,
    tol: float = 1e-7,
) -> pd.Series:
    """Two-stage EMD → VMD denoising filter.

    Stage 1 — EMD (PyEMD):
        Decompose the mean-centred signal into IMFs, discard the highest-
        frequency IMF (index 0), and sum the remainder as a coarse
        reconstruction.  Falls back to a moving average if EMD fails.

    Stage 2 — VMD (vmdpy):
        Further decompose the EMD reconstruction into ``K`` variational
        modes, drop the highest-frequency mode, and add the grand mean back.
    """
    x = ts.values.astype(float)
    grand_mean = np.mean(x)
    x_centered = x - grand_mean

    # --- Stage 1: EMD ---
    try:
        emd = EMD()
        emd.MAX_ITERATION = 1000
        imfs = emd.emd(x_centered, max_imf=max_imfs)
        if imfs.ndim == 1:
            emd_recon = imfs.copy()
        elif imfs.shape[0] >= 2:
            emd_recon = np.sum(imfs[1:], axis=0)
        else:
            emd_recon = imfs[0].copy()
    except Exception as exc:
        warnings.warn(
            f"PyEMD decomposition failed ({exc}); falling back to moving average.",
            stacklevel=2,
        )
        emd_recon = moving_average(_as_series(x_centered, ts.index), window=9).values

    # --- Stage 2: VMD ---
    try:
        u, _, omega = VMD(emd_recon, alpha, tau, K, DC=0, init=1, tol=tol)
        final_omega = omega[-1]
        highest_freq_idx = int(np.argmax(final_omega))
        keep_mask = np.ones(K, dtype=bool)
        keep_mask[highest_freq_idx] = False
        vmd_recon = np.sum(u[keep_mask], axis=0)
    except Exception as exc:
        warnings.warn(
            f"vmdpy VMD failed ({exc}); using EMD reconstruction directly.",
            stacklevel=2,
        )
        vmd_recon = emd_recon

    return _as_series(vmd_recon + grand_mean, ts.index, name="emd_vmd")


# ---------------------------------------------------------------------------
# CEEMDAN + VMD  (Complete Ensemble EMD with Adaptive Noise → VMD)
# ---------------------------------------------------------------------------


@register_filter("CEEMDAN+VMD", slow=True)
def ceemdan_vmd_filter(
    ts: pd.Series,
    trials: int = 50,
    epsilon: float = 0.005,
    K: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
) -> pd.Series:
    """Two-stage CEEMDAN → VMD denoising filter (current SOTA adaptive method).

    Stage 1 — CEEMDAN (PyEMD):
        Complete Ensemble EMD with Adaptive Noise.  More statistically
        consistent than plain EMD or EEMD — each trial adds a different
        realisation of white noise scaled to the residual, yielding nearly
        orthogonal, complete IMFs.  The finest IMF (highest frequency,
        index 0) is discarded as noise.

    Stage 2 — VMD (vmdpy):
        Variational Mode Decomposition on the CEEMDAN reconstruction to
        further separate residual noise from signal modes.

    Parameters
    ----------
    trials:
        Number of CEEMDAN ensemble members (more → lower variance, slower).
    epsilon:
        Noise std relative to signal std used in CEEMDAN.
    K:
        Number of VMD modes.
    alpha, tau, tol:
        VMD bandwidth, dual-ascent step, convergence tolerance.
    """
    x = ts.values.astype(float)
    grand_mean = np.mean(x)
    x_centered = x - grand_mean

    # --- Stage 1: CEEMDAN ---
    try:
        ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)
        ceemdan.noise_seed(_SEED)
        imfs = ceemdan(x_centered)
        if imfs.ndim == 1:
            ceemdan_recon = imfs.copy()
        elif imfs.shape[0] >= 2:
            ceemdan_recon = np.sum(imfs[1:], axis=0)
        else:
            ceemdan_recon = imfs[0].copy()
    except Exception as exc:
        warnings.warn(
            f"CEEMDAN failed ({exc}); falling back to EMD.",
            stacklevel=2,
        )
        try:
            emd = EMD()
            imfs = emd.emd(x_centered)
            ceemdan_recon = (
                np.sum(imfs[1:], axis=0) if imfs.shape[0] >= 2 else imfs[0].copy()
            )
        except Exception:
            ceemdan_recon = moving_average(
                _as_series(x_centered, ts.index), window=9
            ).values

    # --- Stage 2: VMD ---
    try:
        u, _, omega = VMD(ceemdan_recon, alpha, tau, K, DC=0, init=1, tol=tol)
        final_omega = omega[-1]
        highest_freq_idx = int(np.argmax(final_omega))
        keep_mask = np.ones(K, dtype=bool)
        keep_mask[highest_freq_idx] = False
        vmd_recon = np.sum(u[keep_mask], axis=0)
    except Exception as exc:
        warnings.warn(
            f"VMD stage failed ({exc}); using CEEMDAN reconstruction directly.",
            stacklevel=2,
        )
        vmd_recon = ceemdan_recon

    return _as_series(vmd_recon + grand_mean, ts.index, name="ceemdan_vmd")


# ---------------------------------------------------------------------------
# Denoising autoencoder (skipped when fast=True)
# ---------------------------------------------------------------------------


class _DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        hidden_size = max(16, int(hidden_size))
        bottleneck = max(8, hidden_size // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _build_windows(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, (half, half), mode="reflect")
    return np.stack([padded[i : i + window] for i in range(len(values))], axis=0)


@register_filter("Denoising Autoencoder", slow=True)
def train_dae(
    ts: pd.Series,
    window: int = 21,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    noise_std: float = 0.15,
) -> pd.Series:
    """Train a small denoising autoencoder on sliding windows."""
    rng_state = torch.get_rng_state()
    torch.manual_seed(_SEED)

    values = ts.values.astype(np.float32)
    window = _valid_odd_window(len(values), window, minimum=5)
    if window < 5:
        return ts.copy()

    x_clean = _build_windows(values, window)
    x_clean_t = torch.tensor(x_clean, dtype=torch.float32)
    x_noisy_t = x_clean_t + noise_std * torch.randn_like(x_clean_t)

    model = _DenoisingAutoencoder(input_size=window, hidden_size=min(64, window * 3))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = x_clean_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = criterion(model(x_noisy_t[idx]), x_clean_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        center = window // 2
        denoised = model(x_clean_t).numpy()[:, center]

    torch.set_rng_state(rng_state)
    return _as_series(denoised, ts.index, name="dae")


# ---------------------------------------------------------------------------
# Metrics and scoring
# ---------------------------------------------------------------------------


@dataclass
class ScoringWeights:
    """Weights for the composite filter score (must sum to 1)."""

    fidelity_mse: float = 0.40  # low MSE vs original is good
    roughness: float = 0.10  # low roughness — reduced to avoid over-smoothing bias
    residual_autocorr: float = 0.25  # low residual autocorrelation is good
    derivative_corr: float = (
        0.25  # high derivative correlation — increased for shape preservation
    )

    def __post_init__(self) -> None:
        total = (
            self.fidelity_mse
            + self.roughness
            + self.residual_autocorr
            + self.derivative_corr
        )
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"ScoringWeights must sum to 1, got {total:.4f}")


def filter_metrics(
    denoised: pd.Series, original: pd.Series, max_lag: int = 20
) -> dict[str, float]:
    """Compute four quality metrics for a denoised series.

    Returns a dict with keys:
        fidelity_mse       — lower is better
        roughness          — lower is better (std of first-differences)
        residual_autocorr  — lower is better (mean |autocorr| of residual)
        derivative_corr    — higher is better (shape preservation)
    """
    d = denoised.values.astype(float)
    o = original.values.astype(float)
    residual = o - d

    upper = min(max_lag, len(residual) - 1)
    resid_ac = (
        float(np.mean([abs(_autocorr(residual, lag)) for lag in range(1, upper + 1)]))
        if upper >= 1
        else 0.0
    )

    return {
        "fidelity_mse": float(np.mean((d - o) ** 2)),
        "roughness": float(np.std(np.diff(d))),
        "residual_autocorr": resid_ac,
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


def _compute_scores(mdf: pd.DataFrame, weights: ScoringWeights) -> pd.Series:
    fid_n = _rank_normalize(mdf["fidelity_mse"].values)
    rough_n = _rank_normalize(mdf["roughness"].values)
    resid_n = _rank_normalize(mdf["residual_autocorr"].values)
    deriv_n = _rank_normalize(mdf["derivative_corr"].values, invert=True)

    score = (
        weights.fidelity_mse * fid_n
        + weights.roughness * rough_n
        + weights.residual_autocorr * resid_n
        + weights.derivative_corr * deriv_n
    )
    return pd.Series(score, index=mdf.index, name="score")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def auto_filter(
    ts: pd.Series,
    fast: bool = False,
    weights: ScoringWeights | None = None,
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

    Returns
    -------
    best_name:
        Name of the top-ranked filter.
    best_series:
        Denoised series produced by the best filter.
    score_table:
        DataFrame with per-filter metrics and composite score, sorted ascending.
    """
    if weights is None:
        weights = ScoringWeights()

    active = {
        name: fn
        for name, fn in _FILTER_REGISTRY.items()
        if not (fast and name in _SLOW_FILTERS)
    }

    candidates: dict[str, pd.Series] = {}
    for name, fn in active.items():
        try:
            candidates[name] = fn(ts)
        except Exception as exc:
            warnings.warn(
                f"Filter '{name}' raised an exception and was skipped: {exc}",
                stacklevel=2,
            )

    if not candidates:
        raise RuntimeError("All filters failed. Cannot rank.")

    metrics_rows = {
        name: filter_metrics(series, ts) for name, series in candidates.items()
    }
    mdf = pd.DataFrame(metrics_rows).T
    mdf["score"] = _compute_scores(mdf, weights)
    mdf = mdf.sort_values("score")

    best_name = mdf.index[0]
    return best_name, candidates[best_name], mdf


# ---------------------------------------------------------------------------
# Unsupervised weight tuning via Optuna (Bayesian TPE)
# ---------------------------------------------------------------------------


def _unsupervised_proxy(
    winner: pd.Series,
    original: pd.Series,
    candidates: dict[str, pd.Series],
) -> float:
    """Independent quality proxy for the winning filter (lower is better).

    Two complementary criteria, each rank-normalised across all candidates:

    1. **Residual whiteness** — Ljung-Box cumulative Q-statistic at lag 20 on
       the residual.  A good denoiser leaves structureless (white) noise;
       lower Q means whiter residual.

    2. **Output roughness** — std of first-differences of the denoised signal.
       Rewards smooth outputs independent of fidelity.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lags = min(20, len(original) // 5)
    lb_stats: list[float] = []
    roughnesses: list[float] = []

    for s in candidates.values():
        resid = (original - s).values.astype(float)
        try:
            lb = acorr_ljungbox(resid, lags=lags, return_df=True)
            lb_stats.append(float(lb["lb_stat"].iloc[-1]))
        except Exception:
            lb_stats.append(float("nan"))
        roughnesses.append(float(np.std(np.diff(s.values.astype(float)))))

    names = list(candidates.keys())
    winner_name = winner.name
    if winner_name not in names:
        # Fallback: match by values
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
    rough_norm = _rank_norm_list(roughnesses)
    return lb_norm[winner_idx] + rough_norm[winner_idx]


def tune_weights(
    ts: pd.Series,
    n_trials: int = 100,
    fast: bool = True,
    seed: int = _SEED,
    verbose: bool = False,
) -> ScoringWeights:
    """Auto-tune :class:`ScoringWeights` using Optuna (Bayesian TPE, unsupervised).

    All filters are run **once** before the optimisation loop; Optuna only
    searches the 4-dimensional weight simplex.

    Objective (minimised)
    ---------------------
    An independent unsupervised proxy evaluated on whichever filter the trial
    weights select as best:

    * **Residual whiteness** — rank-normalised Ljung-Box Q-statistic on
      (original − denoised).
    * **Output roughness** — rank-normalised std of first-differences.

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
        name: filter_metrics(series, ts) for name, series in candidates.items()
    }
    mdf_base = pd.DataFrame(metrics_rows).T

    def objective(trial: optuna.Trial) -> float:
        logits = np.array([trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(4)])
        logits /= logits.sum()
        w = ScoringWeights(
            fidelity_mse=float(logits[0]),
            roughness=float(logits[1]),
            residual_autocorr=float(logits[2]),
            derivative_corr=float(logits[3]),
        )
        scores = _compute_scores(mdf_base, w)
        best_name = str(scores.idxmin())
        return _unsupervised_proxy(candidates[best_name], ts, candidates)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best = study.best_params
    raw = np.array([best[f"w{i}"] for i in range(4)], dtype=float)
    raw /= raw.sum()

    tuned = ScoringWeights(
        fidelity_mse=float(raw[0]),
        roughness=float(raw[1]),
        residual_autocorr=float(raw[2]),
        derivative_corr=float(raw[3]),
    )

    if verbose:
        print(f"\nTuned ScoringWeights (n_trials={n_trials}):")
        print(f"  fidelity_mse     = {tuned.fidelity_mse:.4f}")
        print(f"  roughness        = {tuned.roughness:.4f}")
        print(f"  residual_autocorr= {tuned.residual_autocorr:.4f}")
        print(f"  derivative_corr  = {tuned.derivative_corr:.4f}")
        print(f"  best trial value = {study.best_value:.6f}")

    return tuned


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
