"""foreblocks.ts_handler.auto_filter.filters.

This module implements the filters pieces for its package.
It belongs to the automatic signal filtering and denoising pipelines area of Foreblocks.
It exposes functions such as moving_average, gaussian_filter, savgol_filter, butter_lowpass.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal, sparse
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import spsolve
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from foreblocks.ts_handler.auto_filter.registry import register_filter


try:
    from statsmodels.tsa.seasonal import STL
except ImportError:  # pragma: no cover - statsmodels ships this in supported envs.
    STL = None


try:  # Real-wavelet basis for wavelet_denoise; falls back to pure-NumPy Haar.
    import pywt
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    pywt = None

try:  # Optional adaptive-decomposition dependency.
    from PyEMD import CEEMDAN, EMD
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    CEEMDAN = None
    EMD = None

try:
    from vmdpy import VMD
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    VMD = None

_SEED = 42

def _as_series(
    values: np.ndarray, index: pd.Index, name: str | None = None
) -> pd.Series:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) != len(index):
        raise ValueError(
            f"Filtered output length {len(arr)} != input length {len(index)}."
        )
    return pd.Series(arr, index=index, name=name)


def _resize_to_match_length(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == target_len:
        return arr
    if arr.size == 0:
        raise ValueError("Cannot resize an empty reconstruction.")
    if target_len <= 0:
        raise ValueError(f"Target length must be positive, got {target_len}.")
    if arr.size == 1:
        return np.full(target_len, float(arr[0]), dtype=float)

    source_grid = np.linspace(0.0, 1.0, num=arr.size)
    target_grid = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(target_grid, source_grid, arr)


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


def moving_average(ts: pd.Series, window: int = 7) -> pd.Series:
    """Centred moving average. Internal fallback only (not a ranked candidate)."""
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


# ---------------------------------------------------------------------------
# Wavelet denoising (real db/sym basis via pywt, BayesShrink + cycle spinning;
# falls back to a pure-NumPy Haar transform when pywt is unavailable)
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

    den_details = _bayes_garrote_thresholds(details)
    return _haar_idwt_multilevel(approx, den_details, lengths)


def _bayes_garrote_thresholds(
    details: list[np.ndarray],
) -> list[np.ndarray]:
    """BayesShrink threshold + non-negative garrote shrinkage, per detail level.

    Noise σ is estimated from the finest detail band via the robust MAD
    estimator (Donoho & Johnstone). Per band we take the smaller of the
    BayesShrink and universal thresholds, with a mild geometric relaxation at
    coarser scales so low-frequency structure is preserved.
    """
    if not details:
        return details
    sigma = np.median(np.abs(details[0])) / 0.6745 + 1e-12
    out = []
    for j, d in enumerate(details):
        sigma_y = np.std(d) + 1e-12
        sigma_x = np.sqrt(max(sigma_y**2 - sigma**2, 0.0))
        bayes_thr = np.inf if sigma_x < 1e-12 else (sigma**2) / sigma_x
        universal_thr = sigma * np.sqrt(2.0 * np.log(d.size + 1.0))
        thr = min(bayes_thr, universal_thr) * (0.92**j)
        out.append(_garrote_threshold(d, thr))
    return out


def _pywt_denoise_once(x: np.ndarray, wavelet: str, levels: int) -> np.ndarray:
    coeffs = pywt.wavedec(x, wavelet, mode="periodization", level=levels)
    approx, details = coeffs[0], list(coeffs[1:])
    details = _bayes_garrote_thresholds(details)
    rec = pywt.waverec([approx, *details], wavelet, mode="periodization")
    return np.asarray(rec, dtype=float)[: len(x)]


@register_filter("Wavelet (Bayes+Garrote)")
def wavelet_denoise(
    ts: pd.Series,
    levels: int = 3,
    cycle_spins: int = 4,
    wavelet: str = "sym8",
) -> pd.Series:
    """Translation-invariant wavelet denoising with BayesShrink + garrote.

    Uses a real Daubechies/symlet basis (``wavelet``, default ``sym8``) via
    pywt — far fewer staircase artefacts than a Haar basis — with cycle
    spinning (averaging over circular shifts) for translation invariance.
    Falls back to a pure-NumPy Haar transform if pywt is unavailable.

    Parameters
    ----------
    levels:
        Number of decomposition levels (clamped to the signal length).
    cycle_spins:
        Number of circular shifts averaged for translation invariance.
    wavelet:
        pywt wavelet name (e.g. ``"sym8"``, ``"db4"``). Ignored on the Haar
        fallback.
    """
    x = ts.values.astype(float)
    n = len(x)
    if n < 8:
        return ts.copy()
    max_levels = max(1, int(np.floor(np.log2(n))) - 1)
    levels = int(np.clip(levels, 1, max_levels))
    spins = max(1, int(cycle_spins))

    if pywt is not None:
        try:
            denoise_once = lambda v: _pywt_denoise_once(v, wavelet, levels)  # noqa: E731
        except Exception:
            denoise_once = lambda v: _haar_wavelet_denoise_once(v, levels=levels)  # noqa: E731
    else:
        denoise_once = lambda v: _haar_wavelet_denoise_once(v, levels=levels)  # noqa: E731

    acc = np.zeros_like(x)
    for shift in range(spins):
        xs = np.roll(x, shift)
        try:
            rec = denoise_once(xs)
        except Exception:
            rec = _haar_wavelet_denoise_once(xs, levels=levels)
        acc += np.roll(rec[:n], -shift)
    return _as_series(acc / spins, ts.index, name="wavelet")


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


@register_filter("Robust LOESS")
def robust_loess_filter(ts: pd.Series, frac: float = 0.10, it: int = 4) -> pd.Series:
    """Outlier-robust local regression (Cleveland's robust LOESS).

    Same local-regression machinery as :func:`lowess_filter` but with more
    robustifying iterations and a slightly wider default window, so bilinear
    weights down-weight outliers/spikes via Tukey's biweight. Prefer this over
    plain LOWESS when the series has heavy-tailed noise or transient spikes.
    """
    x = np.arange(len(ts), dtype=float)
    y = ts.values.astype(float)
    smoothed = sm_lowess(y, x, frac=frac, it=max(3, int(it)), return_sorted=False)
    return _as_series(smoothed, ts.index, name="robust_loess")


# ---------------------------------------------------------------------------
# Bayesian / patch-based smoothers
# ---------------------------------------------------------------------------


@register_filter("Gaussian Process")
def gaussian_process_smoother(
    ts: pd.Series,
    length_scale: float = 12.0,
    noise: float = 0.08,
    max_inducing: int = 256,
) -> pd.Series:
    """Sparse RBF Gaussian-process posterior mean smoother."""
    y = ts.values.astype(float)
    n = len(y)
    if n < 4:
        return ts.copy()

    m = min(max(8, int(max_inducing)), n)
    inducing_idx = np.unique(np.linspace(0, n - 1, m).round().astype(int))
    x_train = inducing_idx.astype(float)[:, None]
    y_train = y[inducing_idx]
    x_all = np.arange(n, dtype=float)[:, None]

    length_scale = max(float(length_scale), 1e-3)
    signal_var = max(float(np.var(y_train)), 1e-8)
    noise_var = max(float(noise), 1e-6) ** 2 * signal_var

    def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist2 = (a - b.T) ** 2
        return signal_var * np.exp(-0.5 * dist2 / (length_scale**2))

    K = rbf(x_train, x_train)
    K[np.diag_indices_from(K)] += noise_var + 1e-8
    Ks = rbf(x_all, x_train)
    centered = y_train - float(np.mean(y_train))
    try:
        alpha = np.linalg.solve(K, centered)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K, centered, rcond=None)[0]
    pred = float(np.mean(y_train)) + Ks @ alpha
    return _as_series(pred, ts.index, name="gp")


@register_filter("Non-local Means 1D")
def non_local_means_filter(
    ts: pd.Series,
    patch_radius: int = 3,
    search_radius: int = 24,
    h: float | None = None,
) -> pd.Series:
    """One-dimensional non-local means smoother."""
    y = ts.values.astype(float)
    n = len(y)
    if n < 4:
        return ts.copy()

    patch_radius = max(1, int(patch_radius))
    search_radius = max(patch_radius + 1, int(search_radius))
    padded = np.pad(y, (patch_radius, patch_radius), mode="reflect")
    patches = np.stack([
        padded[i : i + 2 * patch_radius + 1]
        for i in range(n)
    ])
    if h is None:
        h = float(np.median(np.abs(np.diff(y))) / 0.6745) + 1e-6
    h = max(float(h), 1e-6)

    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - search_radius)
        hi = min(n, i + search_radius + 1)
        d2 = np.mean((patches[lo:hi] - patches[i]) ** 2, axis=1)
        weights = np.exp(-d2 / (h * h))
        out[i] = np.dot(weights, y[lo:hi]) / max(float(weights.sum()), 1e-12)
    return _as_series(out, ts.index, name="nlm")


# ---------------------------------------------------------------------------
# Penalized least-squares smoothers (Whittaker-Eilers, L1 trend filter)
# ---------------------------------------------------------------------------


def _diff_matrix(n: int, order: int) -> sparse.csc_matrix:
    """Sparse order-``order`` finite-difference operator (n-order × n)."""
    D = sparse.eye(n, format="csc")
    for _ in range(order):
        D = D[1:] - D[:-1]
    return D.tocsc()


@register_filter("Whittaker-Eilers")
def whittaker_smoother(ts: pd.Series, lam: float = 1600.0, order: int = 2) -> pd.Series:
    """Whittaker-Eilers penalized-least-squares smoother (Eilers 2003).

    Solves  min_z ‖y − z‖² + λ‖Dᵏ z‖²,  with Dᵏ the k-th difference operator.
    A single banded sparse solve — fast and SOTA for trend/baseline smoothing;
    the modern replacement for spline and Hodrick-Prescott smoothing.

    Parameters
    ----------
    lam:
        Smoothing strength λ. Higher → smoother. (HP's λ=1600 is comparable.)
    order:
        Penalty difference order k. 2 (default) penalises curvature.
    """
    y = ts.values.astype(float)
    n = len(y)
    if n <= order + 1:
        return ts.copy()
    D = _diff_matrix(n, order)
    A = sparse.eye(n, format="csc") + float(max(lam, 0.0)) * (D.T @ D)
    z = spsolve(A.tocsc(), y)
    return _as_series(np.asarray(z, dtype=float), ts.index, name="whittaker")


@register_filter("Hodrick-Prescott")
def hp_filter(ts: pd.Series, lamb: float = 1600.0) -> pd.Series:
    """Hodrick-Prescott trend smoother."""
    return whittaker_smoother(ts, lam=lamb, order=2).rename("hp")


@register_filter("L1 Trend Filter")
def l1_trend_filter(
    ts: pd.Series, lam: float = 1.0, max_iter: int = 200, rho: float = 1.0
) -> pd.Series:
    """ℓ₁ trend filtering (Kim, Koh, Boyd & Gorinevsky 2009).

    Solves  min_z ½‖y − z‖² + λ‖D² z‖₁,  whose ℓ₁ curvature penalty yields a
    piecewise-linear trend with a small number of kinks — the principled
    generalisation of total-variation denoising to trends. Solved here with a
    light ADMM iteration (no external solver dependency).

    Parameters
    ----------
    lam:
        Regularisation weight λ. Higher → fewer kinks, straighter trend.
    max_iter, rho:
        ADMM iteration budget and penalty parameter.
    """
    y = ts.values.astype(float)
    n = len(y)
    if n < 4:
        return ts.copy()

    D = _diff_matrix(n, 2)  # (n-2) × n second-difference operator
    m = D.shape[0]
    rho = float(max(rho, 1e-6))
    eye_n = sparse.eye(n, format="csc")
    # z-update system: (I + ρ DᵀD) z = y + ρ Dᵀ(w − u)
    lhs = (eye_n + rho * (D.T @ D)).tocsc()
    Dz = np.zeros(m)
    w = np.zeros(m)
    u = np.zeros(m)
    thr = lam / rho
    z = y.copy()
    for _ in range(int(max_iter)):
        rhs = y + rho * (D.T @ (w - u))
        z = spsolve(lhs, rhs)
        Dz = D @ z
        # soft-threshold (ℓ₁ prox)
        a = Dz + u
        w = np.sign(a) * np.maximum(np.abs(a) - thr, 0.0)
        u = u + Dz - w
    return _as_series(np.asarray(z, dtype=float), ts.index, name="l1_trend")


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
# Seasonal/decomposition filters
# ---------------------------------------------------------------------------


@register_filter("STL Residual Wavelet")
def stl_residual_denoise(
    ts: pd.Series,
    period: int = 24,
    seasonal: int = 13,
    resid_levels: int = 2,
    cycle_spins: int = 3,
    robust: bool = True,
) -> pd.Series:
    """Seasonal-trend decomposition followed by residual wavelet denoising.

    This is a good fit for hourly generation/load signals: preserve the
    seasonal and trend components, then clean only the leftover innovations.
    """
    if STL is None:
        warnings.warn(
            "statsmodels STL is unavailable; falling back to wavelet denoising.",
            stacklevel=2,
        )
        return wavelet_denoise(ts, levels=resid_levels, cycle_spins=cycle_spins)

    y = ts.values.astype(float)
    if len(y) < max(8, period * 2):
        return wavelet_denoise(ts, levels=resid_levels, cycle_spins=cycle_spins)

    period = int(np.clip(period, 2, max(2, len(y) // 2)))
    seasonal = _valid_odd_window(len(y), int(seasonal), minimum=7)
    if seasonal <= period and seasonal % 2 == 0:
        seasonal += 1

    fit = STL(y, period=period, seasonal=seasonal, robust=robust).fit()
    resid = pd.Series(fit.resid, index=ts.index, name=ts.name)
    resid_clean = wavelet_denoise(
        resid,
        levels=resid_levels,
        cycle_spins=cycle_spins,
    ).values
    return _as_series(fit.trend + fit.seasonal + resid_clean, ts.index, name="stl_wavelet")


@register_filter("VMD", slow=True)
def vmd_filter(
    ts: pd.Series,
    K: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    drop_modes: int = 1,
) -> pd.Series:
    """Variational Mode Decomposition denoising.

    Drops the highest-frequency VMD mode(s) and reconstructs the rest. VMD is
    more adaptive than fixed-kernel smoothers, but still much cheaper than a
    CEEMDAN ensemble.
    """
    compat_module = sys.modules.get("foreblocks.ts_handler.auto_filter")
    vmd_fn = getattr(compat_module, "VMD", VMD)
    if vmd_fn is None:
        warnings.warn(
            "vmdpy is unavailable; falling back to wavelet denoising.",
            stacklevel=2,
        )
        return wavelet_denoise(ts)

    x = ts.values.astype(float)
    grand_mean = float(np.mean(x))
    x_centered = x - grand_mean
    K = int(np.clip(K, 2, min(8, max(2, len(ts) // 8))))
    drop_modes = int(np.clip(drop_modes, 1, K - 1))

    u, _, omega = vmd_fn(x_centered, alpha, tau, K, DC=0, init=1, tol=tol)
    final_omega = omega[-1]
    drop_idx = np.argsort(final_omega)[-drop_modes:]
    keep_mask = np.ones(K, dtype=bool)
    keep_mask[drop_idx] = False
    recon = _resize_to_match_length(np.sum(u[keep_mask], axis=0), len(ts))
    return _as_series(recon + grand_mean, ts.index, name="vmd")


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
    compat_module = sys.modules.get("foreblocks.ts_handler.auto_filter")
    ceemdan_cls = getattr(compat_module, "CEEMDAN", CEEMDAN)
    emd_cls = getattr(compat_module, "EMD", EMD)
    vmd_fn = getattr(compat_module, "VMD", VMD)

    # --- Stage 1: CEEMDAN ---
    try:
        ceemdan = ceemdan_cls(trials=trials, epsilon=epsilon)
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
            emd = emd_cls()
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
        u, _, omega = vmd_fn(ceemdan_recon, alpha, tau, K, DC=0, init=1, tol=tol)
        final_omega = omega[-1]
        highest_freq_idx = int(np.argmax(final_omega))
        keep_mask = np.ones(K, dtype=bool)
        keep_mask[highest_freq_idx] = False
        vmd_recon = _resize_to_match_length(np.sum(u[keep_mask], axis=0), len(ts))
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


class _VariationalAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, latent_size: int = 8) -> None:
        super().__init__()
        hidden_size = max(16, int(hidden_size))
        latent_size = max(2, int(latent_size))
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.decoder(self.mu(h))


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


@register_filter("Variational Autoencoder", slow=True)
def train_vae(
    ts: pd.Series,
    window: int = 25,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    noise_std: float = 0.12,
    beta: float = 0.02,
    latent_size: int = 8,
) -> pd.Series:
    """Train a compact VAE denoiser on sliding windows."""
    rng_state = torch.get_rng_state()
    torch.manual_seed(_SEED)

    values = ts.values.astype(np.float32)
    window = _valid_odd_window(len(values), window, minimum=7)
    if window < 7:
        return ts.copy()

    x_clean = _build_windows(values, window)
    x_clean_t = torch.tensor(x_clean, dtype=torch.float32)
    x_noisy_t = x_clean_t + noise_std * torch.randn_like(x_clean_t)

    model = _VariationalAutoencoder(
        input_size=window,
        hidden_size=min(96, window * 4),
        latent_size=latent_size,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = x_clean_t.shape[0]
    for _ in range(int(epochs)):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            recon, mu, logvar = model(x_noisy_t[idx])
            recon_loss = torch.mean((recon - x_clean_t[idx]) ** 2)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + float(beta) * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        center = window // 2
        denoised = model.reconstruct_mean(x_clean_t).numpy()[:, center]

    torch.set_rng_state(rng_state)
    return _as_series(denoised, ts.index, name="vae")
