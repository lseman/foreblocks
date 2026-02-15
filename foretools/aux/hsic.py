# hsic_sota.py
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------
# Optional Numba
# ----------------------------
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn): return fn
        return deco
    def prange(x): return range(x)

# ----------------------------
# Optional Joblib (matrix parallel)
# ----------------------------
try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except Exception:
    _JOBLIB = False

EPS = 1e-12

# =========================================================
# Numba kernels (hot paths)
# =========================================================

@njit(cache=True, fastmath=True, parallel=True)
def _pairwise_sq_dists_sym_nd(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    out = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        out[i, i] = 0.0
        for j in range(i + 1, n):
            s = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                s += diff * diff
            out[i, j] = s
            out[j, i] = s
    return out

@njit(cache=True, fastmath=True, parallel=True)
def _rbf_from_sq_dists_sym(D2: np.ndarray, sigma: float) -> np.ndarray:
    n = D2.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    denom = 2.0 * sigma * sigma + EPS
    for i in prange(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            v = math.exp(-D2[i, j] / denom)
            out[i, j] = v
            out[j, i] = v
    return out

@njit(cache=True, fastmath=True, parallel=True)
def _rbf_ard_from_sq_dists_sym(X: np.ndarray, sigma_vec: np.ndarray) -> np.ndarray:
    n, d = X.shape
    out = np.empty((n, n), dtype=np.float64)
    inv2 = np.empty(d, dtype=np.float64)
    for k in range(d):
        inv2[k] = 1.0 / (2.0 * sigma_vec[k] * sigma_vec[k] + EPS)
    for i in prange(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            s = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                s += diff * diff * inv2[k]
            v = math.exp(-s)
            out[i, j] = v
            out[j, i] = v
    return out

@njit(cache=True, fastmath=True, parallel=True)
def _linear_gram_sym(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    out = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        s = 0.0
        for k in range(d):
            s += X[i, k] * X[i, k]
        out[i, i] = s
        for j in range(i + 1, n):
            s = 0.0
            for k in range(d):
                s += X[i, k] * X[j, k]
            out[i, j] = s
            out[j, i] = s
    return out

@njit(cache=True, fastmath=True, parallel=True)
def _delta_gram_sym_1d(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            v = 1.0 if x[i] == x[j] else 0.0
            out[i, j] = v
            out[j, i] = v
    return out

@njit(cache=True, fastmath=True)
def _center_inplace(K: np.ndarray) -> None:
    n = K.shape[0]
    row_mean = np.empty(n, dtype=np.float64)
    col_mean = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += K[i, j]
        row_mean[i] = s / n
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += K[i, j]
        col_mean[j] = s / n
    g = 0.0
    for i in range(n):
        for j in range(n):
            g += K[i, j]
    g /= n * n
    for i in range(n):
        for j in range(n):
            K[i, j] = K[i, j] - row_mean[i] - col_mean[j] + g

@njit(cache=True, fastmath=True)
def _trace_prod(A: np.ndarray, B: np.ndarray) -> float:
    n = A.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += A[i, j] * B[j, i]
    return s

@njit(cache=True, fastmath=True)
def _hsic_unbiased_from_grams_numba(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    if n < 4:
        return np.nan
    Ku = K.copy()
    Lu = L.copy()
    for i in range(n):
        Ku[i, i] = 0.0
        Lu[i, i] = 0.0
    term1 = 0.0
    Ku_row = np.zeros(n, dtype=np.float64)
    Lu_row = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            term1 += Ku[i, j] * Lu[i, j]
            Ku_row[i] += Ku[i, j]
            Lu_row[i] += Lu[i, j]
    term3 = 0.0
    for i in range(n):
        term3 += Ku_row[i] * Lu_row[i]
    S_K = 0.0
    S_L = 0.0
    for i in range(n):
        S_K += Ku_row[i]
        S_L += Lu_row[i]
    n1, n2, n3 = n - 1.0, n - 2.0, n - 3.0
    if n3 <= 0:
        return np.nan
    val = (term1 + (S_K * S_L) / (n1 * n2) - 2.0 * term3 / n2) / (n * n3)
    return val if val > 0.0 and np.isfinite(val) else (0.0 if np.isfinite(val) else np.nan)

@njit(cache=True, fastmath=True)
def _hsic_biased_from_centered_perm(Kc: np.ndarray, Lc: np.ndarray, perm: np.ndarray) -> float:
    n = Kc.shape[0]
    s = 0.0
    for i in range(n):
        pi = perm[i]
        for j in range(n):
            pj = perm[j]
            s += Kc[i, j] * Lc[pi, pj]
    return s / ((n - 1.0) ** 2)

@njit(cache=True, fastmath=True)
def _hsic_unbiased_from_grams_perm_numba(K: np.ndarray, L: np.ndarray, perm: np.ndarray) -> float:
    """
    Unbiased HSIC under permutation, without materializing L[perm][:, perm].
    """
    n = K.shape[0]
    if n < 4:
        return np.nan

    # Ku row sums and total (diag-zeroed)
    Ku_row = np.zeros(n, dtype=np.float64)
    S_K = 0.0
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i != j:
                s += K[i, j]
        Ku_row[i] = s
        S_K += s

    # L row sums and diag (for fast permuted row-sum lookup)
    L_row = np.zeros(n, dtype=np.float64)
    L_diag = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += L[i, j]
        L_row[i] = s
        L_diag[i] = L[i, i]

    term1 = 0.0
    for i in range(n):
        pi = perm[i]
        for j in range(n):
            if i == j:
                continue
            pj = perm[j]
            term1 += K[i, j] * L[pi, pj]

    term3 = 0.0
    S_L = 0.0
    for i in range(n):
        pi = perm[i]
        Lu_row_i = L_row[pi] - L_diag[pi]
        term3 += Ku_row[i] * Lu_row_i
        S_L += Lu_row_i

    n1, n2, n3 = n - 1.0, n - 2.0, n - 3.0
    if n3 <= 0:
        return np.nan
    val = (term1 + (S_K * S_L) / (n1 * n2) - 2.0 * term3 / n2) / (n * n3)
    return val if val > 0.0 and np.isfinite(val) else (0.0 if np.isfinite(val) else np.nan)

# =========================================================
# Utilities (NumPy/compat)
# =========================================================

def _as_dtype(x: np.ndarray, prefer_float32: bool) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    # Use np.array(..., copy=False) for NumPy <=1.23 compat
    return np.array(x, dtype=(np.float32 if prefer_float32 else np.float64), copy=False)

def _ensure_1d_or_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    return a.reshape(a.shape[0], -1)

def _nan_filter_two(X: np.ndarray, Y: np.ndarray, warn: bool = True):
    X = _ensure_1d_or_2d(np.asarray(X))
    Y = _ensure_1d_or_2d(np.asarray(Y))
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if warn and mask.sum() < len(mask):
        warnings.warn(f"Removed {len(mask) - int(mask.sum())} rows with NaN/inf.")
    return X[mask], Y[mask]

def _center_numpy_inplace(K: np.ndarray) -> None:
    # stable centering: Kc = K - row_mean - col_mean + g
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    g = K.mean()
    K -= row_mean
    K -= col_mean
    K += g

def _adaptive_sigma_selection(D2: np.ndarray, method: str = "median") -> float:
    positive = D2[D2 > 0]
    if positive.size == 0:
        return 1.0
    if method == "median":
        m = float(np.median(positive))
        sigma = math.sqrt(0.5 * m)
    elif method == "iqr":
        d = np.sqrt(positive)
        iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
        sigma = (iqr / 1.34) if iqr > EPS else 1.0
    else:
        std = float(np.std(np.sqrt(positive)))
        n = positive.size
        sigma = 1.06 * std * (n ** (-1 / 5))
    if not np.isfinite(sigma) or sigma < EPS:
        sigma = 1.0
    p95 = np.percentile(np.sqrt(D2.ravel()), 95) if np.any(D2 > 0) else 1.0
    return float(min(sigma, p95 if p95 > EPS else 1.0))

def _ard_sigma(X: np.ndarray, method: str = "iqr") -> np.ndarray:
    X = _ensure_1d_or_2d(np.asarray(X))
    d = X.shape[1]
    s = np.empty(d, dtype=np.float64)
    for j in range(d):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            s[j] = 1.0
            continue
        if method == "iqr":
            q75, q25 = np.percentile(col, [75, 25])
            iqr = max(q75 - q25, EPS)
            s[j] = iqr / 1.34
        else:
            s[j] = np.std(col) + EPS
    s[~np.isfinite(s)] = 1.0
    return s

# =========================================================
# Null distribution helpers (moment matching)
# =========================================================

def _top_eigs_randomized(Kc: np.ndarray, q: int = 64, oversample: int = 16, n_iter: int = 2) -> np.ndarray:
    n = Kc.shape[0]
    l = min(q + oversample, n)
    rng = np.random.default_rng(0)
    G = rng.standard_normal((n, l))
    Y = Kc @ G
    for _ in range(n_iter):
        Y = Kc @ (Kc @ Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ (Kc @ Q) / n
    w = np.linalg.eigvalsh(B)
    w = np.maximum(w, 0.0)
    return np.sort(w)[::-1]

def _spectrum_based_moments(K: np.ndarray, L: np.ndarray) -> Tuple[float, float, float, float]:
    n = K.shape[0]
    Kc = K.copy(); Lc = L.copy()
    _center_numpy_inplace(Kc); _center_numpy_inplace(Lc)
    Kc_scaled = Kc / n
    Lc_scaled = Lc / n
    try:
        if n > 3000:
            eig_K = _top_eigs_randomized(Kc_scaled, q=min(100, n // 10))
            eig_L = _top_eigs_randomized(Lc_scaled, q=min(100, n // 10))
        else:
            eig_K = np.linalg.eigvalsh(Kc_scaled).real
            eig_L = np.linalg.eigvalsh(Lc_scaled).real
        eig_K = np.maximum(eig_K, 0)
        eig_L = np.maximum(eig_L, 0)
        mu1 = float(np.sum(eig_K) * np.sum(eig_L))
        # Using outer powers is more stable than nested sums
        kron = np.outer(eig_K, eig_L).ravel()
        mu2 = float(2.0 * np.sum(kron**2))
        mu3 = float(8.0 * np.sum(kron**3))
        mu4 = float(48.0 * np.sum(kron**4))
    except (np.linalg.LinAlgError, ValueError):
        tr_K = float(np.trace(Kc_scaled)); tr_L = float(np.trace(Lc_scaled))
        mu1 = tr_K * tr_L
        K_f = float(np.sum(Kc_scaled * Kc_scaled)); L_f = float(np.sum(Lc_scaled * Lc_scaled))
        mu2 = 2.0 * K_f * L_f
        mu3 = 8.0 * mu1**1.5
        mu4 = 48.0 * mu1**2
    return mu1, max(mu2, 1e-15), max(mu3, 1e-15), max(mu4, 1e-15)

def _cornish_fisher_correction(z: float, skew: float, kurt: float) -> float:
    if abs(skew) < 1e-2 and abs(kurt - 3) < 1e-2:
        return z
    cf2 = (z*z - 1.0) * skew / 6.0
    cf3 = (z*z*z - 3.0*z) * (kurt - 3.0) / 24.0
    cf4 = (2.0*z*z*z - 5.0*z) * (skew*skew) / 36.0
    return float(np.clip(z + cf2 + cf3 - cf4, -10.0, 10.0))

def _adaptive_null_method(n: int, eig_K: np.ndarray, eig_L: np.ndarray) -> str:
    if n < 100:
        return "permutation"
    eig_K_pos = eig_K[eig_K > 1e-10]; eig_L_pos = eig_L[eig_L > 1e-10]
    if eig_K_pos.size == 0 or eig_L_pos.size == 0:
        return "permutation"
    min_eff_rank = min(eig_K_pos.size, eig_L_pos.size)
    if min_eff_rank < 10:
        return "gamma"
    if min_eff_rank < 50 and n < 1000:
        return "chi2"  # treated as gaussian here with moments
    if n > 1000 and min_eff_rank > 50:
        return "gaussian"
    return "gamma"

# =========================================================
# Public API
# =========================================================

KernelType   = Literal["auto", "rbf", "linear", "delta", "mixed", "precomputed"]
EstimatorType = Literal["biased", "unbiased", "block", "linear", "rff", "nystrom"]
BandwidthType = Literal["median", "iqr", "silverman"]

class HSIC:
    def __init__(
        self,
        kernel_x: KernelType = "rbf",
        kernel_y: KernelType = "rbf",
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        bandwidth_method: BandwidthType = "iqr",
        estimator: EstimatorType = "biased",
        normalize: bool = True,
        approx_m: int = 2048,         # for linear / Nyström
        rff_features: int = 512,      # for RFF
        prefer_float32: bool = False,
        use_numba: bool = True,
        random_state: Optional[int] = None,
        block_size: int = 1024,       # for block estimator
    ):
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.bandwidth_method = bandwidth_method
        self.estimator = estimator
        self.normalize = normalize
        self.approx_m = int(approx_m)
        self.rff_features = int(rff_features)
        self.prefer_float32 = bool(prefer_float32)
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE)
        self.random_state = random_state
        self.block_size = int(block_size)
        # diagnostics
        self._last_sigma_x: Optional[float] = None
        self._last_sigma_y: Optional[float] = None
        self._cat_mask_x: Optional[np.ndarray] = None
        self._cat_mask_y: Optional[np.ndarray] = None

        self.rng = np.random.default_rng(random_state)

    # ------------- Mixed-data masks (optional) -------------
    def set_mixed_mask_x(self, categorical_mask: np.ndarray):
        self._cat_mask_x = np.array(categorical_mask, dtype=bool, copy=False)

    def set_mixed_mask_y(self, categorical_mask: np.ndarray):
        self._cat_mask_y = np.array(categorical_mask, dtype=bool, copy=False)

    # ------------- Main score -------------
    def score(self, x: np.ndarray, y: np.ndarray, return_components: bool = False) -> Union[float, Tuple[float, dict]]:
        X, Y = _nan_filter_two(x, y)
        X = _as_dtype(X, self.prefer_float32)
        Y = _as_dtype(Y, self.prefer_float32)
        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError("x and y must have the same number of samples")
        if n < 5:
            res = np.nan
            return (res, {"error": "insufficient_data", "n": n}) if return_components else res

        # Fast estimators: linear / RFF / Nyström
        if self.estimator == "linear":
            val = self._lhsic(X, Y)
            comps = {"n": n, "estimator": "linear", "sigma_x": self._last_sigma_x, "sigma_y": self._last_sigma_y}
            return (val, comps) if return_components else val

        if self.estimator == "rff":
            val = self._rff_hsic(X, Y)
            comps = {"n": n, "estimator": "rff", "sigma_x": self._last_sigma_x, "sigma_y": self._last_sigma_y}
            return (val, comps) if return_components else val

        if self.estimator == "nystrom":
            val = self._nystrom_hsic(X, Y, m=self.approx_m)
            comps = {"n": n, "estimator": "nystrom", "m": self.approx_m}
            return (val, comps) if return_components else val

        # Kernel Gram + centering
        K, Kc, xx = self._gram_and_center(X, which="x")
        L, Lc, yy = self._gram_and_center(Y, which="y")

        if self.estimator == "unbiased":
            val = float(_hsic_unbiased_from_grams_numba(K, L)) if self.use_numba else float(self._hsic_unbiased_np(K, L))
            comps = {"n": n, "estimator": "unbiased", "raw": val}
            return (val, comps) if return_components else val

        if self.estimator == "block":
            val = self._hsic_block(Kc, Lc, xx, yy, block_size=self.block_size)
            comps = {"n": n, "estimator": "block", "block_size": self.block_size}
            return (val, comps) if return_components else val

        # biased (default)
        num = float(_trace_prod(Kc, Lc) / ((n - 1.0) ** 2))
        if not self.normalize:
            val = float(max(0.0, num))
        else:
            den = math.sqrt(max(xx, EPS) * max(yy, EPS)) + EPS
            val = float(np.clip(num / den, 0.0, 1.0))

        comps = {"n": n, "estimator": "biased", "sigma_x": self._last_sigma_x, "sigma_y": self._last_sigma_y, "normalization": self.normalize}
        return (val, comps) if return_components else val

    # ------------- P-value -------------
    def pvalue(self, x: np.ndarray, y: np.ndarray, B: int = 200, subsample: Optional[int] = None,
               method: str = "auto", use_corrections: bool = True) -> Tuple[float, float, dict]:
        rng = self.rng
        X, Y = _nan_filter_two(x, y)
        X = _as_dtype(X, self.prefer_float32)
        Y = _as_dtype(Y, self.prefer_float32)
        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError("x and y must have same length")

        if subsample is not None and n > subsample:
            idx = rng.choice(n, size=subsample, replace=False)
            X = X[idx]; Y = Y[idx]; n = X.shape[0]

        # Fast estimators: permutation p-value
        if self.estimator in ("linear", "rff", "nystrom"):
            obs = self.score(X, Y)
            if not np.isfinite(obs):
                return obs, np.nan, {"method": "failed", "n": n}
            null_vals = np.empty(B, dtype=float)
            for b in range(B):
                yp = rng.permutation(n)
                null_vals[b] = self.score(X, Y[yp])
            pval = float((np.sum(null_vals >= obs) + 1.0) / (B + 1.0))
            return obs, pval, {"method": "permutation_approx", "n": n,
                               "null_mean": float(null_vals.mean()), "null_std": float(null_vals.std(ddof=1))}

        K, Kc, xx = self._gram_and_center(X, "x")
        L, Lc, yy = self._gram_and_center(Y, "y")

        # observed statistic
        if self.estimator == "unbiased":
            obs = float(_hsic_unbiased_from_grams_numba(K, L)) if self.use_numba else float(self._hsic_unbiased_np(K, L))
        elif self.estimator == "block":
            obs = self._hsic_block(Kc, Lc, xx, yy, block_size=self.block_size)
        else:
            obs = self._normalized_from_centered(Kc, Lc, xx, yy)

        mu1, mu2, mu3, mu4 = _spectrum_based_moments(K, L)
        skewness = mu3 / (mu2**1.5) if mu2 > 0 else 0.0
        kurtosis = mu4 / (mu2**2) if mu2 > 0 else 3.0

        # choose null
        if method == "auto":
            try:
                eig_K = np.linalg.eigvalsh(Kc / n).real if n <= 2000 else _top_eigs_randomized(Kc / n)
                eig_L = np.linalg.eigvalsh(Lc / n).real if n <= 2000 else _top_eigs_randomized(Lc / n)
                method = _adaptive_null_method(n, eig_K, eig_L)
            except Exception:
                method = "gamma"

        diagnostics = {"method": method, "n": n, "mu1": mu1, "mu2": mu2, "mu3": mu3, "mu4": mu4,
                       "skewness": skewness, "kurtosis": kurtosis, "estimator": self.estimator}

        # Parametric null approximations are derived for unnormalized HSIC moments.
        # If normalized score is requested, use permutation for calibrated p-values.
        if self.normalize and self.estimator in ("biased", "block") and method != "permutation":
            method = "permutation"
            diagnostics["method"] = method
            diagnostics["method_override"] = "normalized_hsic_requires_permutation"

        if method in ("gaussian", "chi2", "gaussian_corrected"):
            if mu2 <= 0:
                return obs, np.nan, diagnostics
            z = (obs - mu1) / math.sqrt(mu2)
            if method == "gaussian_corrected" or (use_corrections and (abs(skewness) > 0.1 or abs(kurtosis - 3) > 0.5)):
                zc = _cornish_fisher_correction(z, skewness, kurtosis)
                pval = 1.0 - stats.norm.cdf(zc)
                diagnostics.update({"correction": "cornish_fisher", "z_original": z, "z_corrected": zc})
            else:
                pval = 1.0 - stats.norm.cdf(z)
                diagnostics["correction"] = "none"

        elif method == "permutation":
            null_vals = np.empty(B, dtype=float)
            if self.estimator == "unbiased":
                for b in range(B):
                    perm = rng.permutation(n).astype(np.int64)
                    if self.use_numba:
                        null_vals[b] = float(
                            _hsic_unbiased_from_grams_perm_numba(K, L, perm)
                        )
                    else:
                        Lp = L[perm][:, perm]
                        null_vals[b] = float(self._hsic_unbiased_np(K, Lp))
            else:
                for b in range(B):
                    perm = rng.permutation(n).astype(np.int64)
                    val = _hsic_biased_from_centered_perm(Kc, Lc, perm)
                    if self.normalize:
                        val = float(np.clip(val / (math.sqrt(max(xx, EPS) * max(yy, EPS)) + EPS), 0.0, 1.0))
                    null_vals[b] = val
            pval = float((np.sum(null_vals >= obs) + 1.0) / (B + 1.0))
            diagnostics.update({"null_mean": float(null_vals.mean()), "null_std": float(null_vals.std(ddof=1))})

        else:  # gamma moment match
            from scipy.stats import gamma
            shape = mu1**2 / mu2 if mu2 > 0 else 1.0
            scale = mu2 / mu1 if mu1 > 0 else 1.0
            pval = 1.0 - gamma.cdf(obs, a=shape, scale=scale)

        pval = float(np.clip(pval, 0.0, 1.0)) if np.isfinite(pval) else np.nan
        return obs, pval, diagnostics

    # ------------- Pairwise matrix -------------
    def matrix(self, df: pd.DataFrame, show_progress: bool = True, n_jobs: int = 1) -> pd.DataFrame:
        cols = df.select_dtypes(include=[np.number]).columns
        if len(cols) != len(df.columns):
            warnings.warn(f"Using only numeric columns: {len(cols)} of {len(df.columns)}")
        p = len(cols)
        M = np.eye(p, dtype=float)

        # Fast estimators path
        if self.estimator in ("linear", "rff", "nystrom"):
            raw = [_ensure_1d_or_2d(df[c].to_numpy()) for c in cols]
            def compute_pair(i, j): return self.score(raw[i], raw[j])
            pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
            if _JOBLIB and n_jobs != 1:
                vals = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(compute_pair)(i, j) for (i, j) in pairs)
                for (i, j), v in zip(pairs, vals):
                    M[i, j] = M[j, i] = v
            else:
                for k, (i, j) in enumerate(pairs, 1):
                    M[i, j] = M[j, i] = compute_pair(i, j)
                    if show_progress and k % max(1, len(pairs) // 10) == 0:
                        print(f"Progress: {k}/{len(pairs)} ({100*k/len(pairs):.1f}%)")
            return pd.DataFrame(M, index=cols, columns=cols)

        # Kernel path: precompute grams once per column
        grams = [self._gram_and_center(_ensure_1d_or_2d(df[c].to_numpy()), which="x") for c in cols]
        pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
        def compute_pair(i, j):
            Ki, Kci, xxi = grams[i]
            Kj, Kcj, xxj = grams[j]
            if self.estimator == "unbiased":
                return float(_hsic_unbiased_from_grams_numba(Ki, Kj)) if self.use_numba else float(self._hsic_unbiased_np(Ki, Kj))
            if self.estimator == "block":
                return self._hsic_block(Kci, Kcj, xxi, xxj, block_size=self.block_size)
            num = float(_trace_prod(Kci, Kcj) / ((Kci.shape[0] - 1.0) ** 2))
            if self.normalize:
                den = math.sqrt(max(xxi, EPS) * max(xxj, EPS)) + EPS
                return float(np.clip(num / den, 0.0, 1.0))
            return float(max(0.0, num))

        if _JOBLIB and n_jobs != 1:
            vals = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(compute_pair)(i, j) for (i, j) in pairs
            )
            for (i, j), v in zip(pairs, vals):
                M[i, j] = M[j, i] = v
        else:
            for k, (i, j) in enumerate(pairs, 1):
                v = compute_pair(i, j)
                M[i, j] = M[j, i] = v
                if show_progress and k % max(1, len(pairs) // 10) == 0:
                    print(f"Progress: {k}/{len(pairs)} ({100*k/len(pairs):.1f}%)")

        return pd.DataFrame(M, index=cols, columns=cols)

    # ------------- Building blocks -------------
    def _gram_and_center(self, A: np.ndarray, which: Literal["x", "y"]) -> Tuple[np.ndarray, np.ndarray, float]:
        kernel = self.kernel_x if which == "x" else self.kernel_y
        sigma  = self.sigma_x  if which == "x" else self.sigma_y
        cat_mask = self._cat_mask_x if which == "x" else self._cat_mask_y

        A = _as_dtype(_ensure_1d_or_2d(A), self.prefer_float32)
        n, d = A.shape

        if kernel == "auto":
            if d == 1:
                uniq = np.unique(A[np.isfinite(A)]).size
                kernel = "delta" if uniq < min(20, n // 10) else "rbf"
            else:
                kernel = "rbf"

        if kernel == "precomputed":
            K = np.array(A, dtype=np.float64, copy=False)
            if K.shape[0] != K.shape[1]:
                raise ValueError("precomputed kernel must be square")
            Kc = K.copy()
            _center_inplace(Kc) if self.use_numba else _center_numpy_inplace(Kc)
            xx = _trace_prod(Kc, Kc) / ((n - 1.0) ** 2)
            return K, Kc, float(xx)

        # Build kernel
        if kernel == "rbf":
            if d > 1 and sigma is None:
                svec = _ard_sigma(A, "iqr")
                if self.use_numba:
                    K = _rbf_ard_from_sq_dists_sym(A.astype(np.float64), svec.astype(np.float64))
                else:
                    denom = (svec[None, None, :] ** 2 + EPS)
                    K = np.exp(-0.5 * ((A[:, None, :] - A[None, :, :]) ** 2 / denom).sum(axis=2))
                sigma_eff = float(np.exp(np.mean(np.log(np.maximum(svec, EPS)))))

            else:
                if self.use_numba:
                    D2 = _pairwise_sq_dists_sym_nd(A.astype(np.float64))
                else:
                    sq = np.sum(A * A, axis=1, keepdims=True)
                    D2 = sq + sq.T - 2.0 * (A @ A.T); np.maximum(D2, 0.0, out=D2)
                sigma_eff = _adaptive_sigma_selection(D2.astype(np.float64), self.bandwidth_method) if sigma is None else float(sigma)
                K = _rbf_from_sq_dists_sym(D2.astype(np.float64), sigma_eff) if self.use_numba else np.exp(-D2 / (2.0 * sigma_eff * sigma_eff + EPS))
            if which == "x":
                self._last_sigma_x = sigma_eff
            else:
                self._last_sigma_y = sigma_eff

        elif kernel == "linear":
            K = _linear_gram_sym(A) if self.use_numba else (A @ A.T)

        elif kernel == "delta":
            if A.shape[1] != 1:
                raise ValueError("delta kernel expects 1D input")
            a1d = A.ravel()
            K = _delta_gram_sym_1d(a1d) if self.use_numba else (a1d[:, None] == a1d[None, :]).astype(float)

        elif kernel == "mixed":
            K = self._mixed_kernel(A.astype(np.float64, copy=False), sigma, cat_mask)

        else:
            raise ValueError(f"Unknown kernel '{kernel}'")

        # symmetrize & diag (if non-numba path)
        if not self.use_numba:
            K = 0.5 * (K + K.T)
            # Only unit-diagonal kernels should be forced to 1.
            if kernel in ("rbf", "delta", "mixed"):
                np.fill_diagonal(K, 1.0)

        if self.estimator == "unbiased":
            return K, K, 1.0  # unbiased uses off-diagonal only

        Kc = K.copy()
        _center_inplace(Kc) if self.use_numba else _center_numpy_inplace(Kc)
        xx = _trace_prod(Kc, Kc) / ((n - 1.0) ** 2)
        return K, Kc, float(xx)

    def _mixed_kernel(self, A: np.ndarray, sigma: Optional[float], cat_mask: Optional[np.ndarray]) -> np.ndarray:
        if cat_mask is None:
            return self._rbf_kernel(A, sigma)
        cat_mask = np.array(cat_mask, dtype=bool, copy=False)
        if A.shape[1] != cat_mask.size:
            raise ValueError("categorical_mask length must match number of features")
        Xn = A[:, ~cat_mask] if (~cat_mask).any() else None
        Xc = A[:, cat_mask] if cat_mask.any() else None

        Kn = None
        if Xn is not None and Xn.shape[1] > 0:
            Kn = self._rbf_kernel(Xn, sigma)
        Kc = None
        if Xc is not None and Xc.shape[1] > 0:
            eq = (Xc[:, None, :] == Xc[None, :, :]).all(axis=2).astype(np.float64)
            Kc = eq
        if Kn is None and Kc is None:
            return np.eye(A.shape[0], dtype=np.float64)
        if Kn is None:
            return Kc
        if Kc is None:
            return Kn
        return 0.5 * (Kn + Kc)

    def _rbf_kernel(self, A: np.ndarray, sigma: Optional[float]) -> np.ndarray:
        if self.use_numba:
            D2 = _pairwise_sq_dists_sym_nd(A.astype(np.float64))
        else:
            sq = np.sum(A * A, axis=1, keepdims=True)
            D2 = sq + sq.T - 2.0 * (A @ A.T)
            np.maximum(D2, 0.0, out=D2)
        sig = _adaptive_sigma_selection(D2, self.bandwidth_method) if sigma is None else float(sigma)
        return _rbf_from_sq_dists_sym(D2, sig) if self.use_numba else np.exp(-D2 / (2.0 * sig * sig + EPS))

    def _normalized_from_centered(self, Kc: np.ndarray, Lc: np.ndarray, xx: float, yy: float) -> float:
        n = Kc.shape[0]
        num = float(_trace_prod(Kc, Lc) / ((n - 1.0) ** 2))
        if not self.normalize:
            return max(0.0, num)
        den = math.sqrt(max(xx, EPS) * max(yy, EPS)) + EPS
        return float(np.clip(num / den, 0.0, 1.0))

    @staticmethod
    def _hsic_unbiased_np(K: np.ndarray, L: np.ndarray) -> float:
        n = K.shape[0]
        if n < 4:
            return np.nan
        Ku = K.copy(); np.fill_diagonal(Ku, 0.0)
        Lu = L.copy(); np.fill_diagonal(Lu, 0.0)
        term1 = float((Ku * Lu).sum())
        Ku_row = Ku.sum(axis=1); Lu_row = Lu.sum(axis=1)
        term3 = float((Ku_row * Lu_row).sum())
        S_K = float(Ku_row.sum()); S_L = float(Lu_row.sum())
        n1, n2, n3 = n - 1.0, n - 2.0, n - 3.0
        if n3 <= 0:
            return np.nan
        hsic_u = (term1 + (S_K * S_L) / (n1 * n2) - 2.0 * term3 / n2) / (n * n3)
        return float(max(0.0, hsic_u)) if np.isfinite(hsic_u) else np.nan

    # ---------- Linear / RFF / Nyström / Block ----------
    def _lhsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        # Linear-time estimator via paired differences (fast, rough)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        m = min(self.approx_m, max(1, n // 2))
        if m < 1:
            return np.nan

        sigx, sigy = self._last_sigma_x, self._last_sigma_y
        if sigx is None or sigy is None:
            idx = rng.choice(n, size=min(2048, n), replace=False)
            D2x = self._pairwise_sq_np(X[idx]); D2y = self._pairwise_sq_np(Y[idx])
            if sigx is None: sigx = _adaptive_sigma_selection(D2x, self.bandwidth_method)
            if sigy is None: sigy = _adaptive_sigma_selection(D2y, self.bandwidth_method)
            self._last_sigma_x, self._last_sigma_y = float(sigx), float(sigy)

        idx = rng.choice(n, size=2 * m, replace=False)
        A = idx[:m]; B = idx[m:]
        s = 0.0
        inv2x = 1.0 / (2.0 * sigx * sigx + EPS); inv2y = 1.0 / (2.0 * sigy * sigy + EPS)
        for i in range(m):
            dx = float(np.dot(X[A[i]] - X[B[i]], X[A[i]] - X[B[i]]))
            dy = float(np.dot(Y[A[i]] - Y[B[i]], Y[A[i]] - Y[B[i]]))
            k = math.exp(-dx * inv2x); l = math.exp(-dy * inv2y)
            s += (k - 1.0) * (l - 1.0)
        val = float(max(0.0, s / m))
        return val

    def _rff_hsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]; D = int(self.rff_features)
        if D < 1: raise ValueError("rff_features must be >= 1")

        sigx, sigy = self._last_sigma_x, self._last_sigma_y
        if sigx is None or sigy is None:
            idx = rng.choice(n, size=min(4096, n), replace=False)
            D2x = self._pairwise_sq_np(X[idx]); D2y = self._pairwise_sq_np(Y[idx])
            if sigx is None: sigx = _adaptive_sigma_selection(D2x, self.bandwidth_method)
            if sigy is None: sigy = _adaptive_sigma_selection(D2y, self.bandwidth_method)
            self._last_sigma_x, self._last_sigma_y = float(sigx), float(sigy)

        Zx = self._rff_features_nd(X.astype(np.float64, copy=False), float(sigx), D, rng)
        Zy = self._rff_features_nd(Y.astype(np.float64, copy=False), float(sigy), D, rng)
        num = self._cross_cov_energy(Zx, Zy)
        if not self.normalize: return float(max(0.0, num))
        den = math.sqrt(max(self._cov_energy(Zx), EPS) * max(self._cov_energy(Zy), EPS)) + EPS
        return float(np.clip(num / den, 0.0, 1.0))

    def _nystrom_hsic(self, X: np.ndarray, Y: np.ndarray, m: int = 512) -> float:
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]; m = min(max(1, m), n)
        idx = rng.choice(n, size=m, replace=False)
        Kx, _, _ = self._gram_and_center(X, which="x")
        Ky, _, _ = self._gram_and_center(Y, which="y")
        Kxx = Kx[np.ix_(idx, idx)]; Kxy = Kx[:, idx]
        Lyy = Ky[np.ix_(idx, idx)]; Lyx = Ky[:, idx]
        tau = 1e-6
        try:
            Cx = np.linalg.solve(Kxx + tau * np.eye(m), Kxy.T)
            Cy = np.linalg.solve(Lyy + tau * np.eye(m), Lyx.T)
        except np.linalg.LinAlgError:
            Cx = np.linalg.pinv(Kxx + tau * np.eye(m)) @ Kxy.T
            Cy = np.linalg.pinv(Lyy + tau * np.eye(m)) @ Lyx.T
        Phi = Cx.T; Psi = Cy.T
        Phi -= Phi.mean(axis=0, keepdims=True); Psi -= Psi.mean(axis=0, keepdims=True)
        num = self._cross_cov_energy(Phi, Psi)
        if not self.normalize: return float(max(0.0, num))
        den = math.sqrt(max(self._cov_energy(Phi), EPS) * max(self._cov_energy(Psi), EPS)) + EPS
        return float(np.clip(num / den, 0.0, 1.0))

    def _hsic_block(self, Kc: np.ndarray, Lc: np.ndarray, xx: float, yy: float, block_size: int = 1024) -> float:
        """Block U-statistic approximation for large n (memory-light, unbiased-ish)."""
        n = Kc.shape[0]
        if n <= block_size:
            return self._normalized_from_centered(Kc, Lc, xx, yy)
        rng = np.random.default_rng(self.random_state)
        B = max(1, n // block_size)
        accum = 0.0
        for _ in range(B):
            idx = rng.choice(n, size=block_size, replace=False)
            Kb = Kc[np.ix_(idx, idx)]; Lb = Lc[np.ix_(idx, idx)]
            xb = float(_trace_prod(Kb, Lb) / ((block_size - 1.0) ** 2))
            if self.normalize:
                # approximate denominators with sub-block energies
                xb_den = math.sqrt(max(float(_trace_prod(Kb, Kb) / ((block_size - 1.0) ** 2)), EPS) *
                                   max(float(_trace_prod(Lb, Lb) / ((block_size - 1.0) ** 2)), EPS)) + EPS
                xb = float(np.clip(xb / xb_den, 0.0, 1.0))
            else:
                xb = float(max(0.0, xb))
            accum += xb
        return float(accum / B)

    # ---------- helpers for fast estimators ----------
    @staticmethod
    def _pairwise_sq_np(A: np.ndarray) -> np.ndarray:
        sq = np.sum(A * A, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (A @ A.T)
        np.maximum(D2, 0.0, out=D2)
        return D2

    def _rff_features_nd(self, X: np.ndarray, sigma: float, D: int, rng: np.random.Generator) -> np.ndarray:
        n, d = X.shape
        # w ~ N(0, 1/sigma^2)
        w = rng.normal(0.0, 1.0 / (sigma + EPS), size=(d, D))
        b = rng.uniform(0.0, 2 * np.pi, size=(D,))
        XW = X @ w + b
        Zc = np.cos(XW); Zs = np.sin(XW)
        Z = np.concatenate([Zc, Zs], axis=1)
        Z *= math.sqrt(1.0 / D)
        return Z

    @staticmethod
    def _cov_energy(Z: np.ndarray) -> float:
        n = Z.shape[0]
        Zc = Z - Z.mean(axis=0, keepdims=True)
        C = (Zc.T @ Zc) / (n - 1.0)
        return float((C * C).sum())

    @staticmethod
    def _cross_cov_energy(Zx: np.ndarray, Zy: np.ndarray) -> float:
        n = Zx.shape[0]
        Zcx = Zx - Zx.mean(axis=0, keepdims=True)
        Zcy = Zy - Zy.mean(axis=0, keepdims=True)
        C = (Zcx.T @ Zcy) / (n - 1.0)
        return float((C * C).sum())


# =========================================================
# Convenience functions
# =========================================================

def hsic_test(x, y, alpha: float = 0.05, method: str = "auto", **kwargs) -> dict:
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if method == "auto":
        if n <= 2000:
            method = "permutation"
        elif n < 10000:
            method = "gamma"
        else:
            method = "gaussian"

    scorer = HSIC(**kwargs)
    score, pval, diagnostics = scorer.pvalue(x, y, method=method)
    return {
        "hsic": score,
        "pvalue": pval,
        "significant": (pval < alpha) if np.isfinite(pval) else False,
        "alpha": alpha,
        "reject_independence": (pval < alpha) if np.isfinite(pval) else False,
        "method": method,
        "n": n,
        "diagnostics": diagnostics
    }

def auto_hsic(x, y, **kwargs) -> float:
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n > 5000:
        kwargs.setdefault("estimator", "block")
        kwargs.setdefault("block_size", 1024)
    elif n < 200:
        kwargs.setdefault("estimator", "unbiased")
    # auto kernels
    if "kernel_x" not in kwargs:
        if x.ndim == 1 and len(np.unique(x)) < min(20, n // 10):
            kwargs["kernel_x"] = "delta"
        else:
            kwargs["kernel_x"] = "rbf"
    if "kernel_y" not in kwargs:
        if y.ndim == 1 and len(np.unique(y)) < min(20, n // 10):
            kwargs["kernel_y"] = "delta"
        else:
            kwargs["kernel_y"] = "rbf"
    return HSIC(**kwargs).score(x, y)


# =========================================================
# Conditional HSIC (cHSIC)
# =========================================================
class ConditionalHSIC:
    """
    Conditional HSIC: tests dependence between X and Y given Z.
    Zhang et al. (2011), NIPS: "Kernel-based Conditional Independence Test".
    
    Improved implementation with better numerical stability and projection methods.
    """

    def __init__(
        self,
        kernel_x: KernelType = "rbf",
        kernel_y: KernelType = "rbf",
        kernel_z: KernelType = "rbf",
        bandwidth_method: BandwidthType = "iqr",
        prefer_float32: bool = False,
        use_numba: bool = True,
        random_state: Optional[int] = None,
        regularization: float = 1e-3,
        projection_method: str = "standard",  # "standard" or "centering"
    ):
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.kernel_z = kernel_z
        self.bandwidth_method = bandwidth_method
        self.prefer_float32 = prefer_float32
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.random_state = random_state
        self.regularization = regularization
        self.projection_method = projection_method
        
        # Cache for last used sigmas
        self._last_sigma_x = None
        self._last_sigma_y = None
        self._last_sigma_z = None

    def _gram_centered(self, A: np.ndarray, kernel: str, sigma: Optional[float] = None):
        """Compute kernel Gram matrix and center it. Reuses HSIC optimizations."""
        A = _as_dtype(_ensure_1d_or_2d(A), self.prefer_float32)
        n, d = A.shape
        
        if kernel == "rbf":
            if self.use_numba:
                D2 = _pairwise_sq_dists_sym_nd(A.astype(np.float64))
                sigma_eff = _adaptive_sigma_selection(D2, self.bandwidth_method) if sigma is None else float(sigma)
                K = _rbf_from_sq_dists_sym(D2, sigma_eff)
            else:
                sq = np.sum(A * A, axis=1, keepdims=True)
                D2 = sq + sq.T - 2.0 * (A @ A.T)
                sigma_eff = _adaptive_sigma_selection(D2, self.bandwidth_method) if sigma is None else float(sigma)
                K = np.exp(-D2 / (2.0 * sigma_eff * sigma_eff + EPS))
                
        elif kernel == "linear":
            K = _linear_gram_sym(A) if self.use_numba else (A @ A.T)
            sigma_eff = None
            
        elif kernel == "delta":
            if A.shape[1] != 1:
                raise ValueError("delta kernel expects 1D input")
            a1d = A.ravel()
            K = _delta_gram_sym_1d(a1d) if self.use_numba else (a1d[:, None] == a1d[None, :]).astype(float)
            sigma_eff = None
            
        else:
            raise ValueError(f"Kernel {kernel} not implemented for ConditionalHSIC")

        # Center the kernel matrix
        Kc = K.copy()
        _center_inplace(Kc) if self.use_numba else _center_numpy_inplace(Kc)
        
        return K, Kc, sigma_eff

    def score(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """
        Compute conditional HSIC statistic between X and Y given Z.
        
        Returns:
            Conditional HSIC statistic (non-negative float)
        """
        X, Y = _nan_filter_two(x, y, warn=False)
        Z = _ensure_1d_or_2d(np.asarray(z))
        
        if not (X.shape[0] == Y.shape[0] == Z.shape[0]):
            raise ValueError("x, y, z must have same number of samples")

        n = X.shape[0]
        if n < 5:
            return np.nan

        # Handle degenerate case where Z has no variation
        if Z.ndim == 1:
            z_var = np.var(Z)
        else:
            z_var = np.mean(np.var(Z, axis=0))
            
        if z_var < 1e-12:
            # Z is constant, return unconditional HSIC
            warnings.warn("Z has no variation, computing unconditional HSIC")
            hsic_obj = HSIC(
                kernel_x=self.kernel_x, 
                kernel_y=self.kernel_y,
                bandwidth_method=self.bandwidth_method,
                prefer_float32=self.prefer_float32,
                use_numba=self.use_numba
            )
            return hsic_obj.score(X, Y)

        try:
            # Compute kernel matrices
            Kx, Kx_c, sigx = self._gram_centered(X, self.kernel_x, self._last_sigma_x)
            Ky, Ky_c, sigy = self._gram_centered(Y, self.kernel_y, self._last_sigma_y)
            Kz, Kz_c, sigz = self._gram_centered(Z, self.kernel_z, self._last_sigma_z)

            # Cache sigmas for next call
            self._last_sigma_x, self._last_sigma_y, self._last_sigma_z = sigx, sigy, sigz

            # Compute conditional HSIC
            stat = self._compute_conditional_hsic(Kx_c, Ky_c, Kz_c, n)
            return stat
            
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

    def _compute_conditional_hsic(self, Kx_c: np.ndarray, Ky_c: np.ndarray, 
                                 Kz_c: np.ndarray, n: int) -> float:
        """
        Compute conditional HSIC using improved projection method.
        
        Two methods available:
        1. "standard": R_z = I - K_z(K_z + τI)^{-1} (Zhang et al. 2011)
        2. "centering": Uses centering matrix with improved stability
        """
        if self.projection_method == "centering":
            return self._conditional_hsic_centering_method(Kx_c, Ky_c, Kz_c, n)
        else:
            return self._conditional_hsic_standard_method(Kx_c, Ky_c, Kz_c, n)

    def _conditional_hsic_standard_method(self, Kx_c: np.ndarray, Ky_c: np.ndarray, 
                                        Kz_c: np.ndarray, n: int) -> float:
        """Standard conditional HSIC following Zhang et al. (2011)"""
        # Adaptive regularization based on trace
        tau = max(self.regularization, 1e-6 * np.trace(Kz_c) / n)
        
        # Add centered kernel back to get original (but we work with centered)
        # For numerical stability, work with the original kernel for projection
        Kz_orig = Kz_c + np.ones((n, n)) / n  # Un-center approximately
        Kz_reg = Kz_orig + tau * np.eye(n)
        
        try:
            # Compute R_z = I - K_z(K_z + τI)^{-1}
            Kz_inv = np.linalg.solve(Kz_reg, np.eye(n))
            Rz = np.eye(n) - Kz_orig @ Kz_inv
            
            # Project centered kernels
            Kx_proj = Rz @ Kx_c @ Rz
            Ky_proj = Rz @ Ky_c @ Rz
            
            # Conditional HSIC statistic
            stat = np.trace(Kx_proj @ Ky_proj) / (n * (n - 1))
            return float(max(0.0, stat))
            
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            Kz_pinv = np.linalg.pinv(Kz_reg)
            Rz = np.eye(n) - Kz_orig @ Kz_pinv
            
            Kx_proj = Rz @ Kx_c @ Rz
            Ky_proj = Rz @ Ky_c @ Rz
            
            stat = np.trace(Kx_proj @ Ky_proj) / (n * (n - 1))
            return float(max(0.0, stat))

    def _conditional_hsic_centering_method(self, Kx_c: np.ndarray, Ky_c: np.ndarray, 
                                         Kz_c: np.ndarray, n: int) -> float:
        """Alternative method using centering matrix for improved stability"""
        # Centering matrix
        H = np.eye(n) - np.ones((n, n)) / n
        
        # Adaptive regularization
        tau = max(self.regularization, 1e-6 * np.trace(Kz_c) / n)
        
        # Regularized centered kernel
        Kz_reg = Kz_c + tau * np.eye(n)
        
        try:
            # Compute projection using centering: R_z = H - H*K_z*pinv(H*K_z*H)*H
            HKzH = H @ Kz_reg @ H
            HKzH_pinv = np.linalg.pinv(HKzH)
            Rz = H - HKzH @ HKzH_pinv
            
            # Project kernels
            Kx_proj = Rz @ Kx_c @ Rz
            Ky_proj = Rz @ Ky_c @ Rz
            
            # Re-center projected kernels
            _center_inplace(Kx_proj) if self.use_numba else _center_numpy_inplace(Kx_proj)
            _center_inplace(Ky_proj) if self.use_numba else _center_numpy_inplace(Ky_proj)
            
            # Conditional HSIC statistic
            stat = np.trace(Kx_proj @ Ky_proj) / (n * (n - 1))
            return float(max(0.0, stat))
            
        except np.linalg.LinAlgError:
            raise ValueError("Numerical instability in conditional HSIC computation")

    def pvalue(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
              B: int = 200, method: str = "permutation") -> Tuple[float, float]:
        """
        Compute p-value for conditional independence test.
        
        Args:
            x, y, z: Input arrays
            B: Number of permutations
            method: "permutation" (only supported method currently)
            
        Returns:
            (statistic, p_value)
        """
        if method != "permutation":
            raise ValueError("Only permutation method supported for conditional HSIC")
            
        rng = np.random.default_rng(self.random_state)
        
        # Observed statistic
        obs = self.score(x, y, z)
        if not np.isfinite(obs):
            return obs, np.nan

        # Permutation test - permute Y while keeping X and Z fixed
        null_vals = np.empty(B, dtype=float)
        n = len(y)
        
        for b in range(B):
            y_perm = rng.permutation(n)
            null_vals[b] = self.score(x, y[y_perm], z)

        # Remove any NaN values from null distribution
        valid_nulls = null_vals[np.isfinite(null_vals)]
        if len(valid_nulls) == 0:
            return obs, np.nan

        # Compute p-value
        pval = (np.sum(valid_nulls >= obs) + 1.0) / (len(valid_nulls) + 1.0)
        return obs, float(pval)

    def test(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
            alpha: float = 0.05, B: int = 200) -> dict:
        """
        Perform conditional independence test.
        
        Returns:
            Dictionary with test results including statistic, p-value, and decision
        """
        stat, pval = self.pvalue(x, y, z, B=B)
        
        return {
            "statistic": stat,
            "pvalue": pval,
            "alpha": alpha,
            "reject_independence": (pval < alpha) if np.isfinite(pval) else False,
            "significant": (pval < alpha) if np.isfinite(pval) else False,
            "method": "conditional_hsic_permutation",
            "n_permutations": B,
            "n_samples": len(x)
        }


# Convenience function
def conditional_hsic_test(x, y, z, alpha: float = 0.05, **kwargs) -> dict:
    """
    Convenience function for conditional independence testing.
    
    Tests: X ⊥ Y | Z (X independent of Y given Z)
    
    Args:
        x, y, z: Input variables
        alpha: Significance level
        **kwargs: Additional arguments for ConditionalHSIC
        
    Returns:
        Dictionary with test results
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    
    tester = ConditionalHSIC(**kwargs)
    return tester.test(x, y, z, alpha=alpha)
