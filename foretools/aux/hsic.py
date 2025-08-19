# hsic_sota.py
import math
import warnings
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ----------------------------
# Optional Numba
# ----------------------------
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    def prange(x):
        return range(x)


# ----------------------------
# DType helpers
# ----------------------------
def _as_dtype(x: np.ndarray, prefer_float32: bool) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if prefer_float32:
        return x.astype(np.float32, copy=False)
    return x.astype(np.float64, copy=False)


def _ensure_1d_or_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    elif a.ndim == 2:
        return a
    else:
        return a.reshape(a.shape[0], -1)


# ----------------------------
# Distance / kernel building blocks (Numba-ready)
# ----------------------------
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
def _pairwise_l1_dists_sym_nd(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    out = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        out[i, i] = 0.0
        for j in range(i + 1, n):
            s = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                s += abs(diff)
            out[i, j] = s
            out[j, i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _rbf_from_sq_dists_sym(D2: np.ndarray, sigma: float) -> np.ndarray:
    n = D2.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    denom = 2.0 * sigma * sigma + 1e-12
    for i in prange(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            v = math.exp(-D2[i, j] / denom)
            out[i, j] = v
            out[j, i] = v
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _laplacian_from_l1_dists_sym(D1: np.ndarray, sigma: float) -> np.ndarray:
    n = D1.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    denom = sigma + 1e-12
    for i in prange(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            v = math.exp(-D1[i, j] / denom)
            out[i, j] = v
            out[j, i] = v
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _matern_from_sq_dists_sym(D2: np.ndarray, sigma: float, nu: float) -> np.ndarray:
    n = D2.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    sqrt2nu = math.sqrt(2.0 * nu)
    inv_sigma = 1.0 / (sigma + 1e-12)
    for i in prange(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            r = math.sqrt(D2[i, j])
            arg = sqrt2nu * r * inv_sigma
            if nu == 0.5:
                v = math.exp(-arg)
            elif nu == 1.5:
                v = (1.0 + arg) * math.exp(-arg)
            elif nu == 2.5:
                v = (1.0 + arg + (arg * arg) / 3.0) * math.exp(-arg)
            else:
                # crude fallback
                v = (1.0 + arg) * math.exp(-arg)
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
def _poly_gram_sym(X: np.ndarray, degree: int, c0: float) -> np.ndarray:
    n, d = X.shape
    out = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        s = 0.0
        for k in range(d):
            s += X[i, k] * X[i, k]
        out[i, i] = (s + c0) ** degree
        for j in range(i + 1, n):
            s = 0.0
            for k in range(d):
                s += X[i, k] * X[j, k]
            v = (s + c0) ** degree
            out[i, j] = v
            out[j, i] = v
    return out


# For discrete 1D (delta kernel); for ND categorical you can encode beforehand
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


# ----------------------------
# Centering and trace
# ----------------------------
@njit(cache=True, fastmath=True)
def _center_inplace(K: np.ndarray) -> None:
    n = K.shape[0]
    row_mean = np.empty(n, dtype=np.float64)
    col_mean = np.empty(n, dtype=np.float64)
    # row means
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += K[i, j]
        row_mean[i] = s / n
    # col means
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += K[i, j]
        col_mean[j] = s / n
    # grand mean
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
def _block_hsic_fast(Kc: np.ndarray, Lc: np.ndarray, block: int = 1024) -> float:
    n = Kc.shape[0]
    if n <= block:
        return _trace_prod(Kc, Lc) / ((n - 1.0) ** 2)
    total = 0.0
    for i in range(0, n, block):
        ie = min(i + block, n)
        for j in range(0, n, block):
            je = min(j + block, n)
            s = 0.0
            for ii in range(i, ie):
                for jj in range(j, je):
                    s += Kc[ii, jj] * Lc[jj, ii]
            total += s
    return total / ((n - 1.0) ** 2)


# ----------------------------
# Bandwidth selection
# ----------------------------
def _adaptive_sigma_selection(
    D2: np.ndarray, method: str = "median", eps: float = 1e-12
) -> float:
    """Heuristics based on pairwise squared distances (can be ND)."""
    # Use only positive entries (ignore zero diag)
    positive = D2[D2 > 0]
    if positive.size == 0:
        return 1.0
    if method == "median":
        m = np.median(positive)
        sigma = math.sqrt(0.5 * m)
    elif method == "silverman":
        std = np.std(np.sqrt(positive))
        n = positive.size
        sigma = 1.06 * std * (n ** (-1 / 5))
    elif method == "scott":
        std = np.std(np.sqrt(positive))
        n = positive.size
        sigma = std * (n ** (-1 / 5))  # d≈1 heuristic
    elif method == "iqr":
        # robust: IQR of distances / 1.34 (≈σ for normal), then convert to σ of kernel
        d = np.sqrt(positive)
        iqr = np.percentile(d, 75) - np.percentile(d, 25)
        sigma = (iqr / 1.34) if iqr > eps else 1.0
    else:
        raise ValueError(f"Unknown bandwidth method: {method}")
    if (not np.isfinite(sigma)) or sigma < eps:
        sigma = 1.0
    p95 = np.percentile(np.sqrt(D2.ravel()), 95) if np.any(D2 > 0) else 1.0
    return float(min(sigma, p95 if p95 > eps else 1.0))


# ----------------------------
# Null approximation (Gaussian/Gamma)
# ----------------------------
def _center_numpy_inplace(K: np.ndarray) -> None:
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    g = K.mean()
    K -= row_mean
    K -= col_mean
    K += g


def _gaussian_hsic_params(K: np.ndarray, L: np.ndarray) -> Tuple[float, float]:
    n = K.shape[0]
    Kc = K.copy()
    Lc = L.copy()
    _center_numpy_inplace(Kc)
    _center_numpy_inplace(Lc)
    try:
        eig_K = np.linalg.eigvals(Kc / n).real
        eig_L = np.linalg.eigvals(Lc / n).real
        eig_K = np.maximum(eig_K, 0)
        eig_L = np.maximum(eig_L, 0)
        eig_K = np.sort(eig_K)[::-1]
        eig_L = np.sort(eig_L)[::-1]
        mean_h0 = np.sum(eig_K) * np.sum(eig_L) / n
        var_h0 = 2.0 * np.sum(np.outer(eig_K, eig_L) ** 2) / n
        var_h0 = max(var_h0, 1e-10)
    except np.linalg.LinAlgError:
        trace_K = float(np.trace(Kc))
        trace_L = float(np.trace(Lc))
        mean_h0 = trace_K * trace_L / (n * n)
        K_fsq = float(np.sum(Kc * Kc))
        L_fsq = float(np.sum(Lc * Lc))
        var_h0 = 2.0 * K_fsq * L_fsq / (n**4)
        var_h0 = max(var_h0, 1e-10)
    return float(mean_h0), float(var_h0)


def _gamma_hsic_params(K: np.ndarray, L: np.ndarray) -> Tuple[float, float]:
    mean_h0, var_h0 = _gaussian_hsic_params(K, L)
    if var_h0 <= 0 or mean_h0 <= 0:
        return 1.0, 1.0
    scale = var_h0 / mean_h0
    shape = mean_h0 * mean_h0 / var_h0
    return float(shape), float(scale)


# ----------------------------
# Fast permutation from centered kernels
# ----------------------------
@njit(cache=True, fastmath=True)
def _hsic_biased_from_centered_perm(
    Kc: np.ndarray, Lc: np.ndarray, perm: np.ndarray
) -> float:
    n = Kc.shape[0]
    s = 0.0
    for i in range(n):
        pi = perm[i]
        for j in range(n):
            pj = perm[j]
            s += Kc[i, j] * Lc[pi, pj]
    return s / ((n - 1.0) ** 2)


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
    return (
        val if val > 0.0 and np.isfinite(val) else (0.0 if np.isfinite(val) else np.nan)
    )


# ----------------------------
# RFF utilities
# ----------------------------
def _rff_features_nd(
    X: np.ndarray, sigma: float, D: int, rng: np.random.Generator
) -> np.ndarray:
    # X: (n,d); w ~ N(0, 1/sigma^2 I_d)
    n, d = X.shape
    w = rng.normal(0.0, 1.0 / (sigma + 1e-12), size=(d, D))
    b = rng.uniform(0.0, 2 * np.pi, size=(D,))
    XW = X @ w + b  # (n, D)
    Zc = np.cos(XW)
    Zs = np.sin(XW)
    Z = np.concatenate([Zc, Zs], axis=1)  # (n, 2D)
    Z *= math.sqrt(1.0 / D)  # sqrt(2/D) split across cos+sin
    return Z


def _cov_energy(Z: np.ndarray) -> float:
    n = Z.shape[0]
    Zc = Z - Z.mean(axis=0, keepdims=True)
    C = (Zc.T @ Zc) / (n - 1.0)
    return float((C * C).sum())


def _cross_cov_energy(Zx: np.ndarray, Zy: np.ndarray) -> float:
    n = Zx.shape[0]
    Zcx = Zx - Zx.mean(axis=0, keepdims=True)
    Zcy = Zy - Zy.mean(axis=0, keepdims=True)
    C = (Zcx.T @ Zcy) / (n - 1.0)
    return float((C * C).sum())


# ----------------------------
# HSIC main class
# ----------------------------
KernelType = Literal[
    "rbf", "laplacian", "linear", "poly", "delta", "matern", "precomputed"
]
EstimatorType = Literal["biased", "unbiased", "block", "linear", "rff"]
BandwidthType = Literal["median", "silverman", "scott", "iqr"]


class HSIC:
    """
    High-performance HSIC with ND inputs, RFF/linear approximations, and fast testing.

    Parameters
    ----------
    kernel_x, kernel_y : KernelType
    sigma_x, sigma_y : Optional[float]
    bandwidth_method : BandwidthType
    degree : int            (poly)
    c0 : float              (poly)
    nu : float              (matern: 0.5, 1.5, 2.5)
    estimator : EstimatorType
        'biased' | 'unbiased' | 'block' | 'linear' | 'rff'
    normalize : bool        (for biased/block/rff)
    approx_m : int          (linear estimator subsample size)
    rff_features : int      (number of base features D; resulting dim is 2D)
    prefer_float32 : bool   (halve memory, often faster; upcasts when needed)
    use_numba : bool
    random_state : Optional[int]
    """

    def __init__(
        self,
        kernel_x: KernelType = "rbf",
        kernel_y: KernelType = "rbf",
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        bandwidth_method: BandwidthType = "median",
        degree: int = 2,
        c0: float = 1.0,
        nu: float = 1.5,
        estimator: EstimatorType = "biased",
        normalize: bool = True,
        approx_m: int = 2048,
        rff_features: int = 512,
        prefer_float32: bool = False,
        use_numba: bool = True,
        random_state: Optional[int] = None,
    ):
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.bandwidth_method = bandwidth_method
        self.degree = degree
        self.c0 = c0
        self.nu = nu
        self.estimator = estimator
        self.normalize = normalize
        self.approx_m = approx_m
        self.rff_features = rff_features
        self.prefer_float32 = prefer_float32
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.random_state = random_state

        self._sigma_cache: Dict[Tuple[int, str, Optional[float], str], float] = {}
        self._last_sigma_x = None
        self._last_sigma_y = None

    # ---------- Public API ----------
    def score(
        self, x: np.ndarray, y: np.ndarray, return_components: bool = False
    ) -> Union[float, Tuple[float, dict]]:
        X = _ensure_1d_or_2d(x)
        Y = _ensure_1d_or_2d(y)
        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError("x and y must have the same number of samples")
        if n < 5:
            res = np.nan
            return (
                (res, {"error": "insufficient_data", "n": n})
                if return_components
                else res
            )

        # Approximations first (no full Gram)
        if self.estimator == "linear":
            val = self._lhsic(X, Y)
            comps = {
                "n": n,
                "estimator": "linear",
                "sigma_x": self._last_sigma_x,
                "sigma_y": self._last_sigma_y,
            }
            return (val, comps) if return_components else val

        if self.estimator == "rff":
            val = self._rff_hsic(X, Y)
            comps = {
                "n": n,
                "estimator": "rff",
                "sigma_x": self._last_sigma_x,
                "sigma_y": self._last_sigma_y,
            }
            return (val, comps) if return_components else val

        # Exact / near-exact HSIC
        K, Kc, xx = self._gram_and_center(X, which="x")
        L, Lc, yy = self._gram_and_center(Y, which="y")

        if self.estimator == "unbiased":
            val = (
                float(_hsic_unbiased_from_grams_numba(K, L))
                if self.use_numba
                else float(self._hsic_unbiased_np(K, L))
            )
            comps = {"n": n, "estimator": "unbiased", "raw": val}
            return (val, comps) if return_components else val

        # biased or block
        if self.estimator == "block":
            num = (
                _block_hsic_fast(Kc, Lc, block=min(1024, max(64, n // 4)))
                if self.use_numba
                else float(_trace_prod(Kc, Lc) / ((n - 1.0) ** 2))
            )
        else:
            num = float(_trace_prod(Kc, Lc) / ((n - 1.0) ** 2))

        if not self.normalize:
            val = float(max(0.0, num))
        else:
            den = math.sqrt(xx * yy) + 1e-12
            val = float(np.clip(num / den, 0.0, 1.0))

        comps = {
            "n": n,
            "estimator": self.estimator,
            "sigma_x": self._last_sigma_x,
            "sigma_y": self._last_sigma_y,
            "normalization": self.normalize,
        }
        return (val, comps) if return_components else val

    # Add inside HSIC class (or as a free function that uses its internals)
    def conditional_score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        lam: float = 1e-3,
        return_components: bool = False,
    ) -> Union[float, Tuple[float, dict]]:
        """
        Conditional HSIC via kernel projection (KCIT-style):
        cHSIC(X,Y|Z) = HSIC( (I-Pz) Kx (I-Pz), (I-Pz) Ky (I-Pz) )
        where Pz = Kz (Kz + lam I)^(-1).
        """
        X = _ensure_1d_or_2d(x)
        Y = _ensure_1d_or_2d(y)
        Z = _ensure_1d_or_2d(z)
        n = X.shape[0]
        if Y.shape[0] != n or Z.shape[0] != n:
            raise ValueError("x, y, z must have the same number of samples")
        if n < 5:
            res = np.nan
            return (res, {"error": "insufficient_data", "n": n}) if return_components else res

        # 1) Build raw (uncentered) Gram matrices using current kernel settings.
        #    We want uncentered here because we will project then center.
        #    Use the same kernel choice for x,y,z as configured for x,y; for z we use kernel_y by default.
        Kx, _, _ = self._gram_and_center(X, which="x")
        Ky, _, _ = self._gram_and_center(Y, which="y")
        # For Z we temporarily force "precomputed" path OFF: we need Kz raw.
        # Temporarily store and restore kernel_y if you want a different kernel for Z.
        kernel_y_backup = self.kernel_y
        try:
            # If you want a separate kernel for Z, you could add self.kernel_z to __init__.
            self.kernel_y = self.kernel_y  # keep as-is; change if desired
            Kz, _, _ = self._gram_and_center(Z, which="y")
        finally:
            self.kernel_y = kernel_y_backup

        # 2) Projection matrix Pz = Kz (Kz + lam I)^-1  (stable ridge)
        #    We will form (I - Pz) implicitly to avoid dense I formation.
        nI = np.eye(n, dtype=np.float64)
        A = Kz + lam * nI
        try:
            A_inv = np.linalg.solve(A, nI)     # (Kz + lam I)^-1
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        Pz = Kz @ A_inv                        # n x n

        # 3) Residualize kernels: K_res = (I - Pz) K (I - Pz)
        #    Compute left and right multiplications explicitly.
        I_minus_P = nI - Pz
        Kx_res = I_minus_P @ Kx @ I_minus_P
        Ky_res = I_minus_P @ Ky @ I_minus_P

        # 4) Center residual kernels
        Kx_c = Kx_res.copy()
        Ky_c = Ky_res.copy()
        _center_numpy_inplace(Kx_c)
        _center_numpy_inplace(Ky_c)

        # 5) Compute (biased or block) normalized HSIC on residual kernels
        if self.estimator == "block":
            num = _block_hsic_fast(Kx_c, Ky_c, block=min(1024, max(64, n // 4)))
        elif self.estimator == "unbiased":
            # unbiased relies on diagonals zeroed in raw grams; here we already projected,
            # so use unbiased formula on the projected (but uncentered) residual kernels:
            num = _hsic_unbiased_from_grams_numba(Kx_res, Ky_res) if self.use_numba else self._hsic_unbiased_np(Kx_res, Ky_res)
            comps = {"n": n, "estimator": "unbiased", "lam": lam, "mode": "conditional"}
            return (float(num), comps) if return_components else float(num)
        else:
            num = _trace_prod(Kx_c, Ky_c) / ((n - 1.0) ** 2)

        if not self.normalize:
            val = float(max(0.0, num))
            comps = {"n": n, "estimator": self.estimator, "lam": lam, "normalized": False, "mode": "conditional"}
            return (val, comps) if return_components else val

        # Normalization uses residual self-energies (same denominator idea as standard HSIC)
        xx = _trace_prod(Kx_c, Kx_c) / ((n - 1.0) ** 2)
        yy = _trace_prod(Ky_c, Ky_c) / ((n - 1.0) ** 2)
        den = math.sqrt(max(xx, 1e-12) * max(yy, 1e-12)) + 1e-12
        val = float(np.clip(num / den, 0.0, 1.0))
        comps = {"n": n, "estimator": self.estimator, "lam": lam, "normalized": True, "mode": "conditional"}
        return (val, comps) if return_components else val

    def pvalue(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 200,
        subsample: Optional[int] = None,
        method: str = "permutation",
    ) -> Tuple[float, float]:
        """Permutation / Gaussian / Gamma p-values. Uses fast permuting of centered kernels."""
        rng = np.random.default_rng(self.random_state)
        X = _ensure_1d_or_2d(x)
        Y = _ensure_1d_or_2d(y)
        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError("x and y must have same length")

        if subsample is not None and n > subsample:
            idx = rng.choice(n, size=subsample, replace=False)
            X = X[idx]
            Y = Y[idx]
            n = X.shape[0]

        # Approximations: use permutation on the approximation for fairness
        if self.estimator in ("linear", "rff"):
            obs = self.score(X, Y)
            if not np.isfinite(obs):
                return obs, np.nan
            if method != "permutation":
                # Gaussian/Gamma not defined for these approximations: fall back to permutation.
                method = "permutation"
            cnt = 0
            for _ in range(B):
                yp = rng.permutation(n)
                v = self.score(X, Y[yp])
                if v >= obs:
                    cnt += 1
            pval = (cnt + 1.0) / (B + 1.0)
            return obs, float(pval)

        # Exact / near-exact paths
        K, Kc, xx = self._gram_and_center(X, "x")
        L, Lc, yy = self._gram_and_center(Y, "y")

        if method == "gaussian":
            obs = self._normalized_from_centered(Kc, Lc, xx, yy)
            from scipy import stats

            mean_h0, var_h0 = _gaussian_hsic_params(K, L)
            if var_h0 <= 0:
                return obs, np.nan
            z = (obs - mean_h0) / math.sqrt(var_h0)
            pval = 1.0 - stats.norm.cdf(z)
            return obs, float(max(0.0, min(1.0, pval)))

        if method == "gamma":
            obs = self._normalized_from_centered(Kc, Lc, xx, yy)
            from scipy import stats

            shape, scale = _gamma_hsic_params(K, L)
            if shape <= 0 or scale <= 0:
                return obs, np.nan
            pval = 1.0 - stats.gamma.cdf(obs, a=shape, scale=scale)
            return obs, float(max(0.0, min(1.0, pval)))

        # permutation: match estimator type
        if self.estimator == "unbiased":
            obs = (
                float(_hsic_unbiased_from_grams_numba(K, L))
                if self.use_numba
                else float(self._hsic_unbiased_np(K, L))
            )
            cnt = 0
            for _ in range(B):
                perm = rng.permutation(n).astype(np.int64)
                # Apply permutation to L rows/cols consistently
                Lp = L[perm][:, perm]
                v = (
                    float(_hsic_unbiased_from_grams_numba(K, Lp))
                    if self.use_numba
                    else float(self._hsic_unbiased_np(K, Lp))
                )
                if v >= obs:
                    cnt += 1
            pval = (cnt + 1.0) / (B + 1.0)
            return obs, float(pval)
        else:
            # biased/block with centered kernels fast path
            obs_num = float(_trace_prod(Kc, Lc) / ((n - 1.0) ** 2))
            obs = (
                obs_num
                if not self.normalize
                else float(np.clip(obs_num / (math.sqrt(xx * yy) + 1e-12), 0.0, 1.0))
            )
            cnt = 0
            for _ in range(B):
                perm = rng.permutation(n).astype(np.int64)
                v_num = _hsic_biased_from_centered_perm(Kc, Lc, perm)
                v = (
                    v_num
                    if not self.normalize
                    else float(np.clip(v_num / (math.sqrt(xx * yy) + 1e-12), 0.0, 1.0))
                )
                if v >= obs:
                    cnt += 1
            pval = (cnt + 1.0) / (B + 1.0)
            return obs, float(pval)

    def matrix(self, df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """Pairwise HSIC matrix with Gram caching. Numeric columns only."""
        cols = df.select_dtypes(include=[np.number]).columns
        if len(cols) != len(df.columns):
            warnings.warn(
                f"Using only numeric columns: {len(cols)} of {len(df.columns)}"
            )
        p = len(cols)
        M = np.eye(p)

        # For approximations, compute once per column
        if self.estimator in ("linear", "rff"):
            Z_or_X: Dict[int, dict] = {}
            for i, c in enumerate(cols):
                arr = df[c].to_numpy()
                X = _ensure_1d_or_2d(arr)
                Z_or_X[i] = {"raw": X}
            total_pairs = p * (p - 1) // 2
            computed = 0
            for i in range(p):
                for j in range(i + 1, p):
                    v = self.score(Z_or_X[i]["raw"], Z_or_X[j]["raw"])
                    M[i, j] = M[j, i] = v
                    computed += 1
                    if show_progress and computed % max(1, total_pairs // 10) == 0:
                        print(
                            f"Progress: {computed}/{total_pairs} ({100*computed/total_pairs:.1f}%)"
                        )
            return pd.DataFrame(M, index=cols, columns=cols)

        # Exact paths: cache grams
        grams = []
        for c in cols:
            arr = df[c].to_numpy()
            K, Kc, xx = self._gram_and_center(_ensure_1d_or_2d(arr), which="x")
            grams.append((K, Kc, xx))
        total_pairs = p * (p - 1) // 2
        computed = 0
        for i in range(p):
            Ki, Kci, xxi = grams[i]
            for j in range(i + 1, p):
                Kj, Kcj, xxj = grams[j]
                if self.estimator == "unbiased":
                    v = (
                        float(_hsic_unbiased_from_grams_numba(Ki, Kj))
                        if self.use_numba
                        else float(self._hsic_unbiased_np(Ki, Kj))
                    )
                else:
                    num = float(_trace_prod(Kci, Kcj) / ((Kci.shape[0] - 1.0) ** 2))
                    if self.normalize:
                        den = math.sqrt(xxi * xxj) + 1e-12
                        v = float(np.clip(num / den, 0.0, 1.0))
                    else:
                        v = float(max(0.0, num))
                M[i, j] = M[j, i] = v
                computed += 1
                if show_progress and computed % max(1, total_pairs // 10) == 0:
                    print(
                        f"Progress: {computed}/{total_pairs} ({100*computed/total_pairs:.1f}%)"
                    )
        return pd.DataFrame(M, index=cols, columns=cols)

    # ---------- Internals ----------
    def _gram_and_center(
        self, A: np.ndarray, which: Literal["x", "y"]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        kernel = self.kernel_x if which == "x" else self.kernel_y
        sigma = self.sigma_x if which == "x" else self.sigma_y

        # Precompute dtype, keep master float64 for stability in traces
        A = _as_dtype(_ensure_1d_or_2d(A), self.prefer_float32)
        n, d = A.shape

        if kernel == "precomputed":
            K = A
            if K.shape[0] != K.shape[1]:
                raise ValueError("precomputed kernel must be square")
            Kc = K.copy()
            _center_inplace(Kc) if self.use_numba else _center_numpy_inplace(Kc)
            xx = _trace_prod(Kc, Kc) / ((n - 1.0) ** 2)
            return K, Kc, float(xx)

        # Distances for certain kernels
        if kernel in ("rbf", "laplacian", "matern"):
            cache_key = (id(A), kernel, sigma, self.bandwidth_method)
            # squared or L1 distances
            if kernel in ("rbf", "matern"):
                D2 = (
                    _pairwise_sq_dists_sym_nd(A) if self.use_numba else ((A @ A.T) * 0)
                )  # placeholder
                if not self.use_numba:
                    # vectorized fallback
                    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
                    sq = np.sum(A * A, axis=1, keepdims=True)
                    D2 = sq + sq.T - 2.0 * (A @ A.T)
                    D2[D2 < 0] = 0.0
                if sigma is None:
                    if cache_key not in self._sigma_cache:
                        sigma = _adaptive_sigma_selection(
                            D2.astype(np.float64), self.bandwidth_method
                        )
                        self._sigma_cache[cache_key] = sigma
                    else:
                        sigma = self._sigma_cache[cache_key]
                if which == "x":
                    self._last_sigma_x = sigma
                else:
                    self._last_sigma_y = sigma
                if kernel == "rbf":
                    K = (
                        _rbf_from_sq_dists_sym(D2.astype(np.float64), float(sigma))
                        if self.use_numba
                        else np.exp(-D2 / (2.0 * sigma * sigma + 1e-12))
                    )
                else:
                    K = (
                        _matern_from_sq_dists_sym(
                            D2.astype(np.float64), float(sigma), float(self.nu)
                        )
                        if self.use_numba
                        else np.exp(-np.sqrt(D2) / (sigma + 1e-12))
                        * (1.0 + np.sqrt(D2) / (sigma + 1e-12))
                    )
            else:  # laplacian
                D1 = (
                    _pairwise_l1_dists_sym_nd(A)
                    if self.use_numba
                    else np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)
                )
                if sigma is None:
                    # reuse D2 heuristic for stability
                    sq = np.sum(A * A, axis=1, keepdims=True)
                    D2 = sq + sq.T - 2.0 * (A @ A.T)
                    D2[D2 < 0] = 0.0
                    sigma = _adaptive_sigma_selection(
                        D2.astype(np.float64), self.bandwidth_method
                    )
                if which == "x":
                    self._last_sigma_x = sigma
                else:
                    self._last_sigma_y = sigma
                K = (
                    _laplacian_from_l1_dists_sym(D1.astype(np.float64), float(sigma))
                    if self.use_numba
                    else np.exp(-D1 / (sigma + 1e-12))
                )

        elif kernel == "linear":
            K = _linear_gram_sym(A) if self.use_numba else (A @ A.T)
        elif kernel == "poly":
            if not isinstance(self.degree, int) or self.degree < 1:
                raise ValueError("degree must be a positive integer")
            K = (
                _poly_gram_sym(A, int(self.degree), float(self.c0))
                if self.use_numba
                else (A @ A.T + self.c0) ** self.degree
            )
        elif kernel == "delta":
            if A.shape[1] != 1:
                raise ValueError(
                    "delta kernel expects 1D discrete input; encode categories otherwise."
                )
            a1d = A.ravel()
            K = (
                _delta_gram_sym_1d(a1d)
                if self.use_numba
                else (a1d[:, None] == a1d[None, :]).astype(float)
            )
        else:
            raise ValueError(f"Unknown kernel '{kernel}'")

        # Center (except unbiased path where we need raw K later; we still return centered + xx)
        if self.estimator == "unbiased":
            return K, K, 1.0

        Kc = K.copy()
        _center_inplace(Kc) if self.use_numba else _center_numpy_inplace(Kc)
        xx = _trace_prod(Kc, Kc) / ((n - 1.0) ** 2)
        return K, Kc, float(xx)

    def _normalized_from_centered(
        self, Kc: np.ndarray, Lc: np.ndarray, xx: float, yy: float
    ) -> float:
        """Biased normalized HSIC from centered kernels."""
        n = Kc.shape[0]
        num = float(_trace_prod(Kc, Lc) / ((n - 1.0) ** 2))
        if not self.normalize:
            return max(0.0, num)
        den = math.sqrt(xx * yy) + 1e-12
        return float(np.clip(num / den, 0.0, 1.0))

    @staticmethod
    def _hsic_unbiased_np(K: np.ndarray, L: np.ndarray) -> float:
        n = K.shape[0]
        if n < 4:
            return np.nan
        Ku = K.copy()
        np.fill_diagonal(Ku, 0.0)
        Lu = L.copy()
        np.fill_diagonal(Lu, 0.0)
        term1 = float((Ku * Lu).sum())
        Ku_row = Ku.sum(axis=1)
        Lu_row = Lu.sum(axis=1)
        term3 = float((Ku_row * Lu_row).sum())
        S_K = float(Ku_row.sum())
        S_L = float(Lu_row.sum())
        n1, n2, n3 = n - 1.0, n - 2.0, n - 3.0
        if n3 <= 0:
            return np.nan
        hsic_u = (term1 + (S_K * S_L) / (n1 * n2) - 2.0 * term3 / n2) / (n * n3)
        return float(max(0.0, hsic_u)) if np.isfinite(hsic_u) else np.nan

    # ----- Linear-time HSIC (L-HSIC) -----
    def _lhsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        m = min(self.approx_m, n // 2) if n >= 2 else 0
        if m < 1:
            return np.nan

        # Choose/bind bandwidths
        sigx = self._last_sigma_x
        sigy = self._last_sigma_y
        if (sigx is None) or (sigy is None):
            # Use quick median heuristic on small subsample
            idx = rng.choice(n, size=min(2048, n), replace=False)
            Xs = X[idx]
            Ys = Y[idx]
            D2x = self._pairwise_sq_np(Xs)
            D2y = self._pairwise_sq_np(Ys)
            if sigx is None:
                sigx = _adaptive_sigma_selection(D2x, self.bandwidth_method)
            if sigy is None:
                sigy = _adaptive_sigma_selection(D2y, self.bandwidth_method)
            self._last_sigma_x = sigx
            self._last_sigma_y = sigy

        # paired subsamples A,B
        idx = rng.choice(n, size=2 * m, replace=False)
        A = idx[:m]
        B = idx[m:]
        s = 0.0
        inv2sigx2 = 1.0 / (2.0 * sigx * sigx + 1e-12)
        inv2sigy2 = 1.0 / (2.0 * sigy * sigy + 1e-12)
        for i in range(m):
            xi = X[A[i]]
            xj = X[B[i]]
            yi = Y[A[i]]
            yj = Y[B[i]]
            dx = float(np.dot(xi - xj, xi - xj))
            dy = float(np.dot(yi - yj, yi - yj))
            k = math.exp(-dx * inv2sigx2)
            l = math.exp(-dy * inv2sigy2)
            s += (k - 1.0) * (l - 1.0)  # centered in expectation for RBF
        return float(max(0.0, s / m))

    # ----- RFF HSIC -----
    def _rff_hsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        D = int(self.rff_features)
        if D < 1:
            raise ValueError("rff_features must be >= 1")

        # Bandwidths
        sigx = self._last_sigma_x
        sigy = self._last_sigma_y
        if (sigx is None) or (sigy is None):
            idx = rng.choice(n, size=min(4096, n), replace=False)
            Xs = X[idx]
            Ys = Y[idx]
            D2x = self._pairwise_sq_np(Xs)
            D2y = self._pairwise_sq_np(Ys)
            if sigx is None:
                sigx = _adaptive_sigma_selection(D2x, self.bandwidth_method)
            if sigy is None:
                sigy = _adaptive_sigma_selection(D2y, self.bandwidth_method)
            self._last_sigma_x = sigx
            self._last_sigma_y = sigy

        Zx = _rff_features_nd(X.astype(np.float64, copy=False), float(sigx), D, rng)
        Zy = _rff_features_nd(Y.astype(np.float64, copy=False), float(sigy), D, rng)
        num = _cross_cov_energy(Zx, Zy)
        if not self.normalize:
            return float(max(0.0, num))
        den = math.sqrt(_cov_energy(Zx) * _cov_energy(Zy)) + 1e-12
        return float(np.clip(num / den, 0.0, 1.0))

    @staticmethod
    def _pairwise_sq_np(A: np.ndarray) -> np.ndarray:
        # vectorized squared Euclidean distances (n,d)
        sq = np.sum(A * A, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (A @ A.T)
        np.maximum(D2, 0.0, out=D2)
        return D2


# ----------------------------
# Convenience functions
# ----------------------------
def hsic_test(x, y, alpha: float = 0.05, method: str = "auto", **kwargs) -> dict:
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if method == "auto":
        if n < 100:
            method = "permutation"
        elif n < 1000:
            method = "gamma"
        else:
            method = "gaussian"
    scorer = HSIC(**kwargs)
    score, pval = scorer.pvalue(x, y, method=method)
    return {
        "hsic": score,
        "pvalue": pval,
        "significant": (pval < alpha) if np.isfinite(pval) else False,
        "alpha": alpha,
        "reject_independence": (pval < alpha) if np.isfinite(pval) else False,
        "method": method,
        "n": n,
    }


def auto_hsic(x, y, **kwargs) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n > 2000:
        kwargs.setdefault("estimator", "block")
    elif n < 200:
        kwargs.setdefault("estimator", "unbiased")
    # Heuristic for discrete 1D
    if "kernel_x" not in kwargs:
        if x.ndim == 1 and len(np.unique(x)) < min(20, n // 10):
            kwargs["kernel_x"] = "delta"
    if "kernel_y" not in kwargs:
        if y.ndim == 1 and len(np.unique(y)) < min(20, n // 10):
            kwargs["kernel_y"] = "delta"
    return HSIC(**kwargs).score(x, y)


def hsic_matrix_test(
    df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "auto",
    correction: str = "bonferroni",
    **kwargs,
) -> pd.DataFrame:
    from scipy.stats import false_discovery_control

    cols = df.select_dtypes(include=[np.number]).columns
    p = len(cols)
    pmat = np.eye(p)
    pvals = []
    positions = []
    scorer = HSIC(**kwargs)
    for i in range(p):
        xi = df[cols[i]].to_numpy()
        for j in range(i + 1, p):
            yi = df[cols[j]].to_numpy()
            _, pval = scorer.pvalue(xi, yi, method=method)
            pmat[i, j] = pmat[j, i] = pval
            pvals.append(pval)
            positions.append((i, j))
    pvals = np.array(pvals)
    if correction == "bonferroni":
        corrected = np.minimum(pvals * len(pvals), 1.0)
    elif correction == "holm":
        order = np.argsort(pvals)
        corrected = np.zeros_like(pvals)
        m = len(pvals)
        for rank, idx in enumerate(order):
            corrected[idx] = min(pvals[idx] * (m - rank), 1.0)
    elif correction == "fdr_bh":
        corrected = false_discovery_control(pvals, method="bh")
    else:
        raise ValueError(f"Unknown correction method: {correction}")
    result = np.eye(p)
    for cp, (i, j) in zip(corrected, positions):
        result[i, j] = result[j, i] = cp
    out = pd.DataFrame(result, index=cols, columns=cols)
    # attach significance matrix for convenience
    sig = np.full((p, p), "", dtype=object)
    for cp, (i, j) in zip(corrected, positions):
        sig_ij = (
            "***" if cp < 1e-3 else "**" if cp < 1e-2 else "*" if cp < alpha else ""
        )
        sig[i, j] = sig[j, i] = sig_ij
    out.significance = pd.DataFrame(sig, index=cols, columns=cols)
    return out
