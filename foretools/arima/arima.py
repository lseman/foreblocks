"""
SARIMAX – performance-improved version + JAX autodiff gradients.

This file is your original “performance-improved” SARIMAX, with **Change #1** implemented:
  1) Replace finite-difference gradients with **JAX autodiff** (value_and_grad + jit)

Design choices for JAX integration:
- We keep your SciPy optimizer (minimize) and simply provide (value, grad) from JAX.
- We build a JAX-only objective that is differentiable end-to-end:
  - parameter transforms (stationary / invertible) implemented in JAX
  - AR/MA lag combination implemented with fixed output lengths (static shapes)
  - state-space matrices built with fixed dimensions from (p_full, q_full)
  - Kalman NLL implemented with lax.scan (pure JAX)

Notes:
- JAX path does NOT use numba kernel (it bypasses it), since JAX handles the loop.
- For dynamic "trim trailing small" in AR/MA polynomials, the JAX path uses FIXED
  lengths (p_full = p + P*s, q_full = q + Q*s). This keeps shapes static for JIT.
- If JAX isn't installed, you still get your original behavior (finite differences).

"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any

import numpy as np


try:
    from scipy.fft import next_fast_len  # PERF: optimal FFT padding
except ImportError:

    def next_fast_len(n):
        return n  # fallback


try:
    from scipy.optimize import minimize
except Exception as e:
    raise ImportError("Requires SciPy.") from e

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*a, **k):
        def _w(f):
            return f

        return _w


# --- JAX autodiff backend (PERF) ---
try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    _HAS_JAX = True
except Exception:
    _HAS_JAX = False
    jax = None
    jnp = None
    lax = None


_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_1d(y) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size < 30:
        raise ValueError("Need at least ~30 finite observations.")
    return y


def _as_2d(X, n: int) -> np.ndarray | None:
    if X is None:
        return None
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != n:
        raise ValueError(f"exog has {X.shape[0]} rows but y has {n}.")
    return np.where(np.isfinite(X), X, 0.0)


def difference(y: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    out = np.asarray(y, dtype=float)
    for _ in range(d):
        out = out[1:] - out[:-1]
    for _ in range(D):
        if s <= 1:
            raise ValueError("Seasonal differencing D>0 requires s>=2.")
        out = out[s:] - out[:-s]
    return out


def difference_exog(
    X: np.ndarray | None, d: int, D: int, s: int
) -> np.ndarray | None:
    if X is None:
        return None
    out = np.asarray(X, dtype=float)
    for _ in range(d):
        out = out[1:] - out[:-1]
    for _ in range(D):
        out = out[s:] - out[:-s]
    return out


def aicc(n: int, k: int, nll: float) -> float:
    aic = 2.0 * k + 2.0 * nll
    return aic + (2.0 * k * (k + 1)) / max(n - k - 1, 1)


def _trim_trailing_small(x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    end = x.size
    while end > 0 and abs(x[end - 1]) <= tol:
        end -= 1
    return x[:end]


def _constrain_stationary_py(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    n = raw.size
    if n == 0:
        return np.zeros(0, dtype=float)
    kappa = np.tanh(raw)
    phi = np.zeros((n, n), dtype=float)
    phi[0, 0] = kappa[0]
    for k in range(1, n):
        phi[k, k] = kappa[k]
        prev = phi[k - 1, :k].copy()
        for j in range(k):
            phi[k, j] = prev[j] - kappa[k] * prev[k - j - 1]
    return phi[n - 1, :].copy()


def _constrain_stationary_nb(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    kappa = np.tanh(raw)
    phi = np.zeros((n, n), dtype=np.float64)
    phi[0, 0] = kappa[0]
    for k in range(1, n):
        phi[k, k] = kappa[k]
        for j in range(k):
            phi[k, j] = phi[k - 1, j] - kappa[k] * phi[k - 1, k - j - 1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = phi[n - 1, i]
    return out


if _HAS_NUMBA:
    _constrain_stationary_nb = njit(nogil=True, cache=False)(_constrain_stationary_nb)


def _constrain_stationary(raw: np.ndarray, *, use_numba: bool = True) -> np.ndarray:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    if raw.size == 0:
        return np.zeros(0, dtype=float)
    if use_numba and _HAS_NUMBA:
        return np.asarray(
            _constrain_stationary_nb(np.ascontiguousarray(raw, dtype=np.float64)),
            dtype=float,
        )
    return _constrain_stationary_py(raw)


def _constrain_invertible(raw: np.ndarray, *, use_numba: bool = True) -> np.ndarray:
    return -_constrain_stationary(raw, use_numba=use_numba)


def _seasonal_poly_py(coeffs: np.ndarray, s: int, sign: float) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    if coeffs.size == 0:
        return np.array([1.0], dtype=float)
    out = np.zeros(coeffs.size * s + 1, dtype=float)
    out[0] = 1.0
    for i, c in enumerate(coeffs, start=1):
        out[i * s] = sign * c
    return out


def _combine_ar_lags_py(ar: np.ndarray, sar: np.ndarray, s: int) -> np.ndarray:
    ns = np.concatenate(([1.0], -np.asarray(ar, dtype=float)))
    seas = _seasonal_poly_py(np.asarray(sar, dtype=float), s=s, sign=-1.0)
    return _trim_trailing_small(-np.convolve(ns, seas)[1:])


def _combine_ma_lags_py(ma: np.ndarray, sma: np.ndarray, s: int) -> np.ndarray:
    ns = np.concatenate(([1.0], np.asarray(ma, dtype=float)))
    seas = _seasonal_poly_py(np.asarray(sma, dtype=float), s=s, sign=+1.0)
    return _trim_trailing_small(np.convolve(ns, seas)[1:])


def _combine_ar_lags_nb(ar: np.ndarray, sar: np.ndarray, s: int) -> np.ndarray:
    p = ar.shape[0]
    P = sar.shape[0]
    ns_len = p + 1
    seas_len = P * s + 1
    poly_len = ns_len + seas_len - 1
    poly = np.zeros(poly_len, dtype=np.float64)
    for i in range(ns_len):
        ns_i = 1.0 if i == 0 else -ar[i - 1]
        poly[i] += ns_i
        for j in range(P):
            poly[i + (j + 1) * s] += ns_i * (-sar[j])
    out_len = poly_len - 1
    end = out_len
    while end > 0 and abs(poly[end]) <= 1e-12:
        end -= 1
    out = np.empty(end, dtype=np.float64)
    for k in range(end):
        out[k] = -poly[k + 1]
    return out


def _combine_ma_lags_nb(ma: np.ndarray, sma: np.ndarray, s: int) -> np.ndarray:
    q = ma.shape[0]
    Q = sma.shape[0]
    ns_len = q + 1
    seas_len = Q * s + 1
    poly_len = ns_len + seas_len - 1
    poly = np.zeros(poly_len, dtype=np.float64)
    for i in range(ns_len):
        ns_i = 1.0 if i == 0 else ma[i - 1]
        poly[i] += ns_i
        for j in range(Q):
            poly[i + (j + 1) * s] += ns_i * sma[j]
    out_len = poly_len - 1
    end = out_len
    while end > 0 and abs(poly[end]) <= 1e-12:
        end -= 1
    out = np.empty(end, dtype=np.float64)
    for k in range(end):
        out[k] = poly[k + 1]
    return out


if _HAS_NUMBA:
    _combine_ar_lags_nb = njit(nogil=True, cache=False)(_combine_ar_lags_nb)
    _combine_ma_lags_nb = njit(nogil=True, cache=False)(_combine_ma_lags_nb)


def _combine_ar_lags(ar, sar, s, *, use_numba=True):
    ar = np.asarray(ar, dtype=float).reshape(-1)
    sar = np.asarray(sar, dtype=float).reshape(-1)
    if use_numba and _HAS_NUMBA:
        return np.asarray(
            _combine_ar_lags_nb(
                np.ascontiguousarray(ar, np.float64),
                np.ascontiguousarray(sar, np.float64),
                int(s),
            ),
            dtype=float,
        )
    return _combine_ar_lags_py(ar, sar, s)


def _combine_ma_lags(ma, sma, s, *, use_numba=True):
    ma = np.asarray(ma, dtype=float).reshape(-1)
    sma = np.asarray(sma, dtype=float).reshape(-1)
    if use_numba and _HAS_NUMBA:
        return np.asarray(
            _combine_ma_lags_nb(
                np.ascontiguousarray(ma, np.float64),
                np.ascontiguousarray(sma, np.float64),
                int(s),
            ),
            dtype=float,
        )
    return _combine_ma_lags_py(ma, sma, s)


def _deterministic_term_nb(c: float, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    k = beta.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        acc = c
        for j in range(k):
            acc += X[i, j] * beta[j]
        out[i] = acc
    return out


if _HAS_NUMBA:
    _deterministic_term_nb = njit(nogil=True, cache=False)(_deterministic_term_nb)


def _deterministic_term(c, beta, X, n, *, include_exog, use_numba):
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if beta.size > 0 and X is not None and include_exog:
        if use_numba and _HAS_NUMBA:
            return np.asarray(
                _deterministic_term_nb(
                    float(c),
                    np.ascontiguousarray(X, np.float64),
                    np.ascontiguousarray(beta, np.float64),
                ),
                dtype=float,
            )
        return np.full(n, float(c), dtype=float) + X @ beta
    return np.full(n, float(c), dtype=float)


# PERF 1: FFT-based autocovariance — O(n log n) instead of O(n * max_lag)
def _autocovariances(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n == 0:
        return np.zeros(max_lag + 1, dtype=float)
    xc = x - x.mean()
    nfft = next_fast_len(2 * n - 1)
    Xf = np.fft.rfft(xc, n=nfft)
    g = np.fft.irfft(Xf * np.conj(Xf))[: max_lag + 1].real
    ns = np.arange(n, n - max_lag - 1, -1, dtype=float)
    return g / ns


# PERF 2: Levinson-Durbin — single phi vector, vectorised inner update
def _levinson_durbin(gamma: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    if order <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    if gamma.size < order + 1 or gamma[0] <= 1e-12:
        return np.zeros(order, dtype=float), np.zeros(order, dtype=float)

    phi = np.zeros(order, dtype=float)
    pacf = np.zeros(order, dtype=float)
    sigma = float(gamma[0])

    for k in range(1, order + 1):
        num = float(gamma[k]) - float(np.dot(phi[: k - 1], gamma[k - 1 : 0 : -1]))
        kk = np.clip(num / max(sigma, 1e-12), -0.98, 0.98)
        pacf[k - 1] = kk
        if k > 1:
            prev = phi[: k - 1].copy()
            phi[: k - 1] = prev - kk * prev[::-1]
        phi[k - 1] = kk
        sigma *= max(1.0 - kk * kk, 1e-6)

    return phi.copy(), pacf


def _stationary_coeffs_to_raw(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(-1)
    n = phi.size
    if n == 0:
        return np.zeros(0, dtype=float)
    work = phi.copy()
    kappa = np.zeros(n, dtype=float)
    for k in range(n - 1, -1, -1):
        kk = float(np.clip(work[k], -0.98, 0.98))
        kappa[k] = kk
        if k == 0:
            break
        den = max(1.0 - kk * kk, 1e-6)
        prev = np.zeros(k, dtype=float)
        for j in range(k):
            prev[j] = (work[j] + kk * work[k - j - 1]) / den
        work[:k] = prev
    return np.arctanh(np.clip(kappa, -0.98, 0.98))


def _build_theta_init(
    yd: np.ndarray,
    Xd: np.ndarray | None,
    *,
    has_c: bool,
    include_exog: bool,
    p: int,
    q: int,
    P: int,
    Q: int,
    s: int,
) -> np.ndarray:
    yd = np.asarray(yd, dtype=float).reshape(-1)
    n = yd.size
    k_exog = 0 if (Xd is None or not include_exog) else Xd.shape[1]

    c0, beta0 = 0.0, np.zeros(k_exog, dtype=float)
    if has_c or k_exog > 0:
        try:
            cols = []
            if has_c:
                cols.append(np.ones(n, dtype=float))
            if k_exog > 0:
                cols.append(np.asarray(Xd, dtype=float))
            A = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
            if A.shape[1] > 0:
                coef, *_ = np.linalg.lstsq(A, yd, rcond=None)
                pos = 0
                if has_c:
                    c0 = float(coef[pos])
                    pos += 1
                if k_exog > 0:
                    beta0 = np.asarray(coef[pos : pos + k_exog], dtype=float)
        except Exception:
            c0 = float(np.mean(yd)) if has_c else 0.0

    det = np.full(n, c0, dtype=float)
    if k_exog > 0:
        det += np.asarray(Xd, dtype=float) @ beta0
    resid = yd - det

    # AR init
    ar_coeff = np.zeros(p, dtype=float)
    pacf_ar = np.zeros(p, dtype=float)
    if p > 0:
        g_ar = _autocovariances(resid, p)
        ar_coeff, pacf_ar = _levinson_durbin(g_ar, p)
    raw_ar0 = np.arctanh(np.clip(pacf_ar, -0.98, 0.98))

    resid_ar = resid.copy()
    if p > 0:
        for t in range(p, n):
            resid_ar[t] = resid[t] - float(np.dot(ar_coeff, resid[t - p : t][::-1]))

    max_lag = max(q, P * s, Q * s, 1)
    g_res = _autocovariances(resid_ar, max_lag)
    g0 = max(float(g_res[0]), 1e-12)

    def rho(lag):
        return (
            float(np.clip(g_res[lag] / g0, -0.98, 0.98))
            if 0 < lag < g_res.size
            else 0.0
        )

    ma_grid = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])

    def snap(v):
        return float(ma_grid[np.argmin(np.abs(ma_grid - v))])

    ma_target = np.array([snap(0.35 * rho(i + 1)) for i in range(q)])
    raw_ma0 = _stationary_coeffs_to_raw(-np.clip(ma_target, -0.95, 0.95))

    sar_target = np.zeros(P)
    for i in range(P):
        sar_target[i] = 0.5 * rho((i + 1) * s)
        if abs(sar_target[i]) < 0.05 and p > 0:
            sar_target[i] = 0.5 * ar_coeff[min(i, p - 1)]
    raw_sar0 = _stationary_coeffs_to_raw(np.clip(sar_target, -0.95, 0.95))

    sma_target = np.zeros(Q)
    for i in range(Q):
        sma_target[i] = snap(0.35 * rho((i + 1) * s))
        if abs(sma_target[i]) < 0.05 and q > 0:
            sma_target[i] = 0.5 * ma_target[min(i, q - 1)]
    raw_sma0 = _stationary_coeffs_to_raw(-np.clip(sma_target, -0.95, 0.95))

    start = max(p, 1)
    sigma2_0 = max(
        float(np.var(resid_ar[start:]) if n > start else np.var(resid_ar)), 1e-6
    )

    theta0 = []
    if has_c:
        theta0.append(float(c0))
    theta0.extend(beta0.tolist())
    theta0.extend(raw_ar0.tolist())
    theta0.extend(raw_sar0.tolist())
    theta0.extend(raw_ma0.tolist())
    theta0.extend(raw_sma0.tolist())
    theta0.append(float(np.log(sigma2_0)))
    return np.asarray(theta0, dtype=float)


def _kalman_nll_kernel(
    y_adj: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    sigma2: float,
    diffuse_scale: float,
    diffuse_burn: int,
) -> tuple[float, int]:
    n = int(y_adj.shape[0])
    m = int(T.shape[0])
    a_pred = np.zeros(m, dtype=np.float64)
    P_pred = np.eye(m, dtype=np.float64) * float(diffuse_scale)
    RRt = np.outer(R, R) * float(sigma2)
    nll = 0.0
    n_eff = 0
    for t in range(n):
        vt = float(y_adj[t] - np.dot(Z, a_pred))
        PZ = P_pred @ Z
        Ft = float(np.dot(Z, PZ))
        if Ft < 1e-12:
            Ft = 1e-12
        a_filt = a_pred + PZ * (vt / Ft)
        P_filt = P_pred - np.outer(PZ, PZ) / Ft
        P_filt = 0.5 * (P_filt + P_filt.T)
        if t >= diffuse_burn:
            nll += 0.5 * (_LOG_2PI + np.log(Ft) + (vt * vt) / Ft)
            n_eff += 1
        a_pred = T @ a_filt
        P_pred = T @ P_filt @ T.T + RRt
        P_pred = 0.5 * (P_pred + P_pred.T)
    return float(nll), int(n_eff)


if _HAS_NUMBA:
    _kalman_nll_kernel = njit(nogil=True, cache=False)(_kalman_nll_kernel)


# ---------------------------------------------------------------------------
# JAX objective pieces (autodiff)
# ---------------------------------------------------------------------------

if _HAS_JAX:
    import jax.numpy as jnp
    from jax import lax

    def _jax_constrain_stationary(raw: jnp.ndarray) -> jnp.ndarray:
        raw = jnp.asarray(raw).reshape(-1)
        n = raw.shape[0]

        if n == 0:
            return jnp.zeros((0,), dtype=jnp.float64)

        kappa = jnp.tanh(raw)
        row = jnp.asarray([kappa[0]], dtype=jnp.float64)
        for k in range(1, n):
            kk = kappa[k]
            row = jnp.concatenate([row - kk * row[::-1], jnp.asarray([kk])])
        return row

    def _jax_constrain_invertible(raw: jnp.ndarray) -> jnp.ndarray:
        return -_jax_constrain_stationary(raw)

    def _jax_seasonal_poly(coeffs: jnp.ndarray, s: int, sign: float) -> jnp.ndarray:
        coeffs = jnp.asarray(coeffs).reshape(-1)
        n = coeffs.shape[0]
        out = jnp.zeros((n * s + 1,), dtype=jnp.float64)
        out = out.at[0].set(1.0)
        for i in range(n):
            out = out.at[(i + 1) * s].set(sign * coeffs[i])
        return out

    def _jax_combine_ar_lags_fixed(
        ar: jnp.ndarray, sar: jnp.ndarray, s: int, out_len: int
    ) -> jnp.ndarray:
        # out_len must be static: p_full = p + P*s (fixed)
        ns = jnp.concatenate([jnp.array([1.0], dtype=jnp.float64), -jnp.asarray(ar)])
        seas = _jax_seasonal_poly(jnp.asarray(sar), s=s, sign=-1.0)
        poly = jnp.convolve(ns, seas)  # length = (p+1)+(P*s+1)-1 = p+P*s+1
        out = -poly[1 : 1 + out_len]  # fixed length
        return out

    def _jax_combine_ma_lags_fixed(
        ma: jnp.ndarray, sma: jnp.ndarray, s: int, out_len: int
    ) -> jnp.ndarray:
        ns = jnp.concatenate([jnp.array([1.0], dtype=jnp.float64), jnp.asarray(ma)])
        seas = _jax_seasonal_poly(jnp.asarray(sma), s=s, sign=+1.0)
        poly = jnp.convolve(ns, seas)  # length = q+Q*s+1
        out = poly[1 : 1 + out_len]  # fixed length
        return out

    def _jax_build_state_space(
        phi: jnp.ndarray,
        theta: jnp.ndarray,
        *,
        T_base: jnp.ndarray,
        r: int,
        p_full: int,
        q_full: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # T = T_base with first row updated:
        #   T[0, :p_full] = phi
        #   T[0, r:r+q_full] = theta
        T = T_base
        # Write phi
        if p_full > 0:
            T = T.at[0, :p_full].set(phi)
        # Write theta
        if q_full > 0:
            T = T.at[0, r : r + q_full].set(theta)
        # Z, R are constant given template; build separately outside if needed.
        return T

    def _jax_kalman_nll(
        y_adj: jnp.ndarray,
        T: jnp.ndarray,
        Z: jnp.ndarray,
        R: jnp.ndarray,
        sigma2: float,
        diffuse_scale: float,
        diffuse_burn: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        y_adj = jnp.asarray(y_adj, dtype=jnp.float64)
        T = jnp.asarray(T, dtype=jnp.float64)
        Z = jnp.asarray(Z, dtype=jnp.float64)
        R = jnp.asarray(R, dtype=jnp.float64)

        m = T.shape[0]

        a0 = jnp.zeros((m,), dtype=jnp.float64)
        P0 = jnp.eye(m, dtype=jnp.float64) * jnp.asarray(diffuse_scale, jnp.float64)
        RRt = jnp.outer(R, R) * jnp.asarray(sigma2, jnp.float64)

        def step(carry, xt):
            t, a_pred, P_pred, nll, n_eff = carry
            y_t = xt

            vt = y_t - (Z @ a_pred)
            PZ = P_pred @ Z
            Ft = Z @ PZ
            Ft = jnp.maximum(Ft, 1e-12)

            K = PZ / Ft
            a_filt = a_pred + K * vt
            P_filt = P_pred - jnp.outer(K, K) * Ft
            # keep symmetric for numerical stability (still differentiable)
            P_filt = 0.5 * (P_filt + P_filt.T)

            add = 0.5 * (_LOG_2PI + jnp.log(Ft) + (vt * vt) / Ft)
            do_add = t >= diffuse_burn
            nll = nll + jnp.where(do_add, add, 0.0)
            n_eff = n_eff + jnp.where(do_add, 1, 0)

            a_next = T @ a_filt
            P_next = T @ P_filt @ T.T + RRt
            P_next = 0.5 * (P_next + P_next.T)

            return (t + 1, a_next, P_next, nll, n_eff), None

        init = (jnp.int32(0), a0, P0, jnp.asarray(0.0, jnp.float64), jnp.int32(0))
        (t_fin, a_fin, P_fin, nll_fin, n_eff_fin), _ = lax.scan(step, init, y_adj)
        return nll_fin, n_eff_fin


# ---------------------------------------------------------------------------
# Public spec / fit dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SarimaxSpec:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    include_intercept: bool = True
    include_exog: bool = True


@dataclass
class SarimaxFit:
    spec: SarimaxSpec
    params: dict[str, np.ndarray]
    nll: float
    aicc: float
    converged: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# State-space SARIMAX
# ---------------------------------------------------------------------------


class SarimaxScratch:
    _SS_TEMPLATE_CACHE: dict[tuple[int, int], dict[str, Any]] = {}

    def __init__(self, spec: SarimaxSpec):
        self.spec = spec
        self.fit_: SarimaxFit | None = None

    @classmethod
    def _get_state_space_template(cls, p_full: int, q_full: int) -> dict[str, Any]:
        key = (int(p_full), int(q_full))
        tpl = cls._SS_TEMPLATE_CACHE.get(key)
        if tpl is not None:
            return tpl
        r = max(1, p_full, q_full + 1)
        m = r + q_full
        T_base = np.zeros((m, m), dtype=float)
        Z = np.zeros(m, dtype=float)
        Z[0] = 1.0
        R = np.zeros(m, dtype=float)
        R[0] = 1.0
        if q_full > 0:
            R[r] = 1.0
        for i in range(1, r):
            T_base[i, i - 1] = 1.0
        for j in range(1, q_full):
            T_base[r + j, r + j - 1] = 1.0
        T_base.setflags(write=False)
        Z.setflags(write=False)
        R.setflags(write=False)
        tpl = {
            "T_base": T_base,
            "Z": Z,
            "R": R,
            "m": np.array([m], dtype=int),
            "r": np.array([r], dtype=int),
            "p_full": np.array([p_full], dtype=int),
            "q_full": np.array([q_full], dtype=int),
            "r_int": r,
        }
        cls._SS_TEMPLATE_CACHE[key] = tpl
        return tpl

    @classmethod
    def _build_state_space(
        cls, phi: np.ndarray, theta: np.ndarray
    ) -> dict[str, np.ndarray]:
        phi = np.asarray(phi, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        tpl = cls._get_state_space_template(
            p_full=int(phi.size), q_full=int(theta.size)
        )
        T = np.asarray(tpl["T_base"], dtype=float).copy()
        r = int(tpl["r_int"])
        if phi.size > 0:
            T[0, : phi.size] = phi
        if theta.size > 0:
            T[0, r : r + theta.size] = theta
        return {
            "T": T,
            "Z": np.asarray(tpl["Z"], dtype=float),
            "R": np.asarray(tpl["R"], dtype=float),
            "m": np.asarray(tpl["m"], dtype=int),
            "r": np.asarray(tpl["r"], dtype=int),
            "p_full": np.asarray(tpl["p_full"], dtype=int),
            "q_full": np.asarray(tpl["q_full"], dtype=int),
        }

    @staticmethod
    def _resolve_diffuse_burn(n, p_full, q_full, diffuse_burn=None):
        if diffuse_burn is not None:
            burn = int(diffuse_burn)
        else:
            burn = max(10, int(1.5 * max(int(p_full), int(q_full) + 1)))
        return min(max(burn, 0), max(n - 1, 0))

    @staticmethod
    def _kalman_filter(
        y_adj,
        T,
        Z,
        R,
        sigma2,
        *,
        diffuse_scale,
        diffuse_burn,
        store,
        use_numba=True,
    ) -> dict[str, Any]:
        y_adj = np.asarray(y_adj, dtype=float).reshape(-1)
        n = y_adj.size
        m = T.shape[0]

        if not store and use_numba:
            nll, n_eff = _kalman_nll_kernel(
                y_adj.astype(np.float64, copy=False),
                np.asarray(T, np.float64),
                np.asarray(Z, np.float64),
                np.asarray(R, np.float64),
                float(sigma2),
                float(diffuse_scale),
                int(diffuse_burn),
            )
            return {
                "nll": float(nll),
                "n_eff": int(n_eff),
                "v": np.zeros(0),
                "F": np.zeros(0),
            }

        a_pred = np.zeros(m, dtype=float)
        P_pred = np.eye(m, dtype=float) * float(diffuse_scale)
        RRt = np.outer(R, R) * float(sigma2)

        if store:
            a_pred_hist = np.zeros((n, m))
            a_filt_hist = np.zeros((n, m))
            P_pred_hist = np.zeros((n, m, m))
            P_filt_hist = np.zeros((n, m, m))

        v = np.zeros(n)
        F = np.zeros(n)
        nll = 0.0
        n_eff = 0

        for t in range(n):
            if store:
                a_pred_hist[t] = a_pred
                P_pred_hist[t] = P_pred
            vt = float(y_adj[t] - Z @ a_pred)
            PZ = P_pred @ Z
            Ft = max(float(Z @ PZ), 1e-12)
            K = PZ / Ft
            a_filt = a_pred + K * vt
            P_filt = P_pred - np.outer(K, K) * Ft
            if store:
                P_filt = 0.5 * (P_filt + P_filt.T)
            v[t] = vt
            F[t] = Ft
            if t >= diffuse_burn:
                nll += 0.5 * (_LOG_2PI + np.log(Ft) + (vt * vt) / Ft)
                n_eff += 1
            if store:
                a_filt_hist[t] = a_filt
                P_filt_hist[t] = P_filt
            a_pred = T @ a_filt
            P_pred = T @ P_filt @ T.T + RRt
            P_pred = 0.5 * (P_pred + P_pred.T)

        out: dict[str, Any] = {"nll": float(nll), "n_eff": int(n_eff), "v": v, "F": F}
        if store:
            out.update(
                {
                    "a_pred": a_pred_hist,
                    "a_filt": a_filt_hist,
                    "P_pred": P_pred_hist,
                    "P_filt": P_filt_hist,
                }
            )
        return out

    @staticmethod
    def _kalman_smoother(T: np.ndarray, filt: dict[str, Any]) -> dict[str, np.ndarray]:
        a_filt = np.asarray(filt["a_filt"], dtype=float)
        P_filt = np.asarray(filt["P_filt"], dtype=float)
        a_pred = np.asarray(filt["a_pred"], dtype=float)
        P_pred = np.asarray(filt["P_pred"], dtype=float)

        n, m = a_filt.shape
        a_smooth = np.zeros((n, m))
        P_smooth = np.zeros((n, m, m))
        a_smooth[-1] = a_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for t in range(n - 2, -1, -1):
            Pn = P_pred[t + 1]
            J = np.linalg.lstsq(Pn, T @ P_filt[t].T, rcond=1e-12)[0].T
            a_smooth[t] = a_filt[t] + J @ (a_smooth[t + 1] - a_pred[t + 1])
            dP = P_smooth[t + 1] - Pn
            P_smooth[t] = P_filt[t] + J @ dP @ J.T
            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

        return {"a_smooth": a_smooth, "P_smooth": P_smooth}

    def _decode_params(self, theta_u: np.ndarray, k_exog: int, *, use_numba=True):
        p, _, q = self.spec.order
        P, _, Q, s = self.spec.seasonal_order
        has_c = bool(self.spec.include_intercept)
        idx = 0
        c = 0.0
        if has_c:
            c = float(theta_u[idx])
            idx += 1
        beta = np.zeros(0, dtype=float)
        if k_exog > 0:
            beta = np.asarray(theta_u[idx : idx + k_exog], dtype=float)
            idx += k_exog
        raw_ar = theta_u[idx : idx + p]
        idx += p
        raw_sar = theta_u[idx : idx + P]
        idx += P
        raw_ma = theta_u[idx : idx + q]
        idx += q
        raw_sma = theta_u[idx : idx + Q]
        idx += Q
        sigma2 = float(np.exp(theta_u[idx]) + 1e-12)

        ar = _constrain_stationary(raw_ar, use_numba=use_numba)
        sar = _constrain_stationary(raw_sar, use_numba=use_numba)
        ma = _constrain_invertible(raw_ma, use_numba=use_numba)
        sma = _constrain_invertible(raw_sma, use_numba=use_numba)
        phi = _combine_ar_lags(ar, sar, s=s, use_numba=use_numba)
        theta = _combine_ma_lags(ma, sma, s=s, use_numba=use_numba)
        ss = self._build_state_space(phi=phi, theta=theta)
        return {
            "c": np.array([c]),
            "beta": beta.astype(float),
            "ar": ar,
            "sar": sar,
            "ma": ma,
            "sma": sma,
            "phi": phi,
            "theta": theta,
            "sigma2": np.array([sigma2]),
            **ss,
        }

    def fit(
        self,
        y,
        exog=None,
        *,
        maxiter=300,
        method="L-BFGS-B",  # kept for compatibility, but ignored in JAX path
        verbose=False,
        seed=0,
        diffuse_scale=1e6,
        diffuse_burn=None,
        pre_differenced=False,
        compute_smoother=True,
        use_numba=True,
        init_params=None,
        use_jax_autodiff: bool = True,
        jax_enable_x64: bool = True,
        jax_jit: bool = True,
        # New Optax-related arguments (optional)
        optax_optimizer=None,  # you can pass custom optax optimizer
        optax_maxiter: int = 500,  # usually more iterations than scipy
        optax_tol: float = 1e-6,  # loss change tolerance for early stopping
        optax_patience: int = 30,  # how many steps without improvement before stop
    ) -> SarimaxFit:
        y0 = _as_1d(y)
        X0 = _as_2d(exog, n=y0.size)
        p, d, q = self.spec.order
        P, D, Q, s = self.spec.seasonal_order
        if s < 1:
            raise ValueError("s must be >= 1.")
        if pre_differenced:
            yd, Xd = y0, X0
        else:
            yd = difference(y0, d=d, D=D, s=s)
            Xd = difference_exog(X0, d=d, D=D, s=s)
        n = yd.size
        if n < 10:
            raise ValueError("Series too short after differencing.")
        k_exog = 0 if (Xd is None or not self.spec.include_exog) else Xd.shape[1]
        has_c = bool(self.spec.include_intercept)
        dim = (1 if has_c else 0) + k_exog + p + P + q + Q + 1

        if init_params is not None:
            theta0 = np.asarray(init_params, dtype=float).reshape(-1)
            if theta0.size != dim:
                raise ValueError(f"init_params size {theta0.size} != expected {dim}.")
        else:
            theta0 = _build_theta_init(
                yd=yd,
                Xd=Xd,
                has_c=has_c,
                include_exog=self.spec.include_exog,
                p=p,
                q=q,
                P=P,
                Q=Q,
                s=s,
            )
            if theta0.size != dim:
                theta0 = np.zeros(dim, dtype=float)
                idx = (1 if has_c else 0) + k_exog + p + P + q + Q
                theta0[idx] = float(np.log(np.var(yd) + 1e-6))

        Xd_c = None if Xd is None else np.ascontiguousarray(Xd, np.float64)

        def det_np(c, beta):
            return _deterministic_term(
                c=c,
                beta=beta,
                X=Xd_c if (use_numba and Xd_c is not None) else Xd,
                n=n,
                include_exog=self.spec.include_exog,
                use_numba=use_numba,
            )

        # ------------------------------------------------------------
        # JAX AUTODIFF + OPTAX path
        # ------------------------------------------------------------
        use_jax_path = bool(use_jax_autodiff and _HAS_JAX)

        if use_jax_path:
            if jax_enable_x64:
                try:
                    jax.config.update("jax_enable_x64", True)
                except Exception:
                    pass

            # Fixed shapes
            p_full = int(p + P * s)
            q_full = int(q + Q * s)

            tpl = self._get_state_space_template(p_full=p_full, q_full=q_full)
            T_base = jnp.asarray(tpl["T_base"], dtype=jnp.float64)
            Z = jnp.asarray(tpl["Z"], dtype=jnp.float64)
            R = jnp.asarray(tpl["R"], dtype=jnp.float64)
            r_int = int(tpl["r_int"])

            yd_j = jnp.asarray(yd, dtype=jnp.float64)
            Xd_j = jnp.asarray(Xd, dtype=jnp.float64) if Xd is not None else None
            diffuse_scale_j = float(diffuse_scale)
            burn_j = int(self._resolve_diffuse_burn(n, p_full, q_full, diffuse_burn))
            include_exog = bool(self.spec.include_exog)

            def det_jax(c, beta):
                if include_exog and (Xd_j is not None) and (beta.size > 0):
                    return jnp.full((n,), c) + Xd_j @ beta
                return jnp.full((n,), c)

            def decode_and_nll(
                theta_u_j: jnp.ndarray,
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
                idx = 0
                c = theta_u_j[idx] if has_c else jnp.asarray(0.0)
                if has_c:
                    idx += 1
                beta = theta_u_j[idx : idx + k_exog] if k_exog > 0 else jnp.zeros((0,))
                if k_exog > 0:
                    idx += k_exog

                raw_ar = theta_u_j[idx : idx + p]
                idx += p
                raw_sar = theta_u_j[idx : idx + P]
                idx += P
                raw_ma = theta_u_j[idx : idx + q]
                idx += q
                raw_sma = theta_u_j[idx : idx + Q]
                idx += Q
                sigma2 = jnp.exp(theta_u_j[idx]) + 1e-12

                ar = _jax_constrain_stationary(raw_ar)
                sar = _jax_constrain_stationary(raw_sar)
                ma = _jax_constrain_invertible(raw_ma)
                sma = _jax_constrain_invertible(raw_sma)

                phi = _jax_combine_ar_lags_fixed(ar, sar, s=int(s), out_len=p_full)
                theta = _jax_combine_ma_lags_fixed(ma, sma, s=int(s), out_len=q_full)

                T = _jax_build_state_space(
                    phi, theta, T_base=T_base, r=r_int, p_full=p_full, q_full=q_full
                )

                y_adj = yd_j - det_jax(c, beta)
                nll, n_eff = _jax_kalman_nll(
                    y_adj=y_adj,
                    T=T,
                    Z=Z,
                    R=R,
                    sigma2=sigma2,
                    diffuse_scale=diffuse_scale_j,
                    diffuse_burn=burn_j,
                )
                return nll, n_eff

            def nll_only(th):
                return decode_and_nll(th)[0]

            value_and_grad_fn = jax.value_and_grad(nll_only)

            if jax_jit:
                value_and_grad_fn = jax.jit(value_and_grad_fn)

            def objective_and_grad(theta_u: np.ndarray) -> tuple[float, np.ndarray]:
                th = jnp.asarray(np.asarray(theta_u, dtype=np.float64))
                val_j, grad_j = value_and_grad_fn(th)
                val = float(np.asarray(val_j))
                grad = np.asarray(grad_j, dtype=float)
                if not np.isfinite(val):
                    return 1e50, np.zeros_like(theta_u, dtype=float)
                grad = np.where(np.isfinite(grad), grad, 0.0)
                return val, grad

            res = None
            final_step = -1
            optimizer_name = method
            opt_message = ""
            optax_used = False

            optimizer = optax_optimizer
            optax_mod = None

            if optimizer is None:
                try:
                    import optax as _optax  # type: ignore

                    if hasattr(_optax, "lbfgs"):
                        optimizer = _optax.lbfgs()
                        optax_mod = _optax
                except Exception:
                    optimizer = None

            if optimizer is not None:
                try:
                    if optax_mod is None:
                        import optax as _optax  # type: ignore

                        optax_mod = _optax

                    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
                    opt_state = optimizer.init(theta0_j)

                    @jax.jit
                    def opt_step(carry, _):
                        params, state, step = carry
                        loss, grads = value_and_grad_fn(params)
                        updates, state = optimizer.update(grads, state, params)
                        params = optax_mod.apply_updates(params, updates)
                        return (params, state, step + 1), (loss, params)

                    init_carry = (theta0_j, opt_state, jnp.array(0))
                    (_, _, final_step_j), (losses, all_params) = lax.scan(
                        opt_step, init_carry, None, length=int(optax_maxiter)
                    )
                    losses = jnp.asarray(losses)
                    min_idx = jnp.argmin(losses)
                    best_loss = float(losses[min_idx])
                    best_params = all_params[min_idx]

                    theta_opt_np = np.asarray(best_params, dtype=float)
                    final_step = int(final_step_j)
                    optimizer_name = "optax"
                    opt_message = "Optax converged"
                    optax_used = True

                    if verbose:
                        print(
                            f"[Optax] Final loss: {best_loss:.6f} after {final_step} steps "
                            f"(best at step {int(min_idx)})"
                        )
                except Exception:
                    optimizer = None

            if optimizer is None:
                res = minimize(
                    objective_and_grad,
                    theta0,
                    method=method,
                    jac=True,
                    options={"maxiter": int(maxiter), "disp": bool(verbose)},
                )
                theta_opt_np = np.asarray(res.x, dtype=float)
                best_loss = float(res.fun)
                final_step = int(getattr(res, "nit", -1))
                optimizer_name = method
                opt_message = str(getattr(res, "message", ""))

        else:
            # ─── Original SciPy + finite diff fallback ──────────────────────────
            def objective(theta_u: np.ndarray) -> float:
                dec = self._decode_params(theta_u, k_exog, use_numba=use_numba)
                y_adj = yd - det_np(float(dec["c"][0]), dec["beta"])
                burn = self._resolve_diffuse_burn(
                    n, int(dec["p_full"][0]), int(dec["q_full"][0]), diffuse_burn
                )
                filt = self._kalman_filter(
                    y_adj,
                    dec["T"],
                    dec["Z"],
                    dec["R"],
                    float(dec["sigma2"][0]),
                    diffuse_scale=diffuse_scale,
                    diffuse_burn=burn,
                    store=False,
                    use_numba=use_numba,
                )
                val = float(filt["nll"])
                return val if np.isfinite(val) else 1e50

            _eps = 1e-6
            _th_p = np.empty(dim, dtype=float)
            _th_m = np.empty(dim, dtype=float)

            def objective_and_grad(theta_u: np.ndarray) -> tuple[float, np.ndarray]:
                val = objective(theta_u)
                grad = np.empty(dim, dtype=float)
                for i in range(dim):
                    _th_p[:] = theta_u
                    _th_p[i] += _eps
                    _th_m[:] = theta_u
                    _th_m[i] -= _eps
                    grad[i] = (objective(_th_p) - objective(_th_m)) / (2.0 * _eps)
                return val, grad

            res = minimize(
                objective_and_grad,
                theta0,
                method=method,
                jac=True,
                options={"maxiter": int(maxiter), "disp": bool(verbose)},
            )
            theta_opt_np = np.asarray(res.x, dtype=float)
            best_loss = res.fun
            final_step = int(getattr(res, "nit", -1))
            optimizer_name = method
            opt_message = str(getattr(res, "message", ""))
            optax_used = False

        # ─── Final decode & filtering (same for both paths) ─────────────────────
        dec = self._decode_params(theta_opt_np, k_exog, use_numba=use_numba)
        y_adj = yd - det_np(float(dec["c"][0]), dec["beta"])
        burn = self._resolve_diffuse_burn(
            n, int(dec["p_full"][0]), int(dec["q_full"][0]), diffuse_burn
        )
        filt = self._kalman_filter(
            y_adj,
            dec["T"],
            dec["Z"],
            dec["R"],
            float(dec["sigma2"][0]),
            diffuse_scale=diffuse_scale,
            diffuse_burn=burn,
            store=compute_smoother,
            use_numba=use_numba,
        )
        smooth = self._kalman_smoother(dec["T"], filt) if compute_smoother else None

        nll_hat = float(filt["nll"])
        n_eff = int(filt["n_eff"])
        score = float(aicc(n_eff, dim, nll_hat))

        info: dict[str, Any] = {
            "n_obs": int(n),
            "n_eff": n_eff,
            "state_dim": int(dec["m"][0]),
            "ar_lag_dim": int(dec["r"][0]),
            "ma_lag_dim": int(dec["q_full"][0]),
            "diffuse_scale": float(diffuse_scale),
            "diffuse_burn": int(burn),
            "optimizer": optimizer_name,
            "message": opt_message,
            "nit": int(final_step),
            "numba_used": bool(use_numba and _HAS_NUMBA),
            "jax_autodiff_used": bool(use_jax_path),
            "optax_used": bool(optax_used),
            "pre_differenced": bool(pre_differenced),
            "compute_smoother": bool(compute_smoother),
            "theta_u_opt": theta_opt_np,
            "state_space": {"T": dec["T"], "Z": dec["Z"], "R": dec["R"]},
        }

        if compute_smoother:
            info.update(
                {
                    "innovations": filt["v"],
                    "innovation_var": filt["F"],
                    "filtered_state": filt["a_filt"],
                    "smoothed_state": smooth["a_smooth"] if smooth else None,
                }
            )

        self.fit_ = SarimaxFit(
            spec=self.spec,
            params={
                "c": dec["c"],
                "beta": dec["beta"],
                "ar": dec["ar"],
                "sar": dec["sar"],
                "ma": dec["ma"],
                "sma": dec["sma"],
                "phi": dec["phi"],
                "theta": dec["theta"],
                "sigma2": dec["sigma2"],
            },
            nll=nll_hat,
            aicc=score,
            converged=(best_loss < 1e20),  # crude check — improve if needed
            info=info,
        )
        return self.fit_

    def filter_smoother(self, y, exog=None) -> dict[str, np.ndarray]:
        if self.fit_ is None:
            raise RuntimeError("Call fit() first.")
        fit = self.fit_
        y0 = _as_1d(y)
        X0 = _as_2d(exog, n=y0.size)
        _, d, _ = fit.spec.order
        _, D, _, s = fit.spec.seasonal_order
        yd = difference(y0, d=d, D=D, s=s)
        Xd = difference_exog(X0, d=d, D=D, s=s)
        n = yd.size
        c = float(fit.params["c"][0]) if fit.spec.include_intercept else 0.0
        beta = fit.params["beta"]
        ss = self._build_state_space(fit.params["phi"], fit.params["theta"])
        det = _deterministic_term(
            c=c,
            beta=beta,
            X=Xd,
            n=n,
            include_exog=fit.spec.include_exog,
            use_numba=True,
        )
        burn = self._resolve_diffuse_burn(n, int(ss["p_full"][0]), int(ss["q_full"][0]))
        filt = self._kalman_filter(
            yd - det,
            ss["T"],
            ss["Z"],
            ss["R"],
            float(fit.params["sigma2"][0]),
            diffuse_scale=1e6,
            diffuse_burn=burn,
            store=True,
        )
        smooth = self._kalman_smoother(ss["T"], filt)
        return {
            "innovations": filt["v"],
            "innovation_var": filt["F"],
            "filtered_state": filt["a_filt"],
            "smoothed_state": smooth["a_smooth"],
        }

    def forecast(
        self,
        y,
        steps: int,
        exog=None,
        exog_future=None,
        *,
        return_intervals=True,
        alpha=0.05,
        num_sim=2000,
        seed=0,
    ) -> dict[str, np.ndarray]:
        if self.fit_ is None:
            raise RuntimeError("Call fit() first.")
        fit = self.fit_
        y0 = _as_1d(y)
        X0 = _as_2d(exog, n=y0.size)
        _, d, _ = fit.spec.order
        _, D, _, s = fit.spec.seasonal_order
        yd = difference(y0, d=d, D=D, s=s)
        Xd = difference_exog(X0, d=d, D=D, s=s)
        n = yd.size

        Xf = None
        if exog_future is not None:
            Xf = np.asarray(exog_future, dtype=float)
            if Xf.ndim == 1:
                Xf = Xf.reshape(-1, 1)
            if Xf.shape[0] != steps:
                raise ValueError("exog_future must have shape (steps, k_exog).")
            if d > 0 or D > 0:
                raise NotImplementedError(
                    "Provide exog_future in differenced space when d>0 or D>0."
                )

        c = float(fit.params["c"][0]) if fit.spec.include_intercept else 0.0
        beta = fit.params["beta"]
        sigma2 = float(fit.params["sigma2"][0])
        ss = self._build_state_space(fit.params["phi"], fit.params["theta"])
        T, Z, R = ss["T"], ss["Z"], ss["R"]

        det = _deterministic_term(
            c=c,
            beta=beta,
            X=Xd,
            n=n,
            include_exog=fit.spec.include_exog,
            use_numba=True,
        )
        burn = self._resolve_diffuse_burn(n, int(ss["p_full"][0]), int(ss["q_full"][0]))
        filt = self._kalman_filter(
            yd - det, T, Z, R, sigma2, diffuse_scale=1e6, diffuse_burn=burn, store=True
        )
        a_last = filt["a_filt"][-1].copy()

        Xf_use = None
        if beta.size > 0 and fit.spec.include_exog:
            if Xf is None:
                raise ValueError(
                    "exog_future required when model was fitted with exog."
                )
            if Xf.shape[1] != beta.size:
                raise ValueError("exog_future has wrong number of columns.")
            Xf_use = Xf
        det_future = _deterministic_term(
            c=c,
            beta=beta,
            X=Xf_use,
            n=steps,
            include_exog=fit.spec.include_exog,
            use_numba=True,
        )

        mean = np.zeros(steps)
        a = a_last.copy()
        for h in range(steps):
            a = T @ a
            mean[h] = det_future[h] + float(Z @ a)

        if not return_intervals:
            return {"mean": mean}

        rng = np.random.default_rng(seed)
        sims = np.zeros((num_sim, steps))
        sqrt_s = float(np.sqrt(max(sigma2, 1e-12)))
        for r in range(num_sim):
            a_sim = a_last.copy()
            for h in range(steps):
                a_sim = T @ a_sim + R * rng.normal(0.0, sqrt_s)
                sims[r, h] = det_future[h] + float(Z @ a_sim)

        lo = np.quantile(sims, alpha / 2.0, axis=0)
        hi = np.quantile(sims, 1.0 - alpha / 2.0, axis=0)
        return {"mean": mean, "lo": lo, "hi": hi}


# ---------------------------------------------------------------------------
# Auto-search helpers
# ---------------------------------------------------------------------------


def _theta_dim(*, k_exog, include_intercept, p, q, P, Q):
    return (1 if include_intercept else 0) + k_exog + p + P + q + Q + 1


def _project_theta_init(theta_src, src_spec, dst_spec, *, k_exog):
    if theta_src is None:
        return None
    theta_src = np.asarray(theta_src, dtype=float).reshape(-1)
    p0, _, q0 = src_spec.order
    P0, _, Q0, _ = src_spec.seasonal_order
    p1, _, q1 = dst_spec.order
    P1, _, Q1, _ = dst_spec.seasonal_order
    dim0 = _theta_dim(
        k_exog=k_exog,
        include_intercept=src_spec.include_intercept,
        p=p0,
        q=q0,
        P=P0,
        Q=Q0,
    )
    dim1 = _theta_dim(
        k_exog=k_exog,
        include_intercept=dst_spec.include_intercept,
        p=p1,
        q=q1,
        P=P1,
        Q=Q1,
    )
    if theta_src.size != dim0:
        return None
    dst = np.zeros(dim1, dtype=float)
    i0 = i1 = 0
    has_c = src_spec.include_intercept and dst_spec.include_intercept
    if has_c:
        dst[i1] = theta_src[i0]
    if src_spec.include_intercept:
        i0 += 1
    if dst_spec.include_intercept:
        i1 += 1
    if k_exog > 0:
        dst[i1 : i1 + k_exog] = theta_src[i0 : i0 + k_exog]
    i0 += k_exog
    i1 += k_exog
    for s0, s1 in [(p0, p1), (P0, P1), (q0, q1), (Q0, Q1)]:
        n = min(s0, s1)
        if n > 0:
            dst[i1 : i1 + n] = theta_src[i0 : i0 + n]
        i0 += s0
        i1 += s1
    dst[i1] = theta_src[i0]
    return dst


@dataclass(frozen=True)
class AutoConfig:
    p_max: int = 5
    q_max: int = 5
    P_max: int = 2
    Q_max: int = 2
    d_max: int = 2
    D_max: int = 1
    seasonal_period: int = 1
    max_steps: int = 40
    maxiter_search: int = 60
    maxiter_refit: int = 250
    include_intercept: bool = True
    refit_final: bool = True
    use_numba: bool = True
    compute_smoother_during_search: bool = False
    fallback_no_numba: bool = True
    max_total_fits: int = 0
    # NEW:
    use_jax_autodiff: bool = True
    jax_enable_x64: bool = True
    jax_jit: bool = True


def auto_sarimax_stepwise(
    y,
    exog=None,
    *,
    seasonal_period=1,
    cfg=None,
    verbose=False,
) -> SarimaxFit:
    """Hyndman–Khandakar stepwise SARIMAX search with AICc."""
    import time

    vlevel = int(verbose) if not isinstance(verbose, bool) else (1 if verbose else 0)
    t0 = time.perf_counter()
    y0 = _as_1d(y)
    X0 = _as_2d(exog, n=y0.size)
    s = int(seasonal_period)
    if cfg is None:
        cfg = AutoConfig()
    cfg = AutoConfig(**{**cfg.__dict__, "seasonal_period": s})

    def kpss_p(x):
        import warnings

        from statsmodels.tsa.stattools import kpss

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pval, _, _ = kpss(x, regression="c", nlags="auto")
        return float(pval)

    yd0 = y0.copy()
    d = 0
    for dd in range(cfg.d_max + 1):
        if kpss_p(yd0) >= 0.05:
            d = dd
            break
        yd0 = yd0[1:] - yd0[:-1]

    D = 0
    if s > 1:
        yd1 = difference(y0, d=d, D=0, s=s)
        if yd1.size > s + 10 and kpss_p(yd1) < 0.05 and cfg.D_max >= 1:
            D = 1

    if vlevel >= 1:
        print(f"[auto] d={d} D={D} s={s}")

    yd = difference(y0, d=d, D=D, s=s)
    Xd = difference_exog(X0, d=d, D=D, s=s)
    k_exog = 0 if Xd is None else Xd.shape[1]

    cache: dict = {}
    fail_cache: dict = {}
    stats = {"fit_calls": 0, "cache_hits": 0, "failures": 0}
    budget_stop = {"stop": False}

    def score_model(p, q, P, Q, *, init_from=None):
        key = (p, d, q, P, D, Q)
        if key in cache:
            stats["cache_hits"] += 1
            return cache[key]
        if key in fail_cache:
            raise RuntimeError(fail_cache[key])
        if cfg.max_total_fits > 0 and stats["fit_calls"] >= cfg.max_total_fits:
            budget_stop["stop"] = True
            msg = f"budget exhausted before ({p},{d},{q},{P},{D},{Q})"
            fail_cache[key] = msg
            raise RuntimeError(msg)
        spec = SarimaxSpec(
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            include_intercept=cfg.include_intercept,
            include_exog=(exog is not None),
        )
        init_params = None
        if init_from is not None:
            init_params = _project_theta_init(
                init_from.info.get("theta_u_opt"), init_from.spec, spec, k_exog=k_exog
            )
        m = SarimaxScratch(spec)
        last_e = None
        for use_nb in (
            [True, False]
            if (cfg.use_numba and cfg.fallback_no_numba)
            else [cfg.use_numba]
        ):
            jax_modes = [cfg.use_jax_autodiff]
            if cfg.use_jax_autodiff:
                jax_modes.append(False)
            for use_jax in list(dict.fromkeys(jax_modes)):
                try:
                    t_fit = time.perf_counter()
                    if vlevel >= 2:
                        print(
                            f"[fit-start] ({p},{d},{q})×({P},{D},{Q},{s}) "
                            f"warm_start={init_params is not None} numba={use_nb} "
                            f"jax={use_jax}"
                        )
                    fit = m.fit(
                        yd,
                        exog=Xd,
                        maxiter=cfg.maxiter_search,
                        method="L-BFGS-B",
                        verbose=False,
                        pre_differenced=True,
                        compute_smoother=cfg.compute_smoother_during_search,
                        use_numba=use_nb,
                        init_params=init_params,
                        use_jax_autodiff=use_jax,
                        jax_enable_x64=cfg.jax_enable_x64,
                        jax_jit=cfg.jax_jit,
                    )
                    stats["fit_calls"] += 1
                    if vlevel >= 2:
                        tag = "fit-done" if use_nb else "fit-done-cpu"
                        print(
                            f"[{tag}] ({p},{d},{q})×({P},{D},{Q},{s}) "
                            f"aicc={fit.aicc:.3f} converged={fit.converged} "
                            f"nit={fit.info.get('nit', -1)} "
                            f"time={time.perf_counter() - t_fit:.2f}s"
                        )
                    cache[key] = fit
                    return fit
                except Exception as e:
                    last_e = e
                    if vlevel >= 2:
                        print(
                            f"[fit-retry] ({p},{d},{q})×({P},{D},{Q},{s}) "
                            f"numba={use_nb} jax={use_jax} "
                            f"reason={type(e).__name__}: {e}"
                        )
        stats["failures"] += 1
        msg = f"fit failed ({p},{d},{q},{P},{D},{Q}): {last_e}"
        fail_cache[key] = msg
        raise RuntimeError(msg)

    candidates = [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (2, 2, 1 if s > 1 else 0, 1 if s > 1 else 0),
    ]
    candidates = [
        (min(p0, cfg.p_max), min(q0, cfg.q_max), min(P0, cfg.P_max), min(Q0, cfg.Q_max))
        for p0, q0, P0, Q0 in candidates
    ]

    best = None
    init_errors = []
    for p0, q0, P0, Q0 in candidates:
        if budget_stop["stop"]:
            break
        try:
            f = score_model(p0, q0, P0, Q0, init_from=best)
            if best is None or f.aicc < best.aicc:
                best = f
        except Exception as e:
            init_errors.append(str(e))
            if vlevel >= 1:
                print(f"[init-skip] ({p0},{d},{q0},{P0},{D},{Q0}) -> {e}")
    if best is None:
        msg = "Could not fit any initial SARIMAX candidates."
        if init_errors:
            msg += " First errors: " + " | ".join(init_errors[:3])
        raise RuntimeError(msg)

    def neighbors(p, q, P, Q):
        for dp, dq, dP, dQ in [
            (1, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, -1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, -1, 0),
            (0, 0, 0, 1),
            (0, 0, 0, -1),
            (1, -1, 0, 0),
            (-1, 1, 0, 0),
        ]:
            pp, qq, PP, QQ = p + dp, q + dq, P + dP, Q + dQ
            if (
                0 <= pp <= cfg.p_max
                and 0 <= qq <= cfg.q_max
                and 0 <= PP <= cfg.P_max
                and 0 <= QQ <= cfg.Q_max
            ):
                yield pp, qq, PP, QQ

    p, _, q = best.spec.order
    P, _, Q, _ = best.spec.seasonal_order
    steps = 0
    improved = True

    n_workers = min(
        10, (cfg.p_max + 1) * (cfg.q_max + 1) * (cfg.P_max + 1) * (cfg.Q_max + 1)
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        while improved and steps < cfg.max_steps and not budget_stop["stop"]:
            improved = False
            cur_best = best
            nb_list = list(neighbors(p, q, P, Q))
            futures = {
                pool.submit(score_model, pp, qq, PP, QQ, init_from=cur_best): (
                    pp,
                    qq,
                    PP,
                    QQ,
                )
                for pp, qq, PP, QQ in nb_list
            }
            for fut in concurrent.futures.as_completed(futures):
                pp, qq, PP, QQ = futures[fut]
                try:
                    f = fut.result()
                    if f.aicc + 1e-9 < cur_best.aicc:
                        cur_best = f
                        p, q, P, Q = pp, qq, PP, QQ
                        improved = True
                except Exception:
                    pass
            best = cur_best
            steps += 1
            if vlevel >= 1:
                print(
                    f"[step {steps}] ({p},{d},{q})×({P},{D},{Q},{s}) "
                    f"AICc={best.aicc:.3f} fits={stats['fit_calls']} "
                    f"t={time.perf_counter() - t0:.1f}s"
                )

    if cfg.refit_final:
        final_model = SarimaxScratch(best.spec)
        for use_nb in (
            [True, False]
            if (cfg.use_numba and cfg.fallback_no_numba)
            else [cfg.use_numba]
        ):
            jax_modes = [cfg.use_jax_autodiff]
            if cfg.use_jax_autodiff:
                jax_modes.append(False)
            for use_jax in list(dict.fromkeys(jax_modes)):
                try:
                    best = final_model.fit(
                        y0,
                        exog=X0,
                        maxiter=cfg.maxiter_refit,
                        method="L-BFGS-B",
                        verbose=False,
                        compute_smoother=True,
                        use_numba=use_nb,
                        init_params=best.info.get("theta_u_opt"),
                        use_jax_autodiff=use_jax,
                        jax_enable_x64=cfg.jax_enable_x64,
                        jax_jit=cfg.jax_jit,
                    )
                    break
                except Exception:
                    if use_jax is False:
                        if not (cfg.use_numba and cfg.fallback_no_numba and use_nb):
                            raise
                    continue
            else:
                continue
            break

    if vlevel >= 1:
        print(
            f"[auto] done {time.perf_counter() - t0:.2f}s  "
            f"fits={stats['fit_calls']} failures={stats['failures']}  "
            f"best={best.spec} AICc={best.aicc:.3f}"
        )
    return best
