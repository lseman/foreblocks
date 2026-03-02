from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
except Exception as e:
    raise ImportError(
        "This implementation requires SciPy (scipy.optimize.minimize)."
    ) from e

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(func):
            return func

        return _wrap


_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------
# Helpers: input shaping, differencing, polynomial transforms, AICc
# ---------------------------------------------------------------------


def _as_1d(y) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size < 30:
        raise ValueError(
            "Need at least ~30 finite observations for robust SARIMAX fitting."
        )
    return y


def _as_2d(X, n: int) -> Optional[np.ndarray]:
    if X is None:
        return None
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != n:
        raise ValueError(f"exog has {X.shape[0]} rows but y has {n}.")
    X = np.where(np.isfinite(X), X, 0.0)
    return X


def difference(y: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    out = np.asarray(y, dtype=float)
    for _ in range(d):
        out = out[1:] - out[:-1]
    for _ in range(D):
        if s <= 1:
            raise ValueError("Seasonal differencing D>0 requires seasonal period s>=2.")
        out = out[s:] - out[:-s]
    return out


def difference_exog(
    X: Optional[np.ndarray], d: int, D: int, s: int
) -> Optional[np.ndarray]:
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
    denom = max(n - k - 1, 1)
    return aic + (2.0 * k * (k + 1)) / denom


def _trim_trailing_small(x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    end = x.size
    while end > 0 and abs(x[end - 1]) <= tol:
        end -= 1
    return x[:end]


def _constrain_stationary_py(raw: np.ndarray) -> np.ndarray:
    """
    Map unconstrained params -> stationary AR coefficients using
    partial autocorrelation recursion (Monahan transform).
    """
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
    # Invertibility of 1 + theta(L) is stationarity of 1 - (-theta)(L).
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
    poly = np.convolve(ns, seas)
    return _trim_trailing_small(-poly[1:])


def _combine_ma_lags_py(ma: np.ndarray, sma: np.ndarray, s: int) -> np.ndarray:
    ns = np.concatenate(([1.0], np.asarray(ma, dtype=float)))
    seas = _seasonal_poly_py(np.asarray(sma, dtype=float), s=s, sign=+1.0)
    poly = np.convolve(ns, seas)
    return _trim_trailing_small(poly[1:])


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
            lag = (j + 1) * s
            poly[i + lag] += ns_i * (-sar[j])

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
            lag = (j + 1) * s
            poly[i + lag] += ns_i * sma[j]

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


def _combine_ar_lags(
    ar: np.ndarray, sar: np.ndarray, s: int, *, use_numba: bool = True
) -> np.ndarray:
    ar = np.asarray(ar, dtype=float).reshape(-1)
    sar = np.asarray(sar, dtype=float).reshape(-1)
    if use_numba and _HAS_NUMBA:
        return np.asarray(
            _combine_ar_lags_nb(
                np.ascontiguousarray(ar, dtype=np.float64),
                np.ascontiguousarray(sar, dtype=np.float64),
                int(s),
            ),
            dtype=float,
        )
    return _combine_ar_lags_py(ar, sar, s)


def _combine_ma_lags(
    ma: np.ndarray, sma: np.ndarray, s: int, *, use_numba: bool = True
) -> np.ndarray:
    ma = np.asarray(ma, dtype=float).reshape(-1)
    sma = np.asarray(sma, dtype=float).reshape(-1)
    if use_numba and _HAS_NUMBA:
        return np.asarray(
            _combine_ma_lags_nb(
                np.ascontiguousarray(ma, dtype=np.float64),
                np.ascontiguousarray(sma, dtype=np.float64),
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


def _deterministic_term(
    c: float,
    beta: np.ndarray,
    X: Optional[np.ndarray],
    n: int,
    *,
    include_exog: bool,
    use_numba: bool,
) -> np.ndarray:
    if beta.size > 0 and X is not None and include_exog:
        if use_numba and _HAS_NUMBA:
            return np.asarray(
                _deterministic_term_nb(
                    float(c),
                    np.ascontiguousarray(X, dtype=np.float64),
                    np.ascontiguousarray(beta, dtype=np.float64),
                ),
                dtype=float,
            )
        out = np.full(n, float(c), dtype=float)
        out += X @ beta
        return out
    return np.full(n, float(c), dtype=float)


def _autocovariances(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    g = np.zeros(max_lag + 1, dtype=float)
    if n == 0:
        return g
    xc = x - float(np.mean(x))
    for lag in range(max_lag + 1):
        m = n - lag
        if m <= 0:
            break
        g[lag] = float(np.dot(xc[lag:], xc[:m])) / float(m)
    return g


def _levinson_durbin(gamma: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    if order <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    if gamma.size < order + 1 or gamma[0] <= 1e-12:
        return np.zeros(order, dtype=float), np.zeros(order, dtype=float)

    phi = np.zeros((order, order), dtype=float)
    pacf = np.zeros(order, dtype=float)
    sigma = float(gamma[0])

    for k in range(1, order + 1):
        num = float(gamma[k])
        if k > 1:
            for j in range(1, k):
                num -= phi[k - 2, j - 1] * gamma[k - j]
        den = max(sigma, 1e-12)
        kk = np.clip(num / den, -0.98, 0.98)
        pacf[k - 1] = kk
        phi[k - 1, k - 1] = kk
        if k > 1:
            for j in range(1, k):
                phi[k - 1, j - 1] = phi[k - 2, j - 1] - kk * phi[k - 2, k - j - 1]
        sigma = sigma * max(1.0 - kk * kk, 1e-6)

    return phi[order - 1, :].copy(), pacf


def _stationary_coeffs_to_raw(phi: np.ndarray) -> np.ndarray:
    """
    Approximate inverse of the Monahan transform:
    stationary AR coeffs -> unconstrained raw params.
    """
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
    Xd: Optional[np.ndarray],
    *,
    has_c: bool,
    include_exog: bool,
    p: int,
    q: int,
    P: int,
    Q: int,
    s: int,
) -> np.ndarray:
    """
    Data-driven initial values for unconstrained optimizer vector:
    [c?] [beta] [raw_ar] [raw_sar] [raw_ma] [raw_sma] [log_sigma2]
    """
    yd = np.asarray(yd, dtype=float).reshape(-1)
    n = yd.size
    k_exog = 0 if (Xd is None or not include_exog) else Xd.shape[1]

    c0 = 0.0
    beta0 = np.zeros(k_exog, dtype=float)

    # OLS for deterministic part.
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
            beta0 = np.zeros(k_exog, dtype=float)

    det = np.full(n, c0, dtype=float)
    if k_exog > 0:
        det += np.asarray(Xd, dtype=float) @ beta0
    resid = yd - det

    # AR init via Levinson-Durbin (Yule-Walker equivalent).
    ar_coeff = np.zeros(p, dtype=float)
    pacf_ar = np.zeros(p, dtype=float)
    if p > 0:
        g_ar = _autocovariances(resid, p)
        ar_coeff, pacf_ar = _levinson_durbin(g_ar, p)
    raw_ar0 = np.arctanh(np.clip(pacf_ar, -0.98, 0.98))

    # Residual after simple AR fit, used for MA/seasonal seeds and sigma2.
    resid_ar = resid.copy()
    if p > 0:
        for t in range(p, n):
            pred = 0.0
            for j in range(p):
                pred += ar_coeff[j] * resid[t - j - 1]
            resid_ar[t] = resid[t] - pred

    max_lag = 0
    if q > 0:
        max_lag = max(max_lag, q)
    if P > 0:
        max_lag = max(max_lag, P * s)
    if Q > 0:
        max_lag = max(max_lag, Q * s)
    g_res = _autocovariances(resid_ar, max_lag) if max_lag > 0 else np.zeros(1, dtype=float)
    g0 = max(float(g_res[0]), 1e-12)

    def rho_at(lag: int) -> float:
        if lag <= 0 or lag >= g_res.size:
            return 0.0
        return float(np.clip(g_res[lag] / g0, -0.98, 0.98))

    def snap_to_grid(val: float, grid: np.ndarray) -> float:
        idx = int(np.argmin(np.abs(grid - val)))
        return float(grid[idx])

    # MA init from residual autocorrelation + small grid around zero.
    ma_grid = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6], dtype=float)
    ma_target = np.zeros(q, dtype=float)
    for i in range(q):
        ma_target[i] = snap_to_grid(0.35 * rho_at(i + 1), ma_grid)
    raw_ma0 = _stationary_coeffs_to_raw(-np.clip(ma_target, -0.95, 0.95))

    # Seasonal AR/MA init from seasonal lags, with fallback copy from non-seasonal.
    sar_target = np.zeros(P, dtype=float)
    for i in range(P):
        sar_target[i] = 0.5 * rho_at((i + 1) * s)
        if abs(sar_target[i]) < 0.05 and p > 0:
            sar_target[i] = 0.5 * ar_coeff[min(i, p - 1)]
    raw_sar0 = _stationary_coeffs_to_raw(np.clip(sar_target, -0.95, 0.95))

    sma_target = np.zeros(Q, dtype=float)
    for i in range(Q):
        sma_target[i] = snap_to_grid(0.35 * rho_at((i + 1) * s), ma_grid)
        if abs(sma_target[i]) < 0.05 and q > 0:
            sma_target[i] = 0.5 * ma_target[min(i, q - 1)]
    raw_sma0 = _stationary_coeffs_to_raw(-np.clip(sma_target, -0.95, 0.95))

    start = max(p, 1)
    sigma2_0 = float(np.var(resid_ar[start:])) if n > start else float(np.var(resid_ar))
    sigma2_0 = max(sigma2_0, 1e-6)

    theta0 = []
    if has_c:
        theta0.append(float(c0))
    if k_exog > 0:
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
) -> Tuple[float, int]:
    """
    Fast likelihood-only Kalman recursion used inside optimization.
    Compiled with numba when available.
    """
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


# ---------------------------------------------------------------------
# Public spec / fit dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SarimaxSpec:
    order: Tuple[int, int, int]  # (p,d,q)
    seasonal_order: Tuple[int, int, int, int]  # (P,D,Q,s)
    include_intercept: bool = True  # intercept on differenced series
    include_exog: bool = True  # use exog if provided


@dataclass
class SarimaxFit:
    spec: SarimaxSpec
    params: Dict[
        str, np.ndarray
    ]  # {"ar","sar","ma","sma","beta","c","sigma2","phi","theta"}
    nll: float
    aicc: float
    converged: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------
# State-space SARIMAX with diffuse Kalman filter/smoother and MLE
# ---------------------------------------------------------------------


class SarimaxScratch:
    """
    SARIMAX via state-space + Kalman filter on the differenced series.

    Model (after applying d and D differencing to y and exog):
      y~_t = c + beta^T x~_t + w_t
      phi(L) w_t = theta(L) eps_t, eps_t ~ N(0, sigma2)

    State-space uses an ARMA companion form:
      alpha_{t+1} = T alpha_t + R eps_{t+1}
      w_t         = Z alpha_t
      y~_t        = d_t + Z alpha_t, d_t = c + beta^T x~_t

    Initial covariance is approximate diffuse: P0 = kappa * I.
    """

    _SS_TEMPLATE_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def __init__(self, spec: SarimaxSpec):
        self.spec = spec
        self.fit_: Optional[SarimaxFit] = None

    @classmethod
    def _get_state_space_template(cls, p_full: int, q_full: int) -> Dict[str, Any]:
        key = (int(p_full), int(q_full))
        tpl = cls._SS_TEMPLATE_CACHE.get(key)
        if tpl is not None:
            return tpl

        p_full = int(p_full)
        q_full = int(q_full)
        r = max(1, p_full, q_full + 1)
        m = r + q_full

        T_base = np.zeros((m, m), dtype=float)
        Z = np.zeros(m, dtype=float)
        R = np.zeros(m, dtype=float)
        Z[0] = 1.0
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

        m_arr = np.array([m], dtype=int)
        r_arr = np.array([r], dtype=int)
        p_arr = np.array([p_full], dtype=int)
        q_arr = np.array([q_full], dtype=int)
        m_arr.setflags(write=False)
        r_arr.setflags(write=False)
        p_arr.setflags(write=False)
        q_arr.setflags(write=False)

        tpl = {
            "T_base": T_base,
            "Z": Z,
            "R": R,
            "m": m_arr,
            "r": r_arr,
            "p_full": p_arr,
            "q_full": q_arr,
            "r_int": r,
        }
        cls._SS_TEMPLATE_CACHE[key] = tpl
        return tpl

    @classmethod
    def _build_state_space(cls, phi: np.ndarray, theta: np.ndarray) -> Dict[str, np.ndarray]:
        phi = np.asarray(phi, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)

        p_full = int(phi.size)
        q_full = int(theta.size)
        tpl = cls._get_state_space_template(p_full=p_full, q_full=q_full)

        T = np.asarray(tpl["T_base"], dtype=float).copy()
        r = int(tpl["r_int"])

        if p_full > 0:
            T[0, :p_full] = phi
        if q_full > 0:
            T[0, r : r + q_full] = theta

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
    def _resolve_diffuse_burn(
        n: int, p_full: int, q_full: int, diffuse_burn: Optional[int] = None
    ) -> int:
        """
        Default burn-in for approximate diffuse initialization.
        If user sets diffuse_burn, it is used directly (clipped to [0, n-1]).
        """
        if diffuse_burn is not None:
            burn = int(diffuse_burn)
        else:
            burn = max(10, int(1.5 * max(int(p_full), int(q_full) + 1)))
        return min(max(burn, 0), max(n - 1, 0))

    @staticmethod
    def _kalman_filter(
        y_adj: np.ndarray,
        T: np.ndarray,
        Z: np.ndarray,
        R: np.ndarray,
        sigma2: float,
        *,
        diffuse_scale: float,
        diffuse_burn: int,
        store: bool,
        use_numba: bool = True,
    ) -> Dict[str, Any]:
        y_adj = np.asarray(y_adj, dtype=float).reshape(-1)
        n = y_adj.size
        m = T.shape[0]

        if not store and use_numba:
            nll, n_eff = _kalman_nll_kernel(
                y_adj=y_adj.astype(np.float64, copy=False),
                T=np.asarray(T, dtype=np.float64),
                Z=np.asarray(Z, dtype=np.float64),
                R=np.asarray(R, dtype=np.float64),
                sigma2=float(sigma2),
                diffuse_scale=float(diffuse_scale),
                diffuse_burn=int(diffuse_burn),
            )
            return {
                "nll": float(nll),
                "n_eff": int(n_eff),
                "v": np.zeros(0, dtype=float),
                "F": np.zeros(0, dtype=float),
            }

        a_pred = np.zeros(m, dtype=float)
        P_pred = np.eye(m, dtype=float) * float(diffuse_scale)
        RRt = np.outer(R, R) * float(sigma2)

        if store:
            a_pred_hist = np.zeros((n, m), dtype=float)
            a_filt_hist = np.zeros((n, m), dtype=float)
            P_pred_hist = np.zeros((n, m, m), dtype=float)
            P_filt_hist = np.zeros((n, m, m), dtype=float)
        else:
            a_pred_hist = a_filt_hist = None
            P_pred_hist = P_filt_hist = None

        v = np.zeros(n, dtype=float)
        F = np.zeros(n, dtype=float)

        nll = 0.0
        n_eff = 0

        for t in range(n):
            if store:
                a_pred_hist[t] = a_pred
                P_pred_hist[t] = P_pred

            vt = y_adj[t] - float(Z @ a_pred)
            Ft = float(Z @ P_pred @ Z)
            Ft = max(Ft, 1e-12)
            K = (P_pred @ Z) / Ft

            a_filt = a_pred + K * vt
            P_filt = P_pred - np.outer(K, K) * Ft
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

        out: Dict[str, Any] = {"nll": float(nll), "n_eff": int(n_eff), "v": v, "F": F}
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
    def _kalman_smoother(T: np.ndarray, filt: Dict[str, Any]) -> Dict[str, np.ndarray]:
        a_pred = np.asarray(filt["a_pred"], dtype=float)
        a_filt = np.asarray(filt["a_filt"], dtype=float)
        P_pred = np.asarray(filt["P_pred"], dtype=float)
        P_filt = np.asarray(filt["P_filt"], dtype=float)

        n, m = a_filt.shape
        a_smooth = np.zeros((n, m), dtype=float)
        P_smooth = np.zeros((n, m, m), dtype=float)

        a_smooth[-1] = a_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for t in range(n - 2, -1, -1):
            Pn = P_pred[t + 1]
            J = P_filt[t] @ T.T @ np.linalg.pinv(Pn, rcond=1e-12)
            a_smooth[t] = a_filt[t] + J @ (a_smooth[t + 1] - a_pred[t + 1])
            P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - Pn) @ J.T
            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

        return {"a_smooth": a_smooth, "P_smooth": P_smooth}

    def _decode_params(
        self,
        theta_u: np.ndarray,
        k_exog: int,
        *,
        use_numba: bool = True,
    ) -> Dict[str, np.ndarray]:
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

        raw_ar = np.asarray(theta_u[idx : idx + p], dtype=float)
        idx += p
        raw_sar = np.asarray(theta_u[idx : idx + P], dtype=float)
        idx += P
        raw_ma = np.asarray(theta_u[idx : idx + q], dtype=float)
        idx += q
        raw_sma = np.asarray(theta_u[idx : idx + Q], dtype=float)
        idx += Q

        log_sigma2 = float(theta_u[idx])
        sigma2 = float(np.exp(log_sigma2) + 1e-12)

        ar = _constrain_stationary(raw_ar, use_numba=use_numba)
        sar = _constrain_stationary(raw_sar, use_numba=use_numba)
        ma = _constrain_invertible(raw_ma, use_numba=use_numba)
        sma = _constrain_invertible(raw_sma, use_numba=use_numba)

        phi = _combine_ar_lags(ar, sar, s=s, use_numba=use_numba)
        theta = _combine_ma_lags(ma, sma, s=s, use_numba=use_numba)

        ss = self._build_state_space(phi=phi, theta=theta)
        return {
            "c": np.array([c], dtype=float),
            "beta": beta.astype(float),
            "ar": ar.astype(float),
            "sar": sar.astype(float),
            "ma": ma.astype(float),
            "sma": sma.astype(float),
            "phi": phi.astype(float),
            "theta": theta.astype(float),
            "sigma2": np.array([sigma2], dtype=float),
            "T": ss["T"],
            "Z": ss["Z"],
            "R": ss["R"],
            "m": ss["m"],
            "r": ss["r"],
            "p_full": ss["p_full"],
            "q_full": ss["q_full"],
        }

    def fit(
        self,
        y,
        exog=None,
        *,
        maxiter: int = 300,
        method: str = "L-BFGS-B",
        verbose: bool = False,
        seed: int = 0,
        diffuse_scale: float = 1e6,
        diffuse_burn: Optional[int] = None,
        pre_differenced: bool = False,
        compute_smoother: bool = True,
        use_numba: bool = True,
        init_params: Optional[np.ndarray] = None,
    ) -> SarimaxFit:
        _ = seed  # kept for API compatibility
        y0 = _as_1d(y)
        X0 = _as_2d(exog, n=y0.size)

        p, d, q = self.spec.order
        P, D, Q, s = self.spec.seasonal_order
        if s < 1:
            raise ValueError("seasonal period s must be >= 1.")
        if (P > 0 or Q > 0 or D > 0) and s == 1:
            # Allowed but effectively non-seasonal.
            pass

        if pre_differenced:
            yd = y0
            Xd = X0
        else:
            yd = difference(y0, d=d, D=D, s=s)
            Xd = difference_exog(X0, d=d, D=D, s=s)
        n = yd.size
        if n < 10:
            raise ValueError("Series too short after differencing.")

        k_exog = 0 if (Xd is None or not self.spec.include_exog) else Xd.shape[1]
        has_c = bool(self.spec.include_intercept)

        # [c?] [beta...] [raw_ar...] [raw_sar...] [raw_ma...] [raw_sma...] [log_sigma2]
        dim = (1 if has_c else 0) + k_exog + p + P + q + Q + 1
        if init_params is not None:
            theta0 = np.asarray(init_params, dtype=float).reshape(-1)
            if theta0.size != dim:
                raise ValueError(f"init_params has size {theta0.size}, expected {dim}.")
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
                # Safety fallback to basic init if heuristic vector mismatches.
                theta0 = np.zeros(dim, dtype=float)
                idx = 0
                if has_c:
                    theta0[idx] = float(np.mean(yd))
                    idx += 1
                idx += k_exog + p + P + q + Q
                theta0[idx] = float(np.log(np.var(yd) + 1e-6))

        Xd_numba = (
            None if Xd is None else np.ascontiguousarray(Xd, dtype=np.float64)
        )

        def deterministic_term(c: float, beta: np.ndarray) -> np.ndarray:
            return _deterministic_term(
                c=c,
                beta=beta,
                X=Xd_numba if (use_numba and Xd_numba is not None) else Xd,
                n=n,
                include_exog=self.spec.include_exog,
                use_numba=use_numba,
            )

        def objective(theta_u: np.ndarray) -> float:
            dec = self._decode_params(theta_u=theta_u, k_exog=k_exog, use_numba=use_numba)
            y_adj = yd - deterministic_term(float(dec["c"][0]), dec["beta"])
            p_full = int(dec["p_full"][0])
            q_full = int(dec["q_full"][0])
            burn = self._resolve_diffuse_burn(
                n=n, p_full=p_full, q_full=q_full, diffuse_burn=diffuse_burn
            )
            filt = self._kalman_filter(
                y_adj=y_adj,
                T=dec["T"],
                Z=dec["Z"],
                R=dec["R"],
                sigma2=float(dec["sigma2"][0]),
                diffuse_scale=diffuse_scale,
                diffuse_burn=burn,
                store=False,
                use_numba=use_numba,
            )
            val = float(filt["nll"])
            if not np.isfinite(val):
                return 1e50
            return val

        def objective_and_grad(theta_u: np.ndarray) -> Tuple[float, np.ndarray]:
            grad = np.zeros_like(theta_u)
            eps = 1e-6
            
            val = objective(theta_u)
            
            for i in range(len(theta_u)):
                theta_plus = theta_u.copy()
                theta_plus[i] += eps
                val_plus = objective(theta_plus)
                grad[i] = (val_plus - val) / eps
                
            return val, grad

        res = minimize(
            objective_and_grad,
            theta0,
            method=method,
            jac=True,
            options={"maxiter": int(maxiter), "disp": bool(verbose)},
        )

        dec = self._decode_params(
            theta_u=np.asarray(res.x, dtype=float),
            k_exog=k_exog,
            use_numba=use_numba,
        )
        y_adj = yd - deterministic_term(float(dec["c"][0]), dec["beta"])
        p_full = int(dec["p_full"][0])
        q_full = int(dec["q_full"][0])
        burn = self._resolve_diffuse_burn(
            n=n, p_full=p_full, q_full=q_full, diffuse_burn=diffuse_burn
        )

        filt = self._kalman_filter(
            y_adj=y_adj,
            T=dec["T"],
            Z=dec["Z"],
            R=dec["R"],
            sigma2=float(dec["sigma2"][0]),
            diffuse_scale=diffuse_scale,
            diffuse_burn=burn,
            store=compute_smoother,
            use_numba=use_numba,
        )
        smooth = self._kalman_smoother(T=dec["T"], filt=filt) if compute_smoother else None

        nll_hat = float(filt["nll"])
        n_eff = int(filt["n_eff"])
        score = float(aicc(n_eff, dim, nll_hat))

        info: Dict[str, Any] = {
            "n_obs": int(n),
            "n_eff": n_eff,
            "state_dim": int(dec["m"][0]),
            "ar_lag_dim": int(dec["r"][0]),
            "ma_lag_dim": int(dec["q_full"][0]),
            "diffuse_scale": float(diffuse_scale),
            "diffuse_burn": int(burn),
            "optimizer": method,
            "message": str(res.message),
            "nit": int(getattr(res, "nit", -1)),
            "numba_used": bool(use_numba and _HAS_NUMBA),
            "pre_differenced": bool(pre_differenced),
            "compute_smoother": bool(compute_smoother),
            "theta_u_opt": np.asarray(res.x, dtype=float),
            "state_space": {
                "T": dec["T"],
                "Z": dec["Z"],
                "R": dec["R"],
            },
        }
        if compute_smoother:
            info["innovations"] = filt["v"]
            info["innovation_var"] = filt["F"]
            info["filtered_state"] = filt["a_filt"]
            info["smoothed_state"] = smooth["a_smooth"] if smooth is not None else None

        fit = SarimaxFit(
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
            converged=bool(res.success),
            info=info,
        )

        self.fit_ = fit
        return fit

    def filter_smoother(self, y, exog=None) -> Dict[str, np.ndarray]:
        """
        Run Kalman filter + RTS smoother with fitted params on provided data.
        """
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
        phi = fit.params["phi"]
        theta = fit.params["theta"]
        sigma2 = float(fit.params["sigma2"][0])

        ss = self._build_state_space(phi=phi, theta=theta)
        T = ss["T"]
        Z = ss["Z"]
        R = ss["R"]
        p_full = int(ss["p_full"][0])
        q_full = int(ss["q_full"][0])

        det = _deterministic_term(
            c=c,
            beta=beta,
            X=Xd,
            n=n,
            include_exog=fit.spec.include_exog,
            use_numba=True,
        )
        y_adj = yd - det

        burn = self._resolve_diffuse_burn(n=n, p_full=p_full, q_full=q_full)
        filt = self._kalman_filter(
            y_adj=y_adj,
            T=T,
            Z=Z,
            R=R,
            sigma2=sigma2,
            diffuse_scale=1e6,
            diffuse_burn=burn,
            store=True,
        )
        smooth = self._kalman_smoother(T=T, filt=filt)

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
        return_intervals: bool = True,
        alpha: float = 0.05,
        num_sim: int = 2000,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Forecasts are returned on the differenced scale.
        """
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
            if (d > 0 or D > 0):
                raise NotImplementedError(
                    "When d>0 or D>0, exog_future should be provided in differenced "
                    "space externally before calling forecast."
                )

        c = float(fit.params["c"][0]) if fit.spec.include_intercept else 0.0
        beta = fit.params["beta"]
        phi = fit.params["phi"]
        theta = fit.params["theta"]
        sigma2 = float(fit.params["sigma2"][0])

        ss = self._build_state_space(phi=phi, theta=theta)
        T = ss["T"]
        Z = ss["Z"]
        R = ss["R"]
        p_full = int(ss["p_full"][0])
        q_full = int(ss["q_full"][0])

        det = _deterministic_term(
            c=c,
            beta=beta,
            X=Xd,
            n=n,
            include_exog=fit.spec.include_exog,
            use_numba=True,
        )
        y_adj = yd - det

        burn = self._resolve_diffuse_burn(n=n, p_full=p_full, q_full=q_full)
        filt = self._kalman_filter(
            y_adj=y_adj,
            T=T,
            Z=Z,
            R=R,
            sigma2=sigma2,
            diffuse_scale=1e6,
            diffuse_burn=burn,
            store=True,
        )

        a_last = filt["a_filt"][-1].copy()

        Xf_use = None
        if beta.size > 0 and fit.spec.include_exog:
            if Xf is None:
                raise ValueError(
                    "Model was fitted with exog; exog_future is required for forecast."
                )
            if Xf.shape[1] != beta.size:
                raise ValueError("exog_future has incompatible number of columns.")
            Xf_use = Xf
        det_future = _deterministic_term(
            c=c,
            beta=beta,
            X=Xf_use,
            n=steps,
            include_exog=fit.spec.include_exog,
            use_numba=True,
        )

        mean = np.zeros(steps, dtype=float)
        a = a_last.copy()
        for h in range(steps):
            a = T @ a
            mean[h] = det_future[h] + float(Z @ a)

        out: Dict[str, np.ndarray] = {"mean": mean}
        if not return_intervals:
            return out

        rng = np.random.default_rng(seed)
        sims = np.zeros((num_sim, steps), dtype=float)
        sqrt_sigma = float(np.sqrt(max(sigma2, 1e-12)))

        for r in range(num_sim):
            a_sim = a_last.copy()
            for h in range(steps):
                eps = rng.normal(0.0, sqrt_sigma)
                a_sim = T @ a_sim + R * eps
                sims[r, h] = det_future[h] + float(Z @ a_sim)

        lo = np.quantile(sims, alpha / 2.0, axis=0)
        hi = np.quantile(sims, 1.0 - alpha / 2.0, axis=0)
        out["lo"] = lo
        out["hi"] = hi
        return out


def _theta_dim(
    *,
    k_exog: int,
    include_intercept: bool,
    p: int,
    q: int,
    P: int,
    Q: int,
) -> int:
    return (1 if include_intercept else 0) + k_exog + p + P + q + Q + 1


def _project_theta_init(
    theta_src: Optional[np.ndarray],
    src_spec: SarimaxSpec,
    dst_spec: SarimaxSpec,
    *,
    k_exog: int,
) -> Optional[np.ndarray]:
    if theta_src is None:
        return None

    theta_src = np.asarray(theta_src, dtype=float).reshape(-1)
    has_c = bool(src_spec.include_intercept and dst_spec.include_intercept)

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

    i0 = 0
    i1 = 0

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

    n = min(p0, p1)
    if n > 0:
        dst[i1 : i1 + n] = theta_src[i0 : i0 + n]
    i0 += p0
    i1 += p1

    n = min(P0, P1)
    if n > 0:
        dst[i1 : i1 + n] = theta_src[i0 : i0 + n]
    i0 += P0
    i1 += P1

    n = min(q0, q1)
    if n > 0:
        dst[i1 : i1 + n] = theta_src[i0 : i0 + n]
    i0 += q0
    i1 += q1

    n = min(Q0, Q1)
    if n > 0:
        dst[i1 : i1 + n] = theta_src[i0 : i0 + n]
    i0 += Q0
    i1 += Q1

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
    max_total_fits: int = 0  # 0 = unlimited


def auto_sarimax_stepwise(
    y,
    exog=None,
    *,
    seasonal_period: int = 1,
    cfg: Optional[AutoConfig] = None,
    verbose: int | bool = False,
) -> SarimaxFit:
    """
    Hyndman–Khandakar style stepwise search over (p,q,P,Q) with AICc.
    Uses a fast search fit and optional full refit for the final model.
    """
    import time

    vlevel = int(verbose) if not isinstance(verbose, bool) else (1 if verbose else 0)

    t_global = time.perf_counter()
    y0 = _as_1d(y)
    X0 = _as_2d(exog, n=y0.size)

    s = int(seasonal_period)
    if cfg is None:
        cfg = AutoConfig()
    cfg = AutoConfig(**{**cfg.__dict__, "seasonal_period": s})
    if vlevel >= 1:
        print(
            f"[auto] start n={y0.size} seasonal_period={s} "
            f"maxiter_search={cfg.maxiter_search} max_steps={cfg.max_steps} "
            f"use_numba={cfg.use_numba and _HAS_NUMBA}"
        )

    # --- choose d, D with simple heuristics ---
    def kpss_p(x):
        import warnings

        from statsmodels.tsa.stattools import kpss

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p, _, _ = kpss(x, regression="c", nlags="auto")
        return float(p)

    yd0 = y0.copy()
    d = 0
    for dd in range(cfg.d_max + 1):
        pval = kpss_p(yd0)
        if vlevel >= 2:
            print(f"[auto] KPSS after d={dd}: p={pval:.4f}")
        if pval >= 0.05:
            d = dd
            break
        yd0 = yd0[1:] - yd0[:-1]

    D = 0
    if s > 1:
        yd1 = difference(y0, d=d, D=0, s=s)
        if yd1.size > s + 10:
            p_seas = kpss_p(yd1)
            if vlevel >= 2:
                print(f"[auto] seasonal KPSS on d={d}: p={p_seas:.4f}")
            if p_seas < 0.05 and cfg.D_max >= 1:
                D = 1
    if vlevel >= 1:
        print(f"[auto] differencing chosen: d={d}, D={D}, s={s}")

    # Pre-difference once for search speed; keep d,D in spec for returned model semantics.
    yd = difference(y0, d=d, D=D, s=s)
    Xd = difference_exog(X0, d=d, D=D, s=s)
    k_exog = 0 if Xd is None else Xd.shape[1]

    cache: Dict[Tuple[int, int, int, int, int, int], SarimaxFit] = {}
    fail_cache: Dict[Tuple[int, int, int, int, int, int], str] = {}
    stats: Dict[str, int] = {"fit_calls": 0, "cache_hits": 0, "failures": 0}
    budget_stop = {"stop": False}

    def score_model(
        p: int,
        q: int,
        P: int,
        Q: int,
        *,
        init_from: Optional[SarimaxFit] = None,
    ) -> SarimaxFit:
        key = (p, d, q, P, D, Q)
        if key in cache:
            stats["cache_hits"] += 1
            if vlevel >= 2:
                print(f"[cache-hit] order={(p, d, q)} seasonal={(P, D, Q, s)}")
            return cache[key]
        if key in fail_cache:
            raise RuntimeError(fail_cache[key])
        if cfg.max_total_fits > 0 and stats["fit_calls"] >= cfg.max_total_fits:
            budget_stop["stop"] = True
            msg = (
                f"fit budget exhausted (max_total_fits={cfg.max_total_fits}) before "
                f"testing (p,d,q,P,D,Q)=({p},{d},{q},{P},{D},{Q})"
            )
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
                init_from.info.get("theta_u_opt"),
                init_from.spec,
                spec,
                k_exog=k_exog,
            )

        m = SarimaxScratch(spec)
        t_fit = time.perf_counter()
        if vlevel >= 2:
            print(
                f"[fit-start] order={(p, d, q)} seasonal={(P, D, Q, s)} "
                f"warm_start={init_params is not None}"
            )
        try:
            fit = m.fit(
                yd,
                exog=Xd,
                maxiter=cfg.maxiter_search,
                method="L-BFGS-B",
                verbose=False,
                pre_differenced=True,
                compute_smoother=cfg.compute_smoother_during_search,
                use_numba=cfg.use_numba,
                init_params=init_params,
            )
            stats["fit_calls"] += 1
            if vlevel >= 2:
                print(
                    f"[fit-done] order={(p, d, q)} seasonal={(P, D, Q, s)} "
                    f"aicc={fit.aicc:.3f} converged={fit.converged} "
                    f"nit={fit.info.get('nit', -1)} "
                    f"time={time.perf_counter() - t_fit:.2f}s"
                )
        except Exception as e1:
            if cfg.use_numba and cfg.fallback_no_numba:
                if vlevel >= 2:
                    print(
                        f"[fit-retry] order={(p, d, q)} seasonal={(P, D, Q, s)} "
                        f"reason={type(e1).__name__}"
                    )
                try:
                    fit = m.fit(
                        yd,
                        exog=Xd,
                        maxiter=cfg.maxiter_search,
                        method="L-BFGS-B",
                        verbose=False,
                        pre_differenced=True,
                        compute_smoother=cfg.compute_smoother_during_search,
                        use_numba=False,
                        init_params=init_params,
                    )
                    stats["fit_calls"] += 1
                    if vlevel >= 2:
                        print(
                            f"[fit-done-cpu] order={(p, d, q)} seasonal={(P, D, Q, s)} "
                            f"aicc={fit.aicc:.3f} converged={fit.converged} "
                            f"nit={fit.info.get('nit', -1)} "
                            f"time={time.perf_counter() - t_fit:.2f}s"
                        )
                except Exception as e2:
                    stats["failures"] += 1
                    msg = (
                        f"fit failed for (p,d,q,P,D,Q)=({p},{d},{q},{P},{D},{Q}) "
                        f"with numba and without numba. "
                        f"numba_err={type(e1).__name__}: {e1}; "
                        f"cpu_err={type(e2).__name__}: {e2}"
                    )
                    fail_cache[key] = msg
                    raise RuntimeError(msg) from e2
            else:
                stats["failures"] += 1
                msg = (
                    f"fit failed for (p,d,q,P,D,Q)=({p},{d},{q},{P},{D},{Q}): "
                    f"{type(e1).__name__}: {e1}"
                )
                fail_cache[key] = msg
                raise RuntimeError(msg) from e1

        cache[key] = fit
        return fit

    candidates = [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (2, 2, 1 if s > 1 else 0, 1 if s > 1 else 0),
    ]
    candidates = [
        (min(p, cfg.p_max), min(q, cfg.q_max), min(P, cfg.P_max), min(Q, cfg.Q_max))
        for p, q, P, Q in candidates
    ]

    best = None
    init_errors = []
    for (p0, q0, P0, Q0) in candidates:
        if budget_stop["stop"]:
            break
        try:
            f = score_model(p0, q0, P0, Q0, init_from=best)
            if best is None or f.aicc < best.aicc:
                best = f
        except Exception as e:
            init_errors.append(str(e))
            if vlevel >= 1:
                print(f"[init-skip] {(p0, d, q0, P0, D, Q0)} -> {e}")
            continue
    if best is None:
        msg = "Could not fit any initial SARIMAX candidates."
        if init_errors:
            msg += " First errors: " + " | ".join(init_errors[:3])
        raise RuntimeError(msg)
    if vlevel >= 1:
        p0, _, q0 = best.spec.order
        P0, _, Q0, _ = best.spec.seasonal_order
        print(
            f"[auto] init best order={(p0, d, q0)} seasonal={(P0, D, Q0, s)} "
            f"AICc={best.aicc:.3f}"
        )

    def neighbors(p, q, P, Q):
        for dp, dq, dP, dQ in [
            (+1, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, +1, 0, 0),
            (0, -1, 0, 0),
            (0, 0, +1, 0),
            (0, 0, -1, 0),
            (0, 0, 0, +1),
            (0, 0, 0, -1),
            (+1, -1, 0, 0),
            (-1, +1, 0, 0),
        ]:
            pp = p + dp
            qq = q + dq
            PP = P + dP
            QQ = Q + dQ
            if (
                0 <= pp <= cfg.p_max
                and 0 <= qq <= cfg.q_max
                and 0 <= PP <= cfg.P_max
                and 0 <= QQ <= cfg.Q_max
            ):
                yield (pp, qq, PP, QQ)

    p, d_, q = best.spec.order
    P, D_, Q, s_ = best.spec.seasonal_order
    assert d_ == d and D_ == D and s_ == s

    steps = 0
    improved = True
    while improved and steps < cfg.max_steps:
        if budget_stop["stop"]:
            break
        improved = False
        cur_best = best
        
        neighbor_candidates = list(neighbors(p, q, P, Q))
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_cand = {
                executor.submit(score_model, pp, qq, PP, QQ, init_from=cur_best): (pp, qq, PP, QQ)
                for (pp, qq, PP, QQ) in neighbor_candidates
            }
            for future in concurrent.futures.as_completed(future_to_cand):
                cand = future_to_cand[future]
                try:
                    f = future.result()
                    results.append((f, cand))
                except Exception as e:
                    if vlevel >= 2:
                        print(f"[neighbor-skip] {(cand[0], d, cand[1], cand[2], D, cand[3])} -> {e}")
        
        for f, (pp, qq, PP, QQ) in results:
            if f.aicc + 1e-9 < cur_best.aicc:
                cur_best = f
                p, q, P, Q = pp, qq, PP, QQ
                improved = True
        best = cur_best
        steps += 1
        if vlevel >= 1:
            print(
                f"[step {steps}] best order={(p, d, q)} seasonal={(P, D, Q, s)} "
                f"AICc={best.aicc:.3f} fits={stats['fit_calls']} "
                f"cache_hits={stats['cache_hits']} elapsed={time.perf_counter() - t_global:.1f}s"
            )

    if cfg.refit_final:
        if vlevel >= 1:
            print(
                f"[refit] final refit maxiter={cfg.maxiter_refit} "
                f"order={best.spec.order} seasonal={best.spec.seasonal_order}"
            )
        final_model = SarimaxScratch(best.spec)
        try:
            best = final_model.fit(
                y0,
                exog=X0,
                maxiter=cfg.maxiter_refit,
                method="L-BFGS-B",
                verbose=False,
                compute_smoother=True,
                use_numba=cfg.use_numba,
                init_params=best.info.get("theta_u_opt"),
            )
        except Exception:
            if not (cfg.use_numba and cfg.fallback_no_numba):
                raise
            best = final_model.fit(
                y0,
                exog=X0,
                maxiter=cfg.maxiter_refit,
                method="L-BFGS-B",
                verbose=False,
                compute_smoother=True,
                use_numba=False,
                init_params=best.info.get("theta_u_opt"),
            )

    if vlevel >= 1:
        print(
            f"[auto] done in {time.perf_counter() - t_global:.2f}s "
            f"fits={stats['fit_calls']} cache_hits={stats['cache_hits']} failures={stats['failures']} "
            f"best={best.spec} AICc={best.aicc:.3f}"
        )

    return best
