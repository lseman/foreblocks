"""foreblocks.ts_handler.auto_filter.filters.penalized.

Penalized least-squares smoothers: Whittaker-Eilers, Hodrick-Prescott, L1 Trend Filter.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from foreblocks.ts_handler.auto_filter.filters.utils import _as_series
from foreblocks.ts_handler.auto_filter.registry import register_filter


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
