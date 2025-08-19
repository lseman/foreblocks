from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import dcor as _dcor_pkg
    HAS_DCOR = True
except Exception:
    HAS_DCOR = False

try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    def njit(*a, **k):
        def wrap(fn): return fn
        return wrap
    prange = range



class DistanceCorrelation:
    """
    Fast & memory-friendly distance correlation.

    Strategy
    --------
    1) If `dcor` is available, use it (vectorized, well-tested).
    2) Else use a *tiled two-pass* algorithm:
       - Pass 1: compute row sums and grand mean of |x_i - x_j| (and for y),
                 streaming over blocks (no n×n allocation).
       - Pass 2: re-stream blocks, apply double-centering on-the-fly, and
                 accumulate <A, B> (dcov_xy) and <A, A>, <B, B> (dcov_xx, dcov_yy).
    3) Optional coreset subsampling (k-center greedy) for very large n.

    Notes
    -----
    - Complexity: O(n^2) time, O(n + b^2) memory, where b is `block_size`.
    - For typical n ≤ 2k (your pipeline already subsamples), this is fast and stable.
    - Uses |x_i - x_j| (L1) distances as per Székely–Rizzo distance correlation definition in 1D.
    """

    def __init__(
        self,
        block_size: int = 1024,
        unbiased: bool = False,
        pearson_gate: float = 0.05,
        max_n: int = 2000,
        random_state: int = 42,
        use_coreset: bool = True,
    ):
        self.block_size = int(block_size)
        self.unbiased = bool(unbiased)
        self.pearson_gate = float(pearson_gate)
        self.max_n = int(max_n)
        self.rs = int(random_state)
        self.use_coreset = bool(use_coreset)

    # ---------- Public API ----------
    def matrix(self, df: pd.DataFrame, pearson_scr: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        cols = list(df.columns)
        p = len(cols)
        if p < 2:
            return pd.DataFrame(np.ones((p, p)), index=cols, columns=cols)

        X = df[cols].to_numpy()
        # Optional coreset subsample for very large n (keeps boundary/diversity)
        if X.shape[0] > self.max_n and self.use_coreset:
            idx = self._kcenter_greedy(self._standardize(X), k=self.max_n, rs=self.rs)
            X = X[idx]

        out = np.eye(p, dtype=float)
        for i in range(p):
            xi = X[:, i].astype(np.float64, copy=False)
            for j in range(i + 1, p):
                if pearson_scr is not None and pearson_scr.iat[i, j] < self.pearson_gate:
                    continue
                yj = X[:, j].astype(np.float64, copy=False)

                # Drop NaNs pairwise
                msk = np.isfinite(xi) & np.isfinite(yj)
                if np.count_nonzero(msk) < 50:
                    continue
                x = xi[msk]
                y = yj[msk]

                try:
                    val = self._dcorr_1d(x, y)
                except Exception:
                    val = np.nan
                if np.isfinite(val):
                    out[i, j] = out[j, i] = float(max(0.0, min(1.0, val)))

        return pd.DataFrame(out, index=cols, columns=cols)

    # ---------- Core single-pair computation ----------
    def _dcorr_1d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance correlation for two real-valued variables."""
        n = x.shape[0]
        if n < 4:
            return np.nan

        if HAS_DCOR:
            # dcor.distance_correlation_fast prefers 2D arrays; ensure shapes
            return float(_dcor_pkg.distance_correlation(x, y))

        # Fallback: two-pass tiled algorithm with double-centering
        # 1) Row sums and grand means for |x_i - x_j| and |y_i - y_j|
        rx, gx = self._row_sums_and_grand_mean_abs(x)
        ry, gy = self._row_sums_and_grand_mean_abs(y)

        # 2) Accumulate inner products of centered distance matrices
        dcov_xy, dcov_xx, dcov_yy = self._accumulate_centered_products(x, y, rx, gx, ry, gy)

        if self.unbiased:
            # unbiased normalization (finite-sample correction)
            m = n * (n - 3)
            if m <= 0:
                return np.nan
            dcov_xy = dcov_xy / m
            dcov_xx = dcov_xx / m
            dcov_yy = dcov_yy / m
        else:
            # biased / "V-statistic" normalization
            m = n * n
            dcov_xy /= m
            dcov_xx /= m
            dcov_yy /= m

        denom = np.sqrt(max(dcov_xx * dcov_yy, 1e-18))
        return float(0.0 if denom == 0.0 else dcov_xy / denom)

    # ---------- Pass 1: row sums & grand mean ----------
    def _row_sums_and_grand_mean_abs(self, v: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes per-row sums of |v_i - v_j| for all i (size n)
        and the grand mean (sum / n^2), without storing the full matrix.

        Uses a sort + prefix-sum trick to get row sums in O(n log n).
        """
        n = v.shape[0]
        order = np.argsort(v, kind="mergesort")
        s = v[order]
        prefix = np.zeros(n + 1, dtype=np.float64)
        prefix[1:] = np.cumsum(s, dtype=np.float64)

        # For sorted s, sum_j |s_i - s_j| =
        #   i*s_i - prefix[i] + (prefix[n] - prefix[i+1]) - (n - i - 1)*s_i
        # then scatter back to original order.
        idxs = np.arange(n, dtype=np.int64)
        left = idxs * s - prefix[:-1]
        right = (prefix[-1] - prefix[1:]) - (n - idxs - 1) * s
        row_sums_sorted = left + right

        row_sums = np.empty_like(row_sums_sorted)
        row_sums[order] = row_sums_sorted

        grand_sum = float(prefix[-1] * n - 2.0 * np.sum((idxs + 1) * s) + np.sum(s))  # equals sum_{i<j} 2*|…|
        # A robust way: the grand mean of the distance matrix:
        grand_mean = float(row_sums.mean() / n)

        return row_sums, grand_mean

    # ---------- Pass 2: centered products accumulation ----------
    @njit(parallel=True, fastmath=True, cache=True)
    def _accumulate_centered_products(x, y, rx, gx, ry, gy, block_size):
        n = x.shape[0]
        n_inv = 1.0 / n

        s_xy = 0.0
        s_xx = 0.0
        s_yy = 0.0

        for i0 in prange(0, n, block_size):   # parallelize across i-blocks
            i1 = min(n, i0 + block_size)
            for j0 in range(0, n, block_size):
                j1 = min(n, j0 + block_size)

                for ii in range(i0, i1):
                    xi = x[ii]
                    rxi = rx[ii]
                    ryi = ry[ii]
                    for jj in range(j0, j1):
                        dx = abs(xi - x[jj])
                        dy = abs(y[ii] - y[jj])

                        ax = dx - rxi * n_inv - rx[jj] * n_inv + gx
                        ay = dy - ryi * n_inv - ry[jj] * n_inv + gy

                        s_xy += ax * ay
                        s_xx += ax * ax
                        s_yy += ay * ay

        return s_xy, s_xx, s_yy
    
    # ---------- Helpers ----------
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, keepdims=True)
        sd = np.where(sd <= 1e-12, 1.0, sd)
        Z = (X - mu) / sd
        return np.asarray(Z, dtype=np.float32)

    def _kcenter_greedy(self, Z: np.ndarray, k: int, rs: int = 42) -> np.ndarray:
        """Coreset sampling for rows of Z (Euclidean)."""
        rng = np.random.default_rng(rs)
        n = Z.shape[0]
        if k >= n:
            return np.arange(n)
        centers = [rng.integers(0, n)]
        d2 = np.sum((Z - Z[centers[0]]) ** 2, axis=1)
        for _ in range(1, k):
            i = int(np.argmax(d2))
            centers.append(i)
            d2 = np.minimum(d2, np.sum((Z - Z[i]) ** 2, axis=1))
        return np.array(centers, dtype=int)
