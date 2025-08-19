from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.stats import norm, rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


class AdaptiveMI:
    """
    Adaptive, fast mutual information estimator for 1D-1D pairs.
    Methods: KSG (kNN), Copula-Gaussian, Miller–Madow binned.
    Returns a bounded [0,1] 'MI-correlation' score: sqrt(1 - exp(-2*MI)).
    """

    def __init__(
        self,
        subsample: int = 2000,
        spearman_gate: float = 0.05,
        min_overlap: int = 50,
        ks: Tuple[int, ...] = (3, 5, 10),
        n_bins: int = 16,
        random_state: int = 42,
    ):
        self.subsample = int(subsample)
        self.spearman_gate = float(spearman_gate)
        self.min_overlap = int(min_overlap)
        self.ks = tuple(int(k) for k in ks)
        self.n_bins = int(n_bins)
        self.random_state = int(random_state)

    # ---------------- Public API ----------------
    def matrix(self, df: pd.DataFrame, spearman_scr: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        cols = list(df.columns)
        p = len(cols)
        if p < 2:
            return pd.DataFrame(np.ones((p, p)), index=cols, columns=cols)

        # One consistent subsample (rows)
        S = self._subsample_df(df, self.subsample, self.random_state)
        S = S[cols]  # keep order

        # Gate using Spearman on the same subsample
        with np.errstate(all="ignore"):
            sp = S.corr(method="spearman", min_periods=self.min_overlap).fillna(0.0).to_numpy()

        out = np.eye(p, dtype=float)

        # Precache columns as arrays
        arrs = [S[c].to_numpy() for c in cols]

        for i in range(p):
            xi_all = arrs[i]
            for j in range(i + 1, p):
                # gate
                if sp[i, j] is not None and abs(sp[i, j]) < self.spearman_gate:
                    continue

                yj_all = arrs[j]
                m = np.isfinite(xi_all) & np.isfinite(yj_all)
                if m.sum() < self.min_overlap:
                    continue

                x = xi_all[m].astype(np.float64, copy=False)
                y = yj_all[m].astype(np.float64, copy=False)

                # choose method
                ties_x = self._tie_fraction(x)
                ties_y = self._tie_fraction(y)
                abs_rho = abs(self._safe_spearman(x, y))

                if max(ties_x, ties_y) > 0.05:
                    mi = self._mi_binned_quantile(x, y, self.n_bins)
                elif abs_rho >= 0.85:
                    mi = self._mi_copula_gaussian(x, y)
                else:
                    mi = self._mi_ksg_avg(x, y, self.ks)

                # map MI (nats) -> [0,1] coefficient
                mi_corr = self._mi_to_coeff(mi)
                if np.isfinite(mi_corr):
                    out[i, j] = out[j, i] = float(np.clip(mi_corr, 0.0, 1.0))

        return pd.DataFrame(out, index=cols, columns=cols)

    # --------------- Estimators ---------------
    def _mi_ksg_avg(self, x: np.ndarray, y: np.ndarray, ks: Tuple[int, ...]) -> float:
        """Average of KSG (type-1) estimates over several k, reusing one NN search."""
        Z = np.column_stack([x, y])
        # Normalize marginals (z-score) helps metric scales (Chebyshev)
        Z = self._zscore(Z)
        xz = Z[:, 0]
        yz = Z[:, 1]
        n = len(Z)
        if n <= 3:
            return 0.0

        # One joint neighbor search with max k
        k_max = max(k for k in ks if k < n)
        if k_max < 1:
            return 0.0
        nn_joint = NearestNeighbors(n_neighbors=k_max + 1, metric="chebyshev", n_jobs=-1)
        nn_joint.fit(Z)
        d_joint, _ = nn_joint.kneighbors(Z, return_distance=True)
        # d_joint[:,0] is self-distance = 0

        mi_vals: List[float] = []
        for k in ks:
            if k >= n:
                continue
            # kth neighbor distance (exclude self at index 0)
            eps = d_joint[:, k] + 1e-12

            # counts in marginals with max norm balls of radius eps
            nx = self._count_within_1d(xz, eps)
            ny = self._count_within_1d(yz, eps)

            mi_k = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
            mi_vals.append(float(max(mi_k, 0.0)))

        if not mi_vals:
            return 0.0
        return float(np.mean(mi_vals))

    def _mi_copula_gaussian(self, x: np.ndarray, y: np.ndarray) -> float:
        """Gaussian MI after rank→Gaussian transform; very fast and robust to monotone warps."""
        # Copula (normal scores)
        xg = self._gauss_rank(x)
        yg = self._gauss_rank(y)
        # Pearson on Gaussianized data
        r = np.corrcoef(xg, yg)[0, 1]
        r = float(np.clip(r, -0.999999, 0.999999))
        return float(-0.5 * np.log(1.0 - r * r))

    def _mi_binned_quantile(self, x: np.ndarray, y: np.ndarray, B: int) -> float:
        """Quantile-binned MI with Miller–Madow bias correction."""
        # Quantile edges are robust to heavy ties/outliers
        qx = self._quantile_bins(x, B)
        qy = self._quantile_bins(y, B)
        # histograms
        Hxy, _, _ = np.histogram2d(qx, qy, bins=[np.arange(B + 1), np.arange(B + 1)])
        Hx = Hxy.sum(axis=1)
        Hy = Hxy.sum(axis=0)
        n = float(Hxy.sum())
        Px = Hx / n
        Py = Hy / n
        Pxy = Hxy / n

        # entropies (base e), with zero-safe logs
        Hx_e = -np.sum(self._zlog(Px))
        Hy_e = -np.sum(self._zlog(Py))
        Hxy_e = -np.sum(self._zlog(Pxy))

        mi = Hx_e + Hy_e - Hxy_e

        # Miller–Madow correction (bias ≈ (K-1)/(2N)), here Kx, Ky, Kxy non-empty bins
        Kx = int(np.count_nonzero(Px))
        Ky = int(np.count_nonzero(Py))
        Kxy = int(np.count_nonzero(Pxy))
        # Bias of MI ≈ (Kx*Ky - Kx - Ky + 1) / (2N)  (rough, safe, non-negative)
        bias = max((Kx * Ky - Kx - Ky + 1), 0) / (2.0 * n + 1e-12)
        mi = max(mi - bias, 0.0)
        return float(mi)

    # --------------- Helpers ---------------

    def _subsample_df(self, df: pd.DataFrame, target: int, rs: int) -> pd.DataFrame:
        if len(df) <= target:
            return df.dropna(how="all")
        rng = check_random_state(rs)
        idx = rng.choice(len(df), size=target, replace=False)
        return df.iloc[idx].dropna(how="all")

    def _zscore(self, Z: np.ndarray) -> np.ndarray:
        mu = np.mean(Z, axis=0, keepdims=True)
        sd = np.std(Z, axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (Z - mu) / sd

    def _tie_fraction(self, v: np.ndarray) -> float:
        # fraction of equal pairs wrt n (proxy via number of unique values)
        n = len(v)
        u = np.unique(v)
        return 1.0 - min(len(u), n) / float(n)

    def _safe_spearman(self, x: np.ndarray, y: np.ndarray) -> float:
        rx = rankdata(x, method="average")
        ry = rankdata(y, method="average")
        r = np.corrcoef(rx, ry)[0, 1]
        return 0.0 if not np.isfinite(r) else float(r)

    def _gauss_rank(self, v: np.ndarray) -> np.ndarray:
        r = rankdata(v, method="average")
        u = (r - 0.5) / len(v)  # (0,1)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return norm.ppf(u)

    def _quantile_bins(self, v: np.ndarray, B: int) -> np.ndarray:
        # Assign each point to its quantile bin in [0, B-1]
        r = rankdata(v, method="average")
        q = np.floor((r - 1) * B / len(v)).astype(int)
        q = np.clip(q, 0, B - 1)
        return q

    def _zlog(self, p: np.ndarray) -> np.ndarray:
        p = p.astype(float, copy=False)
        mask = p > 0
        out = np.zeros_like(p, dtype=float)
        out[mask] = p[mask] * np.log(p[mask])
        return out

    def _count_within_1d(self, v: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """
        For KSG with Chebyshev balls: count points within |v_i - v_j| <= eps_i (excluding self).
        Efficient via two-pointer sweep on sorted array.
        """
        n = len(v)
        order = np.argsort(v, kind="mergesort")
        s = v[order]
        # For each i (original order), find # within [s_i - eps_i, s_i + eps_i]
        # We’ll compute on sorted index positions
        inv = np.empty(n, dtype=int)
        inv[order] = np.arange(n)
        pos = inv  # index of each point in sorted order
        left = np.empty(n, dtype=int)
        right = np.empty(n, dtype=int)

        # Precompute for all possible centers using moving windows
        j_left = 0
        j_right = 0
        # We iterate i in sorted order to reuse windows
        for k, si in enumerate(s):
            e = eps[order[k]]
            # expand right
            while j_right < n and s[j_right] <= si + e + 1e-18:
                j_right += 1
            # shrink left
            while j_left < n and s[j_left] < si - e - 1e-18:
                j_left += 1
            left[k] = j_left
            right[k] = j_right  # exclusive

        counts_sorted = (right - left - 1)  # exclude self
        counts = counts_sorted[pos]
        return counts

    def _mi_to_coeff(self, mi_nats: float) -> float:
        # Map MI (nats) to [0,1]: sqrt(1 - exp(-2*MI)), equals |rho| for Gaussian case
        mi = max(float(mi_nats), 0.0)
        return float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * mi))))
