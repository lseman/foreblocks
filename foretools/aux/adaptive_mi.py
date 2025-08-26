from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numba import njit
from scipy.special import digamma
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from torch import Tensor

try:
    # Often fastest for 2D Chebyshev queries
    from scipy.spatial import cKDTree as _KDTree

    _HAVE_CKDTREE = True
except Exception:
    _HAVE_CKDTREE = False


ArrayLike = Union[np.ndarray, pd.Series]


class InfoNCECritic(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, y):
        # x: [B, x_dim], y: [B, y_dim]
        z = torch.cat([x, y], dim=1)
        return self.net(z)  # [B, 1]


def _mi_neural_infonce(
    x: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 128,
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda",
) -> float:
    """
    Neural MI estimation using InfoNCE bound (in nats).
    """

    # Convert inputs to torch tensors
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    n, x_dim = x.shape
    _, y_dim = y.shape

    critic = InfoNCECritic(x_dim, y_dim, hidden_dim).to(device)
    optimizer = optim.Adam(critic.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            B = xb.size(0)

            # Compute pairwise scores [B, B]
            scores = critic(
                xb.unsqueeze(1).repeat(1, B, 1).reshape(B * B, -1), yb.repeat(B, 1)
            ).reshape(B, B)

            # InfoNCE loss: maximize diagonal elements relative to off-diagonal
            log_probs = F.log_softmax(scores, dim=1)
            loss = -log_probs.diag().mean()  # Negative because we minimize

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Optional: print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

    # Final MI estimate on full dataset
    with torch.no_grad():
        scores = critic(
            x.unsqueeze(1).repeat(1, n, 1).reshape(n * n, -1), y.repeat(n, 1)
        ).reshape(n, n)

        log_probs = F.log_softmax(scores, dim=1)
        # Correct InfoNCE bound: add log(n) bias correction
        mi_est = log_probs.diag().mean().item() + np.log(n)

    return max(0.0, mi_est)  # MI should be non-negative


class AdaptiveMI:
    """
    Adaptive, fast mutual information estimator for 1D-1D pairs.

    Methods:
      - 'ksg' (KSG type-1, k-NN in joint space, Chebyshev metric)
      - 'copula_gaussian' (rank -> Gaussian, exact for Gaussian copula)
      - 'binned_quantile' (quantile bins + Miller–Madow debias)

    Returns by default a bounded [0,1] "MI-correlation": sqrt(1 - exp(-2*MI)).
    (Matches |ρ| for Gaussian case.)

    Design goals:
      - Clean, deterministic, minimal allocations
      - Robust to ties/outliers via quantile and rank transforms
      - One-pass neighbor search reused across k values
      - Subsampling for very large n with fixed RNG

    Public methods: score, fit_score, score_pairwise, matrix, explain_method_choice
    """

    def __init__(
        self,
        subsample: int = 2000,
        spearman_gate: float = 0.05,  # gate low-correlation pairs in matrix()
        min_overlap: int = 50,  # minimum valid paired samples
        ks: Tuple[int, ...] = (3, 5, 10),
        n_bins: int = 16,
        random_state: int = 42,
        ksg_use_ckdtree: bool = True,  # try cKDTree for large n
        ties_threshold: float = 0.05,  # switch to binned when ties exceed this
        rho_threshold: float = 0.85,  # switch to copula when |ρ_spearman| >= this
    ):
        self.subsample = int(subsample)
        self.spearman_gate = float(spearman_gate)
        self.min_overlap = int(min_overlap)
        self.ks = tuple(int(k) for k in ks)
        self.n_bins = int(n_bins)
        self.random_state = int(random_state)
        self.ksg_use_ckdtree = bool(ksg_use_ckdtree)
        self.ties_threshold = float(ties_threshold)
        self.rho_threshold = float(rho_threshold)

    # ---------------- Public API ----------------

    def score(
        self,
        x: ArrayLike,
        y: ArrayLike,
        return_raw_mi: bool = False,
        method: Optional[str] = None,
        return_method: bool = False,
    ) -> Union[float, Tuple[float, str]]:
        """
        Estimate MI between two 1D variables.

        Parameters
        ----------
        x, y : array-like, shape (n,)
        return_raw_mi : if True, return MI in nats; else return [0,1] MI-corr
        method : force 'ksg' | 'copula_gaussian' | 'binned_quantile' | None (auto)
        return_method : if True, return (score, method_used)

        Returns
        -------
        score : float in [0,1] (default) or MI in nats (if return_raw_mi)
        optionally accompanied by the chosen method name
        """
        x = _to_1d_array(x)
        y = _to_1d_array(y)

        if x.size != y.size:
            raise ValueError(
                f"x and y must have same length, got {x.size} and {y.size}"
            )
        if x.size == 0:
            return (0.0, "none") if return_method else 0.0

        m = np.isfinite(x) & np.isfinite(y)
        n_valid = int(m.sum())
        if n_valid < self.min_overlap:
            return (0.0, "insufficient") if return_method else 0.0

        x = x[m].astype(np.float64, copy=False)
        y = y[m].astype(np.float64, copy=False)

        # Deterministic subsample for very large n
        if x.size > self.subsample:
            rng = check_random_state(self.random_state)
            idx = rng.choice(x.size, size=self.subsample, replace=False)
            x, y = x[idx], y[idx]

        # Early outs for constants (MI = 0)
        if _is_constant(x) or _is_constant(y):
            return (0.0, "constant") if return_method else 0.0

        # Choose method
        used = method
        if used is None:
            ties_x = self._tie_fraction(x)
            ties_y = self._tie_fraction(y)
            abs_rho = abs(_spearman_fast(x, y))
            if max(ties_x, ties_y) > self.ties_threshold:
                used = "binned_quantile"
            elif abs_rho >= self.rho_threshold:
                used = "copula_gaussian"
            else:
                used = "ksg"

        if used in ("ksg", "ksg1"):
            mi = self._mi_ksg1_avg(x, y, self.ks)
        elif used == "ksg2":
            mi = self._mi_ksg2_avg(x, y, self.ks)
        elif used == "copula_gaussian":
            mi = self._mi_copula_gaussian(x, y)
        elif used == "binned_quantile":
            mi = self._mi_binned_quantile(x, y, self.n_bins)
        elif used == "infonce":
            print("Using neural MI estimation (InfoNCE)")
            mi = self._mi_neural(x, y)  # Placeholder for neural MI method
        else:
            raise ValueError(f"Unknown method: {used}")

        score = mi if return_raw_mi else _mi_to_coeff(mi)
        score = 0.0 if not np.isfinite(score) else float(np.clip(score, 0.0, 1.0))
        return (score, used) if return_method else score

    def _mi_neural(
        self, x: np.ndarray, y: np.ndarray, method="infonce", **kwargs
    ) -> float:
        if method == "infonce":
            return mi_neural_infonce_coeff(x, y, **kwargs)
        else:
            raise ValueError(f"Unknown neural MI method: {method}")

    def fit_score(
        self, x: ArrayLike, y: ArrayLike, return_raw_mi: bool = False
    ) -> float:
        # For sklearn-compatibility; MI has no fitted state.
        return self.score(x, y, return_raw_mi=return_raw_mi)
                
    def score_pairwise(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: ArrayLike,
        return_raw_mi: bool = False,
    ) -> np.ndarray:
        """
        Compute MI for each column of X against y using AdaptiveMI.score.
        Optimized for lower overhead (no joblib).
        """
        # Ensure numpy 2D
        if hasattr(X, "values"):
            X = X.values  # type: ignore
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y = _to_1d_array(y)

        # Use transposed view to avoid costly X[:, j] copies
        Xt = X.T  

        results = [
            self.score(Xt[j], y, return_raw_mi=return_raw_mi) for j in range(Xt.shape[0])
        ]

        return np.array(results, dtype=float)


    def matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pairwise MI-correlation matrix (symmetric) for DataFrame columns.
        Uses a single row subsample and a Spearman gate to avoid trivial pairs.
        """
        cols = list(df.columns)
        p = len(cols)
        if p <= 1:
            return pd.DataFrame(np.ones((p, p)), index=cols, columns=cols)

        S = self._subsample_df(df, self.subsample, self.random_state)[cols]
        with np.errstate(all="ignore"):
            sp = (
                S.corr(method="spearman", min_periods=self.min_overlap)
                .fillna(0.0)
                .to_numpy()
            )

        arrs = [S[c].to_numpy(dtype=np.float64, copy=False) for c in cols]
        out = np.eye(p, dtype=float)

        for i in range(p):
            xi = arrs[i]
            for j in range(i + 1, p):
                if abs(sp[i, j]) < self.spearman_gate:
                    continue
                x, y, ok = _valid_pair(xi, arrs[j], self.min_overlap)
                if not ok:
                    continue
                ties_x = self._tie_fraction(x)
                ties_y = self._tie_fraction(y)
                abs_rho = abs(_spearman_fast(x, y))
                if max(ties_x, ties_y) > self.ties_threshold:
                    mi = self._mi_binned_quantile(x, y, self.n_bins)
                elif abs_rho >= self.rho_threshold:
                    mi = self._mi_copula_gaussian(x, y)
                else:
                    mi = self._mi_ksg1_avg(x, y, self.ks)
                out[i, j] = out[j, i] = float(np.clip(_mi_to_coeff(mi), 0.0, 1.0))

        return pd.DataFrame(out, index=cols, columns=cols)

    # ---------------- Estimators ----------------
    def _mi_ksg1_avg(self, x: np.ndarray, y: np.ndarray, ks: Tuple[int, ...]) -> float:
        """
        KSG type-1 averaged over ks.
        """
        Z = np.column_stack((_zscore_1d(x), _zscore_1d(y)))
        n = Z.shape[0]
        if n <= 3:
            return 0.0

        ks_eff = [k for k in ks if 1 <= k < n]
        if not ks_eff:
            return 0.0
        k_max = max(ks_eff)

        # Distances to kth neighbor (exclude self)
        if self.ksg_use_ckdtree and _HAVE_CKDTREE and n >= 200:
            tree = _KDTree(Z, compact_nodes=True, balanced_tree=True)
            dists, _ = tree.query(Z, k=k_max + 1, p=np.inf, workers=-1)
        else:
            nn = NearestNeighbors(n_neighbors=k_max + 1, metric="chebyshev", n_jobs=-1)
            nn.fit(Z)
            dists, _ = nn.kneighbors(Z, return_distance=True)

        mi_vals: List[float] = []
        xz = Z[:, 0].copy()
        yz = Z[:, 1].copy()
        for k in ks_eff:
            eps = np.asarray(dists[:, k], dtype=np.float64) + 1e-12
            nx = _count_within_1d(xz, eps)  # excludes self
            ny = _count_within_1d(yz, eps)
            # KSG type-1
            mi_k = float(
                digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
            )
            if mi_k > 0.0 and np.isfinite(mi_k):
                mi_vals.append(mi_k)
        return float(np.mean(mi_vals)) if mi_vals else 0.0

    def _mi_ksg2_avg(self, x: np.ndarray, y: np.ndarray, ks: Tuple[int, ...]) -> float:
        """
        KSG type-2 averaged over ks.
        """
        Z = np.column_stack((_zscore_1d(x), _zscore_1d(y)))
        n = Z.shape[0]
        if n <= 3:
            return 0.0

        ks_eff = [k for k in ks if 1 <= k < n]
        if not ks_eff:
            return 0.0
        k_max = max(ks_eff)

        if self.ksg_use_ckdtree and _HAVE_CKDTREE and n >= 200:
            tree = _KDTree(Z, compact_nodes=True, balanced_tree=True)
            dists, _ = tree.query(Z, k=k_max + 1, p=np.inf, workers=-1)
        else:
            nn = NearestNeighbors(n_neighbors=k_max + 1, metric="chebyshev", n_jobs=-1)
            nn.fit(Z)
            dists, _ = nn.kneighbors(Z, return_distance=True)

        mi_vals: List[float] = []
        xz = Z[:, 0].copy()
        yz = Z[:, 1].copy()
        for k in ks_eff:
            eps = np.asarray(dists[:, k], dtype=np.float64) + 1e-12
            nx = _count_within_1d(xz, eps)
            ny = _count_within_1d(yz, eps)
            # KSG type-2
            mi_k = float(digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny)))
            if mi_k > 0.0 and np.isfinite(mi_k):
                mi_vals.append(mi_k)
        return float(np.mean(mi_vals)) if mi_vals else 0.0

    def _mi_copula_gaussian(self, x: np.ndarray, y: np.ndarray) -> float:
        """Exact for Gaussian copulas; robust via rank->Gaussian."""
        xg = _gauss_rank(x)
        yg = _gauss_rank(y)
        r = float(np.corrcoef(xg, yg)[0, 1])
        r = float(np.clip(r, -0.999999, 0.999999))
        return float(-0.5 * np.log(1.0 - r * r))

    def _mi_binned_quantile(self, x: np.ndarray, y: np.ndarray, B: int) -> float:
        """Quantile-binned MI with Miller–Madow debias."""
        qx = _quantile_bins(x, B)
        qy = _quantile_bins(y, B)
        # bins are [0..B-1], so use arange(B+1) as edges for histogram2d
        Hxy, _, _ = np.histogram2d(qx, qy, bins=[np.arange(B + 1), np.arange(B + 1)])
        n = float(Hxy.sum())
        if n <= 0:
            return 0.0

        Px = Hxy.sum(axis=1) / n
        Py = Hxy.sum(axis=0) / n
        Pxy = Hxy / n

        Hx = -_sum_zlog(Px)
        Hy = -_sum_zlog(Py)
        Hxy = -_sum_zlog(Pxy)

        mi = Hx + Hy - Hxy
        if mi <= 0.0:
            return 0.0

        # Miller–Madow bias: approx (Kx*Ky - Kx - Ky + 1) / (2N)
        Kx = int(np.count_nonzero(Px))
        Ky = int(np.count_nonzero(Py))
        Kxy = int(np.count_nonzero(Pxy))
        bias = max((Kx * Ky - Kx - Ky + 1), 0) / (2.0 * n + 1e-12)
        mi = mi - bias
        return float(max(mi, 0.0))

    # ---------------- Helpers ----------------

    def _subsample_df(self, df: pd.DataFrame, target: int, rs: int) -> pd.DataFrame:
        if len(df) <= target:
            return df
        rng = check_random_state(rs)
        idx = rng.choice(len(df), size=target, replace=False)
        return df.iloc[idx]

    def _tie_fraction(self, v: np.ndarray) -> float:
        # 1 - (#unique / n). Works well as a quick proxy.
        n = v.size
        if n == 0:
            return 0.0
        return 1.0 - (min(np.unique(v).size, n) / float(n))


# ===================== Low-level utilities (tight & fast) =====================


def _to_1d_array(a: ArrayLike) -> np.ndarray:
    if hasattr(a, "values"):
        a = a.values  # type: ignore
    a = np.asarray(a)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a


def _is_constant(v: np.ndarray) -> bool:
    # Much faster than std for large arrays
    return np.nanmin(v) == np.nanmax(v)


def _zscore_1d(v: np.ndarray) -> np.ndarray:
    mu = v.mean()
    sd = v.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


@njit(cache=True, fastmath=True)
def _rank_average(v: np.ndarray) -> np.ndarray:
    """
    Average rank with tie correction (Numba JIT).
    Replaces the Python version for speed, same name for compatibility.
    """
    n = v.size
    order = np.argsort(v)  # Numba supports quicksort/heapsort
    ranks = np.empty(n, dtype=np.float64)

    # assign initial ranks
    for i in range(n):
        ranks[order[i]] = i + 1.0  # ranks start at 1

    # tie correction
    i = 0
    while i < n:
        j = i + 1
        vi = v[order[i]]
        while j < n and v[order[j]] == vi:
            j += 1
        if j - i > 1:  # tie detected
            avg = 0.5 * (ranks[order[i]] + ranks[order[j - 1]])
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    return ranks


@njit(cache=True, fastmath=True)
def _spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation using numba-compatible _rank_average.
    Equivalent to np.corrcoef(rank(x), rank(y))[0,1].
    """
    rx = _rank_average(x)
    ry = _rank_average(y)

    n = rx.size
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += rx[i]
        mean_y += ry[i]
    mean_x /= n
    mean_y /= n

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = rx[i] - mean_x
        dy = ry[i] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    if den_x <= 1e-18 or den_y <= 1e-18:
        return 0.0

    r = num / math.sqrt(den_x * den_y)
    if not math.isfinite(r):
        return 0.0
    return r


def _gauss_rank(v: np.ndarray) -> np.ndarray:
    r = _rank_average(v)
    u = (r - 0.5) / r.size
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return norm.ppf(u)


def _quantile_bins(v: np.ndarray, B: int) -> np.ndarray:
    # Rank -> integer bins in [0, B-1], equal-frequency as much as ties allow
    r = _rank_average(v) - 1.0
    q = np.floor(r * (B / v.size)).astype(int)
    return np.clip(q, 0, B - 1)


def _sum_zlog(p: np.ndarray) -> float:
    # Sum of p*log p with zeros safe
    # p = np.asarray(p, dtype=np.float64, copy=False)
    p = np.array(p, dtype=np.float64, copy=False)  # works across NumPy versions

    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask])))


def _count_within_1d(v: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    For Chebyshev balls at per-point radii eps, count neighbors in 1D within
    [v_i - eps_i, v_i + eps_i], excluding self.
    Linear time using two moving windows over the sorted array.
    """
    n = v.size
    order = np.argsort(v, kind="mergesort")
    s = v[order]
    e = eps[order]

    left = np.empty(n, dtype=np.int32)
    right = np.empty(n, dtype=np.int32)

    jL = 0
    jR = 0
    for k in range(n):
        si = s[k]
        ek = e[k]
        # expand right pointer to <= si + ek
        while jR < n and s[jR] <= si + ek + 1e-18:
            jR += 1
        # move left pointer to first >= si - ek
        while jL < n and s[jL] < si - ek - 1e-18:
            jL += 1
        left[k] = jL
        right[k] = jR  # exclusive

    counts_sorted = right - left - 1  # exclude self
    inv = np.empty(n, dtype=np.int32)
    inv[order] = np.arange(n, dtype=np.int32)
    return counts_sorted[inv]


def _mi_to_coeff(mi_nats: float) -> float:
    # sqrt(1 - exp(-2*MI)) ∈ [0,1]; equals |ρ| for Gaussian
    mi = max(float(mi_nats), 0.0)
    return float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * mi))))


def mi_neural_infonce_coeff(x, y, **kwargs):
    """InfoNCE MI estimation returning correlation coefficient."""
    mi_raw = _mi_neural_infonce(x, y, **kwargs)
    return _mi_to_coeff(mi_raw)


def _valid_pair(
    xi: np.ndarray, yj: np.ndarray, min_overlap: int
) -> Tuple[np.ndarray, np.ndarray, bool]:
    m = np.isfinite(xi) & np.isfinite(yj)
    if m.sum() < min_overlap:
        return xi, yj, False
    x = xi[m].astype(np.float64, copy=False)
    y = yj[m].astype(np.float64, copy=False)
    if _is_constant(x) or _is_constant(y):
        return x, y, False
    return x, y, True
