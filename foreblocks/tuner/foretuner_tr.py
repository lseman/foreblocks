from __future__ import annotations

import collections
import math
import warnings
from dataclasses import dataclass

# ============================================
# ✅ Core Python & Concurrency
# ============================================
from typing import Dict

# ============================================
# ✅ Numerical & Scientific Computing
# ============================================
import numpy as np

# Optimized with Numba JIT compilation for speed
from numba import njit, prange
from pykdtree.kdtree import KDTree

# ============================================
# ✅ Parallelism & Performance
# ============================================
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sobol_seq import i4_sobol_generate

from .foretuner_acq import *

# ============================================
# ✅ Project-Specific
# ============================================
from .foretuner_aux import *
from .foretuner_candidate import *
from .foretuner_sur import *

try:
    from pykdtree.kdtree import KDTree as _OriginalKDTree

    # Create a wrapper that handles pykdtree's quirks
    class KDTree(_OriginalKDTree):
        def query(self, x, k=1, **kwargs):
            x = np.asarray(x, dtype=np.float64)
            single_query = x.ndim == 1

            # Ensure x is 2D for pykdtree compatibility
            if x.ndim == 1:
                x = x.reshape(1, -1)

            distances, indices = super().query(x, k=k, **kwargs)

            # Handle return format for single queries to match scipy behavior
            if single_query:
                if k == 1:
                    return float(distances.flatten()[0]), int(indices.flatten()[0])
                else:
                    return distances.flatten(), indices.flatten()

            return distances, indices

    HAS_KDTREE = True
except Exception:
    try:
        from scipy.spatial import KDTree

        HAS_KDTREE = True
    except Exception:
        HAS_KDTREE = False


warnings.filterwarnings("ignore")

# learned_adaptation.py
# Lightweight learned adaptation for TuRBO-style region management.
# - RadiusPolicy: predicts Δlog r
# - RadiusLearnerAgent: online self-supervised trainer
# - RegionAttention: attention weights over regions
# - ReptileMetaLearner: optional few-shot meta-pretrain across tasks
#
# Safe fallbacks when torch is missing.


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ------------------------------
# Small utilities
# ------------------------------
def _to_tensor(x, dtype=torch.float32, device="cpu"):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _softplus(x: float) -> float:
    # numerically safe softplus
    if x > 20:  # exp overflow guard
        return x
    return math.log1p(math.exp(x))


# ------------------------------
# Learnable modules (if torch)
# ------------------------------
if TORCH_AVAILABLE:

    class RadiusPolicy(nn.Module):
        """
        Predicts Δlog r in [-delta_max, delta_max] from region feature vector.
        new_radius = radius * exp(Δ)
        """

        def __init__(self, d_in: int, hidden: int = 48, delta_max: float = 0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            self.delta_max = float(delta_max)

        @torch.no_grad()
        def step(self, feats, radius: float, min_r: float, max_r: float):
            x = feats if torch.is_tensor(feats) else _to_tensor(feats)
            raw = self.net(x)  # (1,) or scalar
            delta = torch.tanh(raw).item() * self.delta_max  # [-Δmax, Δmax]
            new_r = float(radius * math.exp(delta))
            new_r = float(max(min(new_r, max_r), min_r))
            return new_r, float(delta)

    class RegionAttention(nn.Module):
        """
        Self-attention over regions → weights for allocation/spawn/prune.
        """

        def __init__(self, d_in: int, d_model: int = 48, n_heads: int = 4):
            super().__init__()
            self.inp = nn.Linear(d_in, d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.out = nn.Linear(d_model, 1)

        @torch.no_grad()
        def scores(self, feats_batch: np.ndarray) -> np.ndarray:
            # feats_batch: (R, d_in)
            X = _to_tensor(feats_batch).unsqueeze(0)  # (1,R,d_in)
            H = torch.silu(self.inp(X))
            H2, _ = self.attn(H, H, H, need_weights=False)
            s = self.out(H2).squeeze(0).squeeze(-1)  # (R,)
            w = torch.softmax(s, dim=0).cpu().numpy()
            return w

    @dataclass
    class RLConfig:
        buffer_cap: int = 1024
        lr: float = 1e-3
        train_every: int = 1
        steps_per_train: int = 64
        batch: int = 64
        delta_max: float = 0.15

    class RadiusLearnerAgent:
        """
        Online self-supervised learner for radius changes.
        Stores (features, delta_taken, reward) and regresses to good deltas.
        """

        def __init__(self, d_in: int, cfg: RLConfig | None = None, device: str = "cpu"):
            self.cfg = cfg or RLConfig()
            self.device = device
            self.policy = RadiusPolicy(d_in, delta_max=self.cfg.delta_max).to(device)
            self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
            self.buf: list[tuple[np.ndarray, float, float]] = []
            self._ticks = 0

        def push(self, feats: np.ndarray, delta: float, reward: float):
            if not np.all(np.isfinite(feats)):  # keep buffer clean
                return
            self.buf.append((feats.astype(np.float32), float(delta), float(reward)))
            if len(self.buf) > self.cfg.buffer_cap:
                self.buf.pop(0)

        def maybe_train(self):
            self._ticks += 1
            if self._ticks % self.cfg.train_every != 0:
                return
            if len(self.buf) < 64:
                return
            import random

            for _ in range(self.cfg.steps_per_train):
                B = random.sample(self.buf, min(self.cfg.batch, len(self.buf)))
                X = torch.tensor(
                    [b[0] for b in B], dtype=torch.float32, device=self.device
                )
                y = torch.tensor(
                    [b[1] for b in B], dtype=torch.float32, device=self.device
                ).unsqueeze(1)
                rw = torch.tensor(
                    [_softplus(max(0.0, b[2])) for b in B],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(1)

                pred = torch.tanh(self.policy.net(X)) * self.policy.delta_max
                loss = F.smooth_l1_loss(pred, y, reduction="none")
                loss = (rw * loss).mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()

    class ReptileMetaLearner:
        """
        Simple Reptile meta-learning wrapper for RadiusPolicy (optional).
        """

        def __init__(self, policy: RadiusPolicy, lr_outer: float = 1e-2):
            self.master = policy
            self.lr_outer = lr_outer

        @torch.no_grad()
        def step(self, task_policy: RadiusPolicy):
            for p_master, p_task in zip(
                self.master.parameters(), task_policy.parameters()
            ):
                p_master.add_(self.lr_outer * (p_task - p_master))


else:
    # ------------------------------
    # Fallback stubs if torch missing
    # ------------------------------
    class RadiusPolicy:
        def __init__(self, d_in: int, hidden: int = 48, delta_max: float = 0.15):
            self.delta_max = float(delta_max)

        def step(self, feats, radius: float, min_r: float, max_r: float):
            # neutral: no change
            return float(max(min(radius, max_r), min_r)), 0.0

    class RegionAttention:
        def __init__(self, d_in: int, d_model: int = 48, n_heads: int = 4): ...
        def scores(self, feats_batch: np.ndarray) -> np.ndarray:
            # uniform weights
            n = feats_batch.shape[0]
            return np.ones(n, dtype=np.float64) / max(1, n)

    class RLConfig:
        def __init__(self, *args, **kwargs): ...

    class RadiusLearnerAgent:
        def __init__(self, d_in: int, cfg: RLConfig | None = None, device: str = "cpu"):
            self.policy = RadiusPolicy(d_in)

        def push(self, feats: np.ndarray, delta: float, reward: float): ...
        def maybe_train(self): ...

    class ReptileMetaLearner:
        def __init__(self, policy: RadiusPolicy, lr_outer: float = 1e-2): ...
        def step(self, task_policy: RadiusPolicy): ...


try:
    import seaborn as sns
    from pandas.plotting import parallel_coordinates

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import mplcursors

    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False


class Trial:
    """Simple trial object to hold optimization data"""

    def __init__(
        self, params: Dict[str, float], value: float, is_feasible: bool = True
    ):
        self.params = params
        self.value = value
        self.is_feasible = is_feasible
        self.constraint_violations = []  # For compatibility


@njit(parallel=True, fastmath=True)
def _compute_pairwise_distances_upper_tri(centers):
    n, d = centers.shape
    m = n * (n - 1) // 2
    out = np.empty(m, dtype=np.float64)
    idx = 0
    # Parallel over i; keep inner j serial to preserve write order safely
    for i in prange(n):
        for j in range(i + 1, n):
            s = 0.0
            for k in range(d):
                diff = centers[i, k] - centers[j, k]
                s += diff * diff
            out[idx] = np.sqrt(s)
            idx += 1
    return out


@njit(parallel=True, fastmath=True)
def _coverage_check_numba(X, centers, radii_sq):
    n_points, n_dims = X.shape
    n_regions = centers.shape[0]
    covered = np.zeros(n_points, dtype=np.bool_)
    for i in prange(n_points):
        for j in range(n_regions):
            dist_sq = 0.0
            for k in range(n_dims):
                diff = X[i, k] - centers[j, k]
                dist_sq += diff * diff
            if dist_sq <= radii_sq[j]:
                covered[i] = True
                break
    return covered


def compute_mean_entropy_from_global_gp(surrogate_manager, X):
    """Optimized: reduced function calls and better NaN handling."""
    if surrogate_manager is None or X is None or len(X) == 0:
        return 1.0

    try:
        mean, std = surrogate_manager.predict_global_cached(X)
        # Compute variance directly, handle edge cases efficiently
        var = std * std  # Faster than **2
        mean_var = np.mean(var)
        return float(1.0 if np.isnan(mean_var) else mean_var)
    except Exception:
        return 1.0


def compute_region_spread(regions):
    """Optimized: vectorized distance computation with early returns."""
    n_regions = len(regions)
    if n_regions < 2:
        return 0.0

    # Extract centers once
    centers = np.array([r.center for r in regions])

    # For small numbers, use optimized numba version
    if n_regions <= 50:
        distances = _compute_pairwise_distances_upper_tri(centers)
        return np.mean(distances)

    # For larger numbers, use scipy's optimized cdist
    dmat = cdist(centers, centers)
    # Extract upper triangle without diagonal more efficiently
    triu_indices = np.triu_indices(n_regions, k=1)
    return np.mean(dmat[triu_indices])


def compute_coverage(X, centers, radii):
    """Highly optimized coverage computation with multiple strategies."""
    if centers.shape[0] == 0:
        return 0.0

    X = np.asarray(X)
    centers = np.asarray(centers)
    radii = np.asarray(radii)

    n_points = X.shape[0]
    n_regions = centers.shape[0]

    # Use different strategies based on problem size
    if n_points * n_regions < 10000:  # Small problems: use numba
        radii_sq = (2.0 * radii) ** 2
        covered = _coverage_check_numba(X, centers, radii_sq)
        return np.mean(covered)
    else:  # Large problems: use vectorized operations
        # Vectorized computation with memory-efficient approach
        radius_threshold = (2.0 * radii) ** 2

        # Process in chunks to avoid memory issues for very large datasets
        chunk_size = min(1000, n_points)
        covered_count = 0

        for i in range(0, n_points, chunk_size):
            end_idx = min(i + chunk_size, n_points)
            X_chunk = X[i:end_idx]

            # Compute distances for chunk
            diff = X_chunk[:, None, :] - centers[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)

            # Check coverage
            covered_chunk = np.any(dist_sq <= radius_threshold[None, :], axis=1)
            covered_count += np.sum(covered_chunk)

        return covered_count / n_points


def compute_coverage_fraction(global_X, regions):
    """Optimized with KDTree for large datasets and early returns."""
    if len(global_X) == 0 or len(regions) == 0:
        return 0.0

    global_X = np.asarray(global_X)
    centers = np.array([r.center for r in regions])
    radii = np.array([r.radius for r in regions])

    n_points = len(global_X)
    n_regions = len(regions)

    # For small problems or when scipy not available, use direct computation
    if n_points * n_regions < 5000:
        dists = cdist(global_X, centers)
        covered = np.any(dists <= radii[None, :], axis=1)
        return np.sum(covered) / n_points

    # For larger problems, use spatial tree for efficiency
    try:
        covered = np.zeros(n_points, dtype=bool)

        # Use KDTree for each region (more efficient for many points, few regions)
        for i, (center, radius) in enumerate(zip(centers, radii)):
            if np.any(~covered):  # Only continue if there are uncovered points
                tree = KDTree(global_X[~covered])
                # Query points within radius
                indices = tree.query_ball_point(center, radius)
                # Map back to original indices
                uncovered_indices = np.where(~covered)[0]
                covered[uncovered_indices[indices]] = True

        return np.sum(covered) / n_points

    except ImportError:
        # Fallback to original implementation
        dists = cdist(global_X, centers)
        covered = np.any(dists <= radii[None, :], axis=1)
        return np.sum(covered) / n_points


def compute_mean_entropy(surrogate_manager, global_X):
    """Optimized with better error handling and caching."""
    if surrogate_manager is None or len(global_X) == 0:
        return 1.0  # Return scalar for consistency

    try:
        _, var = surrogate_manager.predict(global_X, return_var=True)
        return float(np.mean(var)) if var is not None else 1.0
    except Exception:
        return 1.0


# Optimized utility functions
@njit
def np_safe_numba(x, default=0.0):
    """Numba-optimized NaN/Inf replacement."""
    if np.isnan(x) or np.isinf(x):
        return default
    return x


def np_safe(x, default=0.0):
    """Vectorized version for arrays, with numba fallback for scalars."""
    if np.isscalar(x):
        return np_safe_numba(float(x), default)

    x = np.asarray(x)
    return np.where(np.isfinite(x), x, default)


@njit
def exp_moving_avg(prev, new_val, alpha=0.1):
    """JIT-compiled for speed in tight loops."""
    return (1.0 - alpha) * prev + alpha * new_val


@njit
def sigmoid(x):
    """Numerically stable sigmoid with JIT compilation."""
    if x > 500:  # Prevent overflow
        return 1.0
    elif x < -500:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


# Optimized advanced utility functions
def _safe_norm01(v):
    """Optimized normalization with better numerical stability."""
    v = np.asarray(v, dtype=np.float64)  # Use higher precision

    # Handle NaN/Inf more efficiently
    mask = np.isfinite(v)
    if not np.any(mask):
        return np.zeros_like(v)

    v = np.where(mask, v, 0.0)
    vmax = np.max(np.abs(v))

    return v / (vmax + np.finfo(np.float64).eps) if vmax > 0 else v


def _eigen_floor_cov(C, floor=1e-6):
    """Optimized eigenvalue flooring with better numerical properties."""
    C = np.asarray(C, dtype=np.float64)  # Ensure high precision

    # Symmetrize more efficiently
    C = 0.5 * (C + C.T)
    original_trace = np.trace(C)

    # Use more stable eigendecomposition
    try:
        w, V = np.linalg.eigh(C)
        w = np.clip(w, floor, None)

        # Reconstruct with trace preservation
        C_new = V @ np.diag(w) @ V.T

        # Preserve trace more accurately
        new_trace = np.trace(C_new)
        if new_trace > np.finfo(float).eps:
            C_new *= original_trace / new_trace

        return C_new

    except np.linalg.LinAlgError:
        # Fallback: add regularization to diagonal
        return C + floor * np.eye(C.shape[0])


@njit(parallel=True, fastmath=True)
def _chol_mahal_sq_numba(centered, L):
    n_points, n_dims = centered.shape
    result = np.empty(n_points, dtype=np.float64)
    y = np.empty(n_dims, dtype=np.float64)
    for i in prange(n_points):
        # forward solve Ly = x
        for j in range(n_dims):
            tmp = centered[i, j]
            for k in range(j):
                tmp -= L[j, k] * y[k]
            y[j] = tmp / L[j, j]
        s = 0.0
        for j in range(n_dims):
            s += y[j] * y[j]
        result[i] = s
    return result


def _chol_mahal_sq(centered, L):
    """Optimized Mahalanobis distance with multiple backends."""
    if centered.shape[0] < 100:  # Small problems: use numba
        return _chol_mahal_sq_numba(centered, L)
    else:  # Large problems: use scipy's optimized solve
        y = np.linalg.solve(L, centered.T)
        return np.sum(y * y, axis=0)


@njit
def _pareto_nondominated_mask_numba(F):
    """Highly optimized Pareto front computation with numba."""
    n_points, n_objectives = F.shape
    is_nondominated = np.ones(n_points, dtype=np.bool_)

    for i in range(n_points):
        if not is_nondominated[i]:
            continue

        for j in range(n_points):
            if i == j or not is_nondominated[j]:
                continue

            # Check if j dominates i
            j_better_equal = True
            j_strictly_better = False

            for k in range(n_objectives):
                if F[j, k] > F[i, k]:  # j worse than i in objective k
                    j_better_equal = False
                    break
                elif F[j, k] < F[i, k]:  # j better than i in objective k
                    j_strictly_better = True

            if j_better_equal and j_strictly_better:
                is_nondominated[i] = False
                break

    return is_nondominated


def _pareto_nondominated_mask(F):
    """Optimized Pareto dominance check."""
    F = np.asarray(F)
    if F.shape[0] <= 1:
        return np.ones(F.shape[0], dtype=bool)

    # Use numba version for better performance
    return _pareto_nondominated_mask_numba(F)


def _try_kdtree(points, query):
    """Optimized nearest neighbor search with better fallbacks."""
    points = np.asarray(points)
    query = np.asarray(query)

    if points.shape[0] == 0:
        return np.full(len(query), np.inf)

    try:
        # Use KDTree (Cython version) for better performance
        tree = KDTree(points)
        distances, _ = tree.query(query)
        return distances
    except ImportError:
        # Fallback to direct computation
        if points.size == 0:
            return np.full(len(query), np.inf)
        return np.min(cdist(query, points), axis=1)


# Batch processing utilities for very large datasets
def compute_coverage_batch(X, centers, radii, batch_size=1000):
    """Process coverage computation in batches for memory efficiency."""
    if centers.shape[0] == 0:
        return 0.0

    n_points = len(X)
    covered_count = 0

    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch_coverage = compute_coverage(X[i:end_idx], centers, radii)
        covered_count += batch_coverage * (end_idx - i)

    return covered_count / n_points


class TrustRegion:
    """
    CMA-like TrustRegion for TuRBO-M+++ (V2):
    - Ellipsoidal covariance adaptation with forgetting + eigen-flooring
    - Velocity/uncertainty-aware radius control (bounded multiplicative steps)
    - Stable entropy from logdet(cov) vs isotropic reference
    - Fully NaN-safe; health is always in [0,1]
    """

    def __init__(self, center, radius, region_id, n_dims, dtype=np.float64):
        self.center = np.array(center, dtype=dtype)
        self.radius = float(radius)
        self.region_id = region_id
        self.n_dims = int(n_dims)

        # Performance metrics
        self.best_value = np.inf
        self.prev_best_value = np.inf
        self.trial_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.last_improvement = 0
        self.stagnation_count = 0
        self.restarts = 0

        # Velocities
        self.stagnation_velocity = 0.0  # EMA of non-improvement gap
        self.improvement_velocity = 1.0  # EMA of improvement magnitude

        # Ellipsoidal covariance (start isotropic)
        self.cov = np.eye(self.n_dims, dtype=dtype) * (self.radius**2)
        self.pca_basis = np.eye(self.n_dims, dtype=dtype)
        self.pca_eigvals = np.ones(self.n_dims, dtype=dtype)
        self.cov_updates_since_reset = 0

        # Local archive
        self.local_X = []
        self.local_y = []

        # Signals
        self.local_entropy = 1.0
        self.local_uncertainty = 1.0
        self._unc_ema = 1.0
        self._health_score_override = None
        self.spawn_score = 0.0
        self.exploration_bonus = 1.0
        self.health_decay_factor = 1.0

    # =====================================
    # Core update per new sample
    # =====================================
    def update(self, x, y, config, surrogate_var=None):
        self.trial_count += 1
        improved = bool(y < self.best_value)
        delta = max(0.0, float(self.best_value - y))

        # Velocity EMAs
        self.stagnation_velocity = 0.9 * self.stagnation_velocity + 0.1 * delta
        if improved:
            self.improvement_velocity = 0.8 * self.improvement_velocity + 0.2 * delta
        else:
            self.improvement_velocity *= 0.97

        # Track/EMA local uncertainty if provided
        if surrogate_var is not None and np.isfinite(surrogate_var):
            self._unc_ema = 0.9 * self._unc_ema + 0.1 * abs(float(surrogate_var))
            self.local_uncertainty = self._unc_ema

        # Best bookkeeping
        if improved:
            self.prev_best_value = self.best_value
            self.best_value = float(y)
            self.success_count += 1
            self.last_improvement = 0
            self.stagnation_count = 0
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.last_improvement += 1
            if self.consecutive_failures > 4:
                self.stagnation_count += 1

        # Local archive & covariance update
        self._update_local_archive(x, float(y), config.max_local_data)
        self._rank_one_cov_update(x)

        # Periodic PCA & entropy refresh
        if self.trial_count % max(5, self.n_dims) == 0:
            self._update_pca_from_cov()
            self.local_entropy = self._compute_entropy()

        # Adaptive radius (bounded)
        if getattr(config, "local_radius_adaptation", True):
            self._adaptive_radius(improved, delta, config)

        # Spawn score combines age, entropy, velocity, uncertainty
        self._update_spawn_score(config)

    # =====================================
    # Rank-one covariance update (with forgetting + PD repair)
    # =====================================
    def _rank_one_cov_update(self, x, alpha=0.12, floor=1e-9):
        """
        cov <- (1-alpha)*cov + alpha*(dx dx^T), with small eigen-flooring.
        """
        dx = (np.asarray(x, dtype=self.center.dtype) - self.center).reshape(-1, 1)
        if not np.all(np.isfinite(dx)):
            # keep at least a tiny jitter to avoid degeneration
            self.cov += 1e-9 * np.eye(self.n_dims, dtype=self.cov.dtype)
            return

        # Exponential forgetting + rank-one add
        self.cov = (1.0 - alpha) * self.cov + alpha * (dx @ dx.T)

        # Numerical stabilization only when needed
        # Ensure SPD via eigen-flooring and trace normalization
        try:
            # quick PD check: attempt cholesky
            np.linalg.cholesky(self.cov + floor * np.eye(self.n_dims))
        except np.linalg.LinAlgError:
            self._repair_cov(floor=floor)

        # Small jitter to keep PD in long runs
        self.cov += floor * np.eye(self.n_dims, dtype=self.cov.dtype)
        self.cov_updates_since_reset += 1

    def _repair_cov(self, floor=1e-9, cap=1e9):
        # Symmetrize
        C = 0.5 * (self.cov + self.cov.T)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, floor, cap)
        C2 = (V * w) @ V.T

        # Optional trace normalization towards radius^2 * n_dims
        target_tr = (self.radius**2) * self.n_dims
        tr = float(np.trace(C2))
        if np.isfinite(tr) and tr > 0:
            C2 *= target_tr / tr

        self.cov = C2

    def _update_pca_from_cov(self):
        C = 0.5 * (self.cov + self.cov.T)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, 1e12)
        idx = np.argsort(w)[::-1]
        self.pca_eigvals = w[idx]
        self.pca_basis = V[:, idx]

    # =====================================
    # Adaptive radius update (bounded multiplicative step)
    # =====================================
    def _adaptive_radius(self, improved, delta, config):
        """
        Uses improvement and uncertainty to scale radius smoothly.
        Prevents violent swings; keeps radius in [min_radius, max_radius].
        """
        # Base expand/contract multipliers
        if improved:
            base = float(getattr(config, "expansion_factor", 1.08))
        else:
            # penalize with stagnation velocity (less than 1)
            st_pen = float(np.exp(-0.5 * self.stagnation_velocity))
            base = float(getattr(config, "contraction_factor", 0.92)) * st_pen

        # Uncertainty bonus (tapers with tanh)
        unc_bonus = 1.0 + 0.3 * np.tanh(self.local_uncertainty)

        # Improvement influence (bigger delta -> a bit more expansion)
        imp_bonus = 1.0 + 0.15 * np.tanh(delta)

        scale = base * unc_bonus * imp_bonus
        # Bound the per-step change tightly
        scale = float(np.clip(scale, 0.85, 1.15))

        # Apply as multiplicative change to isotropic radius proxy
        new_radius = self.radius * scale
        new_radius = float(np.clip(new_radius, config.min_radius, config.max_radius))

        # Optionally align covariance trace to radius (keeps ellipsoid volume in check)
        tr_target = (new_radius**2) * self.n_dims
        tr = float(np.trace(self.cov))
        if np.isfinite(tr) and tr > 0:
            self.cov *= tr_target / tr

        # Update PCA cache after cov rescale
        self._update_pca_from_cov()
        self.radius = new_radius

    # =====================================
    # Local archive & entropy
    # =====================================
    def _update_local_archive(self, x, y, max_size):
        if len(self.local_X) < max_size:
            self.local_X.append(np.asarray(x, dtype=self.center.dtype).copy())
            self.local_y.append(float(y))
        else:
            # reservoir-like replacement with bounded bias
            idx = np.random.randint(0, self.trial_count + 1)
            if idx < max_size:
                self.local_X[idx] = np.asarray(x, dtype=self.center.dtype).copy()
                self.local_y[idx] = float(y)

    def _compute_entropy(self):
        """
        Entropy proxy: 0.5 * (logdet(cov) - logdet(r^2 I)) = log volume ratio.
        Clipped and NaN-safe.
        """
        C = 0.5 * (self.cov + self.cov.T)
        try:
            w = np.linalg.eigvalsh(C)
            w = np.clip(w, 1e-12, 1e12)
            logdet = float(np.sum(np.log(w)))
        except Exception:
            # fallback via SVD
            s = np.linalg.svd(C, compute_uv=False)
            s = np.clip(s, 1e-12, 1e12)
            logdet = float(np.sum(np.log(s)))

        ref_logdet = self.n_dims * np.log(self.radius**2 + 1e-12)
        val = 0.5 * (logdet - ref_logdet)
        return float(np.nan_to_num(val, nan=1.0, posinf=1.0, neginf=1.0))

    # =====================================
    # Spawn score
    # =====================================
    def _update_spawn_score(self, config):
        age_penalty = min(
            1.0, self.last_improvement / max(1, getattr(config, "max_age", 50))
        )
        vel_term = np.tanh(self.stagnation_velocity)
        entropy_term = np.tanh(self.local_entropy)
        uncertainty_term = np.tanh(self.local_uncertainty)

        val = (
            0.3 * self.success_rate
            + 0.3 * vel_term
            + 0.2 * entropy_term
            + 0.2 * uncertainty_term
            - 0.3 * age_penalty
        )
        self.spawn_score = float(np.clip(np.nan_to_num(val, nan=0.0), 0.0, 1.0))
        self.exploration_bonus = float(
            np.clip(1.0 + 0.5 * (1.0 - np.tanh(self.improvement_velocity)), 0.8, 2.0)
        )

    # =====================================
    # Properties
    # =====================================
    @property
    def success_rate(self):
        if self.trial_count <= 0:
            return 0.0
        return float(self.success_count) / float(max(1, self.trial_count))

    @property
    def is_active(self):
        return bool(self.radius > 1e-8)

    @property
    def should_restart(self):
        entropy_low = bool(self.local_entropy < 0.05)
        return (self.stagnation_count > 15 or entropy_low) and (self.radius < 0.05)

    @property
    def health_score(self):
        if self._health_score_override is not None:
            return float(self._health_score_override)
        raw = (
            0.4 * self.success_rate
            + 0.3 * (1.0 - float(self.local_entropy))
            + 0.3 * float(np.tanh(self.local_uncertainty))
        )
        return float(
            np.clip(np.nan_to_num(raw * self.health_decay_factor, nan=0.0), 0.0, 1.0)
        )

    @health_score.setter
    def health_score(self, value):
        self._health_score_override = float(
            np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
        )

    def clear_health_override(self):
        self._health_score_override = None

    def decay_health(self, factor=0.98):
        self.health_decay_factor *= float(factor)


class RunningStats:
    """
    Fast, numerically stable running stats for feature normalization.
    - Exponential moving mean & variance (EMA-Welford)
    - Optional debiasing (like Adam) for warm-up
    - Supports single sample (..., F) or batch (B, F) updates
    - Robust NaN/Inf handling via masking
    """

    __slots__ = ("shape", "alpha", "beta", "t", "mean", "var", "eps", "clip")

    def __init__(self, shape, alpha=0.01, eps=1e-8, clip=10.0):
        self.shape = shape if isinstance(shape, (tuple, list)) else (shape,)
        self.alpha = float(alpha)
        self.beta = 1.0 - self.alpha  # EMA decay
        self.t = 0  # update steps
        self.mean = np.zeros(self.shape, dtype=np.float32)
        self.var = np.ones(self.shape, dtype=np.float32)  # start with unit var
        self.eps = float(eps)
        self.clip = float(clip)

    def _update_single(self, x: np.ndarray):
        # EMA-Welford:
        # m_new = m + α*(x - m)
        # v_new = (1-α)*(v + α*(x - m)^2)  ← more stable than α*(x-m)^2 mix
        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var = self.beta * (self.var + self.alpha * (delta * delta))

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        # Accept either exact feature shape or a batch with trailing feature dims
        if x.shape == self.shape:
            x = x[None, ...]  # (1, F)
        elif x.shape[-len(self.shape) :] == self.shape:
            x = x.reshape(-1, *self.shape)  # (B, F)
        else:
            return  # silently ignore mismatched shapes (keeps your original behavior)

        # mask invalid rows (NaN/Inf anywhere in feature vector)
        mask = np.isfinite(x).all(axis=tuple(range(1, x.ndim)))
        if not mask.any():
            return

        valid = x[mask]
        # Fast path: vectorized loop over batch (small B—use Python loop to keep numpy cheap)
        for row in valid:
            self._update_single(row)
            self.t += 1

    def _debias(self, arr):
        # Debias EMA like Adam: arr_hat = arr / (1 - beta^t)
        if self.t == 0:
            return arr
        corr = 1.0 - (self.beta**self.t)
        return arr / max(corr, 1e-12)

    def normalize(self, x, debias=True):
        x = np.asarray(x, dtype=np.float32)
        if self.t == 0:
            return np.nan_to_num(x)  # no stats yet

        if not np.isfinite(x).all():
            return np.zeros_like(x, dtype=np.float32)

        mean = self._debias(self.mean) if debias else self.mean
        var = self._debias(self.var) if debias else self.var
        std = np.sqrt(var + self.eps)

        z = (x - mean) / std
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)


def safe_region_property(region, prop_name, default_value):
    """Safely get a property from a region with fallback."""
    try:
        value = getattr(region, prop_name, default_value)
        if np.isfinite(value):
            return value
        else:
            return default_value
    except:
        return default_value


def _min_dist_to_set(samples: np.ndarray, existing: np.ndarray) -> np.ndarray:
    """
    Return min Euclidean distance from each sample to the set of existing points.
    samples:  (N, D)
    existing: (R, D)
    """
    if existing.size == 0:
        return np.full(len(samples), np.inf, dtype=np.float32)
    # (N, R, D) diffs → (N, R) min distances
    diffs = samples[:, None, :] - existing[None, :, :]
    d2 = np.einsum("nrd,nrd->nr", diffs, diffs)  # squared distances
    return np.sqrt(d2.min(axis=1, keepdims=False)).astype(np.float32)


def filter_dominated_points(points):
    """Remove dominated points from a set."""
    if len(points) <= 1:
        return points
    
    points = np.asarray(points)
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(i + 1, n):
            if dominated[j]:
                continue
            
            # Check if i dominates j
            if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                dominated[j] = True
            # Check if j dominates i  
            elif np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                dominated[i] = True
                break
    
    return points[~dominated]

def hypervolume_4d_inclusion_exclusion(points, ref_point):
    """
    4D hypervolume using inclusion-exclusion principle.
    Efficient for small point sets (< 15 points).
    """
    points = filter_dominated_points(points)
    if len(points) == 0:
        return 0.0
    
    points = np.asarray(points)
    ref_point = np.asarray(ref_point)
    n = len(points)
    
    # For large point sets, use Monte Carlo approximation
    if n > 12:
        return hypervolume_4d_monte_carlo(points, ref_point)
    
    total_hv = 0.0
    
    # Iterate through all non-empty subsets using inclusion-exclusion
    for i in range(1, 2**n):
        subset_indices = []
        for j in range(n):
            if i & (1 << j):
                subset_indices.append(j)
        
        subset = points[subset_indices]
        
        # Calculate hypervolume of intersection (supremum of subset)
        intersection_point = np.maximum.reduce(subset)
        
        # Check if intersection point is valid (dominated by reference point)
        if np.all(intersection_point < ref_point):
            # Volume of this intersection
            diff = ref_point - intersection_point
            subset_hv = np.prod(np.maximum(diff, 0))
            
            # Apply inclusion-exclusion principle
            if len(subset_indices) % 2 == 1:
                total_hv += subset_hv
            else:
                total_hv -= subset_hv
    
    return max(0.0, total_hv)

def hypervolume_4d_monte_carlo(points, ref_point, n_samples=50000):
    """Monte Carlo approximation for 4D hypervolume."""
    points = filter_dominated_points(points)
    if len(points) == 0:
        return 0.0
    
    points = np.asarray(points)
    ref_point = np.asarray(ref_point)
    
    # Determine sampling bounds
    min_bounds = np.minimum.reduce(points)
    max_bounds = ref_point
    
    # Generate random samples in the bounding box
    samples = np.random.uniform(min_bounds, max_bounds, size=(n_samples, 4))
    
    # Count samples that are dominated by at least one point
    dominated_count = 0
    for sample in samples:
        for point in points:
            if np.all(point <= sample):
                dominated_count += 1
                break
    
    # Estimate hypervolume
    bounding_volume = np.prod(max_bounds - min_bounds)
    return (dominated_count / n_samples) * bounding_volume

def hypervolume_improvement_4d(existing_points, new_point, ref_point):
    """Calculate 4D hypervolume improvement when adding a new point."""
    new_point = np.asarray(new_point)
    ref_point = np.asarray(ref_point)
    
    if len(existing_points) == 0:
        # Single point hypervolume
        diff = ref_point - new_point
        if np.any(diff <= 0):
            return 0.0
        return np.prod(diff)
    
    existing_points = np.asarray(existing_points)
    
    # Quick check: is new point dominated?
    for point in existing_points:
        if np.all(point <= new_point) and np.any(point < new_point):
            return 0.0
    
    # Calculate hypervolume before and after
    hv_before = hypervolume_4d_inclusion_exclusion(existing_points, ref_point)
    
    extended_points = np.vstack([existing_points, new_point.reshape(1, -1)])
    hv_after = hypervolume_4d_inclusion_exclusion(extended_points, ref_point)
    
    return max(0.0, hv_after - hv_before)


class RegionManager:
    """
    Fixed RegionManager for TuRBO-M+++ with robust NaN handling
    - KDTree diversity (fallback to NumPy)
    - Pareto-front spawn selection (EI/UCB/grad/div)
    - Cholesky-based Mahalanobis assignment (no inverses)
    - Covariance eigen-flooring + trace norm
    - Robust radius & health adaptation with NaN protection
    """
    def __init__(self, config, verbose=True):
        self.config  = config
        self.verbose = verbose
        self.regions = []
        self.surrogate_manager = None
        self._entropy_buffer = collections.deque(maxlen=12)
        self._ema_progress = 0.0
        self._iteration = 0

        self._feat_dim = 9

        # ---- NEW: feature gates
        self.use_neural_radius    = getattr(config, "use_neural_radius", False)
        self.use_neural_attention = getattr(config, "use_neural_attention", False)
        self.neural_device        = getattr(config, "neural_device", "cpu")

        # ---- neural radius agent (if enabled & torch available)
        self.radius_agent = None
        if self.use_neural_radius and TORCH_AVAILABLE:
            try:
                rl_config = RLConfig(
                    buffer_cap=max(2048, getattr(config, "max_evals", 1000) // 10),
                    lr=3e-4,
                    train_every=max(1, getattr(config, "max_evals", 1000) // 1000),
                    batch=min(128, max(32, getattr(config, "max_evals", 1000) // 100)),
                    delta_max=0.15,
                )
            except Exception:
                rl_config = RLConfig()
            self.radius_agent = RadiusLearnerAgent(d_in=self._feat_dim,
                                                   cfg=rl_config,
                                                   device=self.neural_device)

        # ---- neural attention (if enabled & torch available)
        self.region_attn = None
        if self.use_neural_attention and TORCH_AVAILABLE:
            self.region_attn = RegionAttention(d_in=self._feat_dim, d_model=64, n_heads=8)

        # rest as-is …
        self._feature_normalizer = None
        self._centers_cache = None
        self.use_hypervolume = getattr(config, "use_hypervolume", True)
        self.verbose = True

    def _region_features(self, r) -> np.ndarray:
        # Pull raw values (allow missing), then one-shot sanitize
        init_radius = getattr(self.config, "init_radius", 0.1)
        n_dims = max(1, safe_region_property(r, "n_dims", 2))
        stagn_norm = max(10.0, 2.0 * n_dims)

        current_radius = max(safe_region_property(r, "radius", init_radius), 1e-12)

        raw = np.array(
            [
                self._iteration
                / (getattr(self.config, "max_evals", 1000) + 1e-9),  # progress
                safe_region_property(r, "success_rate", 0.0),  # success
                safe_region_property(r, "stagnation_count", 0) / stagn_norm,  # stagn
                safe_region_property(r, "local_entropy", 0.5),  # entropy
                safe_region_property(r, "local_uncertainty", 1.0),  # uncertainty
                safe_region_property(r, "improvement_velocity", 0.0),  # vel
                current_radius / (init_radius + 1e-12),  # rel_radius
                np.log(current_radius) - np.log(init_radius + 1e-12),  # logr
                self._region_diversity_score(r) if self.regions else 1.0,  # diversity
            ],
            dtype=np.float32,
        )

        # Replace any bad values with configured safe defaults in one go
        defaults = np.array(
            [0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32
        )
        raw = np.where(np.isfinite(raw), raw, defaults)

        # Init normalizer lazily
        if self._feature_normalizer is None:
            self._feature_normalizer = RunningStats(len(raw))

        # Update stats (batched update accepts (B,F); here it’s a single row)
        self._feature_normalizer.update(raw)

        # Normalize + bound
        z = self._feature_normalizer.normalize(raw)
        z = np.clip(z, -10.0, 10.0)
        z = np.tanh(z)

        if not np.isfinite(z).all():
            if self.verbose:
                print(
                    f"[WARNING] NaN features for region {safe_region_property(r, 'region_id', 'unknown')} → zeros"
                )
            z = np.zeros_like(z, dtype=np.float32)
        return z

    def _compute_adaptive_reward(self, region, delta, old_radius):
        """
        Improved reward signal with robust NaN handling.
        """
        # Improvement component with safety checks
        prev_best = np_safe(safe_region_property(region, "prev_best_value", np.inf))
        current_best = np_safe(safe_region_property(region, "best_value", np.inf))

        if not (np.isfinite(prev_best) and np.isfinite(current_best)):
            delta_impr = 0.0
        else:
            delta_impr = max(0.0, prev_best - current_best)

        # Scale by local data standard deviation if available
        scale = 1.0
        if hasattr(region, "local_y") and len(region.local_y) > 5:
            try:
                local_std = np.std(region.local_y)
                if np.isfinite(local_std) and local_std > 1e-12:
                    scale = local_std
            except:
                scale = 1.0

        improvement_reward = np_safe(delta_impr / (scale + 1e-6))

        # Stability penalty with safety
        trial_count = max(1, safe_region_property(region, "trial_count", 1))
        experience_factor = min(1.0, trial_count / 50.0)
        stability_penalty = np_safe(abs(delta) * experience_factor)

        # Exploration bonus
        entropy = np_safe(safe_region_property(region, "local_entropy", 0.0))
        exploration_bonus = entropy * 0.2

        # Diversity bonus
        try:
            diversity_bonus = np_safe(self._region_diversity_score(region) * 0.1)
        except:
            diversity_bonus = 0.0

        # Combined reward with safety
        raw_reward = (
            math.tanh(improvement_reward)
            + exploration_bonus
            + diversity_bonus
            - 0.1 * stability_penalty
        )

        # Final safety check
        result = np_safe(raw_reward)

        # Clip to reasonable range
        result = float(np.clip(result, -10.0, 10.0))

        return result

    def _region_diversity_score(self, region):
        """
        Compute diversity score with robust error handling.
        """
        if len(self.regions) <= 1:
            return 1.0

        try:
            # Get centers of other regions
            others = []
            for r in self.regions:
                if r is not region and hasattr(r, "center"):
                    center = getattr(r, "center", None)
                    if center is not None and np.all(np.isfinite(center)):
                        others.append(center)

            if not others:
                return 1.0

            others = np.array(others)
            region_center = getattr(region, "center", None)

            if region_center is None or not np.all(np.isfinite(region_center)):
                return 0.5  # Default for invalid center

            # Compute distances safely
            distances = np.linalg.norm(others - region_center, axis=1)

            if len(distances) == 0 or not np.any(np.isfinite(distances)):
                return 1.0

            d = np.min(distances[np.isfinite(distances)])

            # Normalize by local radius
            radius = max(safe_region_property(region, "radius", 0.1), 1e-9)
            result = d / radius

            return np_safe(result, default=1.0)

        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Error in diversity score computation: {e}")
            return 1.0

    # ---------------------------------------------
    # Compatibility
    # ---------------------------------------------
    def set_surrogate_manager(self, surrogate_manager):
        self.surrogate_manager = surrogate_manager

    # ---------------------------------------------
    # Initialization
    # ---------------------------------------------
    def initialize_regions(self, X, y, n_dims, rng=None):
        rng = np.random.default_rng(None if rng is None else rng)
        n_init = min(getattr(self.config, "n_regions", 5), len(X))
        best_idx = int(np.argmin(y))
        selected_idx = [best_idx]

        # perf + diversity greedy selection
        D = cdist(X, X)
        perf_w = np.exp(-0.05 * (y - np.min(y)))
        for _ in range(1, n_init):
            min_d = np.min(D[:, selected_idx], axis=1)
            s = perf_w * (min_d + 1e-9)
            s[selected_idx] = -np.inf
            selected_idx.append(int(np.argmax(s)))

        for rid, center in enumerate(X[selected_idx]):
            base_r = np.percentile(np.linalg.norm(X - center, axis=1), 25)
            base_r = np.clip(
                base_r,
                getattr(self.config, "min_radius", 0.01),
                getattr(self.config, "init_radius", 0.1),
            )
            self.regions.append(TrustRegion(center, float(base_r), rid, n_dims))

        self._refresh_centers_cache()
        if self.verbose:
            print(f"[INIT] {len(self.regions)} trust regions created")

    # ---------------------------------------------
    # Main lifecycle
    # ---------------------------------------------
    def manage_regions(self, bounds, n_dims, rng, global_X, global_y, iteration=0):
        self._iteration = iteration
        self._ema_progress = exp_moving_avg(
            self._ema_progress,
            iteration / (getattr(self.config, "max_evals", 1000) + 1e-12),
            alpha=0.1,
        )
        self._adapt_all(bounds, rng)
        self._maybe_spawn(bounds, n_dims, rng, global_X, global_y)
        self._ensure_min_diversity(bounds, n_dims, rng, global_X, global_y)
        self._adaptive_prune()
        self._refresh_centers_cache()

    def _adapt_all(self, bounds, rng):
        """
        Improved adaptive region management with robust error handling.
        (Neural radius adaptation is gated by self.use_neural_radius)
        """
        dead = []

        for r in self.regions:
            try:
                # --- EMA velocity
                current_vel = np_safe(safe_region_property(r, "improvement_velocity", 0.0))
                old_vel_ema = np_safe(safe_region_property(r, "vel_ema", 0.0))
                r.vel_ema = exp_moving_avg(old_vel_ema, current_vel, alpha=0.3)
                if not np.isfinite(r.vel_ema):
                    r.vel_ema = 0.0

                # --- Covariance refresh from local archive
                if hasattr(r, "local_X") and len(r.local_X) > max(
                    8, safe_region_property(r, "n_dims", 2)
                ):
                    try:
                        local_X_array = np.asarray(r.local_X)
                        if np.all(np.isfinite(local_X_array)) and np.all(np.isfinite(r.center)):
                            centered = local_X_array - r.center
                            sample_cov = np.cov(centered.T) + 1e-9 * np.eye(r.n_dims)
                            if np.all(np.isfinite(sample_cov)):
                                sample_cov = _eigen_floor_cov(sample_cov, floor=1e-9)
                                mix_rate = 0.3 if len(r.local_X) > 20 else 0.1
                                if not np.all(np.isfinite(r.cov)):
                                    r.cov = np.eye(r.n_dims) * (r.radius**2)
                                r.cov = (1 - mix_rate) * r.cov + mix_rate * sample_cov
                                if not np.all(np.isfinite(r.cov)):
                                    r.cov = np.eye(r.n_dims) * (r.radius**2)
                    except Exception as e:
                        if self.verbose:
                            print(f"[WARNING] Covariance update failed for region {r.region_id}: {e}")
                        r.cov = np.eye(r.n_dims) * (r.radius**2)

                old_r = np_safe(r.radius, default=getattr(self.config, "init_radius", 0.1))

                # ====== NEURAL vs CLASSIC RADIUS UPDATE ======
                new_radius, delta = None, 0.0

                if self.use_neural_radius and (self.radius_agent is not None):
                    try:
                        feats = self._region_features(r)
                        if np.all(np.isfinite(feats)):
                            new_radius, delta = self.radius_agent.policy.step(
                                feats,
                                old_r,
                                getattr(self.config, "min_radius", 0.01),
                                getattr(self.config, "max_radius", 1.0),
                            )
                        else:
                            raise ValueError("Invalid features for neural policy")
                    except Exception as e:
                        if self.verbose:
                            print(f"[WARNING] Learned adaptation failed for region {r.region_id}: {e}")
                        new_radius = None  # fall through

                if new_radius is None:
                    # Classic heuristic fallback (or primary if neural disabled)
                    v = np_safe(getattr(r, "vel_ema", 0.0), default=0.0)
                    if v < 0.01:
                        new_radius = min(old_r * 1.05, getattr(self.config, "max_radius", 1.0))
                    elif v > 0.1:
                        new_radius = max(old_r * 0.92, getattr(self.config, "min_radius", 0.01))
                    else:
                        new_radius = float(
                            np.clip(
                                old_r * 0.995,
                                getattr(self.config, "min_radius", 0.01),
                                getattr(self.config, "max_radius", 1.0),
                            )
                        )
                    delta = math.log(max(new_radius, 1e-12) / max(old_r, 1e-12))

                if not np.isfinite(new_radius) or new_radius <= 0:
                    new_radius = max(old_r, getattr(self.config, "min_radius", 0.01))
                    delta = 0.0

                r.radius = float(new_radius)

                # Align cov trace to new radius
                try:
                    tr_target = (r.radius**2) * r.n_dims
                    tr = float(np.trace(r.cov))
                    if np.isfinite(tr) and tr > 1e-12:
                        scale_factor = tr_target / tr
                        if np.isfinite(scale_factor) and 0.1 <= scale_factor <= 10.0:
                            r.cov *= scale_factor
                        else:
                            r.cov = np.eye(r.n_dims) * (r.radius**2)
                    else:
                        r.cov = np.eye(r.n_dims) * (r.radius**2)
                except Exception:
                    r.cov = np.eye(r.n_dims) * (r.radius**2)

                # Update PCA cache
                if hasattr(r, "_update_pca_from_cov"):
                    try:
                        r._update_pca_from_cov()
                    except Exception:
                        pass

                # Health score
                try:
                    div = self._region_diversity_score(r)
                    ent = np_safe(safe_region_property(r, "local_entropy", 0.5))
                    last_impr = max(0, safe_region_property(r, "last_improvement", 0))
                    n_dims = max(1, safe_region_property(r, "n_dims", 2))
                    age_penalty = min(0.5, last_impr / max(20, n_dims * 3))
                    raw_h = 0.4 * np_safe(r.vel_ema) + 0.3 * (1.0 - ent) + 0.2 * div - 0.1 * age_penalty
                    r.health_score = float(np.clip(np_safe(raw_h), 0.0, 1.0))
                    current_decay = np_safe(safe_region_property(r, "health_decay_factor", 1.0))
                    r.health_decay_factor = max(0.1, current_decay * 0.995)
                except Exception as e:
                    if self.verbose:
                        print(f"[WARNING] Health score update failed for region {r.region_id}: {e}")
                    r.health_score = 0.5

                if self.verbose and abs(old_r - r.radius) > 1e-3:
                    print(f"[ADAPT] R#{r.region_id} radius {old_r:.3f}→{r.radius:.3f}")

                # Restart / death rules
                should_restart = safe_region_property(r, "should_restart", False)
                health_score = np_safe(safe_region_property(r, "health_score", 0.5))

                if should_restart or (
                    r.radius < 2.0 * getattr(self.config, "min_radius", 0.01) and health_score < 0.15
                ):
                    if hasattr(self, "_restart_region"):
                        try:
                            self._restart_region(r, bounds, rng)
                        except Exception as e:
                            if self.verbose:
                                print(f"[WARNING] Restart failed for region {r.region_id}: {e}")

                if (r.radius < 1.5 * getattr(self.config, "min_radius", 0.01)) and (health_score < 0.1):
                    dead.append(r)

                # ====== Buffer training data only when neural adaptation is on ======
                if self.use_neural_radius and (self.radius_agent is not None):
                    try:
                        reward = self._compute_adaptive_reward(r, delta, old_r)
                        feats = self._region_features(r)
                        if np.all(np.isfinite(feats)) and np.isfinite(reward):
                            self.radius_agent.push(feats, delta, reward)
                    except Exception as e:
                        if self.verbose:
                            print(f"[WARNING] Reward computation failed for region {r.region_id}: {e}")

            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Region {r.region_id} adaptation failed: {e}")
                dead.append(r)

        # Train policy (if enabled)
        if self.use_neural_radius and (self.radius_agent is not None):
            try:
                self.radius_agent.maybe_train()
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Training step failed: {e}")

        # Replace dead regions if needed
        for r in dead:
            if hasattr(self, "_replace_dead_region"):
                try:
                    self._replace_dead_region(r, bounds, rng)
                except Exception as e:
                    if self.verbose:
                        print(f"[WARNING] Failed to replace dead region {r.region_id}: {e}")
                    self.regions = [rr for rr in self.regions if rr is not r]

    # ---------------------------------------------
    # Spawning new regions (Pareto + scores)
    # ---------------------------------------------
    def _maybe_spawn(self, bounds, n_dims, rng, global_X, global_y):
        coverage = compute_coverage_fraction(global_X, self.regions)
        entropy = compute_mean_entropy_from_global_gp(self.surrogate_manager, global_X)
        self._entropy_buffer.append(np_safe(entropy))
        sm_entropy = (
            float(np.mean(self._entropy_buffer)) if self._entropy_buffer else 0.5
        )

        avg_health = (
            np.mean(
                [
                    np_safe(safe_region_property(r, "health_score", 0.5))
                    for r in self.regions
                ]
            )
            if self.regions
            else 1.0
        )
        exploration_pressure = (1.0 - coverage) + sm_entropy
        exploitation_pressure = self._ema_progress + avg_health

        trigger = sigmoid(3.0 * (exploration_pressure - exploitation_pressure)) > 0.45
        dynamic_cap = int(
            getattr(self.config, "n_regions", 5) * (1.0 + 0.5 * exploration_pressure)
        )

        if trigger and len(self.regions) < dynamic_cap:
            self._force_spawn(bounds, n_dims, rng, global_X, global_y)

    def _estimate_grad_norms_batch(self, X, eps=1e-3):
        if (
            self.surrogate_manager is not None
            and hasattr(self.surrogate_manager, "global_backend")
            and self.surrogate_manager.global_backend == "exact_gp"
            and hasattr(self.surrogate_manager, "global_model")
            and self.surrogate_manager.global_model is not None
        ):
            try:
                grad = self.surrogate_manager.gradient_global_mean(X)  # (N, D)
                grad_norm = np.linalg.norm(grad, axis=1)
                _, std = self.surrogate_manager.predict_global_cached(X)
                return np.nan_to_num(grad_norm * (std + 1e-6))
            except:
                pass

        # fallback: just std
        if self.surrogate_manager is None:
            return np.ones(len(X))

        try:
            _, std = self.surrogate_manager.predict_global_cached(X)
            return np.nan_to_num(std)
        except:
            return np.ones(len(X))

    def _diversity_bonus(self, candidates):
        if not self.regions:
            return np.ones(len(candidates), dtype=np.float32)
        existing = np.asarray([r.center for r in self.regions], dtype=np.float32)
        dmin = _min_dist_to_set(np.asarray(candidates, dtype=np.float32), existing)
        # Avoid zeros; return as-is (bigger → better)
        return np.maximum(dmin, 1e-12)

    def _compute_spawn_radius(self, candidate, existing, global_X, k=8):
        if existing is None or existing.size == 0:
            return getattr(self.config, "init_radius", 0.1)
        # spacing to existing centers
        d_near = np.median(np.linalg.norm(existing - candidate, axis=1))
        # local data density via kNN distances on global_X
        if global_X is not None and len(global_X) > 0:
            if HAS_KDTREE and len(global_X) >= k:
                try:
                    tree = KDTree(global_X)
                    d_k, _ = tree.query(candidate, k=min(k, len(global_X)))
                    d_local = np.median(np.atleast_1d(d_k))
                except:
                    d_local = np.median(np.linalg.norm(global_X - candidate, axis=1))
            else:
                d_local = np.median(np.linalg.norm(global_X - candidate, axis=1))
        else:
            d_local = d_near
        r = float(min(d_near, d_local))
        return float(
            np.clip(
                r,
                getattr(self.config, "min_radius", 0.01),
                getattr(self.config, "init_radius", 0.1),
            )
        )


    def _find_best_spawn_hypervolume(self, bounds, n_dims, rng, global_X, global_y):
        """
        Find best spawn location using 4D hypervolume improvement.
        Objectives: -EI, UCB, -grad_bonus, -diversity (all minimized)
        """
        try:
            sobol = i4_sobol_generate(n_dims, 512)
            cand = bounds[:, 0] + sobol * (bounds[:, 1] - bounds[:, 0])

            # Add jitters around healthy regions
            if self.regions:
                health = np.array([
                    np_safe(safe_region_property(r, "health_score", 0.0))
                    for r in self.regions
                ])
                top = np.argsort(-health)[:min(3, len(self.regions))]
                for idx in top:
                    c = self.regions[idx].center
                    r = self.regions[idx].radius
                    J = c + rng.normal(size=(64, n_dims)) * (0.25 * r)
                    J = np.clip(J, bounds[:, 0], bounds[:, 1])
                    cand = np.vstack([cand, J])

            # Get surrogate predictions
            mean, std = self.surrogate_manager.predict_global_cached(cand)
            f_best = float(np.min(global_y)) if len(global_y) else float(np.min(mean))

            # Compute all objectives (same as original but keep raw values)
            z = (f_best - mean) / (std + 1e-12)
            ei = (f_best - mean) * norm.cdf(z) + std * norm.pdf(z)
            ucb = mean - 2.0 * std  # lower better
            grad_bonus = self._estimate_grad_norms_batch(cand, eps=2e-3)
            div = self._diversity_bonus(cand)

            # Normalize objectives to [0,1] range for stable hypervolume calculation
            def norm01(arr):
                a = np.asarray(arr)
                amin, amax = np.min(a), np.max(a)
                if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
                    return np.zeros_like(a)
                return (a - amin) / (amax - amin)

            # Create 4D objective matrix (all objectives to be minimized)
            objectives = np.column_stack([
                -norm01(ei),          # minimize -EI (maximize EI)
                norm01(ucb),          # minimize UCB  
                -norm01(grad_bonus),  # minimize -grad_bonus (maximize grad_bonus)
                -norm01(div),         # minimize -diversity (maximize diversity)
            ])
            
            # Reference point for hypervolume (nadir point)
            ref_point = np.array([1.2, 1.2, 1.2, 1.2])  # Slightly beyond [0,1] range
            
            # Find current Pareto front in 4D objective space
            pareto_mask = _pareto_nondominated_mask(objectives)
            pareto_front = objectives[pareto_mask]
            
            if self.verbose and len(pareto_front) > 0:
                print(f"[HV-SPAWN] Pareto front has {len(pareto_front)} points")
            
            # Calculate hypervolume improvement for each candidate
            hv_improvements = []
            
            # For efficiency, only calculate HV improvement for promising candidates
            # (Pareto front members + some additional high-potential points)
            if len(pareto_front) > 8:
                # If Pareto front is large, focus on front members and top EI candidates
                ei_top_mask = (-objectives[:, 0]) >= np.percentile(-objectives[:, 0], 90)
                candidate_mask = pareto_mask | ei_top_mask
            else:
                # Small front, evaluate all candidates
                candidate_mask = np.ones(len(objectives), dtype=bool)
            
            for i, obj_vec in enumerate(objectives):
                if not candidate_mask[i]:
                    hv_improvements.append(0.0)
                    continue
                    
                try:
                    if pareto_mask[i]:
                        # Point is on Pareto front, calculate its unique contribution
                        other_points = pareto_front[~np.all(pareto_front == obj_vec, axis=1)]
                        if len(other_points) > 0:
                            hv_improvement = hypervolume_improvement_4d(other_points, obj_vec, ref_point)
                        else:
                            # Only point on front
                            diff = ref_point - obj_vec
                            hv_improvement = np.prod(np.maximum(diff, 0))
                    else:
                        # Point not on front, calculate improvement if added
                        hv_improvement = hypervolume_improvement_4d(pareto_front, obj_vec, ref_point)
                    
                    hv_improvements.append(hv_improvement)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"[WARNING] HV calculation failed for candidate {i}: {e}")
                    hv_improvements.append(0.0)
            
            hv_improvements = np.array(hv_improvements)
            
            # Select candidate with maximum hypervolume improvement
            if np.max(hv_improvements) > 1e-12:
                best_idx = np.argmax(hv_improvements)
                if self.verbose:
                    print(f"[HV-SPAWN] Best HV improvement: {hv_improvements[best_idx]:.6f}")
                    obj = objectives[best_idx]
                    print(f"[HV-SPAWN] Objectives: EI={-obj[0]:.3f}, UCB={obj[1]:.3f}, "
                        f"Grad={-obj[2]:.3f}, Div={-obj[3]:.3f}")
                return cand[best_idx]
            else:
                # Fallback to best point on Pareto front (highest EI among front members)
                if len(pareto_front) > 0:
                    front_indices = np.where(pareto_mask)[0]
                    ei_values = -objectives[front_indices, 0]  # Higher is better for EI
                    best_front_idx = front_indices[np.argmax(ei_values)]
                    if self.verbose:
                        print("[HV-SPAWN] No HV improvement, using best EI from Pareto front")
                    return cand[best_front_idx]
                else:
                    # Ultimate fallback: best EI point overall
                    best_idx = np.argmax(-objectives[:, 0])
                    if self.verbose:
                        print("[HV-SPAWN] No Pareto front, using best EI")
                    return cand[best_idx]

        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 4D Hypervolume spawn failed: {e}")
            # Fallback to random
            return bounds[:, 0] + rng.uniform(0, 1, n_dims) * (bounds[:, 1] - bounds[:, 0])

    # Modified _find_best_spawn method for your RegionManager class:
    def _find_best_spawn(self, bounds, n_dims, rng, global_X, global_y):
        """
        Enhanced spawn finding with hypervolume improvement option.
        Set use_hypervolume=True in config to enable hypervolume-based selection.
        """
        
        if self.use_hypervolume:
            return self._find_best_spawn_hypervolume(bounds, n_dims, rng, global_X, global_y)
        
        # Original Chebyshev implementation (keep as fallback)
        try:
            sobol = i4_sobol_generate(n_dims, 512)
            cand = bounds[:, 0] + sobol * (bounds[:, 1] - bounds[:, 0])

            # local jitters around healthiest centers
            if self.regions:
                health = np.array([
                    np_safe(safe_region_property(r, "health_score", 0.0))
                    for r in self.regions
                ])
                top = np.argsort(-health)[: min(3, len(self.regions))]
                for idx in top:
                    c = self.regions[idx].center
                    r = self.regions[idx].radius
                    J = c + rng.normal(size=(64, n_dims)) * (0.25 * r)
                    J = np.clip(J, bounds[:, 0], bounds[:, 1])
                    cand = np.vstack([cand, J])

            mean, std = self.surrogate_manager.predict_global_cached(cand)
            f_best = float(np.min(global_y)) if len(global_y) else float(np.min(mean))

            z = (f_best - mean) / (std + 1e-12)
            ei = (f_best - mean) * norm.cdf(z) + std * norm.pdf(z)
            ucb = mean - 2.0 * std  # lower better

            grad_bonus = self._estimate_grad_norms_batch(cand, eps=2e-3)
            div = self._diversity_bonus(cand)

            # In-place min-max normalization helpers
            def norm01(arr):
                a = np.asarray(arr)
                amin, amax = np.min(a), np.max(a)
                if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
                    return np.zeros_like(a)
                return (a - amin) / (amax - amin)

            F = np.column_stack([
                -norm01(ei),  # maximize EI  → minimize -EI
                norm01(ucb),  # minimize UCB
                -norm01(grad_bonus),  # maximize grad bonus
                -norm01(div),  # maximize diversity
            ])

            nd_mask = _pareto_nondominated_mask(F)
            PF = F[nd_mask]
            C = cand[nd_mask]

            w = np.array([0.45, 0.25, 0.20, 0.10], dtype=np.float64)
            # Chebyshev scalarization
            PF_shift = PF - PF.min(axis=0, keepdims=True)
            cheb = np.max(w * PF_shift, axis=1)
            best_idx = int(np.argmin(cheb))
            return C[best_idx]

        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Best spawn search failed: {e}")
            # Fallback to random
            return bounds[:, 0] + rng.uniform(0, 1, n_dims) * (bounds[:, 1] - bounds[:, 0])
        
    def _force_spawn(self, bounds, n_dims, rng, global_X, global_y):
        try:
            cand = self._find_best_spawn(bounds, n_dims, rng, global_X, global_y)

            existing = (
                np.array([r.center for r in self.regions])
                if self.regions
                else np.empty((0, n_dims))
            )
            if existing.size > 0:
                min_dist = float(np.min(np.linalg.norm(existing - cand, axis=1)))
                if min_dist < 0.3 * getattr(self.config, "init_radius", 0.1):
                    cand = self._maximin_diverse(bounds, n_dims, rng, existing)
                    if self.verbose:
                        print("[SPAWN] Candidate adjusted for diversity")

            radius = self._compute_spawn_radius(cand, existing, global_X)
            new_r = TrustRegion(cand, radius, len(self.regions), n_dims)
            new_r.exploration_bonus = 1.4
            self.regions.append(new_r)
            if self.verbose:
                print(f"[SPAWN] Region#{new_r.region_id} @ {np.round(cand, 3)}")
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Force spawn failed: {e}")

    def _maximin_diverse(self, bounds, n_dims, rng, existing):
        try:
            samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(128, n_dims))
            dmin = _min_dist_to_set(
                samples.astype(np.float32), existing.astype(np.float32)
            )
            return samples[int(np.argmax(dmin))]
        except Exception:
            # Fallback to uniform
            return bounds[:, 0] + rng.uniform(0, 1, n_dims) * (
                bounds[:, 1] - bounds[:, 0]
            )

    # ---------------------------------------------
    # Adaptive pruning
    # ---------------------------------------------
    def _adaptive_prune(self):
        if not self.regions:
            return
        max_pts = getattr(self.config, "max_total_points", None)
        if not max_pts:
            return
        total_points = sum(self._len_local(r) for r in self.regions)
        if total_points <= max_pts:
            return

        pareto_score = [
            0.6 * np_safe(safe_region_property(r, "health_score", 0.0))
            + 0.4 * self._region_diversity_score(r)
            for r in self.regions
        ]
        order = np.argsort(pareto_score)  # worst first
        remove_n = max(1, int(0.2 * len(self.regions)))
        for idx in order[:remove_n]:
            victim = self.regions[idx]
            if self.verbose:
                print(f"[PRUNE] Region#{victim.region_id} pruned")
            # pop by identity (keep IDs stable if you rely on them)
            self.regions = [r for r in self.regions if r is not victim]

    def _len_local(self, region):
        return (
            len(region.local_X) if getattr(region, "local_X", None) is not None else 0
        )

    # ---------------------------------------------
    # Diversity floor
    # ---------------------------------------------
    def _ensure_min_diversity(self, bounds, n_dims, rng, global_X, global_y):
        min_needed = max(3, int(getattr(self.config, "n_regions", 5) * 0.5))
        while len(self.regions) < min_needed:
            self._force_spawn(bounds, n_dims, rng, global_X, global_y)
            if self.verbose:
                print(f"[DIVERSITY] Forced spawn → {len(self.regions)}/{min_needed}")

    # ---------------------------------------------
    # Region weights
    # ---------------------------------------------
    def safe_region_weights(self, regions):
        """Region weighting with optional neural attention; robust fallbacks."""
        if not regions:
            return np.array([], dtype=float)

        # Neural attention path (gated)
        if self.use_neural_attention and TORCH_AVAILABLE and (self.region_attn is not None):
            try:
                feats_batch = np.stack([self._region_features(r) for r in regions])
                w = self.region_attn.scores(feats_batch)
                if np.all(np.isfinite(w)) and w.sum() > 1e-12:
                    return w / w.sum()
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Attention failed: {e}")

        # Heuristic fallback
        health = np.array([np_safe(safe_region_property(r, "health_score", 0.0)) for r in regions])
        diversity = np.array([self._region_diversity_score(r) for r in regions])
        age = np.array([1.0 / (1.0 + safe_region_property(r, "last_improvement", 0) / 10.0) for r in regions])
        combined = 0.5 * health + 0.3 * diversity + 0.2 * age
        ex = np.exp(combined / 0.8)
        return ex / (ex.sum() + 1e-12)

    # ---------------------------------------------
    # Assign new data to regions (Cholesky Mahalanobis)
    # ---------------------------------------------
    def update_regions_with_new_data(self, X_new, y_new):
        if not self.regions:
            return
        active = [r for r in self.regions if safe_region_property(r, "is_active", True)]
        if not active:
            return
        try:
            n_dims = safe_region_property(active[0], "n_dims", 2)
            centers = np.ascontiguousarray(
                np.stack([r.center for r in active], axis=0), dtype=np.float64
            )

            # Build Cholesky factors (R, D, D)
            L_list = []
            eye = np.eye(n_dims)
            for r in active:
                C = getattr(r, "cov", None)
                if C is None or not np.isfinite(C).all():
                    C = (r.radius**2 + 1e-9) * eye
                try:
                    L = np.linalg.cholesky(C + 1e-12 * eye)
                except np.linalg.LinAlgError:
                    C = _eigen_floor_cov(C, floor=1e-9)
                    L = np.linalg.cholesky(C + 1e-12 * eye)
                L_list.append(L)
            L_stack = np.ascontiguousarray(np.stack(L_list, axis=0), dtype=np.float64)

            X_new = np.ascontiguousarray(X_new, dtype=np.float64)
            diffs = X_new[:, None, :] - centers[None, :, :]  # (N, R, D)

            # Compute Mahalanobis distances per region (R typically small → loop over R is fine)
            N, R, D = diffs.shape
            mahal_sq = np.empty((N, R), dtype=np.float64)
            for j in range(R):
                # solve L y = diff^T → y^T = diff @ L^{-T}; since N can be large, do matmul once
                # Use triangular solve via np.linalg.solve in chunks to keep memory sane
                # Here: y = diff @ inv(L), then mahal^2 = ||y||^2
                y = diffs[:, j, :].dot(np.linalg.inv(L_stack[j]))
                mahal_sq[:, j] = np.einsum("nd,nd->n", y, y)

            avg_radius = float(np.mean([r.radius for r in active]))
            beta = max(
                3.0,
                10.0
                * (getattr(self.config, "init_radius", 0.1) / (avg_radius + 1e-12)),
            )
            with np.errstate(over="ignore"):
                w = np.exp(-beta * mahal_sq)
            w_sum = w.sum(axis=1, keepdims=True) + 1e-12
            w /= w_sum

            # Assign & update
            for i, (x, y) in enumerate(zip(X_new, y_new)):
                rid = int(np.argmax(w[i]))
                if w[i, rid] > 0.05:
                    active[rid].update(x.astype(np.float32), float(y), self.config)

        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Region update failed: {e}")

    # ---------------------------------------------
    # internals
    # ---------------------------------------------
    def _refresh_centers_cache(self):
        if self.regions:
            try:
                self._centers_cache = np.stack([r.center for r in self.regions])
            except:
                self._centers_cache = None
        else:
            self._centers_cache = None

    # --- RNG compat (Generator or RandomState) ---
    def _rng_uniform(self, rng, low, high, size):
        try:
            if hasattr(rng, "uniform"):
                return rng.uniform(low, high, size=size)  # Generator
            return low + (high - low) * rng.rand(*size)  # RandomState
        except:
            # Fallback to numpy random
            return low + (high - low) * np.random.rand(*size)

    # --- In-place restart of an existing region (keep same ID) ---
    def _restart_region(self, region, bounds, rng):
        try:
            n_dims = safe_region_property(region, "n_dims", 2)
            lo, hi = bounds[:, 0], bounds[:, 1]

            # propose a new center: prefer diversity; fall back to uniform
            existing = np.array([r.center for r in self.regions if r is not region])
            if existing.size > 0:
                cand = self._maximin_diverse(bounds, n_dims, rng, existing)
            else:
                cand = self._rng_uniform(rng, lo, hi, size=(n_dims,))

            # radius based on spacing & local density
            radius = self._compute_spawn_radius(
                cand,
                existing if existing.size > 0 else np.empty((0, n_dims)),
                getattr(self, "_global_X", None)
                if hasattr(self, "_global_X")
                else None,
            )

            # reset core state
            region.center = cand.astype(float)
            region.radius = float(
                np.clip(
                    radius,
                    getattr(self.config, "min_radius", 0.01),
                    getattr(self.config, "init_radius", 0.1),
                )
            )
            region.best_value = np.inf
            region.prev_best_value = np.inf
            region.trial_count = 0
            region.success_count = 0
            region.consecutive_failures = 0
            region.last_improvement = 0
            region.stagnation_count = 0
            region.restarts = getattr(region, "restarts", 0) + 1

            region.stagnation_velocity = 0.0
            region.improvement_velocity = 1.0
            region.local_entropy = 1.0
            region.local_uncertainty = 1.0
            region._unc_ema = 1.0
            region._health_score_override = None
            region.spawn_score = 0.0
            region.exploration_bonus = 1.2
            region.health_decay_factor = 1.0

            # clear local archive
            region.local_X = []
            region.local_y = []

            # reset covariance to isotropic ellipsoid ~ r^2 I and PCA cache
            r2 = region.radius**2
            region.cov = np.eye(n_dims) * r2
            region.pca_basis = np.eye(n_dims)
            region.pca_eigvals = np.ones(n_dims) * r2
            region.cov_updates_since_reset = 0

            if self.verbose:
                print(
                    f"[RESTART] Region#{region.region_id} → center={np.round(region.center, 3)}, r={region.radius:.3f}"
                )

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Restart region failed: {e}")

    # --- Replace a "dead" region with a fresh one (keeps list size) ---
    def _replace_dead_region(self, victim, bounds, rng):
        try:
            n_dims = safe_region_property(victim, "n_dims", 2)

            # Try to spawn a strong candidate using your existing logic
            try:
                cand = self._find_best_spawn(
                    bounds,
                    n_dims,
                    rng,
                    getattr(self, "_global_X", []),
                    getattr(self, "_global_y", []),
                )
            except Exception:
                # fallback: just diversify
                existing = np.array([r.center for r in self.regions if r is not victim])
                if existing.size > 0:
                    cand = self._maximin_diverse(bounds, n_dims, rng, existing)
                else:
                    lo, hi = bounds[:, 0], bounds[:, 1]
                    cand = self._rng_uniform(rng, lo, hi, size=(n_dims,))

            existing = np.array([r.center for r in self.regions if r is not victim])
            radius = self._compute_spawn_radius(
                cand,
                existing if existing.size > 0 else np.empty((0, n_dims)),
                getattr(self, "_global_X", None)
                if hasattr(self, "_global_X")
                else None,
            )

            # Re-initialize victim in place (keeps region_id stable)
            victim.center = cand.astype(float)
            victim.radius = float(
                np.clip(
                    radius,
                    getattr(self.config, "min_radius", 0.01),
                    getattr(self.config, "init_radius", 0.1),
                )
            )
            victim.best_value = np.inf
            victim.prev_best_value = np.inf
            victim.trial_count = 0
            victim.success_count = 0
            victim.consecutive_failures = 0
            victim.last_improvement = 0
            victim.stagnation_count = 0
            victim.restarts = getattr(victim, "restarts", 0) + 1

            victim.stagnation_velocity = 0.0
            victim.improvement_velocity = 1.0
            victim.local_entropy = 1.0
            victim.local_uncertainty = 1.0
            victim._unc_ema = 1.0
            victim._health_score_override = None
            victim.spawn_score = 0.0
            victim.exploration_bonus = 1.2
            victim.health_decay_factor = 1.0
            victim.local_X = []
            victim.local_y = []
            r2 = victim.radius**2
            victim.cov = np.eye(n_dims) * r2
            victim.pca_basis = np.eye(n_dims)
            victim.pca_eigvals = np.ones(n_dims) * r2
            victim.cov_updates_since_reset = 0

            if self.verbose:
                print(
                    f"[REPLACE] Region#{victim.region_id} → center={np.round(victim.center, 3)}, r={victim.radius:.3f}"
                )

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Replace dead region failed: {e}")
