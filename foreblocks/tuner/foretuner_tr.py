# --- add inside TrustRegion ---
from __future__ import annotations

import collections
import math
import warnings
from dataclasses import dataclass

# ============================================
# ✅ Core Python & Concurrency
# ============================================
from typing import Dict, Optional, Tuple

# ============================================
# ✅ Numerical & Scientific Computing
# ============================================
import numpy as np
import numpy.typing as npt

# Optimized with Numba JIT compilation for speed
from numba import jit, njit, prange
from pykdtree.kdtree import KDTree

# ============================================
# ✅ Parallelism & Performance
# ============================================
from scipy.spatial.distance import cdist  # if you already have this path, keep it
from scipy.stats import norm
from sobol_seq import i4_sobol_generate

from .foretuner_acq import *

# ============================================
# ✅ Project-Specific
# ============================================
from .foretuner_aux import *
from .foretuner_rl import *
from .foretuner_sur import *

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
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

try:
    from scipy.spatial import cKDTree as KDTreeFast
    HAS_FAST_KD = True
except Exception:
    HAS_FAST_KD = False




class TrustRegion:
    """
    Ellipsoidal Trust Region (CMA-lite) for TuRBO-style BO

    Key features:
      - Rank-1 EMA covariance (SPD, Cholesky-checked) with trace tied to radius^2 * n
      - Log-radius update with burst-on-fail & catastrophic-fail handling
      - Entropy from log |C| via Cholesky (cheap, NaN-safe)
      - Novelty via mean Mahalanobis distance (archive vs center)
      - Ring-buffer archive (O(1) memory)
      - Exposes metric to candidate generator: metric_scales(), chol()
      - Optional ridge direction memory for elongation: prev_dir_white
    """

    _defaults = dict(
        # radius bounds
        min_radius=1e-3, max_radius=1.0,
        # covariance (EMA) + numerics
        cov_alpha=0.12, cov_floor=1e-9, cov_cap=1e9, kappa_max=1e6,
        # local archive
        max_local_data=128,
        # radius adaptation
        local_radius_adaptation=True,
        expansion_factor=1.08, contraction_factor=0.92,
        radius_step_clip=(0.85, 1.15),
        # restart / aging
        max_age=50, restart_entropy_thresh=0.05,
        restart_radius_thresh=0.05, restart_stagnation_steps=15,
        # scores (weights)
        novelty_weight=0.2, entropy_weight=0.2, uncertainty_weight=0.2,
        velocity_weight=0.3, success_weight=0.3, age_penalty_weight=0.3,
        # health
        health_decay_init=1.0,
        health_success_w=0.4, health_entropy_w=0.3, health_uncertainty_w=0.3,
        # dtype
        dtype=np.float64,
        # ---- burst / catastrophic knobs ----
        burst_on_fail=True,
        burst_min_failures=6,
        burst_factor=1.35,
        burst_entropy_thresh=0.10,
        burst_uncertainty_thresh=0.75,
        burst_cooldown=10,
        catastrophic_expand_factor=1.50,
        catastrophic_reset_cov=True,
        # direction memory (for elongation in whitened space)
        dir_beta=0.35, dir_cap=3.0,
    )

    def __init__(self, center, radius, region_id: int, n_dims: int, **kwargs):
        for k, v in {**self._defaults, **kwargs}.items():
            setattr(self, k, v)

        self.n = int(n_dims)
        self.n_dims = self.n
        self.center = np.asarray(center, dtype=self.dtype).reshape(-1)
        assert self.center.shape[0] == self.n

        self.radius = float(np.clip(radius, self.min_radius, self.max_radius))
        self.region_id = int(region_id)

        # perf counters
        self.best_value = math.inf
        self.prev_best_value = math.inf
        self.trial_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.last_improvement = 0
        self.stagnation_count = 0
        self.restarts = 0

        # velocities
        self.stagnation_velocity = 0.0
        self.improvement_velocity = 1.0

        # covariance (tie trace to radius^2 * n)
        self.cov = np.eye(self.n, dtype=self.dtype) * (self.radius ** 2)
        self._chol = np.linalg.cholesky(self.cov + self.cov_floor * np.eye(self.n, dtype=self.dtype))
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))  # log|C|
        self.cov_updates_since_reset = 0

        # archive (ring)
        m = self.max_local_data
        self.local_X = np.zeros((m, self.n), dtype=self.dtype)
        self.local_y = np.full((m,), math.inf, dtype=self.dtype)
        self._buf_size = 0
        self._buf_ptr = 0

        # signals
        self.local_entropy = 1.0
        self.local_uncertainty = 1.0
        self._unc_ema = 1.0
        self._health_score_override = None
        self.spawn_score = 0.0
        self.exploration_bonus = 1.0
        self.health_decay_factor = self.health_decay_init

        # internals
        self._log_radius = self._safe_log(self.radius)
        self._last_burst_trial = -10
        self._trial_index = 0

        # ridge direction memory for ellipsoidal elongation (whitened coords)
        self.prev_dir_white = np.zeros(self.n, dtype=self.dtype)
        self.radius_long = self.radius
        self.radius_lat  = self.radius

    # ---------------- core update ----------------
    def update(self, x, y, surrogate_var: Optional[float] = None) -> Dict[str, float]:
        self.trial_count += 1
        self._trial_index += 1

        x = np.asarray(x, dtype=self.dtype).reshape(-1)
        y_is_finite = np.isfinite(y)
        y_val = float(y) if y_is_finite else np.inf

        improved = (y_val < self.best_value)
        delta = max(0.0, float(self.best_value - y_val)) if np.isfinite(self.best_value) else 0.0

        # velocities
        self.stagnation_velocity = 0.9 * self.stagnation_velocity + 0.1 * delta
        self.improvement_velocity = (0.8 * self.improvement_velocity + 0.2 * delta) if improved else (self.improvement_velocity * 0.97)

        # uncertainty EMA
        sv = self._to_float_maybe(surrogate_var)
        if sv is not None and np.isfinite(sv):
            self._unc_ema = 0.9 * self._unc_ema + 0.1 * abs(sv)
            self.local_uncertainty = self._unc_ema

        # best bookkeeping
        if improved:
            self.prev_best_value = self.best_value
            self.best_value = y_val
            self.success_count += 1
            self.last_improvement = 0
            self.stagnation_count = 0
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.last_improvement += 1
            if self.consecutive_failures > 4:
                self.stagnation_count += 1

        # archive + covariance
        self._archive_push(x, y_val)
        self._rank_one_cov_update(x)

        # entropy from current Cholesky (vs isotropic of same radius)
        self.local_entropy = self._entropy_from_chol()

        # radius schedule + burst / catastrophic
        if self.local_radius_adaptation:
            self._adaptive_radius(improved, delta, catastrophic_fail=(not y_is_finite))

        # spawn/bonus
        self._update_spawn_score()

        return {
            "success_rate": self.success_rate,
            "entropy": self.local_entropy,
            "uncertainty": self.local_uncertainty,
            "stagn_vel": self.stagnation_velocity,
            "impr_vel": self.improvement_velocity,
            "spawn_score": self.spawn_score,
            "radius": self.radius,
        }

    # ---------------- covariance ----------------
    def _rank_one_cov_update(self, x: np.ndarray):
        alpha, floor = self.cov_alpha, self.cov_floor
        dx = (x - self.center).reshape(-1, 1)

        if not self._all_finite_array(dx):
            self.cov += floor * np.eye(self.n, dtype=self.dtype)
        else:
            self.cov = (1.0 - alpha) * self.cov + alpha * (dx @ dx.T)
            self.cov = 0.5 * (self.cov + self.cov.T)

        # SPD & conditioning via Cholesky; repair if needed
        self._ensure_spd(floor=floor)
        self._align_cov_to_radius()
        self.cov_updates_since_reset += 1

    def _ensure_spd(self, floor: float):
        C = 0.5 * (self.cov + self.cov.T) + floor * np.eye(self.n, dtype=self.dtype)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # eigen repair only when necessary
            w, V = np.linalg.eigh(C)
            w = np.clip(w, floor, self.cov_cap)
            # cap condition number
            w_max = float(np.max(w))
            w_min = max(float(np.min(w)), floor)
            if w_max / max(w_min, floor) > self.kappa_max:
                w_min_new = max(w_max / self.kappa_max, floor)
                w = np.maximum(w, w_min_new)
            C = (V * w) @ V.T
            L = np.linalg.cholesky(C)
        self.cov = C
        self._chol = L
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(L))))

    def _align_cov_to_radius(self):
        target_tr = float((self.radius ** 2) * self.n)
        tr = float(np.trace(self.cov))
        if np.isfinite(tr) and tr > 0.0:
            s = target_tr / tr
            self.cov *= s
            self._chol *= math.sqrt(s)  # keep Cholesky in sync
            self._logdet += self.n * math.log(s)

    def _entropy_from_chol(self) -> float:
        # entropy proxy: 0.5(log|C| - log|r^2 I|)
        ref_logdet = self.n * self._safe_log(self.radius ** 2)
        val = 0.5 * (self._logdet - ref_logdet)
        return float(np.clip(val, -50.0, 50.0))

    # ---------------- radius ----------------
    def _adaptive_radius(self, improved: bool, delta: float, catastrophic_fail: bool = False):
        if catastrophic_fail:
            self.radius = float(np.clip(self.radius * self.catastrophic_expand_factor, self.min_radius, self.max_radius))
            if self.catastrophic_reset_cov:
                self._reset_cov_isotropic()
            self._align_cov_to_radius()
            return

        base = self.expansion_factor if improved else (self.contraction_factor * math.exp(-0.5 * self.stagnation_velocity))
        scale = base * (1.0 + 0.30 * self._safe_tanh(self.local_uncertainty)) * (1.0 + 0.15 * self._safe_tanh(delta))

        # burst gate on repeated fails
        burst_fired = False
        if (self.burst_on_fail and not improved and
            self.consecutive_failures >= self.burst_min_failures and
            (self._trial_index - self._last_burst_trial) >= self.burst_cooldown):
            stuck = (self.local_entropy < self.burst_entropy_thresh)
            unsure = (self._safe_tanh(self.local_uncertainty) > self.burst_uncertainty_thresh)
            if stuck or unsure:
                scale = max(scale, self.burst_factor)
                self._last_burst_trial = self._trial_index
                burst_fired = True
                # mild isotropization
                self.cov = 0.8 * self.cov + 0.2 * (np.eye(self.n, dtype=self.dtype) * (self.radius ** 2))
                self._ensure_spd(self.cov_floor)

        lo, hi = self.radius_step_clip
        if burst_fired:
            hi = max(hi, self.burst_factor)
        scale = float(np.clip(scale, lo, hi))

        self._log_radius = self._safe_log(self.radius) + self._safe_log(scale)
        self.radius = float(np.clip(math.exp(self._log_radius), self.min_radius, self.max_radius))
        self._align_cov_to_radius()

    def _reset_cov_isotropic(self):
        self.cov[:] = 0.0
        diag = (self.radius ** 2)
        for i in range(self.n):
            self.cov[i, i] = diag
        self._chol = np.linalg.cholesky(self.cov + self.cov_floor * np.eye(self.n, dtype=self.dtype))
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))

    # ---------------- archive / novelty ----------------
    def _archive_push(self, x: np.ndarray, y: float):
        m = self.max_local_data
        i = self._buf_ptr
        self.local_X[i, :] = x
        self.local_y[i] = y
        self._buf_ptr = (i + 1) % m
        self._buf_size = min(self._buf_size + 1, m)

    def _mean_mahalanobis(self) -> float:
        k = self._buf_size
        if k == 0:
            return 0.0
        L = self._chol
        dx = (self.local_X[:k, :] - self.center[None, :])
        z = np.linalg.solve(L, dx.T)
        d2 = np.sum(z * z, axis=0)
        return float(np.mean(np.sqrt(np.maximum(d2, 0.0))))

    # ---------------- scores ----------------
    def _update_spawn_score(self):
        age_penalty = min(1.0, self.last_improvement / max(1, self.max_age))
        vel_term = self._safe_tanh(self.stagnation_velocity)
        entropy_term = self._safe_tanh(self.local_entropy)
        uncertainty_term = self._safe_tanh(self.local_uncertainty)
        novelty_term = self._safe_tanh(self._mean_mahalanobis())
        val = (
            self.success_weight    * self.success_rate +
            self.velocity_weight   * 0.5 * (vel_term + self.novelty_weight * novelty_term) +
            self.entropy_weight    * entropy_term +
            self.uncertainty_weight* uncertainty_term -
            self.age_penalty_weight* age_penalty
        )
        self.spawn_score = float(np.clip(np.nan_to_num(val, nan=0.0), 0.0, 1.0))
        self.exploration_bonus = float(np.clip(1.0 + 0.5 * (1.0 - self._safe_tanh(self.improvement_velocity)), 0.8, 2.0))

    # ---------------- public: metric to sampler ----------------
    def metric_scales(self) -> np.ndarray:
        """
        Return per-dimension Mahalanobis weights w_j ≈ 1 / ell_j^2 using diagonal(C)^-1.
        Cheap and robust; use with batch diversity. For exact distances, use chol().
        """
        diag = np.clip(np.diag(self.cov), self.cov_floor, self.cov_cap)
        return (1.0 / np.maximum(diag, self.cov_floor)).astype(self.dtype)

    def chol(self) -> np.ndarray:
        """Return Cholesky factor of current covariance (lower triangular)."""
        return self._chol.copy()

    # optional: let the sampler update ridge direction in whitened space
    def update_direction(self, step_white: np.ndarray):
        """Call this on success with step in whitened coords."""
        g = np.asarray(step_white, dtype=self.dtype).reshape(-1)
        nrm = float(np.linalg.norm(g))
        if nrm < 1e-12:
            return
        u = g / nrm
        beta = self.dir_beta
        self.prev_dir_white = beta * u + (1.0 - beta) * self.prev_dir_white
        nu = float(np.linalg.norm(self.prev_dir_white))
        if nu > 1e-12:
            self.prev_dir_white /= nu
        # adjust elongation radii (sampler reads radius_long/lat)
        self.radius_long = min(self.radius_long * 1.6, self.dir_cap * self.radius)
        self.radius_lat  = max(self.radius_lat  / math.sqrt(1.6), 0.25 * self.radius)

    def decay_direction(self):
        """Call on failed step to relax elongation back toward isotropy."""
        self.radius_long = self.radius
        self.radius_lat  = self.radius

    # ---------------- properties ----------------
    @property
    def success_rate(self) -> float:
        return 0.0 if self.trial_count <= 0 else float(self.success_count) / float(self.trial_count)

    @property
    def is_active(self) -> bool:
        return bool(self.radius > 1e-8)

    @property
    def should_restart(self) -> bool:
        entropy_low = bool(self.local_entropy < self.restart_entropy_thresh)
        small = (self.radius < self.restart_radius_thresh)
        stuck = (self.stagnation_count > self.restart_stagnation_steps)
        return (stuck or entropy_low) and small

    # ---------------- helpers ----------------
    @staticmethod
    def _safe_tanh(x: float) -> float:
        return float(np.tanh(np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)))

    @staticmethod
    def _safe_log(x: float) -> float:
        return float(np.log(max(float(x), 1e-12)))

    @staticmethod
    def _to_float_maybe(x):
        if x is None:
            return None
        try:
            import torch  # type: ignore
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().reshape(-1)[0])
        except Exception:
            pass
        try:
            arr = np.asarray(x)
            if arr.size == 0:
                return None
            return float(arr.reshape(-1)[0])
        except Exception:
            return None

    @staticmethod
    def _all_finite_array(a: np.ndarray) -> bool:
        try:
            a = np.asarray(a, dtype=float)
            return np.isfinite(a).all()
        except Exception:
            return False

    # ------------- archive view / maintenance -------------
    def local_len(self) -> int:
        return int(self._buf_size)

    def local_view(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X[:k], y[:k]) in time order (oldest→newest)."""
        k, m, i = int(self._buf_size), int(self.local_X.shape[0]), int(self._buf_ptr)
        if k <= 0:
            return self.local_X[:0], self.local_y[:0]
        if k < m or i == 0:
            return self.local_X[:k], self.local_y[:k]
        Xv = np.vstack((self.local_X[i:], self.local_X[:i]))
        yv = np.concatenate((self.local_y[i:], self.local_y[:i]))
        return Xv, yv

    def clear_archive(self):
        m = self.local_X.shape[0]
        self.local_X[:] = 0.0
        self.local_y[:] = np.inf
        self._buf_size = 0
        self._buf_ptr = 0

# assume helpers exist in your codebase:
# - safe_region_property, np_safe, exp_moving_avg, sigmoid
# - compute_coverage_fraction, compute_mean_entropy_from_global_gp
# - _eigen_floor_cov, _pareto_nondominated_mask, _min_dist_to_set
# - i4_sobol_generate
# - KDTree + HAS_KDTREE guard
# - TORCH_AVAILABLE, RLConfig, RadiusLearnerAgent, RegionAttention
# - TrustRegion (your fixed, ring-buffer version)

class RegionManager:
    """
    RegionManager for TuRBO-M+++ (fixed & faster)
    - Archive-agnostic (list or ring-buffer ndarray)
    - Never passes config into TrustRegion.update (bug fix)
    - Covariance refresh uses ring buffer correctly
    - Safer, slightly faster feature & diversity paths
    - Restart/replace initialize ring buffer archives properly
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

        # ---- feature gates (unchanged interface)
        self.use_neural_radius    = getattr(config, "use_neural_radius", False)
        self.use_neural_attention = getattr(config, "use_neural_attention", False)
        self.neural_device        = getattr(config, "neural_device", "cpu")
        print(f"[INFO] RegionManager: use_neural_radius={self.use_neural_radius}, use_neural_attention={self.use_neural_attention}, neural_device={self.neural_device}")

        self.radius_agent = None
        if self.use_neural_radius and TORCH_AVAILABLE:
            print("[INFO] Initializing neural radius agent...")
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
            self.radius_agent = RadiusLearnerAgent(d_in=self._feat_dim, cfg=rl_config, device=self.neural_device)

        self.region_attn = None
        if self.use_neural_attention and TORCH_AVAILABLE:
            self.region_attn = RegionAttention(d_in=self._feat_dim, d_model=64, n_heads=8)

        self._feature_normalizer = None
        self._centers_cache = None
        self.use_hypervolume = getattr(config, "use_hypervolume", True)

    # --------------- small archive helpers ---------------
    @staticmethod
    def _local_arrays(region):
        """
        Return (X_local, y_local, k) as numpy arrays.
        Uses TrustRegion.local_view() / local_len() if available,
        otherwise falls back to ring-buffer/list detection.
        """
        # Fast path: TrustRegion API
        if hasattr(region, "local_view"):
            Xl, yl = region.local_view()               # views; already sliced to k
            k = Xl.shape[0]
            if k <= 0:
                return None, None, 0
            # ensure contiguous/float for downstream libs
            return (np.ascontiguousarray(Xl, dtype=float),
                    np.ascontiguousarray(yl, dtype=float),
                    k)

        # Fallback: infer ring-buffer / list
        lx = getattr(region, "local_X", None)
        ly = getattr(region, "local_y", None)
        if lx is None or ly is None:
            return None, None, 0

        if isinstance(lx, np.ndarray) and hasattr(region, "_buf_size"):
            k = int(getattr(region, "_buf_size", lx.shape[0]))
            if k <= 0:
                return None, None, 0
            return (np.ascontiguousarray(lx[:k], dtype=float),
                    np.ascontiguousarray(ly[:k], dtype=float),
                    k)

        # list/tuple fallback
        try:
            k = len(lx)
            if k <= 0:
                return None, None, 0
            return (np.ascontiguousarray(lx, dtype=float),
                    np.ascontiguousarray(ly, dtype=float),
                    k)
        except Exception:
            return None, None, 0


    @staticmethod
    def _len_local(region):
        if hasattr(region, "local_len"):
            return int(region.local_len())
        _, _, k = RegionManager._local_arrays(region)
        return int(k)

    # -----------------------------
    # Feature extraction
    # -----------------------------
    def _region_features(self, r) -> np.ndarray:
        init_radius = getattr(self.config, "init_radius", 0.1)
        n_dims = max(1, safe_region_property(r, "n_dims", 2))
        stagn_norm = max(10.0, 2.0 * n_dims)

        current_radius = max(safe_region_property(r, "radius", init_radius), 1e-12)

        raw = np.array(
            [
                self._iteration / (getattr(self.config, "max_evals", 1000) + 1e-9),
                safe_region_property(r, "success_rate", 0.0),
                safe_region_property(r, "stagnation_count", 0) / stagn_norm,
                safe_region_property(r, "local_entropy", 0.5),
                safe_region_property(r, "local_uncertainty", 1.0),
                safe_region_property(r, "improvement_velocity", 0.0),
                current_radius / (init_radius + 1e-12),
                np.log(current_radius) - np.log(init_radius + 1e-12),
                self._region_diversity_score(r) if self.regions else 1.0,
            ],
            dtype=np.float32,
        )

        defaults = np.array([0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, defaults)

        if self._feature_normalizer is None:
            self._feature_normalizer = RunningStats(len(raw))
        self._feature_normalizer.update(raw)

        z = self._feature_normalizer.normalize(raw)
        z = np.clip(z, -10.0, 10.0)
        z = np.tanh(z)

        if not np.isfinite(z).all():
            if self.verbose:
                rid = safe_region_property(r, 'region_id', 'unknown')
                print(f"[WARNING] NaN features for region {rid} → zeros")
            z = np.zeros_like(z, dtype=np.float32)
        return z

    # -----------------------------
    # Reward
    # -----------------------------
    def _compute_adaptive_reward(self, region, delta, old_radius):
        prev_best = np_safe(safe_region_property(region, "prev_best_value", np.inf))
        current_best = np_safe(safe_region_property(region, "best_value", np.inf))
        delta_impr = 0.0 if not (np.isfinite(prev_best) and np.isfinite(current_best)) else max(0.0, prev_best - current_best)

        scale = 1.0
        Xl, yl, k = self._local_arrays(region)
        if k > 5:
            local_std = np.std(yl)
            if np.isfinite(local_std) and local_std > 1e-12:
                scale = local_std

        improvement_reward = np_safe(delta_impr / (scale + 1e-6))
        trial_count = max(1, safe_region_property(region, "trial_count", 1))
        experience_factor = min(1.0, trial_count / 50.0)
        stability_penalty = np_safe(abs(delta) * experience_factor)

        entropy = np_safe(safe_region_property(region, "local_entropy", 0.0))
        exploration_bonus = entropy * 0.2

        try:
            diversity_bonus = np_safe(self._region_diversity_score(region) * 0.1)
        except:
            diversity_bonus = 0.0

        raw_reward = (
            math.tanh(improvement_reward)
            + exploration_bonus
            + diversity_bonus
            - 0.1 * stability_penalty
        )
        return float(np.clip(np_safe(raw_reward), -10.0, 10.0))

    # -----------------------------
    # Diversity score
    # -----------------------------
    def _region_diversity_score(self, region):
        if len(self.regions) <= 1:
            return 1.0
        try:
            centers = []
            Ls = []
            for r in self.regions:
                if r is region:
                    continue
                c = getattr(r, "center", None)
                if c is None or not np.all(np.isfinite(c)):
                    continue
                centers.append(np.asarray(c, dtype=float))
                try:
                    Ls.append(r.chol() if hasattr(r, "chol") else np.linalg.cholesky(r.cov + 1e-9 * np.eye(len(c))))
                except Exception:
                    Ls.append(None)
            if not centers:
                return 1.0
            centers = np.stack(centers, axis=0)
            rc = np.asarray(region.center, dtype=float)
            # use region’s own metric if available, fallback to Euclidean
            try:
                L = region.chol() if hasattr(region, "chol") else np.linalg.cholesky(region.cov + 1e-9 * np.eye(len(rc)))
                diffs = centers - rc[None, :]
                Y = np.linalg.solve(L.T, diffs.T).T
                d = np.sqrt(np.sum(Y * Y, axis=1))
            except Exception:
                d = np.linalg.norm(centers - rc[None, :], axis=1)

            dmin = float(np.min(d)) if d.size else 0.0
            # scale by radius (unitless)
            radius = max(safe_region_property(region, "radius", 0.1), 1e-9)
            return np_safe(dmin / radius, default=1.0)
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Diversity score error: {e}")
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
        self._global_X = np.asarray(global_X) if global_X is not None else None
        self._global_y = np.asarray(global_y) if global_y is not None else None
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

    # ---------------------------------------------
    # Adaptation
    # ---------------------------------------------
    def _adapt_all(self, bounds, rng):
        dead = []
        for r in self.regions:
            try:
                current_vel = np_safe(safe_region_property(r, "improvement_velocity", 0.0))
                old_vel_ema = np_safe(safe_region_property(r, "vel_ema", 0.0))
                r.vel_ema = exp_moving_avg(old_vel_ema, current_vel, alpha=0.3)
                if not np.isfinite(r.vel_ema):
                    r.vel_ema = 0.0

                # Covariance refresh from local archive (supports ring buffer)
                # Optional archive-driven refresh (only if low entropy + enough points)
                Xl, yl, k = self._local_arrays(r)
                if k > max(8, safe_region_property(r, "n_dims", 2)):
                    ent = np_safe(safe_region_property(r, "local_entropy", 0.5))
                    if ent < 0.10:  # only steer geometry when the region collapsed
                        try:
                            centered = Xl - r.center
                            S = np.cov(centered.T) + 1e-9 * np.eye(r.n_dims)
                            S = _eigen_floor_cov(S, floor=1e-9)
                            mix = 0.15 if k > 20 else 0.07
                            C_new = (1 - mix) * getattr(r, "cov", np.eye(r.n_dims) * (r.radius**2)) + mix * S
                            # keep SPD and align trace if TR exposes helpers
                            if hasattr(r, "_ensure_spd"):
                                r.cov = C_new
                                r._ensure_spd(getattr(r, "cov_floor", 1e-9))
                                if hasattr(r, "_align_cov_to_radius"):
                                    r._align_cov_to_radius()
                            else:
                                r.cov = C_new
                        except Exception as e:
                            if self.verbose:
                                print(f"[WARNING] Covariance refresh failed R#{r.region_id}: {e}")
                            r.cov = np.eye(r.n_dims) * (r.radius**2)

                old_r = np_safe(r.radius, default=getattr(self.config, "init_radius", 0.1))

                # Neural vs heuristic radius
                new_radius, delta = None, 0.0
                if self.use_neural_radius and (self.radius_agent is not None):
                    try:
                        feats = self._region_features(r)
                        if np.all(np.isfinite(feats)):
                            new_radius, delta = self.radius_agent.step(
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
                        new_radius = None

                if new_radius is None:
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

                # Align covariance trace to new radius
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

                #if self.verbose and abs(old_r - r.radius) > 1e-3:
                    #print(f"[ADAPT] R#{r.region_id} radius {old_r:.3f}→{r.radius:.3f}")

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

                # Train buffer for neural policy
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

        if self.use_neural_radius and (self.radius_agent is not None):
            try:
                self.radius_agent.maybe_train()
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Training step failed: {e}")

        for r in dead:
            if hasattr(self, "_replace_dead_region"):
                try:
                    self._replace_dead_region(r, bounds, rng)
                except Exception as e:
                    if self.verbose:
                        print(f"[WARNING] Failed to replace dead region {r.region_id}: {e}")
                    self.regions = [rr for rr in self.regions if rr is not r]

    # ---------------------------------------------
    # Spawning
    # ---------------------------------------------
    def _maybe_spawn(self, bounds, n_dims, rng, global_X, global_y):
        coverage = compute_coverage_fraction(global_X, self.regions)
        entropy = compute_mean_entropy_from_global_gp(self.surrogate_manager, global_X)
        self._entropy_buffer.append(np_safe(entropy))
        sm_entropy = float(np.mean(self._entropy_buffer)) if self._entropy_buffer else 0.5

        avg_health = (
            np.mean([np_safe(safe_region_property(r, "health_score", 0.5)) for r in self.regions])
            if self.regions else 1.0
        )
        exploration_pressure = (1.0 - coverage) + sm_entropy
        exploitation_pressure = self._ema_progress + avg_health

        trigger = sigmoid(3.0 * (exploration_pressure - exploitation_pressure)) > 0.45
        dynamic_cap = int(getattr(self.config, "n_regions", 5) * (1.0 + 0.5 * exploration_pressure))

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
        return np.maximum(dmin, 1e-12)

    def _compute_spawn_radius(self, candidate, existing, global_X, k=8):
        if existing is None or existing.size == 0:
            return getattr(self.config, "init_radius", 0.1)
        d_near = np.median(np.linalg.norm(existing - candidate, axis=1))
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
        return float(np.clip(r, getattr(self.config, "min_radius", 0.01), getattr(self.config, "init_radius", 0.1)))

    def _find_best_spawn(self, bounds, n_dims, rng, global_X, global_y):
        sobol = i4_sobol_generate(n_dims, 512)
        cand = bounds[:, 0] + sobol * (bounds[:, 1] - bounds[:, 0])

        if self.regions:
            health = np.array([np_safe(safe_region_property(r, "health_score", 0.0)) for r in self.regions])
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
        ucb = mean - 2.0 * std

        grad_bonus = self._estimate_grad_norms_batch(cand, eps=2e-3)
        div = self._diversity_bonus(cand)

        def norm01(a):
            a = np.asarray(a)
            amin, amax = np.min(a), np.max(a)
            if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
                return np.zeros_like(a)
            return (a - amin) / (amax - amin)

        F = np.column_stack([
            -norm01(ei),
            norm01(ucb),
            -norm01(grad_bonus),
            -norm01(div),
        ])

        nd_mask = _pareto_nondominated_mask(F)
        PF = F[nd_mask]
        C = cand[nd_mask]

        w = np.array([0.45, 0.25, 0.20, 0.10], dtype=np.float64)
        PF_shift = PF - PF.min(axis=0, keepdims=True)
        cheb = np.max(w * PF_shift, axis=1)
        best_idx = int(np.argmin(cheb))
        return C[best_idx]

    def _force_spawn(self, bounds, n_dims, rng, global_X, global_y):
        try:
            cand = self._find_best_spawn(bounds, n_dims, rng, global_X, global_y)
            existing = np.array([r.center for r in self.regions]) if self.regions else np.empty((0, n_dims))
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
            dmin = _min_dist_to_set(samples.astype(np.float32), existing.astype(np.float32))
            return samples[int(np.argmax(dmin))]
        except Exception:
            return bounds[:, 0] + rng.uniform(0, 1, n_dims) * (bounds[:, 1] - bounds[:, 0])

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
            0.6 * np_safe(safe_region_property(r, "health_score", 0.0)) + 0.4 * self._region_diversity_score(r)
            for r in self.regions
        ]
        order = np.argsort(pareto_score)  # worst first
        remove_n = max(1, int(0.2 * len(self.regions)))
        for idx in order[:remove_n]:
            victim = self.regions[idx]
            if self.verbose:
                print(f"[PRUNE] Region#{victim.region_id} pruned")
            self.regions = [r for r in self.regions if r is not victim]

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
        if not regions:
            return np.array([], dtype=float)

        if self.use_neural_attention and TORCH_AVAILABLE and (self.region_attn is not None):
            try:
                feats_batch = np.stack([self._region_features(r) for r in regions])
                w = self.region_attn.scores(feats_batch)
                if np.all(np.isfinite(w)) and w.sum() > 1e-12:
                    return w / w.sum()
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Attention failed: {e}")

        health = np.array([np_safe(safe_region_property(r, "health_score", 0.0)) for r in regions])
        diversity = np.array([self._region_diversity_score(r) for r in regions])
        age = np.array([1.0 / (1.0 + safe_region_property(r, "last_improvement", 0) / 10.0) for r in regions])
        combined = 0.5 * health + 0.3 * diversity + 0.2 * age
        ex = np.exp(combined / 0.8)
        return ex / (ex.sum() + 1e-12)

    # ---------------------------------------------
    # Assign new data to regions (Cholesky Mahalanobis; no matrix inverses)
    # ---------------------------------------------
    def update_regions_with_new_data(self, X_new, y_new):
        if not self.regions:
            return
        active = [r for r in self.regions if safe_region_property(r, "is_active", True)]
        if not active:
            return
        try:
            n_dims = safe_region_property(active[0], "n_dims", 2)
            centers = np.ascontiguousarray([r.center for r in active], dtype=np.float64)

            # get each region’s Cholesky from the TR (much faster than recomputing)
            L_stack = []
            eye = np.eye(n_dims)
            for r in active:
                try:
                    L = r.chol() if hasattr(r, "chol") else np.linalg.cholesky(r.cov + 1e-12 * eye)
                except np.linalg.LinAlgError:
                    C = _eigen_floor_cov(getattr(r, "cov", (r.radius**2) * eye), floor=1e-9)
                    L = np.linalg.cholesky(C + 1e-12 * eye)
                L_stack.append(L)
            L_stack = np.ascontiguousarray(np.stack(L_stack, axis=0), dtype=np.float64)

            X_new = np.ascontiguousarray(X_new, dtype=np.float64)
            diffs = X_new[:, None, :] - centers[None, :, :]  # (N, R, D)

            # Mahalanobis distances using cached chol factors
            N, R, D = diffs.shape
            mahal_sq = np.empty((N, R), dtype=np.float64)
            for j in range(R):
                # solve L^T y = (x-c)  -> y = L^{-T}(x-c)
                Y = np.linalg.solve(L_stack[j].T, diffs[:, j, :].T).T  # (N, D)
                mahal_sq[:, j] = np.einsum("nd,nd->n", Y, Y)

            # soft responsibilities
            avg_radius = float(np.mean([r.radius for r in active]))
            beta = max(3.0, 10.0 * (getattr(self.config, "init_radius", 0.1) / (avg_radius + 1e-12)))
            with np.errstate(over="ignore"):
                w = np.exp(-beta * mahal_sq)
            w /= (w.sum(axis=1, keepdims=True) + 1e-12)

            # update regions; pass predictive variance if available
            for i, (x, y) in enumerate(zip(X_new, y_new)):
                rid = int(np.argmax(w[i]))
                if w[i, rid] > 0.05:
                    s_var = None
                    if self.surrogate_manager is not None:
                        try:
                            _, std = self.surrogate_manager.predict_global_cached(x[None, :])
                            s_var = float(std[0] ** 2)
                        except Exception:
                            s_var = None
                    active[rid].update(x.astype(np.float64), float(y), surrogate_var=s_var)
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

    def _rng_uniform(self, rng, low, high, size):
        try:
            if hasattr(rng, "uniform"):
                return rng.uniform(low, high, size=size)
            return low + (high - low) * rng.rand(*size)
        except:
            return low + (high - low) * np.random.rand(*size)

    def _restart_region(self, region, bounds, rng):
        try:
            n_dims = safe_region_property(region, "n_dims", 2)
            lo, hi = bounds[:, 0], bounds[:, 1]
            existing = np.array([r.center for r in self.regions if r is not region])
            if existing.size > 0:
                cand = self._maximin_diverse(bounds, n_dims, rng, existing)
            else:
                cand = self._rng_uniform(rng, lo, hi, size=(n_dims,))

            radius = self._compute_spawn_radius(
                cand, existing if existing.size > 0 else np.empty((0, n_dims)),
                getattr(self, "_global_X", None) if hasattr(self, "_global_X") else None
            )

            # reset core stats
            region.center = cand.astype(float)
            region.radius = float(np.clip(radius, getattr(self.config, "min_radius", 0.01),
                                          getattr(self.config, "init_radius", 0.1)))
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

            # archive reset — prefer ring buffer API when available
            if hasattr(region, "clear_archive"):
                region.clear_archive()
            else:
                lx = getattr(region, "local_X", None)
                # ring-buffer fallback
                if isinstance(lx, np.ndarray) and hasattr(region, "_buf_size"):
                    region.local_X[:] = 0.0
                    region.local_y[:] = np.inf
                    region._buf_size = 0
                    region._buf_ptr  = 0
                else:
                    # legacy list/tuple fallback
                    region.local_X = []
                    region.local_y = []

            # covariance & PCA
            r2 = region.radius**2
            region.cov = np.eye(n_dims) * r2

            region.cov_updates_since_reset = 0

            if self.verbose:
                print(f"[RESTART] Region#{region.region_id} → center={np.round(region.center, 3)}, r={region.radius:.3f}")
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Restart region failed: {e}")

    def _replace_dead_region(self, victim, bounds, rng):
        try:
            n_dims = safe_region_property(victim, "n_dims", 2)
            try:
                cand = self._find_best_spawn(bounds, n_dims, rng, getattr(self, "_global_X", []),
                                             getattr(self, "_global_y", []))
            except Exception:
                existing = np.array([r.center for r in self.regions if r is not victim])
                if existing.size > 0:
                    cand = self._maximin_diverse(bounds, n_dims, rng, existing)
                else:
                    lo, hi = bounds[:, 0], bounds[:, 1]
                    cand = self._rng_uniform(rng, lo, hi, size=(n_dims,))

            existing = np.array([r.center for r in self.regions if r is not victim])
            radius = self._compute_spawn_radius(
                cand, existing if existing.size > 0 else np.empty((0, n_dims)),
                getattr(self, "_global_X", None) if hasattr(self, "_global_X") else None
            )

            # transplant into victim
            victim.center = cand.astype(float)
            victim.radius = float(np.clip(radius, getattr(self.config, "min_radius", 0.01),
                                          getattr(self.config, "init_radius", 0.1)))
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

            # archive reset — prefer ring buffer API when available
            if hasattr(victim, "clear_archive"):
                victim.clear_archive()
            else:
                lx = getattr(victim, "local_X", None)
                # ring-buffer fallback
                if isinstance(lx, np.ndarray) and hasattr(victim, "_buf_size"):
                    victim.local_X[:] = 0.0
                    victim.local_y[:] = np.inf
                    victim._buf_size = 0
                    victim._buf_ptr  = 0
                else:
                    # legacy list/tuple fallback
                    victim.local_X = []
                    victim.local_y = []


            r2 = victim.radius**2
            victim.cov = np.eye(n_dims) * r2
            victim.pca_basis = np.eye(n_dims)
            victim.pca_eigvals = np.ones(n_dims) * r2
            victim.cov_updates_since_reset = 0

            if self.verbose:
                print(f"[REPLACE] Region#{victim.region_id} → center={np.round(victim.center, 3)}, r={victim.radius:.3f}")
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Replace dead region failed: {e}")
