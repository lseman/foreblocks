# region_manager.py — cleaner, safer, SOTA-ish RegionManager
# --- add inside TrustRegion ---
from __future__ import annotations

import collections
import math
import warnings
from dataclasses import dataclass

# ============================================
# ✅ Core Python & Concurrency
# ============================================
from typing import Dict, List, Optional, Tuple

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
from scipy.spatial.distance import cdist  # if you already have this path, keep it

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
    HAS_FAST_KD = True
except Exception:
    HAS_FAST_KD = False



# ==============================
# Helpers (numerics & ring buffer)
# ==============================
def _safe_log(x: float) -> float:
    return float(np.log(max(float(x), 1e-12)))

def _safe_tanh(x: float) -> float:
    return float(np.tanh(np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)))

def _all_finite(a: np.ndarray) -> bool:
    try:
        a = np.asarray(a, dtype=float)
        return np.isfinite(a).all()
    except Exception:
        return False

def _to_scalar(x):
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


def _chol_rank1_update(L: np.ndarray, x: np.ndarray, sign: float = +1.0) -> np.ndarray:
    """
    In-place Cholesky rank-1 update/downdate: given C = L L^T, produce
    chol(C + sign * x x^T). Requires resulting matrix to remain SPD for +1,
    or PSD with proper floor added by caller. Returns updated L.
    """
    # Implementation adapted from classic LINPACK-style algorithm.
    x = x.astype(L.dtype, copy=False).reshape(-1)
    L = L.copy()
    n = L.shape[0]
    for k in range(n):
        r = math.hypot(L[k, k], math.sqrt(sign) * x[k]) if sign > 0 else math.hypot(L[k, k], x[k])
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            L[k+1:, k] = (L[k+1:, k] + sign * s * x[k+1:]) / c
            x[k+1:] = c * x[k+1:] - s * L[k+1:, k]
    return L


# ==========================================
# Config (kept your defaults; grouped logically)
# ==========================================
@dataclass
class TRConfig:
    # radius bounds & adaptation
    min_radius: float = 1e-3
    max_radius: float = 1.0
    local_radius_adaptation: bool = True
    expansion_factor: float = 1.08
    contraction_factor: float = 0.92
    radius_step_clip: Tuple[float, float] = (0.85, 1.15)

    # covariance & conditioning
    cov_alpha: float = 0.12            # EMA weight for rank-1 updates
    cov_floor: float = 1e-9            # diagonal jitter
    cov_cap: float = 1e9               # eigen cap
    kappa_max: float = 1e6             # condition number cap
    chol_fast_updates: bool = True     # try rank-1 update before full rebuild
    chol_rebuild_every: int = 32       # periodic rebuild to avoid drift

    # local archive (ring buffer)
    max_local_data: int = 128

    # restart / aging
    max_age: int = 50
    restart_entropy_thresh: float = 0.05
    restart_radius_thresh: float = 0.05
    restart_stagnation_steps: int = 15

    # scores (weights)
    novelty_weight: float = 0.2
    entropy_weight: float = 0.2
    uncertainty_weight: float = 0.2
    velocity_weight: float = 0.3
    success_weight: float = 0.3
    age_penalty_weight: float = 0.3

    # health
    health_decay_init: float = 1.0
    health_success_w: float = 0.4
    health_entropy_w: float = 0.3
    health_uncertainty_w: float = 0.3

    # dtype
    dtype: type = np.float64

    # ---- burst / catastrophic knobs ----
    burst_on_fail: bool = True
    burst_min_failures: int = 6
    burst_factor: float = 1.35
    burst_entropy_thresh: float = 0.10
    burst_uncertainty_thresh: float = 0.75
    burst_cooldown: int = 10
    catastrophic_expand_factor: float = 1.50
    catastrophic_reset_cov: bool = True

    # ridge direction memory (whitened space)
    dir_beta: float = 0.35
    dir_cap: float = 3.0


# ==========================
# TrustRegion (clean version)
# ==========================
class TrustRegion:
    """
    Ellipsoidal Trust Region (CMA-lite) for TuRBO-style BO.

    SOTA-ish bits:
      - Trace-tied covariance (tr(C) ≈ r^2 * n), Cholesky tracked
      - Fast rank-1 Cholesky update with periodic full rebuild & eigen repair
      - Explicit condition number control + jitter floor
      - Entropy proxy from log|C| vs isotropic reference
      - Novelty from mean Mahalanobis distance (archive vs center)
      - Log-radius schedule with burst & catastrophic handling
      - Ring-buffer archive (O(1) memory), constant-time push
      - Ridge-direction memory (for sampler elongation in whitened space)
    """

    def __init__(self, center, radius, region_id: int, n_dims: int, **kwargs):
        # config
        self.cfg = TRConfig(**kwargs) if kwargs else TRConfig()
        self.n = int(n_dims)
        self.center = np.asarray(center, dtype=self.cfg.dtype).reshape(-1)
        assert self.center.shape[0] == self.n
        self.radius = float(np.clip(radius, self.cfg.min_radius, self.cfg.max_radius))
        self._log_radius = _safe_log(self.radius)

        # id & perf
        self.region_id = int(region_id)
        self.best_value = math.inf
        self.prev_best_value = math.inf
        self.trial_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.last_improvement = 0
        self.stagnation_count = 0
        self.restarts = 0

        # velocities / signals
        self.stagnation_velocity = 0.0
        self.improvement_velocity = 1.0
        self._unc_ema = 1.0
        self.local_uncertainty = 1.0
        self.local_entropy = 1.0

        # covariance & cholesky
        self.cov = np.eye(self.n, dtype=self.cfg.dtype) * (self.radius ** 2)
        self._chol = np.linalg.cholesky(self.cov + self.cfg.cov_floor * np.eye(self.n, dtype=self.cfg.dtype))
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        self._cov_updates = 0

        # archive (ring)
        m = self.cfg.max_local_data
        self.local_X = np.zeros((m, self.n), dtype=self.cfg.dtype)
        self.local_y = np.full((m,), math.inf, dtype=self.cfg.dtype)
        self._buf_size = 0
        self._buf_ptr = 0

        # internal counters
        self._trial_index = 0
        self._last_burst_trial = -10

        # ridge direction memory (in whitened coords)
        self.prev_dir_white = np.zeros(self.n, dtype=self.cfg.dtype)
        self.radius_long = self.radius
        self.radius_lat  = self.radius
        
        self.n_dims = n_dims
        self.spawn_score = 0.0
        self.exploration_bonus = 1.0
    # ---------------- core update ----------------
    def update(self, x, y, surrogate_var: Optional[float] = None) -> Dict[str, float]:
        self.trial_count += 1
        self._trial_index += 1

        x = np.asarray(x, dtype=self.cfg.dtype).reshape(-1)
        y_is_finite = np.isfinite(y)
        y_val = float(y) if y_is_finite else np.inf

        improved = (y_val < self.best_value)
        delta = max(0.0, float(self.best_value - y_val)) if np.isfinite(self.best_value) else 0.0

        # velocities
        self.stagnation_velocity = 0.9 * self.stagnation_velocity + 0.1 * delta
        self.improvement_velocity = (0.8 * self.improvement_velocity + 0.2 * delta) if improved else (self.improvement_velocity * 0.97)

        # uncertainty EMA
        sv = _to_scalar(surrogate_var)
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
        self._cov_rank1_ema_update(x)

        # entropy proxy
        self.local_entropy = self._entropy_from_chol()

        # radius schedule (+ burst/catastrophic)
        if self.cfg.local_radius_adaptation:
            self._adaptive_radius(improved, delta, catastrophic_fail=(not y_is_finite))

        # (optional) downstream spawn/bonus scores could be computed here
        self._update_scores()

        return {
            "success_rate": self.success_rate,
            "entropy": self.local_entropy,
            "uncertainty": self.local_uncertainty,
            "stagn_vel": self.stagnation_velocity,
            "impr_vel": self.improvement_velocity,
            "radius": self.radius,
            "spawn_score": self.spawn_score,               # <—
            "exploration_bonus": self.exploration_bonus,   # <—
        }

    def _update_scores(self):
        """
        Compute lightweight downstream scores:
          - spawn_score   ∈ [0,1]: higher = more deserving to spawn/expand
          - exploration_bonus ∈ [0.8,2.0]: scalar to bias samplers toward exploration
        Uses Mahalanobis novelty, entropy/uncertainty proxies, success & velocity.
        """
        # Age penalty grows with time since last improvement
        age_penalty = min(1.0, self.last_improvement / max(1, self.cfg.max_age))

        # Signals normalized with tanh to keep bounded and robust
        novelty_term     = _safe_tanh(self._mean_mahalanobis())
        entropy_term     = _safe_tanh(self.local_entropy)
        uncertainty_term = _safe_tanh(self.local_uncertainty)
        velocity_term    = _safe_tanh(self.stagnation_velocity)

        score = (
            self.cfg.success_weight      * self.success_rate +
            self.cfg.velocity_weight     * 0.5 * (velocity_term + self.cfg.novelty_weight * novelty_term) +
            self.cfg.entropy_weight      * entropy_term +
            self.cfg.uncertainty_weight  * uncertainty_term -
            self.cfg.age_penalty_weight  * age_penalty
        )

        # Clamp to sane ranges
        self.spawn_score = float(np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0))

        # Encourage exploration when improvement velocity is low
        self.exploration_bonus = float(
            np.clip(1.0 + 0.5 * (1.0 - _safe_tanh(self.improvement_velocity)), 0.8, 2.0)
        )
        
    # ---------------- covariance & Cholesky ----------------
    def _cov_rank1_ema_update(self, x: np.ndarray):
        """
        EMA over outer products centered at 'center': C ← (1-α)C + α * (dx dx^T)
        Track Cholesky with rank-1 update when possible; repair if needed.
        """
        alpha = self.cfg.cov_alpha
        floor = self.cfg.cov_floor
        dx = (x - self.center).reshape(-1, 1)

        if not _all_finite(dx):
            # just safeguard with jitter
            self.cov += floor * np.eye(self.n, dtype=self.cfg.dtype)
            self._rebuild_from_cov()  # keep chol in sync
            self._align_cov_to_radius()
            return

        # EMA on covariance
        C_new = (1.0 - alpha) * self.cov + alpha * (dx @ dx.T)
        C_new = 0.5 * (C_new + C_new.T)  # ensure symmetry

        # Try fast rank-1 update on Cholesky (optional)
        fast_ok = False
        if self.cfg.chol_fast_updates and self._cov_updates % self.cfg.chol_rebuild_every != 0:
            try:
                # We want chol(C_new). Since C_new = (1-α)C + α xx^T,
                # use: scale L by sqrt(1-α) then rank-1 update with sqrt(α) x.
                L = self._chol * math.sqrt(max(1.0 - alpha, 1e-12))
                L = _chol_rank1_update(L, (math.sqrt(alpha) * dx.reshape(-1)), sign=+1.0)
                self._chol = L
                self.cov = C_new
                self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
                fast_ok = True
            except Exception:
                fast_ok = False

        if not fast_ok:
            self.cov = C_new
            self._ensure_spd_and_rebuild()

        self._align_cov_to_radius()
        self._cov_updates += 1

    def _ensure_spd_and_rebuild(self):
        floor = self.cfg.cov_floor
        C = 0.5 * (self.cov + self.cov.T) + floor * np.eye(self.n, dtype=self.cfg.dtype)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # eigen repair
            w, V = np.linalg.eigh(C)
            w = np.clip(w, floor, self.cfg.cov_cap)
            # cap condition number
            w_max = float(np.max(w))
            w_min = max(float(np.min(w)), floor)
            if w_max / max(w_min, floor) > self.cfg.kappa_max:
                w_min_new = max(w_max / self.cfg.kappa_max, floor)
                w = np.maximum(w, w_min_new)
            C = (V * w) @ V.T
            L = np.linalg.cholesky(C)
        self.cov = C
        self._chol = L
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(L))))

    def _rebuild_from_cov(self):
        C = 0.5 * (self.cov + self.cov.T) + self.cfg.cov_floor * np.eye(self.n, dtype=self.cfg.dtype)
        L = np.linalg.cholesky(C)
        self._chol = L
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(L))))

    def _align_cov_to_radius(self):
        # tie tr(C) to r^2 * n
        target_tr = float((self.radius ** 2) * self.n)
        tr = float(np.trace(self.cov))
        if np.isfinite(tr) and tr > 0.0:
            s = target_tr / tr
            self.cov *= s
            self._chol *= math.sqrt(max(s, 1e-16))  # keep Cholesky consistent
            self._logdet += self.n * math.log(max(s, 1e-16))

    def _entropy_from_chol(self) -> float:
        # entropy proxy: 0.5 (log|C| - log|r^2 I|)
        ref_logdet = self.n * _safe_log(self.radius ** 2)
        val = 0.5 * (self._logdet - ref_logdet)
        return float(np.clip(val, -50.0, 50.0))

    # ---------------- radius schedule ----------------
    def _adaptive_radius(self, improved: bool, delta: float, catastrophic_fail: bool = False):
        if catastrophic_fail:
            self.radius = float(np.clip(self.radius * self.cfg.catastrophic_expand_factor,
                                        self.cfg.min_radius, self.cfg.max_radius))
            if self.cfg.catastrophic_reset_cov:
                self._reset_cov_isotropic()
            self._align_cov_to_radius()
            return

        base = self.cfg.expansion_factor if improved else (
            self.cfg.contraction_factor * math.exp(-0.5 * self.stagnation_velocity)
        )
        scale = base * (1.0 + 0.30 * _safe_tanh(self.local_uncertainty)) * (1.0 + 0.15 * _safe_tanh(delta))

        # burst on repeated fails (entropy low or uncertainty high)
        burst_fired = False
        if (self.cfg.burst_on_fail and not improved and
            self.consecutive_failures >= self.cfg.burst_min_failures and
            (self._trial_index - self._last_burst_trial) >= self.cfg.burst_cooldown):
            stuck  = (self.local_entropy < self.cfg.burst_entropy_thresh)
            unsure = (_safe_tanh(self.local_uncertainty) > self.cfg.burst_uncertainty_thresh)
            if stuck or unsure:
                scale = max(scale, self.cfg.burst_factor)
                self._last_burst_trial = self._trial_index
                burst_fired = True
                # mild isotropization to escape narrow ill-conditioned ellipses
                self.cov = 0.8 * self.cov + 0.2 * (np.eye(self.n, dtype=self.cfg.dtype) * (self.radius ** 2))
                self._ensure_spd_and_rebuild()

        lo, hi = self.cfg.radius_step_clip
        if burst_fired:
            hi = max(hi, self.cfg.burst_factor)
        scale = float(np.clip(scale, lo, hi))

        self._log_radius = _safe_log(self.radius) + _safe_log(scale)
        self.radius = float(np.clip(math.exp(self._log_radius), self.cfg.min_radius, self.cfg.max_radius))
        self._align_cov_to_radius()

    def _reset_cov_isotropic(self):
        self.cov[:] = 0.0
        diag = (self.radius ** 2)
        for i in range(self.n):
            self.cov[i, i] = diag
        self._rebuild_from_cov()

    # ---------------- archive / novelty ----------------
    def _archive_push(self, x: np.ndarray, y: float):
        m = self.cfg.max_local_data
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
        # solve L z = dx^T  → z = L^{-1} dx^T
        z = np.linalg.solve(L, dx.T)
        d2 = np.sum(z * z, axis=0)
        return float(np.mean(np.sqrt(np.maximum(d2, 0.0))))

    # ---------------- public: metric to sampler ----------------
    def metric_scales(self) -> np.ndarray:
        """
        Per-dimension Mahalanobis weights w_j ≈ 1 / ell_j^2 using diagonal(C)^-1.
        Cheap & robust; use with diversity. For exact distances, pass chol().
        """
        diag = np.clip(np.diag(self.cov), self.cfg.cov_floor, self.cfg.cov_cap)
        return (1.0 / np.maximum(diag, self.cfg.cov_floor)).astype(self.cfg.dtype)

    def chol(self) -> np.ndarray:
        """Current Cholesky factor (lower triangular)."""
        return self._chol.copy()

    # ---------------- ridge direction memory ----------------
    def update_direction(self, step_white: np.ndarray):
        g = np.asarray(step_white, dtype=self.cfg.dtype).reshape(-1)
        nrm = float(np.linalg.norm(g))
        if nrm < 1e-12:
            return
        u = g / nrm
        beta = self.cfg.dir_beta
        self.prev_dir_white = beta * u + (1.0 - beta) * self.prev_dir_white
        nu = float(np.linalg.norm(self.prev_dir_white))
        if nu > 1e-12:
            self.prev_dir_white /= nu
        # elongate along ridge (sampler can read radius_long/lat)
        self.radius_long = min(self.radius_long * 1.6, self.cfg.dir_cap * self.radius)
        self.radius_lat  = max(self.radius_lat  / math.sqrt(1.6), 0.25 * self.radius)

    def decay_direction(self):
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
        entropy_low = bool(self.local_entropy < self.cfg.restart_entropy_thresh)
        small = (self.radius < self.cfg.restart_radius_thresh)
        stuck = (self.stagnation_count > self.cfg.restart_stagnation_steps)
        return (stuck or entropy_low) and small

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
# RegionManager (TuRBO-M+++): cleaned, faster, API-compatible
# -----------------------------------------------------------
# Assumes the following helpers/types exist in your codebase:
# - safe_region_property, np_safe, exp_moving_avg, RunningStats, _eigen_floor_cov
# - compute_coverage_fraction, compute_mean_entropy_from_global_gp, _min_dist_to_set
# - TrustRegion, RLConfig, RadiusLearnerAgent, RegionAttention, sigmoid
# - i4_sobol_generate (optional), KDTree/HAS_KDTREE (optional), TORCH_AVAILABLE (bool)
# - self.surrogate_manager.{global_backend, global_model, gradient_global_mean,
#   predict_global_cached} (optional, guarded)
#
# Optional deps used if available:
# - scipy.spatial.distance.cdist
# - scipy.stats.norm
# - scipy.stats.qmc.Sobol
#
# This is a drop-in replacement for your class definition.


# optional imports (guarded)
try:
    from scipy.spatial.distance import cdist
except Exception:
    cdist = None

try:
    from scipy.stats import norm as _scipy_norm
except Exception:
    _scipy_norm = None

try:
    from scipy.stats.qmc import Sobol as _Sobol
except Exception:
    _Sobol = None



# ---------- soft imports & shims ----------
try:
    from scipy.spatial.distance import cdist as _cdist
except Exception:
    _cdist = None
try:
    from scipy.spatial import KDTree as _KDTree
except Exception:
    _KDTree = None
try:
    from scipy.stats import norm as _scipy_norm
except Exception:
    _scipy_norm = None
try:
    from scipy.stats.qmc import Sobol as _Sobol
except Exception:
    _Sobol = None
try:
    from sobol_seq import i4_sobol_generate as _sobol_int
except Exception:
    _sobol_int = None

def _eigen_floor_cov(C, floor=1e-9, cap=1e9, kappa_max=1e6):
    C = 0.5 * (C + C.T)
    try:
        w, V = np.linalg.eigh(C)
        w = np.clip(w, floor, cap)
        w_max = float(np.max(w))
        w_min = max(float(np.min(w)), floor)
        if w_max / max(w_min, floor) > kappa_max:
            w_min_new = max(w_max / kappa_max, floor)
            w = np.maximum(w, w_min_new)
        return (V * w) @ V.T
    except Exception:
        d = C.shape[0]
        return np.eye(d) * max(float(np.trace(C)) / max(d, 1), floor)

def _min_dist_to_set(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B is None or B.size == 0:
        return np.ones(A.shape[0], dtype=np.float32)
    if _cdist is not None:
        D = _cdist(A, B)  # (NA, NB)
        return np.min(D, axis=1).astype(np.float32)
    # fallback
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return np.min(D, axis=1).astype(np.float32)

# ---------- RegionManager ----------
class RegionManager:
    """
    RegionManager for TuRBO-M+++ (clean + faster)
    - Archive-agnostic (list / ring-buffer ndarray / TrustRegion.local_view)
    - Covariance refresh uses ring buffer correctly and aligns trace to radius
    - Vectorized diversity; safe feature extraction with light normalization
    - Spawn path uses Sobol + EI/uncertainty/diversity blend (or your AcqManager)
    - Integrates TrustRegion.spawn_score / exploration_bonus when present
    - Neural radius/attention kept (gated, error-safe)
    """

    def __init__(self, config, verbose: bool = True):
        self.config  = config
        self.verbose = bool(verbose)
        self.regions: List = []
        self.surrogate_manager = None

        self._entropy_buffer = collections.deque(maxlen=12)
        self._ema_progress = 0.0
        self._iteration = 0
        self._feat_dim = 9

        cfg = self.config
        self._cfg = {
            "max_evals":           int(getattr(cfg, "max_evals", 1000)),
            "n_regions":           int(getattr(cfg, "n_regions", 5)),
            "init_radius":         float(getattr(cfg, "init_radius", 0.1)),
            "min_radius":          float(getattr(cfg, "min_radius", 0.01)),
            "max_radius":          float(getattr(cfg, "max_radius", 1.0)),
            "is_periodic_problem": bool(getattr(cfg, "is_periodic_problem", False)),
            "period_phases":       tuple(getattr(cfg, "period_phases", (0.0, 0.25, 0.5, 0.75))),
            "phase_jitter_str":    float(getattr(cfg, "phase_jitter_strength", 0.30)),
            "use_hv":              bool(getattr(cfg, "use_hypervolume", True)),
            "max_total_points":    getattr(cfg, "max_total_points", None),
        }

        # feature gates (optional, guarded)
        self.use_neural_radius    = bool(getattr(cfg, "use_neural_radius", False))
        self.use_neural_attention = bool(getattr(cfg, "use_neural_attention", False))
        self.neural_device        = getattr(cfg, "neural_device", "cpu")
        if self.verbose:
            print(f"[INFO] RegionManager: neural_radius={self.use_neural_radius}, "
                  f"neural_attention={self.use_neural_attention}, device={self.neural_device}")

        # optional neural radius agent
        self.radius_agent = None
        TORCH_AVAILABLE = False
        try:
            import torch as _torch  # noqa: F401
            TORCH_AVAILABLE = True
        except Exception:
            TORCH_AVAILABLE = False

        if self.use_neural_radius and TORCH_AVAILABLE:
            if self.verbose:
                print("[INFO] Initializing neural radius agent...")
            try:
                RLConfig = getattr(cfg, "RLConfig", None)
                RadiusLearnerAgent = getattr(cfg, "RadiusLearnerAgent", None)
                if RLConfig is None or RadiusLearnerAgent is None:
                    raise RuntimeError("RL components missing from config")
                rl_config = RLConfig(
                    buffer_cap=max(2048, self._cfg["max_evals"] // 10),
                    lr=3e-4,
                    train_every=max(1, self._cfg["max_evals"] // 1000),
                    batch=min(128, max(32, self._cfg["max_evals"] // 100)),
                    delta_max=0.15,
                )
                self.radius_agent = RadiusLearnerAgent(d_in=self._feat_dim, cfg=rl_config, device=self.neural_device)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Neural radius disabled: {e}")
                self.use_neural_radius = False

        # optional attention
        self.region_attn = None
        if self.use_neural_attention and TORCH_AVAILABLE:
            try:
                RegionAttention = getattr(cfg, "RegionAttention", None)
                if RegionAttention is None:
                    raise RuntimeError("RegionAttention missing from config")
                self.region_attn = RegionAttention(d_in=self._feat_dim, d_model=64, n_heads=8)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Attention disabled: {e}")
                self.use_neural_attention = False

        self._feature_normalizer = None
        self._centers_cache = None
        self.use_hypervolume = self._cfg["use_hv"]

        # spawn acquisition manager (optional, separate policy)
        self.spawn_acquisition_manager = None
        self._spawn_history = collections.deque(maxlen=20)

    # ---- External: attach a separate acquisition manager just for spawning ---
    def set_acquisition_candidate_manager(self, acq_manager):
        self.spawn_acquisition_manager = acq_manager
        # derive a lean spawn-only config (inherits user overrides if present)
        spawn_config = type('SpawnConfig', (), {})()
        for attr, default in [
            ('batch_size', 1),
            ('max_evals', self._cfg['max_evals']),
            ('is_periodic_problem', self._cfg['is_periodic_problem']),
            ('period_phases', self._cfg['period_phases']),
            ('phase_jitter_strength', self._cfg['phase_jitter_str']),
        ]:
            setattr(spawn_config, attr, getattr(self.config, f'spawn_{attr}', default))
        self.spawn_acquisition_manager.config = spawn_config
        self.spawn_acquisition_manager.thompson_probability = getattr(self.config, 'spawn_thompson_prob', 0.6)

    # --------------- archive helpers ---------------
    @staticmethod
    def _local_arrays(region):
        if hasattr(region, "local_view"):
            Xl, yl = region.local_view()
            k = Xl.shape[0]
            if k <= 0:
                return None, None, 0
            return (np.ascontiguousarray(Xl, dtype=float),
                    np.ascontiguousarray(yl, dtype=float),
                    int(k))
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
                    int(k))
        try:
            k = len(lx)
            if k <= 0:
                return None, None, 0
            return (np.ascontiguousarray(lx, dtype=float),
                    np.ascontiguousarray(ly, dtype=float),
                    int(k))
        except Exception:
            return None, None, 0

    @staticmethod
    def _len_local(region):
        if hasattr(region, "local_len"):
            return int(region.local_len())
        _, _, k = RegionManager._local_arrays(region)
        return int(k)

    # ----------------------------- Feature extraction -------------------------
    def _region_features(self, r) -> np.ndarray:
        init_r  = self._cfg["init_radius"]
        n_dims  = max(1, safe_region_property(r, "n_dims", 2))
        stagn_n = max(10.0, 2.0 * n_dims)
        radius  = float(max(safe_region_property(r, "radius", init_r), 1e-12))
        it_ratio = self._iteration / (self._cfg["max_evals"] + 1e-9)

        raw = np.array([
            it_ratio,
            np_safe(safe_region_property(r, "success_rate", 0.0)),
            np_safe(safe_region_property(r, "stagnation_count", 0)) / stagn_n,
            np_safe(safe_region_property(r, "local_entropy", 0.5)),
            np_safe(safe_region_property(r, "local_uncertainty", 1.0)),
            np_safe(safe_region_property(r, "improvement_velocity", 0.0)),
            radius / (init_r + 1e-12),
            np.log(radius) - np.log(init_r + 1e-12),
            self._region_diversity_score(r) if self.regions else 1.0,
        ], dtype=np.float32)

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
                print(f"[WARN] NaN features for region {rid} → zeros")
            z = np.zeros_like(z, dtype=np.float32)
        return z

    # ----------------------------- Reward (for RL radius) ---------------------
    def _compute_adaptive_reward(self, region, delta, old_radius):
        prev_best = np_safe(safe_region_property(region, "prev_best_value", np.inf))
        current_best = np_safe(safe_region_property(region, "best_value", np.inf))
        delta_impr = 0.0 if not (np.isfinite(prev_best) and np.isfinite(current_best)) else max(0.0, prev_best - current_best)
        scale = 1.0
        Xl, yl, k = self._local_arrays(region)
        if k > 5:
            local_std = float(np.std(yl))
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
        except Exception:
            diversity_bonus = 0.0
        raw_reward = (math.tanh(improvement_reward) + exploration_bonus + diversity_bonus - 0.1 * stability_penalty)
        return float(np.clip(np_safe(raw_reward), -10.0, 10.0))

    # ----------------------------- Diversity score ----------------------------
    def _region_diversity_score(self, region):
        if len(self.regions) <= 1:
            return 1.0
        try:
            centers = []
            for r in self.regions:
                if r is region:
                    continue
                c = getattr(r, "center", None)
                if c is None or not np.all(np.isfinite(c)):
                    continue
                centers.append(np.asarray(c, dtype=float))
            if not centers:
                return 1.0
            centers = np.stack(centers, axis=0)
            rc = np.asarray(region.center, dtype=float)
            try:
                if hasattr(region, "chol"):
                    L = region.chol()
                else:
                    C = getattr(region, "cov", None)
                    if C is None:
                        raise ValueError("cov missing")
                    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(rc)))
                diffs = centers - rc[None, :]
                Y = np.linalg.solve(L.T, diffs.T).T
                d = np.sqrt(np.sum(Y * Y, axis=1))
            except Exception:
                d = np.linalg.norm(centers - rc[None, :], axis=1)
            dmin = float(np.min(d)) if d.size else 0.0
            radius = max(safe_region_property(region, "radius", self._cfg["init_radius"]), 1e-9)
            return np_safe(dmin / radius, default=1.0)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Diversity score error: {e}")
            return 1.0

    # --------------------------------------------- Compatibility --------------
    def set_surrogate_manager(self, surrogate_manager):
        self.surrogate_manager = surrogate_manager

    # --------------------------------------------- Initialization -------------
    def initialize_regions(self, X, y, n_dims, rng=None):
        rng = np.random.default_rng(None if rng is None else rng)
        n_init = min(self._cfg["n_regions"], len(X))
        best_idx = int(np.argmin(y))
        selected_idx = [best_idx]

        # diversify centers by farthest-with-performance heuristic
        if _cdist is not None:
            D = _cdist(X, X)
        else:
            diff = X[:, None, :] - X[None, :, :]
            D = np.linalg.norm(diff, axis=2)
        perf_w = np.exp(-0.05 * (y - np.min(y)))
        for _ in range(1, n_init):
            min_d = np.min(D[:, selected_idx], axis=1)
            s = perf_w * (min_d + 1e-9)
            s[selected_idx] = -np.inf
            selected_idx.append(int(np.argmax(s)))

        for rid, center in enumerate(X[selected_idx]):
            norms = np.linalg.norm(X - center, axis=1)
            base_r = np.percentile(norms, 25)
            base_r = float(np.clip(base_r, self._cfg["min_radius"], self._cfg["init_radius"]))
            self.regions.append(TrustRegion(center, base_r, rid, n_dims))
        self._refresh_centers_cache()
        if self.verbose:
            print(f"[INIT] {len(self.regions)} trust regions created")

    # --------------------------------------------- Main lifecycle -------------
    def manage_regions(self, bounds, n_dims, rng, global_X, global_y, iteration=0):
        self._global_X = np.asarray(global_X) if global_X is not None else None
        self._global_y = np.asarray(global_y) if global_y is not None else None
        self._iteration = int(iteration)
        self._ema_progress = exp_moving_avg(
            self._ema_progress,
            iteration / (self._cfg["max_evals"] + 1e-12),
            alpha=0.1,
        )
        self._adapt_all(bounds, rng)
        self._maybe_spawn(bounds, n_dims, rng, global_X, global_y)
        self._ensure_min_diversity(bounds, n_dims, rng, global_X, global_y)
        self._adaptive_prune()
        self._refresh_centers_cache()
        # diagnostics (optional)
        return {
            "n_regions": len(self.regions),
            "avg_radius": float(np.mean([r.radius for r in self.regions])) if self.regions else 0.0,
            "avg_health": float(np.mean([np_safe(safe_region_property(r, "health_score", 0.5)) for r in self.regions])) if self.regions else 0.0,
        }

    # --------------------------------------------- Adaptation -----------------
    def _adapt_all(self, bounds, rng):
        dead = []
        for r in list(self.regions):
            try:
                current_vel = np_safe(safe_region_property(r, "improvement_velocity", 0.0))
                r.vel_ema = exp_moving_avg(np_safe(safe_region_property(r, "vel_ema", 0.0)), current_vel, alpha=0.3) or 0.0

                # Covariance refresh (archive-driven only when collapsed)
                Xl, yl, k = self._local_arrays(r)
                if k > max(8, safe_region_property(r, "n_dims", 2)):
                    ent = np_safe(safe_region_property(r, "local_entropy", 0.5))
                    if ent < 0.10:
                        try:
                            centered = Xl - r.center
                            S = np.cov(centered.T) + 1e-9 * np.eye(r.n_dims)
                            S = _eigen_floor_cov(S, floor=1e-9)
                            mix = 0.15 if k > 20 else 0.07
                            C_old = getattr(r, "cov", np.eye(r.n_dims) * (r.radius**2))
                            C_new = (1 - mix) * C_old + mix * S
                            if hasattr(r, "_ensure_spd_and_rebuild"):
                                r.cov = C_new
                                r._ensure_spd_and_rebuild()
                                if hasattr(r, "_align_cov_to_radius"):
                                    r._align_cov_to_radius()
                            else:
                                r.cov = C_new
                        except Exception as e:
                            if self.verbose:
                                print(f"[WARN] Cov refresh R#{r.region_id}: {e}")
                            r.cov = np.eye(r.n_dims) * (r.radius**2)

                old_r = float(np_safe(r.radius, default=self._cfg["init_radius"]))

                # Neural vs heuristic radius
                new_radius, delta = None, 0.0
                if self.use_neural_radius and (self.radius_agent is not None):
                    try:
                        feats = self._region_features(r)
                        if np.all(np.isfinite(feats)):
                            new_radius, delta = self.radius_agent.step(
                                feats, old_r, self._cfg["min_radius"], self._cfg["max_radius"]
                            )
                    except Exception as e:
                        if self.verbose:
                            print(f"[WARN] Learned adaptation failed R{r.region_id}: {e}")

                if new_radius is None:
                    v = float(np_safe(getattr(r, "vel_ema", 0.0), default=0.0))
                    if v < 0.01:
                        new_radius = min(old_r * 1.05, self._cfg["max_radius"])
                    elif v > 0.1:
                        new_radius = max(old_r * 0.92, self._cfg["min_radius"])
                    else:
                        new_radius = float(np.clip(old_r * 0.995, self._cfg["min_radius"], self._cfg["max_radius"]))
                    delta = math.log(max(new_radius, 1e-12) / max(old_r, 1e-12))

                if not np.isfinite(new_radius) or new_radius <= 0:
                    new_radius = max(old_r, self._cfg["min_radius"])
                    delta = 0.0

                r.radius = float(new_radius)

                # Tie covariance trace to new radius
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

                # Health score (simple, robust)
                try:
                    div = self._region_diversity_score(r)
                    ent = np_safe(safe_region_property(r, "local_entropy", 0.5))
                    last_impr = max(0, safe_region_property(r, "last_improvement", 0))
                    n_dims = max(1, safe_region_property(r, "n_dims", 2))
                    age_penalty = min(0.5, last_impr / max(20, n_dims * 3))
                    raw_h = 0.4 * np_safe(r.vel_ema) + 0.3 * (1.0 - ent) + 0.2 * div - 0.1 * age_penalty
                    r.health_score = float(np.clip(np_safe(raw_h), 0.0, 1.0))
                    r.health_decay_factor = max(0.1, np_safe(safe_region_property(r, "health_decay_factor", 1.0)) * 0.995)
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Health update R{r.region_id}: {e}")
                    r.health_score = 0.5

                # Restart / death rules
                should_restart = bool(safe_region_property(r, "should_restart", False))
                health_score = float(np_safe(safe_region_property(r, "health_score", 0.5)))
                if should_restart or (r.radius < 2.0 * self._cfg["min_radius"] and health_score < 0.15):
                    try:
                        self._restart_region(r, bounds, rng)
                    except Exception as e:
                        if self.verbose:
                            print(f"[WARN] Restart failed R{r.region_id}: {e}")
                if (r.radius < 1.5 * self._cfg["min_radius"]) and (health_score < 0.1):
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
                            print(f"[WARN] Reward push failed R{r.region_id}: {e}")

            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Region {getattr(r, 'region_id', '?')} adaptation failed: {e}")
                dead.append(r)

        if self.use_neural_radius and (self.radius_agent is not None):
            try:
                self.radius_agent.maybe_train()
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Training step failed: {e}")

        for r in dead:
            try:
                self._replace_dead_region(r, bounds, rng)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Replace dead R{r.region_id} failed: {e}")
                self.regions = [rr for rr in self.regions if rr is not r]

    # --------------------------------------------- Spawning -------------------
    def _maybe_spawn(self, bounds, n_dims, rng, global_X, global_y):
        coverage = compute_coverage_fraction(global_X, self.regions)
        entropy = compute_mean_entropy_from_global_gp(self.surrogate_manager, global_X)
        self._entropy_buffer.append(np_safe(entropy))
        sm_entropy = float(np.mean(self._entropy_buffer)) if self._entropy_buffer else 0.5

        # use TrustRegion signals if available
        avg_health = np.mean([np_safe(safe_region_property(r, "health_score", 0.5)) for r in self.regions]) if self.regions else 1.0
        avg_spawn = np.mean([np_safe(safe_region_property(r, "spawn_score", 0.5)) for r in self.regions]) if self.regions else 0.5

        exploration_pressure = (1.0 - coverage) + sm_entropy + 0.25 * (1.0 - avg_spawn)
        exploitation_pressure = self._ema_progress + 0.5 * avg_health

        try:
            s = sigmoid(3.0 * (exploration_pressure - exploitation_pressure))
        except Exception:
            s = 0.0
        trigger = s > 0.45

        # dynamic cap — allow more regions if exploration pressure high
        dynamic_cap = int(self._cfg["n_regions"] * (1.0 + 0.5 * exploration_pressure))
        if trigger and len(self.regions) < dynamic_cap:
            self._force_spawn(bounds, n_dims, rng, global_X, global_y)


    def _compute_spawn_radius(self, candidate, existing, global_X, k=8):
        if existing is None or existing.size == 0:
            return self._cfg["init_radius"]
        d_near = float(np.median(np.linalg.norm(existing - candidate, axis=1)))
        if global_X is not None and len(global_X) > 0:
            if _KDTree is not None and len(global_X) >= k:
                try:
                    tree = _KDTree(global_X)
                    d_k, _ = tree.query(candidate, k=min(k, len(global_X)))
                    d_local = float(np.median(np.atleast_1d(d_k)))
                except Exception:
                    d_local = float(np.median(np.linalg.norm(global_X - candidate, axis=1)))
            else:
                d_local = float(np.median(np.linalg.norm(global_X - candidate, axis=1)))
        else:
            d_local = d_near
        r = float(min(d_near, d_local))
        return float(np.clip(r, self._cfg["min_radius"], self._cfg["init_radius"]))

    def _find_best_spawn(self, bounds, n_dims, rng, global_X, global_y):
        try:
            if self.spawn_acquisition_manager is None:
                raise RuntimeError("No spawn acquisition manager set")

            # pass context (use region spawn stagnation if tracked)
            if hasattr(self.spawn_acquisition_manager, "set_context"):
                self.spawn_acquisition_manager.set_context(
                    iteration=self._iteration,
                    stagnation_counter=int(np.mean([np_safe(getattr(r, "stagnation_count", 0)) for r in self.regions])) if self.regions else 0
                )

            spawn_regions = self._generate_spawn_region_candidates(bounds, n_dims, rng, global_X, global_y)
            if not spawn_regions:
                raise RuntimeError("No spawn regions generated")
            
            best_candidates, best_scores = [], []
            for spawn_region in spawn_regions:
                try:
                    candidates = self.spawn_acquisition_manager.generate_candidates(
                        bounds, rng, [spawn_region], self.surrogate_manager
                    )
                    if len(candidates) > 0:
                        cand = candidates[0]
                        score = self._score_spawn_candidate(cand, spawn_region, global_X, global_y)
                        # boost with TrustRegion signals if present
                        score *= float(np.clip(np_safe(getattr(spawn_region, "spawn_score", 1.0)) *
                                               np_safe(getattr(spawn_region, "exploration_bonus", 1.0)), 0.8, 2.0))
                        best_candidates.append(cand)
                        best_scores.append(score)
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Spawn acq failed: {e}")
                    continue

            if not best_candidates:
                return self._find_best_spawn_fallback(bounds, n_dims, rng, global_X, global_y)

            best_idx = int(np.argmax(best_scores))
            if self.verbose:
                print(f"[SPAWN] Advanced selection: score={best_scores[best_idx]:.3f}")
            return best_candidates[best_idx]
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Advanced spawn selection failed: {e}, using fallback")
            return self._find_best_spawn_fallback(bounds, n_dims, rng, global_X, global_y)


    def _generate_spawn_region_candidates(self, bounds, n_dims, rng, global_X, global_y, n_candidates=5):
        spawn_regions = []
        try:
            # Strat 1: high global uncertainty (Sobol → std threshold → diverse)
            if self.surrogate_manager and global_X is not None and len(global_X) > 10:
                n_explore = int(min(200, max(50, 10 * n_dims)))
                sobol_points = self._generate_sobol_points(bounds, n_explore)
                _, std = self.surrogate_manager.predict_global_cached(sobol_points)
                if std is not None and len(std) == n_explore:
                    thr = np.percentile(std, 75)
                    high_unc_points = sobol_points[std >= thr]
                    if len(high_unc_points) > 0:
                        centers = self._select_diverse_centers(high_unc_points, min(3, len(high_unc_points)))
                        for c in centers:
                            radius = self._compute_spawn_radius(
                                c, np.array([r.center for r in self.regions]) if self.regions else np.empty((0, n_dims)),
                                global_X
                            )
                            spawn_regions.append(self._create_virtual_spawn_region(c, radius, n_dims))

            # Strat 2: extrapolate from top-health regions
            if self.regions:
                hs = np.array([safe_region_property(r, "health_score", 0.0) for r in self.regions])
                if hs.size > 0:
                    idx = np.argsort(hs)[-2:]
                    for r in (self.regions[i] for i in idx):
                        if np_safe(getattr(r, "improvement_velocity", 0.0)) > 0.05:
                            xc = self._extrapolate_region_center(r, bounds, rng)
                            if xc is not None:
                                spawn_regions.append(self._create_virtual_spawn_region(xc, r.radius * 1.2, n_dims))

            # Strat 3: near global best for refinement
            if (global_y is not None) and len(global_y) > 5:
                bidx = int(np.argmin(global_y))
                bx = global_X[bidx]
                min_dist = np.inf
                if self.regions:
                    centers = np.array([r.center for r in self.regions])
                    min_dist = float(np.min(np.linalg.norm(centers - bx, axis=1)))
                if min_dist > 2.0 * self._cfg["init_radius"]:
                    radius = self._compute_spawn_radius(
                        bx, np.array([r.center for r in self.regions]) if self.regions else np.empty((0, n_dims)),
                        global_X
                    )
                    spawn_regions.append(self._create_virtual_spawn_region(bx, radius, n_dims))
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Spawn region generation failed: {e}")

        return spawn_regions[:n_candidates]

    def _generate_sobol_points(self, bounds, n_points):
        dim = bounds.shape[0]
        # scipy Sobol if available
        if _Sobol is not None:
            try:
                eng = _Sobol(d=dim, scramble=True)
                sob = eng.random_base2(m=int(np.ceil(np.log2(max(2, n_points)))))
                sob = sob[:n_points]
                return bounds[:, 0] + sob * (bounds[:, 1] - bounds[:, 0])
            except Exception:
                pass
        # integer sobol fallback
        if _sobol_int is not None:
            try:
                sob = _sobol_int(dim, n_points)
                return bounds[:, 0] + sob * (bounds[:, 1] - bounds[:, 0])
            except Exception:
                pass
        # uniform last resort
        return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_points, dim))

    def _select_diverse_centers(self, points, k):
        points = np.asarray(points)
        if len(points) <= k:
            return points
        # greedy farthest-first
        sel = [points[0]]
        for _ in range(k - 1):
            dists = np.array([min(np.linalg.norm(p - s) for s in sel) for p in points], dtype=float)
            sel.append(points[int(np.argmax(dists))])
        return np.array(sel)

    def _create_virtual_spawn_region(self, center, radius, n_dims):
        r = type('VirtualSpawnRegion', (), {})()
        r.center = np.asarray(center, dtype=np.float64)
        r.radius = float(np.clip(radius, self._cfg["min_radius"], self._cfg["init_radius"] * 2.0))
        r.n_dims = int(n_dims)
        r.cov = np.eye(n_dims) * (r.radius ** 2)

        r.health_score = 1.0
        r.best_value = np.inf
        r.improvement_velocity = 0.5
        r.local_entropy = 1.0
        r.spawn_score = 1.0
        r.exploration_bonus = 1.2

        r.prev_dir_white = np.zeros(n_dims)
        r.radius_long = r.radius
        r.radius_lat  = r.radius
        return r

    def _score_spawn_candidate(self, candidate, spawn_region, global_X, global_y):
        try:
            sm = self.surrogate_manager
            if sm is None:
                return 0.0
            mean, std = sm.predict_global_cached(candidate.reshape(1, -1))
            f_best = float(np.min(global_y)) if (global_y is not None and len(global_y) > 0) else 0.0

            # EI score
            mu, s = float(mean[0]), float(std[0])
            if s <= 0 or not np.isfinite(s):
                ei_score = 0.0
            else:
                z = (f_best - mu) / s
                if _scipy_norm is not None:
                    ei_score = s * (z * _scipy_norm.cdf(z) + _scipy_norm.pdf(z))
                else:
                    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
                    pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
                    ei_score = s * (z * cdf + pdf)

            # Diversity bonus
            if self.regions:
                centers = np.array([r.center for r in self.regions])
                min_dist = float(np.min(np.linalg.norm(centers - candidate, axis=1)))
                diversity_score = min_dist / (self._cfg["init_radius"] + 1e-12)
            else:
                diversity_score = 1.0

            # Uncertainty bonus (relative to average)
            if global_X is not None and len(global_X) > 0:
                _, std_all = sm.predict_global_cached(global_X)
                ref = float(np.mean(std_all)) if std_all is not None and len(std_all) else 1.0
            else:
                ref = 1.0
            uncertainty_score = s / (ref + 1e-12)

            total = 0.5 * ei_score + 0.3 * diversity_score + 0.2 * uncertainty_score
            return float(np.nan_to_num(total, nan=0.0))
        except Exception:
            return 0.0

    def _extrapolate_region_center(self, region, bounds, rng, extrapolation_factor=1.5):
        try:
            if len(self.regions) <= 1:
                return None
            others = np.array([r.center for r in self.regions if r is not region])
            rc = np.asarray(region.center)
            dirs = rc - others
            norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
            avg_dir = np.mean(dirs / norms, axis=0)
            avg_dir /= (np.linalg.norm(avg_dir) + 1e-12)
            step = float(region.radius) * float(extrapolation_factor)
            xc = rc + avg_dir * step
            return np.clip(xc, bounds[:, 0], bounds[:, 1])
        except Exception:
            return None

    def _force_spawn(self, bounds, n_dims, rng, global_X, global_y):
        try:
            cand = self._find_best_spawn(bounds, n_dims, rng, global_X, global_y)
            existing = np.array([r.center for r in self.regions]) if self.regions else np.empty((0, n_dims))
            if existing.size > 0:
                min_dist = float(np.min(np.linalg.norm(existing - cand, axis=1)))
                if min_dist < 0.3 * self._cfg["init_radius"]:
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
                print(f"[WARN] Force spawn failed: {e}")

    def _maximin_diverse(self, bounds, n_dims, rng, existing):
        try:
            # favor Sobol if possible
            if _Sobol is not None:
                pts = self._generate_sobol_points(bounds, 128)
            else:
                pts = rng.uniform(bounds[:, 0], bounds[:, 1], size=(128, n_dims))
            if existing is not None and existing.size > 0:
                dmin = _min_dist_to_set(pts.astype(np.float32), existing.astype(np.float32))
                return pts[int(np.argmax(dmin))]
            return pts[0]
        except Exception:
            return bounds[:, 0] + rng.uniform(0, 1, n_dims) * (bounds[:, 1] - bounds[:, 0])

    # --------------------------------------------- Adaptive pruning -----------
    def _adaptive_prune(self):
        if not self.regions:
            return
        max_pts = self._cfg["max_total_points"]
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

    # --------------------------------------------- Diversity floor ------------
    def _ensure_min_diversity(self, bounds, n_dims, rng, global_X, global_y):
        min_needed = max(3, int(self._cfg["n_regions"] * 0.5))
        while len(self.regions) < min_needed:
            self._force_spawn(bounds, n_dims, rng, global_X, global_y)
            if self.verbose:
                print(f"[DIVERSITY] Forced spawn → {len(self.regions)}/{min_needed}")

    # --------------------------------------------- Region weights -------------
    def safe_region_weights(self, regions):
        if not regions:
            return np.array([], dtype=float)
        # try attention head if configured
        try:
            if self.use_neural_attention and (self.region_attn is not None):
                feats_batch = np.stack([self._region_features(r) for r in regions])
                w = self.region_attn.scores(feats_batch)
                if np.all(np.isfinite(w)) and w.sum() > 1e-12:
                    return w / w.sum()
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Attention failed: {e}")

        health = np.array([np_safe(safe_region_property(r, "health_score", 0.0)) for r in regions])
        diversity = np.array([self._region_diversity_score(r) for r in regions])
        # gentle bonus from spawn_score/exploration if available
        spawn = np.array([np_safe(getattr(r, "spawn_score", 0.5)) for r in regions])
        xplr  = np.array([np_safe(getattr(r, "exploration_bonus", 1.0)) for r in regions])
        combined = 0.45 * health + 0.25 * diversity + 0.20 * spawn + 0.10 * (xplr - 1.0)
        ex = np.exp(combined / 0.8)
        return ex / (ex.sum() + 1e-12)

    # -------------------------------------- Assign new data to regions --------
    def update_regions_with_new_data(self, X_new, y_new):
        if not self.regions:
            return
        active = [r for r in self.regions if safe_region_property(r, "is_active", True)]
        if not active:
            return
        try:
            n_dims = int(safe_region_property(active[0], "n_dims", 2))
            centers = np.ascontiguousarray([r.center for r in active], dtype=np.float64)

            # prefetch chol
            Ls = []
            I = np.eye(n_dims)
            for r in active:
                try:
                    L = r.chol() if hasattr(r, "chol") else np.linalg.cholesky(getattr(r, "cov", (r.radius**2) * I) + 1e-12 * I)
                except np.linalg.LinAlgError:
                    C = _eigen_floor_cov(getattr(r, "cov", (r.radius**2) * I), floor=1e-9)
                    L = np.linalg.cholesky(C + 1e-12 * I)
                Ls.append(L)
            L_stack = np.ascontiguousarray(np.stack(Ls, axis=0), dtype=np.float64)

            X_new = np.ascontiguousarray(X_new, dtype=np.float64)
            diffs = X_new[:, None, :] - centers[None, :, :]  # (N, R, D)

            N, R, D = diffs.shape
            mahal_sq = np.empty((N, R), dtype=np.float64)
            for j in range(R):
                Y = np.linalg.solve(L_stack[j].T, diffs[:, j, :].T).T  # (N, D)
                mahal_sq[:, j] = np.einsum("nd,nd->n", Y, Y, optimize=True)

            # soft responsibilities
            avg_radius = float(np.mean([r.radius for r in active]))
            beta = max(3.0, 10.0 * (self._cfg["init_radius"] / (avg_radius + 1e-12)))
            with np.errstate(over="ignore"):
                w = np.exp(-beta * mahal_sq)
            w /= (w.sum(axis=1, keepdims=True) + 1e-12)

            sm = self.surrogate_manager
            for i, (x, y) in enumerate(zip(X_new, y_new)):
                rid = int(np.argmax(w[i]))
                if w[i, rid] > 0.05:
                    s_var = None
                    if sm is not None:
                        try:
                            _, std = sm.predict_global_cached(x[None, :])
                            s_var = float(std[0] ** 2)
                        except Exception:
                            s_var = None
                    active[rid].update(x.astype(np.float64), float(y), surrogate_var=s_var)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Region update failed: {e}")

    # --------------------------------------------- internals ------------------
    def _refresh_centers_cache(self):
        if self.regions:
            try:
                self._centers_cache = np.stack([r.center for r in self.regions])
            except Exception:
                self._centers_cache = None
        else:
            self._centers_cache = None

    def _rng_uniform(self, rng, low, high, size):
        try:
            if hasattr(rng, "uniform"):
                return rng.uniform(low, high, size=size)
            return low + (high - low) * rng.rand(*size)
        except Exception:
            return low + (high - low) * np.random.rand(*size)

    def _restart_region(self, region, bounds, rng):
        try:
            n_dims = int(safe_region_property(region, "n_dims", 2))
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

            # reset stats
            region.center = cand.astype(float)
            region.radius = float(np.clip(radius, self._cfg["min_radius"], self._cfg["init_radius"]))
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

            # archive reset
            if hasattr(region, "clear_archive"):
                region.clear_archive()
            else:
                lx = getattr(region, "local_X", None)
                if isinstance(lx, np.ndarray) and hasattr(region, "_buf_size"):
                    region.local_X[:] = 0.0
                    region.local_y[:] = np.inf
                    region._buf_size = 0
                    region._buf_ptr  = 0
                else:
                    region.local_X = []
                    region.local_y = []

            # covariance
            r2 = region.radius**2
            region.cov = np.eye(n_dims) * r2
            if hasattr(region, "cov_updates_since_reset"):
                region.cov_updates_since_reset = 0

            if self.verbose:
                print(f"[RESTART] Region#{region.region_id} → center={np.round(region.center, 3)}, r={region.radius:.3f}")
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Restart region failed: {e}")

    def _replace_dead_region(self, victim, bounds, rng):
        try:
            n_dims = int(safe_region_property(victim, "n_dims", 2))
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

            # transplant
            victim.center = cand.astype(float)
            victim.radius = float(np.clip(radius, self._cfg["min_radius"], self._cfg["init_radius"]))
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

            if hasattr(victim, "clear_archive"):
                victim.clear_archive()
            else:
                lx = getattr(victim, "local_X", None)
                if isinstance(lx, np.ndarray) and hasattr(victim, "_buf_size"):
                    victim.local_X[:] = 0.0
                    victim.local_y[:] = np.inf
                    victim._buf_size = 0
                    victim._buf_ptr  = 0
                else:
                    victim.local_X = []
                    victim.local_y = []

            r2 = victim.radius**2
            victim.cov = np.eye(n_dims) * r2
            if hasattr(victim, "cov_updates_since_reset"):
                victim.cov_updates_since_reset = 0

            if self.verbose:
                print(f"[REPLACE] Region#{victim.region_id} → center={np.round(victim.center, 3)}, r={victim.radius:.3f}")
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Replace dead region failed: {e}")
