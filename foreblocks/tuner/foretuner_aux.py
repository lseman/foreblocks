import math
import warnings
from dataclasses import dataclass

import numpy as np
from numba import jit, vectorize

warnings.filterwarnings("ignore")


@dataclass
class TurboConfig:
    n_init: int = 15
    max_evals: int = 100
    batch_size: int = 1
    n_regions: int = 4
    init_radius: float = 0.25
    min_radius: float = 1e-3
    max_radius: float = 0.9
    expansion_factor: float = 1.5
    contraction_factor: float = 0.8
    acquisition: str = "ts"  # Thompson Sampling as default
    ucb_beta: float = 3.0
    update_frequency: int = 5
    management_frequency: int = 10
    min_local_samples: int = 6
    max_local_data: int = 50
    n_candidates: int = 50
    # Enhanced parameters
    spawn_threshold: float = 0.6
    merge_threshold: float = 0.1
    kill_threshold: int = 15
    diversity_weight: float = 0.2
    exploration_factor: float = 0.4
    restart_threshold: int = 25
    # Thompson Sampling parameters
    n_thompson_samples: int = 10
    n_fantasy_samples: int = 5
    ts_min_variance: float = 1e-6
    # Diversity and selection
    diversity_metric: str = "kl"  # Options: "kl", "wasserstein", "euclidean"
    selection_method: str = "nsga2"  # Options: "nsga", "nsga2", "random"

    merge_interval: int = 5  # How often to check for merges
    restart_stagnation_limit: int = 50  # trigger restart after 50 stagnated steps

    verbose = False  # Enable/disable verbose logging
    min_diversity_threshold: float = (
        0.1  # Minimum diversity threshold for candidate selection
    )
    coverage_metric: str = "default"  # Options: "mahalanobis", "euclidean"
    spawn_w_cov: bool = True  # Use coverage for spawning new regions
    spawn_w_health: bool = True  # Use health for spawning new regions
    spawn_w_entropy: bool = True  # Use entropy for spawning new regions

    max_age: int = 1000  # Maximum age of a region before it is considered for removal


# === Sobol-like Quasi-Random Sequence ===
@jit(nopython=True)
def sobol_sequence(n_points, n_dims, seed=0):
    """
    Numba-friendly quasi-random sequence generator with lightweight scrambling.
    Produces low-discrepancy points using golden ratio additive scrambling.
    """
    np.random.seed(seed)
    # initial pseudo-random uniform
    base_samples = np.random.rand(n_points, n_dims)

    # Scrambling constants
    golden_ratio = 0.6180339887498948482
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])

    n_primes = primes.shape[0]

    for d in range(n_dims):
        # pick offset based on prime or fallback linear increment
        offset = (primes[d] if d < n_primes else (d + 1)) * golden_ratio
        for i in range(n_points):
            val = base_samples[i, d] + offset
            # wrap around [0,1]
            if val >= 1.0:
                val -= np.floor(val)
            base_samples[i, d] = val

    return base_samples


# === Safe sqrt ===
@jit(nopython=True, inline="always")
def safe_sqrt(x):
    """Fast & safe sqrt: clamps negative small noise to 0."""
    return np.sqrt(x) if x > 0.0 else 0.0


@jit(nopython=True)
def compute_coverage(X, centers, radii):
    """
    Fraction of points in X covered by any hypersphere of radius 2*r_j.
    centers: (m,d), radii: (m,) or scalar
    """
    centers = centers.astype(np.float64)
    if radii.ndim == 0:
        radii = np.full(centers.shape[0], float(radii))
    else:
        radii = radii.astype(np.float64)

    if centers.shape[0] == 0 or radii.shape[0] == 0:
        return 0.0

    n = X.shape[0]
    m = centers.shape[0]
    covered = 0
    for i in range(n):
        xi = X[i]
        for j in range(m):
            r = 2.0 * radii[j]
            r2 = r * r
            dist2 = 0.0
            for k in range(xi.shape[0]):
                diff = xi[k] - centers[j, k]
                dist2 += diff * diff
            if dist2 <= r2:
                covered += 1
                break
    return covered / n


def compute_coverage_mahalanobis(X, centers, radii, cov=None):
    """
    Fraction covered under Mahalanobis balls of radius 2*r.
    Implemented by whitening: y = L^{-1} x, where LL^T=cov.
    """
    X = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim == 0:
        radii = np.full(centers.shape[0], float(radii))

    if cov is None:
        cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6

    L = np.linalg.cholesky(cov)  # cov = L L^T
    Linv = np.linalg.inv(L)  # or solve triangular per batch

    Xw = (Linv @ X.T).T  # whitened
    Cw = (Linv @ centers.T).T

    return compute_coverage(Xw, Cw, radii)  # reuse Euclidean version


# ============================================================
# Fast approximations for normal CDF & PDF (Numba-friendly)
# ============================================================

@vectorize(["float64(float64)"], nopython=True)
def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@vectorize(["float64(float64)"], nopython=True)
def norm_cdf(x):
    # 0.5 * [1 + erf(x / sqrt(2))]
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ============================================================
# Acquisition Functions
# ============================================================


@jit(nopython=True)
def expected_improvement(mean, std, best_value):
    std_safe = std + 1e-12
    diff = best_value - mean
    z = diff / std_safe
    return diff * norm_cdf(z) + std_safe * norm_pdf(z)


@jit(nopython=True)
def probability_improvement(mean, std, best_value):
    std_safe = std + 1e-12
    return norm_cdf((best_value - mean) / std_safe)


@jit(nopython=True)
def upper_confidence_bound(mean, std, beta):
    return -mean + beta * std


def log_expected_improvement(mean, std, best_value, eps=1e-9):
    """
    Log-EI for numerical stability (uses scipy for more accurate CDF).
    Only used outside Numba.
    """
    from scipy.stats import norm

    diff = best_value - mean - eps
    std_safe = std + eps
    z = diff / std_safe
    ei = diff * norm.cdf(z) + std_safe * norm.pdf(z)
    return np.log(ei + eps)


@jit(nopython=True)
def predictive_entropy_search(mean, std, best_value):
    """
    Predictive Entropy Search (PES) simplified:
    - just returns std as exploration proxy.
    """
    return std


@jit(nopython=True)
def knowledge_gradient(mean, std, best_value):
    """Knowledge Gradient (KG)"""
    std_safe = np.maximum(std, 1e-9)
    z = (best_value - mean) / std_safe
    kg = std_safe * norm_pdf(z) + (best_value - mean) * norm_cdf(z)
    return np.maximum(kg, 0.0)
