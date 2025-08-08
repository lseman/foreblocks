import warnings

warnings.filterwarnings("ignore")

# Core Python

# Parallelism & Concurrency
from dataclasses import dataclass

# Numerical & Scientific Computing
import numpy as np
# BoTorch & GPyTorch (Gaussian Processes)
from numba import jit

# Numba JIT


# PyTorch Core



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


import numpy as np
from numba import jit


# === Sobol-like Quasi-Random Sequence ===
@jit(nopython=True)
def sobol_sequence(seed, n_points, n_dims):
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


def compute_coverage(X, centers, radii):
    """
    Compute coverage of points X by hyperspheres centered at centers with radius*2.
    Returns fraction of points covered.
    """

    # ✅ Ensure centers/radii are NumPy arrays
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    if centers.size == 0 or radii.size == 0:
        return 0.0  # no regions → no coverage

    n_points = X.shape[0]
    n_regions = centers.shape[0]
    X.shape[1]

    covered_count = 0

    for i in range(n_points):
        for j in range(n_regions):
            # Compute squared distance manually (no sqrt)
            dist_sq = np.sum((X[i] - centers[j]) ** 2)
            if dist_sq <= (radii[j] * 2.0) ** 2:
                covered_count += 1
                break  # early exit once covered

    return covered_count / n_points


def compute_coverage_mahalanobis(X, centers, radii, cov=None):
    """
    Mahalanobis coverage fraction (non-jitted).
    """
    if cov is None:
        cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)

    covered = 0
    for x in X:
        dists = [np.sqrt((x - c) @ cov_inv @ (x - c).T) for c in centers]
        if np.any(np.array(dists) <= radii * 2.0):
            covered += 1
    return covered / len(X)


# ============================================================
# Fast approximations for normal CDF & PDF (Numba-friendly)
# ============================================================


@jit(nopython=True, inline="always")
def norm_pdf(x):
    """Numba-compatible standard normal PDF"""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@jit(nopython=True, inline="always")
def norm_cdf(x):
    """
    Fast Numba-compatible CDF approximation:
    - uses tanh-based approximation of erf()
    - sufficient for acquisition functions
    """
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


# ============================================================
# Acquisition Functions
# ============================================================


@jit(nopython=True)
def expected_improvement(mean, std, best_value):
    """Vectorized Expected Improvement (EI)"""
    std_safe = std + 1e-12  # avoid div by zero
    improvement = best_value - mean
    z = improvement / std_safe
    return improvement * norm_cdf(z) + std_safe * norm_pdf(z)


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
def upper_confidence_bound(mean, std, beta):
    """UCB: -mean + beta * std (maximize exploration)"""
    return -mean + beta * std


@jit(nopython=True)
def probability_improvement(mean, std, best_value):
    """Probability of Improvement (PI)"""
    std_safe = std + 1e-12
    improvement = best_value - mean
    z = improvement / std_safe
    return norm_cdf(z)


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


@jit(nopython=True)
def noisy_expected_improvement(mean, std, best_value, noise):
    """Noisy EI that accounts for observation noise"""
    eff_std = np.sqrt(std * std + noise * noise)
    return expected_improvement(mean, eff_std, best_value)


# ============================================================
# Diversity-aware Candidate Selection (DPP-like)
# ============================================================


@jit(nopython=True)
def rbf_kernel_matrix(X, lengthscale=0.5):
    """Compute an RBF kernel matrix (Numba-friendly)."""
    n, d = X.shape
    K = np.zeros((n, n))
    inv_ls_sq = 1.0 / (lengthscale * lengthscale)

    for i in range(n):
        for j in range(n):
            sqdist = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                sqdist += diff * diff
            K[i, j] = np.exp(-0.5 * inv_ls_sq * sqdist)
    return K


def dpp_select(X, scores, batch_size=5, lengthscale=0.5):
    """
    Greedy DPP-like batch selection:
    - Encourages diversity via RBF kernel.
    - Selects high-score points while penalizing redundancy.
    """
    n = len(X)
    if batch_size >= n:
        return np.argsort(-scores)[:batch_size]

    K = rbf_kernel_matrix(X, lengthscale)
    selected = []
    remaining = list(range(n))

    for _ in range(batch_size):
        best_idx = -1
        best_gain = -1e9

        for j in remaining:
            # diversity penalty = sum of kernel similarities with already selected
            diversity_penalty = 0.0
            if len(selected) > 0:
                diversity_penalty = np.sum(K[j, selected])

            gain = scores[j] - 0.1 * diversity_penalty
            if gain > best_gain:
                best_gain = gain
                best_idx = j

        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.array(selected)


# ======================================================
# === Upgraded AcquisitionManager (EXP3-IX, TS, entropy) ===
# ======================================================


import numpy as np
