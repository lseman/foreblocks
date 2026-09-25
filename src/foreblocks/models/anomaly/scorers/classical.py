"""foreblocks.models.anomaly.scorers.classical.

Classical (non-deep) anomaly detection algorithms for time-series windows.

Provides fast, non-parametric anomaly scoring methods that operate directly on
window-level features without requiring neural network training. Includes:

- IsolationForest: ensemble of random isolation trees; anomalies are instances
  that are easier to isolate (shorter path lengths)
- LocalOutlierFactor: density-based method that scores each window by the
  local density deviation of its neighbours
- PCA+Mahalanobis: projects windows to principal components and scores via
  Mahalanobis distance in the reduced space
- MatrixProfile: computes self-similarity via the matrix profile; low
  self-similarity subsequence windows are anomalous

All functions accept 3-D windows of shape [N, T, D] and return per-window
scores of shape [N].

Core API:
- isolation_forest_score: compute Isolation Forest anomaly scores
- lof_score: compute Local Outlier Factor anomaly scores
- pca_mahalanobis_score: compute PCA + Mahalanobis distance scores
- matrix_profile_score: compute Matrix Profile self-similarity scores

"""

from __future__ import annotations

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------


def _build_random_split(
    rng: np.random.RandomState,
    X: np.ndarray,
    feature_indices: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Build a random split on one of the given features."""
    feat = int(rng.choice(feature_indices))
    col = X[:, feat]
    lo, hi = col.min(), col.max()
    if lo == hi:
        return (lo, feat, np.array([]), np.array([]))
    split = rng.uniform(lo, hi)
    left_mask = col < split
    return (split, feat, X[left_mask], X[~left_mask])


def _build_tree(
    rng: np.random.RandomState,
    X: np.ndarray,
    n_samples: int,
    max_depth: int,
) -> dict:
    """Recursively build one isolation tree."""
    if len(X) <= 1 or max_depth <= 0:
        return {"type": "leaf", "size": len(X)}

    feature_indices = np.arange(X.shape[1])
    split, feat, X_left, X_right = _build_random_split(rng, X, feature_indices)

    if len(X_left) == 0 or len(X_right) == 0:
        return {"type": "leaf", "size": len(X)}

    return {
        "type": "node",
        "feat": feat,
        "split": split,
        "left": _build_tree(rng, X_left, len(X_left), max_depth - 1),
        "right": _build_tree(rng, X_right, len(X_right), max_depth - 1),
    }


def _path_length(
    tree: dict,
    point: np.ndarray,
    depth: int = 0,
) -> float:
    """Expected path length for a single point in a single tree."""
    if tree["type"] == "leaf":
        # Average path length of unsuccessful search in BST
        n = tree["size"]
        if n <= 1:
            return float(depth)
        # cn is the average path length of unsuccessful search in BST
        cn = 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n
        return float(depth) + cn

    if point[int(tree["feat"])] < tree["split"]:
        return _path_length(tree["left"], point, depth + 1)
    else:
        return _path_length(tree["right"], point, depth + 1)


def _average_cdf(n: int) -> float:
    """Average path length of unsuccessful search in BST."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def isolation_forest_score(
    windows: np.ndarray,
    n_estimators: int = 100,
    max_samples: int = 256,
    max_depth: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Compute Isolation Forest anomaly scores for windows.

    Higher scores indicate more anomalous windows.

    Args:
        windows: array of shape [N, T, D].
        n_estimators: number of isolation trees.
        max_samples: subsample size per tree.
        max_depth: explicit max depth (defaults to ceil(log2(max_samples))).
        seed: random seed.

    Returns:
        scores array of shape [N] with values in [0, 1].
    """
    rng = np.random.RandomState(seed)
    windows = np.asarray(windows, dtype=np.float64)
    n, t, d = windows.shape
    flat = windows.reshape(n, -1)

    subsample_size = min(max_samples, n)
    if max_depth is None:
        max_depth = int(np.ceil(np.log2(max(subsample_size, 2))))

    # Build trees with subsamples
    trees = []
    for _ in range(n_estimators):
        indices = rng.choice(n, size=subsample_size, replace=False)
        trees.append(_build_tree(rng, flat[indices], subsample_size, max_depth))

    # Compute path lengths
    avg_paths = np.zeros(n)
    c = _average_cdf(subsample_size)

    for i in range(n):
        total = 0.0
        for tree in trees:
            total += _path_length(tree, flat[i])
        avg_paths[i] = total / n_estimators

    # Convert to anomaly score: s(x, n) = 2^(-E(h(x))/c(n))
    scores = np.power(2.0, -avg_paths / max(c, 1e-8))
    # Shift so normal points are ~0.5, anomalies > 0.5
    scores = (scores - 0.5) * 2.0  # map [0,1] -> [-1,1]
    # Clip to [0, 1]
    scores = np.clip(scores, 0.0, 1.0)
    return scores


# ---------------------------------------------------------------------------
# Local Outlier Factor (simplified, k-distance based)
# ---------------------------------------------------------------------------


def lof_score(
    windows: np.ndarray,
    n_neighbors: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Compute Local Outlier Factor anomaly scores for windows.

    Scores represent the local density deviation of each window relative
    to its k nearest neighbours. Higher scores = more anomalous.

    Uses Euclidean distance on flattened windows. For efficiency on large
    datasets, a random subset of neighbours is used.

    Args:
        windows: array of shape [N, T, D].
        n_neighbors: number of neighbours for LOF computation.
        seed: random seed for tie-breaking.

    Returns:
        scores array of shape [N].
    """
    rng = np.random.RandomState(seed)
    windows = np.asarray(windows, dtype=np.float64)
    n = windows.shape[0]
    flat = windows.reshape(n, -1)

    # Compute pairwise distances (memory-efficient chunking)
    CHUNK = 512
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(0, n, CHUNK):
        end = min(i + CHUNK, n)
        diff = flat[i:end, np.newaxis, :] - flat[np.newaxis, :, :]
        dist_matrix[i:end] = np.sqrt(np.sum(diff ** 2, axis=2))

    # For each point, get k nearest neighbour distances
    k = min(n_neighbors, n - 1)
    k_distances = np.zeros(n)
    k_indices = np.zeros((n, k), dtype=np.intp)

    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf  # exclude self
        idx = np.argpartition(dists, k)[:k]
        k_indices[i] = idx
        k_distances[i] = dists[idx].max()  # k-distance

    # Local reachability density
    lrd = np.zeros(n)
    for i in range(n):
        reach_dists = np.maximum(k_distances[k_indices[i]], dist_matrix[i, k_indices[i]])
        avg_reach = reach_dists.mean()
        lrd[i] = 1.0 / (avg_reach + 1e-8)

    # LOF = average ratio of LRD of neighbours to own LRD
    scores = np.zeros(n)
    for i in range(n):
        neighbour_lrd = lrd[k_indices[i]]
        scores[i] = neighbour_lrd.mean() / (lrd[i] + 1e-8)

    # Shift: normal points ~1.0, anomalies > 1.0
    scores = scores - 1.0
    return np.maximum(scores, 0.0)


# ---------------------------------------------------------------------------
# PCA + Mahalanobis Distance
# ---------------------------------------------------------------------------


def _pca_fit(
    X: np.ndarray,
    n_components: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit PCA via eigendecomposition of covariance.

    Returns:
        components: [d, k] eigenvectors (principal axes)
        explained_variance: [k] eigenvalues
        mean: [d] dataset mean
        total_var: total variance of original data
    """
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if n_components is None:
        n_components = X.shape[1]
    n_components = min(n_components, X.shape[1])

    # Ensure positive eigenvalues for PCA
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_var = eigenvalues.sum()

    return eigenvectors[:, :n_components], eigenvalues[:n_components], mean, total_var


def _mahalanobis_scores(
    X: np.ndarray,
    components: np.ndarray,
    explained_var: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis distance scores in PCA subspace.

    Uses the diagonal covariance in PCA space (principal axes directions
    scaled by their variance).

    Returns:
        Per-sample squared Mahalanobis distances.
    """
    X_centered = X - mean
    proj = X_centered @ components  # [n, k]

    # In PCA space, covariance is diagonal = explained_var
    inv_sqrt_var = 1.0 / np.sqrt(explained_var + 1e-8)
    weighted = proj * inv_sqrt_var[np.newaxis, :]
    scores = np.sum(weighted ** 2, axis=1)
    return scores


def pca_mahalanobis_score(
    windows: np.ndarray,
    n_components: int | None = None,
) -> np.ndarray:
    """Compute PCA + Mahalanobis distance anomaly scores.

    Projects flattened windows to their principal components and scores
    each window by its Mahalanobis distance in the reduced space, where
    each axis is normalised by the corresponding eigenvalue.

    This is sensitive to outliers in the covariance estimate. For robust
    results on highly contaminated data, consider using the mode parameter.

    Args:
        windows: array of shape [N, T, D].
        n_components: number of PCA components (default: min features, samples).

    Returns:
        scores array of shape [N] (higher = more anomalous).
    """
    windows = np.asarray(windows, dtype=np.float64)
    n, t, d = windows.shape
    flat = windows.reshape(n, -1)

    components, explained_var, mean, total_var = _pca_fit(flat, n_components)

    # Mahalanobis in PCA space
    scores = _mahalanobis_scores(flat, components, explained_var, mean)

    # Normalise by total variance so scores are comparable
    scores = scores / (total_var + 1e-8)

    return scores


# ---------------------------------------------------------------------------
# Matrix Profile (STAMP approximation)
# ---------------------------------------------------------------------------


def matrix_profile_score(
    windows: np.ndarray,
    subseq_len: int | None = None,
    top_frac: float = 0.5,
) -> np.ndarray:
    """Compute Matrix Profile self-similarity anomaly scores.

    For each subsequence window, the Matrix Profile value is the distance to
    its nearest neighbour (excluding self). Windows with high Matrix Profile
    values are those that are dissimilar to everything else — classic
    anomalies.

    Uses a sliding-window approximation (STAMP-like) for efficiency.

    Args:
        windows: array of shape [N, T, D].
        subseq_len: query subsequence length (defaults to window_size // 4).
        top_frac: fraction of top distances to use for normalisation.

    Returns:
        scores array of shape [N] (higher = more anomalous).
    """
    windows = np.asarray(windows, dtype=np.float64)
    n, t, d = windows.shape

    if subseq_len is None:
        subseq_len = max(4, t // 4)
    subseq_len = min(subseq_len, t)

    # For multi-feature windows, compute per-feature MP and average
    all_scores = []

    for feat_idx in range(d):
        col = windows[:, :, feat_idx].reshape(n, t)

        # Extract all subsequences of length subseq_len
        n_sub = t - subseq_len + 1
        if n_sub <= 1:
            # Window shorter than subsequence — score uniformly
            all_scores.append(np.ones(n))
            continue

        # Build subsequence matrix [n * n_sub, subseq_len]
        subsequences = []
        for i in range(n):
            for j in range(n_sub):
                subsequences.append(col[i, j : j + subseq_len])
        subsequences = np.array(subsequences)

        # Compute pairwise distances (chunked)
        n_total = subsequences.shape[0]
        CHUNK = 256
        min_dists = np.full(n_total, np.inf)

        for i in range(0, n_total, CHUNK):
            end = min(i + CHUNK, n_total)
            diff = subsequences[i:end, np.newaxis, :] - subsequences[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            # Exclude self-match
            for local_i in range(end - i):
                global_i = i + local_i
                row_idx = global_i // n_sub
                col_idx = global_i % n_sub
                self_global = row_idx * n_sub + col_idx
                dists[local_i, self_global] = np.inf
            min_dists[i:end] = dists.min(axis=1)

        # Aggregate per-window: use max MP distance across all sub-windows
        window_scores = np.zeros(n)
        for i in range(n):
            start = i * n_sub
            end = start + n_sub
            window_scores[i] = min_dists[start:end].max()

        all_scores.append(window_scores)

    # Average across features
    scores = np.mean(all_scores, axis=0)

    # Normalise: shift so median is ~0.5
    median_score = np.median(scores)
    scores = (scores - median_score) / (np.nanpercentile(scores, 99) - median_score + 1e-8)
    scores = np.clip(scores, 0.0, 1.0)

    return scores
