"""Standalone binning strategies for numerical features.

Each function accepts an array of values and returns bin edges or a
transformer suitable for sklearn-compatible ``transform`` calls.
"""

from __future__ import annotations

import warnings
from math import log2

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer

from .stats_safe import safe_skew


def _finite(x: np.ndarray) -> np.ndarray:
    """Return finite elements of x."""
    return x[np.isfinite(x)]


def _clean_edges(edges: np.ndarray) -> np.ndarray:
    """Normalize edges array: finite, sorted, unique, padded."""
    e = np.asarray(edges, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 2:
        return np.array([0.0, 1.0], dtype=float)
    e = np.unique(np.sort(e))
    if e.size < 2:
        return np.array([e[0], e[0] + 1.0], dtype=float)
    e[0] -= 1e-12
    e[-1] += 1e-12
    return e


def _digitize_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin x by edges, returning 0-based bin indices."""
    b = np.digitize(x, edges) - 1
    return np.clip(b, 0, len(edges) - 2)


def _enforce_min_support_edges(
    x: np.ndarray, edges: np.ndarray, min_count: int
) -> np.ndarray:
    """Merge weak bins by removing internal edges until support constraints are met."""
    edges = _clean_edges(edges)
    x_f = _finite(x)
    if x_f.size == 0 or len(edges) <= 2:
        return edges

    while len(edges) > 2:
        b = _digitize_edges(x_f, edges)
        counts = np.bincount(b, minlength=len(edges) - 1)
        weak = np.where(counts < min_count)[0]
        if weak.size == 0:
            break
        i = int(weak[0])
        if i == 0:
            drop_edge_idx = 1
        elif i == len(counts) - 1:
            drop_edge_idx = len(edges) - 2
        else:
            left_c = counts[i - 1]
            right_c = counts[i + 1]
            drop_edge_idx = i if left_c <= right_c else i + 1
        edges = np.delete(edges, drop_edge_idx)

    return _clean_edges(edges)


# ── Strategy functions ───────────────────────────────────────────────


def fd_edges(x: np.ndarray, max_bins: int = 100) -> np.ndarray:
    """Freedman–Diaconis rule → equal-width edges."""
    x_f = _finite(x)
    n = x_f.size
    if n <= 1:
        return np.array([0.0, 1.0])
    iqr = float(np.subtract(*np.percentile(x_f, [75, 25])))
    if iqr <= 0:
        k = min(max_bins, max(1, int(np.sqrt(n))))
    else:
        h = 2 * iqr * (n ** (-1 / 3))
        if h <= 0:
            k = min(max_bins, max(1, int(np.sqrt(n))))
        else:
            k = int(np.ceil((x_f.max() - x_f.min()) / h))
    return np.linspace(x_f.min(), x_f.max(), min(k, max_bins) + 1)


def doane_edges(x: np.ndarray, max_bins: int = 100) -> np.ndarray:
    """Doane's rule (skew-adjusted Sturges) → equal-width edges."""
    x_f = _finite(x)
    n = x_f.size
    if n <= 1:
        return np.array([0.0, 1.0])
    g1 = float(safe_skew(x_f, bias=False)) if n >= 8 else 0.0
    sigma_g1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3) + 1e-12))
    k = 1 + np.log2(n) + np.log2(1 + abs(g1) / (sigma_g1 + 1e-12))
    k = max(2, int(np.clip(int(round(k)), 1, max_bins)))
    return np.linspace(x_f.min(), x_f.max(), k + 1)


def shimazaki_edges(
    x: np.ndarray, max_bins: int = 128, kmin: int = 4, kmax: int = 128
) -> np.ndarray:
    """Shimazaki–Shinomoto optimal bin count → equal-width edges."""
    x_f = _finite(x)
    n = x_f.size
    if n <= 1:
        return np.array([0.0, 1.0])
    xmin, xmax = x_f.min(), x_f.max()
    if xmax - xmin < 1e-12:
        return np.array([0.0, 1.0])
    best_k, best_cost = 1, np.inf
    kmin = max(2, int(kmin))
    kmax = int(min(max_bins, max(kmin + 1, kmax)))
    for k in range(kmin, kmax + 1):
        counts, _ = np.histogram(x_f, bins=k)
        m = counts.mean()
        v = counts.var()
        h = (xmax - xmin) / k
        cost = (2 * m - v) / (h**2 + 1e-12)
        if cost < best_cost:
            best_cost = cost
            best_k = k
    k = max(2, int(np.clip(best_k, 1, max_bins)))
    return np.linspace(xmin, xmax, k + 1)


def kmeans_edges(x: np.ndarray, k: int = 10, random_state: int = 42) -> np.ndarray:
    """1D K-Means edges (handles multi-modality)."""
    x_f = _finite(x)
    if x_f.size == 0:
        return np.array([0.0, 1.0])
    k = max(2, min(k, x_f.size))
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    km.fit(x_f.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.flatten())
    inner = (centers[1:] + centers[:-1]) / 2
    left = x_f.min() - 1e-12
    right = x_f.max() + 1e-12
    return np.concatenate([[left], inner, [right]])


def quantile_transformer(
    x: np.ndarray, n_bins: int = 10, min_count: int = 10, subsample: int = 20000
) -> KBinsDiscretizer | None:
    """Quantile binned via KBinsDiscretizer with backoff on min support."""
    x_2d = _finite(x).reshape(-1, 1)
    x_f = _finite(x)
    if x_f.size < 20 or x_f.size < min_count * 2:
        return None

    k = int(np.clip(n_bins, 2, max(2, min(x_f.size, np.unique(x_f).size))))
    subsample_n = min(subsample, len(x_2d))
    if len(x_2d) > subsample_n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(x_2d), subsample_n, replace=False)
        x_fit = x_2d[idx]
    else:
        x_fit = x_2d

    params = {
        "n_bins": k,
        "encode": "ordinal",
        "strategy": "quantile",
        "subsample": subsample_n,
    }
    try:
        disc = KBinsDiscretizer(**params, quantile_method="averaged_inverted_cdf")
    except TypeError:
        disc = KBinsDiscretizer(**params)

    disc.fit(x_fit)
    binned = disc.transform(x_fit).astype(int).ravel()
    actual_bins = len(np.unique(binned))
    counts = np.bincount(binned, minlength=max(actual_bins, 1))

    if actual_bins >= 2 and counts.min() >= min_count:
        return disc
    return None


def uniform_transformer(
    x: np.ndarray, n_bins: int = 10, min_count: int = 10, subsample: int = 20000
) -> KBinsDiscretizer | None:
    """Uniform (equal-width) binned via KBinsDiscretizer with backoff."""
    x_2d = _finite(x).reshape(-1, 1)
    x_f = _finite(x)
    if x_f.size < 20 or x_f.size < min_count * 2:
        return None

    k = int(np.clip(n_bins, 2, max(2, min(x_f.size, np.unique(x_f).size))))
    subsample_n = min(subsample, len(x_2d))
    if len(x_2d) > subsample_n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(x_2d), subsample_n, replace=False)
        x_fit = x_2d[idx]
    else:
        x_fit = x_2d

    params = {
        "n_bins": k,
        "encode": "ordinal",
        "strategy": "uniform",
        "subsample": subsample_n,
    }
    try:
        disc = KBinsDiscretizer(**params, quantile_method="averaged_inverted_cdf")
    except TypeError:
        disc = KBinsDiscretizer(**params)

    disc.fit(x_fit)
    binned = disc.transform(x_fit).astype(int).ravel()
    actual_bins = len(np.unique(binned))
    counts = np.bincount(binned, minlength=max(actual_bins, 1))

    if actual_bins >= 2 and counts.min() >= min_count:
        return disc
    return None


def mdlp_edges(x: np.ndarray, y: np.ndarray, max_bins: int = 100) -> np.ndarray:
    """Fayyad–Irani MDL supervised binning → edges."""
    x_f = _finite(x)
    mask = np.isfinite(y)
    x_v, y_v = x_f[mask], y[mask]
    if x_v.size == 0:
        return np.array([0.0, 1.0])

    order = np.argsort(x_v)
    x_v, y_v = x_v[order], y_v[order]

    def candidate_splits(xv, yv):
        cs = []
        for i in range(1, xv.size):
            if xv[i] != xv[i - 1] and yv[i] != yv[i - 1]:
                cs.append((xv[i - 1] + xv[i]) / 2.0)
        return cs

    def entropy(labels):
        _, cnt = np.unique(labels, return_counts=True)
        p = cnt / cnt.sum()
        return float(-(p * np.log2(p + 1e-12)).sum())

    def mdl_stop(parent_y, left_y, right_y, N, k):
        H_p = entropy(parent_y)
        H_l, H_r = entropy(left_y), entropy(right_y)
        Nl, Nr = len(left_y), len(right_y)
        gain = H_p - (Nl / N) * H_l - (Nr / N) * H_r
        k_l, k_r = len(np.unique(left_y)), len(np.unique(right_y))
        delta = log2(3**k - 2) - (k * H_p - k_l * H_l - k_r * H_r)
        threshold = (log2(max(N - 1, 1)) + delta) / max(N, 1)
        return gain > threshold, gain

    def split_rec(xv, yv):
        if len(np.unique(yv)) <= 1 or xv.size < 2:
            return []
        k = len(np.unique(yv))
        splits = candidate_splits(xv, yv)
        if not splits:
            return []
        best_s, best_gain = None, -np.inf
        for s in splits:
            left_mask = xv <= s
            yl, yr = yv[left_mask], yv[~left_mask]
            if yl.size == 0 or yr.size == 0:
                continue
            ok, gain = mdl_stop(yv, yl, yr, xv.size, k)
            if ok and gain > best_gain:
                best_gain, best_s = gain, s
        if best_s is None:
            return []
        left = xv <= best_s
        return (
            split_rec(xv[left], yv[left]) + [best_s] + split_rec(xv[~left], yv[~left])
        )

    cut_points = split_rec(x_v, y_v)
    cut_points = sorted(set(cut_points))
    if len(cut_points) + 1 > max_bins:
        step = max(1, int(np.ceil(len(cut_points) / (max_bins - 1))))
        cut_points = cut_points[::step][: max(0, max_bins - 1)]

    return np.array([x_v.min() - 1e-12] + cut_points + [x_v.max() + 1e-12], dtype=float)


def woe_edges_and_map(
    x: np.ndarray, y: np.ndarray, max_bins: int = 100
) -> tuple[np.ndarray, dict[int, float], float]:
    """MDLP edges + Weight of Evidence map + Information Value.

    Returns (edges, woe_map, iv).
    """
    edges = mdlp_edges(x, y, max_bins)
    binned = _digitize_edges(x, edges)
    df = pd.DataFrame({"bin": binned, "target": y})
    total_events = float(df["target"].sum())
    total_non_events = len(df) - total_events
    if total_events == 0 or total_non_events == 0:
        return edges, {i: 0.0 for i in range(len(edges) - 1)}, 0.0

    stats = df.groupby("bin")["target"].agg(["sum", "count"])
    stats["non_sum"] = stats["count"] - stats["sum"]
    eps = 1e-6
    dist_event = (stats["sum"] + eps) / (total_events + eps)
    dist_non_event = (stats["non_sum"] + eps) / (total_non_events + eps)
    stats["woe"] = np.log(dist_event / dist_non_event)
    iv = float(((dist_event - dist_non_event) * stats["woe"]).sum())

    return edges, stats["woe"].to_dict(), iv
