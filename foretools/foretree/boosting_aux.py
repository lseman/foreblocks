# ================================================================
# Unified numeric kernels and utilities for tree training/inference
# ================================================================
from __future__ import annotations

import warnings
from collections import OrderedDict, defaultdict
from threading import RLock
from typing import Dict, Optional, Tuple

import numba
import numpy as np
from numba import njit, prange

# ---------------------------
# Optional GPU (CuPy) support
# ---------------------------
try:
    import cupy as cp
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False


# ================================================================
# Core objective helpers (leaves, gains)
# ================================================================

@njit(fastmath=True, cache=True)
def calc_leaf_value_newton(g_sum: float, h_sum: float, reg_lambda: float,
                           alpha: float = 0.0, max_delta_step: float = 0.0) -> float:
    """Optimal leaf value via Newton step with L1 (soft-threshold) + optional clipping."""
    if h_sum <= 1e-16:
        return 0.0

    denom = h_sum + reg_lambda

    if alpha > 0.0:
        # soft threshold
        if g_sum > alpha:
            leaf_val = -(g_sum - alpha) / denom
        elif g_sum < -alpha:
            leaf_val = -(g_sum + alpha) / denom
        else:
            leaf_val = 0.0
    else:
        leaf_val = -g_sum / denom

    if max_delta_step > 0.0:
        if leaf_val > max_delta_step:
            leaf_val = max_delta_step
        elif leaf_val < -max_delta_step:
            leaf_val = -max_delta_step

    return leaf_val


@njit(fastmath=True, cache=True)
def _leaf_objective_optimal(g_sum: float, h_sum: float, reg_lambda: float,
                            alpha: float, max_delta_step: float) -> float:
    """
    Optimal objective value for a leaf node (used by CCP pruning).
    Matches calc_leaf_value_newton when clipping is active; otherwise uses closed-form.
    """
    if max_delta_step > 0.0:
        v = calc_leaf_value_newton(g_sum, h_sum, reg_lambda, alpha, max_delta_step)
        return g_sum * v + 0.5 * (h_sum + reg_lambda) * v * v + alpha * abs(v)

    denom = h_sum + reg_lambda
    if denom <= 0.0:
        return 0.0

    if alpha > 0.0:
        if g_sum > alpha:
            gsh = g_sum - alpha
        elif g_sum < -alpha:
            gsh = g_sum + alpha
        else:
            gsh = 0.0
    else:
        gsh = g_sum

    return -0.5 * (gsh * gsh) / denom


@njit(fastmath=True, cache=True)
def compute_gain(GL: float, HL: float, GR: float, HR: float, lam: float, gamma: float) -> float:
    """Single source of truth for split gain."""
    G = GL + GR
    H = HL + HR
    return 0.5 * ((GL * GL) / (HL + lam) + (GR * GR) / (HR + lam) - (G * G) / (H + lam)) - gamma


# ================================================================
# Pre-binning (CPU/GPU) over flattened edges
# ================================================================

def prebin_data(
    X: np.ndarray,
    bin_edges: np.ndarray,      # shape: (n_features, n_edges_per_feature) – uniform edges per feature
    n_bins_total: int,          # = actual_max_bins + 1 (last id is missing)
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Pre-bin dense matrix X using the same number of edges per feature.
    Reserves the last bin id as missing (id == actual_max_bins).
    """
    if bin_edges.ndim != 2 or bin_edges.shape[1] < 2:
        raise ValueError("bin_edges must be 2D with at least 2 edges per feature")

    n_features, n_edges_per_feat = bin_edges.shape

    # Flatten edges row-major for kernels
    edges_flat = np.asarray(bin_edges, dtype=np.float64, order="C").ravel()
    edge_starts = np.arange(0, edges_flat.size, n_edges_per_feat, dtype=np.int32)
    edge_counts = np.full(n_features, n_edges_per_feat, dtype=np.int32)

    # Capacity and reserved missing id
    actual_max_bins = int(n_bins_total) - 1
    if actual_max_bins < 1:
        raise ValueError("n_bins_total must be >= 2 (>=1 finite bin + missing)")

    # Intrinsic per-feature bins
    intrinsic_bins = n_edges_per_feat - 1
    if actual_max_bins > intrinsic_bins:
        actual_max_bins = intrinsic_bins
    missing_id = actual_max_bins  # last id

    if use_gpu and GPU_AVAILABLE:
        return _prebin_gpu(X, edges_flat, edge_starts, edge_counts, actual_max_bins, missing_id)
    else:
        return _prebin_cpu_kernel(X, edges_flat, edge_starts, edge_counts, actual_max_bins, missing_id)


@njit(parallel=True, fastmath=True, cache=True)
def _prebin_cpu_kernel(
    X: np.ndarray,               # (n_samples, n_features)
    edges_flat: np.ndarray,      # float64
    edge_starts: np.ndarray,     # int32
    edge_counts: np.ndarray,     # int32
    actual_max_bins: int,        # global finite capacity
    missing_id: int,             # reserved id (== actual_max_bins)
) -> np.ndarray:
    n_samples, n_features = X.shape
    B = np.empty((n_samples, n_features), np.int32)

    for f in prange(n_features):
        es = edge_starts[f]
        ec = edge_counts[f]
        n_edges = ec
        nb = n_edges - 1
        if nb < 1:
            for i in range(n_samples):
                B[i, f] = missing_id
            continue

        if nb > actual_max_bins:
            nb = actual_max_bins
        last_bin = nb - 1

        e0 = edges_flat[es]
        eN = edges_flat[es + n_edges - 1]

        for i in range(n_samples):
            v = X[i, f]
            # fast finite check
            if not (v == v and -np.inf < v < np.inf):
                B[i, f] = missing_id
                continue

            if v <= e0:
                B[i, f] = 0
            elif v > eN:
                B[i, f] = last_bin
            else:
                lo = 0
                hi = n_edges - 1
                while lo < hi - 1:
                    mid = (lo + hi) >> 1
                    if v <= edges_flat[es + mid]:
                        hi = mid
                    else:
                        lo = mid
                b = lo
                if b > last_bin:
                    b = last_bin
                B[i, f] = b
    return B


def _prebin_gpu(
    X: np.ndarray,
    edges_flat: np.ndarray,
    edge_starts: np.ndarray,
    edge_counts: np.ndarray,
    actual_max_bins: int,
    missing_id: int,
) -> np.ndarray:
    if not GPU_AVAILABLE:  # defensive (should be guarded by caller)
        raise RuntimeError("GPU not available")

    n_samples, n_features = X.shape
    Xg = cp.asarray(X, dtype=cp.float32)
    Ef = cp.asarray(edges_flat, dtype=cp.float32)
    Es = cp.asarray(edge_starts, dtype=cp.int32)
    Ec = cp.asarray(edge_counts, dtype=cp.int32)
    Bg = cp.empty((n_samples, n_features), dtype=cp.int32)

    for f in range(n_features):
        es = int(Es[f])
        ec = int(Ec[f])
        nb = ec - 1
        if nb < 1:
            Bg[:, f] = missing_id
            continue
        if nb > actual_max_bins:
            nb = actual_max_bins
        last_bin = nb - 1

        el = Ef[es : es + ec]
        right = el[1:]                 # edges[1:]

        col = Xg[:, f]
        is_finite = cp.isfinite(col)
        bins = cp.searchsorted(right, col, side="left")
        bins = cp.clip(bins, 0, last_bin)
        bins = cp.where(is_finite, bins, missing_id)
        Bg[:, f] = bins

    return Bg.get().astype(np.int32, copy=False)


# ================================================================
# Edges creation (gradient-weighted quantiles), CPU/GPU
# ================================================================

def create_bin_edges(
    X: np.ndarray, gradients: np.ndarray, n_bins: int, use_gpu: bool = False
) -> np.ndarray:
    """Create per-feature edges using gradient-weighted quantiles."""
    if use_gpu and GPU_AVAILABLE:
        try:
            return _create_bins_gpu(X, gradients, n_bins)
        except Exception as e:
            warnings.warn(f"GPU binning failed: {e}; falling back to CPU.")
    return _create_bins_cpu(X, gradients, n_bins)


@njit(parallel=True, fastmath=True, cache=True)
def _create_bins_cpu(X: np.ndarray, gradients: np.ndarray, n_bins: int) -> np.ndarray:
    n_samples, n_features = X.shape
    bin_edges = np.empty((n_features, n_bins + 1), dtype=np.float64)
    abs_gradients = np.abs(gradients)

    for f in prange(n_features):
        feature_data = X[:, f]
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)

        if max_val - min_val < 1e-12:
            bin_edges[f] = np.linspace(min_val - 1e-6, max_val + 1e-6, n_bins + 1)
            continue

        valid_mask = np.isfinite(feature_data)
        if np.sum(valid_mask) < 2:
            bin_edges[f] = np.linspace(min_val, max_val, n_bins + 1)
            continue

        valid_data = feature_data[valid_mask]
        valid_weights = abs_gradients[valid_mask]

        order = np.argsort(valid_data)
        sorted_vals = valid_data[order]
        sorted_weights = valid_weights[order]

        total_weight = np.sum(sorted_weights)
        if total_weight <= 1e-12:
            bin_edges[f] = np.linspace(min_val, max_val, n_bins + 1)
            continue

        edges = np.empty(n_bins + 1, dtype=np.float64)
        edges[0] = sorted_vals[0] - 1e-6
        edges[-1] = sorted_vals[-1] + 1e-6

        cumsum_weights = np.cumsum(sorted_weights)
        for i in range(1, n_bins):
            target_weight = (i / n_bins) * total_weight
            idx = np.searchsorted(cumsum_weights, target_weight)
            if idx < sorted_vals.size:
                edges[i] = sorted_vals[idx]
            else:
                edges[i] = sorted_vals[-1]

        bin_edges[f] = edges

    return bin_edges


def _create_bins_gpu(X: np.ndarray, gradients: np.ndarray, n_bins: int) -> np.ndarray:
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    Xg = cp.asarray(X, dtype=cp.float32)
    wg = cp.abs(cp.asarray(gradients, dtype=cp.float32))
    n_features = X.shape[1]
    out = np.empty((n_features, n_bins + 1), dtype=np.float64)

    for f in range(n_features):
        vals = Xg[:, f]
        min_val = float(cp.min(vals))
        max_val = float(cp.max(vals))
        if max_val - min_val < 1e-12:
            out[f] = np.linspace(min_val - 1e-6, max_val + 1e-6, n_bins + 1)
            continue

        valid_mask = cp.isfinite(vals)
        if int(valid_mask.sum()) < 2:
            out[f] = np.linspace(min_val, max_val, n_bins + 1)
            continue

        v = vals[valid_mask]
        w = wg[valid_mask]
        order = cp.argsort(v)
        sv = v[order]
        sw = w[order]

        tw = float(cp.sum(sw))
        if tw <= 1e-12:
            out[f] = np.linspace(min_val, max_val, n_bins + 1)
            continue

        csum = cp.cumsum(sw)
        qpos = cp.linspace(0.0, tw, n_bins + 1)
        edges = cp.interp(qpos, csum, sv)
        edges[0] = sv[0] - 1e-6
        edges[-1] = sv[-1] + 1e-6
        out[f] = edges.get().astype(np.float64)

    return out


# ================================================================
# Prediction kernels (continuous & binned)
# ================================================================

@njit(fastmath=True, cache=True)
def predict_tree_numba_with_missing_mask(
    X: np.ndarray,
    missing_mask: np.ndarray,
    node_features: np.ndarray,
    node_thresholds: np.ndarray,
    node_missing_go_left: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    leaf_values: np.ndarray,
    is_leaf_flags: np.ndarray,
    feature_map_array: np.ndarray,
    root_idx: int = 0,
) -> np.ndarray:
    n_samples = X.shape[0]
    predictions = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        nid = root_idx
        for _ in range(100):  # guard
            if nid < 0 or nid >= is_leaf_flags.size:
                predictions[i] = 0.0
                break
            if is_leaf_flags[nid]:
                predictions[i] = leaf_values[nid]
                break

            gfi = node_features[nid]
            if gfi < 0 or gfi >= feature_map_array.size:
                predictions[i] = 0.0
                break

            lfi = feature_map_array[gfi]
            if lfi < 0 or lfi >= X.shape[1]:
                predictions[i] = 0.0
                break

            thr = node_thresholds[nid]
            if missing_mask[i, lfi]:
                go_left = node_missing_go_left[nid]
            else:
                go_left = X[i, lfi] <= thr

            nid = left_children[nid] if go_left else right_children[nid]
        else:
            predictions[i] = 0.0
    return predictions


@njit(fastmath=True, cache=True)
def predict_tree_binned_with_missingbin(
    Xb: np.ndarray,                 # (n_samples, n_features) int codes
    missing_bin_id: int,            # == actual_max_bins
    nf: np.ndarray,                 # node_features (global feature id)
    nbin: np.ndarray,               # node_bin_idx (split bin; -1 if not using bins)
    nm: np.ndarray,                 # node_missing_go_left
    lc: np.ndarray,                 # left_children
    rc: np.ndarray,                 # right_children
    lv: np.ndarray,                 # leaf_values
    is_leaf_flags: np.ndarray,      # bool per node
    fmap: np.ndarray,               # global->local map
    root_id: int,                   # root node id
) -> np.ndarray:
    n_samples = Xb.shape[0]
    out = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        nid = root_id
        while nid >= 0 and not is_leaf_flags[nid]:
            gfi = nf[nid]
            if gfi < 0:
                break

            lfi = -1
            if gfi < fmap.size:
                lfi = fmap[gfi]

            go_left = nm[nid]  # default to learned missing direction
            if 0 <= lfi < Xb.shape[1]:
                b = int(Xb[i, lfi])
                if b != missing_bin_id:
                    split_bin = nbin[nid]
                    if split_bin >= 0:
                        go_left = b <= split_bin

            nid = lc[nid] if go_left else rc[nid]
            if nid < 0:
                break

        if nid >= 0:
            out[i] = lv[nid] if is_leaf_flags[nid] else lv[root_id]
        else:
            out[i] = 0.0

    return out


@njit(fastmath=True, cache=True)
def predict_tree_numba(
    X: np.ndarray,
    node_features: np.ndarray,
    node_thresholds: np.ndarray,
    node_missing_left: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    leaf_values: np.ndarray,
    is_leaf: np.ndarray,
    feature_map: np.ndarray,
    root_idx: int = 0,
) -> np.ndarray:
    """Predict using continuous data (non-binned)."""
    n_samples = X.shape[0]
    predictions = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        node = root_idx
        for _ in range(100):
            if node < 0 or node >= is_leaf.size or is_leaf[node]:
                predictions[i] = leaf_values[node] if node >= 0 else 0.0
                break

            gfi = node_features[node]
            if gfi < 0 or gfi >= feature_map.size:
                predictions[i] = 0.0
                break

            lfi = feature_map[gfi]
            if lfi < 0 or lfi >= X.shape[1]:
                predictions[i] = 0.0
                break

            x = X[i, lfi]
            thr = node_thresholds[node]
            if not (x == x):  # NaN check
                go_left = node_missing_left[node]
            else:
                go_left = x <= thr

            node = left_children[node] if go_left else right_children[node]
        else:
            predictions[i] = 0.0

    return predictions


# ================================================================
# Split search (histograms) with explicit missing routing
# ================================================================

@njit(fastmath=True, cache=True)
def find_best_split_with_missing(
    hist_g: np.ndarray,           # (n_features, n_finite_bins)
    hist_h: np.ndarray,           # (n_features, n_finite_bins)
    g_miss_arr: np.ndarray,       # (n_features,)
    h_miss_arr: np.ndarray,       # (n_features,)
    reg_lambda: float,
    gamma: float,
    n_bins: int,                  # finite bins to consider (<= hist_g.shape[1])
    min_child_weight: float = 1e-6,
    monotone_constraints: Optional[np.ndarray] = None,  # (n_features,) in {-1,0,1}
) -> Tuple[int, int, float, int]:
    """
    Returns: (best_feat_idx_local, best_bin_idx, best_gain, missing_left_flag)
    best_bin_idx indexes finite edges; threshold := edges[best_bin_idx].
    Left uses bins [0 .. best_bin_idx-1].
    """
    n_features = hist_g.shape[0]
    best_feat = -1
    best_bin = -1
    best_gain = -1.0e308
    best_missing = 0

    max_bins_avail = hist_g.shape[1]
    max_valid_bin = n_bins if n_bins < max_bins_avail else max_bins_avail
    if max_valid_bin <= 1:
        return best_feat, best_bin, best_gain, best_missing

    for f in range(n_features):
        constraint = 0
        if monotone_constraints is not None and f < monotone_constraints.size:
            constraint = int(monotone_constraints[f])

        G_miss = g_miss_arr[f] if f < g_miss_arr.size else 0.0
        H_miss = h_miss_arr[f] if f < h_miss_arr.size else 0.0

        G_prefix = np.empty(max_valid_bin, dtype=np.float64)
        H_prefix = np.empty(max_valid_bin, dtype=np.float64)
        G_prefix[0] = hist_g[f, 0]
        H_prefix[0] = hist_h[f, 0]
        for b in range(1, max_valid_bin):
            G_prefix[b] = G_prefix[b - 1] + hist_g[f, b]
            H_prefix[b] = H_prefix[b - 1] + hist_h[f, b]

        G_finite = G_prefix[max_valid_bin - 1]
        H_finite = H_prefix[max_valid_bin - 1]

        # totals including missing (kept for feasibility; parent score is absorbed in compute_gain)
        H_tot = H_finite + H_miss
        if H_tot < 2.0 * min_child_weight:
            continue

        for split_bin in range(1, max_valid_bin):
            G_Lp = G_prefix[split_bin - 1]
            H_Lp = H_prefix[split_bin - 1]
            G_Rp = G_finite - G_Lp
            H_Rp = H_finite - H_Lp

            if (H_Lp < min_child_weight) or (H_Rp < min_child_weight):
                continue

            # missing -> left
            G_L = G_Lp + G_miss
            H_L = H_Lp + H_miss
            G_R = G_Rp
            H_R = H_Rp
            if (H_L >= min_child_weight) and (H_R >= min_child_weight):
                gain = compute_gain(G_L, H_L, G_R, H_R, reg_lambda, gamma)
                if gain > best_gain:
                    if constraint != 0 and (H_L > 1e-16) and (H_R > 1e-16):
                        w_L = -G_L / (H_L + reg_lambda)
                        w_R = -G_R / (H_R + reg_lambda)
                        if (constraint == 1 and w_L > w_R + 1e-10) or (constraint == -1 and w_L < w_R - 1e-10):
                            pass
                        else:
                            best_gain = gain
                            best_feat = f
                            best_bin = split_bin
                            best_missing = 1
                    else:
                        best_gain = gain
                        best_feat = f
                        best_bin = split_bin
                        best_missing = 1

            # missing -> right
            G_L = G_Lp
            H_L = H_Lp
            G_R = G_Rp + G_miss
            H_R = H_Rp + H_miss
            if (H_L >= min_child_weight) and (H_R >= min_child_weight):
                gain = compute_gain(G_L, H_L, G_R, H_R, reg_lambda, gamma)
                if gain > best_gain:
                    if constraint != 0 and (H_L > 1e-16) and (H_R > 1e-16):
                        w_L = -G_L / (H_L + reg_lambda)
                        w_R = -G_R / (H_R + reg_lambda)
                        if (constraint == 1 and w_L > w_R + 1e-10) or (constraint == -1 and w_L < w_R - 1e-10):
                            continue
                    best_gain = gain
                    best_feat = f
                    best_bin = split_bin
                    best_missing = 0

    return best_feat, best_bin, best_gain, best_missing


# ================================================================
# Exact-scan helper (single-feature split over sorted order)
# ================================================================

@njit(fastmath=True, cache=True)
def best_split_on_feature_list(
    sorted_ids_for_g: np.ndarray,
    v_seq: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    g_miss: float,
    h_miss: float,
    n_miss: int,
    min_samples_leaf: int,
    min_child_weight: float,
    lambda_reg: float,
    gamma: float,
    monotone_constraint: int = 0,
):
    """Find best split on a single feature using exact sorted order with missing routing."""
    m = sorted_ids_for_g.size
    if m < 2 * min_samples_leaf:
        return -1.0, 0.0, False, 0, 0

    g_seq = np.empty(m, dtype=np.float64)
    h_seq = np.empty(m, dtype=np.float64)
    for i in range(m):
        idx = sorted_ids_for_g[i]
        g_seq[i] = grad[idx]
        h_seq[i] = hess[idx]

    for i in range(1, m):
        g_seq[i] += g_seq[i - 1]
        h_seq[i] += h_seq[i - 1]

    total_g = g_seq[m - 1] + g_miss
    total_h = h_seq[m - 1] + h_miss
    parent_score = (total_g * total_g) / (total_h + lambda_reg) if (total_h + lambda_reg) > 0.0 else 0.0

    best_gain = -1.0
    best_thr = 0.0
    best_mleft = False
    best_nL = 0
    best_nR = 0

    prev = v_seq[0]
    for i in range(1, m):
        cur = v_seq[i]
        if cur <= prev:
            continue

        nL_f = i
        nR_f = m - i
        if nL_f < min_samples_leaf or nR_f < min_samples_leaf:
            prev = cur
            continue

        gL_f = g_seq[i - 1]
        hL_f = h_seq[i - 1]
        gR_f = g_seq[m - 1] - gL_f
        hR_f = h_seq[m - 1] - hL_f

        # missing -> left
        gL = gL_f + g_miss
        hL = hL_f + h_miss
        nL = nL_f + n_miss
        gR = gR_f
        hR = hR_f
        nR = nR_f
        if (hL >= min_child_weight) and (hR >= min_child_weight) and (nL >= min_samples_leaf) and (nR >= min_samples_leaf):
            left_score = (gL * gL) / (hL + lambda_reg) if (hL + lambda_reg) > 0.0 else 0.0
            right_score = (gR * gR) / (hR + lambda_reg) if (hR + lambda_reg) > 0.0 else 0.0
            gain = 0.5 * (left_score + right_score - parent_score) - gamma
            if gain > best_gain:
                if monotone_constraint != 0:
                    wL = -gL / (hL + lambda_reg) if (hL + lambda_reg) > 0.0 else 0.0
                    wR = -gR / (hR + lambda_reg) if (hR + lambda_reg) > 0.0 else 0.0
                    if monotone_constraint * (wR - wL) >= 0:
                        best_gain = gain
                        best_thr = 0.5 * (prev + cur)
                        best_mleft = True
                        best_nL = nL
                        best_nR = nR
                else:
                    best_gain = gain
                    best_thr = 0.5 * (prev + cur)
                    best_mleft = True
                    best_nL = nL
                    best_nR = nR

        # missing -> right
        gL = gL_f
        hL = hL_f
        nL = nL_f
        gR = gR_f + g_miss
        hR = hR_f + h_miss
        nR = nR_f + n_miss
        if (hL >= min_child_weight) and (hR >= min_child_weight) and (nL >= min_samples_leaf) and (nR >= min_samples_leaf):
            left_score = (gL * gL) / (hL + lambda_reg) if (hL + lambda_reg) > 0.0 else 0.0
            right_score = (gR * gR) / (hR + lambda_reg) if (hR + lambda_reg) > 0.0 else 0.0
            gain = 0.5 * (left_score + right_score - parent_score) - gamma
            if gain > best_gain:
                if monotone_constraint != 0:
                    wL = -gL / (hL + lambda_reg) if (hL + lambda_reg) > 0.0 else 0.0
                    wR = -gR / (hR + lambda_reg) if (hR + lambda_reg) > 0.0 else 0.0
                    if monotone_constraint * (wR - wL) >= 0:
                        best_gain = gain
                        best_thr = 0.5 * (prev + cur)
                        best_mleft = False
                        best_nL = nL
                        best_nR = nR
                else:
                    best_gain = gain
                    best_thr = 0.5 * (prev + cur)
                    best_mleft = False
                    best_nL = nL
                    best_nR = nR

        prev = cur

    return best_gain, best_thr, best_mleft, best_nL, best_nR


# ================================================================
# Histogram utilities
# ================================================================

@njit(fastmath=True, cache=True)
def subtract_sibling_histograms(
    parent_hist_g: np.ndarray, parent_hist_h: np.ndarray,
    sibling_hist_g: np.ndarray, sibling_hist_h: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract sibling histograms from parent to get the other child."""
    result_g = np.zeros_like(parent_hist_g)
    result_h = np.zeros_like(parent_hist_h)
    for i in range(parent_hist_g.shape[0]):
        for j in range(parent_hist_g.shape[1]):
            result_g[i, j] = parent_hist_g[i, j] - sibling_hist_g[i, j]
            result_h[i, j] = parent_hist_h[i, j] - sibling_hist_h[i, j]
    return result_g, result_h


# ================================================================
# Membership masks (fast left-mask creation)
# ================================================================

@njit
def _bitmap_fill(bitmap: np.ndarray, left_vals: np.ndarray, offset: int):
    for i in range(left_vals.size):
        v = int(left_vals[i]) - offset
        if 0 <= v < bitmap.size:
            bitmap[v] = 1


@njit
def _bitmap_gather(bitmap: np.ndarray, node_vals: np.ndarray, offset: int, out_mask: np.ndarray):
    for i in range(node_vals.size):
        v = int(node_vals[i]) - offset
        out_mask[i] = 1 if (0 <= v < bitmap.size and bitmap[v] != 0) else 0


def _mask_bitmap(node_indices: np.ndarray, left_indices: np.ndarray, scratch: Dict | None = None) -> np.ndarray:
    lo = int(left_indices.min())
    hi = int(left_indices.max())
    width = hi - lo + 1

    if scratch is None:
        scratch = {}
    bitmap = scratch.get("bitmap")
    if bitmap is None or bitmap.size < width:
        bitmap = np.zeros(width, dtype=np.uint8)
        scratch["bitmap"] = bitmap
    else:
        bitmap[:width] = 0

    _bitmap_fill(bitmap[:width], left_indices, lo)

    out = scratch.get("mask_buf")
    if out is None or out.size < node_indices.size:
        out = np.empty(node_indices.size, dtype=np.uint8)
        scratch["mask_buf"] = out

    _bitmap_gather(bitmap[:width], node_indices, lo, out[: node_indices.size])
    return out[: node_indices.size].astype(bool, copy=False)


def _mask_searchsorted(node_indices: np.ndarray, left_indices: np.ndarray, left_sorted: np.ndarray | None = None) -> np.ndarray:
    if left_sorted is None:
        left_sorted = np.sort(left_indices, kind="mergesort")
    pos = np.searchsorted(left_sorted, node_indices, side="left")
    in_left = pos < left_sorted.size
    eq = np.zeros_like(in_left, dtype=bool)
    if left_sorted.size:
        eq[in_left] = left_sorted[pos[in_left]] == node_indices[in_left]
    return eq


def create_left_mask_adaptive(
    node_indices: np.ndarray,
    left_indices: np.ndarray,
    *,
    scratch: Dict | None = None,
    bitmap_max_bytes: int = 4_000_000,
) -> np.ndarray:
    """Adaptive membership mask with bitmap or searchsorted fallback."""
    n_node = node_indices.size
    n_left = left_indices.size
    if n_left == 0 or n_node == 0:
        return np.zeros(n_node, dtype=bool)
    if n_node < 96:
        return np.isin(node_indices, left_indices, assume_unique=False)

    lo = int(left_indices.min())
    hi = int(left_indices.max())
    width = hi - lo + 1

    if width > 0 and width <= bitmap_max_bytes and width <= 16 * n_left:
        node_i = np.ascontiguousarray(node_indices.astype(np.int64, copy=False))
        left_i = np.ascontiguousarray(left_indices.astype(np.int64, copy=False))
        return _mask_bitmap(node_i, left_i, scratch)

    return _mask_searchsorted(np.ascontiguousarray(node_indices), np.ascontiguousarray(left_indices))


# ================================================================
# Feature importance & histogram cache
# ================================================================

class FeatureImportance:
    def __init__(self):
        self._store = {
            "gain": defaultdict(float),
            "cover": defaultdict(float),
            "split": defaultdict(float),
            # optional model-agnostic slots:
            "perm": defaultdict(float),
            "input_grad": defaultdict(float),
        }

    def add(self, importance_type: str, feat_idx: int, value: float):
        self._store[importance_type][int(feat_idx)] += float(value)

    def add_split(self, feat_idx: int, gain: float, cover: float):
        self._store["gain"][int(feat_idx)] += float(gain)
        self._store["cover"][int(feat_idx)] += float(cover)
        self._store["split"][int(feat_idx)] += 1.0

    def get_importance(self, importance_type: str) -> dict:
        return dict(self._store.get(importance_type, {}))


class HistogramCache:
    """Simple thread-safe LRU cache for histograms."""

    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = RLock()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key, hist_g, hist_h):
        with self._lock:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = (hist_g, hist_h)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
