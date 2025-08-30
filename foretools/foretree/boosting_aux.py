import warnings
from collections import OrderedDict, defaultdict
from threading import RLock
from typing import Optional, Tuple, Union

import cupy as cp
import numba
import numpy as np

# ---------- CPU (Numba) ----------
# -------- CPU: Numba kernel over flattened edges --------
# ================================================================
# Tree prediction
# ================================================================
from numba import njit, prange
from numba.typed import List as NTypedList

# GPU support detection
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# ================================================================
# Core numerical computations
# ================================================================


@njit(fastmath=True, cache=True)
def calc_leaf_value_newton(g_sum, h_sum, reg_lambda, alpha=0.0, max_delta_step=0.0):
    """Compute optimal leaf value using Newton's method."""
    if h_sum <= 1e-16:
        return 0.0

    denom = h_sum + reg_lambda

    # L1 regularization (soft thresholding)
    if alpha > 0.0:
        if g_sum > alpha:
            leaf_val = -(g_sum - alpha) / denom
        elif g_sum < -alpha:
            leaf_val = -(g_sum + alpha) / denom
        else:
            leaf_val = 0.0
    else:
        leaf_val = -g_sum / denom

    # Apply step size constraint
    if max_delta_step > 0.0:
        leaf_val = max(-max_delta_step, min(max_delta_step, leaf_val))

    return leaf_val


@njit(fastmath=True, cache=True)
def _leaf_objective_optimal(g_sum, h_sum, reg_lambda, alpha, max_delta_step):
    """
    Compute the optimal objective value for a leaf node.
    Used in cost-complexity pruning.
    """
    if max_delta_step and max_delta_step > 0.0:
        # When clipping is involved, compute exact objective at optimal value
        v = calc_leaf_value_newton(g_sum, h_sum, reg_lambda, alpha, max_delta_step)
        return g_sum * v + 0.5 * (h_sum + reg_lambda) * v * v + alpha * abs(v)

    # Fast closed-form when no clipping
    denom = h_sum + reg_lambda
    if denom <= 0.0:
        return 0.0

    # Apply soft thresholding for L1 regularization
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
def calc_gain(G_L, H_L, G_R, H_R, reg_lambda, gamma):
    """Calculate split gain."""

    def score(g, h):
        denom = h + reg_lambda
        return (g * g) / denom if denom > 1e-16 else 0.0

    G_total = G_L + G_R
    H_total = H_L + H_R

    gain_left = score(G_L, H_L)
    gain_right = score(G_R, H_R)
    gain_total = score(G_total, H_total)

    return 0.5 * (gain_left + gain_right - gain_total) - gamma


#########################################################
# Pre-binning data
#########################################################

def prebin_data(
    X: np.ndarray,
    bin_edges: np.ndarray,  # shape: (n_features, n_edges_per_feature), uniform per-feature
    n_bins: int,  # TOTAL bins = actual_max_bins + 1 (last is missing)
    use_gpu: bool = False,
) -> np.ndarray:
    # Validate shape
    if bin_edges.ndim != 2 or bin_edges.shape[1] < 2:
        raise ValueError("bin_edges must be 2D with at least 2 edges per feature")

    n_features, n_edges_per_feat = bin_edges.shape

    # Flatten edges row-major for kernels
    edges_flat = np.asarray(bin_edges, dtype=np.float64, order="C").ravel()
    edge_starts = np.arange(0, edges_flat.size, n_edges_per_feat, dtype=np.int32)
    edge_counts = np.full(n_features, n_edges_per_feat, dtype=np.int32)

    # Capacity: total bins includes reserved missing
    actual_max_bins = int(n_bins) - 1
    if actual_max_bins < 1:
        raise ValueError("n_bins must be >= 2 (at least 1 non-missing bin + 1 missing)")

    # The reserved missing id is the last column
    missing_id = actual_max_bins

    # Optional clamp: don’t allow global capacity to exceed per-feature intrinsic capacity
    intrinsic_bins = n_edges_per_feat - 1
    if actual_max_bins > intrinsic_bins:
        actual_max_bins = intrinsic_bins
        missing_id = actual_max_bins  # keep 'missing == last'

    if use_gpu and GPU_AVAILABLE:
        return _prebin_gpu(
            X,
            edges_flat,
            edge_starts,
            edge_counts,
            actual_max_bins,
            missing_id,
        )
    else:
        return _prebin_cpu_kernel(
            X,
            edges_flat,
            edge_starts,
            edge_counts,
            actual_max_bins,
            missing_id,
        )


@njit(parallel=True, fastmath=True, cache=True)
def _prebin_cpu_kernel(
    X: np.ndarray,  # (n_samples, n_features), float64 or float32
    edges_flat: np.ndarray,  # concatenated edges for all features, float64
    edge_starts: np.ndarray,  # start index in edges_flat per feature, int32
    edge_counts: np.ndarray,  # number of edges per feature, int32  (>=2)
    actual_max_bins: int,  # global non-missing capacity
    missing_id: int,  # reserved missing id (usually == actual_max_bins)
) -> np.ndarray:
    n_samples, n_features = X.shape
    B = np.empty((n_samples, n_features), np.int32)

    for f in prange(n_features):
        es = edge_starts[f]
        ec = edge_counts[f]  # number of edges for feature f
        n_edges = ec
        nb = n_edges - 1  # intrinsic non-missing bins for this feature
        if nb < 1:
            # degenerate guard: everything is missing
            for i in range(n_samples):
                B[i, f] = missing_id
            continue

        # respect global cap
        if nb > actual_max_bins:
            nb = actual_max_bins
        last_bin = nb - 1

        # slice view of edges
        # edges[i] <= v < edges[i+1], we search in edges[1:]
        for i in range(n_samples):
            v = X[i, f]
            # missing?
            if not (v == v and -np.inf < v < np.inf):
                B[i, f] = missing_id
                continue

            e0 = edges_flat[es]
            eN = edges_flat[es + n_edges - 1]

            if v <= e0:
                B[i, f] = 0
            elif v > eN:
                B[i, f] = last_bin
            else:
                # binary search in local edges (right array = edges[1:])
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


# -------- GPU: CuPy implementation over flattened edges --------
def _prebin_gpu(
    X: np.ndarray,
    edges_flat: np.ndarray,
    edge_starts: np.ndarray,
    edge_counts: np.ndarray,
    actual_max_bins: int,
    missing_id: int,
) -> np.ndarray:
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    n_samples, n_features = X.shape
    # to GPU (float32 is fine; keep edges in float32 too)
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
        nb = nb if nb <= actual_max_bins else actual_max_bins
        last_bin = nb - 1

        # local edges view for the feature
        el = Ef[es : es + ec]  # edges (len >= 2)
        right = el[1:]  # edges[1:]

        col = Xg[:, f]
        is_finite = cp.isfinite(col)
        # search in right edges (right-closed on left bin)
        bins = cp.searchsorted(right, col, side="left")
        # clamp to capacity
        bins = cp.clip(bins, 0, last_bin)
        # missing assignment
        bins = cp.where(is_finite, bins, missing_id)
        Bg[:, f] = bins

    return Bg.get().astype(np.int32, copy=False)

###########################################################
# Create bin edges using gradient-weighted quantiles
###########################################################

def create_bin_edges(
    X: np.ndarray, gradients: np.ndarray, n_bins: int, use_gpu: bool = False
) -> np.ndarray:
    """Create bin edges using gradient-weighted quantiles."""
    if use_gpu and GPU_AVAILABLE:
        try:
            return _create_bins_gpu(X, gradients, n_bins)
        except Exception as e:
            warnings.warn(f"GPU binning failed: {e}, using CPU")

    return _create_bins_cpu(X, gradients, n_bins)


@njit(parallel=True, fastmath=True, cache=True)
def _create_bins_cpu(X, gradients, n_bins):
    """CPU implementation of gradient-weighted binning."""
    n_samples, n_features = X.shape
    bin_edges = np.empty((n_features, n_bins + 1), dtype=np.float64)
    abs_gradients = np.abs(gradients)

    for f in prange(n_features):
        feature_data = X[:, f]

        # Handle constant features
        min_val, max_val = np.min(feature_data), np.max(feature_data)
        if max_val - min_val < 1e-12:
            bin_edges[f] = np.linspace(min_val - 1e-6, max_val + 1e-6, n_bins + 1)
            continue

        # Filter valid values
        valid_mask = np.isfinite(feature_data)
        if np.sum(valid_mask) < 2:
            bin_edges[f] = np.linspace(min_val, max_val, n_bins + 1)
            continue

        valid_data = feature_data[valid_mask]
        valid_weights = abs_gradients[valid_mask]

        # Sort by feature values
        sorted_idx = np.argsort(valid_data)
        sorted_vals = valid_data[sorted_idx]
        sorted_weights = valid_weights[sorted_idx]

        # Compute cumulative weights
        total_weight = np.sum(sorted_weights)
        if total_weight <= 1e-12:
            bin_edges[f] = np.linspace(min_val, max_val, n_bins + 1)
            continue

        # Create weighted quantiles
        edges = np.empty(n_bins + 1, dtype=np.float64)
        edges[0] = sorted_vals[0] - 1e-6
        edges[-1] = sorted_vals[-1] + 1e-6

        cumsum_weights = np.cumsum(sorted_weights)

        for i in range(1, n_bins):
            target_weight = (i / n_bins) * total_weight
            # Find position by binary search
            idx = np.searchsorted(cumsum_weights, target_weight)
            if idx < len(sorted_vals):
                edges[i] = sorted_vals[idx]
            else:
                edges[i] = sorted_vals[-1]

        bin_edges[f] = edges

    return bin_edges


def _create_bins_gpu(X, gradients, n_bins):
    """GPU implementation of binning (simplified)."""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    # Convert to GPU arrays
    X_gpu = cp.asarray(X, dtype=cp.float32)
    grad_gpu = cp.abs(cp.asarray(gradients, dtype=cp.float32))

    n_features = X.shape[1]
    bin_edges = np.empty((n_features, n_bins + 1), dtype=np.float64)

    for f in range(n_features):
        feat_vals = X_gpu[:, f]

        # Handle constant features
        min_val, max_val = float(cp.min(feat_vals)), float(cp.max(feat_vals))
        if max_val - min_val < 1e-12:
            bin_edges[f] = np.linspace(min_val - 1e-6, max_val + 1e-6, n_bins + 1)
            continue

        # Filter valid values
        valid_mask = cp.isfinite(feat_vals)
        if cp.sum(valid_mask) < 2:
            bin_edges[f] = np.linspace(min_val, max_val, n_bins + 1)
            continue

        valid_vals = feat_vals[valid_mask]
        valid_weights = grad_gpu[valid_mask]

        # Sort and create quantiles
        sorted_idx = cp.argsort(valid_vals)
        sorted_vals = valid_vals[sorted_idx]
        sorted_weights = valid_weights[sorted_idx]

        total_weight = float(cp.sum(sorted_weights))
        if total_weight <= 1e-12:
            bin_edges[f] = np.linspace(min_val, max_val, n_bins + 1)
            continue

        # Compute cumulative weights and quantiles
        cumsum_weights = cp.cumsum(sorted_weights)
        quantile_positions = cp.linspace(0, total_weight, n_bins + 1)

        # Interpolate to find bin edges
        edges = cp.interp(quantile_positions, cumsum_weights, sorted_vals)
        edges[0] = sorted_vals[0] - 1e-6
        edges[-1] = sorted_vals[-1] + 1e-6

        bin_edges[f] = edges.get().astype(np.float64)

    return bin_edges

##########################################################
# Find best split with missing value handling
##########################################################

try:
    import cupy as cp
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

@njit(fastmath=True, cache=True)
def find_best_split_with_missing(
    hist_g,
    hist_h,
    g_miss_arr,
    h_miss_arr,
    reg_lambda,
    gamma,
    n_bins,
    min_child_weight=1e-6,
    monotone_constraints=None,
):
    """
    Find the best split from histograms with explicit missing value handling.

    Args:
        hist_g, hist_h: Histogram arrays (n_features, max_bins)
        g_miss_arr, h_miss_arr: Missing value statistics per feature (n_features,)
        reg_lambda, gamma: Regularization parameters
        n_bins: Number of bins to consider per feature
        min_child_weight: Minimum child weight constraint
        monotone_constraints: Monotone constraints per feature
    """
    n_features = hist_g.shape[0]
    best_feat, best_bin, best_gain, best_missing = -1, -1, -np.inf, 0

    for f in range(n_features):
        # Get monotone constraint
        constraint = 0
        if monotone_constraints is not None and f < len(monotone_constraints):
            constraint = monotone_constraints[f]

        # Get missing statistics for this feature
        G_miss = g_miss_arr[f] if f < len(g_miss_arr) else 0.0
        H_miss = h_miss_arr[f] if f < len(h_miss_arr) else 0.0

        # Compute total statistics from finite bins only
        G_finite = 0.0
        H_finite = 0.0
        max_valid_bin = min(n_bins, hist_g.shape[1])

        for b in range(max_valid_bin):
            G_finite += hist_g[f, b]
            H_finite += hist_h[f, b]

        # Total including missing
        G_tot = G_finite + G_miss
        H_tot = H_finite + H_miss

        if H_tot < 2 * min_child_weight:
            continue

        # Compute prefix sums for finite values
        G_prefix = np.zeros(max_valid_bin, dtype=numba.float64)
        H_prefix = np.zeros(max_valid_bin, dtype=numba.float64)

        if max_valid_bin > 0:
            G_prefix[0] = hist_g[f, 0]
            H_prefix[0] = hist_h[f, 0]
            for b in range(1, max_valid_bin):
                G_prefix[b] = G_prefix[b - 1] + hist_g[f, b]
                H_prefix[b] = H_prefix[b - 1] + hist_h[f, b]

        # Evaluate splits
        for split_bin in range(1, max_valid_bin):
            G_Lp = G_prefix[split_bin - 1]  # Left without missing
            H_Lp = H_prefix[split_bin - 1]
            G_Rp = G_finite - G_Lp  # Right without missing
            H_Rp = H_finite - H_Lp

            # Skip if base split is invalid
            if H_Lp < min_child_weight or H_Rp < min_child_weight:
                continue

            # Try both missing directions
            for missing_left in [True, False]:
                if missing_left:
                    G_L, H_L = G_Lp + G_miss, H_Lp + H_miss
                    G_R, H_R = G_Rp, H_Rp
                    go_left = 1
                else:
                    G_L, H_L = G_Lp, H_Lp
                    G_R, H_R = G_Rp + G_miss, H_Rp + H_miss
                    go_left = 0

                # Check child weight constraints
                if H_L < min_child_weight or H_R < min_child_weight:
                    continue

                gain = calc_gain(G_L, H_L, G_R, H_R, reg_lambda, gamma)

                if gain <= best_gain:
                    continue

                # Check monotone constraint
                if constraint != 0 and H_L > 1e-16 and H_R > 1e-16:
                    w_L = -G_L / (H_L + reg_lambda)
                    w_R = -G_R / (H_R + reg_lambda)
                    if constraint == 1 and w_L > w_R + 1e-10:
                        continue
                    elif constraint == -1 and w_L < w_R - 1e-10:
                        continue

                best_gain = gain
                best_feat = f
                best_bin = split_bin
                best_missing = go_left

    return best_feat, best_bin, best_gain, best_missing




#############################################################
# Pre-binning kernel for single feature (for testing)
##############################################################

@njit(fastmath=True, cache=True)
def prebin_kernel(
    values: np.ndarray,
    edges: np.ndarray,  # strictly increasing, len >= 2
    nb: int,  # non-missing bins = len(edges) - 1, capped
    miss_id: int,
) -> np.ndarray:
    """
    Return int32 bin ids for 'values' w.r.t. 'edges'. Missing -> miss_id.
    Non-missing bins are 0..nb-1; last id (miss_id) is reserved for missing.
    """
    out = np.empty(values.shape[0], np.int32)
    # search on right edges (edges[1:]); NaNs will be overwritten to miss_id
    idx = np.searchsorted(edges[1:], values, side="left")
    if nb > 1:
        # clip into [0, nb-1]
        for i in range(idx.size):
            if idx[i] < 0:
                idx[i] = 0
            elif idx[i] >= nb:
                idx[i] = nb - 1
    # assign + missing handling
    for i in range(idx.size):
        vi = values[i]
        if np.isfinite(vi):
            out[i] = idx[i]
        else:
            out[i] = miss_id
    return out

##############################################################
# Tree prediction with missing value handling
##############################################################

@njit
def predict_tree_numba_with_missing_mask(
    X,
    missing_mask,
    node_features,
    node_thresholds,
    node_missing_go_left,
    left_children,
    right_children,
    leaf_values,
    is_leaf_flags,
    feature_map_array,
    root_idx=0,
):
    n_samples = X.shape[0]
    predictions = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        node_idx = root_idx
        for _ in range(100):  # max depth guard
            if node_idx < 0 or node_idx >= len(is_leaf_flags):
                predictions[i] = 0.0
                break
            if is_leaf_flags[node_idx]:
                predictions[i] = leaf_values[node_idx]
                break

            global_feat_idx = node_features[node_idx]
            if global_feat_idx < 0 or global_feat_idx >= len(feature_map_array):
                predictions[i] = 0.0
                break

            local_feat_idx = feature_map_array[global_feat_idx]
            if local_feat_idx < 0 or local_feat_idx >= X.shape[1]:
                predictions[i] = 0.0
                break

            x = X[i, local_feat_idx]
            thr = node_thresholds[node_idx]

            # Use precomputed missing mask
            go_left = (
                node_missing_go_left[node_idx]
                if missing_mask[i, local_feat_idx]
                else (x <= thr)
            )
            node_idx = left_children[node_idx] if go_left else right_children[node_idx]
        else:
            predictions[i] = 0.0
    return predictions

@njit(cache=True, fastmath=True)
def predict_tree_binned_with_missingbin(
    Xb: np.ndarray,  # (n_samples, n_features) int bins; last id = missing_bin_id
    missing_bin_id: int,  # == self._actual_max_bins
    nf: np.ndarray,  # node_features (global feature id) shape (N,)
    nbin: np.ndarray,  # node_bin_idx (split bin, -1 if not using bins)
    nm: np.ndarray,  # node_missing_go_left (bool) shape (N,)
    lc: np.ndarray,  # left_children (int) shape (N,)
    rc: np.ndarray,  # right_children (int) shape (N,)
    lv: np.ndarray,  # leaf_values (float) shape (N,)
    is_leaf_flags: np.ndarray,  # bool shape (N,)
    fmap: np.ndarray,  # global->local feature index map (int, -1 if absent)
    root_id: int,  # id of root node
) -> np.ndarray:
    n_samples = Xb.shape[0]
    out = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        nid = root_id
        while nid >= 0 and not is_leaf_flags[nid]:
            gfi = nf[nid]
            # If no feature recorded, bail to leaf value
            if gfi < 0:
                break
            lfi = -1
            if gfi < fmap.size:
                lfi = fmap[gfi]

            # Default: if mapping invalid, treat as "missing" and follow learned direction
            go_left = nm[nid]
            if lfi >= 0 and lfi < Xb.shape[1]:
                b = int(Xb[i, lfi])
                if b != missing_bin_id:
                    # Compare real bin to stored split bin
                    split_bin = nbin[nid]
                    # If nbin is invalid, fall back to missing policy (rare)
                    if split_bin >= 0:
                        go_left = b <= split_bin

            nid = lc[nid] if go_left else rc[nid]
            if nid < 0:
                break

        # If we fell off or hit a leaf, emit value
        if nid >= 0:
            out[i] = lv[nid] if is_leaf_flags[nid] else lv[root_id]
        else:
            out[i] = 0.0  # conservative fallback

    return out

@njit
def predict_tree_numba(
    X,
    node_features,
    node_thresholds,
    node_missing_left,
    left_children,
    right_children,
    leaf_values,
    is_leaf,
    feature_map,
    root_idx=0,
):
    """Predict using continuous data (non-binned version)."""
    n_samples = X.shape[0]
    predictions = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        node = root_idx

        # Traverse tree (with depth limit for safety)
        for _ in range(100):
            if node < 0 or node >= len(is_leaf) or is_leaf[node]:
                predictions[i] = leaf_values[node] if node >= 0 else 0.0
                break

            # Get feature and threshold
            global_feat = node_features[node]
            if global_feat < 0 or global_feat >= len(feature_map):
                predictions[i] = 0.0
                break

            local_feat = feature_map[global_feat]
            if local_feat < 0 or local_feat >= X.shape[1]:
                predictions[i] = 0.0
                break

            feat_val = X[i, local_feat]
            threshold = node_thresholds[node]

            # Determine direction
            if not (feat_val == feat_val):  # NaN check
                go_left = node_missing_left[node]
            else:
                go_left = feat_val < threshold

            node = left_children[node] if go_left else right_children[node]
        else:
            predictions[i] = 0.0

    return predictions

###############################################################
# Best split on single feature with missing value handling
###############################################################

@njit
def best_split_on_feature_list(
    sorted_ids_for_g,
    v_seq,
    grad,
    hess,
    g_miss,
    h_miss,
    n_miss,
    min_samples_leaf,
    min_child_weight,
    lambda_reg,
    gamma,
    monotone_constraint=0,
):
    """Find best split on a single feature using sorted sample order."""
    m = sorted_ids_for_g.size
    if m < 2 * min_samples_leaf:
        return -1.0, 0.0, False, 0, 0

    # Gather gradients/hessians in feature-sorted order
    g_seq = np.empty(m, dtype=np.float64)
    h_seq = np.empty(m, dtype=np.float64)
    for i in range(m):
        idx = sorted_ids_for_g[i]
        g_seq[i] = grad[idx]
        h_seq[i] = hess[idx]

    # Compute prefix sums
    for i in range(1, m):
        g_seq[i] += g_seq[i - 1]
        h_seq[i] += h_seq[i - 1]

    total_g = g_seq[m - 1] + g_miss
    total_h = h_seq[m - 1] + h_miss
    parent_score = (
        (total_g * total_g) / (total_h + lambda_reg)
        if (total_h + lambda_reg) > 0.0
        else 0.0
    )

    best_gain = -1.0
    best_thr = 0.0
    best_mleft = False
    best_nL = 0
    best_nR = 0

    prev = v_seq[0]
    for i in range(1, m):
        cur = v_seq[i]
        if cur <= prev:
            continue  # Skip duplicate values

        nL_f = i
        nR_f = m - i
        if nL_f < min_samples_leaf or nR_f < min_samples_leaf:
            prev = cur
            continue

        gL_f = g_seq[i - 1]
        hL_f = h_seq[i - 1]
        gR_f = g_seq[m - 1] - gL_f
        hR_f = h_seq[m - 1] - hL_f

        # Try both missing directions
        for mleft in (True, False):
            if mleft:
                gL = gL_f + g_miss
                hL = hL_f + h_miss
                nL = nL_f + n_miss
                gR = gR_f
                hR = hR_f
                nR = nR_f
            else:
                gL = gL_f
                hL = hL_f
                nL = nL_f
                gR = gR_f + g_miss
                hR = hR_f + h_miss
                nR = nR_f + n_miss

            if (
                hL < min_child_weight
                or hR < min_child_weight
                or nL < min_samples_leaf
                or nR < min_samples_leaf
            ):
                continue

            left_score = (
                (gL * gL) / (hL + lambda_reg) if (hL + lambda_reg) > 0.0 else 0.0
            )
            right_score = (
                (gR * gR) / (hR + lambda_reg) if (hR + lambda_reg) > 0.0 else 0.0
            )
            gain = 0.5 * (left_score + right_score - parent_score) - gamma

            if gain <= best_gain:
                continue

            # Check monotone constraint
            if monotone_constraint != 0:
                wL = -gL / (hL + lambda_reg) if (hL + lambda_reg) > 0.0 else 0.0
                wR = -gR / (hR + lambda_reg) if (hR + lambda_reg) > 0.0 else 0.0
                if monotone_constraint * (wR - wL) < 0:
                    continue

            best_gain = gain
            best_thr = 0.5 * (prev + cur)
            best_mleft = mleft
            best_nL = nL
            best_nR = nR

        prev = cur

    return best_gain, best_thr, best_mleft, best_nL, best_nR

##################################################################

@njit(fastmath=True, cache=True)
def subtract_sibling_histograms(
    parent_hist_g, parent_hist_h, sibling_hist_g, sibling_hist_h
):
    """Fast sibling subtraction for histogram computation."""
    result_g = np.zeros_like(parent_hist_g)
    result_h = np.zeros_like(parent_hist_h)

    for i in range(parent_hist_g.shape[0]):
        for j in range(parent_hist_g.shape[1]):
            result_g[i, j] = parent_hist_g[i, j] - sibling_hist_g[i, j]
            result_h[i, j] = parent_hist_h[i, j] - sibling_hist_h[i, j]

    return result_g, result_h

# ================================================================
# Utility classes
# ================================================================

class FeatureImportance:
    def __init__(self):
        self._store = {
            "gain": defaultdict(float),
            "cover": defaultdict(float),
            "split": defaultdict(float),
            # model-agnostic / neural-friendly:
            "perm": defaultdict(float),
            "input_grad": defaultdict(float),
        }

    def add(self, importance_type: str, feat_idx: int, value: float):
        self._store[importance_type][int(feat_idx)] += float(value)

    def add_split(self, feat_idx: int, gain: float, cover: float):
        self._store["gain"][int(feat_idx)]  += float(gain)
        self._store["cover"][int(feat_idx)] += float(cover)
        self._store["split"][int(feat_idx)] += 1.0

    def get_importance(self, importance_type: str) -> dict:
        return dict(self._store.get(importance_type, {}))
    
class HistogramCache:
    """Simple LRU cache for histograms."""

    def __init__(self, max_size=1000):
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
            else:
                self.misses += 1
                return None

    def put(self, key, hist_g, hist_h):
        with self._lock:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = (hist_g, hist_h)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ================================================================
# GPU utilities (simplified)
# ================================================================

