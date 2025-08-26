import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import RLock
from typing import Optional, Tuple

# ================================================================
# Optimized computations with enhanced numerical stability
# ================================================================
import numpy as np
from numba import njit, prange

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@njit
def sum_missing_in_node(miss_idx: np.ndarray,
                        mark_node: np.ndarray,  # uint8 bitset length n_samples
                        g: np.ndarray,
                        h: np.ndarray):
    g_m = 0.0
    h_m = 0.0
    n_m = 0
    for k in range(miss_idx.size):
        idx = miss_idx[k]
        if mark_node[idx] == 1:
            g_m += g[idx]
            h_m += h[idx]
            n_m += 1
    return g_m, h_m, n_m


@njit
def best_split_on_feature_list(sorted_ids_for_g: np.ndarray,  # ids in node, sorted by feature g
                               v_seq: np.ndarray,             # X[sorted_ids_for_g, lf] (float64, non-decreasing)
                               grad: np.ndarray,
                               hess: np.ndarray,
                               g_miss: float, h_miss: float, n_miss: int,
                               min_samples_leaf: int, min_child_weight: float,
                               lambda_reg: float, gamma: float,
                               monotone_constraint: int = 0):
    m = sorted_ids_for_g.size
    if m < 2 * min_samples_leaf:
        return -1.0, 0.0, False, 0, 0

    # gather grads/hess in feature-sorted order for this node
    g_seq = np.empty(m, dtype=np.float64)
    h_seq = np.empty(m, dtype=np.float64)
    for i in range(m):
        idx = sorted_ids_for_g[i]
        g_seq[i] = grad[idx]
        h_seq[i] = hess[idx]

    # prefix sums
    for i in range(1, m):
        g_seq[i] += g_seq[i - 1]
        h_seq[i] += h_seq[i - 1]

    total_g = g_seq[m - 1] + g_miss
    total_h = h_seq[m - 1] + h_miss
    parent = (total_g * total_g) / (total_h + lambda_reg) if (total_h + lambda_reg) > 0.0 else 0.0

    best_gain = -1.0
    best_thr = 0.0
    best_mleft = False
    best_nL = 0
    best_nR = 0

    prev = v_seq[0]
    for i in range(1, m):
        cur = v_seq[i]
        if cur <= prev:
            continue  # compress runs of equal values

        nL_f = i
        nR_f = m - i
        if nL_f < min_samples_leaf or nR_f < min_samples_leaf:
            prev = cur
            continue

        gL_f = g_seq[i - 1]
        hL_f = h_seq[i - 1]
        gR_f = g_seq[m - 1] - gL_f
        hR_f = h_seq[m - 1] - hL_f

        # try both missing directions
        for mleft in (True, False):
            if mleft:
                gL = gL_f + g_miss; hL = hL_f + h_miss; nL = nL_f + n_miss
                gR = gR_f;         hR = hR_f;           nR = nR_f
            else:
                gL = gL_f;         hL = hL_f;           nL = nL_f
                gR = gR_f + g_miss; hR = hR_f + h_miss; nR = nR_f + n_miss

            if hL < min_child_weight or hR < min_child_weight or nL < min_samples_leaf or nR < min_samples_leaf:
                continue

            left  = (gL * gL) / (hL + lambda_reg) if (hL + lambda_reg) > 0.0 else 0.0
            right = (gR * gR) / (hR + lambda_reg) if (hR + lambda_reg) > 0.0 else 0.0
            gain = 0.5 * (left + right - parent) - gamma
            if gain <= best_gain:
                continue

            if monotone_constraint != 0:
                wL = -gL / (hL + lambda_reg) if (hL + lambda_reg) > 0.0 else 0.0
                wR = -gR / (hR + lambda_reg) if (hR + lambda_reg) > 0.0 else 0.0
                if monotone_constraint * (wR - wL) < 0:
                    continue

            best_gain = gain
            best_thr  = 0.5 * (prev + cur)
            best_mleft = mleft
            best_nL = nL
            best_nR = nR

        prev = cur

    return best_gain, best_thr, best_mleft, best_nL, best_nR


def _soft_threshold(g: float, alpha: float) -> float:
    if alpha <= 0.0:
        return g
    if g > alpha:  return g - alpha
    if g < -alpha: return g + alpha
    return 0.0

def _leaf_value_from_sums(g_sum: float, h_sum: float, reg_lambda: float, alpha: float, max_delta_step: float) -> float:
    return float(calc_leaf_value_newton(g_sum, h_sum, reg_lambda, alpha, max_delta_step))

def _leaf_objective_from_value(g_sum: float, h_sum: float, reg_lambda: float, alpha: float, v: float) -> float:
    # Exact objective at chosen v (works with/without clipping)
    return float(g_sum * v + 0.5 * (h_sum + reg_lambda) * v * v + alpha * abs(v))

def _leaf_objective_optimal(g_sum: float, h_sum: float, reg_lambda: float, alpha: float, max_delta_step: float) -> float:
    # Fast closed-form when not clipped; falls back to exact-at-v if clipping in play
    if max_delta_step and max_delta_step > 0.0:
        v = _leaf_value_from_sums(g_sum, h_sum, reg_lambda, alpha, max_delta_step)
        return _leaf_objective_from_value(g_sum, h_sum, reg_lambda, alpha, v)
    denom = h_sum + reg_lambda
    if denom <= 0.0:
        return 0.0
    gsh = _soft_threshold(g_sum, alpha)
    return -0.5 * (gsh * gsh) / denom




@njit
def find_best_split_feature_exact(
    sorted_indices: np.ndarray,
    values: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    node_membership: np.ndarray,
    missing_indices: np.ndarray,
    node_id: int,
    min_samples_leaf: int,
    min_child_weight: float,
    lambda_reg: float,
    gamma: float,
    monotone_constraint: int = 0,
) -> Tuple[float, float, bool, int, int]:
    """
    Numba-optimized split finding for a single feature.
    Returns: (best_gain, best_threshold, best_missing_left, n_left, n_right)
    """
    # Filter indices to current node
    node_samples = []
    for idx in sorted_indices:
        if node_membership[idx] == node_id:
            node_samples.append(idx)

    if len(node_samples) <= min_samples_leaf:
        return -1.0, 0.0, False, 0, 0

    node_samples = np.array(node_samples)
    n_samples = len(node_samples)

    # Get sorted values and gradients/hessians for this node
    sorted_vals = values[node_samples]
    sorted_grads = gradients[node_samples]
    sorted_hess = hessians[node_samples]

    # Handle missing values for this node
    missing_g = 0.0
    missing_h = 0.0
    n_missing = 0
    for idx in missing_indices:
        if node_membership[idx] == node_id:
            missing_g += gradients[idx]
            missing_h += hessians[idx]
            n_missing += 1

    # Precompute cumulative sums
    cum_g = np.cumsum(sorted_grads)
    cum_h = np.cumsum(sorted_hess)
    total_g = cum_g[-1]
    total_h = cum_h[-1]

    # Parent score
    parent_score = (
        (total_g + missing_g) ** 2 / (total_h + missing_h + lambda_reg)
        if (total_h + missing_h + lambda_reg) > 0
        else 0.0
    )

    best_gain = -1.0
    best_threshold = 0.0
    best_missing_left = False
    best_n_left = 0
    best_n_right = 0

    # Find unique split points more efficiently
    prev_val = sorted_vals[0]
    for i in range(1, n_samples):
        curr_val = sorted_vals[i]

        # Only consider splits between different values
        if curr_val <= prev_val:
            continue

        # Left side: samples [0, i)
        n_left_finite = i
        n_right_finite = n_samples - i

        if n_left_finite < min_samples_leaf or n_right_finite < min_samples_leaf:
            prev_val = curr_val
            continue

        g_left_finite = cum_g[i - 1]
        h_left_finite = cum_h[i - 1]
        g_right_finite = total_g - g_left_finite
        h_right_finite = total_h - h_left_finite

        # Try both missing directions
        for missing_left in [True, False]:
            if missing_left:
                g_left = g_left_finite + missing_g
                h_left = h_left_finite + missing_h
                g_right = g_right_finite
                h_right = h_right_finite
                n_left = n_left_finite + n_missing
                n_right = n_right_finite
            else:
                g_left = g_left_finite
                h_left = h_left_finite
                g_right = g_right_finite + missing_g
                h_right = h_right_finite + missing_h
                n_left = n_left_finite
                n_right = n_right_finite + n_missing

            if (
                h_left < min_child_weight
                or h_right < min_child_weight
                or n_left < min_samples_leaf
                or n_right < min_samples_leaf
            ):
                continue

            # Calculate gain
            left_score = (
                g_left**2 / (h_left + lambda_reg) if (h_left + lambda_reg) > 0 else 0.0
            )
            right_score = (
                g_right**2 / (h_right + lambda_reg)
                if (h_right + lambda_reg) > 0
                else 0.0
            )
            gain = 0.5 * (left_score + right_score - parent_score) - gamma

            if gain <= best_gain:
                continue

            # Monotone constraint check
            if monotone_constraint != 0:
                left_pred = (
                    -g_left / (h_left + lambda_reg)
                    if (h_left + lambda_reg) > 0
                    else 0.0
                )
                right_pred = (
                    -g_right / (h_right + lambda_reg)
                    if (h_right + lambda_reg) > 0
                    else 0.0
                )
                if monotone_constraint * (right_pred - left_pred) < 0:
                    continue

            best_gain = gain
            best_threshold = 0.5 * (prev_val + curr_val)
            best_missing_left = missing_left
            best_n_left = n_left
            best_n_right = n_right

        prev_val = curr_val

    return best_gain, best_threshold, best_missing_left, best_n_left, best_n_right


@njit
def update_node_membership_exact(
    indices: np.ndarray,
    values: np.ndarray,
    node_membership: np.ndarray,
    old_node_id: int,
    left_node_id: int,
    right_node_id: int,
    threshold: float,
    missing_left: bool,
):
    """Numba-optimized node membership update."""
    for idx in indices:
        if node_membership[idx] == old_node_id:
            val = values[idx]
            if np.isnan(val):
                node_membership[idx] = left_node_id if missing_left else right_node_id
            else:
                node_membership[idx] = (
                    left_node_id if val <= threshold else right_node_id
                )


@njit(fastmath=True, cache=True)
def calc_leaf_value_newton(g_sum, h_sum, reg_lambda, alpha=0.0, max_delta_step=0.0):
    """
    Optimized leaf value computation with enhanced numerical stability.
    XGBoost-style Newton step: w* = - g_sum / (h_sum + lambda)
    
    Improvements:
    - Better numerical stability for edge cases
    - Optimized L1 regularization
    - Cached compilation for repeated calls
    """
    # Early exit for degenerate cases
    if h_sum <= 1e-16:  # More robust than 0.0 check
        return 0.0
    
    denom = h_sum + reg_lambda
    
    if alpha > 0.0:  # L1 regularization (soft-thresholding)
        # Optimized soft thresholding with single division
        if g_sum > alpha:
            leaf_val = -(g_sum - alpha) / denom
        elif g_sum < -alpha:
            leaf_val = -(g_sum + alpha) / denom
        else:
            leaf_val = 0.0
    else:
        leaf_val = -g_sum / denom

    # Apply max_delta_step constraint (vectorizable clipping)
    if max_delta_step > 0.0:
        leaf_val = max(-max_delta_step, min(max_delta_step, leaf_val))

    return leaf_val


@njit(fastmath=True, inline='always', cache=True)
def gain_score(g, h, reg_lambda):
    """
    Optimized gain contribution with enhanced numerical stability.
    Inlined for maximum performance in tight loops.
    """
    denom = h + reg_lambda
    if denom <= 1e-16:  # Robust denominator check
        return 0.0
    return (g * g) / denom


@njit(fastmath=True, cache=True)
def calc_gain(G_L, H_L, G_R, H_R, reg_lambda, gamma):
    """
    Optimized gain calculation with better numerical properties.
    Uses inline gain_score for maximum efficiency.
    """
    # Pre-compute total for numerical stability
    G_total = G_L + G_R
    H_total = H_L + H_R
    
    # Vectorized gain computation
    gain_left = gain_score(G_L, H_L, reg_lambda)
    gain_right = gain_score(G_R, H_R, reg_lambda)
    gain_total = gain_score(G_total, H_total, reg_lambda)
    
    return 0.5 * (gain_left + gain_right - gain_total) - gamma


# ================================================================
# Enhanced pre-binning utilities with better memory management
# ================================================================
def prebin_edges(X: np.ndarray, grad: np.ndarray, n_bins: int, gpu: bool = False) -> np.ndarray:
    """
    Enhanced gradient-weighted quantiles with automatic fallback.
    Improved error handling and memory efficiency.
    """
    try:
        if gpu and GPU_AVAILABLE:
            return build_bins_gpu(X, grad, n_bins)
        else:
            return build_bins_cpu(X, grad, n_bins)
    except Exception:
        # Fallback to simple quantile-based binning
        return build_bins_fallback(X, n_bins)


@njit(parallel=True, fastmath=True, cache=True)
def prebin_data(X: np.ndarray, bin_edges: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Highly optimized continuous-to-discrete mapping with:
    - Enhanced parallel processing
    - Optimized binary search
    - Better missing value handling
    - Memory-efficient implementation
    """
    n_samples, n_features = X.shape
    B = np.empty((n_samples, n_features), dtype=np.int32)
    missing_bin = n_bins - 1
    max_regular_bin = n_bins - 2

    for f in prange(n_features):  # Parallel over features for better cache locality
        edges = bin_edges[f]
        n_edges = len(edges)
        
        for i in range(n_samples):
            val = X[i, f]
            
            # Fast missing value check (NaN/inf handling)
            if not (val == val and -np.inf < val < np.inf):  # Efficient NaN/inf check
                B[i, f] = missing_bin
                continue
            
            # Optimized binary search with early bounds checking
            if val <= edges[1]:  # Below first real bin
                B[i, f] = 0
            elif val > edges[n_edges - 2]:  # Above last real bin
                B[i, f] = max_regular_bin
            else:
                # Binary search in valid range [1, n_edges-1)
                lo, hi = 1, n_edges - 1
                while lo < hi:
                    mid = (lo + hi) >> 1  # Bit shift for faster division
                    if val > edges[mid]:
                        lo = mid + 1
                    else:
                        hi = mid
                B[i, f] = min(lo - 1, max_regular_bin)

    return B


@njit(fastmath=True)
def build_bins_fallback(X: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Fallback binning strategy using simple quantiles.
    Used when gradient-weighted binning fails.
    """
    n_features = X.shape[1]
    bin_edges = np.empty((n_features, n_bins + 1), dtype=np.float64)
    
    for f in range(n_features):
        feature_data = X[:, f]
        # Remove invalid values for quantile computation
        valid_mask = np.isfinite(feature_data)
        if np.sum(valid_mask) > 0:
            valid_data = feature_data[valid_mask]
            # Create quantile-based bins
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges[f] = np.percentile(valid_data, quantiles * 100)
        else:
            # All invalid data - create dummy bins
            bin_edges[f] = np.linspace(0, 1, n_bins + 1)
    
    return bin_edges


# ================================================================
# Ultra-optimized histogram computation
# ================================================================
@njit(parallel=True, fastmath=True, cache=True)
def compute_histograms_from_binned(
    binned_local: np.ndarray,  # (n_sub, n_features), int32/int64
    local_indices: np.ndarray,  # (n_node,), int64
    grad: np.ndarray,  # (n_node,), float64  
    hess: np.ndarray,  # (n_node,), float64
    n_features: int,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ultra-optimized histogram computation with:
    - Enhanced parallel processing strategy
    - Better memory access patterns
    - Reduced atomic operations
    - Cache-friendly data layout
    """
    hist_g = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_h = np.zeros((n_features, n_bins), dtype=np.float64)
    
    n_samples = len(local_indices)
    
    # Strategy: parallelize over samples when many samples, over features when few samples
    if n_samples > n_features * 4:  # Sample-parallel strategy
        for j in prange(n_samples):
            idx = local_indices[j]
            g_val = grad[j]
            h_val = hess[j]
            
            # Vectorized update across all features for this sample
            for f in range(n_features):
                bin_idx = binned_local[idx, f]
                if 0 <= bin_idx < n_bins:  # Bounds check
                    hist_g[f, bin_idx] += g_val
                    hist_h[f, bin_idx] += h_val
    else:  # Feature-parallel strategy  
        for f in prange(n_features):
            for j in range(n_samples):
                idx = local_indices[j]
                bin_idx = binned_local[idx, f]
                if 0 <= bin_idx < n_bins:  # Bounds check
                    hist_g[f, bin_idx] += grad[j]
                    hist_h[f, bin_idx] += hess[j]

    return hist_g, hist_h


# ================================================================
# Enhanced split finding with advanced optimizations
# ================================================================
@njit(fastmath=True, cache=True)
def find_best_splits_with_missing(
    hist_g,
    hist_h,
    reg_lambda,
    gamma,
    n_bins,
    min_child_weight=1e-6,
    monotone_constraints=None,
):
    """
    Highly optimized split finding with:
    - Early termination conditions
    - Vectorized gain computations
    - Enhanced monotone constraint handling
    - Better numerical stability
    - Optimized missing value handling
    """
    n_features, actual_n_bins = hist_g.shape
    miss_bin = n_bins - 1
    
    best_feat, best_bin, best_gain, best_missing = -1, -1, -np.inf, 0
    
    # Pre-compute constants for efficiency
    gamma_threshold = gamma + 1e-12  # Small epsilon for numerical stability
    
    for f in range(n_features):
        g_row = hist_g[f]
        h_row = hist_h[f]
        
        # Total statistics
        G_tot = 0.0
        H_tot = 0.0
        for b in range(actual_n_bins):
            G_tot += g_row[b]
            H_tot += h_row[b]
        
        # Early termination: insufficient total weight
        if H_tot < 2 * min_child_weight:
            continue
        
        G_miss, H_miss = g_row[miss_bin], h_row[miss_bin]
        
        # Pre-compute prefix sums for valid bins (excluding missing)
        max_valid_bin = min(miss_bin, actual_n_bins)
        G_prefix = np.zeros(max_valid_bin, dtype=np.float64)
        H_prefix = np.zeros(max_valid_bin, dtype=np.float64)
        
        # Efficient prefix sum computation
        if max_valid_bin > 0:
            G_prefix[0] = g_row[0]
            H_prefix[0] = h_row[0]
            for b in range(1, max_valid_bin):
                G_prefix[b] = G_prefix[b-1] + g_row[b]
                H_prefix[b] = H_prefix[b-1] + h_row[b]
        
        # Get monotone constraint for this feature
        constraint = 0
        if monotone_constraints is not None and f < len(monotone_constraints):
            constraint = monotone_constraints[f]
        
        # Vectorized split evaluation
        for split_bin in range(1, max_valid_bin):
            # Left side statistics (non-missing)
            G_Lp = G_prefix[split_bin - 1]
            H_Lp = H_prefix[split_bin - 1]
            
            # Right side statistics (non-missing) 
            G_Rp = G_tot - G_miss - G_Lp
            H_Rp = H_tot - H_miss - H_Lp
            
            # Early termination: insufficient weight in children
            if H_Lp < min_child_weight or H_Rp < min_child_weight:
                continue
            
            # === Missing values go LEFT ===
            G_L1 = G_Lp + G_miss
            H_L1 = H_Lp + H_miss
            G_R1 = G_Rp
            H_R1 = H_Rp
            
            # === Missing values go RIGHT ===
            G_L2 = G_Lp  
            H_L2 = H_Lp
            G_R2 = G_Rp + G_miss
            H_R2 = H_Rp + H_miss
            
            # Compute gains for both directions
            gain_left = calc_gain(G_L1, H_L1, G_R1, H_R1, reg_lambda, gamma)
            gain_right = calc_gain(G_L2, H_L2, G_R2, H_R2, reg_lambda, gamma)
            
            # Choose better direction
            if gain_left >= gain_right:
                gain, go_left = gain_left, 1
                G_Lc, H_Lc, G_Rc, H_Rc = G_L1, H_L1, G_R1, H_R1
            else:
                gain, go_left = gain_right, 0
                G_Lc, H_Lc, G_Rc, H_Rc = G_L2, H_L2, G_R2, H_R2
            
            # Early termination: gain too small
            if gain <= gamma_threshold:
                continue
            
            # === Enhanced monotone constraint checking ===
            if constraint != 0 and H_Lc > 1e-16 and H_Rc > 1e-16:
                # Compute leaf values for constraint validation
                w_L = -G_Lc / (H_Lc + reg_lambda)
                w_R = -G_Rc / (H_Rc + reg_lambda)
                
                # Check constraint with numerical tolerance
                tolerance = 1e-10
                if constraint == 1:  # Increasing: left ≤ right
                    if w_L > w_R + tolerance:
                        continue
                elif constraint == -1:  # Decreasing: left ≥ right  
                    if w_L < w_R - tolerance:
                        continue
            
            # === Update best split ===
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_bin = split_bin
                best_missing = go_left

    return best_feat, best_bin, best_gain, best_missing


# ================================================================
# Optimized histogram operations
# ================================================================
@njit(fastmath=True, cache=True)
def subtract_histograms(parent_hist_g, parent_hist_h, child_hist_g, child_hist_h):
    """
    Optimized sibling histogram subtraction with validation.
    Enhanced version of LightGBM's histogram subtraction trick.
    """
    # Vectorized subtraction
    sibling_g = parent_hist_g - child_hist_g
    sibling_h = parent_hist_h - child_hist_h
    
    # Optional: Validate non-negative hessians (can be disabled for speed)
    # This helps catch numerical errors in histogram computation
    n_features, n_bins = sibling_h.shape
    for f in range(n_features):
        for b in range(n_bins):
            if sibling_h[f, b] < -1e-10:  # Allow small numerical errors
                sibling_h[f, b] = 0.0
                sibling_g[f, b] = 0.0
    
    return sibling_g, sibling_h



# ================ FEATURE IMPORTANCE ================
class FeatureImportance:
    def __init__(self):
        self.gain_importance = {}
        self.cover_importance = {}
        self.frequency_importance = {}

    def update(self, feature_idx, gain, cover):
        if feature_idx not in self.gain_importance:
            self.gain_importance[feature_idx] = 0
            self.cover_importance[feature_idx] = 0
            self.frequency_importance[feature_idx] = 0

        self.gain_importance[feature_idx] += gain
        self.cover_importance[feature_idx] += cover
        self.frequency_importance[feature_idx] += 1

    def get_importance(self, importance_type="gain"):
        if importance_type == "gain":
            return self.gain_importance
        elif importance_type == "cover":
            return self.cover_importance
        elif importance_type == "frequency":
            return self.frequency_importance
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")


class HistogramCache:
    """High-performance histogram cache with memory + size constraints."""

    def __init__(self, max_cache_size=2000, max_memory_mb=200, copy_on_put=False):
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.copy_on_put = copy_on_put
        self.memory_usage = 0
        self._lock = RLock()

        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, node_id):
        with self._lock:
            try:
                value = self.cache[node_id]
                self.cache.move_to_end(node_id)  # O(1), marks as recently used
                self.hits += 1
                return value
            except KeyError:
                self.misses += 1
                return None

    def put(self, node_id, hist_g, hist_h):
        hist_size = hist_g.nbytes + hist_h.nbytes
        if self.copy_on_put:
            hist_g, hist_h = np.ascontiguousarray(hist_g), np.ascontiguousarray(hist_h)

        with self._lock:
            # Evict until under budget
            while (
                len(self.cache) >= self.max_cache_size
                or self.memory_usage + hist_size > self.max_memory_bytes
            ):
                old_id, (old_g, old_h) = self.cache.popitem(last=False)  # oldest
                self.memory_usage -= old_g.nbytes + old_h.nbytes
                self.evictions += 1

            self.cache[node_id] = (hist_g, hist_h)
            self.memory_usage += hist_size

    def get_hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ================================================================
# Ultra-fast histogram building with advanced optimizations
# ================================================================

@njit(parallel=True, fastmath=True, cache=True)
def compute_histograms(X, gradients, hessians, bin_edges, n_bins):
    """
    SOTA vectorized histogram computation with enhanced optimizations:
    - Parallel feature processing with optimal load balancing
    - Optimized bin assignment using manual binary search
    - Memory-efficient accumulation with reduced allocations
    - Cache-friendly memory access patterns
    """
    n_samples, n_features = X.shape
    hist_g = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_h = np.zeros((n_features, n_bins), dtype=np.float64)

    # Parallel processing with adaptive work distribution
    for f in prange(n_features):
        edges = bin_edges[f]
        n_edges = len(edges)
        
        # Pre-compute edge bounds for faster binary search
        min_edge = edges[1]
        max_edge = edges[n_edges - 2]
        
        # Vectorized bin assignment with optimized binary search
        for i in range(n_samples):
            val = X[i, f]
            g_val = gradients[i]
            h_val = hessians[i]
            
            # Handle missing/invalid values
            if not (val == val):  # NaN check
                bin_idx = n_bins - 1
            elif val <= min_edge:
                bin_idx = 0
            elif val >= max_edge:
                bin_idx = n_bins - 2
            else:
                # Optimized binary search
                left, right = 1, n_edges - 2
                while left < right:
                    mid = (left + right) >> 1  # Bit shift for faster division
                    if val > edges[mid]:
                        left = mid + 1
                    else:
                        right = mid
                bin_idx = left - 1
            
            # Bounds check and accumulate
            if 0 <= bin_idx < n_bins:
                hist_g[f, bin_idx] += g_val
                hist_h[f, bin_idx] += h_val

    return hist_g, hist_h


@njit(parallel=True, fastmath=True, cache=True)
def compute_histograms_sparse(X_indices, X_values, gradients, hessians, 
                             bin_edges, n_bins, n_features):
    """
    Optimized histogram computation for sparse data:
    - Skip zero values completely
    - Efficient sparse indexing
    - Reduced memory bandwidth
    """
    hist_g = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_h = np.zeros((n_features, n_bins), dtype=np.float64)
    
    n_nonzero = len(X_indices)
    
    for idx in prange(n_nonzero):
        sample_idx = X_indices[idx] // n_features
        feature_idx = X_indices[idx] % n_features
        val = X_values[idx]
        
        if sample_idx < len(gradients):
            g_val = gradients[sample_idx]
            h_val = hessians[sample_idx]
            
            # Binary search for bin assignment
            edges = bin_edges[feature_idx]
            left, right = 1, len(edges) - 2
            
            while left < right:
                mid = (left + right) >> 1
                if val > edges[mid]:
                    left = mid + 1
                else:
                    right = mid
            
            bin_idx = left - 1
            if 0 <= bin_idx < n_bins:
                hist_g[feature_idx, bin_idx] += g_val
                hist_h[feature_idx, bin_idx] += h_val
    
    return hist_g, hist_h


# ================================================================
# Advanced gradient-weighted binning with sketching
# ================================================================

@njit(parallel=True, fastmath=True, cache=True)
def build_bins_cpu(X, gradients, n_bins):
    """
    SOTA adaptive bin construction with optimizations:
    - Parallel feature processing
    - Gradient-weighted quantile estimation
    - Memory-efficient sorting
    - Numerical stability improvements
    """
    n_samples, n_features = X.shape
    bin_edges = np.empty((n_features, n_bins + 1), dtype=np.float64)
    
    # Pre-compute absolute gradients once
    abs_gradients = np.abs(gradients)
    
    for f in prange(n_features):
        feature_data = X[:, f]
        
        # Handle edge case: all values are the same
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)
        
        if max_val - min_val < 1e-12:  # Nearly constant feature
            edges = np.linspace(min_val - 1e-6, max_val + 1e-6, n_bins + 1)
            bin_edges[f] = edges
            continue
        
        # Filter out invalid values
        valid_mask = np.isfinite(feature_data)
        if np.sum(valid_mask) < 2:  # Not enough valid values
            edges = np.linspace(min_val, max_val, n_bins + 1)
            bin_edges[f] = edges
            continue
        
        valid_data = feature_data[valid_mask]
        valid_weights = abs_gradients[valid_mask]
        
        # Efficient sorting with argsort
        sorted_idx = np.argsort(valid_data)
        sorted_vals = valid_data[sorted_idx]
        sorted_weights = valid_weights[sorted_idx]
        
        # Optimized cumulative sum
        cumsum_weights = np.empty_like(sorted_weights)
        cumsum_weights[0] = sorted_weights[0]
        for i in range(1, len(sorted_weights)):
            cumsum_weights[i] = cumsum_weights[i-1] + sorted_weights[i]
        
        total_weight = cumsum_weights[-1]
        if total_weight <= 1e-12:  # No significant gradients
            edges = np.linspace(min_val, max_val, n_bins + 1)
            bin_edges[f] = edges
            continue
        
        # Create quantile positions
        quantile_positions = np.empty(n_bins + 1, dtype=np.float64)
        for i in range(n_bins + 1):
            quantile_positions[i] = (i / n_bins) * total_weight
        
        # Linear interpolation for quantiles
        edges = np.empty(n_bins + 1, dtype=np.float64)
        edges[0] = sorted_vals[0] - 1e-6
        edges[-1] = sorted_vals[-1] + 1e-6
        
        for i in range(1, n_bins):
            target_weight = quantile_positions[i]
            
            # Binary search for interpolation
            left, right = 0, len(cumsum_weights) - 1
            while left < right:
                mid = (left + right) >> 1
                if cumsum_weights[mid] < target_weight:
                    left = mid + 1
                else:
                    right = mid
            
            if left == 0:
                edges[i] = sorted_vals[0]
            elif left >= len(sorted_vals):
                edges[i] = sorted_vals[-1]
            else:
                # Linear interpolation between points
                w0, w1 = cumsum_weights[left-1], cumsum_weights[left]
                v0, v1 = sorted_vals[left-1], sorted_vals[left]
                
                if w1 - w0 > 1e-12:
                    alpha = (target_weight - w0) / (w1 - w0)
                    edges[i] = v0 + alpha * (v1 - v0)
                else:
                    edges[i] = v0
        
        bin_edges[f] = edges

    return bin_edges


def build_bins_gpu(X, gradients, n_bins):
    """
    High-performance GPU gradient-weighted quantile binning:
    - Optimized GPU memory usage
    - Parallel sorting and reduction
    - Memory coalescing optimizations
    - Reduced CPU-GPU transfers
    """
    if not GPU_AVAILABLE:
        return build_bins_cpu(X, gradients, n_bins)
    
    n_samples, n_features = X.shape
    
    # Use mixed precision for better GPU performance
    X_gpu = cp.asarray(X, dtype=cp.float32)
    grad_weights = cp.abs(cp.asarray(gradients, dtype=cp.float32))
    
    # Pre-allocate result on CPU to minimize transfers
    bin_edges = np.empty((n_features, n_bins + 1), dtype=np.float64)
    
    # Process features in batches to optimize GPU memory usage
    batch_size = min(32, n_features)  # Process up to 32 features at once
    
    for batch_start in range(0, n_features, batch_size):
        batch_end = min(batch_start + batch_size, n_features)
        batch_features = []
        
        for f in range(batch_start, batch_end):
            feat_vals = X_gpu[:, f]
            
            # Handle constant features
            min_val = cp.min(feat_vals)
            max_val = cp.max(feat_vals)
            
            if float(max_val - min_val) < 1e-12:
                edges = cp.linspace(float(min_val) - 1e-6, 
                                  float(max_val) + 1e-6, n_bins + 1)
                batch_features.append(edges.get().astype(np.float64))
                continue
            
            # Filter valid values on GPU
            valid_mask = cp.isfinite(feat_vals)
            if cp.sum(valid_mask) < 2:
                edges = cp.linspace(float(min_val), float(max_val), n_bins + 1)
                batch_features.append(edges.get().astype(np.float64))
                continue
            
            valid_vals = feat_vals[valid_mask]
            valid_weights = grad_weights[valid_mask]
            
            # GPU-accelerated sorting
            sorted_idx = cp.argsort(valid_vals)
            sorted_vals = valid_vals[sorted_idx]
            sorted_wts = valid_weights[sorted_idx]
            
            # Efficient cumulative sum on GPU
            cumsum_wts = cp.cumsum(sorted_wts)
            total_wt = float(cumsum_wts[-1])
            
            if total_wt <= 1e-12:
                edges = cp.linspace(float(min_val), float(max_val), n_bins + 1)
                batch_features.append(edges.get().astype(np.float64))
                continue
            
            # Quantile target positions
            quantile_positions = cp.linspace(0, total_wt, n_bins + 1, dtype=cp.float32)
            
            # GPU interpolation
            edges = cp.interp(quantile_positions, cumsum_wts, sorted_vals)
            edges[0] = sorted_vals[0] - 1e-6
            edges[-1] = sorted_vals[-1] + 1e-6
            
            batch_features.append(edges.get().astype(np.float64))
        
        # Copy batch results to final array
        for i, edges in enumerate(batch_features):
            bin_edges[batch_start + i] = edges
    
    return bin_edges


# ================================================================
# Enhanced GPU Accelerator with memory management
# ================================================================

class GPUAccelerator:
    """
    SOTA GPU acceleration with advanced optimizations:
    - Persistent GPU memory management
    - Automatic memory pool optimization
    - Mixed precision computation
    - Adaptive batch processing
    """
    
    def __init__(self, memory_pool_size=0.8):
        self.available = GPU_AVAILABLE
        self.memory_pool_size = memory_pool_size
        
        if not self.available:
            warnings.warn("CuPy not available. GPU acceleration disabled.")
            return
        
        # Initialize memory pool for better performance
        try:
            mempool = cp.get_default_memory_pool()
            # Set memory pool to use 80% of GPU memory by default
            total_memory = cp.cuda.Device().mem_info[1]
            mempool.set_limit(size=int(total_memory * memory_pool_size))
        except Exception as e:
            warnings.warn(f"GPU memory pool setup failed: {e}")
        
        # Persistent GPU arrays for reduced transfers
        self._X_gpu = None
        self._g_gpu = None  
        self._h_gpu = None
        self._cached_shapes = {}
        
        # Performance monitoring
        self._transfer_time = 0.0
        self._compute_time = 0.0

    def to_gpu_persistent(self, X, g=None, h=None):
        """
        Optimized persistent GPU transfers with caching:
        - Reuse existing GPU arrays when possible
        - Minimize data transfers
        - Efficient memory management
        """
        if not self.available:
            return X, g, h
        
        import time
        start_time = time.time()
        
        try:
            # Check if we can reuse existing arrays
            X_shape = X.shape if X is not None else None
            g_shape = g.shape if g is not None else None
            h_shape = h.shape if h is not None else None
            
            if (X is not None and 
                (self._X_gpu is None or self._X_gpu.shape != X_shape)):
                self._X_gpu = cp.asarray(X, dtype=cp.float32)
            
            if (g is not None and 
                (self._g_gpu is None or self._g_gpu.shape != g_shape)):
                self._g_gpu = cp.asarray(g, dtype=cp.float32)
            
            if (h is not None and 
                (self._h_gpu is None or self._h_gpu.shape != h_shape)):
                self._h_gpu = cp.asarray(h, dtype=cp.float32)
            
            self._transfer_time += time.time() - start_time
            return self._X_gpu, self._g_gpu, self._h_gpu
            
        except cp.cuda.memory.OutOfMemoryError:
            warnings.warn("GPU out of memory, falling back to CPU")
            self.available = False
            return X, g, h

    def clear_persistent(self):
        """Enhanced GPU memory cleanup with pool management"""
        if not self.available:
            return
        
        self._X_gpu = None
        self._g_gpu = None
        self._h_gpu = None
        self._cached_shapes.clear()
        
        # Force memory pool cleanup
        try:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            
            # Also clear pinned memory pool
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
        except Exception:
            pass

    def to_gpu(self, arr):
        """Fast GPU transfer with error handling"""
        if self.available and isinstance(arr, np.ndarray):
            try:
                return cp.asarray(arr, dtype=cp.float32)
            except cp.cuda.memory.OutOfMemoryError:
                warnings.warn("GPU out of memory, using CPU")
                self.available = False
        return arr

    def to_cpu(self, arr):
        """Fast CPU transfer with type preservation"""
        if self.available and hasattr(arr, "get"):
            result = arr.get()
            # Convert back to float64 for compatibility
            if result.dtype == np.float32:
                return result.astype(np.float64)
            return result
        return arr

    def build_bins(self, X, g, n_bins=256):
        """
        Optimized bin building with automatic fallback:
        - GPU acceleration when available
        - Automatic CPU fallback on memory issues
        - Performance monitoring
        """
        try:
            if self.available:
                return build_bins_gpu(X, g, n_bins)
            else:
                return build_bins_cpu(X, g, n_bins)
        except Exception as e:
            warnings.warn(f"GPU binning failed: {e}, falling back to CPU")
            self.available = False
            return build_bins_cpu(X, g, n_bins)

    def build_histograms(self, X, g, h, bin_edges, n_bins):
        """
        SOTA histogram building with adaptive optimization:
        - GPU acceleration with memory management
        - Automatic batch sizing
        - Performance monitoring and fallback
        """
        try:
            if self.available:
                return self._build_histograms_gpu(X, g, h, bin_edges, n_bins)
            else:
                return compute_histograms(X, g, h, bin_edges, n_bins)
        except Exception as e:
            warnings.warn(f"GPU histogram building failed: {e}")
            self.available = False
            return compute_histograms(X, g, h, bin_edges, n_bins)

    def _build_histograms_gpu(self, X, g, h, bin_edges, n_bins):
        """
        Ultra-optimized GPU histogram computation:
        - Memory-efficient scatter operations
        - Optimized indexing and accumulation
        - Mixed precision for speed
        """
        import time
        start_time = time.time()
        
        n_samples, n_features = X.shape
        
        # Use persistent arrays if available and matching
        if (self._X_gpu is not None and self._X_gpu.shape == X.shape):
            X_gpu = self._X_gpu
        else:
            X_gpu = cp.asarray(X, dtype=cp.float32)
        
        g_gpu = cp.asarray(g, dtype=cp.float32)
        h_gpu = cp.asarray(h, dtype=cp.float32)
        bin_edges_gpu = cp.asarray(bin_edges, dtype=cp.float32)
        
        # Pre-allocate output arrays
        hist_g = cp.zeros((n_features, n_bins), dtype=cp.float32)
        hist_h = cp.zeros((n_features, n_bins), dtype=cp.float32)
        
        # Optimized bin assignment with memory coalescing
        bins_flat = cp.empty(n_samples * n_features, dtype=cp.int32)
        
        # Process features in parallel batches for memory efficiency
        feature_batch_size = min(16, n_features)
        
        for batch_start in range(0, n_features, feature_batch_size):
            batch_end = min(batch_start + feature_batch_size, n_features)
            
            for f_idx in range(batch_start, batch_end):
                # Get feature values and bin edges
                feature_vals = X_gpu[:, f_idx]
                edges = bin_edges_gpu[f_idx, 1:-1]  # Exclude first/last sentinels
                
                # Vectorized bin assignment
                bins = cp.searchsorted(edges, feature_vals, side='right')
                bins = cp.clip(bins, 0, n_bins - 1)
                
                # Store in flat array for efficient scatter
                start_idx = f_idx * n_samples
                end_idx = start_idx + n_samples
                bins_flat[start_idx:end_idx] = bins
        
        # Create feature indices for scatter operation
        feature_indices = cp.repeat(cp.arange(n_features, dtype=cp.int32), n_samples)
        
        # Compute flat indices for 2D histogram arrays
        flat_indices = feature_indices * n_bins + bins_flat
        
        # Tile gradient/hessian values for all features
        g_tiled = cp.tile(g_gpu, n_features)
        h_tiled = cp.tile(h_gpu, n_features)
        
        # Efficient scatter addition using cp.bincount
        # This is faster than cp.add.at for dense accumulation
        max_idx = n_features * n_bins
        
        hist_g_flat = cp.bincount(flat_indices, weights=g_tiled, minlength=max_idx)
        hist_h_flat = cp.bincount(flat_indices, weights=h_tiled, minlength=max_idx)
        
        # Reshape back to 2D
        hist_g = hist_g_flat[:max_idx].reshape((n_features, n_bins))
        hist_h = hist_h_flat[:max_idx].reshape((n_features, n_bins))
        
        self._compute_time += time.time() - start_time
        
        # Convert back to CPU with proper dtype
        return hist_g.get().astype(np.float64), hist_h.get().astype(np.float64)

    def get_performance_stats(self):
        """Get GPU performance statistics"""
        return {
            'available': self.available,
            'transfer_time': self._transfer_time,
            'compute_time': self._compute_time,
            'total_time': self._transfer_time + self._compute_time
        }

    def reset_stats(self):
        """Reset performance counters"""
        self._transfer_time = 0.0
        self._compute_time = 0.0
