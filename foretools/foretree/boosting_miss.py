from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numba
import numpy as np
from numba import jit, njit, prange


@njit(cache=True)
def _flatten_edges_for_numba(bin_edges_list):
    n_features = len(bin_edges_list)
    total = 0
    for e in bin_edges_list:
        total += len(e)
    flat = np.empty(total, dtype=np.float64)
    starts = np.empty(n_features, dtype=np.int32)
    off = 0
    for i, e in enumerate(bin_edges_list):
        starts[i] = off
        m = len(e)
        flat[off : off + m] = e
        off += m
    return flat, starts


@njit(parallel=True, fastmath=True, cache=True)
def encode_with_reserved_missing_all_features(X, bin_edges_flat, bin_start_indices):
    """Encode to integer bins with a dedicated missing bin."""
    n_samples, n_features = X.shape
    out = np.empty((n_samples, n_features), dtype=np.int32)

    for f in prange(n_features):
        start = bin_start_indices[f]
        end = bin_start_indices[f + 1] if f < n_features - 1 else len(bin_edges_flat)
        edges = bin_edges_flat[start:end]
        n_bins = len(edges) - 1
        missing_id = n_bins

        for i in range(n_samples):
            v = X[i, f]
            if not np.isfinite(v):
                out[i, f] = missing_id
                continue

            j = 1
            while j < len(edges) and v > edges[j]:
                j += 1
            b = j - 1
            if b >= n_bins:
                b = n_bins - 1
            out[i, f] = b
    return out


@njit(fastmath=True, cache=True)
def compute_optimal_missing_direction(
    g_left, h_left, g_right, h_right, g_missing, h_missing, lambda_reg=0.1
):
    """
    Compute optimal direction for missing values using gain comparison.
    Returns: (go_left: bool, gain_improvement: float)
    """
    if h_missing <= 0:
        return True, 0.0
    
    # Current gain (missing go left)
    g_left_with_missing = g_left + g_missing
    h_left_with_missing = h_left + h_missing
    gain_left = (g_left_with_missing * g_left_with_missing) / (h_left_with_missing + lambda_reg) + \
                (g_right * g_right) / (h_right + lambda_reg)
    
    # Alternative gain (missing go right)
    g_right_with_missing = g_right + g_missing
    h_right_with_missing = h_right + h_missing
    gain_right = (g_left * g_left) / (h_left + lambda_reg) + \
                 (g_right_with_missing * g_right_with_missing) / (h_right_with_missing + lambda_reg)
    
    if gain_left > gain_right:
        return True, gain_left - gain_right
    else:
        return False, gain_right - gain_left


@njit(fastmath=True, cache=True)
def find_best_surrogate_split(
    X_surrogates, missing_mask, primary_split_mask, min_samples_leaf=5
):
    """
    Find best surrogate split that mimics primary split behavior.
    Returns: (best_feature_idx, best_threshold, best_agreement)
    """
    n_samples, n_features = X_surrogates.shape
    best_agreement = 0.0
    best_feature = -1
    best_threshold = 0.0
    
    # Only evaluate on samples with missing primary feature
    missing_indices = np.where(missing_mask)[0]
    if len(missing_indices) < min_samples_leaf * 2:
        return best_feature, best_threshold, best_agreement
    
    target_decisions = primary_split_mask[missing_indices]
    
    for f in range(n_features):
        # Get surrogate feature values for missing samples
        surrogate_values = X_surrogates[missing_indices, f]
        
        # Skip if surrogate also has missing values
        valid_surrogate = np.isfinite(surrogate_values)
        if np.sum(valid_surrogate) < min_samples_leaf * 2:
            continue
            
        valid_missing_indices = missing_indices[valid_surrogate]
        valid_surrogate_values = surrogate_values[valid_surrogate]
        valid_target_decisions = target_decisions[valid_surrogate]
        
        # Sort for threshold candidates
        sorted_indices = np.argsort(valid_surrogate_values)
        sorted_values = valid_surrogate_values[sorted_indices]
        sorted_decisions = valid_target_decisions[sorted_indices]
        
        # Try thresholds between unique values
        for i in range(min_samples_leaf, len(sorted_values) - min_samples_leaf):
            if sorted_values[i] == sorted_values[i-1]:
                continue
                
            threshold = (sorted_values[i] + sorted_values[i-1]) / 2.0
            
            # Compute agreement
            surrogate_decisions = valid_surrogate_values <= threshold
            agreement = np.mean(surrogate_decisions == valid_target_decisions)
            
            if agreement > best_agreement:
                best_agreement = agreement
                best_feature = f
                best_threshold = threshold
    
    return best_feature, best_threshold, best_agreement


@njit(fastmath=True, cache=True)
def apply_surrogate_splits(
    X, missing_mask, primary_split_mask, surrogate_features, surrogate_thresholds, 
    surrogate_agreements, default_direction=True
):
    """
    Apply surrogate splits to handle missing values.
    Uses best available surrogate or falls back to default direction.
    """
    n_samples = len(missing_mask)
    final_decisions = primary_split_mask.copy()
    
    missing_indices = np.where(missing_mask)[0]
    
    for idx in missing_indices:
        decision_made = False
        
        # Try surrogates in order of agreement
        for s in range(len(surrogate_features)):
            if surrogate_features[s] == -1:
                break
                
            surrogate_val = X[idx, surrogate_features[s]]
            if np.isfinite(surrogate_val):
                final_decisions[idx] = surrogate_val <= surrogate_thresholds[s]
                decision_made = True
                break
        
        # Use default direction if no surrogate available
        if not decision_made:
            final_decisions[idx] = default_direction
    
    return final_decisions


class MissingStrategy(Enum):
    """Strategy for handling missing values during splits."""
    LEARN_DIRECTION = "learn"
    ALWAYS_LEFT = "left" 
    ALWAYS_RIGHT = "right"
    SURROGATE_SPLITS = "surrogate"


@dataclass
class SplitResult:
    """Container for split results with missing handling."""
    left_indices: np.ndarray
    right_indices: np.ndarray
    missing_direction: bool
    gain_improvement: float = 0.0
    surrogate_splits: Optional[List[Tuple[int, float, float]]] = None
    

@dataclass  
class MissingValueConfig:
    """Configuration for missing value handling."""
    strategy: MissingStrategy = MissingStrategy.LEARN_DIRECTION
    lambda_reg: float = 0.1  # L2 regularization for gain computation
    max_surrogate_splits: int = 3  # Max number of surrogate splits to find
    min_surrogate_agreement: float = 0.55  # Minimum agreement for useful surrogate
    min_samples_leaf: int = 5
    surrogate_search_features: int = 50  # Limit features searched for surrogates
    finite_check_mode = "strict"
    

class UnifiedMissingHandler:
    """
    SOTA Missing value handler with learned directions and surrogate splits.
    Optimized for performance while maintaining comprehensive functionality.
    """

    def __init__(self, config: MissingValueConfig = None):
        self.config = config or MissingValueConfig()
        self._cache_size_limit = 1000  # Prevent memory bloat
        self._missing_masks_cache: Dict[int, np.ndarray] = {}
        
    def _evict_cache_if_needed(self):
        """Simple LRU-style cache eviction."""
        if len(self._missing_masks_cache) > self._cache_size_limit:
            # Remove oldest 20% of entries (simple heuristic)
            keys_to_remove = list(self._missing_masks_cache.keys())[
                :len(self._missing_masks_cache) // 5
            ]
            for key in keys_to_remove:
                del self._missing_masks_cache[key]

    @njit(fastmath=True, cache=True)
    def _detect_missing_numba(values):
        """Fast missing detection."""
        return ~np.isfinite(values)

    def compute_split_with_missing(
        self, 
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray, 
        node_indices: np.ndarray,
        feature_idx: int,
        threshold: float,
        X_surrogates: Optional[np.ndarray] = None
    ) -> SplitResult:
        """
        Comprehensive split computation with SOTA missing handling.
        """
        feature_values = X[node_indices, feature_idx]
        missing_mask = ~np.isfinite(feature_values)
        n_missing = np.sum(missing_mask)
        
        # If no missing values, use standard split
        if n_missing == 0:
            standard_split = feature_values <= threshold
            left_idx = node_indices[standard_split]
            right_idx = node_indices[~standard_split]
            return SplitResult(left_idx, right_idx, missing_direction=True)
        
        # Compute primary split on non-missing values  
        non_missing_mask = ~missing_mask
        primary_split_mask = np.full(len(feature_values), False, dtype=bool)
        primary_split_mask[non_missing_mask] = feature_values[non_missing_mask] <= threshold
        
        # Strategy-specific missing value handling
        if self.config.strategy == MissingStrategy.LEARN_DIRECTION:
            missing_direction, gain_improvement = self._learn_optimal_direction(
                gradients, hessians, node_indices, primary_split_mask, missing_mask
            )
            
        elif self.config.strategy == MissingStrategy.SURROGATE_SPLITS:
            if X_surrogates is not None:
                return self._handle_surrogate_splits(
                    X_surrogates, node_indices, missing_mask, primary_split_mask
                )
            else:
                # Fall back to learned direction
                missing_direction, gain_improvement = self._learn_optimal_direction(
                    gradients, hessians, node_indices, primary_split_mask, missing_mask
                )
        else:
            # Simple fixed directions
            missing_direction = self.config.strategy == MissingStrategy.ALWAYS_LEFT
            gain_improvement = 0.0
        
        # Apply missing direction
        final_split_mask = primary_split_mask.copy()
        final_split_mask[missing_mask] = missing_direction
        
        left_indices = node_indices[final_split_mask]
        right_indices = node_indices[~final_split_mask]
        
        return SplitResult(
            left_indices, right_indices, missing_direction, gain_improvement
        )
    
    def _learn_optimal_direction(
        self, gradients, hessians, node_indices, primary_split_mask, missing_mask
    ):
        """Learn optimal direction for missing values."""
        node_g = gradients[node_indices]
        node_h = hessians[node_indices]
        
        # Compute statistics for left/right sides (non-missing only)
        non_missing = ~missing_mask
        left_mask = primary_split_mask & non_missing
        right_mask = (~primary_split_mask) & non_missing
        missing_only = missing_mask
        
        g_left = np.sum(node_g[left_mask]) if np.any(left_mask) else 0.0
        h_left = np.sum(node_h[left_mask]) if np.any(left_mask) else 0.0
        g_right = np.sum(node_g[right_mask]) if np.any(right_mask) else 0.0
        h_right = np.sum(node_h[right_mask]) if np.any(right_mask) else 0.0
        g_missing = np.sum(node_g[missing_only]) if np.any(missing_only) else 0.0
        h_missing = np.sum(node_h[missing_only]) if np.any(missing_only) else 0.0
        
        return compute_optimal_missing_direction(
            g_left, h_left, g_right, h_right, g_missing, h_missing, self.config.lambda_reg
        )
    
    def _handle_surrogate_splits(
        self, X_surrogates, node_indices, missing_mask, primary_split_mask
    ):
        """Handle missing values using surrogate splits."""
        # Limit features to search for performance
        n_features = X_surrogates.shape[1]
        if n_features > self.config.surrogate_search_features:
            # Randomly sample features to search
            np.random.seed(42)  # Reproducible
            search_features = np.random.choice(
                n_features, self.config.surrogate_search_features, replace=False
            )
            X_search = X_surrogates[:, search_features]
        else:
            X_search = X_surrogates
            
        X_node = X_search[node_indices]
        
        # Find surrogate splits
        surrogate_splits = []
        remaining_missing_mask = missing_mask.copy()
        
        for _ in range(self.config.max_surrogate_splits):
            if np.sum(remaining_missing_mask) < self.config.min_samples_leaf:
                break
                
            best_feat, best_thresh, best_agreement = find_best_surrogate_split(
                X_node, remaining_missing_mask, primary_split_mask, self.config.min_samples_leaf
            )
            
            if best_feat == -1 or best_agreement < self.config.min_surrogate_agreement:
                break
                
            surrogate_splits.append((best_feat, best_thresh, best_agreement))
            
            # Update remaining missing mask (remove samples handled by this surrogate)
            surrogate_values = X_node[remaining_missing_mask, best_feat]
            handled_mask = np.isfinite(surrogate_values)
            remaining_missing_indices = np.where(remaining_missing_mask)[0]
            remaining_missing_mask[remaining_missing_indices[handled_mask]] = False
        
        # Apply surrogate splits
        if surrogate_splits:
            surrogate_features = np.array([s[0] for s in surrogate_splits] + [-1] * (3 - len(surrogate_splits)))
            surrogate_thresholds = np.array([s[1] for s in surrogate_splits] + [0.0] * (3 - len(surrogate_splits)))
            surrogate_agreements = np.array([s[2] for s in surrogate_splits] + [0.0] * (3 - len(surrogate_splits)))
            
            final_split_mask = apply_surrogate_splits(
                X_node, missing_mask, primary_split_mask,
                surrogate_features, surrogate_thresholds, surrogate_agreements,
                default_direction=True  # Fallback for unsolved cases
            )
        else:
            # No good surrogates found, use default direction
            final_split_mask = primary_split_mask.copy()
            final_split_mask[missing_mask] = True
        
        left_indices = node_indices[final_split_mask]
        right_indices = node_indices[~final_split_mask]
        
        return SplitResult(
            left_indices, right_indices, 
            missing_direction=True,  # Not applicable for surrogates
            surrogate_splits=surrogate_splits
        )

    def prebin_matrix_with_reserved_missing(
        self, X: np.ndarray, bin_edges_list: list, max_bins: int, out_dtype=np.int32
    ) -> Tuple[np.ndarray, int]:
        """Vectorized prebinning with dedicated missing bin."""
        flat, starts = _flatten_edges_for_numba(bin_edges_list)
        binned = encode_with_reserved_missing_all_features(X, flat, starts)
        if out_dtype != np.int32:
            binned = binned.astype(out_dtype, copy=False)
        return binned, max_bins  # missing bin id

    def detect_missing(self, X: np.ndarray, feature_idx: int = None) -> np.ndarray:
        """Unified missing detection with caching."""
        if feature_idx is not None:
            # Single feature case with caching
            cache_key = feature_idx
            if cache_key in self._missing_masks_cache:
                return self._missing_masks_cache[cache_key]

            col = X[:, feature_idx] if X.ndim == 2 else X
            missing_mask = ~np.isfinite(col)
            
            self._evict_cache_if_needed()
            self._missing_masks_cache[cache_key] = missing_mask
            return missing_mask
        else:
            # All features
            return ~np.isfinite(X)

    def get_missing_stats(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        data_indices: np.ndarray,
        feature_idx: int,
    ) -> Dict:
        """Fast missing statistics computation."""
        col_values = X[data_indices, feature_idx]
        missing_mask = self.detect_missing(col_values)

        # Compute stats
        missing_indices_mask = missing_mask
        n_missing = np.sum(missing_indices_mask)
        
        if n_missing > 0:
            g_missing = np.sum(g[data_indices][missing_indices_mask])
            h_missing = np.sum(h[data_indices][missing_indices_mask])
            missing_indices = data_indices[missing_indices_mask]
        else:
            g_missing = 0.0
            h_missing = 0.0
            missing_indices = np.array([], dtype=np.int64)

        stats = {
            "n_missing": n_missing,
            "g_missing": g_missing,
            "h_missing": h_missing,
            "missing_indices": missing_indices,
            "missing_mask": missing_mask,
        }

        return stats

    def apply_missing_split(
        self,
        X: np.ndarray,
        node_indices: np.ndarray,
        feature_idx: int,
        threshold: float,
        missing_go_left: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fast split application with missing handling (backward compatibility)."""
        col_values = X[node_indices, feature_idx]
        missing_mask = ~np.isfinite(col_values)

        # Apply split logic
        split_mask = np.empty(len(col_values), dtype=bool)
        for i in range(len(col_values)):
            if missing_mask[i]:
                split_mask[i] = missing_go_left
            else:
                split_mask[i] = col_values[i] <= threshold

        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]

        return left_indices, right_indices

    def missing_bin_id(self, max_bins: int) -> int:
        """Return the reserved bin index for missing values."""
        return max_bins  # last id

    def clear_cache(self):
        """Clear caches to free memory."""
        self._missing_masks_cache.clear()
