import heapq

import numpy as np
from boosting_aux import *
from boosting_loss import *
from numba import njit

try:
    import cupy as cp  # isort: skip
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# ================ LEAF NODE FOR LEAF-WISE GROWTH ================
class LeafNode:
    """Represents a leaf in the tree with potential for splitting"""

    __slots__ = (
        "node_id",
        "data_indices",
        "gradients",
        "hessians",
        "depth",
        "parent_hist",
        "is_left_child",
        "n_samples",
        "g_sum",
        "h_sum",
        "best_feature",
        "best_threshold",
        "best_gain",
        "best_bin_idx",
        "histograms",
        "left_child",
        "right_child",
        "leaf_value",
        "is_leaf",
        "sibling_node_id",
        "missing_go_left",
    )

    def __init__(
        self,
        node_id,
        data_indices,
        gradients,
        hessians,
        depth,
        parent_hist=None,
        is_left_child=False,
    ):
        self.node_id = node_id
        self.data_indices = data_indices
        self.gradients = gradients
        self.hessians = hessians
        self.depth = depth
        self.parent_hist = parent_hist
        self.is_left_child = is_left_child

        # Computed properties
        self.n_samples = len(data_indices)
        self.g_sum = np.sum(gradients)
        self.h_sum = np.sum(hessians)

        # Split information (computed when evaluated)
        self.best_feature = None
        self.best_threshold = None
        self.best_gain = -np.inf
        self.best_bin_idx = None
        self.histograms = None

        # Tree structure
        self.left_child = None
        self.right_child = None
        self.leaf_value = None
        self.is_leaf = True

        # For sibling histogram subtraction
        self.sibling_node_id = None
        self.missing_go_left = False

    def __lt__(self, other):
        """For priority queue ordering (max heap based on gain)"""
        return self.best_gain > other.best_gain  # Reverse for max heap


@njit
def partition_row_idx(row_idx, feat_vals, threshold, missing_go_left):
    """Optimized in-place partitioning for row indices"""
    n = feat_vals.shape[0]
    left_pos, right_pos = 0, n-1
    while left_pos <= right_pos:
        x = feat_vals[left_pos]
        if np.isnan(x):  # missing
            go_left = missing_go_left
        else:
            go_left = x <= threshold
        if go_left:
            left_pos += 1
        else:
            # swap
            row_idx[left_pos], row_idx[right_pos] = row_idx[right_pos], row_idx[left_pos]
            right_pos -= 1
    return left_pos  # size of left partition


@njit
def predict_tree_numba(X, node_features, node_thresholds, node_missing_go_left, 
                      left_children, right_children, leaf_values, is_leaf_flags, 
                      feature_map_array, root_idx=0):
    """Fast numba prediction for a single tree"""
    n_samples = X.shape[0]
    predictions = np.empty(n_samples, dtype=np.float64)
    
    for i in range(n_samples):
        node_idx = root_idx
        
        while not is_leaf_flags[node_idx]:
            global_feat_idx = node_features[node_idx]
            local_feat_idx = feature_map_array[global_feat_idx]
            threshold = node_thresholds[node_idx]
            missing_go_left = node_missing_go_left[node_idx]
            
            x = X[i, local_feat_idx]
            if np.isnan(x):
                go_left = missing_go_left
            else:
                go_left = x <= threshold
                
            if go_left:
                node_idx = left_children[node_idx]
            else:
                node_idx = right_children[node_idx]
        
        predictions[i] = leaf_values[node_idx]
    
    return predictions


# ================ LEAF-WISE TREE BUILDER ================
class LeafWiseTreeBuilder:
    """Builds trees using leaf-wise growth strategy for maximum efficiency"""

    def __init__(
        self,
        max_leaves=31,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=5,
        lambda_=1.0,
        gamma=0.0,
        alpha=0.0,
        feature_indices=None,
        n_bins=256,
        bin_edges=None,
        gpu_accelerator=None,
        monotone_constraints=None,
        max_delta_step=0.0,
        min_child_weight=1e-3,
    ):
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.feature_indices = feature_indices
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.gpu_accelerator = gpu_accelerator
        self.monotone_constraints = monotone_constraints or {}
        self.max_delta_step = max_delta_step
        self.min_child_weight = min_child_weight
        self.binned = None
        self.binned_local = None  # (n_sub, n_local_features), set by caller

        # Tree state
        self.nodes = {}  # node_id -> LeafNode
        self.leaf_queue = []  # Priority queue of leaves to potentially split
        self.next_node_id = 0
        self.histogram_cache = HistogramCache()

        # Precompute feature lookup for speed (global→local)
        if self.feature_indices is not None:
            self.feature_map = {f: i for i, f in enumerate(self.feature_indices)}
            self._temp_mask = np.empty(0, dtype=bool)
        else:
            self.feature_map = {}

        # Root node placeholder
        self.root = None
        
        # Prediction optimization
        self._prediction_arrays = None

    def _get_next_node_id(self):
        """Get unique node ID"""
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

    def _compute_histograms_with_reuse(self, X, node, use_cache=True):
        """Compute histograms with caching and sibling reuse (Numba inside)."""
        # Try cache
        if use_cache:
            cached = self.histogram_cache.get(node.node_id)
            if cached is not None:
                return cached

        # Try sibling subtraction
        if (
            node.parent_hist is not None
            and hasattr(node, "sibling_node_id")
            and node.sibling_node_id is not None
            and use_cache
        ):
            sibling_hist = self.histogram_cache.get(node.sibling_node_id)
            if sibling_hist is not None:
                sibling_hist_g, sibling_hist_h = sibling_hist
                parent_hist_g, parent_hist_h = node.parent_hist
                hist_g, hist_h = subtract_histograms(
                    parent_hist_g, parent_hist_h, sibling_hist_g, sibling_hist_h
                )
                self.histogram_cache.put(node.node_id, hist_g, hist_h)
                return hist_g, hist_h

        if self.binned_local is not None:
            idx = np.asarray(node.data_indices, dtype=np.int64)
            hist_g, hist_h = compute_histograms_from_binned(
                self.binned_local,
                idx,
                node.gradients,
                node.hessians,
                n_features=len(self.feature_indices),
                n_bins=self.n_bins,
            )
        else:
            # Fallback to original raw-data histogram functions
            X_subset = X[node.data_indices]
            feature_bin_edges = np.array(
                [self.bin_edges[i] for i in self.feature_indices]
            )
            if self.gpu_accelerator and hasattr(self.gpu_accelerator, 'available') and self.gpu_accelerator.available:
                hist_g, hist_h = self.gpu_accelerator.build_histograms(
                    X_subset,
                    node.gradients,
                    node.hessians,
                    feature_bin_edges,
                    self.n_bins,
                )
            else:
                hist_g, hist_h = compute_histograms(
                    X_subset,
                    node.gradients,
                    node.hessians,
                    feature_bin_edges,
                    self.n_bins,
                )

        if use_cache:
            self.histogram_cache.put(node.node_id, hist_g, hist_h)
        return hist_g, hist_h

    def _prepare_monotone_constraints(self, n_features):
        """Prepare monotone constraints array with bounds checking"""
        if self.monotone_constraints is None:
            return np.zeros(n_features, dtype=np.int8)
        arr = np.zeros(n_features, dtype=np.int8)
        for f, c in self.monotone_constraints.items():
            if 0 <= f < n_features:  # Bounds check
                arr[f] = np.int8(c)  # enforce -1, 0, +1
        return arr

    def _evaluate_split(self, X, node):
        """Optimized split evaluation with early termination"""
        # Early termination checks
        if (
            node.n_samples < self.min_samples_split
            or node.depth >= self.max_depth
            or node.h_sum < self.min_child_weight
        ):
            return False

        hist_g, hist_h = self._compute_histograms_with_reuse(X, node)
        node.histograms = (hist_g, hist_h)

        # Prepare monotone constraints only if needed
        mono_arr = None
        if self.monotone_constraints:
            mono_arr = self._prepare_monotone_constraints(len(self.feature_indices))

        # Use optimized split finding
        local_feat_idx, bin_idx, gain, go_left_if_missing = (
            find_best_splits_with_missing(
                hist_g,
                hist_h,
                self.lambda_,
                self.gamma,
                self.n_bins,
                self.min_child_weight,
                mono_arr,
            )
        )

        node.missing_go_left = bool(go_left_if_missing)

        if local_feat_idx == -1 or gain <= 0:
            return False

        # Bounds check for feature indices
        if local_feat_idx >= len(self.feature_indices):
            return False

        actual_feat_idx = self.feature_indices[local_feat_idx]
        threshold = self.bin_edges[actual_feat_idx][bin_idx]

        node.best_feature = actual_feat_idx
        node.best_threshold = threshold
        node.best_gain = gain
        node.best_bin_idx = bin_idx
        return True

    def _split_node(self, X, node):
        """Optimized node splitting with memory reuse"""
        if node.best_feature is None:
            return False

        local_feat_idx = self.feature_map[node.best_feature]
        feat_values = X[node.data_indices, local_feat_idx]

        # Always use raw threshold values for consistency
        threshold = node.best_threshold

        # Reuse mask array to reduce allocations
        if len(self._temp_mask) != len(feat_values):
            self._temp_mask = np.empty(len(feat_values), dtype=bool)

        mask = self._temp_mask

        # Handle missing values efficiently
        finite_mask = np.isfinite(feat_values)
        mask[:] = node.missing_go_left  # Default for missing values
        mask[finite_mask] = feat_values[finite_mask] <= threshold

        left_indices = node.data_indices[mask]
        right_indices = node.data_indices[~mask]

        # Minimum samples check
        if (
            len(left_indices) < self.min_samples_leaf
            or len(right_indices) < self.min_samples_leaf
        ):
            return False

        # Create children with optimized gradient/hessian slicing
        left_node_id = self._get_next_node_id()
        right_node_id = self._get_next_node_id()

        left_node = LeafNode(
            left_node_id,
            left_indices,
            node.gradients[mask],
            node.hessians[mask],
            node.depth + 1,
            parent_hist=node.histograms,
            is_left_child=True,
        )
        right_node = LeafNode(
            right_node_id,
            right_indices,
            node.gradients[~mask],
            node.hessians[~mask],
            node.depth + 1,
            parent_hist=node.histograms,
            is_left_child=False,
        )

        # Set sibling references
        left_node.sibling_node_id = right_node_id
        right_node.sibling_node_id = left_node_id

        # Update tree structure
        node.left_child = left_node
        node.right_child = right_node
        node.is_leaf = False

        self.nodes[left_node_id] = left_node
        self.nodes[right_node_id] = right_node
        return True

    def build_tree(self, X, gradients, hessians, feature_importance=None):
        """Optimized tree building with better heap management"""
        n_samples = len(gradients)
        root_indices = np.arange(n_samples, dtype=np.int64)

        root_id = self._get_next_node_id()
        self.root = LeafNode(root_id, root_indices, gradients, hessians, depth=0)
        self.nodes[root_id] = self.root

        # Evaluate root
        if self._evaluate_split(X, self.root):
            heapq.heappush(self.leaf_queue, self.root)

        leaves_created = 1

        # Optimized main loop with heap size monitoring
        while self.leaf_queue and leaves_created < self.max_leaves:
            best_leaf = heapq.heappop(self.leaf_queue)

            if self._split_node(X, best_leaf):
                leaves_created += 1

                # Update feature importance efficiently
                if feature_importance is not None:
                    feature_importance.update(
                        best_leaf.best_feature, best_leaf.best_gain, best_leaf.n_samples
                    )

                # Add children to queue with better filtering
                for child in [best_leaf.left_child, best_leaf.right_child]:
                    if (
                        child.n_samples >= self.min_samples_split
                        and child.depth < self.max_depth
                        and self._evaluate_split(X, child)
                    ):
                        heapq.heappush(self.leaf_queue, child)

        self._set_leaf_values()
        self._build_prediction_arrays()
        return self.root

    def _set_leaf_values(self):
        """Set leaf values for all leaf nodes"""
        for node in self.nodes.values():
            if node.is_leaf:
                node.leaf_value = calc_leaf_value_newton(
                    node.g_sum,
                    node.h_sum,
                    self.lambda_,
                    self.alpha,
                    self.max_delta_step,
                )

    def _build_prediction_arrays(self):
        """Build arrays for fast prediction - conservative approach"""
        try:
            if not self.nodes or not self.feature_indices:
                return
                
            max_node_id = max(self.nodes.keys())
            n_nodes = max_node_id + 1
            
            # Initialize arrays
            node_features = np.full(n_nodes, -1, dtype=np.int32)
            node_thresholds = np.full(n_nodes, np.nan, dtype=np.float64)
            node_missing_go_left = np.full(n_nodes, False, dtype=bool)
            left_children = np.full(n_nodes, -1, dtype=np.int32)
            right_children = np.full(n_nodes, -1, dtype=np.int32)
            leaf_values = np.full(n_nodes, 0.0, dtype=np.float64)
            is_leaf_flags = np.full(n_nodes, True, dtype=bool)
            
            # Build feature map array for numba
            max_feat = max(self.feature_indices) if self.feature_indices else 0
            feature_map_array = np.full(max_feat + 1, -1, dtype=np.int32)
            for global_feat, local_feat in self.feature_map.items():
                feature_map_array[global_feat] = local_feat
            
            # Fill arrays
            for node_id, node in self.nodes.items():
                if not node.is_leaf:
                    node_features[node_id] = node.best_feature
                    node_thresholds[node_id] = node.best_threshold
                    node_missing_go_left[node_id] = node.missing_go_left
                    left_children[node_id] = node.left_child.node_id
                    right_children[node_id] = node.right_child.node_id
                    is_leaf_flags[node_id] = False
                else:
                    leaf_values[node_id] = node.leaf_value
            
            self._prediction_arrays = (
                node_features, node_thresholds, node_missing_go_left,
                left_children, right_children, leaf_values, is_leaf_flags,
                feature_map_array
            )
        except Exception:
            # If building prediction arrays fails, just set to None
            # and fall back to regular prediction
            self._prediction_arrays = None

    def predict_node(self, X_sample, node):
        """Non-recursive node prediction (kept for compatibility)"""
        while not node.is_leaf:
            lf = self.feature_map[node.best_feature]
            x = X_sample[lf]
            if not np.isfinite(x):
                node = node.left_child if node.missing_go_left else node.right_child
            else:
                node = node.left_child if x <= node.best_threshold else node.right_child
        return node.leaf_value

    def predict(self, X):
        """Optimized prediction with fallback"""
        # Try fast prediction if arrays are available
        if self._prediction_arrays is not None:
            try:
                (node_features, node_thresholds, node_missing_go_left,
                 left_children, right_children, leaf_values, is_leaf_flags,
                 feature_map_array) = self._prediction_arrays
                
                return predict_tree_numba(
                    X, node_features, node_thresholds, node_missing_go_left,
                    left_children, right_children, leaf_values, is_leaf_flags,
                    feature_map_array, self.root.node_id
                )
            except Exception:
                # Fall back to regular prediction if numba fails
                pass

        # Fallback: vectorized prediction with batch processing
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=np.float64)

        # Batch process for better cache locality
        batch_size = min(1024, n_samples)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            for j in range(i, end_idx):
                predictions[j] = self.predict_node(X[j], self.root)

        return predictions

    def predict_vectorized(self, X: np.ndarray) -> np.ndarray:
        """Vectorized tree traversal - try optimized first, then fallback"""
        # Try optimized prediction first
        if self._prediction_arrays is not None:
            try:
                (node_features, node_thresholds, node_missing_go_left,
                 left_children, right_children, leaf_values, is_leaf_flags,
                 feature_map_array) = self._prediction_arrays
                
                return predict_tree_numba(
                    X, node_features, node_thresholds, node_missing_go_left,
                    left_children, right_children, leaf_values, is_leaf_flags,
                    feature_map_array, self.root.node_id
                )
            except Exception:
                pass

        # Fallback: original vectorized approach
        n_samples = X.shape[0]
        preds = np.zeros(n_samples, dtype=np.float64)

        # Worklist: queue of (node, row_indices)
        stack = [(self.root, np.arange(n_samples, dtype=np.int64))]

        while stack:
            node, row_idx = stack.pop()

            if node.is_leaf:
                preds[row_idx] = node.leaf_value
                continue

            lf = self.feature_map[node.best_feature]
            feat_vals = X[row_idx, lf]

            # Handle missing values
            finite_mask = np.isfinite(feat_vals)
            go_left = np.full(row_idx.shape, node.missing_go_left, dtype=bool)
            go_left[finite_mask] = feat_vals[finite_mask] <= node.best_threshold

            # Partition
            left_idx = row_idx[go_left]
            right_idx = row_idx[~go_left]

            if left_idx.size > 0:
                stack.append((node.left_child, left_idx))
            if right_idx.size > 0:
                stack.append((node.right_child, right_idx))

        return preds


# ================ ENHANCED SINGLE TREE WITH LEAF-WISE GROWTH ================
class LeafWiseSingleTree:
    """SingleTree that uses leaf-wise growth for better performance"""

    def __init__(
        self,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        lambda_=1.0,
        gamma=0.0,
        alpha=0.0,
        feature_indices=None,
        n_jobs=4,
        tree_method="hist",
        n_bins=256,
        bin_edges=None,
        monotone_constraints=None,
        interaction_constraints=None,
        gpu_accelerator=None,
        max_delta_step=0.0,
        max_leaves=31,
        min_child_weight=1e-3,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.feature_indices = feature_indices
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.monotone_constraints = monotone_constraints or {}
        self.interaction_constraints = interaction_constraints
        self.gpu_accelerator = gpu_accelerator
        self.max_delta_step = max_delta_step
        self.max_leaves = max_leaves
        self.min_child_weight = min_child_weight
        self._binned = None  # injected by BoostRegressor
        self._row_indexer = None  # (n_sub,) global row ids for this tree
        self._feature_mask = None  # (n_local_features,)
        self._binned_local = None  # (n_sub, n_local_features) built in fit
        # Tree builder
        self.builder = None
        self.root = None

    def fit(self, X, g, h, depth=0, feature_importance=None):
        """Optimized fit with better memory management"""
        # Build optimized binned view
        if (
            self._binned is not None
            and self._row_indexer is not None
            and self._feature_mask is not None
        ):
            # Use advanced indexing with proper ordering
            self._binned_local = self._binned[
                np.ix_(self._row_indexer, self._feature_mask)
            ]
        else:
            self._binned_local = None

        # Create optimized builder
        self.builder = LeafWiseTreeBuilder(
            max_leaves=self.max_leaves,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            lambda_=self.lambda_,
            gamma=self.gamma,
            alpha=self.alpha,
            feature_indices=self.feature_indices,
            n_bins=self.n_bins,
            bin_edges=self.bin_edges,
            gpu_accelerator=self.gpu_accelerator,
            monotone_constraints=self.monotone_constraints,
            max_delta_step=self.max_delta_step,
            min_child_weight=self.min_child_weight,
        )

        self.builder.binned = self._binned
        self.builder.binned_local = self._binned_local

        # Build tree
        self.root = self.builder.build_tree(X, g, h, feature_importance)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.builder is None or self.root is None:
            return np.zeros(X.shape[0])
        return self.builder.predict_vectorized(X)
