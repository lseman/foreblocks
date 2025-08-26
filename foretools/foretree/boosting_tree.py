from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# -------------------------------------------------------------------
# Keep your existing imports/utilities
from boosting_aux import *
from boosting_aux import _leaf_objective_optimal, _leaf_value_from_sums
from boosting_bin import *
from boosting_loss import *
from numba import njit

# -------------------------------------------------------------------


# ================= SAFE PREDICTION (unchanged core) =================
@njit
def predict_tree_numba(
    X,
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
            go_left = node_missing_go_left[node_idx] if np.isnan(x) else (x <= thr)
            node_idx = left_children[node_idx] if go_left else right_children[node_idx]
        else:
            predictions[i] = 0.0
    return predictions


# =============== UNIFIED / SLOTTED NODE ===============
@dataclass(slots=True)
class TreeNode:
    node_id: int
    data_indices: np.ndarray
    gradients: np.ndarray
    hessians: np.ndarray
    depth: int
    parent_hist: Optional[Tuple[np.ndarray, np.ndarray]] = None
    is_left_child: bool = False

    # computed
    n_samples: int = 0
    g_sum: float = 0.0
    h_sum: float = 0.0

    # split info
    best_feature: Optional[int] = None
    best_threshold: float = np.nan
    best_gain: float = -np.inf
    best_bin_idx: Optional[int] = None
    missing_go_left: bool = False
    histograms: Optional[Tuple[np.ndarray, np.ndarray]] = None
    sibling_node_id: Optional[int] = None

    # structure
    left_child: Optional["TreeNode"] = None
    right_child: Optional["TreeNode"] = None
    leaf_value: Optional[float] = None
    is_leaf: bool = True

    _prune_leaves: int = 0
    _prune_internal: int = 0
    _prune_R_subtree: float = 0.0
    _prune_R_collapse: float = 0.0
    _prune_alpha_star: float = np.inf

    sorted_lists: Optional[List[np.ndarray]] = None  # len = n_local_features

    def init_sums(self):
        self.n_samples = len(self.data_indices)
        self.g_sum = float(np.sum(self.gradients)) if self.gradients.size else 0.0
        self.h_sum = float(np.sum(self.hessians)) if self.hessians.size else 0.0


# ================= UNIFIED TREE =================
class UnifiedTree:
    """
    Single-class tree with strategy dispatch (hist/exact, leaf/level).
    Minimal changes: exact now uses presorted indices + global g/h.
    """

    # ---- construction ----
    def __init__(
        self,
        growth_policy="leaf_wise",
        max_depth=6,
        max_leaves=31,
        min_samples_split=10,
        min_samples_leaf=5,
        min_child_weight=1e-3,
        lambda_=1.0,
        gamma=0.0,
        alpha=0.0,
        max_delta_step=0.0,
        tree_method="hist",
        n_bins=256,
        feature_indices=None,
        bin_edges=None,
        monotone_constraints=None,
        interaction_constraints=None,
        gpu_accelerator=None,
        n_jobs=4,
        adaptive_hist=False,
    ):

        if growth_policy not in ("leaf_wise", "level_wise"):
            raise ValueError("growth_policy must be 'leaf_wise' or 'level_wise'")
        if tree_method not in ("hist", "exact"):
            raise ValueError("tree_method must be 'hist' or 'exact'")

        # params
        self.growth_policy = growth_policy
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.max_delta_step = max_delta_step

        self.tree_method = tree_method
        self.n_bins = n_bins
        self.feature_indices = np.array(
            feature_indices if feature_indices is not None else [], dtype=np.int32
        )
        self.bin_edges = bin_edges
        self.monotone_constraints: Dict[int, int] = monotone_constraints or {}
        self.interaction_constraints = interaction_constraints
        self.gpu_accelerator = gpu_accelerator
        self.n_jobs = n_jobs

        # state
        self.root: Optional[TreeNode] = None
        self.nodes: Dict[int, TreeNode] = {}
        self.next_node_id = 0
        self.histogram_cache = (
            HistogramCache() if "HistogramCache" in globals() else None
        )

        # binned views prepared by caller
        self.binned_local = None

        if feature_indices is not None:
            self.feature_indices = np.array(feature_indices, dtype=np.int32)
        else:
            self.feature_indices = np.array([], dtype=np.int32)

        # FIXED: Build feature map more robustly
        if self.feature_indices.size > 0:
            self.feature_map = {int(f): i for i, f in enumerate(self.feature_indices)}
        else:
            self.feature_map = {}

        # prepared monotone vector for fast checks (local indices)
        self._mono_local = None
        if self.monotone_constraints and self.feature_indices.size:
            self._mono_local = np.zeros(len(self.feature_indices), dtype=np.int8)
            for g, c in self.monotone_constraints.items():
                if g in self.feature_map:
                    self._mono_local[self.feature_map[g]] = np.int8(c)

        # dispatch
        # ---- dispatch ----
        if self.tree_method == "hist":
            if getattr(self, "adaptive_hist", False):
                self._eval_split: Callable[[np.ndarray, TreeNode], bool] = self._eval_hist_adaptive
            else:
                self._eval_split: Callable[[np.ndarray, TreeNode], bool] = self._eval_hist
        elif self.tree_method == "exact":
            self._eval_split: Callable[[np.ndarray, TreeNode], bool] = self._eval_exact
        else:
            raise ValueError("tree_method must be 'hist' or 'exact'")

        self._grow: Callable[
            [np.ndarray, np.ndarray, np.ndarray, Optional[object]], None
        ] = (
            self._grow_leaf_wise
            if self.growth_policy == "leaf_wise"
            else self._grow_level_wise
        )

        # prediction arrays
        self._pred_arrays = None

        # ---- NEW: exact presort buffers & global g/h ----
        self._g_global: Optional[np.ndarray] = None
        self._h_global: Optional[np.ndarray] = None
        self._sorted_finite_idx: Optional[List[np.ndarray]] = (
            None  # per feature (global index)
        )
        self._sorted_missing_idx: Optional[List[np.ndarray]] = (
            None  # per feature (global index)
        )
        self._node_of_sample: Optional[np.ndarray] = None  # membership by node_id

        self._mono_constraints_array = None

        self._work_buffers = {}
        self._partition_cache = {}

        # IT IS EXPERIMENTAL, CAN EXPLODE!!!!
        self.adaptive_hist = adaptive_hist  # turn on to use adaptive per-feature bins
        self._adaptive_binner = AdaptiveMultiLevelBinner()
        self._adaptive_edges = None  # dict: gfi -> np.ndarray of edges

    # ---- small utilities ----
    def _new_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def _hist_from_cache_or_build(
        self, X: np.ndarray, node: TreeNode
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.histogram_cache:
            cached = self.histogram_cache.get(node.node_id)
            if cached is not None:
                return cached

        # sibling subtraction
        if (
            node.parent_hist is not None
            and node.sibling_node_id is not None
            and self.histogram_cache
        ):
            sib = self.histogram_cache.get(node.sibling_node_id)
            if sib is not None:
                ph_g, ph_h = node.parent_hist
                sh_g, sh_h = sib
                hg, hh = subtract_histograms(ph_g, ph_h, sh_g, sh_h)
                self.histogram_cache.put(node.node_id, hg, hh)
                return hg, hh

        # compute fresh
        if self.binned_local is not None:
            idx = node.data_indices.astype(np.int64, copy=False)
            hg, hh = compute_histograms_from_binned(
                self.binned_local,
                idx,
                node.gradients,
                node.hessians,
                n_features=len(self.feature_indices),
                n_bins=self.n_bins,
            )
        else:
            Xs = X[node.data_indices]
            fe = np.array([self.bin_edges[i] for i in self.feature_indices])
            if self.gpu_accelerator and getattr(
                self.gpu_accelerator, "available", False
            ):
                hg, hh = self.gpu_accelerator.build_histograms(
                    Xs, node.gradients, node.hessians, fe, self.n_bins
                )
            else:
                hg, hh = compute_histograms(
                    Xs, node.gradients, node.hessians, fe, self.n_bins
                )

        if self.histogram_cache:
            self.histogram_cache.put(node.node_id, hg, hh)
        return hg, hh

    def _eval_hist_adaptive(self, X: np.ndarray, node: TreeNode) -> bool:
        """
        Per-feature adaptive bins. Builds histograms on the fly per feature
        for this node’s samples; handles missing->left/right and selects best.
        """
        best_gain = -np.inf
        best_feature = -1
        best_thr = np.nan
        best_miss_left = False

        g_all = node.gradients  # float32
        h_all = node.hessians   # float32
        idx = node.data_indices.astype(np.int64, copy=False)

        for lfi, gfi in enumerate(self.feature_indices):
            gfi = int(gfi)
            lf = self.feature_map.get(gfi, gfi)
            edges = self._adaptive_edges.get(gfi, None)
            if edges is None or edges.shape[0] < 2:
                continue

            vals = X[idx, lf].astype(np.float64, copy=False)
            g = g_all
            h = h_all

            # build finite-only histogram for this node/feature
            hist_g, hist_h = _build_histogram_numba(vals, g, h, edges)

            # missing aggregates for this node/feature
            finite_mask = np.isfinite(vals)
            if finite_mask.any():
                g_miss = float(np.sum(g[~finite_mask], dtype=np.float64))
                h_miss = float(np.sum(h[~finite_mask], dtype=np.float64))
            else:
                g_miss = float(np.sum(g, dtype=np.float64))
                h_miss = float(np.sum(h, dtype=np.float64))

            # find best split (bin idx) + policy for missing
            gain, split_bin, miss_left = _best_split_from_hist_with_missing_numba(
                hist_g, hist_h, g_miss, h_miss,
                self.lambda_, self.gamma, self.min_child_weight
            )

            if gain > best_gain and split_bin >= 0:
                # threshold is the right edge of split_bin (or left of next)
                thr = edges[split_bin + 1]
                best_gain = gain
                best_feature = gfi
                best_thr = float(thr)
                best_miss_left = bool(miss_left)

        if best_feature == -1 or best_gain <= 0.0:
            return False

        node.best_feature = int(best_feature)
        node.best_threshold = float(best_thr)
        node.best_gain = float(best_gain)
        node.best_bin_idx = None
        node.missing_go_left = bool(best_miss_left)
        node.histograms = None  # not used in adaptive path
        return True


    # ---- histogram evaluator (unchanged) ----
    def _eval_hist(self, X: np.ndarray, node: TreeNode) -> bool:
        if (
            node.n_samples < self.min_samples_split
            or node.depth >= self.max_depth
            or node.h_sum < self.min_child_weight
        ):
            return False

        hg, hh = self._hist_from_cache_or_build(X, node)
        node.histograms = (hg, hh)

        mono = self._mono_local
        lfi, bin_idx, gain, missing_left = find_best_splits_with_missing(
            hg, hh, self.lambda_, self.gamma, self.n_bins, self.min_child_weight, mono
        )

        if lfi == -1 or gain <= 0:
            return False

        gfi = int(self.feature_indices[lfi])
        thr = self.bin_edges[gfi][bin_idx]

        node.best_feature = gfi
        node.best_threshold = float(thr)
        node.best_gain = float(gain)
        node.best_bin_idx = int(bin_idx)
        node.missing_go_left = bool(missing_left)
        return True

    def _prepare_presort_exact(self, X: np.ndarray):
        """Optimized presort with better memory layout"""
        n, p = X.shape

        if self.feature_indices.size > 0:
            features_to_sort = self.feature_indices.astype(np.int32)
        else:
            features_to_sort = np.arange(p, dtype=np.int32)

        max_feat = int(features_to_sort.max()) if features_to_sort.size > 0 else 0
        self._mono_constraints_array = np.zeros(max_feat + 1, dtype=np.int8)
        for gfi, constraint in self.monotone_constraints.items():
            if gfi <= max_feat:
                self._mono_constraints_array[gfi] = np.int8(constraint)

        # Pre-allocate work buffers
        self._work_buffers = {
            "mark_left": np.zeros(n, dtype=np.uint8),
            "mark_node": np.zeros(n, dtype=np.uint8),
            "temp_indices": np.empty(n, dtype=np.int32),
        }

        # Use more efficient sorting and storage
        self._sorted_idx = {}
        self._missing_idx = {}

        for gfi in features_to_sort:
            gfi = int(gfi)
            lf = self.feature_map.get(gfi, gfi)
            if lf >= p:
                self._sorted_idx[gfi] = np.empty(0, dtype=np.int32)
                self._missing_idx[gfi] = np.empty(0, dtype=np.int32)
                continue

            col = X[:, lf]
            # Use mergesort for better cache performance on partially sorted data
            order = np.argsort(col, kind="mergesort").astype(np.int32)

            # More efficient finite check
            finite_mask = np.isfinite(col[order])
            finite_end = np.sum(finite_mask)

            self._sorted_idx[gfi] = order[:finite_end]
            self._missing_idx[gfi] = (
                order[finite_end:] if finite_end < n else np.empty(0, dtype=np.int32)
            )

        self._node_of_sample = np.full(n, -1, dtype=np.int32)

    def _eval_exact(self, X: np.ndarray, node: TreeNode) -> bool:
        # Basic stopping
        if (
            node.n_samples < self.min_samples_split
            or node.depth >= self.max_depth
            or node.h_sum < self.min_child_weight
        ):
            return False

        g = self._g_global
        h = self._h_global

        best_gain = -np.inf
        best_feature = -1
        best_thr = np.nan
        best_miss_left = False

        # Cheap optimistic upper bound: if even a perfect split cannot beat current best, skip the feature
        G = node.g_sum
        H = node.h_sum
        leaf_upper_bound = (G * G) / (H + self.lambda_) if H > 0.0 else 0.0

        # Iterate over local feature indices mapped to global ones
        for lfi, gfi in enumerate(self.feature_indices):
            gfi = int(gfi)
            lst = node.sorted_lists[lfi] if node.sorted_lists is not None else None
            if lst is None or lst.size < 2 * self.min_samples_leaf:
                continue

            # UB pruning per feature (same cheap bound works feature-agnostically)
            if leaf_upper_bound <= best_gain:
                # No feature can beat current best if this is already tight; continue to next (kept here for clarity)
                pass  # (kept for readability; we still evaluate in case of very loose bound)
            # NOTE: If you want aggressive pruning, uncomment the next two lines:
            # if leaf_upper_bound <= best_gain:
            #     continue

            # Local feature index
            lf = self.feature_map.get(gfi, gfi)

            # Values for this node in feature-order (already sorted). Avoid dtype copy.
            vals_seq = X[lst, lf]

            # Missing indices for this feature (global)
            miss_idx = self._missing_idx.get(gfi, np.empty(0, dtype=np.int32))

            # Aggregate missing that belong to THIS node via intersection with the node's feature-sorted list.
            if miss_idx.size:
                # Intersect on sorted arrays; assume_unique=False is safe here.
                missing_in_node = np.intersect1d(miss_idx, lst, assume_unique=False)
                if missing_in_node.size:
                    g_miss = float(np.sum(g[missing_in_node]))
                    h_miss = float(np.sum(h[missing_in_node]))
                    n_miss = int(missing_in_node.size)
                else:
                    g_miss = 0.0
                    h_miss = 0.0
                    n_miss = 0
            else:
                g_miss = 0.0
                h_miss = 0.0
                n_miss = 0

            # Monotone constraint (if any) for this global feature index
            mono = 0
            if self._mono_constraints_array is not None and gfi < len(
                self._mono_constraints_array
            ):
                mono = int(self._mono_constraints_array[gfi])

            # Find best split along this pre-sorted list
            gain, thr, miss_left, _, _ = best_split_on_feature_list(
                lst,
                vals_seq,
                g,
                h,
                g_miss,
                h_miss,
                n_miss,
                self.min_samples_leaf,
                self.min_child_weight,
                self.lambda_,
                self.gamma,
                mono,
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = gfi
                best_thr = thr
                best_miss_left = bool(miss_left)

        if best_feature == -1 or best_gain <= 0.0:
            return False

        node.best_feature = int(best_feature)
        node.best_threshold = float(best_thr)
        node.best_gain = float(best_gain)
        node.best_bin_idx = None
        node.missing_go_left = bool(best_miss_left)
        return True

    def _apply_split(
        self, X: np.ndarray, node: TreeNode
    ) -> Optional[Tuple[TreeNode, TreeNode]]:
        """Unified split application handling both hist and exact methods optimally."""
        if node.best_feature is None:
            return None

        gfi = int(node.best_feature)
        lf = self.feature_map[gfi]
        threshold = float(node.best_threshold)
        missing_left = bool(node.missing_go_left)

        # ==================== EXACT METHOD PATH ====================
        if self.tree_method == "exact" and node.sorted_lists is not None:
            n_total = X.shape[0]

            # Reuse pre-allocated buffers
            mark_left = self._work_buffers["mark_left"]
            mark_left[:] = 0

            lfi_split = self.feature_map[gfi]
            parent_list_on_f = node.sorted_lists[lfi_split]
            if parent_list_on_f is None or parent_list_on_f.size == 0:
                return None

            vals_f = X[parent_list_on_f, lf]
            finite_mask = np.isfinite(vals_f)
            left_mask = finite_mask & (vals_f <= threshold)

            if np.any(left_mask):
                mark_left[parent_list_on_f[left_mask]] = 1

            # Missing handling: intersect missing idx with this node’s feature-sorted list
            miss_f = self._missing_idx.get(gfi, np.empty(0, dtype=np.int32))
            if miss_f.size > 0 and missing_left:
                missing_in_node = np.intersect1d(
                    miss_f, parent_list_on_f, assume_unique=False
                )
                if missing_in_node.size:
                    mark_left[missing_in_node] = 1

            # Create left/right children indices
            node_left_mask = mark_left[node.data_indices] == 1
            li = node.data_indices[node_left_mask]
            ri = node.data_indices[~node_left_mask]

            if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
                return None

            lid, rid = self._new_id(), self._new_id()
            ln = TreeNode(
                lid,
                li,
                node.gradients[node_left_mask],
                node.hessians[node_left_mask],
                node.depth + 1,
                parent_hist=node.histograms,
                is_left_child=True,
            )
            rn = TreeNode(
                rid,
                ri,
                node.gradients[~node_left_mask],
                node.hessians[~node_left_mask],
                node.depth + 1,
                parent_hist=node.histograms,
                is_left_child=False,
            )

            ln.sibling_node_id, rn.sibling_node_id = rid, lid
            ln.init_sums()
            rn.init_sums()

            # Partition sorted lists efficiently
            nlf = len(self.feature_indices)
            ln.sorted_lists = [None] * nlf
            rn.sorted_lists = [None] * nlf
            for lfi in range(nlf):
                parent_list = node.sorted_lists[lfi]
                if parent_list is None or parent_list.size == 0:
                    ln.sorted_lists[lfi] = np.empty(0, dtype=np.int32)
                    rn.sorted_lists[lfi] = np.empty(0, dtype=np.int32)
                    continue
                child_left_mask = mark_left[parent_list] == 1
                ln.sorted_lists[lfi] = parent_list[child_left_mask]
                rn.sorted_lists[lfi] = parent_list[~child_left_mask]

            if self._node_of_sample is not None:
                self._node_of_sample[li] = lid
                self._node_of_sample[ri] = rid

        # ==================== HISTOGRAM METHOD PATH ====================
        else:
            # (unchanged from your version)
            vals = X[node.data_indices, lf]
            finite_mask = np.isfinite(vals)
            split_mask = np.full(vals.shape[0], missing_left, dtype=bool)
            split_mask[finite_mask] = vals[finite_mask] <= threshold
            li = node.data_indices[split_mask]
            ri = node.data_indices[~split_mask]
            if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
                return None
            lid, rid = self._new_id(), self._new_id()
            ln = TreeNode(
                lid,
                li,
                node.gradients[split_mask],
                node.hessians[split_mask],
                node.depth + 1,
                parent_hist=node.histograms,
                is_left_child=True,
            )
            rn = TreeNode(
                rid,
                ri,
                node.gradients[~split_mask],
                node.hessians[~split_mask],
                node.depth + 1,
                parent_hist=node.histograms,
                is_left_child=False,
            )
            ln.sibling_node_id, rn.sibling_node_id = rid, lid
            ln.init_sums()
            rn.init_sums()
            ln.sorted_lists = None
            rn.sorted_lists = None

        # ==================== COMMON FINALIZATION ====================
        node.left_child, node.right_child = ln, rn
        node.is_leaf = False
        self.nodes[lid], self.nodes[rid] = ln, rn
        return ln, rn

    def _grow_leaf_wise(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, fi=None):
        n = len(g)

        # Cast per-node payload to float32 (bandwidth win); keep accumulations in float64
        g32 = g.astype(np.float32, copy=False)
        h32 = h.astype(np.float32, copy=False)

        root_idx = np.arange(n, dtype=np.int64)
        rid = self._new_id()
        root = TreeNode(rid, root_idx, g32, h32, depth=0)
        root.init_sums()
        self.root = root
        self.nodes[rid] = root

        # EXACT: initialize root's per-feature sorted lists
        if self.tree_method == "exact":
            nlf = len(self.feature_indices)
            root.sorted_lists = [None] * nlf
            for lfi, gfi in enumerate(self.feature_indices):
                root.sorted_lists[lfi] = self._sorted_idx[int(gfi)]

            # Track membership (optional)
            if self._node_of_sample is not None:
                self._node_of_sample[:] = -1
                self._node_of_sample[root_idx] = rid

        heap = []
        if self._eval_split(X, root):
            # root.histograms is set inside _eval_hist (hist mode)
            heapq.heappush(heap, (-root.best_gain, root.node_id, root))
        else:
            # No split possible → leaf
            root.leaf_value = calc_leaf_value_newton(
                root.g_sum, root.h_sum, self.lambda_, self.alpha, self.max_delta_step
            )

        leaves = 1
        while heap and leaves < self.max_leaves:
            _, _, best = heapq.heappop(heap)

            children = self._apply_split(X, best)
            if not children:
                # finalize as leaf
                best.leaf_value = calc_leaf_value_newton(
                    best.g_sum,
                    best.h_sum,
                    self.lambda_,
                    self.alpha,
                    self.max_delta_step,
                )
                continue

            ln, rn = children
            leaves += 1

            # HIST: ensure children know parent's hist for sibling subtraction
            if self.tree_method == "hist" and best.histograms is not None:
                ln.parent_hist = best.histograms
                rn.parent_hist = best.histograms

            # Evaluate children; push only if splittable
            for ch in (ln, rn):
                if (
                    ch.n_samples < self.min_samples_split
                    or ch.depth >= self.max_depth
                    or ch.h_sum < self.min_child_weight
                ):
                    ch.leaf_value = calc_leaf_value_newton(
                        ch.g_sum,
                        ch.h_sum,
                        self.lambda_,
                        self.alpha,
                        self.max_delta_step,
                    )
                    continue
                if self._eval_split(X, ch):
                    heapq.heappush(heap, (-ch.best_gain, ch.node_id, ch))
                else:
                    ch.leaf_value = calc_leaf_value_newton(
                        ch.g_sum,
                        ch.h_sum,
                        self.lambda_,
                        self.alpha,
                        self.max_delta_step,
                    )

    def _grow_level_wise(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, fi=None):
        n = len(g)

        # Cast per-node payload to float32 (bandwidth win); keep accumulations in float64
        g32 = g.astype(np.float32, copy=False)
        h32 = h.astype(np.float32, copy=False)

        rid = self._new_id()
        root = TreeNode(rid, np.arange(n, dtype=np.int64), g32, h32, depth=0)
        root.init_sums()
        self.root = root
        self.nodes[rid] = root

        # EXACT: initialize root's per-feature sorted lists
        if self.tree_method == "exact":
            nlf = len(self.feature_indices)
            root.sorted_lists = [None] * nlf
            for lfi, gfi in enumerate(self.feature_indices):
                root.sorted_lists[lfi] = self._sorted_idx[int(gfi)]

            if self._node_of_sample is not None:
                self._node_of_sample[:] = -1
                self._node_of_sample[root.data_indices] = rid

        q = [root]
        while q:
            node = q.pop(0)

            stop = (
                node.depth >= self.max_depth
                or node.n_samples < self.min_samples_split
                or node.h_sum < self.min_child_weight
            )
            if stop or not self._eval_split(X, node):
                node.leaf_value = calc_leaf_value_newton(
                    node.g_sum,
                    node.h_sum,
                    self.lambda_,
                    self.alpha,
                    self.max_delta_step,
                )
                continue

            children = self._apply_split(X, node)
            if not children:
                node.leaf_value = calc_leaf_value_newton(
                    node.g_sum,
                    node.h_sum,
                    self.lambda_,
                    self.alpha,
                    self.max_delta_step,
                )
                continue

            ln, rn = children

            # HIST: ensure sibling subtraction path is primed
            if self.tree_method == "hist" and node.histograms is not None:
                ln.parent_hist = node.histograms
                rn.parent_hist = node.histograms

            q.extend((ln, rn))

    # ---- public API ----
    def fit(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        depth: int = 0,
        feature_importance=None,
    ):
        # reset
        self.nodes.clear()
        self.next_node_id = 0
        self.root = None
        self._pred_arrays = None

        # ---- exact vs hist setup
        if self.tree_method == "exact":
            self._g_global = g
            self._h_global = h
            self._prepare_presort_exact(X)
            # hist-specific state off
            self._adaptive_edges = None
        else:
            self._g_global = None
            self._h_global = None
            self._sorted_finite_idx = None
            self._sorted_missing_idx = None
            self._node_of_sample = None

            # Precompute per-feature adaptive edges (once per tree), if enabled
            if getattr(self, "adaptive_hist", False):
                if not hasattr(self, "_adaptive_binner") or self._adaptive_binner is None:
                    self._adaptive_binner = AdaptiveMultiLevelBinner()

                self._adaptive_edges = {}

                # which features?
                if self.feature_indices.size == 0:
                    feats = np.arange(X.shape[1], dtype=np.int32)
                else:
                    feats = self.feature_indices.astype(np.int32, copy=False)

                g32 = g.astype(np.float32, copy=False)
                h32 = h.astype(np.float32, copy=False)

                for gfi in feats:
                    lf = self.feature_map.get(int(gfi), int(gfi))
                    col = X[:, lf].astype(np.float64, copy=False)

                    edges, _ = self._adaptive_binner.create_adaptive_bins(
                        col, g32, h32,
                        feature_idx=int(gfi),
                        lambda_reg=self.lambda_,
                        gamma=self.gamma,
                    )

                    # robust fallback: handle columns that are all-NaN or constant
                    if edges.shape[0] < 2:
                        if np.all(np.isnan(col)):
                            edges = np.array([0.0, 1.0], dtype=np.float64)
                        else:
                            vmin = float(np.nanmin(col))
                            vmax = float(np.nanmax(col))
                            if not np.isfinite(vmin) or not np.isfinite(vmax):
                                edges = np.array([0.0, 1.0], dtype=np.float64)
                            elif vmin == vmax:
                                edges = np.array([vmin, vmax], dtype=np.float64)
                            else:
                                edges = np.array([vmin, vmax], dtype=np.float64)

                    self._adaptive_edges[int(gfi)] = edges
            else:
                self._adaptive_edges = None

        # if caller prepared binned data into self._binned/_row_indexer/_feature_mask, set self.binned_local here
        if (
            hasattr(self, "_binned")
            and hasattr(self, "_row_indexer")
            and hasattr(self, "_feature_mask")
        ):
            if (
                self._binned is not None
                and self._row_indexer is not None
                and self._feature_mask is not None
            ):
                self.binned_local = self._binned[np.ix_(self._row_indexer, self._feature_mask)]
        else:
            self.binned_local = None

        # grow
        self._grow(X, g, h, feature_importance)

        # set leaf values for any leftover leaf without value
        for n in self.nodes.values():
            if n.is_leaf and n.leaf_value is None:
                n.leaf_value = calc_leaf_value_newton(
                    n.g_sum, n.h_sum, self.lambda_, self.alpha, self.max_delta_step
                )

        self._build_pred_arrays()
        # make sure arrays are contiguous for numba
        if self._pred_arrays is not None:
            self._pred_arrays = tuple(
                np.ascontiguousarray(a) if isinstance(a, np.ndarray) else a
                for a in self._pred_arrays
            )

        return self

    def _build_pred_arrays(self):
        if not self.nodes:
            self._pred_arrays = None
            return

        max_id = max(self.nodes.keys())
        N = max_id + 1
        node_features = np.full(N, -1, dtype=np.int32)
        node_thresholds = np.full(N, np.nan, dtype=np.float64)
        node_missing_go_left = np.zeros(N, dtype=np.bool_)
        left_children = np.full(N, -1, dtype=np.int32)
        right_children = np.full(N, -1, dtype=np.int32)
        leaf_values = np.zeros(N, dtype=np.float64)
        is_leaf_flags = np.ones(N, dtype=np.bool_)

        max_feat = int(self.feature_indices.max()) if self.feature_indices.size else -1
        fmap = np.full(max_feat + 1 if max_feat >= 0 else 1, -1, dtype=np.int32)
        for g, l in self.feature_map.items():
            if g >= 0:
                if g >= fmap.size:
                    # expand if features aren't dense
                    new = np.full(g + 1, -1, dtype=np.int32)
                    new[: fmap.size] = fmap
                    fmap = new
                fmap[g] = l

        for nid, nd in self.nodes.items():
            if nd.is_leaf:
                leaf_values[nid] = nd.leaf_value or 0.0
            else:
                node_features[nid] = (
                    nd.best_feature if nd.best_feature is not None else -1
                )
                node_thresholds[nid] = nd.best_threshold
                node_missing_go_left[nid] = nd.missing_go_left
                left_children[nid] = nd.left_child.node_id if nd.left_child else -1
                right_children[nid] = nd.right_child.node_id if nd.right_child else -1
                is_leaf_flags[nid] = False

        self._pred_arrays = (
            node_features,
            node_thresholds,
            node_missing_go_left,
            left_children,
            right_children,
            leaf_values,
            is_leaf_flags,
            fmap,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        if self._pred_arrays is not None:
            nf, nt, nm, lc, rc, lv, lf, fmap = self._pred_arrays
            try:
                return predict_tree_numba(
                    X, nf, nt, nm, lc, rc, lv, lf, fmap, self.root.node_id
                )
            except Exception:
                pass
        # simple fallback (stack loop)
        out = np.zeros(X.shape[0], dtype=np.float64)
        stack: List[Tuple[TreeNode, np.ndarray]] = [
            (self.root, np.arange(X.shape[0], dtype=np.int64))
        ]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                out[idx] = node.leaf_value
                continue
            lf = self.feature_map[node.best_feature]
            vals = X[idx, lf]
            finite = np.isfinite(vals)
            go_left = np.full(idx.shape[0], node.missing_go_left, dtype=bool)
            go_left[finite] = vals[finite] <= node.best_threshold
            li = idx[go_left]
            ri = idx[~go_left]
            if li.size:
                stack.append((node.left_child, li))
            if ri.size:
                stack.append((node.right_child, ri))
        return out

    # convenience
    def get_feature_importance(self):
        imp: Dict[int, float] = {}
        for nd in self.nodes.values():
            if not nd.is_leaf and nd.best_feature is not None and nd.best_gain > 0:
                imp[nd.best_feature] = (
                    imp.get(nd.best_feature, 0.0) + nd.best_gain * nd.n_samples
                )
        return imp

    def get_depth(self) -> int:
        if not self.root:
            return 0
        maxd = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            maxd = max(maxd, n.depth)
            if not n.is_leaf:
                stack.append(n.left_child)
                stack.append(n.right_child)
        return maxd

    def get_n_leaves(self) -> int:
        return sum(1 for n in self.nodes.values() if n.is_leaf)

    def post_prune_ccp(self, ccp_alpha: float) -> None:
        """
        Cost-complexity post-pruning using the same objective as training:
        soft-thresholded Newton step + optional max_delta_step clipping,
        and gamma penalty per split.
        """
        if not self.root:
            return

        # ----- Bottom-up accumulation of subtree stats -----
        def _acc(node):
            if node is None:
                return 0, 0, 0.0  # n_leaves, n_internal, R_subtree

            if node.is_leaf:
                R_leaf = _leaf_objective_optimal(
                    node.g_sum,
                    node.h_sum,
                    self.lambda_,
                    self.alpha,
                    self.max_delta_step,
                )
                node._prune_leaves = 1
                node._prune_internal = 0
                node._prune_R_subtree = R_leaf
                node._prune_R_collapse = R_leaf
                node._prune_alpha_star = np.inf
                return 1, 0, R_leaf

            # children
            _acc(node.left_child)
            _acc(node.right_child)

            n_leaves = node.left_child._prune_leaves + node.right_child._prune_leaves
            n_internal = (
                node.left_child._prune_internal + node.right_child._prune_internal + 1
            )

            # subtree objective = sum leaf objectives - gamma * (#splits in this subtree)
            R_subtree = (
                node.left_child._prune_R_subtree + node.right_child._prune_R_subtree
            ) - self.gamma * 1.0

            # collapsed objective at this node as single leaf
            R_collapse = _leaf_objective_optimal(
                node.g_sum, node.h_sum, self.lambda_, self.alpha, self.max_delta_step
            )

            # weakest-link alpha for this node
            denom = max(n_leaves - 1, 1)
            alpha_star = (R_collapse - R_subtree) / denom

            node._prune_leaves = n_leaves
            node._prune_internal = n_internal
            node._prune_R_subtree = R_subtree
            node._prune_R_collapse = R_collapse
            node._prune_alpha_star = alpha_star
            return n_leaves, n_internal, R_subtree

        _acc(self.root)

        # ----- Apply pruning (post-order), collapsing nodes with alpha* <= ccp_alpha -----
        def _apply(node):
            if node is None or node.is_leaf:
                return
            _apply(node.left_child)
            _apply(node.right_child)
            if not node.is_leaf and node._prune_alpha_star <= ccp_alpha:
                # collapse
                node.left_child = None
                node.right_child = None
                node.is_leaf = True
                node.leaf_value = _leaf_value_from_sums(
                    node.g_sum,
                    node.h_sum,
                    self.lambda_,
                    self.alpha,
                    self.max_delta_step,
                )
                node.best_feature = None
                node.best_threshold = np.nan
                node.missing_go_left = False
                node.best_gain = -np.inf
                node.best_bin_idx = None

        _apply(self.root)
        self._build_pred_arrays()


# ================= COMPAT SHIMS =================
class SingleTree(UnifiedTree):
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
        adaptive_hist=False,
    ):
        super().__init__(
            growth_policy="level_wise",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            lambda_=lambda_,
            gamma=gamma,
            alpha=alpha,
            max_delta_step=max_delta_step,
            tree_method=tree_method,
            n_bins=n_bins,
            feature_indices=feature_indices,
            bin_edges=bin_edges,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            gpu_accelerator=gpu_accelerator,
            n_jobs=n_jobs,
            adaptive_hist=False,
        )


class LeafWiseSingleTree(UnifiedTree):
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
        adaptive_hist=False,
    ):
        super().__init__(
            growth_policy="leaf_wise",
            max_depth=max_depth,
            max_leaves=max_leaves,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_child_weight=min_child_weight,
            lambda_=lambda_,
            gamma=gamma,
            alpha=alpha,
            max_delta_step=max_delta_step,
            tree_method=tree_method,
            n_bins=n_bins,
            feature_indices=feature_indices,
            bin_edges=bin_edges,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            gpu_accelerator=gpu_accelerator,
            n_jobs=n_jobs,
            adaptive_hist=adaptive_hist,
        )
