from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

# -------------------------------------------------------------------
# Keep your existing imports/utilities
from boosting_aux import *
from boosting_aux import _leaf_objective_optimal
from boosting_bin import *
from boosting_loss import *
from node import TreeNode
from splits import *

from foretools.foretree.boosting_bin_reg import BinRegistry
from foretools.foretree.boosting_miss import (
    MissingStrategy,
    MissingValueConfig,
    UnifiedMissingHandler,
)

# ================================ UNIFIED TREE ==================================


class UnifiedTree:
    """
    Single-class tree with strategy dispatch (binned/exact) and hist/approx/adaptive binning,
    including optional adaptive refinement. This version cleans duplication while preserving
    all original features and signatures.
    """

    # ---- construction ----
    def __init__(
        self,
        growth_policy: str = "leaf_wise",
        max_depth: int = 6,
        max_leaves: int = 31,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        min_child_weight: float = 1e-3,
        lambda_: float = 1.0,
        gamma: float = 0.0,
        alpha: float = 0.0,
        max_delta_step: float = 0.0,
        tree_method: str = "binned",  # {"binned","exact"}
        binned_mode: str = "hist",    # {"hist","approx","adaptive"}
        n_bins: int = 256,
        feature_indices=None,
        bin_edges=None,
        monotone_constraints=None,
        interaction_constraints=None,
        gpu_accelerator=None,
        n_jobs: int = 4,
        adaptive_hist: bool = False,  # enables global adaptive + optional per-node refinement
        use_gpu: bool = False,
        feature_importance_=None,
    ):
        if growth_policy not in ("leaf_wise", "level_wise"):
            raise ValueError("growth_policy must be 'leaf_wise' or 'level_wise'")
        if tree_method not in ("binned", "exact"):
            raise ValueError("tree_method must be 'binned' or 'exact'")
        if binned_mode not in ("hist", "approx", "adaptive"):
            raise ValueError("binned_mode must be 'hist', 'approx', or 'adaptive'")

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
        self.binned_mode = binned_mode
        self.n_bins = int(n_bins)

        self.feature_indices = np.array(
            feature_indices if feature_indices is not None else [], dtype=np.int32
        )
        self.bin_edges: list[np.ndarray] = bin_edges if bin_edges is not None else []

        self.monotone_constraints: Dict[int, int] = monotone_constraints or {}
        self.interaction_constraints = interaction_constraints
        self.gpu_accelerator = gpu_accelerator
        self.n_jobs = n_jobs
        self.adaptive_hist = adaptive_hist
        self.use_gpu = use_gpu

        # state
        self.root: Optional[TreeNode] = None
        self.nodes: Dict[int, TreeNode] = {}
        self.next_node_id = 0
        self.histogram_cache = (
            HistogramCache() if "HistogramCache" in globals() else None
        )

        # binned views prepared by caller (optional fast path)
        self.binned_local = None

        # feature mapping via unified registry
        if self.feature_indices.size > 0:
            self.bins = BinRegistry(self.feature_indices)
            self.feature_map = self.bins.feature_map  # global->local index
        else:
            self.bins = BinRegistry(np.array([], dtype=np.int32))
            self.feature_map = {}

        # monotone vector (local space) for some paths
        self._mono_local = None
        if self.monotone_constraints and self.feature_indices.size:
            self._mono_local = np.zeros(len(self.feature_indices), dtype=np.int8)
            for g, c in self.monotone_constraints.items():
                if g in self.feature_map:
                    self._mono_local[self.feature_map[g]] = np.int8(c)

        # prediction arrays
        self._pred_arrays = None

        # ---- exact & shared buffers ----
        self._g_global: Optional[np.ndarray] = None
        self._h_global: Optional[np.ndarray] = None
        self._sorted_idx: Optional[Dict[int, np.ndarray]] = None
        self._missing_idx: Optional[Dict[int, np.ndarray]] = None
        self._node_of_sample: Optional[np.ndarray] = None
        self._work_buffers = {}
        self._partition_cache = {}

        # ---- approx/adaptive edges ----
        self._approx_edges: Optional[Dict[int, np.ndarray]] = None
        self._approx_weight_mode = "hess"  # or "abs_g" or "gain" etc.
        self._adaptive_edges: Optional[Dict[int, np.ndarray]] = None
        self._feature_bin_counts: Dict[int, int] = {}
        self._feature_metadata: Dict[int, dict] = {}

        # adaptive refinement switches
        self._adaptive_binner: Optional[GradientBinner] = (
            GradientBinner() if "GradientBinner" in globals() else None
        )
        self._node_adaptive_edges: Dict[Tuple[int, int], np.ndarray] = {}
        self._refinement_enabled = False
        self._adaptive_refine_topk = 8
        self._adaptive_weight_mode = "hess"

        # mono constraints array for exact path (global size)
        self._mono_constraints_array = np.zeros(
            int(self.feature_indices.max()) + 1 if self.feature_indices.size else 1,
            dtype=np.int8,
        )
        for gfi, constraint in self.monotone_constraints.items():
            if gfi < len(self._mono_constraints_array):
                self._mono_constraints_array[gfi] = np.int8(constraint)

        # Missing-handling config/handler
        missing_config = MissingValueConfig(
            strategy=MissingStrategy.LEARN_DIRECTION,
            lambda_reg=0.1,
            max_surrogate_splits=3,
            min_surrogate_agreement=0.55,
            min_samples_leaf=5,
            surrogate_search_features=50,
        )
        self.missing_handler = UnifiedMissingHandler(missing_config)

        # active strategy
        if self.tree_method == "binned":
            binner = self._adaptive_binner if self.binned_mode == "adaptive" else None
            self._strategy = BinnedStrategy(mode=self.binned_mode, binner=binner)
        else:
            self._strategy = ExactStrategy()

        self.feature_importance_ = feature_importance_
        self._actual_max_bins = int(self.n_bins)  # refined per-mode during fit
        self._missing_bin_id = int(self.n_bins)   # reserved last bin, refined per-mode

        # training slice cached
        self._X_train_cols: Optional[np.ndarray] = None
        self._plan_kind_counts = defaultdict(int)

    # --------------------------------------------------------------------------
    # Helpers (centralize shared logic)
    # --------------------------------------------------------------------------
    
    def report_split_plan_usage(self):
        total = sum(self._plan_kind_counts.values())
        return {
            k: {"count": v, "pct": (100.0 * v / total if total else 0.0)}
            for k, v in sorted(self._plan_kind_counts.items(), key=lambda kv: -kv[1])
        }

    def dump_splits(self, max_nodes: int = 20):
        rows = []
        for nid, nd in sorted(self.nodes.items()):
            if nd.is_leaf:
                continue
            plan = getattr(nd, "_split_plan", None)
            kind = getattr(plan, "kind", "axis" if nd.best_feature is not None else "unknown")
            if kind == "axis":
                rows.append((nid, nd.depth, kind, {"gfi": nd.best_feature, "thr": nd.best_threshold, "miss_left": nd.missing_go_left}))
            elif kind == "kway":
                rows.append((nid, nd.depth, kind, {"gfi": plan.gfi, "left_groups": list(plan.left_groups)[:8], "miss_left": plan.missing_left}))
            elif kind in ("oblique", "oblique_interaction"):
                rows.append((nid, nd.depth, kind, {"features": list(plan.features)[:6], "weights_norm1": float(np.sum(np.abs(plan.weights))), "thr": plan.threshold}))
            else:
                rows.append((nid, nd.depth, kind, {}))
            if len(rows) >= max_nodes:
                break
        return rows

    def _dtype_for_bins(self, max_bins: int):
        if max_bins <= 255:
            return np.uint8
        if max_bins <= 65535:
            return np.uint16
        return np.int32

    def _calc_actual_max_bins(self, edges_list: list[np.ndarray]) -> int:
        if not edges_list:
            return int(self.n_bins)
        return int(max(1, min(int(self.n_bins), max(len(e) - 1 for e in edges_list))))

    def _finalize_leaf(self, node: "TreeNode"):
        node.leaf_value = calc_leaf_value_newton(
            node.g_sum, node.h_sum, self.lambda_, self.alpha, self.max_delta_step
        )

    def _register_edges_with_registry(self, *, mode: str, edges_list: list[np.ndarray]):
        actual_max = self._calc_actual_max_bins(edges_list)
        self._actual_max_bins = actual_max
        self._missing_bin_id = actual_max
        bin_dtype = self._dtype_for_bins(actual_max)
        self.bins.register_global_edges(
            mode=mode,
            edges_list_in_feature_index_order=edges_list,
            actual_max_bins=actual_max,
            out_dtype=bin_dtype,
        )
        return actual_max, bin_dtype

    def _ensure_mode_registered(self, mode_str: str):
        try:
            _ = self.bins.get_layout(mode=mode_str)
            return
        except Exception:
            pass

        if mode_str == "hist":
            edges_list = self.bin_edges
        elif mode_str == "approx":
            edges_list = [
                self._approx_edges.get(int(g), np.array([0.0, 1.0], dtype=np.float64))
                for g in self.feature_indices
            ]
        elif mode_str == "adaptive":
            edges_list = [
                self._adaptive_edges.get(int(g), np.array([0.0, 1.0], dtype=np.float64))
                for g in self.feature_indices
            ]
        else:
            raise KeyError(f"Unknown bin mode '{mode_str}'")

        if not edges_list or any(getattr(e, "size", 0) < 2 for e in edges_list):
            raise RuntimeError(f"Cannot re-register mode '{mode_str}': invalid edges")

        self._register_edges_with_registry(mode=mode_str, edges_list=edges_list)

    def missing_bin_id(self, max_bins: int) -> int:
        """Return the reserved bin index for missing values."""
        return max_bins  # last id

    # --------------------------------------------------------------------------
    # EXACT PRE-SORT
    # --------------------------------------------------------------------------
    def _prepare_presort_exact(self, X: np.ndarray):
        """Optimized presort with better memory layout."""
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
            order = np.argsort(col, kind="mergesort").astype(np.int32)

            missing_mask_ordered = self.missing_handler.detect_missing(col[order])
            finite_mask = ~missing_mask_ordered

            self._sorted_idx[gfi] = order[finite_mask]         # finite values
            self._missing_idx[gfi] = order[missing_mask_ordered]  # missing values

        self._node_of_sample = np.full(n, -1, dtype=np.int32)

    # --------------------------------------------------------------------------
    # APPLY SPLIT
    # --------------------------------------------------------------------------
    def _new_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid
        
    def _project_oblique(self, Xcols: np.ndarray, rows: np.ndarray, sp) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute oblique decision values and the 'any_missing' mask consistently
        with what the generator intended.

        Canonicalizes the decision to:  z = w^T x' + b'   and we branch on (z <= 0).
        Where x' may be standardized/imputed according to plan fields:
        - sp.features: sequence of global feature ids
        - sp.weights:  array-like weights (same length as features)
        - sp.bias:     original bias term (default 0)
        - sp.threshold: original threshold (default 0)  -> we fold into bias
        - optional sp.centers, sp.scales: per-feature standardization
        - optional sp.impute: per-feature imputation for missing (else 0.0)
        - sp.missing_left: bool — when ANY participating feature is missing
        Returns:
        z: np.ndarray shape (rows.size,), the canonical score (<=0 => left)
        any_missing: np.ndarray bool mask where any participating feature was missing
        """
        # Pull fields with safe defaults
        gfi_list = list(sp.features)
        w = np.asarray(sp.weights, dtype=np.float64)
        b = float(getattr(sp, "bias", 0.0))
        thr = float(getattr(sp, "threshold", 0.0))

        centers = getattr(sp, "centers", None)
        scales = getattr(sp, "scales", None)
        impute = getattr(sp, "impute", None)

        # Fold threshold into bias so we ALWAYS compare against 0
        # (If the generator used w·x <= t, this becomes (b - t). If it used w·x + b <= 0, thr=0.)
        b_canon = b - thr

        z = np.full(rows.size, b_canon, dtype=np.float64)
        any_missing = np.zeros(rows.size, dtype=np.bool_)

        for j, gfi in enumerate(gfi_list):
            lfi = self.feature_map.get(int(gfi), -1)
            if lfi < 0 or lfi >= Xcols.shape[1]:
                any_missing |= True
                continue

            v = Xcols[rows, lfi]
            m = ~np.isfinite(v)
            any_missing |= m

            # impute missing in the projection space (so decision uses same values as training)
            if impute is not None:
                v = np.where(m, float(impute[j]), v)
            else:
                v = np.where(m, 0.0, v)

            # standardize if requested
            if centers is not None:
                v = v - float(centers[j])
            if scales is not None:
                s = float(scales[j])
                if s != 0.0:
                    v = v / s

            z += w[j] * v

        return z, any_missing

    def _apply_split(
        self, X: np.ndarray, node: "TreeNode"
    ) -> Optional[Tuple["TreeNode", "TreeNode"]]:
        """
        Apply the split stored on `node`. Supports:
        - Axis-aligned (legacy fast path; uses UnifiedMissingHandler)
        - Categorical K-way (chooses one side as a set of category bins)
        - Oblique / interaction-seeded (linear projection threshold)
        """
        # If a generator produced a SplitPlan, prefer it.
        plan = getattr(node, "_split_plan", None)
        sp = SplitPlanView(plan) if plan is not None else None

        # -------------------- AXIS-ALIGNED (legacy) --------------------
        if sp is None or sp.kind == "axis":
            if (sp is None and node.best_feature is None) or (sp is not None and sp.gfi < 0):
                return None
            gfi = int(sp.gfi) if sp is not None else int(node.best_feature)
            lf = self.feature_map[gfi]
            threshold = float(sp.threshold) if sp is not None else float(node.best_threshold)
            split = self.missing_handler.compute_split_with_missing(
                X, node.gradients, node.hessians, node.data_indices, lf, threshold
            )
            li, ri = split.left_indices, split.right_indices
            if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
                return None
            return self._make_binary_children(node, li, ri)

        # -------------------- CATEGORICAL K-WAY --------------------
        if sp.kind == "kway":
            gfi = int(sp.gfi)
            lf = self.feature_map[gfi]
            mode = getattr(self, "binned_mode", "hist")
            node_id_for_prebin = node.node_id if (
                mode == "adaptive" and getattr(self.bins, "node_has_any_override", lambda *a, **k: False)("adaptive", node.node_id)
            ) else None
            X_sub = self._X_train_cols[node.data_indices]
            codes, missing_bin_id = self.bins.prebin_matrix(
                X_sub, mode=mode, node_id=node_id_for_prebin,
                cache_key=f"{mode}:kway:{node.node_id}:{X_sub.shape}",
            )
            lfi = self.feature_map[gfi]
            col = codes[:, lfi].astype(np.int32, copy=False)
            nb_total = int(self.bins.get_layout(mode=mode).actual_max_bins) + 1
            left_map = np.zeros(nb_total, dtype=np.uint8)
            for b in sp.left_groups:
                if 0 <= int(b) < nb_total - 1:
                    left_map[int(b)] = 1
            if bool(sp.missing_left):
                left_map[missing_bin_id] = 1
            go_left = left_map[col] == 1
            li = node.data_indices[go_left]
            ri = node.data_indices[~go_left]
            if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
                return None
            return self._make_binary_children(node, li, ri)

        # -------------------- OBLIQUE / INTERACTION-SEEDED --------------------
        if sp.kind in ("oblique", "oblique_interaction"):
            Xcols = self._X_train_cols
            rows = node.data_indices
            z, any_missing = self._project_oblique(Xcols, rows, sp)
            miss_left = bool(getattr(sp, "missing_left", True))
            go_left = np.empty(rows.size, dtype=np.bool_)
            go_left[any_missing] = miss_left
            finite_mask = ~any_missing
            go_left[finite_mask] = z[finite_mask] <= 0.0  # canonical decision

            li = rows[go_left]; ri = rows[~go_left]
            if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
                return None
            return self._make_binary_children(node, li, ri)

        # Unknown plan kind
        return None

    def _make_binary_children(
        self, parent: "TreeNode", left_idx: np.ndarray, right_idx: np.ndarray
    ) -> Tuple["TreeNode", "TreeNode"]:
        """Common child creation (used by all split kinds)."""
        lid, rid = self._new_id(), self._new_id()

        # create boolean mask w.r.t parent order for g/h slicing
        left_mask = create_left_mask_adaptive(parent.data_indices, left_idx)

        ln = TreeNode(
            lid,
            left_idx,
            parent.gradients[left_mask],
            parent.hessians[left_mask],
            parent.depth + 1,
            parent_hist=parent.histograms,
            is_left_child=True,
        )
        rn = TreeNode(
            rid,
            right_idx,
            parent.gradients[~left_mask],
            parent.hessians[~left_mask],
            parent.depth + 1,
            parent_hist=parent.histograms,
            is_left_child=False,
        )

        # for exact path: partition sorted lists exactly as before
        if self.tree_method == "exact" and parent.sorted_lists is not None:
            mark_left = self._work_buffers.get("mark_left", np.zeros(self._X_train_cols.shape[0], dtype=np.uint8))
            mark_left[:] = 0
            mark_left[left_idx] = 1
            nlf = len(self.feature_indices)
            ln.sorted_lists = [None] * nlf
            rn.sorted_lists = [None] * nlf
            for lfi in range(nlf):
                pl = parent.sorted_lists[lfi]
                if pl is None or pl.size == 0:
                    ln.sorted_lists[lfi] = np.empty(0, dtype=np.int32)
                    rn.sorted_lists[lfi] = np.empty(0, dtype=np.int32)
                else:
                    m = mark_left[pl] == 1
                    ln.sorted_lists[lfi] = pl[m]
                    rn.sorted_lists[lfi] = pl[~m]
        else:
            ln.sorted_lists = None
            rn.sorted_lists = None

        # finalize
        for ch in (ln, rn):
            ch._tree_ref = self
            ch.init_sums()

        parent.left_child, parent.right_child = ln, rn
        parent.is_leaf = False
        ln.sibling_node_id, rn.sibling_node_id = rid, lid

        self.nodes[lid] = ln
        self.nodes[rid] = rn
        return ln, rn

    # --------------------------------------------------------------------------
    # GROWTH
    # --------------------------------------------------------------------------
    def _grow_root(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> "TreeNode":
        n = len(g)
        # Per-node payload in float32 for bandwidth; accumulations in float64 handled internally
        g32 = g.astype(np.float32, copy=False)
        h32 = h.astype(np.float32, copy=False)

        rid = self._new_id()
        root = TreeNode(rid, np.arange(n, dtype=np.int64), g32, h32, depth=0)
        root.init_sums()
        self.root = root
        self.nodes[rid] = root
        root._tree_ref = self

        if self.tree_method == "exact":
            nlf = len(self.feature_indices)
            root.sorted_lists = [None] * nlf
            for lfi, gfi in enumerate(self.feature_indices):
                root.sorted_lists[lfi] = self._sorted_idx[int(gfi)]
            if self._node_of_sample is not None:
                self._node_of_sample[:] = -1
                self._node_of_sample[root.data_indices] = rid
        return root

    def _grow_leaf_wise(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, fi=None):
        root = self._grow_root(X, g, h)

        heap = []
        if self._strategy.eval_split(self, X, root):
            heapq.heappush(heap, (-root.best_gain, root.node_id, root))
        else:
            self._finalize_leaf(root)

        leaves = 1
        while heap and leaves < self.max_leaves:
            _, _, best = heapq.heappop(heap)
            children = self._apply_split(X, best)
            if not children:
                self._finalize_leaf(best)
                continue

            ln, rn = children
            leaves += 1

            # HIST-like: let children see parent's hist for sibling subtraction
            if self.tree_method != "exact" and best.histograms is not None:
                ln.parent_hist = best.histograms
                rn.parent_hist = best.histograms

            for ch in (ln, rn):
                if should_stop(self, ch):
                    self._finalize_leaf(ch)
                    continue
                if self._strategy.eval_split(self, X, ch):
                    heapq.heappush(heap, (-ch.best_gain, ch.node_id, ch))
                else:
                    self._finalize_leaf(ch)

    def _grow_level_wise(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, fi=None):
        root = self._grow_root(X, g, h)
        q = [root]
        while q:
            node = q.pop(0)
            if should_stop(self, node) or not self._strategy.eval_split(self, X, node):
                self._finalize_leaf(node)
                continue

            children = self._apply_split(X, node)
            if not children:
                self._finalize_leaf(node)
                continue

            ln, rn = children
            if self.tree_method != "exact" and node.histograms is not None:
                ln.parent_hist = node.histograms
                rn.parent_hist = node.histograms
            q.extend((ln, rn))

    # --------------------------------------------------------------------------
    # Legacy seeding (kept for backward compat / fast path)
    # --------------------------------------------------------------------------
    def _seed_registry_with_legacy_codes(
        self, *, mode: str, edges_list, codes, legacy_missing_code: int | None = None
    ):
        actual_max_bins = self._calc_actual_max_bins(edges_list)
        self._actual_max_bins = int(actual_max_bins)
        self._missing_bin_id = int(actual_max_bins)

        out_dtype = self._dtype_for_bins(actual_max_bins)

        self.bins.register_global_edges(
            mode=mode,
            edges_list_in_feature_index_order=edges_list,
            actual_max_bins=self._actual_max_bins,
            out_dtype=out_dtype,
        )

        codes = np.asarray(codes, dtype=out_dtype)
        if legacy_missing_code is not None and (codes == legacy_missing_code).any():
            codes = codes.copy()
            codes[codes == legacy_missing_code] = self._missing_bin_id

        np.clip(codes, 0, self._missing_bin_id, out=codes)

        spec = self.bins._specs[mode]
        spec.cached_full_codes = codes  # single source of truth for training slices

        key = (
            id(self._X_train_cols),
            -1,
            hash(f"{mode}:train:{codes.shape}"),
            codes.shape,
        )
        self.bins._prebin_cache[key] = codes

    # --------------------------------------------------------------------------
    # FIT
    # --------------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        depth: int = 0,
        feature_importance=None,
    ):
        # reset per-fit state
        self.nodes.clear()
        self.next_node_id = 0
        self.root = None
        self._pred_arrays = None
        self._node_adaptive_edges.clear()

        self._g_global = None
        self._h_global = None

        # Keep training slice aligned to feature order
        X_train_cols = (
            X[:, self.feature_indices] if self.feature_indices.size and X.shape[1] != len(self.feature_indices) else X
        )
        self._X_train_cols = X_train_cols

        # Prepare by mode
        if self.tree_method == "exact":
            self._approx_edges = None
            self._adaptive_edges = None
        else:
            mode = getattr(self, "binned_mode", "hist")  # "hist" | "approx" | "adaptive"

            if mode == "hist":
                edges_list = self.bin_edges
                if not edges_list or any(getattr(e, "size", 0) < 2 for e in edges_list):
                    raise ValueError("For binned_mode='hist', valid bin_edges must be provided.")

                # register and seed from already pre-binned legacy matrices if available
                self._register_edges_with_registry(mode="hist", edges_list=edges_list)

                # If caller precomputed binned matrix, seed it
                if getattr(self, "_binned", None) is not None and \
                   hasattr(self, "_row_indexer") and hasattr(self, "_feature_mask"):
                    binned_local = self._binned[np.ix_(self._row_indexer, self._feature_mask)]
                    self._seed_registry_with_legacy_codes(
                        mode="hist",
                        edges_list=self.bin_edges,
                        codes=binned_local,
                        legacy_missing_code=-1,  # change if legacy used different missing code
                    )
                else:
                    # fallback: build training prebin now
                    _, _ = self.bins.prebin_matrix(
                        X_train_cols, mode="hist", node_id=None,
                        cache_key=f"hist:train:{X_train_cols.shape}",
                    )

            elif mode == "approx":
                self._prepare_approx(X, g, h)

            elif mode == "adaptive":
                self._prepare_adaptive(X, g, h, enable_refinement=True)

            else:
                raise ValueError(f"Unknown binned_mode: {mode}")

        # strategy extra prep (exact presort, etc.)
        self._strategy.prepare(self, X, g, h)

        # grow
        grow_fn = self._grow_leaf_wise if self.growth_policy == "leaf_wise" else self._grow_level_wise
        grow_fn(X, g, h, feature_importance)

        # finalize any leaves lacking value
        for n in self.nodes.values():
            if n.is_leaf and n.leaf_value is None:
                self._finalize_leaf(n)

        # Build predictor arrays
        self._build_pred_arrays()
        if self._pred_arrays is not None:
            self._pred_arrays = tuple(
                np.ascontiguousarray(a) if isinstance(a, np.ndarray) else a
                for a in self._pred_arrays
            )
                    
        return self

    # --------------------------------------------------------------------------
    # APPROX (gradient-aware edges)
    # --------------------------------------------------------------------------
    def _compute_importance_weights(
        self, g: np.ndarray, h: np.ndarray, mode
    ) -> np.ndarray:
        lam = float(getattr(self, "lambda_", 1.0))
        eps = float(getattr(self, "_approx_weight_eps", 1e-12))
        if callable(mode):
            w = mode(g, h, lam)
            return np.asarray(w, dtype=np.float64)

        m = (mode or "gain").lower()
        g = g.astype(np.float64, copy=False)
        h = h.astype(np.float64, copy=False)

        if m in ("gain", "g2_over_h+lam", "g2_over_h"):
            return (g * g) / (h + lam + eps)
        if m == "abs_g":
            return np.abs(g)
        if m == "hess":
            return np.maximum(h, 0.0)
        if m in ("abs_g_over_sqrt_h", "abs_g_over_sqrt(h)"):
            return np.abs(g) / np.sqrt(h + lam + eps)
        if m == "unit":
            return np.ones_like(g, dtype=np.float64)
        return (g * g) / (h + lam + eps)  # default

    def _gradient_aware_quantile_edges(
        self, x: np.ndarray, w: np.ndarray, n_bins: int
    ) -> np.ndarray:
        mask = np.isfinite(x) & np.isfinite(w)
        if not np.any(mask):
            return np.array([0.0, 1.0], dtype=np.float64)

        x = x[mask].astype(np.float64, copy=False)
        w = w[mask].astype(np.float64, copy=False)

        order = np.argsort(x)
        x = x[order]
        w = w[order]

        uniq, start_idx = np.unique(x, return_index=True)
        W = np.add.reduceat(w, start_idx)

        sw = int(getattr(self, "_approx_smooth_window", 3))
        if sw >= 3 and W.size >= 9:
            W_s = W.copy()
            W_s[1:-1] = (W[:-2] + W[1:-1] + W[2:]) / 3.0
            W = W_s

        k = int(max(1, min(n_bins, max(1, uniq.size - 1))))
        cW = np.cumsum(W)
        total = cW[-1]
        if total <= 0 or k == 1:
            lo, hi = float(uniq[0]), float(uniq[-1])
            if lo == hi:
                hi = np.nextafter(lo, np.inf)
            return np.array([lo, hi], dtype=np.float64)

        targets = np.arange(1, k) * (total / k)
        cut_locs = np.searchsorted(cW, targets, side="left")
        cut_locs = np.clip(cut_locs, 0, uniq.size - 2)

        edges = [float(uniq[0])]
        for pos in cut_locs:
            a, b = float(uniq[pos]), float(uniq[pos + 1])
            mid = 0.5 * (a + b) if a < b else np.nextafter(a, np.inf)
            if mid <= edges[-1]:
                mid = np.nextafter(edges[-1], np.inf)
            edges.append(mid)
        edges.append(float(uniq[-1]))

        edges = np.asarray(edges, dtype=np.float64)
        for i in range(1, edges.size):
            if not (edges[i] > edges[i - 1]):
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        return edges

    def _prepare_approx(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> None:
        """Build gradient-aware edges with selectable weighting w (feature-normalized)."""
        self._approx_edges = {}
        self._feature_bin_counts = {}

        feats = (
            self.feature_indices.astype(np.int32, copy=False)
            if self.feature_indices.size
            else np.arange(X.shape[1], dtype=np.int32)
        )

        mode = getattr(self, "_approx_weight_mode", "gain")
        w_all = self._compute_importance_weights(g, h, mode)

        for gfi in feats:
            lf = int(self.feature_map.get(int(gfi), int(gfi)))
            if lf < 0 or lf >= X.shape[1]:
                edges = np.array([0.0, 1.0], dtype=np.float64)
                self._approx_edges[int(gfi)] = edges
                self._feature_bin_counts[int(gfi)] = 1
                continue

            col = X[:, lf].astype(np.float64, copy=False)
            finite = np.isfinite(col)
            if not np.any(finite):
                edges = np.array([0.0, 1.0], dtype=np.float64)
                self._approx_edges[int(gfi)] = edges
                self._feature_bin_counts[int(gfi)] = 1
                continue

            x_f = col[finite]
            w_f = w_all[finite].astype(np.float64, copy=False)

            mu = float(w_f.mean())
            if mu > 0:
                w_f = w_f / mu
            hi = np.percentile(w_f, 99.9)
            if hi > 0:
                w_f = np.minimum(w_f, hi)

            n_unique = np.unique(x_f).size
            max_bins_feat = int(max(1, min(self.n_bins, max(1, n_unique - 1))))

            edges = self._gradient_aware_quantile_edges(x_f, w_f, max_bins_feat)
            if edges.size < 2:
                vmin, vmax = float(np.min(x_f)), float(np.max(x_f))
                if vmin == vmax:
                    edges = np.array([vmin, np.nextafter(vmin, np.inf)], dtype=np.float64)
                else:
                    edges = np.array([vmin, vmax], dtype=np.float64)

            self._approx_edges[int(gfi)] = edges
            self._feature_bin_counts[int(gfi)] = max(1, edges.size - 1)

        edges_list = [
            self._approx_edges.get(int(gfi), np.array([0.0, 1.0], dtype=np.float64))
            for gfi in self.feature_indices
        ]
        self._register_edges_with_registry(mode="approx", edges_list=edges_list)

        # Prebin training matrix using registry (reserves last bin as missing)
        X_train_cols = (
            X[:, self.feature_indices] if self.feature_indices.size and X.shape[1] != len(self.feature_indices) else X
        )
        self._X_train_cols = X_train_cols
        _, _ = self.bins.prebin_matrix(
            X_train_cols,
            mode="approx",
            node_id=None,
            cache_key=f"approx:train:{X_train_cols.shape}",
        )

    # --------------------------------------------------------------------------
    # ADAPTIVE (GradientBinner global edges + optional node refinement)
    # --------------------------------------------------------------------------
    def _prepare_adaptive(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        *,
        subsample: int = 0,
        weight_mode: str = "hess",  # kept for signature compat
        rng_seed: int = 42,
        enable_refinement: bool = True,
    ) -> None:
        self._adaptive_edges = {}
        self._feature_bin_counts = {}
        self._feature_metadata = {}

        feats = (
            self.feature_indices.astype(np.int32, copy=False)
            if self.feature_indices.size
            else np.arange(X.shape[1], dtype=np.int32)
        )
        n_samples, n_features = X.shape

        # Instantiate binner if missing
        if not hasattr(self, "_adaptive_binner") or self._adaptive_binner is None:
            base_bins = int(np.clip(np.sqrt(max(n_samples, 1) / 100.0), 16, 128))
            max_bins = 64 if n_samples < 1_000 else (256 if n_samples < 10_000 else 512)
            gain_thr = 0.05 if n_samples < 1_000 else (0.02 if n_samples < 10_000 else 0.01)
            refinement_fraction = float(np.clip(50.0 / max(n_features, 50), 0.1, 0.5))

            cfg = BinningConfig(
                coarse_bins=base_bins,
                max_total_bins=min(max_bins, int(self.n_bins) * 2),
                min_child_weight=getattr(self, "min_child_weight", 1e-3),
                min_gain_threshold=gain_thr,
                top_regions=max(4, min(12, n_features // 50)),
                use_density_aware_binning=True,
                use_parallel=(len(feats) > 20),
                max_workers=int(np.clip(len(feats) // 25, 1, 6)),
                cache_size=min(2000, max(100, len(feats) * 2)),
                node_refinement_threshold=max(100, n_samples // 200),
                max_refinement_depth=min(15, int(getattr(self, "max_depth", 8)) + 2),
                refinement_feature_fraction=refinement_fraction,
                refinement_min_correlation=0.05,
                categorical_threshold=32,
                overlap_merge_threshold=2,
                eps=1e-12,
            )
            self._adaptive_binner = GradientBinner(cfg)

        # Subsample optionally
        if subsample and subsample < X.shape[0]:
            effective_subsample = min(subsample, max(10_000, n_features * 100))
            rs = np.random.RandomState(rng_seed)
            idx = rs.choice(X.shape[0], size=effective_subsample, replace=False)
        else:
            idx = np.arange(X.shape[0], dtype=np.int64)

        g_use = g[idx].astype(np.float64, copy=False)
        h_use = h[idx].astype(np.float64, copy=False)
        X_sub = X[idx]

        # Batch create bins
        batch_data, valid_feats, fallback_feats = [], [], []
        for gfi in feats:
            lf = int(self.feature_map.get(int(gfi), int(gfi)))
            if lf < 0 or lf >= X.shape[1]:
                fallback_feats.append(gfi)
                continue
            col = X_sub[:, lf].astype(np.float64, copy=False)
            finite = np.isfinite(col)
            if (finite.sum() < 10) or (np.std(col[finite]) < 1e-12):
                fallback_feats.append(gfi)
                continue
            batch_data.append((col, g_use, h_use, int(gfi)))
            valid_feats.append(gfi)

        if batch_data:
            results = self._adaptive_binner.batch_create_bins(
                batch_data,
                lambda_reg=getattr(self, "lambda_", 1.0),
                gamma=getattr(self, "gamma", 0.0),
            )
            for gfi, (edges, meta) in zip(valid_feats, results):
                edges = np.unique(np.asarray(edges, np.float64))
                if edges.size < 2:
                    edges = np.array([0.0, 1.0], np.float64)
                    meta["strategy"] = "fallback_minimal"
                if edges.size - 1 > self.n_bins:
                    q = np.linspace(0, 1, self.n_bins + 1)
                    lo, hi = edges[0], edges[-1]
                    edges = np.unique(lo + q * (hi - lo))
                    meta["capped_at"] = int(self.n_bins)
                self._adaptive_edges[int(gfi)] = edges
                self._feature_bin_counts[int(gfi)] = max(1, edges.size - 1)
                self._feature_metadata[int(gfi)] = meta

        # Align/register in the registry (guarantee edges for all features)
        edges_list = []
        for gfi in self.feature_indices:
            e = self._adaptive_edges.get(int(gfi))
            if e is None or e.size < 2:
                # fallback from full X to be robust
                lf = int(self.feature_map.get(int(gfi), int(gfi)))
                if 0 <= lf < X.shape[1]:
                    col = X[:, lf].astype(np.float64, copy=False)
                    finite = np.isfinite(col)
                    if np.any(finite):
                        vmin = float(np.min(col[finite]))
                        vmax = float(np.max(col[finite]))
                        if vmin == vmax:
                            e = np.array([vmin, np.nextafter(vmin, np.inf)], dtype=np.float64)
                        else:
                            q = np.linspace(0.0, 100.0, min(self.n_bins, 64) + 1)
                            cuts = np.percentile(col[finite], q).astype(np.float64)
                            for i in range(1, cuts.size):
                                if not (cuts[i] > cuts[i - 1]):
                                    cuts[i] = np.nextafter(cuts[i - 1], np.inf)
                            e = cuts
                    else:
                        e = np.array([0.0, 1.0], dtype=np.float64)
                else:
                    e = np.array([0.0, 1.0], dtype=np.float64)
                self._adaptive_edges[int(gfi)] = e
                self._feature_bin_counts[int(gfi)] = max(1, e.size - 1)
                self._feature_metadata[int(gfi)] = {"strategy": "fallback_dense"}
            edges_list.append(e)

        self._register_edges_with_registry(mode="adaptive", edges_list=edges_list)

        # Prebin training matrix
        X_train_cols = (
            X[:, self.feature_indices] if self.feature_indices.size and X.shape[1] != len(self.feature_indices) else X
        )
        self._X_train_cols = X_train_cols
        _, _ = self.bins.prebin_matrix(
            X_train_cols,
            mode="adaptive",
            node_id=None,
            cache_key=f"adaptive:train:{X_train_cols.shape}",
        )
        self._refinement_enabled = bool(enable_refinement)

    # --------------------------------------------------------------------------
    # PREDICTION ARRAYS
    # --------------------------------------------------------------------------
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
        node_bin_idx = np.full(N, -1, dtype=np.int32)
        node_uses_bin = np.zeros(N, dtype=np.bool_)

        max_feat = int(self.feature_indices.max()) if self.feature_indices.size else -1
        fmap = np.full(max_feat + 1 if max_feat >= 0 else 1, -1, dtype=np.int32)
        for g, l in self.feature_map.items():
            if g >= 0:
                if g >= fmap.size:
                    new = np.full(g + 1, -1, dtype=np.int32)
                    new[: fmap.size] = fmap
                    fmap = new
                fmap[g] = l

        for nid, nd in self.nodes.items():
            if nd.is_leaf:
                leaf_values[nid] = nd.leaf_value or 0.0
            else:
                node_features[nid] = nd.best_feature if nd.best_feature is not None else -1
                node_thresholds[nid] = nd.best_threshold
                node_missing_go_left[nid] = nd.missing_go_left
                left_children[nid] = nd.left_child.node_id if nd.left_child else -1
                right_children[nid] = nd.right_child.node_id if nd.right_child else -1
                is_leaf_flags[nid] = False

            # only hist-like paths have stable bin indices
            if self.tree_method != "exact" and nd.best_bin_idx is not None:
                node_bin_idx[nid] = int(nd.best_bin_idx)
                node_uses_bin[nid] = True

        self._pred_arrays = (
            node_features,
            node_thresholds,
            node_missing_go_left,
            left_children,
            right_children,
            leaf_values,
            is_leaf_flags,
            fmap,
            node_bin_idx,
            node_uses_bin,
        )

    # --------------------------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        if self._pred_arrays is None:
            raise RuntimeError("Prediction arrays not built; cannot predict.")

        # If the model contains any non-axis splits, run the fallback traverser.
        # (Axis-aligned trees keep the fast array/binned paths exactly as before.)
        has_non_axis = any(
            getattr(nd, "_split_plan", None) is not None
            and getattr(nd._split_plan, "kind", "axis") != "axis"
            for nd in self.nodes.values()
        )
        if has_non_axis:
            return self._predict_generic(X)

        # ---- your existing fast paths (unchanged) ----
        (nf, nt, nm, lc, rc, lv, is_leaf_flags, fmap, nbin, nuse) = self._pred_arrays
        X_local = X[:, self.feature_indices] if X.shape[1] != len(self.feature_indices) else X

        if self.tree_method != "exact" and np.any(nuse):
            mode = getattr(self, "binned_mode", "hist")
            try:
                Xb_pred, missing_bin_id = self.bins.prebin_matrix(
                    X_local, mode=mode, node_id=None, cache_key=f"{mode}:pred:{X_local.shape}"
                )
            except KeyError:
                # re-register edges if needed (kept from your code)
                _ensure_mode_registered = getattr(self, "_ensure_mode_registered", None)
                if _ensure_mode_registered is not None:
                    _ensure_mode_registered(mode)
                    Xb_pred, missing_bin_id = self.bins.prebin_matrix(
                        X_local, mode=mode, node_id=None, cache_key=f"{mode}:pred:{X_local.shape}"
                    )
                else:
                    raise
            y = predict_tree_binned_with_missingbin(
                Xb_pred,
                missing_bin_id,
                nf, nbin, nm, lc, rc, lv, is_leaf_flags, fmap,
                self.root.node_id,
            )
            return np.asarray(y, dtype=np.float64)

        # exact path (unchanged)
        if self.missing_handler.config.finite_check_mode == "strict":
            y = predict_tree_numba(
                X_local, nf, nt, nm, lc, rc, lv, is_leaf_flags, fmap, self.root.node_id
            )
        else:
            missing_mask = self.missing_handler.detect_missing(X_local)
            y = predict_tree_numba_with_missing_mask(
                X_local, missing_mask, nf, nt, nm, lc, rc, lv, is_leaf_flags, fmap, self.root.node_id
            )
        return np.asarray(y, dtype=np.float64)

    def _predict_generic(self, X: np.ndarray) -> np.ndarray:
        """
        Traversal that can evaluate axis, categorical k-way, and oblique splits.
        Kept simple & vectorized per-node where possible; assumes columns are
        aligned to self.feature_indices.
        """
        Xloc = X[:, self.feature_indices] if X.shape[1] != len(self.feature_indices) else X
        n = Xloc.shape[0]
        out = np.empty(n, dtype=np.float64)

        # iterative per-sample traversal (depths are small; this is fine for mixed trees)
        for i in range(n):
            nd = self.root
            while nd is not None and not nd.is_leaf:
                plan = getattr(nd, "_split_plan", None)

                if plan is None or plan.kind == "axis":
                    # axis path: use feature map array from pred arrays
                    gfi = int(nd.best_feature)
                    lfi = self.feature_map.get(gfi, -1)
                    if lfi < 0 or lfi >= Xloc.shape[1]:
                        # defensive: treat as missing → learned side
                        go_left = bool(nd.missing_go_left)
                    else:
                        v = Xloc[i, lfi]
                        if not np.isfinite(v):
                            go_left = bool(nd.missing_go_left)
                        else:
                            go_left = v <= float(nd.best_threshold)

                elif plan.kind == "kway":
                    gfi = int(plan.gfi)
                    lfi = self.feature_map.get(gfi, -1)
                    miss_left = bool(getattr(plan, "missing_left", True))
                    if lfi < 0 or lfi >= Xloc.shape[1]:
                        go_left = miss_left
                    else:
                        mode = getattr(self, "binned_mode", "hist")
                        # one-row prebin for this feature
                        e = self.bins.get_edges(gfi, mode=mode, node_id=None)
                        if e is None or e.size < 2:
                            go_left = miss_left
                        else:
                            # scalar binning
                            v = Xloc[i, lfi]
                            if not np.isfinite(v):
                                go_left = miss_left
                            else:
                                # binary search
                                lo, hi = 0, e.size - 1
                                if v < e[0]:
                                    b = 0
                                elif v >= e[-1]:
                                    b = (e.size - 1) - 1
                                else:
                                    while hi - lo > 1:
                                        mid = (lo + hi) // 2
                                        if v >= e[mid]:
                                            lo = mid
                                        else:
                                            hi = mid
                                    b = lo
                                go_left = int(b) in set(plan.left_groups)

                elif plan.kind in ("oblique", "oblique_interaction"):
                    # Build a tiny view for "one row" without copying full arrays
                    gfi_list = list(plan.features)
                    w = np.asarray(plan.weights, dtype=np.float64)
                    b0 = float(getattr(plan, "bias", 0.0))
                    thr = float(getattr(plan, "threshold", 0.0))
                    miss_left = bool(getattr(plan, "missing_left", True))
                    centers = getattr(plan, "centers", None)
                    scales = getattr(plan, "scales", None)
                    impute = getattr(plan, "impute", None)

                    # Manually compute one-sample projection (cheaper than creating rows=idx array)
                    acc = b0 - thr  # canonical bias
                    missing = False
                    for j, gfi in enumerate(gfi_list):
                        lfi = self.feature_map.get(int(gfi), -1)
                        if lfi < 0 or lfi >= Xloc.shape[1]:
                            missing = True
                            break
                        val = Xloc[i, lfi]
                        if not np.isfinite(val):
                            missing = True
                            # impute for the decision value so behavior matches training
                            val = float(impute[j]) if impute is not None else 0.0
                        if centers is not None:
                            val = val - float(centers[j])
                        if scales is not None:
                            s = float(scales[j])
                            if s != 0.0:
                                val = val / s
                        acc += w[j] * val

                    go_left = miss_left if missing else (acc <= 0.0)

                else:
                    # unknown plan kind → safe default
                    go_left = True

                nd = nd.left_child if go_left else nd.right_child

            out[i] = 0.0 if nd is None else float(nd.leaf_value)
        return out


    # --------------------------------------------------------------------------
    # UTILS
    # --------------------------------------------------------------------------
    def get_feature_importance(self, mode: str = "gain", cover_def: str = "hessian"):
        """
        mode ∈ {"gain", "cover", "split", "gain_x_cover"}
        cover_def ∈ {"hessian", "samples"}
        Returns: dict {global_feat_idx: importance_value}
        """
        imp = defaultdict(float)
        for nd in self.nodes.values():
            if nd.is_leaf or nd.best_feature is None:
                continue

            j_global = int(nd.best_feature)
            gain = float(getattr(nd, "best_gain", 0.0))
            if gain <= 0.0:
                continue

            if cover_def == "hessian":
                cover = float(nd.h_sum)
            elif cover_def == "samples":
                cover = float(nd.n_samples)
            else:
                raise ValueError(f"Invalid cover_def: {cover_def}")

            if mode == "gain":
                imp[j_global] += gain
            elif mode == "cover":
                imp[j_global] += cover
            elif mode == "split":
                imp[j_global] += 1.0
            elif mode == "gain_x_cover":
                imp[j_global] += gain * cover
            else:
                raise ValueError(f"Invalid mode: {mode}")

        return dict(imp)

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

        def _acc(node):
            if node is None:
                return 0, 0, 0.0
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

            _acc(node.left_child)
            _acc(node.right_child)
            n_leaves = node.left_child._prune_leaves + node.right_child._prune_leaves
            n_internal = node.left_child._prune_internal + node.right_child._prune_internal + 1
            R_subtree = (node.left_child._prune_R_subtree + node.right_child._prune_R_subtree) - self.gamma * 1.0
            R_collapse = _leaf_objective_optimal(
                node.g_sum, node.h_sum, self.lambda_, self.alpha, self.max_delta_step
            )
            denom = max(n_leaves - 1, 1)
            alpha_star = (R_collapse - R_subtree) / denom
            node._prune_leaves = n_leaves
            node._prune_internal = n_internal
            node._prune_R_subtree = R_subtree
            node._prune_R_collapse = R_collapse
            node._prune_alpha_star = alpha_star
            return n_leaves, n_internal, R_subtree

        _acc(self.root)

        def _apply(node):
            if node is None or node.is_leaf:
                return
            _apply(node.left_child)
            _apply(node.right_child)
            if not node.is_leaf and node._prune_alpha_star <= ccp_alpha:
                node.left_child = None
                node.right_child = None
                node.is_leaf = True
                self._finalize_leaf(node)
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
        tree_method="binned",
        n_bins=256,
        bin_edges=None,
        monotone_constraints=None,
        interaction_constraints=None,
        gpu_accelerator=None,
        max_delta_step=0.0,
        adaptive_hist=False,
        use_gpu=False,
        feature_importance_=None,  # ignored; for sklearn compatibility
        binned_mode="hist",
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
            adaptive_hist=adaptive_hist,
            use_gpu=use_gpu,
            feature_importance_=feature_importance_,
            binned_mode=binned_mode,
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
        tree_method="binned",
        n_bins=256,
        bin_edges=None,
        monotone_constraints=None,
        interaction_constraints=None,
        gpu_accelerator=None,
        max_delta_step=0.0,
        max_leaves=31,
        min_child_weight=1e-3,
        adaptive_hist=False,
        use_gpu=False,
        feature_importance_=None,  # ignored; for sklearn compatibility
        binned_mode="hist",
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
            use_gpu=use_gpu,
            feature_importance_=feature_importance_,
            binned_mode=binned_mode,
        )
