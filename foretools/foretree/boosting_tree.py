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
        binned_mode: str = "hist",  # {"hist","approx","adaptive"}
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
        enable_interactions: bool = False,
    ):
        if growth_policy not in ("leaf_wise", "level_wise"):
            raise ValueError("growth_policy must be 'leaf_wise' or 'level_wise'")
        if tree_method not in ("binned", "exact"):
            raise ValueError("tree_method must be 'binned' or 'exact'")
        if tree_method == "binned" and binned_mode not in (
            "hist",
            "approx",
            "adaptive",
        ):
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
        self.enable_interactions = enable_interactions

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

        self.has_refinement = False

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
            # Initialize histogram system instead of old binning
            config = HistogramConfig(
                method=self.binned_mode,  # "hist", "approx", "grad_aware"
                max_bins=self.n_bins,
                lambda_reg=self.lambda_,
                gamma=self.gamma,
                use_parallel=True,
                random_state=42,
            )
            self._histogram_system = GradientHistogramSystem(config)
            self._strategy = BinnedStrategy(
                histogram_system=self._histogram_system,
                enable_interactions=enable_interactions,
            )
        else:
            self._strategy = ExactStrategy()

        self.feature_importance_ = feature_importance_
        self._actual_max_bins = int(self.n_bins)  # refined per-mode during fit
        self._missing_bin_id = int(self.n_bins)  # reserved last bin, refined per-mode

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
    def _finalize_leaf(self, node: "TreeNode"):
        node.leaf_value = calc_leaf_value_newton(
            node.g_sum, node.h_sum, self.lambda_, self.alpha, self.max_delta_step
        )

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

            self._sorted_idx[gfi] = order[finite_mask]  # finite values
            self._missing_idx[gfi] = order[missing_mask_ordered]  # missing values

        self._node_of_sample = np.full(n, -1, dtype=np.int32)

    # --------------------------------------------------------------------------
    # APPLY SPLIT
    # --------------------------------------------------------------------------
    def _new_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def _project_oblique(
        self, Xcols: np.ndarray, rows: np.ndarray, sp
    ) -> tuple[np.ndarray, np.ndarray]:
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
            if (sp is None and node.best_feature is None) or (
                sp is not None and sp.gfi < 0
            ):
                return None
            gfi = int(sp.gfi) if sp is not None else int(node.best_feature)
            lf = self.feature_map[gfi]
            threshold = (
                float(sp.threshold) if sp is not None else float(node.best_threshold)
            )
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
            X_sub = self._X_train_cols[node.data_indices]

            # NEW: Use histogram system for binning if available
            if hasattr(self, "_strategy") and hasattr(
                self._strategy, "histogram_system"
            ):
                # Get binned codes from histogram system
                codes = self._histogram_system._precomputed_indices[node.data_indices]
                missing_bin_id = (
                    self._strategy.histogram_system.config.missing_bin_id
                )
                nb_total = self._strategy.histogram_system.config.total_bins

            else:
                # No binning system available
                return None

            # Apply categorical split using binned codes
            lfi = self.feature_map[gfi]
            col = codes[:, lfi].astype(np.int32, copy=False)

            # Create left mapping
            left_map = np.zeros(nb_total, dtype=np.uint8)
            for b in sp.left_groups:
                if 0 <= int(b) < nb_total - 1:  # Exclude missing bin from range check
                    left_map[int(b)] = 1
            if bool(sp.missing_left):
                left_map[missing_bin_id] = 1

            # Apply split
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

            li = rows[go_left]
            ri = rows[~go_left]
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
            mark_left = self._work_buffers.get(
                "mark_left", np.zeros(self._X_train_cols.shape[0], dtype=np.uint8)
            )
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

    def _priority_gain_depth(self, gain, node):
        # Depth-regularized priority: gain / (1 + alpha * depth)
        alpha = getattr(self, "leaf_depth_penalty", 0.0)  # default 0 = no-op
        d = getattr(node, "depth", 0)
        return gain / (1.0 + alpha * float(d))

    def _priority_gain_hess(self, gain, node):
        # Hessian-weighted priority: favors splits where parent has mass
        beta = getattr(self, "leaf_hess_boost", 0.0)  # default 0 = no-op
        Hp = max(0.0, float(getattr(node, "sum_hess", 0.0)))
        return gain * (1.0 + beta * Hp)

    def _compose_priority(self, gain, node):
        # Choose a prioritization recipe (defaults to pure gain)
        policy = getattr(
            self, "leaf_prioritization", "gain"
        )  # "gain"|"gain_depth"|"gain_hess"|"gain_depth_hess"
        pr = gain
        if "depth" in policy:
            pr = self._priority_gain_depth(pr, node)
        if "hess" in policy:
            pr = self._priority_gain_hess(pr, node)
        return pr

    def _grow_leaf_wise(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, fi=None):
        root = self._grow_root(X, g, h)

        # Deterministic heap tie-breaker (persists across calls)
        uid = int(getattr(self, "_heap_uid", 0))

        def _next_uid():
            nonlocal uid
            uid += 1
            return uid

        # Persist counter for subsequent trees (optional)
        self._heap_uid = uid

        gain_eps = float(
            getattr(self, "leaf_gain_eps", 0.0)
        )  # tiny floor to ignore micro-gains (default 0)
        allow_zero = bool(
            getattr(self, "leaf_allow_zero_gain", False)
        )  # if True, accept zero gains

        def _finite_pos_gain(x) -> bool:
            try:
                if not np.isfinite(x) or np.isnan(x):
                    return False
                if allow_zero:
                    return x >= gain_eps
                return x > max(0.0, gain_eps)
            except Exception:
                return False

        def _pass_child_guards(ch) -> bool:
            mnil = int(getattr(self, "min_data_in_leaf", 0))
            if mnil and int(getattr(ch, "n_samples", mnil)) < mnil:
                return False
            mcw = float(getattr(self, "min_child_weight", 0.0))
            if mcw and float(getattr(ch, "sum_hess", mcw)) < mcw:
                return False
            return True

        def _push_if_good(node):
            # Evaluate split and compute priority
            if not self._strategy.eval_split(self, X, node):
                return False
            bg = float(node.best_gain)
            if not _finite_pos_gain(bg):
                return False
            pr = float(self._compose_priority(bg, node))
            # min-heap → push negative priority
            heapq.heappush(heap, (-pr, _next_uid(), node))
            return True

        heap = []
        if not _push_if_good(root):
            self._finalize_leaf(root)
            return

        leaves = 1
        while heap and leaves < self.max_leaves:
            _, _, best = heapq.heappop(heap)

            children = self._apply_split(X, best)
            if not children:
                self._finalize_leaf(best)
                continue

            ln, rn = children

            # Optional: reject split if either child too small/weak
            if not (_pass_child_guards(ln) and _pass_child_guards(rn)):
                self._finalize_leaf(best)
                continue

            # One leaf → two leaves
            leaves += 1

            # HIST-like: sibling subtraction handoff (robust to container shape)
            if (
                self.tree_method != "exact"
                and getattr(best, "histograms", None) is not None
            ):
                p_hg, p_hh = best.histograms[0], best.histograms[1]
                if p_hg is not None and p_hh is not None:
                    ln.parent_hist = (p_hg, p_hh)
                    rn.parent_hist = (p_hg, p_hh)
                best.histograms = None  # avoid stale reuse / memory growth

            for ch in (ln, rn):
                if should_stop(self, ch):
                    self._finalize_leaf(ch)
                    continue
                if not _push_if_good(ch):
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
                p_hg = node.histograms[0]
                p_hh = node.histograms[1]
                ln.parent_hist = (p_hg, p_hh)
                rn.parent_hist = (p_hg, p_hh)

            q.extend((ln, rn))

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
            X[:, self.feature_indices]
            if self.feature_indices.size and X.shape[1] != len(self.feature_indices)
            else X
        )
        self._X_train_cols = X_train_cols

        # Prepare by mode
        if self.tree_method == "exact":
            self._approx_edges = None
            self._adaptive_edges = None
        else:
            self._strategy = BinnedStrategy(
                histogram_system=self._histogram_system,
                enable_interactions=False,
            )

        # strategy extra prep (exact presort, etc.)
        self._strategy.prepare(self, X, g, h)

        # grow
        grow_fn = (
            self._grow_leaf_wise
            if self.growth_policy == "leaf_wise"
            else self._grow_level_wise
        )
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
                node_features[nid] = (
                    nd.best_feature if nd.best_feature is not None else -1
                )
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
        has_non_axis = any(
            getattr(nd, "_split_plan", None) is not None
            and getattr(nd._split_plan, "kind", "axis") != "axis"
            for nd in self.nodes.values()
        )

        if has_non_axis:
            return self._predict_generic(X)

        (nf, nt, nm, lc, rc, lv, is_leaf_flags, fmap, nbin, nuse) = self._pred_arrays
        X_local = (
            X[:, self.feature_indices] if X.shape[1] != len(self.feature_indices) else X
        )

        # NEW: Use histogram system for binned prediction
        if self.tree_method != "exact" and np.any(nuse):
            # Check if we have histogram system
            if hasattr(self, "_strategy") and hasattr(
                self._strategy, "histogram_system"
            ):
                # Use histogram system for prediction binning

                Xb_pred = self._strategy.histogram_system.prebin_dataset(X_local)
                missing_bin_id = self._strategy.histogram_system.config.missing_bin_id

                y = predict_tree_binned_with_missingbin(
                    Xb_pred,
                    missing_bin_id,
                    nf,
                    nbin,
                    nm,
                    lc,
                    rc,
                    lv,
                    is_leaf_flags,
                    fmap,
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
                X_local,
                missing_mask,
                nf,
                nt,
                nm,
                lc,
                rc,
                lv,
                is_leaf_flags,
                fmap,
                self.root.node_id,
            )
        return np.asarray(y, dtype=np.float64)

    def _predict_generic(self, X: np.ndarray) -> np.ndarray:
        """
        Traversal that can evaluate axis, categorical k-way, and oblique splits.
        Updated to work with histogram system.
        """
        Xloc = (
            X[:, self.feature_indices] if X.shape[1] != len(self.feature_indices) else X
        )
        n = Xloc.shape[0]
        out = np.empty(n, dtype=np.float64)

        # iterative per-sample traversal
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
                        # NEW: Use histogram system for edge lookup
                        if hasattr(self, "_strategy") and hasattr(
                            self._strategy, "histogram_system"
                        ):
                            try:
                                # Get edges from histogram system feature info
                                feature_info = (
                                    self._strategy.histogram_system.get_feature_info(
                                        lfi
                                    )
                                )
                                e = feature_info.get("edges")
                            except Exception:
                                e = None
                        else:
                            # Legacy fallback
                            mode = getattr(self, "binned_mode", "hist")
                            e = (
                                self.bins.get_edges(gfi, mode=mode, node_id=None)
                                if hasattr(self, "bins")
                                else None
                            )

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
                    # oblique logic remains unchanged - no histogram system dependency
                    gfi_list = list(plan.features)
                    w = np.asarray(plan.weights, dtype=np.float64)
                    b0 = float(getattr(plan, "bias", 0.0))
                    thr = float(getattr(plan, "threshold", 0.0))
                    miss_left = bool(getattr(plan, "missing_left", True))
                    centers = getattr(plan, "centers", None)
                    scales = getattr(plan, "scales", None)
                    impute = getattr(plan, "impute", None)

                    # Manually compute one-sample projection
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
            n_internal = (
                node.left_child._prune_internal + node.right_child._prune_internal + 1
            )
            R_subtree = (
                node.left_child._prune_R_subtree + node.right_child._prune_R_subtree
            ) - self.gamma * 1.0
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
