# binned_strategy.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
from boosting_aux import *
from boosting_bin import *
from boosting_loss import *
from node import TreeNode

# import kernels/utilities you already have in your project
from plan import *

from foretools.foretree.boosting_bin_reg import _build_histograms_from_codes_njit


class SplitStrategy(Protocol):
    def prepare(
        self, tree: "UnifiedTree", X: np.ndarray, g: np.ndarray, h: np.ndarray
    ) -> None: ...
    def eval_split(
        self, tree: "UnifiedTree", X: np.ndarray, node: "TreeNode"
    ) -> bool: ...


EdgesProvider = Callable[["UnifiedTree", TreeNode, int], Optional[np.ndarray]]

# ================================ STRATEGIES ====================================


def should_stop(tree: "UnifiedTree", node: "TreeNode") -> bool:
    return (
        node.n_samples < tree.min_samples_split or node.depth >= tree.max_depth
        # or node.h_sum < tree.min_child_weight
    )



class BinnedStrategy:
    """
    Composite binned strategy that integrates with the new GradientHistogramSystem.
    Supports three modes:
      - 'hist'     : Simple histogram method (like XGBoost hist)
      - 'approx'   : Approximate with quantile sketching (like XGBoost approx)  
      - 'grad_aware': Gradient-aware sophisticated binning
    
    Key integration points:
    - Uses GradientHistogramSystem for all binning and histogram operations
    - Handles GOSS subsampling through index mapping
    - Supports both global pre-binning and per-node refinement
    """

    def __init__(
        self,
        mode: str = "hist",
        histogram_system: Optional[GradientHistogramSystem] = None,
        *,
        enable_categorical_kway: bool = True,
        enable_oblique: bool = False,
        enable_interactions: bool = True,
        cat_max_groups: int = 8,
        oblique_k: int = 6,
    ):
        if mode not in {"hist", "approx", "grad_aware"}:
            raise ValueError(
                "BinnedStrategy.mode must be one of {'hist','approx','grad_aware'}"
            )
        self.mode = mode
        self.histogram_system = histogram_system

        # Split generators (Axis always first to ensure baseline candidate)
        self._gens: List = [AxisGenerator()]
        if enable_categorical_kway:
            self._gens.append(CategoricalKWayGenerator(max_groups=cat_max_groups))
        if enable_interactions:
            self._gens.append(InteractionSeededGenerator())
        if enable_oblique:
            self._gens.append(ObliqueGenerator(k_features=oblique_k))

    # -------------------------------------------------------------------------
    # Strategy lifecycle
    # -------------------------------------------------------------------------
    def prepare(
        self, tree: "UnifiedTree", X: np.ndarray, g: np.ndarray, h: np.ndarray
    ) -> None:
        """Initialize histogram system and handle pre-binning"""
        if self.histogram_system is None:
            print("Initializing histogram system...")
            # Create histogram system if not provided
            config = HistogramConfig(
                method=self.mode,
                max_bins=getattr(tree, 'n_bins', 256),
                lambda_reg=getattr(tree, 'lambda_', 1.0),
                gamma=getattr(tree, 'gamma', 0.0),
                use_parallel=True,
                random_state=42
            )
            self.histogram_system = GradientHistogramSystem(config)
            
            # Fit bins using training data
            self.histogram_system.fit_bins(X, g, h)
        
        # Pre-bin the training data for fast histogram building
        # if not hasattr(tree, '_precomputed_indices') or tree._precomputed_indices is None:
        tree._precomputed_indices = self.histogram_system._precomputed_indices
        
        # Store training data reference
        tree._X_train_cols = X
        tree._global_gradients = g
        tree._global_hessians = h

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _lfi(self, tree, gfi: int) -> int:
        """Map global feature index -> local (training) column index or -1."""
        if hasattr(tree, "feature_indices") and tree.feature_indices.size:
            try:
                pos = np.where(tree.feature_indices == int(gfi))[0]
                if pos.size:
                    return int(pos[0])
            except Exception:
                pass
        return int(tree.feature_map.get(int(gfi), -1))

    def _get_node_sample_indices(self, tree: "UnifiedTree", node: "TreeNode") -> np.ndarray:
        """
        Get the global sample indices for this node, handling GOSS subsampling.
        This is key for mapping node-local indices back to the global pre-binned matrix.
        """
        # If tree was built with GOSS, map local indices to global
        if hasattr(tree, '_goss_selected_indices') and tree._goss_selected_indices is not None:
            # Node's data_indices are relative to the GOSS subsample
            # Map them back to original dataset indices
            return tree._goss_selected_indices[node.data_indices]
        else:
            # No subsampling - node indices are already global
            return node.data_indices

    # -------------------------------------------------------------------------
    # Histogram building with GOSS support
    # -------------------------------------------------------------------------

    def _build_histograms_fast(
        self, tree: "UnifiedTree", node: "TreeNode"
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build histograms using the fast pre-binned approach.
        Handles GOSS subsampling correctly by mapping indices.
        """
        # Get global sample indices for this node
        global_indices = self._get_node_sample_indices(tree, node)
        
        # Use fast histogram building from histogram system
        try:
            if hasattr(tree, '_goss_selected_indices') and tree._goss_selected_indices is not None:
                # GOSS case: use full gradient/hessian arrays with global indices
                grad_hist, hess_hist = self.histogram_system.build_histograms_fast(
                    tree._global_gradients,
                    tree._global_hessians,
                    sample_indices=global_indices
                )
            else:
                # Normal case: use node's gradients/hessians with node indices
                grad_hist, hess_hist = self.histogram_system.build_histograms_fast(
                    node.gradients.astype(np.float64),
                    node.hessians.astype(np.float64), 
                    sample_indices=global_indices
                )
            
            n_bins_total = self.histogram_system.config.total_bins
            return grad_hist, hess_hist, n_bins_total
            
        except Exception as e:
            # Fallback to slower method if fast path fails
            return self._build_histograms_fallback(tree, node)

    def _build_histograms_fallback(
        self, tree: "UnifiedTree", node: "TreeNode"
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Fallback histogram building for cases where fast path fails.
        Uses direct data processing without pre-binned indices.
        """
        # Get node's feature data
        global_indices = self._get_node_sample_indices(tree, node)
        X_node = tree._X_train_cols[global_indices]
        
        # Get gradients and hessians for this node
        if hasattr(tree, '_goss_selected_indices') and tree._goss_selected_indices is not None:
            g_node = tree._global_gradients[global_indices]
            h_node = tree._global_hessians[global_indices] 
        else:
            g_node = node.gradients.astype(np.float64)
            h_node = node.hessians.astype(np.float64)
        
        # Build histograms directly from data
        grad_hist, hess_hist = self.histogram_system.build_histograms(
            X_node, g_node, h_node
        )
        
        n_bins_total = self.histogram_system.config.total_bins
        return grad_hist, hess_hist, n_bins_total

    def _build_histograms_with_cache(
        self, tree: "UnifiedTree", node: "TreeNode"
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Main histogram building method with caching support.
        Integrates with the new histogram system while maintaining compatibility.
        """
        # Try cached histograms first
        if hasattr(tree, 'histogram_cache') and tree.histogram_cache is not None:
            cached_hist = tree.histogram_cache.get(node.node_id)
            if cached_hist is not None:
                hist_g, hist_h = cached_hist
                return hist_g, hist_h, hist_g.shape[1]

        # Try sibling subtraction if available
        if self._can_use_sibling_subtraction(tree, node):
            sib_hist = tree.histogram_cache.get(node.sibling_node_id)
            if sib_hist is not None:
                p_g, p_h = node.parent_hist
                s_g, s_h = sib_hist
                if p_g.shape == s_g.shape == p_h.shape == s_h.shape:
                    hist_g, hist_h = subtract_sibling_histograms(p_g, p_h, s_g, s_h)
                    tree.histogram_cache.put(node.node_id, hist_g, hist_h)
                    return hist_g, hist_h, hist_g.shape[1]

        # Build histograms using histogram system
        hist_g, hist_h, n_bins_total = self._build_histograms_fast(tree, node)

        # Cache the results
        if hasattr(tree, 'histogram_cache') and tree.histogram_cache is not None:
            tree.histogram_cache.put(node.node_id, hist_g, hist_h)

        return hist_g, hist_h, n_bins_total

    def _can_use_sibling_subtraction(self, tree, node) -> bool:
        """Check if sibling subtraction infrastructure is available."""
        return (
            hasattr(tree, 'histogram_cache') and
            tree.histogram_cache is not None and
            node.parent_hist is not None and
            node.sibling_node_id is not None
        )

    # -------------------------------------------------------------------------
    # Feature importance tracking  
    # -------------------------------------------------------------------------

    def _update_feature_importance_for_plan(
        self, tree, node, sp: "SplitPlanView", hist_pair
    ):
        """Feature importance update logic"""
        fi = getattr(tree, "feature_importance_", None)
        if fi is None:
            return
        hg, hh, max_bins_total = hist_pair
        finite_bins = max_bins_total - 1

        def _book(gfi, GL, HL, GR, HR, weight=1.0):
            gain = compute_gain(
                float(GL),
                float(HL),
                float(GR),
                float(HR),
                float(tree.lambda_),
                float(tree.gamma),
            )
            cover_h = float(HL + HR)
            if weight != 1.0:
                gain *= weight
                cover_h *= weight
            fi.add_split(int(gfi), float(gain), float(cover_h))
            fi.add("split", int(gfi), float(1.0 * weight))

        if sp.kind == "axis":
            gfi = int(sp.gfi)
            lfi = (
                int(np.where(tree.feature_indices == gfi)[0][0])
                if gfi in set(tree.feature_indices)
                else int(tree.feature_map.get(gfi, -1))
            )
            if lfi < 0 or lfi >= hg.shape[0] or sp.bin_idx is None:
                return
            b = int(sp.bin_idx)
            g_f = hg[lfi, :finite_bins]
            h_f = hh[lfi, :finite_bins]
            Gm = float(hg[lfi, finite_bins])
            Hm = float(hh[lfi, finite_bins])
            GL = float(g_f[: b + 1].sum())
            HL = float(h_f[: b + 1].sum())
            GR = float(g_f[b + 1 :].sum())
            HR = float(h_f[b + 1 :].sum())
            if bool(sp.missing_left):
                GL += Gm
                HL += Hm
            else:
                GR += Gm
                HR += Hm
            _book(gfi, GL, HL, GR, HR)
            return

        if sp.kind == "kway":
            gfi = int(sp.gfi)
            lfi = (
                int(np.where(tree.feature_indices == gfi)[0][0])
                if gfi in set(tree.feature_indices)
                else int(tree.feature_map.get(gfi, -1))
            )
            if lfi < 0 or lfi >= hg.shape[0]:
                return
            left_bins = np.array(
                sorted(
                    set(int(b) for b in sp.left_groups if 0 <= int(b) < finite_bins)
                ),
                dtype=np.int32,
            )
            GL = float(hg[lfi, left_bins].sum()) if left_bins.size else 0.0
            HL = float(hh[lfi, left_bins].sum()) if left_bins.size else 0.0
            all_bins = np.arange(finite_bins, dtype=np.int32)
            mask = np.ones(finite_bins, dtype=bool)
            if left_bins.size:
                mask[left_bins] = False
            GR = float(hg[lfi, all_bins[mask]].sum()) if mask.any() else 0.0
            HR = float(hh[lfi, all_bins[mask]].sum()) if mask.any() else 0.0
            Gm = float(hg[lfi, finite_bins])
            Hm = float(hh[lfi, finite_bins])
            if bool(sp.missing_left):
                GL += Gm
                HL += Hm
            else:
                GR += Gm
                HR += Hm
            _book(gfi, GL, HL, GR, HR)
            return

        if sp.kind in ("oblique", "oblique_interaction"):
            rows = node.data_indices
            g = node.gradients.astype(np.float64, copy=False)
            h = node.hessians.astype(np.float64, copy=False)
            gfi_list = [int(x) for x in sp.features]
            w = np.asarray(sp.weights, dtype=np.float64)

            z, any_missing = tree._project_oblique(tree._X_train_cols, rows, sp)
            thr = 0.0
            miss_left = bool(sp.missing_left)
            go_left = np.empty(rows.size, dtype=np.bool_)
            go_left[any_missing] = miss_left
            finite_mask = ~any_missing
            go_left[finite_mask] = z[finite_mask] <= thr

            GL = float(g[go_left].sum())
            HL = float(h[go_left].sum())
            GR = float(g[~go_left].sum())
            HR = float(h[~go_left].sum())

            absw = np.abs(w)
            denom = float(absw.sum()) if absw.size else 1.0
            shares = (
                (absw / denom)
                if denom > 0.0
                else np.full(len(gfi_list), 1.0 / max(len(gfi_list), 1))
            )
            for gfi, share in zip(gfi_list, shares):
                _book(int(gfi), GL, HL, GR, HR, float(share))
            return

    # -------------------------------------------------------------------------
    # Strategy entrypoint
    # -------------------------------------------------------------------------
    def eval_split(self, tree: "UnifiedTree", X: np.ndarray, node: "TreeNode") -> bool:
        """
        Main split evaluation method.
        Now uses the new histogram system for all binning and histogram operations.
        """
        # Guard conditions
        if (
            node.n_samples < tree.min_samples_split
            or node.depth >= tree.max_depth
            or node.h_sum < tree.min_child_weight
        ):
            return False

        # Build histograms using the new system
        try:
            hg, hh, max_bins_total = self._build_histograms_with_cache(tree, node)
            node.histograms = (hg, hh, max_bins_total)
        except Exception as e:
            # If histogram building fails, cannot split
            return False

        # Run Axis generator first (fast baseline)
        best_axis: Optional[Candidate] = None
        axis_gain_max = -np.inf

        axis_gen = None
        other_gens = []
        for gen in self._gens:
            if isinstance(gen, AxisGenerator):
                axis_gen = gen
            else:
                other_gens.append(gen)

        if axis_gen is None:
            axis_gen = AxisGenerator()

        axis_cands = axis_gen.generate(tree, X, node, (hg, hh, max_bins_total))
        if axis_cands:
            best_axis = max(axis_cands, key=lambda c: c.gain)
            axis_gain_max = best_axis.gain

        # Ask other generators
        best: Optional[Candidate] = best_axis
        for gen in other_gens:
            cands = gen.generate(
                tree, X, node, (hg, hh, max_bins_total), axis_gain_max=axis_gain_max
            )
            for c in cands:
                if (best is None) or (c.gain > best.gain):
                    best = c

        if best is None or best.gain <= 0.0:
            return False

        # Convert to SplitPlan
        sp: SplitPlan = to_split_plan(best)
        node._split_plan = sp
        node.best_gain = float(sp.gain)

        # Set legacy fields for axis-aligned splits (backward compatibility)
        if sp.kind == "axis":
            node.best_feature = int(sp.gfi)
            node.best_threshold = float(sp.threshold)
            node.missing_go_left = bool(sp.missing_left)
            node.best_bin_idx = int(sp.bin_idx) if sp.bin_idx is not None else None
        else:
            node.best_feature = None
            node.best_threshold = np.nan
            node.missing_go_left = False
            node.best_bin_idx = None

        # Update feature importance
        self._update_feature_importance_for_plan(
            tree, node, sp, (hg, hh, max_bins_total)
        )

        return True

    # -------------------------------------------------------------------------
    # Utility methods for compatibility
    # -------------------------------------------------------------------------

    def get_feature_bins_info(self, tree: "UnifiedTree", feature_idx: int) -> Dict:
        """Get binning information for a specific feature"""
        if self.histogram_system is None:
            return {}
        
        try:
            return self.histogram_system.get_feature_info(feature_idx)
        except Exception:
            return {}

    def get_all_strategies(self, tree: "UnifiedTree") -> List[str]:
        """Get binning strategies used for all features"""
        if self.histogram_system is None:
            return []
        
        try:
            return self.histogram_system.get_all_strategies()
        except Exception:
            return []


# Helper functions that need to be imported or defined elsewhere
def subtract_sibling_histograms(p_g, p_h, s_g, s_h):
    """Subtract sibling histograms from parent to get other child"""
    return p_g - s_g, p_h - s_h

def compute_gain(GL, HL, GR, HR, lambda_reg, gamma):
    """Compute split gain using gradient boosting formula"""
    if HL <= 0 or HR <= 0:
        return 0.0
    return 0.5 * ((GL * GL) / (HL + lambda_reg) + 
                  (GR * GR) / (HR + lambda_reg) - 
                  ((GL + GR) * (GL + GR)) / (HL + HR + lambda_reg)) - gamma

class ExactStrategy:
    """
    Exact strategy: presort once; evaluate thresholds directly on raw values,
    while delegating missing stats to the UnifiedMissingHandler. Keeps caches tidy.
    """

    def prepare(
        self, tree: "UnifiedTree", X: np.ndarray, g: np.ndarray, h: np.ndarray
    ) -> None:
        tree._g_global = g
        tree._h_global = h
        tree._prepare_presort_exact(X)
        if not hasattr(tree, "_exact_missing_cache"):
            tree._exact_missing_cache = {}

    def _missing_stats_for(
        self, tree: "UnifiedTree", X: np.ndarray, node: "TreeNode", lf: int
    ):
        key = (node.node_id, int(lf))
        cache = tree._exact_missing_cache
        if key in cache:
            return cache[key]
        stats = tree.missing_handler.get_missing_stats(
            X, tree._g_global, tree._h_global, node.data_indices, lf
        )
        tup = (
            int(stats["n_missing"]),
            float(stats["g_missing"]),
            float(stats["h_missing"]),
        )
        cache[key] = tup
        return tup

    def eval_split(self, tree: "UnifiedTree", X: np.ndarray, node: "TreeNode") -> bool:
        if should_stop(tree, node):
            return False

        g = tree._g_global
        h = tree._h_global
        best_gain = -np.inf
        best_feature = -1
        best_thr = np.nan
        best_miss_left = False
        best_ctx = None

        for lfi, gfi0 in enumerate(tree.feature_indices):
            gfi = int(gfi0)
            lst = node.sorted_lists[lfi] if node.sorted_lists is not None else None
            if lst is None or lst.size < 2 * tree.min_samples_leaf:
                continue

            lf = tree.feature_map.get(gfi, gfi)
            vals_seq = X[lst, lf]
            finite_mask = np.isfinite(vals_seq)
            if np.count_nonzero(finite_mask) < 2 * tree.min_samples_leaf:
                continue

            lst_finite = lst[finite_mask]
            vals_finite = vals_seq[finite_mask]
            n_miss, g_miss, h_miss = self._missing_stats_for(tree, X, node, lf)

            mono = 0
            if tree._mono_constraints_array is not None and gfi < len(
                tree._mono_constraints_array
            ):
                mono = int(tree._mono_constraints_array[gfi])

            gain, thr, miss_left, _, _ = best_split_on_feature_list(
                lst_finite,
                vals_finite,
                g,
                h,
                g_miss,
                h_miss,
                n_miss,
                tree.min_samples_leaf,
                tree.min_child_weight,
                tree.lambda_,
                tree.gamma,
                mono,
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = gfi
                best_thr = thr
                best_miss_left = bool(miss_left)
                best_ctx = (lst_finite, vals_finite, g_miss, h_miss)

        if best_feature == -1 or best_gain <= 0.0:
            return False

        node.best_feature = int(best_feature)
        node.best_threshold = float(best_thr)
        node.best_gain = float(best_gain)
        node.best_bin_idx = None
        node.missing_go_left = bool(best_miss_left)

        # Optional importances
        if (
            best_ctx is not None
            and getattr(tree, "feature_importance_", None) is not None
        ):
            lst_finite, vals_finite, g_miss, h_miss = best_ctx
            left_mask = vals_finite <= best_thr
            GL = float(g[lst_finite[left_mask]].sum())
            HL = float(h[lst_finite[left_mask]].sum())
            GR = float(g[lst_finite[~left_mask]].sum())
            HR = float(h[lst_finite[~left_mask]].sum())
            if best_miss_left:
                GL += float(g_miss)
                HL += float(h_miss)
            else:
                GR += float(g_miss)
                HR += float(h_miss)
            split_gain = compute_gain(
                GL, HL, GR, HR, float(tree.lambda_), float(tree.gamma)
            )
            cover_hessian = float(h[node.data_indices].sum())
            tree.feature_importance_.add_split(
                int(best_feature), split_gain, cover_hessian
            )
            tree.feature_importance_.add("split", int(best_feature), 1.0)

        return True
