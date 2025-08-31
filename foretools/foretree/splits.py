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
        node.n_samples < tree.min_samples_split
        or node.depth >= tree.max_depth
        # or node.h_sum < tree.min_child_weight
    )


class BinnedStrategy:
    """
    Composite binned strategy that can operate in three modes selected at runtime:
      - 'hist'     : fixed global edges from tree.bin_edges (via BinRegistry)
      - 'approx'   : global learned edges (tree._approx_edges registered in BinRegistry)
      - 'adaptive' : global learned edges with optional per-node refinement via a binner

    It keeps the original axis-aligned behavior fast, while optionally adding:
      - Categorical k-way splits,
      - Oblique (linear-combination) splits,
      - Interaction-seeded oblique splits.

    The best candidate across enabled generators is turned into a SplitPlan that
    materializes the children; axis-aligned plans still fill legacy fields (thresholds etc.).
    """

    def __init__(
        self,
        mode: str = "hist",
        binner: Optional["GradientBinner"] = None,
        edges_provider: Optional["EdgesProvider"] = None,
        *,
        enable_categorical_kway: bool = True,
        enable_oblique: bool = False,
        enable_interactions: bool = False,
        cat_max_groups: int = 8,
        oblique_k: int = 6,
    ):
        if mode not in {"hist", "approx", "adaptive"}:
            raise ValueError(
                "BinnedStrategy.mode must be one of {'hist','approx','adaptive'}"
            )
        self.mode = mode
        self.binner = binner
        self._custom_provider = edges_provider

        # plug-in generators (Axis always first to ensure baseline candidate)
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
        # nothing to do here (exact path pre-sorts; binned path is prepared in tree.fit)
        return

    # -------------------------------------------------------------------------
    # Edges resolution (registry-first, with adaptive overrides)
    # -------------------------------------------------------------------------
    def _edges_for(
        self, tree: "UnifiedTree", node: "TreeNode", gfi: int
    ) -> Optional[np.ndarray]:
        """
        Resolve edges for (node, global_feature_index=gfi) according to this strategy's mode.
        - hist/approx: use registry.
        - adaptive: use registry global; if a node override exists, return it; otherwise
                    optionally refine and register a node override.
        If a custom edges_provider is set, it wins outright.
        """
        # 1) Custom provider wins outright
        if self._custom_provider is not None:
            return self._custom_provider(tree, node, gfi)

        mode = self.mode  # "hist" | "approx" | "adaptive"

        # 2) Registry lookup (adaptive can have per-node overrides)
        e = None
        try:
            e = tree.bins.get_edges(
                int(gfi),
                mode=mode,
                node_id=(node.node_id if mode == "adaptive" else None),
            )
            if e is not None and e.size >= 2:
                if mode != "adaptive":
                    return e  # hist/approx done
                # adaptive: if a node override already exists, just use it
                try:
                    if tree.bins.has_node_override(mode, node.node_id, int(gfi)):
                        return e
                except AttributeError:
                    # Registry doesn't expose has_node_override; fall through and possibly refine.
                    pass
        except KeyError:
            # mode not registered; fall back to legacy below
            e = None

        # 3) Legacy parent edges fallback
        parent_edges = None
        if mode == "hist":
            lfi = tree.feature_map.get(int(gfi))
            if lfi is not None and lfi < len(tree.bin_edges):
                parent_edges = tree.bin_edges[lfi]
            return parent_edges

        if mode == "approx":
            parent_edges = getattr(tree, "_approx_edges", {}).get(int(gfi))
            if parent_edges is None:
                lfi = tree.feature_map.get(int(gfi))
                if lfi is not None and lfi < len(tree.bin_edges):
                    parent_edges = tree.bin_edges[lfi]
            return parent_edges

        # ---- ADAPTIVE MODE ----
        # Prefer registry global (e); else legacy adaptive; else legacy hist
        if e is not None and e.size >= 2:
            parent_edges = e
        if parent_edges is None:
            parent_edges = getattr(tree, "_adaptive_edges", {}).get(int(gfi))
        if parent_edges is None:
            lfi = tree.feature_map.get(int(gfi))
            if lfi is not None and lfi < len(tree.bin_edges):
                parent_edges = tree.bin_edges[lfi]

        # If still nothing or degenerate, bail
        if parent_edges is None or parent_edges.size < 2:
            return parent_edges

        # 4) Node-local refinement cache
        key = (node.node_id, int(gfi))
        if hasattr(tree, "_node_adaptive_edges"):
            cached = tree._node_adaptive_edges.get(key)
            if cached is not None and cached.size >= 2:
                return cached

        # 5) If refinement disabled or no binner, return parent
        if not getattr(tree, "_refinement_enabled", False) or self.binner is None:
            return parent_edges

        # 6) Need local feature index
        lf = tree.feature_map.get(int(gfi))
        if lf is None:
            return parent_edges

        # 7) Node values
        try:
            v_node = node.get_feature_values(int(lf))
        except Exception:
            return parent_edges

        if not np.any(np.isfinite(v_node)):
            if not hasattr(tree, "_node_adaptive_edges"):
                tree._node_adaptive_edges = {}
            tree._node_adaptive_edges[key] = parent_edges
            return parent_edges

        # 8) Refine
        refined, _meta = self.binner.create_node_refined_bins(
            feature_values=v_node,
            gradients=node.gradients,
            hessians=node.hessians,
            feature_idx=int(gfi),
            node_id=str(node.node_id),
            tree_depth=int(node.depth),
            parent_edges=parent_edges,
            lambda_reg=float(tree.lambda_),
            gamma=float(tree.gamma),
        )

        if refined is None or refined.size < 2:
            refined = parent_edges

        # 9) Cap to capacity and enforce strictly increasing
        nb = refined.size - 1
        cap = int(getattr(tree, "_actual_max_bins", nb))
        if nb > cap:
            q = np.linspace(0.0, 1.0, cap + 1)
            lo, hi = float(refined[0]), float(refined[-1])
            refined = lo + q * (hi - lo)

        ref = np.array(refined, dtype=np.float64, copy=True)
        for i in range(1, ref.size):
            if not (ref[i] > ref[i - 1]):
                ref[i] = np.nextafter(ref[i - 1], np.inf)
        refined = ref

        # 10) Cache & register node override
        if not hasattr(tree, "_node_adaptive_edges"):
            tree._node_adaptive_edges = {}
        tree._node_adaptive_edges[key] = refined

        try:
            tree.bins.set_node_override("adaptive", node.node_id, int(gfi), refined)
        except KeyError:
            # Mode not registered: ignore silently (training still proceeds)
            pass

        node._used_refinement = True
        return refined

    # -------------------------------------------------------------------------
    # Histogram build (+ sibling subtraction if valid)
    # -------------------------------------------------------------------------
    def _build_histograms_with_cache(
        self, tree: "UnifiedTree", node: "TreeNode"
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # Sibling subtraction is only valid if both children share identical edges/codes.
        # If adaptive overrides were used for this node, skip subtraction.
        use_subtraction = (
            tree.histogram_cache is not None
            and node.parent_hist is not None
            and node.sibling_node_id is not None
            and getattr(tree, "binned_mode", "hist") != "adaptive"
            and not getattr(tree.bins, "node_has_any_override", lambda *a, **k: False)(
                "adaptive", node.node_id
            )
        )
        if use_subtraction:
            sib = tree.histogram_cache.get(node.sibling_node_id)
            if sib is not None:
                p_g, p_h = node.parent_hist
                s_g, s_h = sib
                if p_g.shape == s_g.shape == p_h.shape == s_h.shape:
                    hist_g, hist_h = subtract_sibling_histograms(p_g, p_h, s_g, s_h)
                    tree.histogram_cache.put(node.node_id, hist_g, hist_h)
                    return hist_g, hist_h, hist_g.shape[1]

        mode = getattr(tree, "binned_mode", "hist")  # "hist" | "approx" | "adaptive"

        # Decide which codes to use:
        # - adaptive + any per-node override => re-prebin rows with node_id
        # - else => slice the global codes view if available (fast-path)
        use_node_prebin = False
        if mode == "adaptive":
            try:
                use_node_prebin = tree.bins.node_has_any_override(
                    "adaptive", node.node_id
                )
            except AttributeError:
                use_node_prebin = bool(getattr(node, "_used_refinement", False))

        if use_node_prebin:
            # Re-prebin ONLY these rows; overrides (per-node, per-feature) are applied inside the registry
            X_sub = tree._X_train_cols[node.data_indices]
            codes, missing_bin_id = tree.bins.prebin_matrix(
                X_sub,
                mode="adaptive",
                node_id=node.node_id,
                cache_key=f"adaptive:node:{node.node_id}:{X_sub.shape}",
            )
            layout = tree.bins.get_layout(mode="adaptive")
            n_bins_total = int(layout.actual_max_bins) + 1
        else:
            # Slice the global codes view if available, or synthesize it on demand
            try:
                global_codes = tree.bins.get_codes_view(mode=mode)
                if (
                    global_codes is None
                    or global_codes.shape[0] < np.max(node.data_indices) + 1
                ):
                    raise KeyError(
                        "global codes view missing or mismatched; fallback to prebin rows"
                    )
                codes = global_codes[node.data_indices]
                layout = tree.bins.get_layout(mode=mode)
                n_bins_total = int(layout.actual_max_bins) + 1
            except Exception:
                # Fallback: prebin just these rows without overrides
                X_sub = tree._X_train_cols[node.data_indices]
                codes, missing_bin_id = tree.bins.prebin_matrix(
                    X_sub,
                    mode=mode,
                    node_id=None,
                    cache_key=f"{mode}:rows:{X_sub.shape}",
                )
                layout = tree.bins.get_layout(mode=mode)
                n_bins_total = int(layout.actual_max_bins) + 1

        # Histograms from codes (Numba)
        codes = codes.astype(np.int32, copy=False)
        g_node = node.gradients.astype(np.float64, copy=False)
        h_node = node.hessians.astype(np.float64, copy=False)

        # Use the BinRegistry's builder to guarantee same missing id/capacity semantics
        hist_g, hist_h = tree.bins.build_histograms_from_codes(
            codes, g_node, h_node, n_bins_total
        )

        # Cache only if subtraction remains valid (i.e., no per-node overrides)
        if tree.histogram_cache is not None and not use_node_prebin:
            tree.histogram_cache.put(node.node_id, hist_g, hist_h)

        return hist_g, hist_h, n_bins_total

    def _update_feature_importance_for_plan(self, tree, node, sp: "SplitPlanView", hist_pair):
        fi = getattr(tree, "feature_importance_", None)
        if fi is None:
            return
        hg, hh, max_bins_total = hist_pair
        finite_bins = max_bins_total - 1

        def _book(gfi, GL, HL, GR, HR, weight=1.0):
            gain = compute_gain(float(GL), float(HL), float(GR), float(HR), float(tree.lambda_), float(tree.gamma))
            cover_h = float(HL + HR)
            if weight != 1.0:
                gain *= weight; cover_h *= weight
            fi.add_split(int(gfi), float(gain), float(cover_h))
            fi.add("split", int(gfi), float(1.0 * weight))

        if sp.kind == "axis":
            gfi = int(sp.gfi)
            lfi = int(np.where(tree.feature_indices == gfi)[0][0]) if gfi in set(tree.feature_indices) \
                else int(tree.feature_map.get(gfi, -1))
            if lfi < 0 or lfi >= hg.shape[0] or sp.bin_idx is None:
                return
            b = int(sp.bin_idx)
            g_f = hg[lfi, :finite_bins]; h_f = hh[lfi, :finite_bins]
            Gm = float(hg[lfi, finite_bins]); Hm = float(hh[lfi, finite_bins])
            GL = float(g_f[: b + 1].sum()); HL = float(h_f[: b + 1].sum())
            GR = float(g_f[b + 1 :].sum()); HR = float(h_f[b + 1 :].sum())
            if bool(sp.missing_left): GL += Gm; HL += Hm
            else: GR += Gm; HR += Hm
            _book(gfi, GL, HL, GR, HR)
            return

        if sp.kind == "kway":
            gfi = int(sp.gfi)
            lfi = int(np.where(tree.feature_indices == gfi)[0][0]) if gfi in set(tree.feature_indices) \
                else int(tree.feature_map.get(gfi, -1))
            if lfi < 0 or lfi >= hg.shape[0]:
                return
            left_bins = np.array(sorted(set(int(b) for b in sp.left_groups if 0 <= int(b) < finite_bins)), dtype=np.int32)
            GL = float(hg[lfi, left_bins].sum()) if left_bins.size else 0.0
            HL = float(hh[lfi, left_bins].sum()) if left_bins.size else 0.0
            all_bins = np.arange(finite_bins, dtype=np.int32)
            mask = np.ones(finite_bins, dtype=bool)
            if left_bins.size: mask[left_bins] = False
            GR = float(hg[lfi, all_bins[mask]].sum()) if mask.any() else 0.0
            HR = float(hh[lfi, all_bins[mask]].sum()) if mask.any() else 0.0
            Gm = float(hg[lfi, finite_bins]); Hm = float(hh[lfi, finite_bins])
            if bool(sp.missing_left): GL += Gm; HL += Hm
            else: GR += Gm; HR += Hm
            _book(gfi, GL, HL, GR, HR)
            return

        if sp.kind in ("oblique", "oblique_interaction"):
            rows = node.data_indices
            Xcols = tree._X_train_cols
            g = node.gradients.astype(np.float64, copy=False)
            h = node.hessians.astype(np.float64, copy=False)
            gfi_list = [int(x) for x in sp.features]
            w = np.asarray(sp.weights, dtype=np.float64)
            b0 = float(sp.bias)
            thr = float(sp.threshold)
            miss_left = bool(sp.missing_left)

            z = np.full(rows.size, b0, dtype=np.float64)
            any_miss = np.zeros(rows.size, dtype=np.bool_)
            for j, gfi in enumerate(gfi_list):
                lfi = tree.feature_map.get(int(gfi), -1)
                if lfi < 0 or lfi >= Xcols.shape[1]:
                    any_miss[:] = True; break
                col = Xcols[rows, lfi]
                any_miss |= ~np.isfinite(col)
                z += w[j] * np.where(np.isfinite(col), col, 0.0)

            z, any_missing = tree._project_oblique(tree._X_train_cols, rows, sp)
            thr = 0.0  # canonical
            miss_left = bool(sp.missing_left)

            go_left = np.empty(rows.size, dtype=np.bool_)
            go_left[any_missing] = miss_left
            finite_mask = ~any_missing
            go_left[finite_mask] = z[finite_mask] <= thr

            GL = float(g[go_left].sum()); HL = float(h[go_left].sum())
            GR = float(g[~go_left].sum()); HR = float(h[~go_left].sum())

            absw = np.abs(w)
            denom = float(absw.sum()) if absw.size else 1.0
            shares = (absw / denom) if denom > 0.0 else np.full(len(gfi_list), 1.0 / max(len(gfi_list),1))
            for gfi, share in zip(gfi_list, shares):
                _book(int(gfi), GL, HL, GR, HR, float(share))
            return

        # otherwise ignore

    # -------------------------------------------------------------------------
    # Strategy entrypoint
    # -------------------------------------------------------------------------
    def eval_split(self, tree: "UnifiedTree", X: np.ndarray, node: "TreeNode") -> bool:
        # guard
        if (
            node.n_samples < tree.min_samples_split
            or node.depth >= tree.max_depth
            or node.h_sum < tree.min_child_weight
        ):
            return False

        # (1) ensure edges prepared/refined (lazy touch so adaptive overrides happen)
        for gfi in tree.feature_indices:
            _ = self._edges_for(tree, node, int(gfi))

        # (2) histograms for this node
        hg, hh, max_bins_total = self._build_histograms_with_cache(tree, node)
        node.histograms = (hg, hh)

        # (3) query all generators, pick the best candidate
        best: Optional[Candidate] = None
        for gen in self._gens:
            cands = gen.generate(tree, X, node, (hg, hh, max_bins_total))
            for c in cands:
                if (best is None) or (c.gain > best.gain):
                    best = c
        if best is None or best.gain <= 0.0:
            return False

        # (4) convert into SplitPlan and stash on node
        plan: SplitPlan = to_split_plan(best)
        sp = SplitPlanView(plan)  # NEW
        node._split_plan = plan   # keep the raw plan around if you want
        node.best_gain = float(sp.gain)
        tree._plan_kind_counts[plan.kind] += 1

        node.best_gain = float(plan.gain)

        # For axis we still set the legacy fields (for compatibility)
        if plan.kind == "axis":
            node.best_feature = int(plan.gfi)
            node.best_threshold = float(plan.threshold)
            node.missing_go_left = bool(plan.missing_left)
            node.best_bin_idx = int(plan.bin_idx) if plan.bin_idx is not None else None
        else:
            node.best_feature = None
            node.best_threshold = np.nan
            node.missing_go_left = False
            node.best_bin_idx = None

        # ---------- NEW: feature importance update ----------
        # use the exact histograms we just computed so numbers match split search
        self._update_feature_importance_for_plan(
            tree, node, sp, (hg, hh, max_bins_total)
        )

        return True
        return True


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
