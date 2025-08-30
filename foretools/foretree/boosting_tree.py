from __future__ import annotations

import heapq
from collections import defaultdict
from configparser import Error
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numba
import numpy as np
import torch

# -------------------------------------------------------------------
# Keep your existing imports/utilities
from boosting_aux import *
from boosting_aux import _leaf_objective_optimal
from boosting_bin import *
from boosting_loss import *
from boosting_miss import _flatten_edges_for_numba
from numba import jit, njit

from foretools.foretree.boosting_miss import (
    MissingStrategy,
    MissingValueConfig,
    UnifiedMissingHandler,
)

# -------------------------------------------------------------------


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
    
    _used_refinement: bool = False  # whether adaptive edges were used at this node

    sorted_lists: Optional[List[np.ndarray]] = None  # len = n_local_features

    def init_sums(self):
        self.n_samples = len(self.data_indices)
        self.g_sum = float(np.sum(self.gradients)) if self.gradients.size else 0.0
        self.h_sum = float(np.sum(self.hessians)) if self.hessians.size else 0.0

    _tree_ref: Optional["UnifiedTree"] = (
        None  # Reference to parent tree for adaptive edges
    )
    
    def get_feature_values(self, lf: int) -> np.ndarray:
        """
        Return RAW (float) values for this node's samples at local feature index lf.
        Falls back to midpoints-from-codes only if raw matrix is unavailable.
        """
        tree = self._tree_ref
        if tree is None:
            raise RuntimeError("Node has no tree reference.")

        # Preferred: raw training slice aligned to feature_indices
        X_raw = getattr(tree, "_X_train_cols", None)
        if X_raw is not None:
            vals = X_raw[self.data_indices, lf]
            return vals.astype(np.float64, copy=False)

        # Fallback: reconstruct approximate values from BIN CODES (midpoints)
        if getattr(tree, "binned_local", None) is None:
            raise RuntimeError("No raw data or binned codes available on tree.")
        codes = tree.binned_local[self.data_indices, lf]

        # Resolve edges for this local feature (aligned to feature_indices)
        edges_list = getattr(tree, "bin_edges", None)
        if edges_list is None or lf >= len(edges_list):
            return np.full(codes.shape[0], np.nan, dtype=np.float64)

        edges = edges_list[lf]
        if edges is None or edges.size < 2:
            return np.full(codes.shape[0], np.nan, dtype=np.float64)

        mids = 0.5 * (edges[:-1] + edges[1:])
        miss_id = int(getattr(tree, "_missing_bin_id", len(mids)))
        out = np.empty_like(codes, dtype=np.float64)
        mask_nonmiss = codes != miss_id
        out[mask_nonmiss] = mids[np.clip(codes[mask_nonmiss], 0, len(mids) - 1)]
        out[~mask_nonmiss] = np.nan
        return out
# ====================== STRATEGY INTERFACES & SHARED UTILS ======================


class SplitStrategy(Protocol):
    def prepare(
        self, tree: "UnifiedTree", X: np.ndarray, g: np.ndarray, h: np.ndarray
    ) -> None: ...
    def eval_split(
        self, tree: "UnifiedTree", X: np.ndarray, node: "TreeNode"
    ) -> bool: ...


def _should_stop(tree: "UnifiedTree", node: "TreeNode") -> bool:
    return (
        node.n_samples < tree.min_samples_split
        or node.depth >= tree.max_depth
        or node.h_sum < tree.min_child_weight
    )

# ================================ STRATEGIES ====================================

def _build_histograms_with_cache(
    X: np.ndarray,
    idx: np.ndarray,
    feature_indices: np.ndarray,
    feature_map: Dict[int, int],
    edges_dict: Dict[int, np.ndarray],
    g_node: np.ndarray,
    h_node: np.ndarray,
    missing_handler,
    node,
    histogram_cache: Optional[HistogramCache] = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Histogram building with cache + sibling subtraction, using a RESERVED LAST BIN for missing.
    Returns:
        hist_g: (n_features, max_bins_total) where max_bins_total = actual_max_bins + 1
        hist_h: (n_features, max_bins_total)
        max_bins_total: includes the reserved missing bin
        g_miss_arr: placeholder (unused; zeros)
        h_miss_arr: placeholder (unused; zeros)
    Notes:
        - Pre-binned data must encode missing as bin == actual_max_bins (reserved last bin).
        - No per-feature missing mask scans here anymore.
    """
    n_features = len(feature_indices)
    if n_features == 0:
        return np.zeros((0, 1)), np.zeros((0, 1)), 1, np.zeros(0), np.zeros(0)

    # # --- Sibling subtraction fast path ---
    if (
        histogram_cache is not None
        and getattr(node, "parent_hist", None) is not None
        and getattr(node, "sibling_node_id", None) is not None
        and getattr(node, "_used_refinement", False) is False
    ):
        sib = histogram_cache.get(node.sibling_node_id)
        if sib is not None:
            parent_hist_g, parent_hist_h = node.parent_hist
            sibling_hist_g, sibling_hist_h = sib

            # Shapes must match (including the reserved-missing bin column)
            if (
                parent_hist_g.shape == sibling_hist_g.shape
                and parent_hist_h.shape == sibling_hist_h.shape
            ):
                hist_g, hist_h = subtract_sibling_histograms(
                    parent_hist_g, parent_hist_h, sibling_hist_g, sibling_hist_h
                )
                max_bins_total = hist_g.shape[1]

                # Cache this node’s histogram
                if histogram_cache is not None:
                    histogram_cache.put(node.node_id, hist_g, hist_h)

                # Placeholders: not used anymore
                g_miss_arr = np.zeros(n_features, dtype=np.float64)
                h_miss_arr = np.zeros(n_features, dtype=np.float64)
                return hist_g, hist_h, max_bins_total, g_miss_arr, h_miss_arr

    # --- Pre-binned fast path (required for this implementation) ---
    tree = getattr(node, "_tree_ref", None)
    if getattr(tree, "binned_local", None) is not None:
        # real (non-missing) bins that were used when pre-binning
        if hasattr(tree, "_actual_max_bins"):
            actual_max_bins = int(tree._actual_max_bins)
        elif isinstance(getattr(tree, "bin_edges", None), list) and tree.bin_edges:
            actual_max_bins = max(len(e) - 1 for e in tree.bin_edges)
        else:
            actual_max_bins = int(tree.n_bins)

        # indices for this node (already a subset)
        sample_indices = idx

        # Build histograms including the reserved missing bin.
        # Expectation: binned_local uses last id == actual_max_bins to represent missing.
        # n_bins_total = actual_max_bins + 1
        n_bins_total = actual_max_bins + 1
        hg = np.zeros((n_features, n_bins_total), dtype=np.float64)
        hh = np.zeros((n_features, n_bins_total), dtype=np.float64)

        Xb_sub = tree.binned_local[sample_indices]  # (n_node, n_features)
        # Vectorized per-feature bincount
        for lfi in range(n_features):
            bins_f = Xb_sub[:, lfi].astype(np.int64, copy=False)
            # Guard: clip any stray codes into [0, n_bins_total-1]
            np.clip(bins_f, 0, n_bins_total - 1, out=bins_f)
            hg[lfi] = np.bincount(bins_f, weights=g_node, minlength=n_bins_total)
            hh[lfi] = np.bincount(bins_f, weights=h_node, minlength=n_bins_total)

        # Cache for sibling subtraction
        if histogram_cache is not None:
            histogram_cache.put(node.node_id, hg, hh)

        # Placeholders: not used anymore
        g_miss_arr = np.zeros(n_features, dtype=np.float64)
        h_miss_arr = np.zeros(n_features, dtype=np.float64)
        return hg, hh, n_bins_total, g_miss_arr, h_miss_arr

    # --- No pre-binned data available: explicit for this path ---
    raise RuntimeError("Pre-binned data with reserved missing bin is required (tree.binned_local is None).")

def _split_stats_from_hist(
    hg: np.ndarray, hh: np.ndarray,
    g_miss_arr: np.ndarray, h_miss_arr: np.ndarray,
    lfi: int, bin_idx: int, missing_left: bool
):
    """
    Return (gL,hL,gR,hR,gP,hP) for the chosen (feature= lfi, bin=bin_idx).
    hg/hh shapes: (n_features, max_bins)
    g_miss_arr/h_miss_arr: (n_features,)
    """
    g_hist = hg[lfi]  # shape (B,)
    h_hist = hh[lfi]

    # cumulative up to and including bin_idx
    gL = float(np.sum(g_hist[:bin_idx+1]))
    hL = float(np.sum(h_hist[:bin_idx+1]))

    gR = float(np.sum(g_hist[bin_idx+1:]))
    hR = float(np.sum(h_hist[bin_idx+1:]))

    # assign missing mass
    if missing_left:
        gL += float(g_miss_arr[lfi]); hL += float(h_miss_arr[lfi])
    else:
        gR += float(g_miss_arr[lfi]); hR += float(h_miss_arr[lfi])

    gP = gL + gR
    hP = hL + hR
    return gL, hL, gR, hR, gP, hP


class HistLikeStrategy:
    """
    Unifies 'hist', 'approx', and 'adaptive' by taking an edges_provider:
        edges_provider(tree, node, gfi) -> np.ndarray[bin_edges] or None
    """

    def __init__(
        self,
        edges_provider: Callable[
            ["UnifiedTree", "TreeNode", int], Optional[np.ndarray]
        ],
    ):
        self.edges_provider = edges_provider

    def prepare(
        self, tree: "UnifiedTree", X: np.ndarray, g: np.ndarray, h: np.ndarray
    ) -> None:
        # No mandatory work here; edges may already be built by tree.fit() (approx/adaptive).
        pass

    def eval_split(self, tree: "UnifiedTree", X: np.ndarray, node: "TreeNode") -> bool:
        if _should_stop(tree, node):
            return False

        # Collect edges per feature (can be per-node if provider uses node)
        edges_dict: Dict[int, np.ndarray] = {}
        for gfi in tree.feature_indices:
            e = self.edges_provider(tree, node, int(gfi))
            if e is not None and e.shape[0] >= 2:
                edges_dict[int(gfi)] = e

        idx = node.data_indices.astype(np.int64, copy=False)
        hg, hh, max_bins, g_miss_arr, h_miss_arr = _build_histograms_with_cache(
            X,
            idx,
            tree.feature_indices,
            tree.feature_map,
            edges_dict,
            node.gradients,
            node.hessians,
            tree.missing_handler,
            node,
            tree.histogram_cache,
        )

        if g_miss_arr is None:
            g_miss_arr = np.zeros(n_features, dtype=np.float64)
        if h_miss_arr is None:
            h_miss_arr = np.zeros(n_features, dtype=np.float64)
    
        lfi, bin_idx, gain, missing_left = find_best_split_with_missing(
            hg,
            hh,
            g_miss_arr,
            h_miss_arr,
            tree.lambda_,
            tree.gamma,
            max_bins,
            tree.min_child_weight,
            tree._mono_local,
        )
        
        #print(bin_idx)
        if lfi == -1 or gain <= -1e-6:
            #print(gain)
            return False
        # Global feature id
        gfi = int(tree.feature_indices[lfi])

        # Prefer node-provided edges; otherwise fall back to tree.bin_edges
        edges = edges_dict.get(gfi)
        if edges is None:
            lfi_map = tree.feature_map.get(gfi, None)
            if lfi_map is None or lfi_map >= len(tree.bin_edges):
                return False  # no valid edges available
            edges = tree.bin_edges[lfi_map]

        # Validate bin_idx and compute a consistent numeric threshold.
        # Convention: split uses bins <= bin_idx to the left -> threshold is edges[bin_idx+1]
        if not (0 <= bin_idx < len(edges) - 1):
            return False
        thr = float(edges[bin_idx])

        node.best_feature = gfi
        node.best_threshold = thr
        node.best_gain = float(gain)
        node.best_bin_idx = int(bin_idx)
        node.missing_go_left = bool(missing_left)
        node.histograms = (hg, hh)  # used by sibling subtraction in hist fast-path
        
        # Recompute the exact components used by the gain (for clarity & robustness)
        gL, hL, gR, hR, gP, hP = _split_stats_from_hist(hg, hh, g_miss_arr, h_miss_arr, lfi, bin_idx, missing_left)

        # Split gain (same formula the splitter used)
        lam = float(tree.lambda_)
        # If your 'gain' above already equals this quantity minus gamma, you could also trust node.best_gain.
        split_gain = 0.5 * ( (gL*gL)/(hL+lam) + (gR*gR)/(hR+lam) - (gP*gP)/(hP+lam) ) - float(tree.gamma)
        if split_gain <= 0.0:
            # numerical safety: keep what the splitter returned
            split_gain = float(gain)
        # Cover: prefer Hessian mass (industry standard); you can switch to sample count if you prefer
        cover_hessian = hP

        # Aggregate into your global tracker (gain/cover/split)
        if getattr(tree, "feature_importance_", None) is not None:
            # or, if the booster injected it on the splitter/builder
            tree.feature_importance_.add_split(gfi, split_gain, cover_hessian)
            tree.feature_importance_.add("split", gfi, 1.0)
            
        return True

def _split_stats_from_list(
    lst_finite: np.ndarray,          # indices of finite rows in this node (sorted by feature)
    vals_finite: np.ndarray,         # corresponding feature values
    thr: float,                      # chosen numeric threshold
    g: np.ndarray, h: np.ndarray,    # global grad/hess arrays (1D over samples)
    g_miss: float, h_miss: float,    # total grad/hess of missing rows for this feature in this node
    miss_left: bool
):
    # partition by threshold (left: <= thr; right: > thr)
    left_mask = vals_finite <= thr
    right_mask = ~left_mask

    GL = float(g[lst_finite[left_mask]].sum())
    HL = float(h[lst_finite[left_mask]].sum())
    GR = float(g[lst_finite[right_mask]].sum())
    HR = float(h[lst_finite[right_mask]].sum())

    if miss_left:
        GL += float(g_miss); HL += float(h_miss)
    else:
        GR += float(g_miss); HR += float(h_miss)

    G = GL + GR
    H = HL + HR
    return GL, HL, GR, HR, G, H


class ExactStrategy:
    """Exact presort + scan with cached missing stats (no reserved-bin change)."""

    def prepare(
        self, tree: "UnifiedTree", X: np.ndarray, g: np.ndarray, h: np.ndarray
    ) -> None:
        tree._g_global = g
        tree._h_global = h
        tree._prepare_presort_exact(X)
        # optional: build a small cache on the tree to avoid recomputing per node
        if not hasattr(tree, "_exact_missing_cache"):
            tree._exact_missing_cache = {}  # key: (node_id, lf) -> (n_miss, g_miss, h_miss)

    def _missing_stats_for(self, tree, X, g, h, node, lf):
        key = (node.node_id, int(lf))
        cache = tree._exact_missing_cache
        if key in cache:
            return cache[key]
        stats = tree.missing_handler.get_missing_stats(
            X, g, h, node.data_indices, lf
        )
        tup = (int(stats["n_missing"]),
               float(stats["g_missing"]),
               float(stats["h_missing"]))
        cache[key] = tup
        return tup
    
    def eval_split(self, tree: "UnifiedTree", X: np.ndarray, node: "TreeNode") -> bool:
        if _should_stop(tree, node):
            return False

        g = tree._g_global
        h = tree._h_global

        best_gain = -np.inf
        best_feature = -1
        best_thr = np.nan
        best_miss_left = False

        # keep context to recompute stats for importance
        best_ctx = None  # (lst_finite, vals_finite, g_miss, h_miss, lfi_local, gfi_global)

        for lfi, gfi0 in enumerate(tree.feature_indices):
            gfi = int(gfi0)
            lst = node.sorted_lists[lfi] if node.sorted_lists is not None else None
            if lst is None or lst.size < 2 * tree.min_samples_leaf:
                continue

            lf = tree.feature_map.get(gfi, gfi)
            vals_seq = X[lst, lf]

            # finite rows only
            finite_mask = np.isfinite(vals_seq)
            if np.count_nonzero(finite_mask) < 2 * tree.min_samples_leaf:
                continue
            lst_finite = lst[finite_mask]
            vals_finite = vals_seq[finite_mask]

            # per-feature missing stats on this node
            n_miss, g_miss, h_miss = self._missing_stats_for(tree, X, g, h, node, lf)

            mono = 0
            if tree._mono_constraints_array is not None and gfi < len(tree._mono_constraints_array):
                mono = int(tree._mono_constraints_array[gfi])

            gain, thr, miss_left, _, _ = best_split_on_feature_list(
                lst_finite,
                vals_finite,
                g, h,
                g_miss, h_miss,
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
                best_ctx = (lst_finite, vals_finite, g_miss, h_miss, lfi, gfi)

        if best_feature == -1 or best_gain <= 0.0:
            return False

        node.best_feature = int(best_feature)
        node.best_threshold = float(best_thr)
        node.best_gain = float(best_gain)
        node.best_bin_idx = None
        node.missing_go_left = bool(best_miss_left)

        # ---------- Feature importance (gain / cover / split) ----------
        if best_ctx is not None:
            lst_finite, vals_finite, g_miss, h_miss, lfi_best, gfi_best = best_ctx

            GL, HL, GR, HR, G, H = _split_stats_from_list(
                lst_finite, vals_finite, best_thr, g, h, g_miss, h_miss, best_miss_left
            )

            lam = float(tree.lambda_)
            split_gain = 0.5 * ((GL*GL)/(HL+lam) + (GR*GR)/(HR+lam) - (G*G)/(H+lam)) - float(tree.gamma)
            if not np.isfinite(split_gain) or split_gain <= 0.0:
                # fall back to the splitter’s evaluated gain (already finite/valid)
                split_gain = float(best_gain)

            # Hessian cover of the parent (industry standard)
            # Parent is the current node, i.e., all data_indices
            idx = node.data_indices
            cover_hessian = float(h[idx].sum())

            # push into the global aggregator (your booster attaches this)
            agg = tree.feature_importance_ if hasattr(tree, "feature_importance_") else None
            if agg is None:
                agg = getattr(tree, "feature_importance", None)
            if agg is not None:
                agg.add_split(int(gfi_best), float(split_gain), float(cover_hessian))
                agg.add("split", int(gfi_best), 1.0)

        return True

# -------------------- bitmap helpers --------------------


@njit
def _bitmap_fill(bitmap: np.ndarray, left_vals: np.ndarray, offset: int):
    for i in range(left_vals.size):
        v = int(left_vals[i]) - offset
        if 0 <= v < bitmap.size:
            bitmap[v] = 1


@njit
def _bitmap_gather(
    bitmap: np.ndarray, node_vals: np.ndarray, offset: int, out_mask: np.ndarray
):
    # out_mask is uint8 (0/1). Caller can view as bool
    for i in range(node_vals.size):
        v = int(node_vals[i]) - offset
        out_mask[i] = 1 if (0 <= v < bitmap.size and bitmap[v] != 0) else 0


def _mask_bitmap(
    node_indices: np.ndarray, left_indices: np.ndarray, scratch: dict | None = None
) -> np.ndarray:
    lo = int(left_indices.min())
    hi = int(left_indices.max())
    width = hi - lo + 1

    # reuse buffers when available
    if scratch is None:
        scratch = {}
    bitmap = scratch.get("bitmap")
    if bitmap is None or bitmap.size < width:
        bitmap = np.zeros(width, dtype=np.uint8)
        scratch["bitmap"] = bitmap
    else:
        bitmap[:width] = 0  # reset only used slice

    _bitmap_fill(bitmap[:width], left_indices, lo)

    out = scratch.get("mask_buf")
    if out is None or out.size < node_indices.size:
        out = np.empty(node_indices.size, dtype=np.uint8)
        scratch["mask_buf"] = out

    _bitmap_gather(bitmap[:width], node_indices, lo, out[: node_indices.size])
    return out[: node_indices.size].astype(bool, copy=False)


# -------------------- searchsorted path --------------------


def _mask_searchsorted(
    node_indices: np.ndarray,
    left_indices: np.ndarray,
    left_sorted: np.ndarray | None = None,
) -> np.ndarray:
    # Sort left once; node can be unsorted
    if left_sorted is None:
        left_sorted = np.sort(left_indices, kind="mergesort")  # stable & fast for ints
    pos = np.searchsorted(left_sorted, node_indices, side="left")
    # Check equality at found positions (guard bounds)
    in_left = pos < left_sorted.size
    eq = np.zeros_like(in_left, dtype=bool)
    if left_sorted.size:
        eq[in_left] = left_sorted[pos[in_left]] == node_indices[in_left]
    return eq


# -------------------- fast adaptive mask --------------------


def create_left_mask_adaptive(
    node_indices: np.ndarray,
    left_indices: np.ndarray,
    *,
    scratch: dict | None = None,
    bitmap_max_bytes: int = 4_000_000,
) -> np.ndarray:
    """
    Faster membership mask:
      - tiny nodes: np.isin
      - compact-range left: bitmap (O(n) & very fast)
      - otherwise: vectorized searchsorted (O(n log m))
    Reuses buffers via `scratch` to minimize allocations.
    """
    n_node = node_indices.size
    n_left = left_indices.size

    if n_left == 0 or n_node == 0:
        return np.zeros(n_node, dtype=bool)
    if n_node < 96:  # tiny → np.isin is quite optimized
        return np.isin(node_indices, left_indices, assume_unique=False)

    # Heuristic: try bitmap if the value range is "reasonable" in size
    lo = int(left_indices.min())
    hi = int(left_indices.max())
    width = hi - lo + 1

    # Budget bitmap memory: width bytes (uint8). Keep it below bitmap_max_bytes.
    # Also avoid bitmap if range is wildly larger than set size.
    if width > 0 and width <= bitmap_max_bytes and width <= 16 * n_left:
        # bitmap path
        # Make sure dtype is integer & contiguous for numba kernels
        node_i32 = np.ascontiguousarray(node_indices.astype(np.int64, copy=False))
        left_i32 = np.ascontiguousarray(left_indices.astype(np.int64, copy=False))
        return _mask_bitmap(node_i32, left_i32, scratch)

    # Default: searchsorted path (sort left once, fully vectorized)
    # Works great for both small and large arrays.
    return _mask_searchsorted(
        np.ascontiguousarray(node_indices), np.ascontiguousarray(left_indices)
    )


# ================================ UNIFIED TREE ==================================


class UnifiedTree:
    """
    Single-class tree with strategy dispatch (hist/exact/approx), optional adaptive refinement.
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
        tree_method: str = "hist",  # {"hist","approx","exact"}
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
        if tree_method not in ("hist", "exact", "approx"):
            raise ValueError("tree_method must be 'hist', 'approx', or 'exact'")

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
        self.bin_edges: list[np.ndarray] = bin_edges if bin_edges is not None else []

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

        # binned views prepared by caller (optional fast path)
        self.binned_local = None

        # feature mapping
        if self.feature_indices.size > 0:
            self.feature_map = {int(f): i for i, f in enumerate(self.feature_indices)}
        else:
            self.feature_map = {}

        # monotone vector (local)
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
        self._approx_weight_mode = "hess"  # or "abs_g"
        self._adaptive_edges: Optional[Dict[int, np.ndarray]] = None
        self._feature_bin_counts: Dict[int, int] = {}
        self._feature_metadata: Dict[int, dict] = {}

        # adaptive refinement switches
        self.adaptive_hist = adaptive_hist
        self._adaptive_binner: Optional[GradientBinner] = (
            GradientBinner() if "GradientBinner" in globals() else None
        )
        self._node_adaptive_edges: Dict[Tuple[int, int], np.ndarray] = (
            {}
        )  # (node_id, gfi) -> edges
        self._refinement_enabled = False
        self._adaptive_refine_topk = 8
        self._adaptive_weight_mode = "hess"
        self.use_gpu = use_gpu

        # mono constraints array for exact path
        self._mono_constraints_array = np.zeros(
            int(self.feature_indices.max()) + 1 if self.feature_indices.size else 1,
            dtype=np.int8,
        )
        for gfi, constraint in self.monotone_constraints.items():
            if gfi < len(self._mono_constraints_array):
                self._mono_constraints_array[gfi] = np.int8(constraint)

        missing_config = MissingValueConfig(
            strategy=MissingStrategy.LEARN_DIRECTION,
            lambda_reg=0.1,  # L2 regularization for gain computation
            max_surrogate_splits=3,  # Max surrogates to find (if using surrogate strategy)
            min_surrogate_agreement=0.55,  # Minimum agreement threshold
            min_samples_leaf=5,  # Minimum samples per leaf for robust splits
            surrogate_search_features=50,  # Limit features searched for surrogates
        )
        self.missing_handler = UnifiedMissingHandler(missing_config)

        # ---- strategy wiring ----
        if self.tree_method == "exact":
            self._strategy: SplitStrategy = ExactStrategy()
        else:
            # Choose edges provider depending on mode; supports per-node refinement
            if self.adaptive_hist:

                def _adaptive_edges_provider(
                    tree: "UnifiedTree", node: "TreeNode", gfi: int
                ) -> Optional[np.ndarray]:
                    return tree._node_edges(node, gfi)  # per-node when enabled

                self._strategy = HistLikeStrategy(_adaptive_edges_provider)
            elif self.tree_method == "approx":

                def _approx_edges_provider(
                    tree: "UnifiedTree", node: "TreeNode", gfi: int
                ) -> Optional[np.ndarray]:
                    return (
                        None
                        if tree._approx_edges is None
                        else tree._approx_edges.get(int(gfi))
                    )

                self._strategy = HistLikeStrategy(_approx_edges_provider)
            else:  # classic histogram with fixed global bin_edges

                def _hist_edges_provider(
                    tree: "UnifiedTree", node: "TreeNode", gfi: int
                ) -> Optional[np.ndarray]:
                    lfi = tree.feature_map.get(int(gfi))
                    self._actual_max_bins = (
                        max(len(e) - 1 for e in tree.bin_edges) if tree.bin_edges else tree.n_bins
                    )
                    if lfi is None or lfi >= len(tree.bin_edges):
                        return None
                    return tree.bin_edges[lfi]  # list aligned to feature_indices

                self._strategy = HistLikeStrategy(_hist_edges_provider)
        self.feature_importance_ = feature_importance_
    # ----------------------------- EXACT PRE-SORT --------------------------------

    def _prepare_presort_exact(self, X: np.ndarray):
        """Optimized presort with better memory layout (kept from your code)."""
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

            # CHANGED: Use missing handler for consistent detection
            missing_mask_ordered = self.missing_handler.detect_missing(col[order])
            finite_mask = ~missing_mask_ordered
            # finite_end = np.sum(finite_mask)

            self._sorted_idx[gfi] = order[finite_mask]  # Only finite values
            self._missing_idx[gfi] = order[missing_mask_ordered]  # Missing values

        self._node_of_sample = np.full(n, -1, dtype=np.int32)

    # ------------------------------ APPLY SPLIT ----------------------------------

    def _new_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def _apply_split(
        self, X: np.ndarray, node: "TreeNode"
    ) -> Optional[Tuple["TreeNode", "TreeNode"]]:
        """Apply split with optimized missing value handling."""
        if node.best_feature is None:
            return None

        gfi = int(node.best_feature)
        lf = self.feature_map[gfi]
        threshold = float(node.best_threshold)
        missing_left = bool(node.missing_go_left)

        # ==================== EXACT PATH ====================
        if self.tree_method == "exact" and node.sorted_lists is not None:
            # Use missing handler for consistency
            gradients = node.gradients
            hessians = node.hessians
            split_result = self.missing_handler.compute_split_with_missing(
                X, gradients, hessians, node.data_indices, lf, threshold
            )
            li, ri = split_result.left_indices, split_result.right_indices


            if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
                return None

            lid, rid = self._new_id(), self._new_id()

            # Efficient boolean mask for gradients/hessians
            # node_indices_set = set(node.data_indices)
            left_mask = create_left_mask_adaptive(node.data_indices, li)

            ln = TreeNode(
                lid,
                li,
                node.gradients[left_mask],
                node.hessians[left_mask],
                node.depth + 1,
                parent_hist=node.histograms,
                is_left_child=True,
            )
            rn = TreeNode(
                rid,
                ri,
                node.gradients[~left_mask],
                node.hessians[~left_mask],
                node.depth + 1,
                parent_hist=node.histograms,
                is_left_child=False,
            )

            # Partition sorted lists using the same split
            mark_left = self._work_buffers["mark_left"]
            mark_left[:] = 0
            mark_left[li] = 1  # Mark all left indices

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

        # ==================== HIST/APPROX/ADAPTIVE PATH ====================
        else:
            # Use optimized missing handler
            gradients = node.gradients
            hessians = node.hessians
            split_result = self.missing_handler.compute_split_with_missing(
                X, gradients, hessians, node.data_indices, lf, threshold
            )
            li, ri = split_result.left_indices, split_result.right_indices

            if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
                return None

            lid, rid = self._new_id(), self._new_id()
            left_mask = create_left_mask_adaptive(node.data_indices, li)

            ln = TreeNode(
                lid,
                li,
                node.gradients[left_mask],
                node.hessians[left_mask],
                node.depth + 1,
                parent_hist=node.histograms,
                is_left_child=True,
            )
            rn = TreeNode(
                rid,
                ri,
                node.gradients[~left_mask],
                node.hessians[~left_mask],
                node.depth + 1,
                parent_hist=node.histograms,
                is_left_child=False,
            )
            ln.sorted_lists = None
            rn.sorted_lists = None
            ln._tree_ref = self  # Add reference to tree
            rn._tree_ref = self  # Add reference to tree

        # ==================== COMMON FINALIZATION ====================
        ln.sibling_node_id, rn.sibling_node_id = rid, lid
        ln.init_sums()
        rn.init_sums()

        node.left_child, node.right_child = ln, rn
        node.is_leaf = False
        self.nodes[lid], self.nodes[rid] = ln, rn

        return ln, rn

    # ------------------------------- GROWTH --------------------------------------

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
        root._tree_ref = self  # Add reference to tree
        self.nodes[rid] = root

        # EXACT: initialize root's per-feature sorted lists
        if self.tree_method == "exact":
            nlf = len(self.feature_indices)
            root.sorted_lists = [None] * nlf
            for lfi, gfi in enumerate(self.feature_indices):
                root.sorted_lists[lfi] = self._sorted_idx[int(gfi)]
            if self._node_of_sample is not None:
                self._node_of_sample[:] = -1
                self._node_of_sample[root_idx] = rid

        heap = []
        if self._strategy.eval_split(self, X, root):
            heapq.heappush(heap, (-root.best_gain, root.node_id, root))
        else:
            root.leaf_value = calc_leaf_value_newton(
                root.g_sum, root.h_sum, self.lambda_, self.alpha, self.max_delta_step
            )

        leaves = 1
        while heap and leaves < self.max_leaves:
            _, _, best = heapq.heappop(heap)
            children = self._apply_split(X, best)
            if not children:
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
            if self.tree_method != "exact" and best.histograms is not None:
                ln.parent_hist = best.histograms
                rn.parent_hist = best.histograms

            for ch in (ln, rn):
                if _should_stop(self, ch):
                    ch.leaf_value = calc_leaf_value_newton(
                        ch.g_sum,
                        ch.h_sum,
                        self.lambda_,
                        self.alpha,
                        self.max_delta_step,
                    )
                    continue
                if self._strategy.eval_split(self, X, ch):
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
        g32 = g.astype(np.float32, copy=False)
        h32 = h.astype(np.float32, copy=False)

        rid = self._new_id()
        root = TreeNode(rid, np.arange(n, dtype=np.int64), g32, h32, depth=0)
        root.init_sums()
        self.root = root
        root._tree_ref = self  # Add reference to tree
        self.nodes[rid] = root

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
            if _should_stop(self, node) or not self._strategy.eval_split(self, X, node):
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
            if self.tree_method != "exact" and node.histograms is not None:
                ln.parent_hist = node.histograms
                rn.parent_hist = node.histograms
            q.extend((ln, rn))

    # ------------------------------- FIT -----------------------------------------

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
        self._node_adaptive_edges.clear()

        # strategy-specific preparation
        if self.tree_method == "exact":
            # exact strategy will call _prepare_presort_exact in its prepare()
            self._approx_edges = None
            self._adaptive_edges = None
        elif self.tree_method == "approx":
            self._g_global = None
            self._h_global = None
            self._prepare_approx(
                X, g, h
            )  # builds self._approx_edges and self.bin_edges
        else:  # hist
            self._g_global = None
            self._h_global = None
            if self.adaptive_hist:
                self._prepare_adaptive(
                    X, g, h
                )  # builds self._adaptive_edges and self.bin_edges

            # prebinned fast path if caller prepared arrays
            if not self.adaptive_hist:
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
                        self.binned_local = self._binned[
                            np.ix_(self._row_indexer, self._feature_mask)
                        ]
                else:
                    self.binned_local = None

        # let strategy do any final prep
        self._strategy.prepare(self, X, g, h)

        # grow
        grow_fn = (
            self._grow_leaf_wise
            if self.growth_policy == "leaf_wise"
            else self._grow_level_wise
        )
        grow_fn(X, g, h, feature_importance)

        # finalize leaves without value
        for n in self.nodes.values():
            if n.is_leaf and n.leaf_value is None:
                n.leaf_value = calc_leaf_value_newton(
                    n.g_sum, n.h_sum, self.lambda_, self.alpha, self.max_delta_step
                )

        self._build_pred_arrays()
        if self._pred_arrays is not None:
            self._pred_arrays = tuple(
                np.ascontiguousarray(a) if isinstance(a, np.ndarray) else a
                for a in self._pred_arrays
            )
        return self

    # ----------------------- APPROX EDGES (UNCHANGED CORE) -----------------------
    def _gradient_aware_quantile_edges(
        self, x: np.ndarray, g: np.ndarray, h: np.ndarray, n_bins: int
    ) -> np.ndarray:
        finite_mask = np.isfinite(x) & np.isfinite(g) & np.isfinite(h) & (h > 1e-12)
        if not np.any(finite_mask):
            return np.array([0.0, 1.0], dtype=np.float64)
        x_f = x[finite_mask].astype(np.float64, copy=False)
        g_f = g[finite_mask].astype(np.float64, copy=False)
        h_f = h[finite_mask].astype(np.float64, copy=False)
        order = np.argsort(x_f)
        x_sorted = x_f[order]
        g_sorted = g_f[order]
        h_sorted = h_f[order]

        window_size = max(20, len(x_f) // (n_bins * 3))
        if window_size <= 0:
            return np.array([x_sorted[0], x_sorted[-1]], dtype=np.float64)

        importance = []
        positions = []
        step = max(1, window_size // 2)
        for i in range(0, len(x_f) - window_size, step):
            window_g = g_sorted[i : i + window_size]
            window_h = h_sorted[i : i + window_size]
            imp = np.mean(np.abs(window_g)) / (np.mean(window_h) + 1e-8)
            importance.append(imp)
            positions.append(i)
        if not importance:
            return np.array([x_sorted[0], x_sorted[-1]], dtype=np.float64)

        importance = np.array(importance, dtype=np.float64)
        importance /= importance.sum()
        cum_importance = np.cumsum(importance)
        target_probs = np.linspace(0, 1, n_bins + 1)[1:-1]
        cut_indices = np.searchsorted(cum_importance, target_probs)

        edges = [x_sorted[0]]
        for idx in cut_indices:
            pos = positions[min(idx, len(positions) - 1)]
            edges.append(x_sorted[min(pos + window_size // 2, len(x_sorted) - 1)])
        edges.append(x_sorted[-1])
        return np.unique(np.array(edges, dtype=np.float64))
    
    def _prepare_approx(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> None:
        """Build per-feature gradient-aware (or weighted-quantile) edges, then prebin with a reserved missing bin."""
        self._approx_edges = {}

        # Features to consider
        feats = (
            self.feature_indices.astype(np.int32, copy=False)
            if self.feature_indices.size
            else np.arange(X.shape[1], dtype=np.int32)
        )

        # Choose weighting mode for edge placement (kept from your behavior)
        weight_mode = getattr(self, "_approx_weight_mode", "hess")
        if weight_mode == "abs_g":
            w_all = np.abs(g).astype(np.float64, copy=False)
        else:
            w_all = (h + 1e-12).astype(np.float64, copy=False)

        # Build per-feature edges on finite values
        for gfi in feats:
            lf = int(self.feature_map.get(int(gfi), int(gfi)))
            if lf < 0 or lf >= X.shape[1]:
                self._approx_edges[int(gfi)] = np.array([0.0, 1.0], dtype=np.float64)
                continue

            col = X[:, lf].astype(np.float64, copy=False)
            finite = np.isfinite(col)
            if not np.any(finite):
                self._approx_edges[int(gfi)] = np.array([0.0, 1.0], dtype=np.float64)
                continue

            x_f = col[finite]
            # If you ever want to use the weights explicitly in your gradient-aware routine,
            # pass w_f (currently unused by _gradient_aware_quantile_edges)
            # w_f = w_all[finite]

            uniq = np.unique(x_f)
            max_bins_feat = max(1, min(self.n_bins, uniq.size - 1))

            edges = self._gradient_aware_quantile_edges(
                x_f, g[finite], h[finite], max_bins_feat
            )
            edges = np.unique(np.asarray(edges, dtype=np.float64))
            if edges.size < 2:
                vmin, vmax = float(np.min(x_f)), float(np.max(x_f))
                edges = np.array([vmin, vmax + (1e-12 if vmin == vmax else 0.0)], dtype=np.float64)

            self._approx_edges[int(gfi)] = edges

        # Materialize edges aligned to self.feature_indices (training column order)
        edges_list = [
            self._approx_edges.get(int(gfi), np.array([0.0, 1.0], dtype=np.float64))
            for gfi in self.feature_indices
        ]

        # Actual real (non-missing) bins used by any feature
        actual_max_bins = (
            max((len(edges) - 1) for edges in edges_list) if edges_list else self.n_bins
        )
        actual_max_bins = max(1, min(actual_max_bins, self.n_bins))
        self._actual_max_bins = actual_max_bins  # real bins only

        # Pick compact dtype for binned matrix
        if self._actual_max_bins <= 255:
            bin_dtype = np.uint8
        elif self._actual_max_bins <= 65535:
            bin_dtype = np.uint16
        else:
            bin_dtype = np.int32

        # Prebin with RESERVED last bin for missing (id == actual_max_bins)
        X_train_cols = X[:, self.feature_indices] if X.shape[1] != len(self.feature_indices) else X
        self._binned, self._missing_bin_id = self.missing_handler.prebin_matrix_with_reserved_missing(
            X_train_cols, edges_list, self._actual_max_bins, out_dtype=bin_dtype
        )
        self.binned_local = self._binned  # expose to fast path

        # Store edges for predict-time binning parity
        self.bin_edges = edges_list  # list aligned to feature_indices

    # ------------------- ADAPTIVE EDGES + PER-NODE REFINEMENT --------------------

    def _create_enhanced_fallback(
        self, gfi: int, X: np.ndarray, idx: np.ndarray, lf: int = None
    ):
        if lf is None:
            lf = int(self.feature_map.get(int(gfi), int(gfi)))

        if lf < 0 or lf >= X.shape[1]:
            edges = np.array([0.0, 1.0], dtype=np.float64)
            reason = "invalid_feature_index"
        else:
            col = X[:, lf] if idx is None else X[idx, lf]
            finite = np.isfinite(col)
            if not np.any(finite):
                edges = np.array([0.0, 1.0], dtype=np.float64)
                reason = "no_finite_values"
            else:
                col_clean = col[finite].astype(np.float64, copy=False)
                vmin, vmax = float(np.min(col_clean)), float(np.max(col_clean))
                if vmin == vmax:
                    edges = np.array([vmin, vmin + 1e-12], dtype=np.float64)
                    reason = "constant_feature"
                else:
                    n_unique = len(np.unique(col_clean))
                    if n_unique <= 32:
                        edges_u = np.unique(col_clean).astype(np.float64)
                        if len(edges_u) == n_unique and n_unique > 1:
                            bnds = [
                                (edges_u[i] + edges_u[i + 1]) / 2
                                for i in range(len(edges_u) - 1)
                            ]
                            edges = np.array(
                                [edges_u[0] - 1e-12] + bnds + [edges_u[-1] + 1e-12]
                            )
                        else:
                            edges = np.array([vmin, vmax], dtype=np.float64)
                        reason = "categorical_fallback"
                    else:
                        n_bins = min(32, max(8, len(col_clean) // 100))
                        edges = np.quantile(col_clean, np.linspace(0, 1, n_bins + 1))
                        edges = np.unique(edges).astype(np.float64)
                        reason = "quantile_fallback"

        self._adaptive_edges[int(gfi)] = edges
        self._feature_bin_counts[int(gfi)] = max(1, len(edges) - 1)
        self._feature_metadata[int(gfi)] = {
            "strategy": "enhanced_fallback",
            "reason": reason,
        }
        
    def _prepare_adaptive(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        *,
        subsample: int = 0,
        weight_mode: str = "hess",   # kept for signature compat; not used by new binner
        rng_seed: int = 42,
        enable_refinement: bool = True,
    ) -> None:
        """
        Build per-feature adaptive edges using the new GradientBinner.
        Pre-bin with a RESERVED last bin for missing (bin id = actual_max_bins).
        """
        self._adaptive_edges = {}
        self._feature_bin_counts = {}
        self._feature_metadata = {}

        feats = (
            self.feature_indices.astype(np.int32, copy=False)
            if self.feature_indices.size
            else np.arange(X.shape[1], dtype=np.int32)
        )
        n_samples, n_features = X.shape

        # ---- Instantiate SOTA binner with compatible config ----
        if not hasattr(self, "_adaptive_binner") or self._adaptive_binner is None:
            # Heuristics
            base_bins = int(np.clip(np.sqrt(max(n_samples, 1) / 100.0), 16, 128))
            max_bins = 64 if n_samples < 1_000 else (256 if n_samples < 10_000 else 512)
            gain_thr = 0.05 if n_samples < 1_000 else (0.02 if n_samples < 10_000 else 0.01)
            # fraction of features to consider for refinement decisions (kept for parity)
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

                # node refinement knobs (the class will still auto-fallback by threshold/depth)
                node_refinement_threshold=max(100, n_samples // 200),
                max_refinement_depth=min(15, int(getattr(self, "max_depth", 8)) + 2),
                refinement_feature_fraction=refinement_fraction,
                refinement_min_correlation=0.05,

                categorical_threshold=32,
                overlap_merge_threshold=2,
                eps=1e-12,
            )
            self._adaptive_binner = GradientBinner(cfg)

        # ---- Subsample (optional) for building global edges ----
        if subsample and subsample < X.shape[0]:
            effective_subsample = min(subsample, max(10_000, n_features * 100))
            rs = np.random.RandomState(rng_seed)
            idx = rs.choice(X.shape[0], size=effective_subsample, replace=False)
        else:
            idx = np.arange(X.shape[0], dtype=np.int64)

        g_use = g[idx].astype(np.float64, copy=False)
        h_use = h[idx].astype(np.float64, copy=False)
        X_sub = X[idx]

        # ---- Parallel batch path ----
        if self._adaptive_binner.config.use_parallel and len(feats) > 10:
            batch_data, valid_feats, fallback_feats = [], [], []
            for gfi in feats:
                lf = int(self.feature_map.get(int(gfi), int(gfi)))
                if lf < 0 or lf >= X.shape[1]:
                    fallback_feats.append(gfi); continue
                col = X_sub[:, lf].astype(np.float64, copy=False)
                finite = np.isfinite(col)
                if (finite.sum() < 10) or (np.std(col[finite]) < 1e-12):
                    fallback_feats.append(gfi); continue
                batch_data.append((col, g_use, h_use, int(gfi)))
                valid_feats.append(gfi)

            if batch_data:
                try:
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
                except Exception as e:
                    print(f"Warning: adaptive batch processing failed: {e}")
                    fallback_feats.extend(valid_feats)

            for gfi in fallback_feats:
                self._create_enhanced_fallback(gfi, X_sub, idx)

        # ---- Sequential path ----
        else:
            for gfi in feats:
                lf = int(self.feature_map.get(int(gfi), int(gfi)))
                if lf < 0 or lf >= X_sub.shape[1]:
                    self._create_enhanced_fallback(gfi, X_sub, idx); continue
                col = X_sub[:, lf].astype(np.float64, copy=False)
                finite = np.isfinite(col)
                if finite.sum() < 5:
                    self._create_enhanced_fallback(gfi, X_sub, idx, lf); continue
                try:
                    edges, meta = self._adaptive_binner.create_bins(
                        col, g_use, h_use,
                        feature_idx=int(gfi),
                        lambda_reg=getattr(self, "lambda_", 1.0),
                        gamma=getattr(self, "gamma", 0.0),
                    )
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
                except Exception as e:
                    print(f"Warning: adaptive binning failed for feature {gfi}: {e}")
                    self._create_enhanced_fallback(gfi, X_sub, idx, lf)

        # Ensure every feature has edges
        for gfi in feats:
            if int(gfi) not in self._adaptive_edges:
                self._create_enhanced_fallback(gfi, X, idx)

        # Align + prebin (reserve last bin for missing)
        edges_list = []
        for gfi in self.feature_indices:
            e = self._adaptive_edges.get(int(gfi))
            if e is None or e.size < 2:
                lf = int(self.feature_map.get(int(gfi), int(gfi)))
                if 0 <= lf < X.shape[1]:
                    col = X[:, lf]
                    finite = np.isfinite(col)
                    if np.any(finite):
                        vmin, vmax = np.min(col[finite]), np.max(col[finite])
                        e = np.array([vmin, vmax + 1e-12] if vmin == vmax else [vmin, vmax], np.float64)
                    else:
                        e = np.array([0.0, 1.0], np.float64)
                else:
                    e = np.array([0.0, 1.0], np.float64)
                self._adaptive_edges[int(gfi)] = e
            edges_list.append(self._adaptive_edges[int(gfi)])

        actual_max_bins = max((len(e) - 1) for e in edges_list) if edges_list else int(self.n_bins)
        actual_max_bins = max(1, min(actual_max_bins, int(self.n_bins)))
        self._actual_max_bins = actual_max_bins

        if self._actual_max_bins <= 255:
            bin_dtype = np.uint8
        elif self._actual_max_bins <= 65535:
            bin_dtype = np.uint16
        else:
            bin_dtype = np.int32

        X_train_cols = X[:, self.feature_indices] if X.shape[1] != len(self.feature_indices) else X
        self._binned, self._missing_bin_id = self.missing_handler.prebin_matrix_with_reserved_missing(
            X_train_cols, edges_list, self._actual_max_bins, out_dtype=bin_dtype
        )
        self.binned_local = self._binned
        self.bin_edges = edges_list  # store for predict parity

        # Node-level refinement toggle (external to binner)
        self._refinement_enabled = bool(enable_refinement)
        self._node_binned_cache = {}
        self._X_train_cols = X_train_cols  # shape: (n_samples, len(self.feature_indices))


    def _prebin_values_with_reserved_missing(self, values: np.ndarray, edges: np.ndarray):
        """
        Return (possibly downsampled) edges and per-sample bin ids using reserved missing id.
        JITs only the binning loop; keeps object-y bits in Python.
        """
        # --- sanitize edges minimalistically in Python ---
        edges = np.asarray(edges, np.float64)
        if edges.size < 2 or not np.all(np.isfinite(edges)):
            m = 0.0
            edges = np.array([m - 1e-12, m + 1e-12], np.float64)

        nb = edges.size - 1
        cap = int(getattr(self, "_actual_max_bins", nb))
        if nb > cap:
            # regular linspace is fine; no need for np.unique if we ensure strictly increasing
            q = np.linspace(0.0, 1.0, cap + 1)
            lo, hi = float(edges[0]), float(edges[-1])
            edges = lo + q * (hi - lo)
            # enforce strict monotonicity (handles degenerate tiny ranges)
            for i in range(1, edges.size):
                if not np.isfinite(edges[i]) or edges[i] <= edges[i-1]:
                    edges[i] = np.nextafter(edges[i-1], np.inf)
            nb = cap

        miss_id = int(getattr(self, "_missing_bin_id", nb))  # typically == cap

        # --- numba hot path: values -> int32 codes ---
        codes32 = prebin_kernel(values.astype(np.float64, copy=False), edges, nb, miss_id)

        # match the global binned dtype if needed
        if hasattr(self, "_binned"):
            out_dtype = self._binned.dtype
            if out_dtype != np.int32:
                codes = codes32.astype(out_dtype, copy=False)
            else:
                codes = codes32
        else:
            codes = codes32

        return edges, codes


    def _node_edges(self, node: "TreeNode", gfi: int) -> Optional[np.ndarray]:
        """
        Return edges for (node, gfi). Uses GradientBinner per-node refinement:
        start from global edges and, when enabled/applicable, subdivide important parent bins.
        Also writes refined bin codes back into binned_local for this node/feature only.
        """
        # ---------- Resolve parent/global edges ----------
        parent_edges = self._adaptive_edges.get(int(gfi))
        if parent_edges is None and isinstance(self.bin_edges, (list, tuple)):
            lfi_fallback = self.feature_map.get(int(gfi))
            if lfi_fallback is not None and 0 <= lfi_fallback < len(self.bin_edges):
                parent_edges = self.bin_edges[lfi_fallback]
        if parent_edges is None or parent_edges.size < 2:
            return parent_edges

        # ---------- Refinement gate ----------
        if not getattr(self, "_refinement_enabled", False):
            return parent_edges

        # Node/feature cache hit
        key = (node.node_id, int(gfi))
        if hasattr(self, "_node_adaptive_edges"):
            cached = self._node_adaptive_edges.get(key)
            if cached is not None and cached.size >= 2:
                return cached

        # ---------- Resolve local feature index (lf) aligned to _X_train_cols / binned_local ----------
        lf = self.feature_map.get(int(gfi), None)
        if lf is None or not (0 <= int(lf) < len(self.feature_indices)):
            # fallback O(n) scan if feature_map was not populated fully
            try:
                lf = int(np.where(self.feature_indices == int(gfi))[0][0])
            except Exception:
                return parent_edges  # cannot resolve; bail safely
        lf = int(lf)

        # ---------- Fetch RAW values for this node ----------
        vals_accessor = getattr(node, "get_feature_values", None)
        if not callable(vals_accessor):
            return parent_edges
        v_node = vals_accessor(lf)  # MUST be raw (float) values

        # If we accidentally got codes (all small non-negative ints), decode to midpoints
        act_bins = int(getattr(self, "_actual_max_bins", 0))
        if act_bins > 0:
            v = v_node[np.isfinite(v_node)]
            if v.size and np.all((v == np.floor(v)) & (v >= 0) & (v <= act_bins)):
                # Looks like codes → decode using current (parent) edges midpoints
                edges_for_decode = parent_edges
                mids = 0.5 * (edges_for_decode[:-1] + edges_for_decode[1:])
                miss_id = int(getattr(self, "_missing_bin_id", len(mids)))
                nonmiss = v_node != miss_id
                # convert in-place
                v_node = v_node.astype(np.float64, copy=False)
                v_node[nonmiss] = mids[np.clip(v_node[nonmiss].astype(np.int64, copy=False), 0, len(mids) - 1)]
                v_node[~nonmiss] = np.nan

        # If no finite values, nothing to refine—keep parent edges and mark missing bins later
        if not np.any(np.isfinite(v_node)):
            if not hasattr(self, "_node_adaptive_edges"):
                self._node_adaptive_edges = {}
            self._node_adaptive_edges[key] = parent_edges
            return parent_edges

        # ---------- Ask binner to refine ----------
        try:
            refined, meta = self._adaptive_binner.create_node_refined_bins(
                feature_values=v_node,
                gradients=node.gradients,
                hessians=node.hessians,
                feature_idx=int(gfi),
                node_id=str(node.node_id),
                tree_depth=int(getattr(node, "depth", 0)),
                parent_edges=parent_edges,
                lambda_reg=getattr(self, "lambda_", 1.0),
                gamma=getattr(self, "gamma", 0.0),
            )
            # print(f"Node {node.node_id} Feature {gfi} refinement: {meta}")
        except Exception as e:
            refined, meta = parent_edges, {"strategy": "refinement_failed", "refined": False, "error": str(e)}

        # Fallback to parent if degenerate
        if refined is None or refined.size < 2:
            refined = parent_edges

        # ---------- Prebin node values with refined edges (capacity-capped, reserved missing id) ----------

        # Ensure helper attributes exist
        if not hasattr(self, "_missing_bin_id"):
            # by design, reserved missing id == _actual_max_bins
            setattr(self, "_missing_bin_id", int(getattr(self, "_actual_max_bins", 0)))

        refined, node_bins = self._prebin_values_with_reserved_missing(v_node, refined)

        # Cache edges for provider
        if not hasattr(self, "_node_adaptive_edges"):
            self._node_adaptive_edges = {}
        self._node_adaptive_edges[key] = refined

        # ---------- Write-through into global pre-binned matrix for JUST this node/feature ----------
        idx_rows = node.data_indices.astype(np.int64, copy=False)
        # guard: shape must match
        if node_bins.shape[0] == idx_rows.shape[0] and 0 <= lf < self.binned_local.shape[1]:
            self.binned_local[idx_rows, lf] = node_bins

            # mark this node as refined (children must avoid sibling-subtraction)
            setattr(node, "_used_refinement", True)
            # print(f"Node {node.node_id} Feature {gfi} used refinement with {len(refined)-1} bins.")

            # Optional: quick sanity (can be commented out in prod)
            # codes must be within [0, _actual_max_bins] where last is missing
            # assert np.all((node_bins >= 0) & (node_bins <= int(self._actual_max_bins)))

        return refined


    def missing_bin_id(self, max_bins: int) -> int:
        """Return the reserved bin index for missing values."""
        return max_bins  # last id

    def prebin_matrix_with_reserved_missing(
        self, X: np.ndarray, bin_edges_list: list, max_bins: int, out_dtype=np.int32
    ) -> Tuple[np.ndarray, int]:
        """
        Vectorized prebinning with a dedicated last bin for missing.
        Returns (Xb, missing_bin_id), where bins in each column are 0..max_bins-1 (real)
        and 'missing' uses index = max_bins.
        """
        flat, starts = _flatten_edges_for_numba(bin_edges_list)
        binned = encode_with_reserved_missing_all_features(X, flat, starts)
        if out_dtype != np.int32:
            binned = binned.astype(out_dtype, copy=False)
        return binned, self.missing_bin_id(max_bins)
    
    # --------------------------- PREDICTION ARRAYS -------------------------------

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

    # -------------------------------- PREDICT ------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        if self._pred_arrays is None:
            raise RuntimeError("Prediction arrays not built; cannot predict.")

        (nf, nt, nm, lc, rc, lv, is_leaf_flags, fmap, nbin, nuse) = self._pred_arrays

        # Ensure the same column order as training
        X_local = X[:, self.feature_indices] if X.shape[1] != len(self.feature_indices) else X

        # Prefer binned path for hist-like trees
        if self.tree_method != "exact" and np.any(nuse):
            # Use the same edges & bin count used at training
            edges_local = [
                e if (e is not None and e.size >= 2) else np.array([0.0, 1.0], dtype=np.float64)
                for e in self.bin_edges
            ]
            # Prebin with RESERVED last bin for missing (id == self._actual_max_bins)
            out_dtype = getattr(self, "_binned", None).dtype if getattr(self, "_binned", None) is not None else (
                np.uint8 if getattr(self, "_actual_max_bins", 256) <= 255 else
                (np.uint16 if self._actual_max_bins <= 65535 else np.int32)
            )
            Xb_pred, missing_bin_id = self.missing_handler.prebin_matrix_with_reserved_missing(
                X_local, edges_local, self._actual_max_bins, out_dtype=out_dtype
            )

            # Binned predictor must treat `missing_bin_id` specially:
            # if feature bin == missing_bin_id: follow nm[node_id] (missing_go_left),
            # else compare bin to stored node.best_bin_idx.
            return predict_tree_binned_with_missingbin(
                Xb_pred,
                missing_bin_id,
                nf, nbin, nm, lc, rc, lv, is_leaf_flags, fmap, self.root.node_id
            )

        # Exact path: original fast numba predictors
        if self.missing_handler.config.finite_check_mode == "strict":
            return predict_tree_numba(
                X_local, nf, nt, nm, lc, rc, lv, is_leaf_flags, fmap, self.root.node_id
            )
        else:
            missing_mask = self.missing_handler.detect_missing(X_local)
            return predict_tree_numba_with_missing_mask(
                X_local, missing_mask, nf, nt, nm, lc, rc, lv, is_leaf_flags, fmap, self.root.node_id
            )

    # --------------------------------- UTILS -------------------------------------


    def get_feature_importance(self, mode: str = "gain", cover_def: str = "hessian"):
        """
        mode ∈ {"gain", "cover", "split", "gain_x_cover"}
        cover_def ∈ {"hessian", "samples"}
        Returns: dict {global_feat_idx: importance_value}
        """
        imp = defaultdict(float)

        for nd in self.nodes.values():
            # Only accepted internal splits
            if nd.is_leaf or nd.best_feature is None:
                continue

            # Map local index -> global dataset index (critical!)
            j_local = int(nd.best_feature)
            j_global = int(self.feature_indices[j_local])  # or self._feature_mask[j_local]

            # Gain: use the exact quantity you optimized to decide the split.
            # If nd.best_gain already stores the net improvement (after λ, γ), use it.
            gain = float(getattr(nd, "best_gain", 0.0))
            if gain <= 0.0:
                continue

            # Cover: prefer Hessian mass if you split with Newton stats; else samples.
            if cover_def == "hessian":
                cover = float(getattr(nd, "sum_hess", nd.n_samples))
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
                node.leaf_value = calc_leaf_value_newton(
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
        use_gpu=False,
        feature_importance_=None,  # ignored; for sklearn compatibility
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
        use_gpu=False,
        feature_importance_=None,  # ignored; for sklearn compatibility
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
        )
