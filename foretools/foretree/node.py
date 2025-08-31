from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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
    _split_plan: Optional["SplitPlan"] = None  # Optional plan for this node's split
    
    def get_feature_values(self, lf: int) -> np.ndarray:
        """
        Return RAW (float) values for this node's samples at local feature index lf.
        Preferred: raw training slice (tree._X_train_cols).
        Fallback: reconstruct from bin codes using edges from BinRegistry (honors adaptive node overrides).
        """
        tree = self._tree_ref
        if tree is None:
            raise RuntimeError("Node has no tree reference.")

        # 1) Preferred: raw values aligned to training feature order
        X_raw = getattr(tree, "_X_train_cols", None)
        if X_raw is not None:
            vals = X_raw[self.data_indices, lf]
            return vals.astype(np.float64, copy=False)

        # 2) Fallback: reconstruct from codes + edges coming from the registry
        if not hasattr(tree, "bins"):
            raise RuntimeError("BinRegistry not available on tree.")

        mode = getattr(tree, "binned_mode", "hist")  # "hist" | "approx" | "adaptive"

        # Map local feature index -> global feature id used by the registry/spec
        if hasattr(tree, "feature_indices") and lf < len(tree.feature_indices):
            gfi = int(tree.feature_indices[lf])
        else:
            gfi = int(lf)

        # Decide if we must prebin this node with overrides (adaptive only)
        use_node_prebin = False
        if mode == "adaptive":
            try:
                use_node_prebin = tree.bins.has_node_override(
                    "adaptive", self.node_id, gfi
                )
            except AttributeError:
                use_node_prebin = bool(getattr(self, "_used_refinement", False))

        # 2a) Get codes for just these rows (node-aware if needed)
        if use_node_prebin:
            # Re-prebin the node rows so per-node overrides are applied
            # We don't have raw X here; reconstructing from registry requires X rows.
            # But since raw X is unavailable (X_raw is None), this path should be rare.
            # If you *must* support this, ensure the registry can prebin from stored full-matrix raw data.
            # Otherwise, fall back to global codes view (may be slightly inconsistent if overrides exist).
            codes_view = tree.bins.get_codes_view(mode=mode)
            if codes_view is not None:
                codes = codes_view[self.data_indices, lf]
                missing_bin_id = int(
                    getattr(
                        tree,
                        "_missing_bin_id",
                        tree.bins.get_layout(mode=mode).actual_max_bins,
                    )
                )
            else:
                # Last resort: try a generic row prebin without node overrides.
                # (Still better than crashing; results may be slightly approximate.)
                X_local = getattr(tree, "_X_train_cols", None)
                if X_local is None:
                    raise RuntimeError(
                        "No raw matrix or global codes available to reconstruct values."
                    )
                X_sub = X_local[self.data_indices]
                codes_full, missing_bin_id = tree.bins.prebin_matrix(
                    X_sub, mode=mode, node_id=None
                )
                codes = codes_full[:, lf]
        else:
            # Use the cached full-matrix codes view from the registry
            codes_view = tree.bins.get_codes_view(mode=mode)
            if codes_view is None:
                # Seed on demand by prebinning these rows (no node overrides)
                X_local = getattr(tree, "_X_train_cols", None)
                if X_local is None:
                    raise RuntimeError(
                        "No raw matrix or cached codes available to reconstruct values."
                    )
                X_sub = X_local[self.data_indices]
                codes_full, missing_bin_id = tree.bins.prebin_matrix(
                    X_sub,
                    mode=mode,
                    node_id=None,
                    cache_key=f"{mode}:nodevals:{X_sub.shape}",
                )
                codes = codes_full[:, lf]
            else:
                codes = codes_view[self.data_indices, lf]
                layout = tree.bins.get_layout(mode=mode)
                missing_bin_id = int(
                    getattr(tree, "_missing_bin_id", layout.actual_max_bins)
                )

        # 2b) Resolve the exact edges used for this feature (honor adaptive override if present)
        edges = None
        try:
            if mode == "adaptive" and use_node_prebin:
                edges = tree.bins.get_edges(gfi, mode=mode, node_id=self.node_id)
            else:
                edges = tree.bins.get_edges(gfi, mode=mode, node_id=None)
        except KeyError:
            edges = None

        # Legacy fallback if registry didn’t return edges
        if edges is None:
            edges_list = getattr(tree, "bin_edges", None)
            if edges_list is not None and lf < len(edges_list):
                edges = edges_list[lf]

        # If still missing/degenerate, give NaNs
        if edges is None or getattr(edges, "size", 0) < 2:
            return np.full(codes.shape[0], np.nan, dtype=np.float64)

        # 2c) Map codes -> midpoints; treat reserved last bin as missing
        edges = np.asarray(edges, dtype=np.float64)
        mids = 0.5 * (edges[:-1] + edges[1:])
        miss_id = int(missing_bin_id if missing_bin_id is not None else len(mids))

        codes_i = np.asarray(codes, dtype=np.int64, copy=False)
        out = np.empty_like(codes_i, dtype=np.float64)
        nonmiss = codes_i != miss_id
        # Clamp just in case
        if mids.size > 0:
            idx = np.clip(codes_i[nonmiss], 0, mids.size - 1)
            out[nonmiss] = mids[idx]
        else:
            out[nonmiss] = np.nan
        out[~nonmiss] = np.nan
        return out
