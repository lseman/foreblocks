from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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

    _used_refinement: bool = False
    sorted_lists: Optional[List[np.ndarray]] = None

    # references set by the builder
    _tree_ref: Optional["UnifiedTree"] = None
    _split_plan: Optional["SplitPlan"] = None
    
    _has_refinements: bool = False  # whether the tree has any refinement enabled
    _codes : Optional[np.ndarray] = None  # cached codes for this node if any
    
    best_split : Optional[dict] = None  # dictionary to hold best split info
    _ver : int = 1  # versioning for future changes

    def __post_init__(self):
        # Auto-init sums to avoid forgetting init_sums()
        self.n_samples = int(self.data_indices.size)
        if self.gradients.size:
            self.g_sum = float(self.gradients.sum())
        if self.hessians.size:
            self.h_sum = float(self.hessians.sum())

    def init_sums(self):
        # kept for backward-compat callers
        self.__post_init__()

    def get_feature_values(self, lf: int) -> np.ndarray:
        """
        Return raw (float) values for this node at local feature index lf.

        Priority:
          1) Fast path: raw training matrix `tree._X_train_cols`.
          2) Reconstruct from bin codes + bin edges (honors adaptive overrides when possible).

        Returns float64 array of length len(self.data_indices), with NaN for missing.
        """
        tree = self._tree_ref
        if tree is None:
            raise RuntimeError("Node has no tree reference.")

        # --- 1) Fast path: raw matrix present ---
        X_raw = getattr(tree, "_X_train_cols", None)
        if X_raw is not None:
            # NOTE: assumes X_raw is aligned to LOCAL features (lf)
            vals = X_raw[self.data_indices, lf]
            return vals.astype(np.float64, copy=False)

        # --- 2) Reconstruct from codes + edges ---
        if not hasattr(tree, "bins"):
            raise RuntimeError("BinRegistry not available on tree.")

        mode = getattr(tree, "binned_mode", "hist")  # "hist" | "approx" | "adaptive"

        # local->global feature id mapping used by the registry
        if hasattr(tree, "feature_indices") and lf < len(tree.feature_indices):
            gfi = int(tree.feature_indices[lf])
        else:
            gfi = int(lf)

        # Check if this node has adaptive overrides for this feature
        use_node_override = False
        if mode == "adaptive":
            bins = tree.bins
            try:
                use_node_override = bins.has_node_override("adaptive", self.node_id, gfi)
            except AttributeError:
                use_node_override = bool(getattr(self, "_used_refinement", False))
        else:
            bins = tree.bins

        # Get a codes view (full matrix) if available
        codes_view = bins.get_codes_view(mode=mode)

        # Layout & missing id helpers
        layout = bins.get_layout(mode=mode) if hasattr(bins, "get_layout") else None
        default_missing_id = layout.actual_max_bins if layout is not None else None

        def _resolve_edges() -> Optional[np.ndarray]:
            # Respect node override when `adaptive` and available
            try:
                if mode == "adaptive" and use_node_override:
                    return bins.get_edges(gfi, mode=mode, node_id=self.node_id)
                return bins.get_edges(gfi, mode=mode, node_id=None)
            except KeyError:
                return None

        # --- codes for these rows at (lf) ---
        # Prefer global codes_view -> cheap gather for our rows.
        # If absent, prebin just these rows (no override), which is still acceptable as a fallback.
        if codes_view is not None:
            codes_col = codes_view[self.data_indices, lf]
            missing_bin_id = int(getattr(tree, "_missing_bin_id", default_missing_id))
        else:
            # Last resort: try to compute codes from raw if tree exposes it; otherwise bail
            X_train = getattr(tree, "_X_train_cols", None)
            if X_train is None:
                raise RuntimeError("No raw matrix or cached codes available.")
            X_sub = X_train[self.data_indices]
            codes_full, missing_bin_id = bins.prebin_matrix(
                X_sub, mode=mode, node_id=None, cache_key=f"{mode}:nodevals:{X_sub.shape}"
            )
            codes_col = codes_full[:, lf]
            missing_bin_id = int(
                getattr(tree, "_missing_bin_id", missing_bin_id if missing_bin_id is not None else default_missing_id)
            )

        # --- edges for this feature ---
        edges = _resolve_edges()
        if edges is None:
            # fallback: tree.bin_edges (legacy path)
            edges_list = getattr(tree, "bin_edges", None)
            if edges_list is not None and lf < len(edges_list):
                edges = edges_list[lf]

        # If still missing or degenerate, return NaNs
        if edges is None:
            return np.full(codes_col.shape[0], np.nan, dtype=np.float64)

        edges = np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            return np.full(codes_col.shape[0], np.nan, dtype=np.float64)

        # Map code -> midpoint, reserve last bin as missing
        mids = 0.5 * (edges[:-1] + edges[1:])
        miss_id = missing_bin_id if missing_bin_id is not None else len(mids)

        codes_i = np.asarray(codes_col, dtype=np.int64, copy=False)
        out = np.empty(codes_i.shape[0], dtype=np.float64)

        nonmiss = codes_i != miss_id
        if mids.size > 0:
            idx = np.clip(codes_i[nonmiss], 0, mids.size - 1)
            out[nonmiss] = mids[idx]
        else:
            out[nonmiss] = np.nan
        out[~nonmiss] = np.nan
        return out
