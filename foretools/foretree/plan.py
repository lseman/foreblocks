from __future__ import annotations

# =====================================================================
# 5. Main Ultra-Fast Generator Class (Compatible Names)
# =====================================================================
# Add missing import at the top
# Add missing import at the top
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from boosting_aux import *

# =====================================================================
# Optimized Conservative Version (Best of Both Worlds)
# =====================================================================
from numba import njit, prange


class SplitPlanView:
    """
    Read-only adapter that normalizes different SplitPlan/Candidate shapes to a common API.

    Expected normalized fields (properties):
      - kind: "axis" | "kway" | "oblique" | "oblique_interaction"
      - gain: float
      - gfi: int (axis/kway)
      - bin_idx: Optional[int]  (axis)
      - threshold: float        (axis/oblique)
      - missing_left: bool
      - left_groups: Iterable[int] (kway)
      - features: List[int]       (oblique)
      - weights: np.ndarray       (oblique)
      - bias: float               (oblique, optional)
    """

    def __init__(self, plan):
        self._p = plan

    # ---- basics ----
    @property
    def kind(self):
        return getattr(self._p, "kind", getattr(self._p, "type", "axis"))

    @property
    def gain(self):
        return float(getattr(self._p, "gain", getattr(self._p, "best_gain", 0.0)))

    @property
    def bin_idx(self):
        # aliases: bin_idx, split_bin, best_bin
        for k in ("bin_idx", "split_bin", "best_bin"):
            if hasattr(self._p, k):
                v = getattr(self._p, k)
                return None if v is None else int(v)
        return None

    @property
    def gfi(self):
        for k in (
            "gfi",
            "feature",
            "feature_id",
            "global_feat",
            "global_feature",
            "cat_gfi",
        ):
            if hasattr(self._p, k):
                return int(getattr(self._p, k))
        return -1

    @property
    def threshold(self):
        for k in ("threshold", "thr", "t", "oblique_thr"):
            if hasattr(self._p, k):
                v = getattr(self._p, k)
                if v is None:
                    return np.nan
                try:
                    return float(v)
                except Exception:
                    return np.nan
        return np.nan

    @property
    def missing_left(self):
        for k in (
            "missing_left",
            "go_left_if_missing",
            "miss_left",
            "oblique_missing_left",
        ):
            if hasattr(self._p, k):
                return bool(getattr(self._p, k))
        return True

    @property
    def left_groups(self):
        # support both "left_groups" and "groups"/"missing_group" (binary emulation)
        if hasattr(self._p, "left_groups"):
            try:
                return [int(x) for x in getattr(self._p, "left_groups")]
            except Exception:
                return []
        if hasattr(self._p, "groups"):
            groups = getattr(self._p, "groups")
            if groups:
                try:
                    return [int(x) for x in groups[0]]
                except Exception:
                    pass
        return []

    @property
    def features(self):
        # Prefer explicit "features" (global ids). Do NOT map w_lfi here (no tree access).
        for k in ("features", "feature_ids", "feat_ids", "indices"):
            if hasattr(self._p, k):
                v = getattr(self._p, k)
                try:
                    return [int(x) for x in v]
                except Exception:
                    pass
        return []

    @property
    def weights(self):
        for k in ("weights", "coeffs", "coef", "w", "w_vals"):
            if hasattr(self._p, k):
                return np.asarray(getattr(self._p, k), dtype=np.float64)
        return np.zeros((0,), dtype=np.float64)

    @property
    def bias(self):
        for k in ("bias", "intercept", "b0"):
            if hasattr(self._p, k):
                return float(getattr(self._p, k))
        return 0.0


@dataclass
class SplitPlan:
    kind: str  # "axis" | "kway" | "oblique"
    gain: float

    # axis
    gfi: Optional[int] = None
    threshold: Optional[float] = None
    missing_left: Optional[bool] = None
    bin_idx: Optional[int] = None

    # kway (binary emulation)
    left_groups: Optional[List[int]] = None  # finite bin ids to go left
    # re-use gfi + missing_left for kway

    # oblique
    features: Optional[List[int]] = None  # GLOBAL feature ids
    weights: Optional[np.ndarray] = None
    bias: Optional[float] = 0.0
    threshold: Optional[float] = 0.0  # same field name for uniform access

    def apply(self, tree: "UnifiedTree", node: "TreeNode", X: np.ndarray):
        """
        Materialize children for this node. Handles:
        - axis (binary, reuses your MissingHandler)
        - categorical_kway (multi-child, uses BinRegistry codes)
        - oblique (binary on projection)
        """
        if self.kind == "axis":
            # delegate to existing missing handler split
            lf = int(tree.feature_map[self.gfi])
            res = tree.missing_handler.compute_split_with_missing(
                X,
                node.gradients,
                node.hessians,
                node.data_indices,
                lf,
                float(self.threshold),
            )
            li, ri = res.left_indices, res.right_indices
            if li.size < tree.min_samples_leaf or ri.size < tree.min_samples_leaf:
                return None
            ln, rn = tree._make_binary_children(node, li, ri)
            node.best_feature = int(self.gfi)
            node.best_threshold = float(self.threshold)
            node.missing_go_left = bool(self.missing_left)
            node.best_bin_idx = int(self.bin_idx) if self.bin_idx is not None else None
            node.node_kind = "axis"
            return (ln, rn)

        if self.kind == "categorical_kway":
            # Prebin just these rows with the active binned mode
            mode = getattr(tree, "binned_mode", "hist")
            X_sub = tree._X_train_cols[node.data_indices]
            codes, missing_bin_id = tree.bins.prebin_matrix(
                X_sub, mode=mode, node_id=None, cache_key=f"{mode}:rows:{X_sub.shape}"
            )
            lf = int(tree.feature_map[int(self.cat_gfi)])
            col = codes[:, lf].astype(np.int32, copy=False)

            # route rows into groups (missing -> missing_group)
            child_indices: List[np.ndarray] = []
            for gi, bins in enumerate(self.groups):
                if gi == self.missing_group:
                    # this group also collects missing
                    mask = np.isin(col, bins, assume_unique=False) | (
                        col == missing_bin_id
                    )
                else:
                    mask = np.isin(col, bins, assume_unique=False)
                child_indices.append(node.data_indices[mask])

            # sanity: keep only non-empty children
            kept = [
                (i, idxs)
                for i, idxs in enumerate(child_indices)
                if idxs.size >= tree.min_samples_leaf
            ]
            if len(kept) < 2:
                return None

            children_nodes = []
            for _, idxs in kept:
                ch = tree._make_child(node, idxs)
                children_nodes.append(ch)

            # record on node
            node.node_kind = "categorical_kway"
            node.best_children = [c.node_id for c in children_nodes]
            node.best_split_payload = {
                "gfi": int(self.cat_gfi),
                "groups": [np.asarray(g, dtype=np.int32) for g in self.groups],
                "missing_group": int(self.missing_group),
            }
            return tuple(children_nodes)

        if self.kind == "oblique":
            # project z = w^T x  (NaN in any used feature => missing)
            X_local = tree._X_train_cols[node.data_indices]
            w_idx = np.asarray(self.w_lfi, dtype=np.int32)
            w_val = np.asarray(self.w_vals, dtype=np.float64)
            colset = X_local[:, w_idx].astype(np.float64, copy=False)
            # missing mask: any non-finite across used features
            miss_mask = ~np.isfinite(colset).all(axis=1)

            z = (colset * w_val.reshape(1, -1)).sum(axis=1)
            # apply split with missing policy
            go_left = np.empty(z.shape[0], dtype=bool)
            go_left[:] = self.oblique_missing_left
            finite = ~miss_mask
            go_left[finite] = z[finite] <= float(self.oblique_thr)

            left_idx = node.data_indices[go_left]
            right_idx = node.data_indices[~go_left]
            if (
                left_idx.size < tree.min_samples_leaf
                or right_idx.size < tree.min_samples_leaf
            ):
                return None

            ln, rn = tree._make_binary_children(node, left_idx, right_idx)
            node.node_kind = "oblique"
            node.best_children = (ln.node_id, rn.node_id)
            node.best_split_payload = {
                "w_lfi": w_idx,
                "w_vals": w_val,
                "thr": float(self.oblique_thr),
                "missing_left": bool(self.oblique_missing_left),
            }
            return (ln, rn)

        raise ValueError(f"Unknown SplitPlan.kind={self.kind}")


@dataclass
class Candidate:
    kind: str
    gain: float
    payload: dict


# --- splits.py (or wherever CandidateGenerator lives) ---


class CandidateGenerator:
    def generate(self, tree, X, node, hist_tuple, **kwargs) -> List[Candidate]:
        raise NotImplementedError


class AxisGenerator(CandidateGenerator):
    def __init__(self):
        self._edges_cache = {}  # (gfi, mode) -> np.ndarray

    def generate(self, tree, X, node, hist_tuple, **kwargs) -> List[Candidate]:
        hg, hh, n_bins_total = hist_tuple
        finite_bins = n_bins_total - 1
        gmiss = hg[:, finite_bins]
        hmiss = hh[:, finite_bins]
        gfin = hg[:, :finite_bins]
        hfin = hh[:, :finite_bins]

        lfi, bin_idx, gain, missing_left = find_best_split_with_missing(
            gfin,
            hfin,
            gmiss,
            hmiss,
            tree.lambda_,
            tree.gamma,
            finite_bins,
            tree.min_child_weight,
            tree._mono_local,
        )
        if lfi == -1 or gain <= 0.0:
            return []

        gfi_global = int(tree.feature_indices[lfi])

        # NEW: Get threshold from histogram system instead of old registry
        if hasattr(tree, "_strategy") and hasattr(tree._strategy, "histogram_system"):
            # Get feature info from histogram system
            feature_info = tree._strategy.histogram_system.get_feature_info(lfi)
            edges = feature_info.get("edges")
            if edges is not None and bin_idx < len(edges) - 1:
                thr = float(edges[int(bin_idx)])
            else:
                # Fallback: use bin midpoint
                thr = float(bin_idx)
        else:
            # Legacy fallback
            mode = getattr(tree, "binned_mode", "hist")
            key = (gfi_global, mode)
            edges = self._edges_cache.get(key)
            if edges is None:
                # Try to get from old registry if available
                if hasattr(tree, "bins") and tree.bins is not None:
                    edges = tree.bins.get_edges(gfi_global, mode=mode, node_id=None)
                    self._edges_cache[key] = edges

            if edges is not None and bin_idx < len(edges) - 1:
                thr = float(edges[int(bin_idx)])
            else:
                thr = float(bin_idx)

        return [
            Candidate(
                "axis",
                float(gain),
                dict(
                    gfi=gfi_global,
                    threshold=thr,
                    missing_left=bool(missing_left),
                    bin_idx=int(bin_idx),
                ),
            )
        ]


class CategoricalKWayGenerator(CandidateGenerator):
    def __init__(
        self,
        feature_types: Optional[Dict[int, str]] = None,
        max_groups: int = 8,
        **kwargs,
    ):
        self.feature_types = feature_types or {}
        self.max_groups = int(max_groups)

    def generate(self, tree, X, node, hist_tuple, **kwargs) -> List[Candidate]:
        hg, hh, n_bins_total = hist_tuple
        finite_bins = n_bins_total - 1
        out: List[Candidate] = []
        lam = float(tree.lambda_)

        for lpos, gfi in enumerate(tree.feature_indices):
            gfi = int(gfi)

            # Detect categorical features - updated for histogram system
            is_cat = False

            # First try: histogram system categorical detection
            if hasattr(tree, "_strategy") and hasattr(
                tree._strategy, "histogram_system"
            ):
                try:
                    feature_info = tree._strategy.histogram_system.get_feature_info(
                        lpos
                    )
                    strategy = feature_info.get("strategy", "simple")
                    n_bins = feature_info.get("n_bins", 0)

                    # Heuristic: categorical if explicitly marked or few bins
                    is_cat = (
                        "categorical" in strategy
                        or strategy == "simple"
                        and n_bins <= 32
                        or n_bins <= 10  # Very few bins suggests categorical
                    )
                except Exception:
                    pass

            # Second try: old BinRegistry feature types
            if (
                not is_cat
                and hasattr(tree, "bins")
                and hasattr(tree.bins, "feature_types")
            ):
                is_cat = tree.bins.feature_types.get(gfi, "num") == "cat"

            # Third try: explicit feature types passed to generator
            if not is_cat:
                is_cat = self.feature_types.get(gfi, "num") == "cat"

            # Fourth try: heuristic based on histogram shape
            if not is_cat:
                # If we have very few non-zero bins relative to total, likely categorical
                G = hg[lpos, :finite_bins]
                non_zero_bins = np.count_nonzero(G)
                if finite_bins > 0 and non_zero_bins <= min(32, finite_bins // 2):
                    is_cat = True

            if not is_cat:
                continue

            G = hg[lpos, :finite_bins]
            H = hh[lpos, :finite_bins]
            K = G.shape[0]
            if K < 2:
                continue

            # Filter out empty bins to avoid division by zero
            non_empty_mask = (G != 0) | (H > 0)
            if not np.any(non_empty_mask):
                continue

            G_filtered = G[non_empty_mask]
            H_filtered = H[non_empty_mask]
            bin_indices = np.where(non_empty_mask)[0]
            K_filtered = G_filtered.shape[0]

            if K_filtered < 2:
                continue

            # Rank categories by gradient signal strength
            score = np.abs(G_filtered) / (H_filtered + lam + 1e-12)
            order_filtered = np.argsort(-score)

            # Map back to original bin indices
            order = bin_indices[order_filtered]

            top = min(self.max_groups - 1, K_filtered)
            if top > 0 and top < K_filtered:
                # Use partition for efficiency
                base_filtered = np.argpartition(-score, top - 1)[:top]
                order_filtered = base_filtered[np.argsort(-score[base_filtered])]
                order = bin_indices[order_filtered]
            else:
                order = bin_indices[np.argsort(-score)[:top]]

            # Form groups (top singletons + tail)
            groups: List[np.ndarray] = []
            if top <= 1:
                g0 = np.array([order[0]], dtype=np.int32)
                # Only include non-empty bins in the complement
                remaining_bins = np.setdiff1d(bin_indices, g0, assume_unique=True)
                if remaining_bins.size > 0:
                    groups = [g0, remaining_bins]
                else:
                    continue  # Can't form valid split
            else:
                # Create singleton groups for top categories
                for i in range(min(top - 1, len(order))):
                    groups.append(np.array([order[i]], dtype=np.int32))

                # Tail group with remaining categories
                used_bins = (
                    np.concatenate(groups) if groups else np.array([], dtype=np.int32)
                )
                tail = np.setdiff1d(bin_indices, used_bins, assume_unique=True)
                if tail.size > 0:
                    groups.append(tail)

                # Optionally add last strong category as separate group
                if len(groups) < self.max_groups and top < len(order):
                    last = np.array([order[top - 1]], dtype=np.int32)
                    groups = [last] + groups

            if len(groups) < 2:
                continue

            # Choose LEFT group (binary emulation) - use first group
            left_groups = groups[0]

            # Missing group heuristic: group with maximum hessian (most stable)
            H_group = np.array([H[g].sum() for g in groups], dtype=np.float64)
            missing_group = int(np.argmax(H_group))
            missing_left = missing_group == 0

            # Compute binary gain: LEFT=left_groups (+missing if missing_left) vs REST
            G_tot = float(G.sum() + hg[lpos, finite_bins])  # Include missing bin
            H_tot = float(H.sum() + hh[lpos, finite_bins])  # Include missing bin

            if H_tot <= lam:  # Not enough total weight
                continue

            parent_score = (G_tot * G_tot) / (H_tot + lam)

            # Left child: selected bins + missing if missing_left
            GL = float(G[left_groups].sum())
            HL = float(H[left_groups].sum())
            if missing_left:
                GL += float(hg[lpos, finite_bins])
                HL += float(hh[lpos, finite_bins])

            # Right child: remaining bins + missing if not missing_left
            GR = float(G.sum() - G[left_groups].sum())
            HR = float(H.sum() - H[left_groups].sum())
            if not missing_left:
                GR += float(hg[lpos, finite_bins])
                HR += float(hh[lpos, finite_bins])

            # Check minimum child weight constraint
            if HL < tree.min_child_weight or HR < tree.min_child_weight:
                continue

            # Compute gain using standard gradient boosting formula
            child_score = (GL * GL) / (HL + lam) + (GR * GR) / (HR + lam)
            gain = 0.5 * (child_score - parent_score) - float(tree.gamma)

            if gain <= 0.0:
                continue

            out.append(
                Candidate(
                    "kway",
                    float(gain),
                    dict(
                        gfi=gfi,
                        left_groups=[int(x) for x in left_groups],
                        missing_left=bool(missing_left),
                    ),
                )
            )

        return out


@njit(fastmath=True, cache=True, nogil=True)
def _build_normal_eq(XS, g, h, ridge_plus_lam):
    n, k = XS.shape
    A = np.zeros((k, k), dtype=np.float64)
    bvec = np.zeros(k, dtype=np.float64)

    # b = -X^T g
    for i in range(n):
        gi = g[i]
        xi = XS[i]
        for j in range(k):
            bvec[j] -= xi[j] * gi

    # A = X^T diag(h) X  (only upper triangle)
    for i in range(n):
        hi = h[i]
        xi = XS[i]
        for r in range(k):
            hrx = hi * xi[r]
            # compute c >= r only
            for c in range(r, k):
                A[r, c] += hrx * xi[c]

    # symmetrize + ridge
    for r in range(k):
        for c in range(r + 1, k):
            A[c, r] = A[r, c]
        A[r, r] += ridge_plus_lam

    return A, bvec


@njit(fastmath=True, cache=True, nogil=True)
def _best_split_on_projection(z, g, h, miss_mask, lam, gamma, min_child_weight):
    n = z.size

    # Count & fast exit
    n_missing = 0
    for i in range(n):
        if miss_mask[i]:
            n_missing += 1
    n_finite = n - n_missing
    if n_finite < 4:
        return -1.0, 0.0, True

    # Build finite-only compact views (no sort yet)
    zf = np.empty(n_finite, dtype=np.float64)
    gf = np.empty(n_finite, dtype=np.float64)
    hf = np.empty(n_finite, dtype=np.float64)

    p = 0
    gm = 0.0
    hm = 0.0
    for i in range(n):
        if miss_mask[i]:
            gm += g[i]
            hm += h[i]
        else:
            zf[p] = z[i]
            gf[p] = g[i]
            hf[p] = h[i]
            p += 1

    # Sort only the finite slice
    order = np.argsort(zf)
    # Reorder in-place (permute into tmp, then copy back)
    zt = zf.copy()
    gt = gf.copy()
    ht = hf.copy()
    for i in range(n_finite):
        j = order[i]
        zf[i] = zt[j]
        gf[i] = gt[j]
        hf[i] = ht[j]

    # Prefix sums
    for i in range(1, n_finite):
        gf[i] += gf[i - 1]
        hf[i] += hf[i - 1]

    g_total = gf[n_finite - 1] + gm
    h_total = hf[n_finite - 1] + hm
    parent_score = (g_total * g_total) / (h_total + lam)

    best_gain = -1.0
    best_thr = 0.0
    best_mleft = True

    i = 1
    while i < n_finite:
        # skip ties
        if zf[i] <= zf[i - 1] + 1e-15:
            i += 1
            continue

        gl_base = gf[i - 1]
        hl_base = hf[i - 1]
        gr_base = gf[n_finite - 1] - gl_base
        hr_base = hf[n_finite - 1] - hl_base
        thr = 0.5 * (zf[i - 1] + zf[i])

        # missing left
        hl_left = hl_base + hm
        hr_left = hr_base
        if hl_left >= min_child_weight and hr_left >= min_child_weight:
            gl_left = gl_base + gm
            gr_left = gr_base
            child = (gl_left * gl_left) / (hl_left + lam) + (gr_left * gr_left) / (
                hr_left + lam
            )
            gain = 0.5 * (child - parent_score) - gamma
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
                best_mleft = True

        # missing right
        hl_right = hl_base
        hr_right = hr_base + hm
        if hl_right >= min_child_weight and hr_right >= min_child_weight:
            gl_right = gl_base
            gr_right = gr_base + gm
            child = (gl_right * gl_right) / (hl_right + lam) + (gr_right * gr_right) / (
                hr_right + lam
            )
            gain = 0.5 * (child - parent_score) - gamma
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
                best_mleft = False

        i += 1

    return best_gain, best_thr, best_mleft


@njit(fastmath=True, cache=True, inline="always")
def solve_2x2(A, b):
    """Direct solution for 2x2 system"""
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    inv_det = 1.0 / det
    return np.array(
        [
            (A[1, 1] * b[0] - A[0, 1] * b[1]) * inv_det,
            (A[0, 0] * b[1] - A[1, 0] * b[0]) * inv_det,
        ]
    )


@njit(fastmath=True, cache=True, inline="always")
def solve_3x3(A, b):
    """Direct solution for 3x3 system using Cramer's rule"""
    # Calculate determinant
    det = (
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )

    inv_det = 1.0 / det

    # Cramer's rule
    x0 = (
        b[0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - b[1] * (A[0, 1] * A[2, 2] - A[0, 2] * A[2, 1])
        + b[2] * (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1])
    ) * inv_det

    x1 = (
        A[0, 0] * (b[1] * A[2, 2] - b[2] * A[1, 2])
        - A[0, 1] * (b[0] * A[2, 2] - b[2] * A[2, 0])
        + A[0, 2] * (b[0] * A[1, 2] - b[1] * A[2, 0])
    ) * inv_det

    x2 = (
        A[0, 0] * (A[1, 1] * b[2] - A[1, 2] * b[1])
        - A[0, 1] * (A[1, 0] * b[2] - A[1, 2] * b[0])
        + A[0, 2] * (A[1, 0] * b[1] - A[1, 1] * b[0])
    ) * inv_det

    return np.array([x0, x1, x2])


@njit(fastmath=True, cache=True)
def cholesky_solve(A, b):
    """Cholesky decomposition solver for symmetric positive definite matrices"""
    n = A.shape[0]
    L = np.zeros_like(A)

    # Cholesky decomposition: A = L * L.T
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Diagonal elements
                sum_sq = 0.0
                for k in range(j):
                    sum_sq += L[j, k] * L[j, k]
                L[i, j] = math.sqrt(A[i, i] - sum_sq)
            else:
                sum_prod = 0.0
                for k in range(j):
                    sum_prod += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]

    # Forward substitution: L * y = b
    y = np.zeros_like(b)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = (b[i] - sum_val) / L[i, i]

    # Backward substitution: L.T * x = y
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += L[j, i] * x[j]
        x[i] = (y[i] - sum_val) / L[i, i]

    return x


@njit(fastmath=True, cache=True)
def is_symmetric(A, tol=1e-12):
    """Check if matrix is symmetric"""
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j] - A[j, i]) > tol:
                return False
    return True


@njit(fastmath=True, cache=True)
def is_positive_definite_check(A):
    """Quick check for positive definiteness (all diagonal elements > 0)"""
    n = A.shape[0]
    for i in range(n):
        if A[i, i] <= 0:
            return False
    return True


@njit(fastmath=True, cache=True)
def solve_sym_optimized(A, b):
    """
    Optimized solver that chooses the best method based on matrix size and properties.
    For small k (k<=16), automatically selects fastest approach.
    """
    n = A.shape[0]

    # Ensure contiguous memory layout
    if not A.flags.c_contiguous:
        A = np.ascontiguousarray(A)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b)

    # Direct methods for very small matrices
    if n == 1:
        return np.array([b[0] / A[0, 0]])
    elif n == 2:
        return solve_2x2(A, b)
    elif n == 3:
        return solve_3x3(A, b)

    # For larger matrices, check if we can use Cholesky
    if n <= 16 and is_symmetric(A) and is_positive_definite_check(A):
        try:
            return cholesky_solve(A, b)
        except:
            pass  # Fall back to general solver

    # Default to numpy's optimized solver
    return np.linalg.solve(A, b)


@njit(fastmath=True, cache=True, nogil=True)
def _build_Ab_2col(x1, x2, g, h, ridge_plus_lam):
    A00 = 0.0
    A01 = 0.0
    A11 = 0.0
    b0 = 0.0
    b1 = 0.0
    n = x1.size
    for i in range(n):
        xi = x1[i]
        yi = x2[i]
        hi = h[i]
        gi = g[i]
        if np.isfinite(xi):
            b0 -= xi * gi
            A00 += hi * xi * xi
        if np.isfinite(yi):
            b1 -= yi * gi
            A11 += hi * yi * yi
        if np.isfinite(xi) and np.isfinite(yi):
            A01 += hi * xi * yi
    A = np.empty((2, 2), np.float64)
    A[0, 0] = A00 + ridge_plus_lam
    A[0, 1] = A01
    A[1, 0] = A01
    A[1, 1] = A11 + ridge_plus_lam
    return A, np.array([b0, b1], np.float64)


@njit(fastmath=True, cache=True, nogil=True, inline="always")
def _proj_2col(x1, x2, w0, w1):
    n = x1.size
    z = np.empty(n, np.float64)
    for i in range(n):
        z[i] = x1[i] * w0 + x2[i] * w1
    return z


@njit(fastmath=True, cache=True, nogil=True)
def _missmask_2col(x1, x2):
    n = x1.size
    m = np.empty(n, np.bool_)
    for i in range(n):
        m[i] = not (np.isfinite(x1[i]) and np.isfinite(x2[i]))
    return m


@njit(fastmath=True, cache=True, nogil=True)
def _process_inter_pair_fast(
    x1, x2, g, h, ridge_plus_lam, lam, gamma, min_child_weight
):
    A, b = _build_Ab_2col(x1, x2, g, h, ridge_plus_lam)
    w = solve_2x2(A, b)
    if not np.isfinite(w[0]) or not np.isfinite(w[1]):
        return -1.0, 0.0, True, np.zeros(2)
    z = _proj_2col(x1, x2, w[0], w[1])
    miss_mask = _missmask_2col(x1, x2)
    best_gain, best_thr, best_mleft = _best_split_on_projection(
        z, g, h, miss_mask, lam, gamma, min_child_weight
    )
    return best_gain, best_thr, best_mleft, w

class ObliqueGenerator(CandidateGenerator):
    """
    Small oblique candidate from top-k features (by |corr(x, g)|) and ridge WLS:
        (X^T H X + λI) w = - X^T g
    Early-exit if axis is already very strong, downsample big nodes, cache XS.
    """

    def __init__(
        self,
        k_features: int = 6,
        ridge: float = 1e-3,
        max_candidates: int = 8,
        axis_guard_factor: float = 1.02,  # skip if axis_gain_max * 1.02 >= oblique best
        max_rows_for_oblique: int = 16000,  # downsample target
        weighted_subsample: bool = True,  # use |g|+h weights
        xs_cache_capacity: int = 16,  # per-node XS LRU
    ):
        self.k = int(k_features)
        self.ridge = float(ridge)
        self.m = int(max_candidates)
        self.axis_guard_factor = float(axis_guard_factor)
        self.max_rows_for_oblique = int(max_rows_for_oblique)
        self.weighted_subsample = bool(weighted_subsample)
        self.xs_cache_capacity = int(xs_cache_capacity)

    def _get_node_xs_cache(self, node):
        # attach a tiny LRU dict to the node
        cache = getattr(node, "_xs_cache", None)
        if cache is None:
            cache = OrderedDict()
            node._xs_cache = cache
        return cache

    def _cache_get_XS(self, node, X_loc, S_tuple):
        """
        Cheap XS cache per node keyed by tuple(local_feature_indices).
        Stores a *view* slice (no copy), so rows must not be permuted.
        """
        cache = self._get_node_xs_cache(node)
        key = ("XS", S_tuple)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return hit
        XS = X_loc[:, np.fromiter(S_tuple, dtype=np.int32)]
        cache[key] = XS
        if len(cache) > self.xs_cache_capacity:
            cache.popitem(last=False)
        return XS

    def _maybe_downsample_pos(
        self, rng, n: int, g: np.ndarray, h: np.ndarray
    ) -> Optional[np.ndarray]:
        if n <= self.max_rows_for_oblique:
            return None
        m = self.max_rows_for_oblique
        if not self.weighted_subsample:
            return rng.choice(n, size=m, replace=False)
        w = np.abs(g) + h + 1e-12
        w /= w.sum()
        # IMPORTANT: choose *row positions* in [0..n)
        return rng.choice(n, size=m, replace=False, p=w)

    def generate(self, tree, X, node, hist_tuple, **kwargs) -> List[Candidate]:
        axis_gain_max = float(kwargs.get("axis_gain_max", -np.inf))
        
        # Early exit guard for very strong axis splits
        if (np.isfinite(axis_gain_max) and 
            axis_gain_max > 0 and 
            self.axis_guard_factor >= 1.0):
            # We'll still compute candidates but may exit early
            pass

        out: List[Candidate] = []
        ridge_plus_lam = float(self.ridge + tree.lambda_)

        # FIXED: Remove redundant variable assignments
        idx = node.data_indices
        g_full = node.gradients.astype(np.float64, copy=False)
        h_full = node.hessians.astype(np.float64, copy=False)
        X_loc_full = tree._X_train_cols[idx].astype(np.float64, copy=False)

        # Downsample large nodes (stable seed per node id)
        rng = np.random.default_rng(seed=0xC0FFEE ^ int(node.node_id))
        pos = self._maybe_downsample_pos(rng, idx.size, g_full, h_full)
        if pos is None:
            X_loc = X_loc_full
            g_sub = g_full
            h_sub = h_full
        else:
            X_loc = X_loc_full[pos]
            g_sub = g_full[pos]
            h_sub = h_full[pos]

        # Feature ranking by |corr(x, g)| (vectorized)
        corrs = _abs_corr_cols_ignore_nan_optimized(X_loc, g_sub)
        kneed = max(2, self.k)
        
        if corrs.size > kneed:
            # Pick top-k efficiently using partition
            base = np.argpartition(-corrs, kneed - 1)[:kneed]
            # Order those k by score
            base = base[np.argsort(-corrs[base])]
        else:
            base = np.argsort(-corrs)[:kneed]
            
        if base.size < 2:
            return []
            
        S_local = tuple(int(x) for x in base)

        # Get cached feature subset
        XS = self._cache_get_XS(node, X_loc, S_local)

        # Build normal equations and solve
        A, b = _build_normal_eq(np.ascontiguousarray(XS), g_sub, h_sub, ridge_plus_lam)
        try:
            w = solve_sym_optimized(A, b)
        except Exception:
            return out

        # Project to 1D and find best split
        z = XS @ w
        miss_mask = ~np.isfinite(XS).all(axis=1)
        best_gain, best_thr, best_mleft = _best_split_on_projection(
            np.ascontiguousarray(z, dtype=np.float64),
            g_sub,
            h_sub,
            miss_mask,
            float(tree.lambda_),
            float(tree.gamma),
            float(tree.min_child_weight),
        )
        
        if best_gain <= 0.0:
            return out

        # Early exit if axis is already good enough
        if (np.isfinite(axis_gain_max) and 
            axis_gain_max * self.axis_guard_factor >= best_gain):
            return out  # Keep axis split instead

        # Convert local feature indices to global
        features_global = [int(tree.feature_indices[j]) for j in S_local]
        
        out.append(
            Candidate(
                "oblique",
                float(best_gain),
                dict(
                    features=features_global,
                    weights=w.astype(np.float64, copy=False),
                    threshold=float(best_thr),
                    missing_left=bool(best_mleft),
                    bias=0.0,
                ),
            )
        )
        return out

@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def _process_multiple_pairs_conservative(
    X_loc, g, h, pairs_list, ridge_plus_lam, lam, gamma, min_child_weight
):
    n_pairs = len(pairs_list)

    gains = np.full(n_pairs, -1.0, dtype=np.float64)
    thrs = np.zeros(n_pairs, dtype=np.float64)
    mlefts = np.zeros(n_pairs, dtype=np.bool_)
    a_idx = np.zeros(n_pairs, dtype=np.int32)
    b_idx = np.zeros(n_pairs, dtype=np.int32)
    w_out = np.zeros((n_pairs, 2), dtype=np.float64)

    # Parallel, independent across pairs
    for k in prange(n_pairs):
        a_local = pairs_list[k][0]
        b_local = pairs_list[k][1]

        x1 = np.ascontiguousarray(X_loc[:, a_local])
        x2 = np.ascontiguousarray(X_loc[:, b_local])

        A, b = _build_Ab_2col(x1, x2, g, h, ridge_plus_lam)
        w = solve_2x2(A, b)

        if np.isfinite(w[0]) and np.isfinite(w[1]):
            z = _proj_2col(x1, x2, w[0], w[1])
            miss_mask = _missmask_2col(x1, x2)
            gain, thr, mleft = _best_split_on_projection(
                z, g, h, miss_mask, lam, gamma, min_child_weight
            )
            if gain > 0.0:
                gains[k] = gain
                thrs[k] = thr
                mlefts[k] = mleft
                a_idx[k] = a_local
                b_idx[k] = b_local
                w_out[k, 0] = w[0]
                w_out[k, 1] = w[1]

    # Count valid
    valid = 0
    for k in range(n_pairs):
        if gains[k] > 0.0:
            valid += 1

    # Compact to exact-size arrays
    gains_v = np.empty(valid, dtype=np.float64)
    thrs_v = np.empty(valid, dtype=np.float64)
    mlefts_v = np.empty(valid, dtype=np.bool_)
    a_v = np.empty(valid, dtype=np.int32)
    b_v = np.empty(valid, dtype=np.int32)
    w_v = np.empty((valid, 2), dtype=np.float64)

    p = 0
    for k in range(n_pairs):
        if gains[k] > 0.0:
            gains_v[p] = gains[k]
            thrs_v[p] = thrs[k]
            mlefts_v[p] = mlefts[k]
            a_v[p] = a_idx[k]
            b_v[p] = b_idx[k]
            w_v[p, 0] = w_out[k, 0]
            w_v[p, 1] = w_out[k, 1]
            p += 1

    return gains_v, thrs_v, mlefts_v, a_v, b_v, w_v


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def _abs_corr_cols_ignore_nan_optimized(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    n, d = X.shape
    out = np.zeros(d, dtype=np.float64)

    # First pass: accumulate counts and means (per column)
    cnt = np.zeros(d, dtype=np.int64)
    sx = np.zeros(d, dtype=np.float64)
    sg = 0.0
    # accumulate sg once (only depends on g where x finite; using full g mean is fine because same subset used below)
    for i in range(n):
        sg += g[i]
    mg_global = sg / n

    for j in prange(d):
        c = 0
        s = 0.0
        for i in range(n):
            x = X[i, j]
            if np.isfinite(x):
                c += 1
                s += x
        cnt[j] = c
        sx[j] = s

    # Second pass: compute corr components per column
    for j in prange(d):
        c = cnt[j]
        if c < 2:
            out[j] = 0.0
            continue
        mx = sx[j] / c

        sxx = 0.0
        sgg = 0.0
        sxg = 0.0
        for i in range(n):
            x = X[i, j]
            if np.isfinite(x):
                dx = x - mx
                dg = g[i] - mg_global  # stable enough for ranking
                sxx += dx * dx
                sgg += dg * dg
                sxg += dx * dg

        denom = np.sqrt(sxx * sgg) + 1e-12
        out[j] = 0.0 if denom < 1e-12 else abs(sxg / denom)

    return out

class InteractionSeededGenerator:
    """
    Conservative optimization maintaining exact original algorithm.
    Optimizations: memory layout, numba compilation, pre-allocation.
    No algorithmic changes - guaranteed same results as original.
    """

    def __init__(self, pairs: int = 5, max_top_features: int = 8, **kwargs):
        self.pairs = pairs
        self.max_top_features = max_top_features

    def generate(self, tree, X, node, hist_tuple, **kwargs):
        idx = node.data_indices
        g = node.gradients.astype(np.float64, copy=False)
        h = node.hessians.astype(np.float64, copy=False)
        X_loc = tree._X_train_cols[idx].astype(np.float64, copy=False)

        n_features = X_loc.shape[1]
        if n_features < 2:
            return []

        # Exact original feature selection - just faster execution
        var = np.nanvar(X_loc, axis=0)
        topv = np.argsort(-var)[:min(16, n_features)]

        if topv.size < 2:
            return []

        # Original correlation computation - use optimized version
        corrs = _abs_corr_cols_ignore_nan_optimized(X_loc, g)
        scores = corrs[topv]
        ord2 = topv[np.argsort(-scores)]

        if ord2.size < 2:
            return []

        max_features_for_pairs = min(self.max_top_features, ord2.size)
        ridge_plus_lam = 1e-3 + tree.lambda_

        # Generate pairs list once (avoid repeated indexing)
        pairs_list = []
        for i in range(min(4, max_features_for_pairs)):
            for j in range(i + 1, min(max_features_for_pairs, 8)):
                if len(pairs_list) >= self.pairs:
                    break
                pairs_list.append((int(ord2[i]), int(ord2[j])))
            if len(pairs_list) >= self.pairs:
                break

        if not pairs_list:
            return []

        # Process all pairs with optimized numba kernel
        gains_v, thrs_v, mlefts_v, a_v, b_v, w_v = _process_multiple_pairs_conservative(
            X_loc,
            g,
            h,
            pairs_list,
            ridge_plus_lam,
            tree.lambda_,
            tree.gamma,
            tree.min_child_weight,
        )

        # Convert results to candidates
        out = []
        for i in range(gains_v.shape[0]):
            gain = float(gains_v[i])
            thr = float(thrs_v[i])
            mleft = bool(mlefts_v[i])
            a_local = int(a_v[i])
            b_local = int(b_v[i])
            weights = w_v[i].copy()  # (2,) float64

            # Map local feature indices to global
            features_global = [
                int(tree.feature_indices[a_local]),
                int(tree.feature_indices[b_local]),
            ]

            out.append(
                Candidate(
                    "oblique",
                    gain,
                    dict(
                        features=features_global,
                        weights=weights,
                        threshold=thr,
                        missing_left=mleft,
                        bias=0.0,
                    ),
                )
            )

        return out

# Keep the original function available for compatibility
def to_split_plan(c: Candidate) -> SplitPlan:
    """Convert Candidate to SplitPlan - unchanged for compatibility"""
    if c.kind == "axis":
        return SplitPlan(kind="axis", gain=c.gain, **c.payload)
    if c.kind == "kway":
        return SplitPlan(kind="kway", gain=c.gain, **c.payload)
    if c.kind == "oblique":
        return SplitPlan(kind="oblique", gain=c.gain, **c.payload)
    raise ValueError(c.kind)
    raise ValueError(c.kind)
