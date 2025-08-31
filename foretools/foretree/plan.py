from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from boosting_aux import *


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
        for k in ("gfi", "feature", "feature_id", "global_feat", "global_feature", "cat_gfi"):
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
        for k in ("missing_left", "go_left_if_missing", "miss_left", "oblique_missing_left"):
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
    kind: str                # "axis" | "kway" | "oblique"
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
    features: Optional[List[int]] = None     # GLOBAL feature ids
    weights: Optional[np.ndarray] = None
    bias: Optional[float] = 0.0
    threshold: Optional[float] = 0.0         # same field name for uniform access

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
                X, node.gradients, node.hessians, node.data_indices, lf, float(self.threshold)
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
                    mask = np.isin(col, bins, assume_unique=False) | (col == missing_bin_id)
                else:
                    mask = np.isin(col, bins, assume_unique=False)
                child_indices.append(node.data_indices[mask])

            # sanity: keep only non-empty children
            kept = [(i, idxs) for i, idxs in enumerate(child_indices) if idxs.size >= tree.min_samples_leaf]
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
            if left_idx.size < tree.min_samples_leaf or right_idx.size < tree.min_samples_leaf:
                return None

            ln, rn = tree._make_binary_children(node, left_idx, right_idx)
            node.node_kind = "oblique"
            node.best_children = (ln.node_id, rn.node_id)
            node.best_split_payload = {
                "w_lfi": w_idx, "w_vals": w_val,
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


class CandidateGenerator:
    def generate(self, tree, X, node, hist_tuple) -> List[Candidate]:
        raise NotImplementedError


class AxisGenerator(CandidateGenerator):
    """Reuse your existing histogram search + missing routing."""
    def generate(self, tree, X, node, hist_tuple) -> List[Candidate]:
        hg, hh, n_bins_total = hist_tuple
        finite_bins = n_bins_total - 1
        gmiss = hg[:, finite_bins]
        hmiss = hh[:, finite_bins]
        gfin = hg[:, :finite_bins]
        hfin = hh[:, :finite_bins]

        lfi, bin_idx, gain, missing_left = find_best_split_with_missing(
            gfin, hfin, gmiss, hmiss,
            tree.lambda_, tree.gamma, finite_bins,
            tree.min_child_weight, tree._mono_local
        )
        if lfi == -1 or gain <= 0.0:
            return []
        gfi_global = int(tree.feature_indices[lfi])
        # derive threshold from registry/hist edges
        edges = tree.bins.get_edges(gfi_global, mode=getattr(tree, "binned_mode", "hist"), node_id=None)
        thr = float(edges[int(bin_idx)])
        return [Candidate("axis", float(gain), dict(
            gfi=gfi_global, threshold=thr, missing_left=bool(missing_left), bin_idx=int(bin_idx)
        ))]

class CategoricalKWayGenerator(CandidateGenerator):
    def __init__(self, feature_types: Optional[Dict[int, str]] = None, max_groups: int = 8):
        self.feature_types = feature_types or {}
        self.max_groups = int(max_groups)

    def generate(self, tree, X, node, hist_tuple) -> List[Candidate]:
        hg, hh, n_bins_total = hist_tuple
        finite_bins = n_bins_total - 1
        out: List[Candidate] = []
        lam = float(tree.lambda_)

        for lpos, gfi in enumerate(tree.feature_indices):
            gfi = int(gfi)
            # detect categorical
            is_cat = False
            if hasattr(tree.bins, "feature_types"):
                is_cat = (tree.bins.feature_types.get(gfi, "num") == "cat")
            else:
                is_cat = (self.feature_types.get(gfi, "num") == "cat")
            if not is_cat:
                continue

            G = hg[lpos, :finite_bins]
            H = hh[lpos, :finite_bins]
            K = G.shape[0]
            if K < 2:
                continue

            # rank categories
            score = np.abs(G) / (H + lam + 1e-12)
            order = np.argsort(-score)
            top = min(self.max_groups - 1, K)

            # form groups (top singletons + tail)
            groups: List[np.ndarray] = []
            if top <= 1:
                g0 = np.array([order[0]], dtype=np.int32)
                g1 = np.setdiff1d(np.arange(K, dtype=np.int32), g0, assume_unique=False)
                groups = [g0, g1]
            else:
                for i in range(top - 1):
                    groups.append(np.array([order[i]], dtype=np.int32))
                tail = np.setdiff1d(np.arange(K, dtype=np.int32),
                                    np.concatenate(groups), assume_unique=False)
                if tail.size > 0:
                    groups.append(tail)
                # also add the last strong category as its own group in front
                if len(groups) < self.max_groups and top >= 1:
                    last = np.array([order[top - 1]], dtype=np.int32)
                    groups = [last] + groups

            if len(groups) < 2:
                continue

            # choose the actual LEFT group (binary emulation)
            left_groups = groups[0]  # simple + stable
            # missing group heuristic: the group with max H
            H_group = np.array([H[g].sum() for g in groups], dtype=np.float64)
            missing_group = int(np.argmax(H_group))
            missing_left = (missing_group == 0)

            # compute binary gain: LEFT=left_groups (+missing if missing_left) vs REST
            G_tot = float(G.sum() + hg[lpos, finite_bins])
            H_tot = float(H.sum() + hh[lpos, finite_bins])
            parent = (G_tot * G_tot) / (H_tot + lam)

            GL = float(G[left_groups].sum())
            HL = float(H[left_groups].sum())
            if missing_left:
                GL += float(hg[lpos, finite_bins])
                HL += float(hh[lpos, finite_bins])

            GR = float(G.sum() - G[left_groups].sum())
            HR = float(H.sum() - H[left_groups].sum())
            if not missing_left:
                GR += float(hg[lpos, finite_bins])
                HR += float(hh[lpos, finite_bins])

            if HL < tree.min_child_weight or HR < tree.min_child_weight:
                continue

            child = (GL * GL) / (HL + lam) + (GR * GR) / (HR + lam)
            gain = 0.5 * (child - parent) - float(tree.gamma)
            if gain <= 0.0:
                continue

            out.append(Candidate("kway", float(gain), dict(
                gfi=gfi,
                left_groups=[int(x) for x in left_groups],
                missing_left=bool(missing_left),
            )))
        return out

class ObliqueGenerator(CandidateGenerator):
    def __init__(self, k_features: int = 6, ridge: float = 1e-3, max_candidates: int = 8):
        self.k = int(k_features); self.ridge = float(ridge); self.m = int(max_candidates)

    def generate(self, tree, X, node, hist_tuple) -> List[Candidate]:
        idx = node.data_indices
        g = node.gradients.astype(np.float64, copy=False)   # shape (|idx|,)
        h = node.hessians.astype(np.float64, copy=False)    # shape (|idx|,)
        X_loc = tree._X_train_cols[idx].astype(np.float64, copy=False)  # (|idx|, p_local)

        # rank features by |corr(x_j, g)|
        corrs = []
        for j in range(X_loc.shape[1]):
            xj = X_loc[:, j]
            m = np.isfinite(xj)
            if m.sum() < 2:
                corrs.append(0.0); continue
            xv = xj[m] - xj[m].mean()
            gv = g[m] - g[m].mean()
            denom = (np.sqrt((xv * xv).sum()) * np.sqrt((gv * gv).sum()) + 1e-12)
            corrs.append(float(abs((xv * gv).sum()) / denom))
        order = np.argsort(-np.asarray(corrs))
        base = order[: max(2, self.k)]

        cands: List[Candidate] = []
        for _ in range(min(self.m, max(1, len(base)))):
            S_local = base[: min(len(base), self.k)]
            XS = X_loc[:, S_local]                                # (n, k)
            A = XS.T @ (h.reshape(-1, 1) * XS)                    # X^T H X
            A.flat[:: A.shape[0] + 1] += (self.ridge + tree.lambda_)  # + ridge I
            b = -(XS.T @ g)                                       # -X^T g
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue

            z = XS @ w
            miss_mask = ~np.isfinite(XS).all(axis=1)
            order_z = np.argsort(z)
            z_ord = z[order_z]; g_ord = g[order_z]; h_ord = h[order_z]
            finite_mask = ~miss_mask[order_z]
            z_f = z_ord[finite_mask]; g_f = g_ord[finite_mask]; h_f = h_ord[finite_mask]
            gm = float(g_ord[~finite_mask].sum()); hm = float(h_ord[~finite_mask].sum())
            if z_f.size < 2 * tree.min_samples_leaf:
                continue

            Gp = np.cumsum(g_f); Hp = np.cumsum(h_f)
            parent = ((Gp[-1] + gm) ** 2) / (Hp[-1] + hm + tree.lambda_)

            best_gain = -1.0; best_thr = 0.0; best_mleft = True
            prev = z_f[0]
            for i in range(1, z_f.size):
                cur = z_f[i]
                if cur <= prev: continue
                GLp = float(Gp[i-1]); HLp = float(Hp[i-1])
                GRp = float(Gp[-1] - Gp[i-1]); HRp = float(Hp[-1] - Hp[i-1])

                # missing→left
                GL = GLp + gm; HL = HLp + hm; GR = GRp; HR = HRp
                if HL >= tree.min_child_weight and HR >= tree.min_child_weight:
                    gain = 0.5 * ((GL*GL)/(HL+tree.lambda_) + (GR*GR)/(HR+tree.lambda_) - parent) - tree.gamma
                    if gain > best_gain:
                        best_gain = gain; best_thr = 0.5*(prev+cur); best_mleft = True

                # missing→right
                GL = GLp; HL = HLp; GR = GRp + gm; HR = HRp + hm
                if HL >= tree.min_child_weight and HR >= tree.min_child_weight:
                    gain = 0.5 * ((GL*GL)/(HL+tree.lambda_) + (GR*GR)/(HR+tree.lambda_) - parent) - tree.gamma
                    if gain > best_gain:
                        best_gain = gain; best_thr = 0.5*(prev+cur); best_mleft = False

                prev = cur

            if best_gain <= 0.0:
                continue

            # map LOCAL → GLOBAL features
            features_global = [int(tree.feature_indices[j]) for j in S_local]
            cands.append(Candidate("oblique", float(best_gain), dict(
                features=features_global,
                weights=w.astype(np.float64, copy=False),
                threshold=float(best_thr),
                missing_left=bool(best_mleft),
                bias=0.0,
            )))
        return cands

class InteractionSeededGenerator(CandidateGenerator):
    def __init__(self, pairs: int = 5):
        self.pairs = int(pairs)

    def generate(self, tree, X, node, hist_tuple) -> List[Candidate]:
        idx = node.data_indices
        g = node.gradients.astype(np.float64, copy=False)
        h = node.hessians.astype(np.float64, copy=False)
        X_loc = tree._X_train_cols[idx].astype(np.float64, copy=False)

        var = np.nanvar(X_loc, axis=0)
        topv = np.argsort(-var)[: min(16, X_loc.shape[1])]
        scores = []
        for j in topv:
            xj = X_loc[:, j]
            m = np.isfinite(xj)
            if m.sum() < 3:
                scores.append(0.0); continue
            xv = xj[m] - xj[m].mean(); gv = g[m] - g[m].mean()
            denom = np.sqrt((xv*xv).sum()) * np.sqrt((gv*gv).sum()) + 1e-12
            scores.append(float(abs((xv*gv).sum()) / denom))
        ord2 = topv[np.argsort(-np.asarray(scores))]
        if ord2.size < 2:
            return []

        pairs = [(int(ord2[i]), int(ord2[j]))
                 for i in range(min(4, ord2.size))
                 for j in range(i+1, min(ord2.size, 8))]

        out: List[Candidate] = []
        for (a_local, b_local) in pairs[: self.pairs]:
            XS = X_loc[:, [a_local, b_local]]
            A = XS.T @ (h.reshape(-1, 1) * XS)
            A.flat[:: A.shape[0] + 1] += (1e-3 + tree.lambda_)
            bvec = -(XS.T @ g)
            try:
                w = np.linalg.solve(A, bvec)
            except np.linalg.LinAlgError:
                continue

            z = XS @ w
            miss_mask = ~np.isfinite(XS).all(axis=1)
            order_z = np.argsort(z)
            z_ord = z[order_z]; g_ord = g[order_z]; h_ord = h[order_z]
            finite_mask = ~miss_mask[order_z]
            z_f = z_ord[finite_mask]; g_f = g_ord[finite_mask]; h_f = h_ord[finite_mask]
            gm = float(g_ord[~finite_mask].sum()); hm = float(h_ord[~finite_mask].sum())
            if z_f.size < 2 * tree.min_samples_leaf:
                continue
            Gp = np.cumsum(g_f); Hp = np.cumsum(h_f)
            parent = ((Gp[-1] + gm) ** 2) / (Hp[-1] + hm + tree.lambda_)

            best_gain = -1.0; best_thr = 0.0; best_mleft = True
            prev = z_f[0]
            for i in range(1, z_f.size):
                cur = z_f[i]
                if cur <= prev: continue
                GLp = float(Gp[i-1]); HLp = float(Hp[i-1])
                GRp = float(Gp[-1] - Gp[i-1]); HRp = float(Hp[-1] - Hp[i-1])

                # missing→left
                GL = GLp + gm; HL = HLp + hm; GR = GRp; HR = HRp
                if HL >= tree.min_child_weight and HR >= tree.min_child_weight:
                    gain = 0.5 * ((GL*GL)/(HL+tree.lambda_) + (GR*GR)/(HR+tree.lambda_) - parent) - tree.gamma
                    if gain > best_gain:
                        best_gain = gain; best_thr = 0.5*(prev+cur); best_mleft = True

                # missing→right
                GL = GLp; HL = HLp; GR = GRp + gm; HR = HRp + hm
                if HL >= tree.min_child_weight and HR >= tree.min_child_weight:
                    gain = 0.5 * ((GL*GL)/(HL+tree.lambda_) + (GR*GR)/(HR+tree.lambda_) - parent) - tree.gamma
                    if gain > best_gain:
                        best_gain = gain; best_thr = 0.5*(prev+cur); best_mleft = False

                prev = cur

            if best_gain <= 0.0:
                continue

            # map LOCAL → GLOBAL
            features_global = [int(tree.feature_indices[a_local]),
                               int(tree.feature_indices[b_local])]
            out.append(Candidate("oblique", float(best_gain), dict(
                features=features_global,
                weights=w.astype(np.float64, copy=False),
                threshold=float(best_thr),
                missing_left=bool(best_mleft),
                bias=0.0,
            )))
        return out


# Adapter from Candidate → SplitPlan
def to_split_plan(c: Candidate) -> SplitPlan:
    if c.kind == "axis":
        return SplitPlan(kind="axis", gain=c.gain, **c.payload)
    if c.kind == "kway":
        return SplitPlan(kind="kway", gain=c.gain, **c.payload)
    if c.kind == "oblique":
        return SplitPlan(kind="oblique", gain=c.gain, **c.payload)
    raise ValueError(c.kind)
