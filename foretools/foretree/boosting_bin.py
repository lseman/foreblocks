import threading
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numba
import numpy as np
from numba import njit, prange

warnings.filterwarnings("ignore")


# =========================================================
#                     Low-level kernels
# =========================================================


@njit(fastmath=True, cache=True)
def _kahan_cumsum(x: np.ndarray) -> np.ndarray:
    n = x.size
    out = np.empty_like(x)
    s = 0.0
    c = 0.0
    for i in range(n):
        y = x[i] - c
        t = s + y
        c = (t - s) - y
        s = t
        out[i] = s
    return out


@njit(fastmath=True, cache=True)
def _histogram_sum(
    values: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-friendly histogram using searchsorted on right edges.
    Returns sums of gradients and hessians per bin.
    """
    n_bins = edges.size - 1
    if n_bins <= 0:
        return np.zeros(0, np.float64), np.zeros(0, np.float64)

    mask = np.isfinite(values) & np.isfinite(gradients) & np.isfinite(hessians)
    if not np.any(mask):
        return np.zeros(n_bins, np.float64), np.zeros(n_bins, np.float64)

    v = values[mask]
    g = gradients[mask]
    h = hessians[mask]

    # Assign bins: index i such that edges[i] <= v < edges[i+1]
    # Using right edges for searchsorted yields [0, n_bins]; clamp to [0, n_bins-1]
    idx = np.searchsorted(edges[1:], v, side="left")
    if n_bins > 1:
        idx = np.minimum(idx, n_bins - 1)

    hist_g = np.zeros(n_bins, np.float64)
    hist_h = np.zeros(n_bins, np.float64)
    for i in range(v.size):
        vi = v[i]
        # reject values strictly outside [edges[0], edges[-1]]
        if vi < edges[0] or vi > edges[-1]:
            continue
        b = idx[i]
        hist_g[b] += g[i]
        hist_h[b] += h[i]
    return hist_g, hist_h


@njit(fastmath=True, cache=True)
def _split_gains(
    hist_g: np.ndarray,
    hist_h: np.ndarray,
    lambda_reg: float,
    gamma: float,
    min_child_weight: float,
) -> np.ndarray:
    """
    Vectorized split gain per split between consecutive bins.
    """
    n_bins = hist_g.size
    if n_bins <= 1:
        return np.zeros(0, np.float64) - np.inf

    c_g = _kahan_cumsum(hist_g)
    c_h = _kahan_cumsum(hist_h)

    G = c_g[-1]
    H = c_h[-1]
    if H <= 1e-16:
        return np.zeros(n_bins - 1, np.float64) - np.inf

    parent = (G * G) / (H + lambda_reg)

    gL = c_g[:-1]
    hL = c_h[:-1]
    gR = G - gL
    hR = H - hL

    valid = (hL >= min_child_weight) & (hR >= min_child_weight)

    left = (gL * gL) / (hL + lambda_reg)
    right = (gR * gR) / (hR + lambda_reg)

    gains = 0.5 * (left + right - parent) - gamma
    out = np.zeros(n_bins - 1, np.float64) - np.inf
    for i in range(gains.size):
        if valid[i]:
            out[i] = gains[i]
    return out


@njit(fastmath=True, cache=True)
def _adaptive_quantile_edges(
    values: np.ndarray, weights: np.ndarray, k_bins: int, density_aware: bool = True
) -> np.ndarray:
    """
    Weighted quantile edges with simple density-awareness (works well in practice).
    """
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.array([0.0, 1.0], np.float64)

    v = values[mask].astype(np.float64)
    w = weights[mask].astype(np.float64)

    # unique shortcut
    if v.size <= k_bins:
        u = np.unique(v)
        if u.size == 1:
            return np.array([u[0], u[0] + 1e-12], np.float64)
        return u

    order = np.argsort(v, kind="mergesort")
    vs = v[order]
    ws = w[order]

    cw = _kahan_cumsum(ws)
    tw = cw[-1]
    if tw <= 0:
        return np.array([vs[0], vs[-1]], np.float64)

    cw = cw / tw

    if density_aware:
        edges = np.empty(k_bins + 1, np.float64)
        edges[0] = vs[0]
        edges[-1] = vs[-1]
        for i in range(1, k_bins):
            q = i / k_bins
            j = np.searchsorted(cw, q, side="left")
            if j >= vs.size:
                j = vs.size - 1
            edges[i] = vs[j]
    else:
        qs = np.linspace(0.0, 1.0, k_bins + 1)
        edges = np.interp(qs, cw, vs)

    # uniq/order guard
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([vs[0], vs[-1]], np.float64)
    return edges


@njit(fastmath=True, cache=True)
def _lightgbm_plus_refine(
    values: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    coarse_edges: np.ndarray,
    hist_g: np.ndarray,
    hist_h: np.ndarray,
    lambda_reg: float,
    max_bins: int,
    complexity: float,
) -> np.ndarray:
    """
    Allocate sub-bins to coarse bins proportionally to an importance metric.
    """
    n_coarse = coarse_edges.size - 1
    if n_coarse <= 0:
        return coarse_edges

    # budget: scale with complexity in [0.5, 2.0]
    cf = complexity
    if cf < 0.5:
        cf = 0.5
    if cf > 2.0:
        cf = 2.0
    budget = int(max_bins * cf)
    if budget <= n_coarse:
        return coarse_edges

    imp = np.zeros(n_coarse, np.float64)
    Hsum = 0.0
    for i in range(n_coarse):
        Hsum += hist_h[i]
    for i in range(n_coarse):
        if hist_h[i] > 0:
            gi = np.abs(hist_g[i]) / np.sqrt(hist_h[i] + lambda_reg)
            hi = hist_h[i] / (Hsum + 1e-12)
            imp[i] = 0.7 * gi + 0.3 * hi

    total_imp = np.sum(imp)
    if total_imp <= 0:
        return coarse_edges

    # start: keep all coarse edges
    out = [coarse_edges[0]]
    # distribute at least one sub-bin per coarse bin
    remaining = budget - n_coarse

    for i in range(n_coarse):
        left = coarse_edges[i]
        right = coarse_edges[i + 1]
        if remaining <= 0:
            out.append(right)
            continue

        # proportional allocation
        extra = int((imp[i] / total_imp) * remaining)
        n_sub = 1 + max(0, extra)  # at least 1
        # collect samples in this range
        if i == n_coarse - 1:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)

        if not np.any(mask) or n_sub <= 1:
            out.append(right)
            continue

        v = values[mask]
        w = hessians[mask] + 1e-12
        sub = _adaptive_quantile_edges(v, w, n_sub, True)
        # inject interior edges only
        for k in range(1, sub.size - 1):
            e = float(sub[k])
            if (e > left) and (e < right):
                out.append(e)
        out.append(right)

    arr = np.array(out, np.float64)
    arr = np.unique(np.array(out, np.float64))
    # final cap
    if arr.size - 1 > max_bins:
        q = np.linspace(0.0, 1.0, max_bins + 1)
        lo, hi = arr[0], arr[-1]
        arr = np.unique(lo + q * (hi - lo))
    return arr


# =========================================================
#                        Config
# =========================================================


@dataclass
class BinningConfig:
    coarse_bins: int = 64
    max_total_bins: int = 256
    min_child_weight: float = 1e-3
    min_gain_threshold: float = 1e-2
    top_regions: int = 8
    overlap_merge_threshold: int = 2

    categorical_threshold: int = 32
    node_refinement_threshold: int = 1000
    max_refinement_depth: int = 8
    refinement_feature_fraction: float = 0.3
    refinement_min_correlation: float = 0.10

    use_density_aware_binning: bool = True
    use_parallel: bool = True
    max_workers: int = 4

    cache_size: int = 1024
    refinement_cache_size: int = 2048

    eps: float = 1e-12  # numeric epsilon for edge dedup


# =========================================================
#               Gradient-aware SOTA binner
# =========================================================


class GradientBinner:
    """
    Clean, SOTA-oriented gradient-aware binning with per-node refinement.
    """

    def __init__(self, config: Optional[BinningConfig] = None):
        self.config = config or BinningConfig()
        self._lock = threading.Lock()

        # LRU caches
        self._feature_edges: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._feature_stats: Dict[
            int, Tuple[float, float, float, int, float, float]
        ] = {}
        self._feature_complex: Dict[int, float] = {}

        self._node_edges: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self._executor = (
            ThreadPoolExecutor(max_workers=self.config.max_workers)
            if self.config.use_parallel
            else None
        )

    def __del__(self):
        if self._executor:
            self._executor.shutdown(wait=False)

    # ---------- Public API ----------

    def create_bins(
        self,
        feature_values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        feature_idx: Optional[int] = None,
        lambda_reg: float = 1.0,
        gamma: float = 0.0,
    ) -> Tuple[np.ndarray, Dict]:
        v = np.asarray(feature_values, np.float64)
        g = np.asarray(gradients, np.float64)
        h = np.asarray(hessians, np.float64)
        if not (v.size == g.size == h.size):
            raise ValueError("feature_values, gradients, hessians must match in length")

        # cache hit?
        if feature_idx is not None and self._feature_cached_valid(feature_idx, v):
            with self._lock:
                edges = self._feature_edges[feature_idx]
                stats = self._feature_stats.get(feature_idx, (0, 0, 1, 0, 0, 0))
                # LRU touch
                self._feature_edges.move_to_end(feature_idx)
            return edges, {
                "strategy": "cached",
                "n_bins": edges.size - 1,
                "cache_hit": True,
                "feature_stats": stats,
            }

        stats = self._compute_feature_stats(v, g, h)
        complexity = self._estimate_complexity(v, g, h)

        strategy = self._select_strategy(stats, complexity)
        if strategy == "categorical":
            edges = self._categorical_edges(v)
            regions: List[Tuple[int, int, float]] = []
        elif strategy == "uniform":
            edges = self._uniform_edges(v, self.config.coarse_bins)
            regions = []
        else:
            edges, regions = self._gradient_aware(
                v, g, h, complexity, lambda_reg, gamma
            )

        # commit to cache
        if feature_idx is not None:
            with self._lock:
                self._feature_edges[feature_idx] = edges
                self._feature_stats[feature_idx] = stats
                self._feature_complex[feature_idx] = complexity
                self._evict_lru(self._feature_edges, self.config.cache_size)

        meta = {
            "strategy": strategy,
            "n_bins": edges.size - 1,
            "n_regions": len(regions),
            "feature_complexity": complexity,
            "cache_hit": False,
            "regions": regions,
        }
        return edges, meta

    def create_node_refined_bins(
        self,
        feature_values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        feature_idx: int,
        node_id: str,
        tree_depth: int = 0,
        parent_edges: Optional[np.ndarray] = None,
        lambda_reg: float = 1.0,
        gamma: float = 0.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Per-node refinement. Uses parent edges if present; otherwise falls back to global bins.
        Only refines when the node/feature is promising. Always returns a valid edges array
        (float64, sorted, unique, len>=2) and a metadata dict.
        """
        # --- normalize inputs ---
        v = np.asarray(feature_values, dtype=np.float64)
        g = np.asarray(gradients, dtype=np.float64)
        h = np.asarray(hessians, dtype=np.float64)

        # finite mask
        finite = np.isfinite(v) & np.isfinite(g) & np.isfinite(h)
        n_fin = int(np.sum(finite))
        if n_fin == 0:
            # no usable data: fallback to parent or minimal
            if parent_edges is not None and parent_edges.size >= 2:
                return parent_edges.astype(np.float64, copy=False), {
                    "strategy": "no_finite_parent",
                    "refined": False,
                }
            base = np.array([0.0, 1.0], dtype=np.float64)
            return base, {"strategy": "no_finite_minimal", "refined": False}

        v = v[finite]
        g = g[finite]
        h = h[finite]

        # --- quick applicability checks (use finite count) ---
        if (n_fin < int(self.config.node_refinement_threshold)) or (
            tree_depth > int(self.config.max_refinement_depth)
        ):
            if parent_edges is not None and parent_edges.size >= 2:
                return parent_edges.astype(np.float64, copy=False), {
                    "strategy": "parent_fallback",
                    "refined": False,
                }
            base_edges, _ = self.create_bins(
                v, g, h, feature_idx=feature_idx, lambda_reg=lambda_reg, gamma=gamma
            )
            base_edges = np.asarray(base_edges, dtype=np.float64)
            return base_edges, {"strategy": "global_fallback", "refined": False}

        # --- cache lookup (use same key format everywhere) ---
        cache_key = f"{node_id}_{int(feature_idx)}"
        if not hasattr(self, "_node_edges_cache"):
            # prefer OrderedDict for LRU, but a plain dict also works with simple eviction
            from collections import OrderedDict

            self._node_edges_cache = OrderedDict()
        with self._lock:
            e_cached = self._node_edges_cache.get(cache_key)
            if e_cached is not None and e_cached.size >= 2:
                # touch for LRU
                try:
                    self._node_edges_cache.move_to_end(cache_key)
                except Exception:
                    pass
                return e_cached, {
                    "strategy": "cached_node",
                    "refined": True,
                    "cache_hit": True,
                    "actual_bins": int(e_cached.size - 1),
                }

        # --- decide whether to refine ---
        try:
            should = self._should_refine_feature(
                v, g, h, feature_idx, tree_depth
            )  # align name with your class
        except TypeError:
            # if your method signature is _should_refine(...), adapt here:
            should = self._should_refine(v, g, feature_idx, tree_depth)
        if not should:
            if parent_edges is not None and parent_edges.size >= 2:
                return parent_edges.astype(np.float64, copy=False), {
                    "strategy": "parent_inherited",
                    "refined": False,
                }
            base_edges, _ = self.create_bins(
                v, g, h, feature_idx=feature_idx, lambda_reg=lambda_reg, gamma=gamma
            )
            base_edges = np.asarray(base_edges, dtype=np.float64)
            return base_edges, {"strategy": "global_fallback", "refined": False}

        # --- obtain a starting edge set ---
        if parent_edges is not None and parent_edges.size >= 2:
            start_edges = np.asarray(parent_edges, dtype=np.float64)
        else:
            start_edges, _ = self.create_bins(
                v, g, h, feature_idx=feature_idx, lambda_reg=lambda_reg, gamma=gamma
            )
            start_edges = np.asarray(start_edges, dtype=np.float64)

        # --- refine ---
        try:
            edges = self._refine_from_parent(
                v, g, h, start_edges, feature_idx, tree_depth, lambda_reg
            )
            # allow _refine_from_parent to return None → fallback to start_edges
            if edges is None or np.asarray(edges).size < 2:
                edges = start_edges
        except Exception:
            edges = start_edges

        # --- sanitize edges: sorted, unique, finite, len>=2 ---
        edges = np.asarray(edges, dtype=np.float64)
        edges = edges[np.isfinite(edges)]
        if edges.size < 2:
            edges = start_edges
        edges = np.unique(edges)
        if edges.size < 2:
            # last resort: tight 2-point span around median
            m = float(np.nanmedian(v))
            edges = np.array([m - 1e-12, m + 1e-12], dtype=np.float64)

        # Optionally ensure strictly increasing (guard tiny duplicates)
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)

        meta = {
            "strategy": (
                "parent_refined"
                if (parent_edges is not None and parent_edges.size >= 2)
                else "new_refined"
            ),
            "refined": True,
            "node_depth": int(tree_depth),
            "node_samples": int(n_fin),
            "actual_bins": int(edges.size - 1),
            "cache_hit": False,
        }

        # --- cache result (LRU) ---
        with self._lock:
            self._node_edges_cache[cache_key] = edges
            # Evict if over capacity
            try:
                limit = int(self.config.refinement_cache_size)
            except Exception:
                limit = 1000
            if limit > 0:
                while len(self._node_edges_cache) > limit:
                    try:
                        self._node_edges_cache.popitem(last=False)  # OrderedDict LRU
                    except Exception:
                        # plain dict fallback
                        k = next(iter(self._node_edges_cache))
                        del self._node_edges_cache[k]

        max_bins = int(self.config.max_total_bins)
        if edges.size - 1 > max_bins:
            q = np.linspace(0.0, 1.0, max_bins + 1)
            lo, hi = edges[0], edges[-1]
            edges = np.unique(lo + q * (hi - lo))
            meta["actual_bins"] = int(edges.size - 1)

        return edges, meta

    def batch_create_bins(
        self,
        features_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]],
        lambda_reg: float = 1.0,
        gamma: float = 0.0,
    ) -> List[Tuple[np.ndarray, Dict]]:
        if not self._executor or len(features_data) < 2:
            return [
                self.create_bins(v, g, h, idx, lambda_reg, gamma)
                for (v, g, h, idx) in features_data
            ]

        futures = [
            self._executor.submit(self.create_bins, v, g, h, idx, lambda_reg, gamma)
            for (v, g, h, idx) in features_data
        ]

        out: List[Tuple[np.ndarray, Dict]] = []
        for k, fut in enumerate(futures):
            try:
                out.append(fut.result(timeout=30))
            except Exception as e:
                v = features_data[k][0]
                edges = self._uniform_edges(v, 32)
                out.append((edges, {"strategy": "fallback_uniform", "error": str(e)}))
        return out

    def batch_create_node_bins(
        self,
        node_data: List[
            Tuple[
                np.ndarray, np.ndarray, np.ndarray, int, str, int, Optional[np.ndarray]
            ]
        ],
        lambda_reg: float = 1.0,
        gamma: float = 0.0,
    ) -> List[Tuple[np.ndarray, Dict]]:
        if not self._executor or len(node_data) < 2:
            return [
                self.create_node_refined_bins(
                    v, g, h, fi, nid, dep, pedges, lambda_reg, gamma
                )
                for (v, g, h, fi, nid, dep, pedges) in node_data
            ]

        futures = [
            self._executor.submit(
                self.create_node_refined_bins,
                v,
                g,
                h,
                fi,
                nid,
                dep,
                pedges,
                lambda_reg,
                gamma,
            )
            for (v, g, h, fi, nid, dep, pedges) in node_data
        ]
        out: List[Tuple[np.ndarray, Dict]] = []
        for k, fut in enumerate(futures):
            try:
                out.append(fut.result(timeout=30))
            except Exception as e:
                v = node_data[k][0]
                edges = self._uniform_edges(v, 16)
                out.append((edges, {"strategy": "parallel_fallback", "error": str(e)}))
        return out

    # ---------- Internals ----------

    def _gradient_aware(
        self,
        v: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        complexity: float,
        lambda_reg: float,
        gamma: float,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        n_coarse = int(
            min(self.config.coarse_bins, max(16, 32 * max(0.5, min(2.0, complexity))))
        )
        coarse = self._coarse_from_quantiles(v, n_coarse)

        hist_g, hist_h = _histogram_sum(v, g, h, coarse)
        gains = _split_gains(
            hist_g, hist_h, lambda_reg, gamma, self.config.min_child_weight
        )

        regions = self._regions_from_gains(gains, coarse, complexity)

        # LightGBM+ refinement (good default)
        edges = _lightgbm_plus_refine(
            v,
            g,
            h,
            coarse,
            hist_g,
            hist_h,
            lambda_reg,
            self.config.max_total_bins,
            complexity,
        )
        return edges, regions

    def _regions_from_gains(
        self, gains: np.ndarray, edges: np.ndarray, complexity: float
    ) -> List[Tuple[int, int, float]]:
        if gains.size == 0:
            return []
        thr = self.config.min_gain_threshold * (1.0 + complexity)
        idx = np.where(gains > thr)[0]
        if idx.size == 0:
            return []

        # take top candidates, merge small overlaps
        maxr = max(self.config.top_regions, int(self.config.top_regions * complexity))
        order = idx[np.argsort(gains[idx])[::-1]]
        order = order[: (maxr * 2)]
        order = np.sort(order)

        regions: List[Tuple[int, int, float]] = []
        i = 0
        while i < order.size:
            s = order[i]
            e = s + 1
            gmax = gains[s]
            j = i + 1
            while (
                j < order.size and order[j] <= s + self.config.overlap_merge_threshold
            ):
                e = max(e, order[j] + 1)
                if gains[order[j]] > gmax:
                    gmax = gains[order[j]]
                j += 1
            if gmax > thr:
                regions.append((s, e, float(gmax)))
            i = j
        regions.sort(key=lambda t: t[2], reverse=True)
        return regions[:maxr]

    def _refine_from_parent(
        self,
        v: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        parent_edges: Optional[np.ndarray],
        feature_idx: int,
        depth: int,
        lambda_reg: float,
    ) -> np.ndarray:
        # target bins ↓ with depth
        base = max(8, min(64, v.size // 20))
        target = max(4, int(base * max(0.3, 1.0 - 0.05 * depth)))

        if parent_edges is None or parent_edges.size < 2:
            return _adaptive_quantile_edges(v, np.maximum(h, 1e-12), target, True)

        # parent stats
        pg, ph = _histogram_sum(v, g, h, parent_edges)
        imp = np.zeros(pg.size, np.float64)
        for i in range(pg.size):
            if ph[i] > 0:
                imp[i] = np.abs(pg[i]) / np.sqrt(ph[i] + lambda_reg)

        total = np.sum(imp)
        if total <= 0:
            return parent_edges

        edges_set = _edges_to_set(parent_edges, self.config.eps)
        remaining = max(0, target - (parent_edges.size - 1))

        order = np.argsort(imp)[::-1]
        for k in order:
            if remaining <= 0:
                break
            if imp[k] <= 0:
                continue
            left = parent_edges[k]
            right = parent_edges[k + 1]

            # collect samples in parent bin
            if k == parent_edges.size - 2:
                mask = (v >= left) & (v <= right)
            else:
                mask = (v >= left) & (v < right)
            if not np.any(mask):
                continue

            vv = v[mask]
            ww = np.maximum(h[mask], 1e-12)
            # allocate proportional internal edges
            want = min(
                remaining, max(1, int((imp[k] / (total + 1e-12)) * remaining) + 1)
            )
            sub = _adaptive_quantile_edges(vv, ww, want + 1, True)
            # add interior edges with tolerance
            for t in range(1, sub.size - 1):
                e = float(sub[t])
                if (
                    (e > left)
                    and (e < right)
                    and _edgeset_add(edges_set, e, self.config.eps)
                ):
                    remaining -= 1
                    if remaining <= 0:
                        break

        out = np.array(sorted([kv / 1.0 for kv in edges_set.keys()]), np.float64)
        # cap if we overshot
        if out.size - 1 > target:
            q = np.linspace(0.0, 1.0, target + 1)
            lo, hi = out[0], out[-1]
            out = lo + q * (hi - lo)
            out = np.unique(out)
        return out

    # ---------- Stats / Strategy ----------

    def _compute_feature_stats(
        self, v: np.ndarray, g: np.ndarray, h: np.ndarray
    ) -> Tuple[float, float, float, int, float, float]:
        mask = np.isfinite(v) & np.isfinite(g) & np.isfinite(h)
        if not np.any(mask):
            return 0.0, 0.0, 1.0, 0, 0.0, 0.0
        vv = v[mask]
        gg = g[mask]
        hh = h[mask]
        n_unique = int(np.unique(vv).size)
        miss_rate = 1.0 - (vv.size / max(1, v.size))

        def _corr(a, b):
            sa = np.std(a)
            sb = np.std(b)
            if sa <= 1e-12 or sb <= 1e-12:
                return 0.0
            try:
                return float(np.corrcoef(a, b)[0, 1])
            except Exception:
                return 0.0

        corr_g = abs(_corr(vv, gg))
        corr_h = abs(_corr(vv, hh))
        skew = 0.0
        sv = np.std(vv)
        if sv > 1e-12:
            mu = np.mean(vv)
            skew = float(np.mean(((vv - mu) / sv) ** 3))
        return corr_g, skew, miss_rate, n_unique, corr_h, float(sv)

    def _estimate_complexity(
        self, v: np.ndarray, g: np.ndarray, h: np.ndarray
    ) -> float:
        mask = np.isfinite(v) & np.isfinite(g) & np.isfinite(h)
        vv = v[mask]
        gg = g[mask]
        hh = h[mask]
        if vv.size < 10:
            return 0.3

        gv = np.sqrt(max(0.0, float(np.var(gg)))) if gg.size > 1 else 0.0
        hv = np.sqrt(max(0.0, float(np.var(hh)))) if hh.size > 1 else 0.0

        sidx = np.argsort(vv)
        gs = gg[sidx]

        # second differences: gs[i+1] - 2*gs[i] + gs[i-1]
        m = max(0, gs.size - 2)
        if m > 0:
            local = float(np.sum(np.abs(gs[2:] - 2.0 * gs[1:-1] + gs[:-2])))
            local /= m
        else:
            local = 0.0

        smooth = 1.0 / (1.0 + local)
        comp = 0.45 * gv + 0.35 * hv + 0.20 * (1.0 - smooth)
        return float(np.clip(comp, 0.1, 2.0))

    def _select_strategy(self, stats: Tuple, complexity: float) -> str:
        corr_g, skew, miss, n_unique, corr_h, sv = stats
        if n_unique <= self.config.categorical_threshold:
            return "categorical"
        if (
            (complexity < 0.3)
            and (abs(corr_g) < 0.1)
            and (abs(corr_h) < 0.1)
            and (abs(skew) < 1.0)
        ):
            return "uniform"
        return "gradient_aware"

    # ---------- Edges helpers ----------

    def _coarse_from_quantiles(self, v: np.ndarray, n_bins: int) -> np.ndarray:
        fv = v[np.isfinite(v)]
        if fv.size == 0:
            return np.array([0.0, 1.0], np.float64)
        u = np.unique(fv)
        if u.size <= n_bins:
            return u.astype(np.float64)
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        e = np.quantile(fv, qs, method="linear")
        e = np.unique(e).astype(np.float64)
        # enforce tiny separation
        minsep = (e[-1] - e[0]) / max(2, e.size * 1000)
        for i in range(1, e.size):
            if e[i] - e[i - 1] < minsep:
                e[i] = e[i - 1] + minsep
        return e

    def _categorical_edges(self, v: np.ndarray) -> np.ndarray:
        fv = v[np.isfinite(v)]
        if fv.size == 0:
            return np.array([0.0, 1.0], np.float64)
        u = np.unique(fv)
        if u.size == 1:
            x = float(u[0])
            return np.array([x - self.config.eps, x + self.config.eps], np.float64)
        edges = np.empty(u.size + 1, np.float64)
        edges[0] = float(u[0]) - self.config.eps
        edges[-1] = float(u[-1]) + self.config.eps
        for i in range(1, u.size):
            edges[i] = 0.5 * (u[i - 1] + u[i])
        return edges

    def _uniform_edges(self, v: np.ndarray, n_bins: int) -> np.ndarray:
        fv = v[np.isfinite(v)]
        if fv.size == 0:
            return np.array([0.0, 1.0], np.float64)
        lo, hi = float(np.min(fv)), float(np.max(fv))
        if lo == hi:
            return np.array([lo, hi + self.config.eps], np.float64)
        return np.linspace(lo, hi, n_bins + 1, dtype=np.float64)

    # ---------- Policies ----------

    def _should_refine(
        self, v: np.ndarray, g: np.ndarray, feature_idx: int, depth: int
    ) -> bool:
        fv = np.isfinite(v) & np.isfinite(g)
        vv = v[fv]
        gg = g[fv]
        if vv.size < 20:
            return False
        # correlation
        sg = np.std(gg)
        sv = np.std(vv)
        if sg <= 1e-12 or sv <= 1e-12:
            return False
        try:
            corr = abs(float(np.corrcoef(vv, gg)[0, 1]))
        except Exception:
            return False
        if corr < self.config.refinement_min_correlation:
            return False

        depth_factor = min(1.0, 0.3 + 0.1 * depth)
        complexity = self._feature_complex.get(feature_idx, 0.5)
        score = corr * complexity * depth_factor
        return score > (2.0 * self.config.refinement_min_correlation)

    # ---------- Caching ----------
    def _feature_cached_valid(
        self, feature_idx: int, current_values: np.ndarray
    ) -> bool:
        with self._lock:
            if feature_idx not in self._feature_edges:
                return False
            edges = self._feature_edges[feature_idx]
        fv = current_values[np.isfinite(current_values)]
        if fv.size == 0:
            return True
        lo, hi = float(np.min(fv)), float(np.max(fv))
        elo, ehi = float(edges[0]), float(edges[-1])

        abs_range = max(ehi - elo, 0.0)
        tol = max(1e-6 * max(abs_range, 1.0), 10.0 * self.config.eps)  # floor + eps
        return (lo >= (elo - tol)) and (hi <= (ehi + tol))

    @staticmethod
    def _evict_lru(od: "OrderedDict", limit: int) -> None:
        while len(od) > limit:
            od.popitem(last=False)


# =========================================================
#                 Small edge-set with tolerance
# =========================================================


def _edges_to_set(edges: np.ndarray, eps: float) -> Dict[int, float]:
    """
    Store edges quantized by eps for O(1) tolerant membership checks.
    """
    out: Dict[int, float] = {}
    inv = 1.0 / max(eps, 1e-18)
    for e in edges:
        k = int(round(e * inv))
        out[k] = float(e)
    return out


def _edgeset_add(es: Dict[int, float], val: float, eps: float) -> bool:
    inv = 1.0 / max(eps, 1e-18)
    k = int(round(val * inv))
    if k in es:
        return False
    es[k] = float(val)
    return True
