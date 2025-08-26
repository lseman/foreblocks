from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numba import njit


@njit
def _find_split_gains_numba(hist_g: np.ndarray, hist_h: np.ndarray, 
                           lambda_reg: float, gamma: float) -> np.ndarray:
    """Numba-optimized gain calculation for all split points."""
    n_bins = len(hist_g)
    gains = np.full(n_bins - 1, -np.inf, dtype=np.float64)
    
    # Calculate cumulative sums
    cum_g = np.cumsum(hist_g)
    cum_h = np.cumsum(hist_h)
    total_g = cum_g[-1]
    total_h = cum_h[-1]
    
    if total_h <= 0:
        return gains
    
    # Parent score
    parent_score = total_g * total_g / (total_h + lambda_reg)
    
    # Calculate gains for all split points
    for i in range(n_bins - 1):
        g_left = cum_g[i]
        h_left = cum_h[i]
        g_right = total_g - g_left
        h_right = total_h - h_left
        
        if h_left > 0 and h_right > 0:
            left_score = g_left * g_left / (h_left + lambda_reg)
            right_score = g_right * g_right / (h_right + lambda_reg)
            gain = 0.5 * (left_score + right_score - parent_score) - gamma
            gains[i] = gain
    
    return gains

@njit(cache=True, fastmath=True)
def _build_histogram_numba(values, gradients, hessians, bin_edges):
    """
    Build gradient/hessian histograms for a single feature with given edges.
    values: float64[::1], gradients: float32/64[::1], hessians: float32/64[::1]
    bin_edges: float64[::1] (length = n_bins+1)
    """
    n_bins = len(bin_edges) - 1
    hist_g = np.zeros(n_bins, dtype=np.float64)
    hist_h = np.zeros(n_bins, dtype=np.float64)

    # Pre-slice to avoid repeated materialization
    upper = bin_edges[1:]

    for i in range(values.shape[0]):
        v = values[i]
        if np.isnan(v):
            continue
        # Binary search: place v in first upper edge >= v
        b = np.searchsorted(upper, v)
        if 0 <= b < n_bins:
            hist_g[b] += gradients[i]
            hist_h[b] += hessians[i]
    return hist_g, hist_h


@njit(cache=True, fastmath=True)
def _best_split_from_hist_with_missing_numba(hist_g, hist_h, g_miss, h_miss, lambda_reg, gamma, min_child_weight):
    """
    Returns: (best_gain, best_bin_idx, missing_left_flag)
    Uses XGBoost-style score: 0.5 * (left + right - parent) - gamma
    Tries both policies: missing -> left and missing -> right.
    """
    n_bins = hist_g.shape[0]
    if n_bins <= 1:
        return -np.inf, -1, False

    # Cumulative sums
    cum_g = np.cumsum(hist_g)
    cum_h = np.cumsum(hist_h)
    G = cum_g[-1]
    H = cum_h[-1]

    if H <= 0.0:
        return -np.inf, -1, False

    parent = (G * G) / (H + lambda_reg)
    best_gain = -np.inf
    best_idx = -1
    best_miss_left = False

    for i in range(n_bins - 1):
        # finite-only left/right
        gL = cum_g[i]
        hL = cum_h[i]
        gR = G - gL
        hR = H - hL

        # Skip hopeless splits (child weights too small)
        if hL < min_child_weight or hR < min_child_weight:
            continue

        # Case A: missing -> left
        gL_A = gL + g_miss
        hL_A = hL + h_miss
        gR_A = gR
        hR_A = hR
        if hL_A >= min_child_weight and hR_A >= min_child_weight:
            left_A  = (gL_A * gL_A) / (hL_A + lambda_reg)
            right_A = (gR_A * gR_A) / (hR_A + lambda_reg)
            gain_A  = 0.5 * (left_A + right_A - parent) - gamma
            if gain_A > best_gain:
                best_gain = gain_A
                best_idx = i
                best_miss_left = True

        # Case B: missing -> right
        gL_B = gL
        hL_B = hL
        gR_B = gR + g_miss
        hR_B = hR + h_miss
        if hL_B >= min_child_weight and hR_B >= min_child_weight:
            left_B  = (gL_B * gL_B) / (hL_B + lambda_reg)
            right_B = (gR_B * gR_B) / (hR_B + lambda_reg)
            gain_B  = 0.5 * (left_B + right_B - parent) - gamma
            if gain_B > best_gain:
                best_gain = gain_B
                best_idx = i
                best_miss_left = False

    return best_gain, best_idx, best_miss_left


@dataclass
class BinningConfig:
    coarse_bins: int = 32
    max_total_bins: int = 128
    top_regions: int = 3
    min_region_samples: int = 50
    min_gain_threshold: float = 0.1
    overlap_merge_threshold: int = 2

class MultiLevelBinner:
    def __init__(self, config: BinningConfig = None):
        self.config = config or BinningConfig()

    def create_adaptive_bins(self, feature_values: np.ndarray, gradients: np.ndarray,
                             hessians: np.ndarray, lambda_reg: float = 1.0,
                             gamma: float = 0.0) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        finite_mask = np.isfinite(feature_values)
        finite_values = feature_values[finite_mask]
        n_samples = finite_values.shape[0]
        n_unique = int(np.unique(finite_values).shape[0])

        # Early exit: uniform-ish
        if (n_samples < self.config.min_region_samples * 2) or (n_unique <= self.config.coarse_bins):
            uniform_bins = self._create_uniform_bins(finite_values, min(n_unique, self.config.max_total_bins))
            return uniform_bins, []

        # Step 1: coarse bins
        coarse_edges = self._create_uniform_bins(finite_values, self.config.coarse_bins)

        # Step 2: coarse histogram
        g_coarse, h_coarse = _build_histogram_numba(feature_values.astype(np.float64, copy=False),
                                                    gradients, hessians,
                                                    coarse_edges.astype(np.float64, copy=False))

        # Step 3: promising regions
        regions = self._find_promising_regions_fixed(g_coarse, h_coarse, coarse_edges, lambda_reg, gamma)
        if not regions:
            return coarse_edges, []

        # Step 4: refine within budget
        refined_edges = self._create_refined_bins_with_budget(finite_values, coarse_edges, regions)
        return refined_edges, regions

    def _create_uniform_bins(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        if values.shape[0] == 0:
            return np.array([0.0, 1.0], dtype=np.float64)
        unique_vals = np.unique(values)
        if unique_vals.shape[0] <= n_bins:
            # Ensure edges (include min/max twice if single value)
            if unique_vals.shape[0] == 1:
                v = unique_vals[0]
                return np.array([v, v], dtype=np.float64)
            return unique_vals.astype(np.float64, copy=False)
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(unique_vals, q)
        # Guarantee strictly non-decreasing and unique
        return np.unique(edges).astype(np.float64, copy=False)

    def _find_promising_regions_fixed(self, hist_g: np.ndarray, hist_h: np.ndarray,
                                      bin_edges: np.ndarray, lambda_reg: float, gamma: float) -> List[Tuple[int, int, float]]:
        gains = _find_split_gains_numba(hist_g, hist_h, lambda_reg, gamma)
        good = np.where(gains > self.config.min_gain_threshold)[0]
        if good.size == 0:
            return []

        # take more candidates than top_regions, then merge
        cand = good[np.argsort(gains[good])][-self.config.top_regions * 3:]
        cand.sort()

        merged: List[Tuple[int, int, float]] = []
        cs = int(cand[0]); ce = cs + 1; cg = float(gains[cs])
        for k in range(1, cand.size):
            idx = int(cand[k])
            if idx <= ce + self.config.overlap_merge_threshold:
                ce = max(ce, idx + 1)
                cg = max(cg, float(gains[idx]))
            else:
                if cg > self.config.min_gain_threshold:
                    merged.append((cs, ce, cg))
                cs = idx; ce = idx + 1; cg = float(gains[idx])
        if cg > self.config.min_gain_threshold:
            merged.append((cs, ce, cg))

        merged.sort(key=lambda t: t[2], reverse=True)
        return merged[: self.config.top_regions]

    def _create_refined_bins_with_budget(self, values: np.ndarray, coarse_edges: np.ndarray,
                                         regions: List[Tuple[int, int, float]]) -> np.ndarray:
        all_edges = set(coarse_edges.tolist())
        budget = self.config.max_total_bins - len(all_edges)
        if budget <= 0:
            return np.array(sorted(all_edges), dtype=np.float64)

        total_gain = float(sum(g for _, _, g in regions))
        if total_gain <= 0.0:
            return np.array(sorted(all_edges), dtype=np.float64)

        for s, e, g in regions:
            if budget <= 0:
                break
            share = max(4, int(budget * (g / total_gain)))
            # region bounds in value-space
            lo = coarse_edges[s]
            hi = coarse_edges[min(e + 1, coarse_edges.shape[0] - 1)]
            mask = (values >= lo) & (values <= hi)
            region_vals = values[mask]
            if region_vals.shape[0] >= self.config.min_region_samples:
                region_edges = self._create_uniform_bins(region_vals, share)
                new_edges = [v for v in region_edges.tolist() if v not in all_edges]
                if new_edges:
                    for v in new_edges:
                        all_edges.add(v)
                    budget -= len(new_edges)

        return np.array(sorted(all_edges), dtype=np.float64)


class AdaptiveMultiLevelBinner(MultiLevelBinner):
    def create_adaptive_bins(self, feature_values: np.ndarray, gradients: np.ndarray,
                             hessians: np.ndarray, feature_idx: int = None,
                             lambda_reg: float = 1.0, gamma: float = 0.0):
        stats = self._analyze_feature(feature_values, gradients)
        original = self.config
        self.config = self._adapt_config(stats)
        try:
            return super().create_adaptive_bins(feature_values, gradients, hessians, lambda_reg, gamma)
        finally:
            self.config = original

    def _analyze_feature(self, vals: np.ndarray, grads: np.ndarray) -> dict:
        finite = np.isfinite(vals)
        v = vals[finite]
        g = grads[finite]
        if v.shape[0] < 10:
            return {'n_unique': v.shape[0], 'missing_rate': 1.0 - v.shape[0] / vals.shape[0], 'correlation': 0.0, 'skewness': 0.0}
        nuniq = int(np.unique(v).shape[0])
        miss = 1.0 - v.shape[0] / vals.shape[0]
        corr = 0.0
        if np.std(v) > 1e-8 and np.std(g) > 1e-8:
            corr = float(np.corrcoef(v, g)[0, 1])
        skew = 0.0
        if v.shape[0] > 2:
            m = float(np.mean(v)); s = float(np.std(v))
            if s > 1e-8:
                skew = float(np.mean(((v - m) / s) ** 3))
        return {'n_unique': nuniq, 'missing_rate': miss, 'correlation': corr, 'skewness': skew}

    def _adapt_config(self, st: dict) -> BinningConfig:
        cfg = BinningConfig()
        if st['n_unique'] < 20:
            cfg.coarse_bins = min(16, st['n_unique'])
            cfg.max_total_bins = max(4, min(32, st['n_unique'] * 2))
            cfg.top_regions = 1
            cfg.min_gain_threshold = 0.01
        elif abs(st.get('correlation', 0.0)) > 0.3:
            cfg.coarse_bins = 24
            cfg.max_total_bins = 96
            cfg.top_regions = 4
            cfg.min_gain_threshold = 0.05
        elif abs(st.get('skewness', 0.0)) > 2.0:
            cfg.coarse_bins = 20
            cfg.max_total_bins = 80
            cfg.top_regions = 2
            cfg.min_gain_threshold = 0.1
        else:
            cfg.coarse_bins = 32
            cfg.max_total_bins = 128
            cfg.top_regions = 3
            cfg.min_gain_threshold = 0.1
        return cfg
