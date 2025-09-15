# gradient_histogram_system.py
# Unified gradient-aware histogram system combining sophisticated binning with clean API

from __future__ import annotations

import copy
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

# ---- New: tiny view type + helpers (zero-copy) ----
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numba import njit, prange

# ============================================================================
# Numba kernels from both systems
# ============================================================================

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


# ============================================================================
# Quantile sketching for approximate method
# ============================================================================

@njit(fastmath=True, cache=True)
def _quantile_sketch_merge(
    values1: np.ndarray, weights1: np.ndarray,
    values2: np.ndarray, weights2: np.ndarray,
    eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge two quantile sketches maintaining eps-approximation guarantee"""
    # Simple merge-and-compress approach
    all_values = np.concatenate([values1, values2])
    all_weights = np.concatenate([weights1, weights2])
    
    if len(all_values) == 0:
        return np.array([0.0]), np.array([1.0])
    
    # Sort by values
    order = np.argsort(all_values)
    sorted_vals = all_values[order]
    sorted_weights = all_weights[order]
    
    # Compress to maintain sketch size bounds
    total_weight = np.sum(sorted_weights)
    if total_weight <= 0:
        return np.array([0.0]), np.array([1.0])
    
    # Target number of quantiles based on eps
    target_size = max(10, int(1.0 / eps))
    if len(sorted_vals) <= target_size:
        return sorted_vals, sorted_weights
    
    # Sample quantiles uniformly
    cumsum = _kahan_cumsum(sorted_weights)
    quantiles = np.linspace(0, total_weight, target_size)
    
    sketch_vals = np.empty(target_size, dtype=np.float64)
    sketch_weights = np.empty(target_size, dtype=np.float64)
    
    j = 0
    for i in range(target_size):
        target = quantiles[i]
        while j < len(cumsum) - 1 and cumsum[j] < target:
            j += 1
        sketch_vals[i] = sorted_vals[j]
        if i == 0:
            sketch_weights[i] = cumsum[j]
        else:
            sketch_weights[i] = cumsum[j] - cumsum[j-1] if j > 0 else cumsum[j]
    
    return sketch_vals, sketch_weights




@njit(fastmath=True, cache=True) 
def _build_quantile_sketch(
    values: np.ndarray, 
    weights: np.ndarray, 
    eps: float,
    max_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a quantile sketch from weighted samples"""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.array([0.0]), np.array([1.0])
    
    vals = values[mask]
    wts = weights[mask]
    
    if len(vals) <= max_size:
        # Small enough - use exact
        order = np.argsort(vals)
        return vals[order], wts[order]
    
    # Build sketch by sampling
    target_samples = min(max_size, max(50, int(2.0 / eps)))
    step = len(vals) // target_samples
    
    if step <= 1:
        order = np.argsort(vals)
        return vals[order], wts[order]
    
    # Systematic sampling
    indices = np.arange(0, len(vals), step)[:target_samples]
    sampled_vals = vals[indices]
    sampled_wts = wts[indices]
    
    order = np.argsort(sampled_vals)
    return sampled_vals[order], sampled_wts[order]
    """Weighted quantile edges with density-awareness"""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.array([0.0, 1.0], np.float64)

    v = values[mask].astype(np.float64)
    w = weights[mask].astype(np.float64)

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

    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([vs[0], vs[-1]], np.float64)
    return edges


@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def _build_histograms_numba(
    X: np.ndarray,  # (n, p) 
    gradients: np.ndarray,  # (n,)
    hessians: np.ndarray,   # (n,)
    edges_flat: np.ndarray,  # (p, max_edges)
    uniform_flags: np.ndarray,  # (p,) uint8
    uniform_lo: np.ndarray,     # (p,) 
    uniform_widths: np.ndarray, # (p,)
    n_bins_per_feature: np.ndarray,  # (p,) int32
    missing_bin_id: int,
    total_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    n_samples, n_features = X.shape
    grad_hist = np.zeros((n_features, total_bins), dtype=np.float64)
    hess_hist = np.zeros((n_features, total_bins), dtype=np.float64)
    
    for j in prange(n_features):
        n_bins = n_bins_per_feature[j]
        is_uniform = uniform_flags[j] == 1
        
        for i in range(n_samples):
            value = X[i, j]
            
            # Handle missing values
            if not np.isfinite(value):
                if missing_bin_id >= 0:
                    bin_idx = missing_bin_id
                else:
                    continue
            else:
                # Find bin index
                if is_uniform:
                    # O(1) uniform binning
                    lo = uniform_lo[j]
                    width = uniform_widths[j]
                    bin_idx = int((value - lo) / width)
                    bin_idx = max(0, min(bin_idx, n_bins - 1))
                else:
                    # Binary search
                    edges = edges_flat[j]
                    bin_idx = 0
                    for k in range(1, n_bins + 1):
                        if value < edges[k]:
                            break
                        bin_idx = k
                    bin_idx = min(bin_idx, n_bins - 1)
            
            # Accumulate
            grad_hist[j, bin_idx] += gradients[i]
            hess_hist[j, bin_idx] += hessians[i]
    
    return grad_hist, hess_hist


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HistogramConfig:
    """Unified configuration combining both systems"""
    # Basic histogram settings
    method: str = "grad_aware"  # "hist", "approx", "grad_aware", "adaptive"
    max_bins: int = 256
    missing_bin: bool = True
    dtype: np.dtype = np.uint8
    
    # Gradient-aware binning settings
    coarse_bins: int = 64
    min_child_weight: float = 1e-3
    min_gain_threshold: float = 1e-2
    categorical_threshold: int = 32
    use_density_aware_binning: bool = True
    
    # Approximate method settings
    sketch_eps: float = 0.1  # Quantile sketch precision (lower = more accurate)
    subsample_ratio: float = 0.3  # Fraction of data to sample
    min_sketch_size: int = 10000  # Minimum samples for sketching
    global_sketching: bool = True  # Use global vs local sketching
    sketch_method: str = "weighted_quantile"  # "uniform", "weighted_quantile", "adaptive"
    
    # Node refinement settings
    node_refinement_threshold: int = 1000
    max_refinement_depth: int = 8
    refinement_feature_fraction: float = 0.3
    refinement_min_correlation: float = 0.10
    
    # Categorical settings
    use_optimal_categorical_encoding: bool = True
    cat_min_count: int = 20
    cat_prior_strength: float = 50.0
    
    # Performance settings
    use_parallel: bool = False
    max_workers: int = 16
    cache_size: int = 1024
    
    # Regularization
    lambda_reg: float = 1.0
    gamma: float = 0.0
    eps: float = 1e-12
    random_state: Optional[int] = 42
    
    @property
    def total_bins(self) -> int:
        return self.max_bins + (1 if self.missing_bin else 0)
    
    @property
    def missing_bin_id(self) -> int:
        return self.max_bins if self.missing_bin else -1


# ============================================================================
# Feature bins with gradient-aware logic
# ============================================================================

@dataclass 
class GradientFeatureBins:
    """Enhanced feature bins with gradient-aware metadata"""
    edges: np.ndarray
    is_uniform: bool = False
    strategy: str = "uniform"  # "uniform", "categorical", "gradient_aware", "adaptive"
    complexity: float = 0.0
    feature_stats: Optional[Tuple] = None
    
    # Uniform bin optimization params
    _lo: float = 0.0
    _width: float = 1.0
    
    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.float64)
        if self.edges.size < 2:
            self.edges = np.array([0.0, 1.0])
        self._check_uniform()
    
    def _check_uniform(self, tol: float = 1e-9):
        """Check if bins are uniform for O(1) binning optimization"""
        if len(self.edges) < 3:
            self.is_uniform = True
            self._lo = float(self.edges[0])
            self._width = float(self.edges[1] - self.edges[0])
            return
            
        widths = np.diff(self.edges)
        mean_width = np.mean(widths)
        if mean_width > 0 and np.max(np.abs(widths - mean_width)) <= tol * mean_width:
            self.is_uniform = True
            self._lo = float(self.edges[0])
            self._width = float(mean_width)
        else:
            self.is_uniform = False
    
    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1


# ============================================================================
# Binning strategies with gradient awareness
# ============================================================================

class BinningStrategy(ABC):
    """Abstract base for binning strategies"""
    
    @abstractmethod
    def create_bins(
        self,
        values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        config: HistogramConfig,
    ) -> GradientFeatureBins:
        pass


class SimpleBinning(BinningStrategy):
    """Simple uniform/quantile binning for baseline comparison"""
    
    def create_bins(
        self,
        values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        config: HistogramConfig,
    ) -> GradientFeatureBins:
        
        v = values[np.isfinite(values)]
        if len(v) == 0:
            edges = np.array([0.0, 1.0])
        elif len(np.unique(v)) <= config.max_bins:
            unique_vals = np.unique(v)
            edges = np.concatenate([
                [unique_vals[0] - 1e-6],
                (unique_vals[:-1] + unique_vals[1:]) / 2,
                [unique_vals[-1] + 1e-6]
            ])
        else:
            quantiles = np.linspace(0, 1, config.max_bins + 1)
            edges = np.quantile(v, quantiles)
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([v.min() - 1e-6, v.max() + 1e-6])
        
        return GradientFeatureBins(edges=edges, strategy="simple")


class GradientAwareBinning(BinningStrategy):
    """Sophisticated gradient-aware binning from your GradientBinner"""
    
    def __init__(self):
        self._rng = np.random.default_rng(42)
    
    def create_bins(
        self,
        values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        config: HistogramConfig,
    ) -> GradientFeatureBins:
        
        v = np.asarray(values, dtype=np.float64)
        g = np.asarray(gradients, dtype=np.float64)
        h = np.asarray(hessians, dtype=np.float64)
        
        # Compute feature statistics and complexity
        stats = self._compute_feature_stats(v, g, h)
        complexity = self._estimate_complexity(v, g, h)
        strategy = self._select_strategy(stats, complexity, config)
        
        if strategy == "categorical":
            if config.use_optimal_categorical_encoding:
                edges = self._cat_target_mean_edges(v, g, h, config)
            else:
                edges = self._categorical_edges(v)
        elif strategy == "uniform":
            edges = self._uniform_edges(v, config.coarse_bins)
        else:  # "gradient_aware"
            edges = self._gradient_aware_edges(v, g, h, complexity, config)
        
        return GradientFeatureBins(
            edges=edges,
            strategy=strategy,
            complexity=complexity,
            feature_stats=stats,
        )
    
    def _gradient_aware_edges(
        self,
        v: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        complexity: float,
        config: HistogramConfig,
    ) -> np.ndarray:
        """Core gradient-aware edge creation"""
        # Adaptive coarse bins based on complexity
        scale = max(0.5, min(2.0, complexity))
        n_coarse = int(min(config.coarse_bins, max(16, int(32 * scale))))
        
        # Create coarse bins using weighted quantiles
        coarse = self._coarse_from_quantiles(v, n_coarse)
        if config.use_density_aware_binning:
            coarse = self._density_refined_edges(v, np.maximum(h, 1e-12), n_coarse)
        
        return coarse
    
    def _select_strategy(self, stats: Tuple, complexity: float, config: HistogramConfig) -> str:
        corr_g, skew, _miss, n_unique, corr_h, _sv = stats
        if n_unique <= config.categorical_threshold:
            return "categorical"
        if (complexity < 0.3) and (abs(corr_g) < 0.1) and (abs(corr_h) < 0.1) and (abs(skew) < 1.0):
            return "uniform"
        return "gradient_aware"
    
    def _compute_feature_stats(self, v: np.ndarray, g: np.ndarray, h: np.ndarray) -> Tuple[float, float, float, int, float, float]:
        mask = np.isfinite(v) & np.isfinite(g) & np.isfinite(h)
        if not np.any(mask):
            return 0.0, 0.0, 1.0, 0, 0.0, 0.0
        
        vv, gg, hh = v[mask], g[mask], h[mask]
        n_unique = int(np.unique(vv).size)
        miss_rate = 1.0 - (vv.size / max(1, v.size))
        
        def _corr(a, b) -> float:
            sa, sb = np.std(a), np.std(b)
            if sa <= 1e-12 or sb <= 1e-12:
                return 0.0
            try:
                return float(np.corrcoef(a, b)[0, 1])
            except Exception:
                return 0.0
        
        corr_g = abs(_corr(vv, gg))
        corr_h = abs(_corr(vv, hh))
        sv = float(np.std(vv))
        
        skew = 0.0
        if sv > 1e-12:
            mu = float(np.mean(vv))
            skew = float(np.mean(((vv - mu) / sv) ** 3))
        
        return corr_g, skew, miss_rate, n_unique, corr_h, sv
    
    def _estimate_complexity(self, v: np.ndarray, g: np.ndarray, h: np.ndarray) -> float:
        mask = np.isfinite(v) & np.isfinite(g) & np.isfinite(h)
        vv, gg, hh = v[mask], g[mask], h[mask]
        
        if vv.size < 10:
            return 0.3
        
        gv = float(np.sqrt(max(0.0, np.var(gg)))) if gg.size > 1 else 0.0
        hv = float(np.sqrt(max(0.0, np.var(hh)))) if hh.size > 1 else 0.0
        
        # Local variation in gradients
        sidx = np.argsort(vv)
        gs = gg[sidx]
        m = max(0, gs.size - 2)
        if m > 0:
            local = float(np.sum(np.abs(gs[2:] - 2.0 * gs[1:-1] + gs[:-2]))) / m
        else:
            local = 0.0
        
        smooth = 1.0 / (1.0 + local)
        comp = 0.45 * gv + 0.35 * hv + 0.20 * (1.0 - smooth)
        return float(np.clip(comp, 0.1, 2.0))
    
    def _density_refined_edges(self, v: np.ndarray, w: np.ndarray, k_bins: int) -> np.ndarray:
        return _adaptive_quantile_edges(v, w, k_bins, density_aware=True)
    
    @staticmethod
    def _uniform_edges(v: np.ndarray, n_bins: int) -> np.ndarray:
        fv = v[np.isfinite(v)]
        if fv.size == 0:
            return np.array([0.0, 1.0])
        lo, hi = float(np.min(fv)), float(np.max(fv))
        if lo == hi:
            return np.array([lo, hi + 1e-12])
        return np.linspace(lo, hi, n_bins + 1)
    
    @staticmethod
    def _categorical_edges(v: np.ndarray) -> np.ndarray:
        fv = v[np.isfinite(v)]
        if fv.size == 0:
            return np.array([0.0, 1.0])
        u = np.unique(fv)
        if u.size == 1:
            x = float(u[0])
            return np.array([x - 1e-12, x + 1e-12])
        edges = np.empty(u.size + 1)
        edges[0] = float(u[0]) - 1e-12
        edges[-1] = float(u[-1]) + 1e-12
        for i in range(1, u.size):
            edges[i] = 0.5 * (u[i - 1] + u[i])
        return edges
    
    @staticmethod
    def _coarse_from_quantiles(v: np.ndarray, n_bins: int) -> np.ndarray:
        fv = v[np.isfinite(v)]
        if fv.size == 0:
            return np.array([0.0, 1.0])
        u = np.unique(fv)
        if u.size <= n_bins:
            return u.astype(np.float64)
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        return np.unique(np.quantile(fv, qs)).astype(np.float64)
    
    def _cat_target_mean_edges(self, v: np.ndarray, g: np.ndarray, h: np.ndarray, config: HistogramConfig) -> np.ndarray:
        # Simplified categorical encoding - could expand with your full logic
        return self._categorical_edges(v)


class ApproximateBinning(BinningStrategy):
    """Approximate binning using quantile sketching (like XGBoost 'approx' method)"""
    
    def __init__(self, rng_seed: Optional[int] = None):
        self._rng = np.random.default_rng(rng_seed)
    
    def create_bins(
        self,
        values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        config: HistogramConfig,
    ) -> GradientFeatureBins:
        
        v = np.asarray(values, dtype=np.float64)
        g = np.asarray(gradients, dtype=np.float64)
        h = np.asarray(hessians, dtype=np.float64)
        
        # Filter finite values
        mask = np.isfinite(v) & np.isfinite(g) & np.isfinite(h)
        if not np.any(mask):
            return GradientFeatureBins(np.array([0.0, 1.0]), strategy="approx_empty")
        
        v_clean = v[mask]
        g_clean = g[mask]
        h_clean = h[mask]
        
        # Check if we should use exact method instead
        if len(v_clean) < config.min_sketch_size:
            # Too small for approximation - use simple binning
            return SimpleBinning().create_bins(values, gradients, hessians, config)
        
        # Determine sketch method and create edges
        if config.sketch_method == "uniform":
            edges = self._uniform_sketch_edges(v_clean, config)
        elif config.sketch_method == "weighted_quantile":
            edges = self._weighted_quantile_sketch_edges(v_clean, h_clean, config)
        elif config.sketch_method == "adaptive":
            edges = self._adaptive_sketch_edges(v_clean, g_clean, h_clean, config)
        else:
            # Fallback to weighted quantile
            edges = self._weighted_quantile_sketch_edges(v_clean, h_clean, config)
        
        return GradientFeatureBins(edges=edges, strategy="approx", complexity=0.5)
    
    def _uniform_sketch_edges(self, values: np.ndarray, config: HistogramConfig) -> np.ndarray:
        """Simple uniform sampling approximation"""
        n_samples = len(values)
        sample_size = max(1000, int(n_samples * config.subsample_ratio))
        
        if sample_size >= n_samples:
            # Sample is too large - use all data
            return self._quantile_edges_exact(values, config.max_bins)
        
        # Random sampling
        indices = self._rng.choice(n_samples, sample_size, replace=False)
        sample_values = values[indices]
        
        return self._quantile_edges_exact(sample_values, config.max_bins)
    
    def _weighted_quantile_sketch_edges(
        self, 
        values: np.ndarray, 
        weights: np.ndarray, 
        config: HistogramConfig
    ) -> np.ndarray:
        """Weighted quantile sketching using hessians as weights"""
        # Build quantile sketch
        sketch_vals, sketch_weights = _build_quantile_sketch(
            values, np.maximum(weights, 1e-12), config.sketch_eps
        )
        
        # Convert sketch to edges
        if len(sketch_vals) <= config.max_bins:
            # Sketch is small enough - use directly
            unique_vals = np.unique(sketch_vals)
            if len(unique_vals) <= 1:
                val = float(unique_vals[0]) if len(unique_vals) == 1 else 0.0
                return np.array([val - 1e-6, val + 1e-6])
            
            # Create edges between unique values
            edges = np.empty(len(unique_vals) + 1)
            edges[0] = unique_vals[0] - 1e-6
            edges[-1] = unique_vals[-1] + 1e-6
            for i in range(1, len(unique_vals)):
                edges[i] = 0.5 * (unique_vals[i-1] + unique_vals[i])
            return edges
        
        # Use sketch to create quantile-based edges
        total_weight = np.sum(sketch_weights)
        if total_weight <= 0:
            return np.array([values.min(), values.max()])
        
        # Cumulative weights for quantile calculation
        cum_weights = _kahan_cumsum(sketch_weights) / total_weight
        
        # Target quantiles
        quantiles = np.linspace(0, 1, config.max_bins + 1)
        edges = np.interp(quantiles, cum_weights, sketch_vals)
        
        return np.unique(edges)
    
    def _adaptive_sketch_edges(
        self, 
        values: np.ndarray, 
        gradients: np.ndarray, 
        hessians: np.ndarray, 
        config: HistogramConfig
    ) -> np.ndarray:
        """Adaptive sketching based on gradient/hessian patterns"""
        # Use both gradient variance and hessian magnitude for adaptive sampling
        abs_gradients = np.abs(gradients)
        weights = 0.7 * abs_gradients + 0.3 * np.maximum(hessians, 1e-12)
        
        # Importance-based sampling
        total_weight = np.sum(weights)
        if total_weight <= 0:
            return self._uniform_sketch_edges(values, config)
        
        probabilities = weights / total_weight
        n_samples = len(values)
        sample_size = max(1000, int(n_samples * config.subsample_ratio))
        sample_size = min(sample_size, n_samples)
        
        # Weighted sampling without replacement (approximated)
        if sample_size >= n_samples:
            selected_values = values
            selected_weights = weights
        else:
            # Sample with replacement based on probabilities
            indices = self._rng.choice(n_samples, sample_size, replace=True, p=probabilities)
            selected_values = values[indices]
            selected_weights = weights[indices]
        
        # Build weighted quantile sketch
        return self._weighted_quantile_sketch_edges(selected_values, selected_weights, config)
    
    @staticmethod
    def _quantile_edges_exact(values: np.ndarray, max_bins: int) -> np.ndarray:
        """Create exact quantile edges for small datasets"""
        if len(values) == 0:
            return np.array([0.0, 1.0])
        
        unique_vals = np.unique(values)
        if len(unique_vals) <= max_bins:
            # Few unique values - use them directly
            if len(unique_vals) == 1:
                val = float(unique_vals[0])
                return np.array([val - 1e-6, val + 1e-6])
            
            edges = np.empty(len(unique_vals) + 1)
            edges[0] = unique_vals[0] - 1e-6
            edges[-1] = unique_vals[-1] + 1e-6
            for i in range(1, len(unique_vals)):
                edges[i] = 0.5 * (unique_vals[i-1] + unique_vals[i])
            return edges
        
        # Use quantiles
        quantiles = np.linspace(0, 1, max_bins + 1)
        edges = np.quantile(values, quantiles)
        return np.unique(edges)
    
# ============================================================================
# Pre-binning for speed optimization
# ============================================================================

@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def _prebin_matrix_numba(
    X: np.ndarray,  # (n_samples, n_features)
    edges_flat: np.ndarray,  # (n_features, max_edges)
    uniform_flags: np.ndarray,  # (n_features,) uint8
    uniform_lo: np.ndarray,     # (n_features,)
    uniform_widths: np.ndarray, # (n_features,)
    n_bins_per_feature: np.ndarray,  # (n_features,) int32
    missing_bin_id: int,
) -> np.ndarray:
    """Pre-bin entire dataset into bin indices for fast histogram building"""
    n_samples, n_features = X.shape
    bin_indices = np.empty((n_samples, n_features), dtype=np.int32)
    
    for j in prange(n_features):
        n_bins = n_bins_per_feature[j]
        is_uniform = uniform_flags[j] == 1
        
        for i in range(n_samples):
            value = X[i, j]
            
            if not np.isfinite(value):
                if missing_bin_id >= 0:
                    bin_indices[i, j] = missing_bin_id
                else:
                    bin_indices[i, j] = 0  # fallback
            else:
                if is_uniform:
                    # O(1) uniform binning
                    lo = uniform_lo[j]
                    width = uniform_widths[j]
                    bin_idx = int((value - lo) / width)
                    bin_idx = max(0, min(bin_idx, n_bins - 1))
                    bin_indices[i, j] = bin_idx
                else:
                    # Binary search
                    edges = edges_flat[j]
                    bin_idx = 0
                    for k in range(1, n_bins + 1):
                        if value < edges[k]:
                            break
                        bin_idx = k
                    bin_idx = min(bin_idx, n_bins - 1)
                    bin_indices[i, j] = bin_idx
    
    return bin_indices


@njit(cache=True, nogil=True, fastmath=False, parallel=True)  
def _build_histograms_from_indices(
    bin_indices: np.ndarray,  # (n_samples, n_features) int32
    gradients: np.ndarray,    # (n_samples,)
    hessians: np.ndarray,     # (n_samples,)
    total_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast histogram building from pre-computed bin indices"""
    n_samples, n_features = bin_indices.shape
    grad_hist = np.zeros((n_features, total_bins), dtype=np.float64)
    hess_hist = np.zeros((n_features, total_bins), dtype=np.float64)
    
    for j in prange(n_features):
        for i in range(n_samples):
            bin_idx = bin_indices[i, j]
            if 0 <= bin_idx < total_bins:
                grad_hist[j, bin_idx] += gradients[i]
                hess_hist[j, bin_idx] += hessians[i]
    
    return grad_hist, hess_hist


# ============================================================================
# Main unified system
# ============================================================================


@dataclass(frozen=True)
class HistogramView:
    """A lightweight view over bin edges and pre-binned codes for a (rows, features) slice."""
    bin_edges: List[np.ndarray]          # len = n_features_view, each (n_bins_j+1,)
    codes: np.ndarray                    # shape (n_rows_view, n_features_view), int32

    @property
    def n_rows(self) -> int:
        return int(self.codes.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.codes.shape[1])

def _subset_edges(all_edges: List[np.ndarray], feature_idx: np.ndarray) -> List[np.ndarray]:
    return [all_edges[int(j)] for j in feature_idx]

class GradientHistogramSystem:
    """Unified system combining sophisticated gradient-aware binning with efficient histograms"""
    
    def __init__(self, config: Optional[HistogramConfig] = None):
        self.config = config or HistogramConfig()
        self.feature_bins: List[GradientFeatureBins] = []
        self._precomputed_indices: Optional[np.ndarray] = None  # Cached pre-binned data
        self._lock = threading.Lock()
        
        # Initialize strategy based on method
        if self.config.method in ["grad_aware", "gradient_aware", "adaptive"]:
            self.binning_strategy = GradientAwareBinning()
        elif self.config.method == "approx":
            self.binning_strategy = ApproximateBinning(self.config.random_state)
        else:  # "hist" or other simple methods
            self.binning_strategy = SimpleBinning()
        
        # Thread pool for parallel processing
        self._executor = (
            ThreadPoolExecutor(max_workers=self.config.max_workers)
            if self.config.use_parallel else None
        )
    
    def __del__(self):
        if self._executor:
            self._executor.shutdown(wait=False)
            
    def clone(
        self,
        feature_idx: np.ndarray,
        row_idx: Optional[np.ndarray] = None,
    ) -> "GradientHistogramSystem":
        """
        Create an EXACT GradientHistogramSystem instance, but restricted to
        the provided feature and row subsets. The clone has:
          - same config (deep-copied)
          - same binning strategy type
          - feature_bins subset (deep-copied)
          - pre-binned matrix sliced to (rows, features)

        After cloning, use clone.build_histograms_fast(gradients, hessians).
        """
        if self._precomputed_indices is None:
            raise ValueError("prebin_dataset(X) must be called before clone(...)")
        if not self.feature_bins:
            raise ValueError("fit_bins(...) must be called before clone(...)")

        # Normalize indices
        feat = np.asarray(feature_idx, dtype=np.int32).ravel()
        if row_idx is None:
            rows = None
            codes_slice = self._precomputed_indices[:, feat]
        else:
            rows = np.asarray(row_idx, dtype=np.int64).ravel()
            codes_slice = self._precomputed_indices[rows][:, feat]

        # New system with SAME config
        new_cfg = copy.deepcopy(self.config)
        cloned = GradientHistogramSystem(config=new_cfg)

        # Rebuild SAME strategy type (mirrors __init__ logic)
        if cloned.config.method in ["grad_aware", "gradient_aware", "adaptive"]:
            cloned.binning_strategy = GradientAwareBinning()
        elif cloned.config.method == "approx":
            cloned.binning_strategy = ApproximateBinning(cloned.config.random_state)
        else:
            cloned.binning_strategy = SimpleBinning()

        # Deep-copy ONLY selected features' bins (edges, stats, etc.)
        cloned.feature_bins = [copy.deepcopy(self.feature_bins[int(j)]) for j in feat]

        # Install sliced pre-binned matrix (ready to read)
        cloned._precomputed_indices = np.ascontiguousarray(codes_slice, dtype=np.int32)

        # Fresh lock/executor are already created by __init__; nothing else to do.
        return cloned


    def fit_bins(
        self, 
        X: np.ndarray, 
        gradients: np.ndarray, 
        hessians: np.ndarray,
    ) -> None:
        """Fit bins for all features using gradient information"""
        n_samples, n_features = X.shape
        
        if not (len(gradients) == len(hessians) == n_samples):
            raise ValueError("X, gradients, and hessians must have compatible shapes")
        
        self.feature_bins = []
        
        # Process features in parallel if enabled
        if self._executor and n_features > 1:
            futures = []
            for j in range(n_features):
                future = self._executor.submit(
                    self.binning_strategy.create_bins,
                    X[:, j], gradients, hessians, self.config
                )
                futures.append(future)
            
            for future in futures:
                try:
                    bins = future.result(timeout=30)
                    self.feature_bins.append(bins)
                except Exception as e:
                    # Fallback to simple uniform bins on error
                    values = X[:, len(self.feature_bins)]
                    fallback_bins = SimpleBinning().create_bins(values, gradients, hessians, self.config)
                    self.feature_bins.append(fallback_bins)
        else:
            # Sequential processing
            for j in range(n_features):
                bins = self.binning_strategy.create_bins(X[:, j], gradients, hessians, self.config)
                self.feature_bins.append(bins)
                
    def prebin_dataset(self, X: np.ndarray, *, inplace: bool = True) -> np.ndarray:
        """
        Pre-bin dataset. If `inplace=True`, cache into self._precomputed_indices.
        If `inplace=False`, just return the codes without touching the cache.
        """
        if not self.feature_bins:
            raise ValueError("Must call fit_bins() first")

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        n_features = len(self.feature_bins)
        max_edges = max(len(bins.edges) for bins in self.feature_bins)

        edges_flat = np.zeros((n_features, max_edges), dtype=np.float64)
        uniform_flags = np.zeros(n_features, dtype=np.uint8)
        uniform_lo = np.zeros(n_features, dtype=np.float64)
        uniform_widths = np.zeros(n_features, dtype=np.float64)
        n_bins_per_feature = np.zeros(n_features, dtype=np.int32)

        for i, bins in enumerate(self.feature_bins):
            edges_flat[i, :len(bins.edges)] = bins.edges
            uniform_flags[i] = 1 if bins.is_uniform else 0
            uniform_lo[i] = bins._lo
            uniform_widths[i] = bins._width
            n_bins_per_feature[i] = bins.n_bins

        codes = _prebin_matrix_numba(
            X_c, edges_flat, uniform_flags, uniform_lo, uniform_widths,
            n_bins_per_feature, self.config.missing_bin_id
        )

        if inplace:
            self._precomputed_indices = codes
        return codes

    # (Optional tiny helper, if you prefer a separate name)
    def prebin_view(self, X: np.ndarray) -> np.ndarray:
        return self.prebin_dataset(X, inplace=False)

    def build_histograms_fast(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ultra-fast histogram building using pre-computed bin indices.
        
        Args:
            gradients: Gradient values for samples
            hessians: Hessian values for samples  
            sample_indices: Optional subset of samples to use (for tree nodes)
        
        Returns:
            (grad_hist, hess_hist): Histograms for each feature
        """
        if self._precomputed_indices is None:
            raise ValueError("Must call prebin_dataset() first for fast histograms")
        
        g_c = np.ascontiguousarray(gradients, dtype=np.float64)
        h_c = np.ascontiguousarray(hessians, dtype=np.float64)
        
        if sample_indices is not None:
            # Use subset of samples (e.g., samples in a tree node)
            indices_subset = self._precomputed_indices[sample_indices]
            g_subset = g_c[sample_indices]
            h_subset = h_c[sample_indices]
            return _build_histograms_from_indices(indices_subset, g_subset, h_subset, self.config.total_bins)
        else:
            # Use all samples
            return _build_histograms_from_indices(self._precomputed_indices, g_c, h_c, self.config.total_bins)
    
    def hist_from_codes(
        self,
        bin_indices: np.ndarray,          # (n_rows, n_features_sel) int
        gradients: np.ndarray,            # (n_rows,)
        hessians: np.ndarray,             # (n_rows,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build histograms directly from provided bin indices
        (e.g., a slice of self._precomputed_indices[:, feature_mask]).
        """
        if bin_indices.ndim != 2:
            raise ValueError("bin_indices must be 2D (n_rows, n_features_sel)")
        n_rows, n_feat = bin_indices.shape
        if gradients.shape[0] != n_rows or hessians.shape[0] != n_rows:
            raise ValueError("bin_indices, gradients, hessians must align on rows")

        bin_c = np.ascontiguousarray(bin_indices, dtype=np.int32)
        g_c = np.ascontiguousarray(gradients, dtype=np.float64)
        h_c = np.ascontiguousarray(hessians, dtype=np.float64)

        return _build_histograms_from_indices(bin_c, g_c, h_c, self.config.total_bins)

    def build_histograms(
        self,
        X: np.ndarray,
        gradients: np.ndarray, 
        hessians: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build histograms using fitted bins (slower path - bins on-the-fly).
        Use build_histograms_fast() after prebin_dataset() for better performance.
        """
        if not self.feature_bins:
            raise ValueError("Must call fit_bins() first")
        
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        g_c = np.ascontiguousarray(gradients, dtype=np.float64)
        h_c = np.ascontiguousarray(hessians, dtype=np.float64)
        
        # Prepare data for numba kernel (same as before)
        n_features = len(self.feature_bins)
        max_edges = max(len(bins.edges) for bins in self.feature_bins)
        
        edges_flat = np.zeros((n_features, max_edges), dtype=np.float64)
        uniform_flags = np.zeros(n_features, dtype=np.uint8)
        uniform_lo = np.zeros(n_features, dtype=np.float64)
        uniform_widths = np.zeros(n_features, dtype=np.float64)
        n_bins_per_feature = np.zeros(n_features, dtype=np.int32)
        
        for i, bins in enumerate(self.feature_bins):
            edges_flat[i, :len(bins.edges)] = bins.edges
            uniform_flags[i] = 1 if bins.is_uniform else 0
            uniform_lo[i] = bins._lo
            uniform_widths[i] = bins._width
            n_bins_per_feature[i] = bins.n_bins
        
        return _build_histograms_numba(
            X_c, g_c, h_c,
            edges_flat, uniform_flags, uniform_lo, uniform_widths, n_bins_per_feature,
            self.config.missing_bin_id, self.config.total_bins
        )
    
    def get_feature_info(self, feature_idx: int) -> Dict:
        """Get detailed information about a feature's binning"""
        if feature_idx >= len(self.feature_bins):
            raise IndexError(f"Feature {feature_idx} not found")
        
        bins = self.feature_bins[feature_idx]
        return {
            "edges": bins.edges,
            "n_bins": bins.n_bins,
            "is_uniform": bins.is_uniform,
            "strategy": bins.strategy,
            "complexity": bins.complexity,
            "feature_stats": bins.feature_stats,
        }
    
    def get_all_strategies(self) -> List[str]:
        """Get binning strategy used for each feature"""
        return [bins.strategy for bins in self.feature_bins]
