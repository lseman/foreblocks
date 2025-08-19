import os
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numba import jit
from scipy import fft
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder,
    QuantileTransformer,
)
from sklearn.utils.validation import check_is_fitted

from foretools.aux.adaptive_mi import AdaptiveMI
from foretools.aux.hsic import HSIC

# Type hints for better code clarity
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]
ModelLike = Any


@dataclass
class FeatureConfig:
    """Configuration class for feature engineering parameters."""
    
    # Core settings
    task: str = "classification"
    random_state: int = 42
    corr_threshold: float = 0.95
    rare_threshold: float = 0.01
    
    # Feature creation flags
    create_interactions: bool = True
    create_math_features: bool = True
    create_binning: bool = True
    create_clustering: bool = True
    create_statistical: bool = True
    create_fourier: bool = False
    use_autoencoder: bool = True
    
    # Feature limits
    max_interactions: int = 50
    max_selected_interactions: int = 20
    n_bins: int = 5
    n_clusters: int = 8
    n_fourier_terms: int = 3
    
    # Selection parameters
    use_quantile_transform: bool = True
    target_encode_threshold: int = 10
    mi_threshold: float = 0.001
    shap_threshold: float = 0.001
    
    # Autoencoder settings
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 64
    autoencoder_lr: float = 1e-3
    autoencoder_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder_latent_ratio: float = 0.25

    # Interaction-specific settings
    create_interactions: bool = True
    max_pairs_screen: int = 200               # pairs to score in stage-1
    max_interactions: int = 200               # hard cap of generated features
    max_selected_interactions: int = 64       # final kept features
    scorer: str = "mi"                        # {"mi", "hsic"}
    task: str = "regression"                  # {"regression","classification"}
    n_splits: int = 5                         # CV splits for stability selection
    min_selected_per_fold: int = 20           # per-fold top-k before aggregating
    importance_agg: str = "median"            # {"mean","median","max"}
    random_state: int = 42
    max_rows_score: int = 50000               # speed guard for scoring
    model_backend: str = "xgb"                # {"xgb","lgb"}
    device: Optional[str] = None              # "cuda" for xgb if desired
    # robust ops toggles
    include_sum: bool = True
    include_diff: bool = True
    include_prod: bool = True
    include_ratio: bool = True
    include_norm_ratio: bool = True
    include_minmax: bool = True
    # screening knobs
    pair_corr_with_y: bool = True             # use |corr(feature, y)| to order cols
    pair_max_per_feature: int = 32            # limit partners per anchor to reduce blow-up
    corr_avoid_redundancy: float = 0.995      # skip pairs with |corr(a,b)| above this
    # safety
    epsilon: float = 1e-8
    dtype_out: str = "float32"
     # speed guard for scoring


class BaseFeatureTransformer(ABC):
    """Abstract base class for feature transformers."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseFeatureTransformer':
        """Fit the transformer."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class DateTimeTransformer(BaseFeatureTransformer):
    """Handles datetime feature extraction with cyclical encoding."""
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DateTimeTransformer':
        self.datetime_cols_ = X.select_dtypes(include=['datetime64']).columns.tolist()
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.datetime_cols_:
            return pd.DataFrame(index=X.index)
        
        features = {}
        for col in self.datetime_cols_:
            if col not in X.columns:
                continue
                
            dt = pd.to_datetime(X[col], errors='coerce')
            
            # Basic features
            features.update({
                f"{col}_year": dt.dt.year,
                f"{col}_month": dt.dt.month,
                f"{col}_day": dt.dt.day,
                f"{col}_weekday": dt.dt.weekday,
                f"{col}_quarter": dt.dt.quarter,
                f"{col}_is_weekend": (dt.dt.weekday >= 5).astype(int),
                f"{col}_hour": dt.dt.hour,
                f"{col}_elapsed_days": (dt - dt.min()).dt.days,
            })
            
            # Cyclical encoding
            features.update({
                f"{col}_month_sin": np.sin(2 * np.pi * dt.dt.month / 12),
                f"{col}_month_cos": np.cos(2 * np.pi * dt.dt.month / 12),
                f"{col}_weekday_sin": np.sin(2 * np.pi * dt.dt.weekday / 7),
                f"{col}_weekday_cos": np.cos(2 * np.pi * dt.dt.weekday / 7),
                f"{col}_hour_sin": np.sin(2 * np.pi * dt.dt.hour / 24),
                f"{col}_hour_cos": np.cos(2 * np.pi * dt.dt.hour / 24),
            })
        
        return pd.DataFrame(features, index=X.index)


class MathematicalTransformer(BaseFeatureTransformer):
    """Creates mathematical transformations of numerical features."""
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MathematicalTransformer':
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.valid_transforms_ = {}
        
        for col in self.numerical_cols_:
            if col not in X.columns or X[col].isna().mean() > 0.5:
                continue
                
            data = X[col]
            transforms = []
            
            # Log transform for positive values
            if (data > 0).any():
                log_data = np.log1p(np.maximum(data, 0))
                if log_data.var() > 1e-6:
                    transforms.append('log')
            
            # Square root for non-negative values
            if (data >= 0).any():
                sqrt_data = np.sqrt(np.maximum(data, 0))
                if sqrt_data.var() > 1e-6:
                    transforms.append('sqrt')
            
            # Reciprocal (careful with zero division)
            if (np.abs(data) > 1e-6).any():
                recip_data = 1 / data.replace(0, np.nan)
                if recip_data.var() > 1e-6:
                    transforms.append('reciprocal')
            
            if transforms:
                self.valid_transforms_[col] = transforms
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.create_math_features:
            return pd.DataFrame(index=X.index)
        
        features = {}
        for col, transforms in self.valid_transforms_.items():
            if col not in X.columns:
                continue
                
            data = X[col]
            
            if 'log' in transforms:
                features[f"{col}_log"] = np.log1p(np.maximum(data, 0))
            if 'sqrt' in transforms:
                features[f"{col}_sqrt"] = np.sqrt(np.maximum(data, 0))
            if 'reciprocal' in transforms:
                features[f"{col}_reciprocal"] = 1 / data.replace(0, np.nan)
        
        return pd.DataFrame(features, index=X.index)

# -----------------------
# SOTA Interaction Transformer
# -----------------------
# Optimized mathematical operations with Numba
@jit(nopython=True, fastmath=True)
def _safe_ratio_vectorized(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Vectorized safe ratio with epsilon protection."""
    denominator = np.where(np.abs(b) < eps, np.sign(b) * eps, b)
    return a / denominator

@jit(nopython=True, fastmath=True)
def _safe_norm_ratio_vectorized(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Vectorized normalized ratio: (a-b)/(a+b+eps)."""
    denominator = np.abs(a) + np.abs(b) + eps
    return (a - b) / denominator

@jit(nopython=True, fastmath=True)
def _compute_all_interactions(a: np.ndarray, b: np.ndarray, ops: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute all interaction types efficiently."""
    n = len(a)
    n_ops = len(ops)
    result = np.empty((n_ops, n), dtype=np.float32)
    
    for i in range(n_ops):
        op = ops[i]
        if op == 0:  # sum
            result[i] = a + b
        elif op == 1:  # diff
            result[i] = a - b
        elif op == 2:  # product
            result[i] = a * b
        elif op == 3:  # ratio
            result[i] = _safe_ratio_vectorized(a, b, eps)
        elif op == 4:  # norm_ratio
            result[i] = _safe_norm_ratio_vectorized(a, b, eps)
        elif op == 5:  # min
            result[i] = np.minimum(a, b)
        elif op == 6:  # max
            result[i] = np.maximum(a, b)
    
    return result



class InteractionTransformer(BaseFeatureTransformer):
    """
    High-performance interaction feature generator with modern optimizations:
    
    Key improvements:
    1. Numba JIT compilation for 10-100x speedup on math operations
    2. Vectorized batch processing instead of loop-based
    3. Memory-optimized operations
    4. Smarter feature selection using custom MI implementations
    5. Clean integration with AdaptiveMI and HSIC scorers
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.numerical_cols_: List[str] = []
        self.selected_interactions_: List[str] = []
        self.interaction_pairs_: List[Tuple[str, str]] = []
        self.is_fitted = False
        
        # Initialize custom MI scorers
        self.hsic_scorer = HSIC(
            kernel_x="rbf",
            kernel_y="rbf", 
            estimator="biased",
            normalize=True,
            use_numba=True,
        )

        self.ami_scorer = AdaptiveMI(
            subsample=min(getattr(self.config, "max_rows_score", 2000), 2000),
            spearman_gate=getattr(self.config, "candidate_screen_abs_spearman", 0.05),
            min_overlap=50,
            ks=(3, 5, 10),
            n_bins=getattr(self.config, "mi_bins", 16),
            random_state=self.config.random_state,
        )
        
        # Pre-compile operation codes for Numba
        self.op_codes = self._get_operation_codes()
        
        # Cache for expensive computations
        self._correlation_cache: Dict[Tuple[str, str], float] = {}

    def _clean_interaction_feature(self, feature: np.ndarray) -> np.ndarray:
        """
        Clean interaction feature by handling infinities and extreme values.
        """
        # Replace infinities with NaN
        feature = np.where(np.isinf(feature), np.nan, feature)
        
        # Cap extreme values to prevent overflow
        if np.any(np.isfinite(feature)):
            finite_vals = feature[np.isfinite(feature)]
            if len(finite_vals) > 0:
                # Use robust percentile-based capping
                p1, p99 = np.percentile(finite_vals, [1, 99])
                cap_range = max(abs(p1), abs(p99))
                
                # Cap at 100x the 99th percentile to prevent extreme outliers
                max_val = min(cap_range * 100, 1e10)  # Hard cap at 1e10
                feature = np.clip(feature, -max_val, max_val)
        
        return feature.astype(np.float32)

    def _get_operation_codes(self) -> np.ndarray:
        """Map enabled operations to integer codes for Numba."""
        ops = []
        if getattr(self.config, 'include_sum', True): ops.append(0)
        if getattr(self.config, 'include_diff', True): ops.append(1)
        if getattr(self.config, 'include_prod', True): ops.append(2)
        if getattr(self.config, 'include_ratio', True): ops.append(3)
        if getattr(self.config, 'include_norm_ratio', True): ops.append(4)
        if getattr(self.config, 'include_minmax', True): 
            ops.extend([5, 6])  # min, max
        return np.array(ops, dtype=np.int32)
    
    def _smart_pair_screening(self, X_data: np.ndarray, col_names: List[str], 
                            y: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Smart pair screening using variance-based pre-filtering and correlation analysis.
        """
        n_cols = len(col_names)
        
        # Fast variance screening
        variances = np.nanvar(X_data, axis=0)
        high_var_mask = variances > np.percentile(variances, 25)  # Keep top 75%
        valid_indices = np.where(high_var_mask)[0]
        
        if len(valid_indices) < 2:
            return []
        
        # Efficient pairwise correlation screening
        pairs = []
        max_pairs = getattr(self.config, 'max_pairs_screen', 10000)
        corr_threshold = getattr(self.config, 'corr_avoid_redundancy', 0.95)
        
        # Subsample for correlation calculation if too large
        n_rows = X_data.shape[0]
        if n_rows > 10000:
            sample_idx = np.random.choice(n_rows, 10000, replace=False)
            X_sample = X_data[sample_idx]
        else:
            X_sample = X_data
        
        # Use AdaptiveMI's correlation method for consistency
        for i, idx_i in enumerate(valid_indices):
            for idx_j in valid_indices[i+1:]:
                if len(pairs) >= max_pairs:
                    break
                    
                # Fast correlation check using AdaptiveMI's method
                xi = X_sample[:, idx_i]
                xj = X_sample[:, idx_j]
                
                # Skip if too many NaNs
                valid = np.isfinite(xi) & np.isfinite(xj)
                if valid.sum() < 50:
                    continue
                
                # Use AdaptiveMI's Spearman calculation
                corr = self.ami_scorer._safe_spearman(xi[valid], xj[valid])
                if abs(corr) < corr_threshold:
                    pairs.append((idx_i, idx_j))
                
                if len(pairs) >= max_pairs:
                    break
        
        return pairs
    
    def _batch_interaction_generation(self, X_data: np.ndarray, pairs: List[Tuple[int, int]], 
                                    batch_size: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate interactions in batches with vectorized operations.
        """
        interactions = {}
        n_pairs = len(pairs)
        
        for batch_start in range(0, n_pairs, batch_size):
            batch_end = min(batch_start + batch_size, n_pairs)
            batch_pairs = pairs[batch_start:batch_end]
            
            for i, j in batch_pairs:
                col_i = self.numerical_cols_[i]
                col_j = self.numerical_cols_[j]
                
                a = X_data[:, i].astype(np.float32)
                b = X_data[:, j].astype(np.float32)
                
                # Compute all interactions at once using Numba
                results = _compute_all_interactions(a, b, self.op_codes, 
                                                 getattr(self.config, 'epsilon', 1e-8))
                
                # Map results to named features
                op_names = ['plus', 'diff', 'times', 'div', 'ndiv', 'min', 'max']
                for k, op_code in enumerate(self.op_codes):
                    op_name = op_names[op_code]
                    feature_name = f"{col_i}_{op_name}_{col_j}"
                    # Clean the interaction feature
                    cleaned_feature = self._clean_interaction_feature(results[k])
                    interactions[feature_name] = cleaned_feature
                            
        return interactions
    
    def _score_single_feature(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Score single feature using custom MI implementations.
        """
        try:
            # Use the appropriate scorer based on config
            if getattr(self.config, 'scorer', 'mi') == "hsic":
                return self.hsic_scorer.score(x, y)
            else:  # Use AdaptiveMI (default)
                return self.ami_scorer.score(x, y)
        
        except Exception:
            return 0.0
    
    def _sequential_scoring(self, interactions: Dict[str, np.ndarray], 
                           y: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Sequential scoring of interactions using custom MI implementations.
        """
        if y is None:
            # Unsupervised: use variance
            return {name: np.nanvar(arr) for name, arr in interactions.items()}
        
        # Choose the appropriate scorer and use batch processing if available
        if getattr(self.config, 'scorer', 'mi') == "hsic":
            # HSIC: score one by one (no batch method available)
            scores = {}
            for name, arr in interactions.items():
                scores[name] = self._score_single_feature(arr, y)
        else:
            # AdaptiveMI: use efficient batch scoring when possible
            interaction_matrix = np.column_stack(list(interactions.values()))
            feature_names = list(interactions.keys())
            
            try:
                # Try batch scoring for efficiency
                scores_array = self.ami_scorer.score_pairwise(interaction_matrix, y)
                scores = dict(zip(feature_names, scores_array))
            except Exception:
                # Fallback to individual scoring
                scores = {}
                for name, arr in interactions.items():
                    scores[name] = self._score_single_feature(arr, y)
        
        return scores
    
    def _adaptive_selection(self, scores: Dict[str, float], 
                          X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Modern feature selection combining score-based ranking with diversity maximization.
        """
        if not scores:
            return []
        
        # Sort by score
        ranked_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top candidates
        max_candidates = min(len(ranked_features), 
                           getattr(self.config, 'max_interactions', 1000))
        candidates = [name for name, _ in ranked_features[:max_candidates]]
        
        # Diversity-based selection if we have too many
        max_selected = getattr(self.config, 'max_selected_interactions', 100)
        if len(candidates) <= max_selected:
            return candidates
        
        # Use greedy diversity selection
        selected = [candidates[0]]  # Start with best
        remaining = set(candidates[1:])
        
        while len(selected) < max_selected and remaining:
            # Find feature with minimum correlation to selected set
            best_candidate = None
            min_max_corr = float('inf')
            
            for candidate in remaining:
                # Compute max correlation with already selected features
                max_corr = 0.0
                for selected_feat in selected:
                    corr = self._compute_feature_correlation(candidate, selected_feat, X)
                    max_corr = max(max_corr, abs(corr))
                
                if max_corr < min_max_corr:
                    min_max_corr = max_corr
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _compute_feature_correlation(self, feat1: str, feat2: str, X: pd.DataFrame) -> float:
        """Fast feature correlation computation with caching."""
        cache_key = (feat1, feat2) if feat1 < feat2 else (feat2, feat1)
        
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]
        
        try:
            # Reconstruct the features
            x1_data = self._reconstruct_feature_data(feat1, X)
            x2_data = self._reconstruct_feature_data(feat2, X)
            
            # Use AdaptiveMI's built-in Spearman calculation for consistency
            corr = self.ami_scorer._safe_spearman(x1_data, x2_data)
            
            self._correlation_cache[cache_key] = corr
            return corr
        
        except Exception:
            return 0.0
    
    def _reconstruct_feature_data(self, feature_name: str, X: pd.DataFrame) -> np.ndarray:
        """Reconstruct feature data from name."""
        # Parse feature name to recreate the interaction
        for op in ["_plus_", "_diff_", "_times_", "_div_", "_ndiv_", "_min_", "_max_"]:
            if op in feature_name:
                f1, f2 = feature_name.split(op)
                if f1 in X.columns and f2 in X.columns:
                    a = X[f1].to_numpy().astype(np.float32)
                    b = X[f2].to_numpy().astype(np.float32)
                    
                    eps = getattr(self.config, 'epsilon', 1e-8)
                    
                    if op == "_plus_":
                        result = a + b
                    elif op == "_diff_":
                        result = a - b
                    elif op == "_times_":
                        result = a * b
                    elif op == "_div_":
                        result = _safe_ratio_vectorized(a, b, eps)
                    elif op == "_ndiv_":
                        result = _safe_norm_ratio_vectorized(a, b, eps)
                    elif op == "_min_":
                        result = np.minimum(a, b)
                    elif op == "_max_":
                        result = np.maximum(a, b)
                    
                    # Clean the reconstructed feature
                    return self._clean_interaction_feature(result)
                break
        
        # Fallback
        return np.random.randn(len(X))
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "InteractionTransformer":
        """
        Optimized fit method with modern techniques.
        """
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not getattr(self.config, 'create_interactions', True) or len(self.numerical_cols_) < 2:
            self.is_fitted = True
            return self
        
        # Convert to optimized format
        X_numeric = X[self.numerical_cols_].astype(np.float32, copy=False)
        X_data = X_numeric.to_numpy()
        
        # Smart pair screening
        pairs = self._smart_pair_screening(X_data, self.numerical_cols_, 
                                         y.to_numpy() if y is not None else None)
        
        if not pairs:
            self.is_fitted = True
            return self
        
        # Batch interaction generation
        interactions = self._batch_interaction_generation(X_data, pairs)
        
        if not interactions:
            self.is_fitted = True
            return self
        
        # Sequential scoring with custom MI
        scores = self._sequential_scoring(interactions, 
                                        y.to_numpy() if y is not None else None)
        
        # Modern feature selection
        self.selected_interactions_ = self._adaptive_selection(scores, X, y)
        
        # Store pairs needed for transform
        self.interaction_pairs_ = pairs
        
        print(f"Selected {len(self.selected_interactions_)} interactions from {len(interactions)} candidates.")
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized transform with vectorized operations.
        """
        if not self.is_fitted or not getattr(self.config, 'create_interactions', True):
            return pd.DataFrame(index=X.index)
        
        if not self.interaction_pairs_ or not self.selected_interactions_:
            return pd.DataFrame(index=X.index)
        
        X_numeric = X[self.numerical_cols_].astype(np.float32, copy=False)
        X_data = X_numeric.to_numpy()
        
        # Generate only selected interactions
        interactions = self._batch_interaction_generation(X_data, self.interaction_pairs_)
        
        # Filter to selected features only
        selected_data = {name: arr for name, arr in interactions.items() 
                        if name in self.selected_interactions_}
        
        result_df = pd.DataFrame(selected_data, index=X.index)
        return result_df.astype(getattr(self.config, 'dtype_out', np.float32))
    
class ClusteringTransformer(BaseFeatureTransformer):
    """Creates clustering-based features for complex relationships."""
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ClusteringTransformer':
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.config.create_clustering or len(numerical_cols) < 2:
            self.is_fitted = True
            return self
        
        # Select top features by variance for clustering
        X_clust = X[numerical_cols].fillna(X[numerical_cols].median())
        top_features = X_clust.var().nlargest(min(10, len(numerical_cols))).index.tolist()
        
        self.cluster_features_ = top_features
        self.cluster_model_ = KMeans(
            n_clusters=min(self.config.n_clusters, len(X_clust) // 10),
            random_state=self.config.random_state,
            n_init=10,
        )
        
        try:
            self.cluster_model_.fit(X_clust[top_features])
        except Exception as e:
            warnings.warn(f"Clustering failed: {e}")
            self.cluster_model_ = None
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.create_clustering or not hasattr(self, 'cluster_model_') or self.cluster_model_ is None:
            return pd.DataFrame(index=X.index)
        
        try:
            # Prepare data
            available_features = [col for col in self.cluster_features_ if col in X.columns]
            if not available_features:
                return pd.DataFrame(index=X.index)
            
            X_clust = X[available_features].fillna(X[available_features].median())
            
            # Get cluster assignments and distances
            clusters = self.cluster_model_.predict(X_clust)
            distances = self.cluster_model_.transform(X_clust)
            
            features = {
                'cluster_id': clusters,
                'cluster_distance_min': distances.min(axis=1),
                'cluster_distance_mean': distances.mean(axis=1),
            }
            
            return pd.DataFrame(features, index=X.index)
            
        except Exception as e:
            warnings.warn(f"Cluster prediction failed: {e}")
            return pd.DataFrame(index=X.index)


class StatisticalTransformer(BaseFeatureTransformer):
    """Creates statistical aggregation features across rows."""
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'StatisticalTransformer':
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.create_statistical or len(self.numerical_cols_) < 2:
            return pd.DataFrame(index=X.index)
        
        available_cols = [col for col in self.numerical_cols_ if col in X.columns]
        if len(available_cols) < 2:
            return pd.DataFrame(index=X.index)
        
        X_stats = X[available_cols]
        
        features = {
            'row_mean': X_stats.mean(axis=1),
            'row_std': X_stats.std(axis=1),
            'row_min': X_stats.min(axis=1),
            'row_max': X_stats.max(axis=1),
            'row_median': X_stats.median(axis=1),
            'row_skew': X_stats.skew(axis=1),
            'row_non_null_count': X_stats.notna().sum(axis=1),
            'row_null_ratio': X_stats.isna().sum(axis=1) / len(available_cols),
        }
        
        features['row_range'] = features['row_max'] - features['row_min']
        
        return pd.DataFrame(features, index=X.index)


class AutoencoderTransformer(BaseFeatureTransformer):
    """Deep autoencoder for latent feature extraction."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AutoencoderTransformer':
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.config.use_autoencoder or len(numerical_cols) < 3:
            self.is_fitted = True
            return self
        
        # Prepare and scale data
        X_ae = X[numerical_cols].fillna(X[numerical_cols].median())
        
        # Use quantile transformer for better handling of outliers
        self.scaler_ = QuantileTransformer(
            n_quantiles=min(1000, max(10, len(X_ae) // 2)),
            output_distribution='normal',
            random_state=self.config.random_state,
        )
        X_scaled = self.scaler_.fit_transform(X_ae)
        
        # Create and train autoencoder
        n_features = X_scaled.shape[1]
        latent_dim = max(8, int(n_features * self.config.autoencoder_latent_ratio))
        
        self.model_ = SimpleAutoencoder(n_features, latent_dim).to(self.config.autoencoder_device)
        self._train_autoencoder(X_scaled)
        
        # Store feature names
        self.feature_names_ = [f"ae_latent_{i}" for i in range(latent_dim)]
        self.numerical_cols_ = numerical_cols
        self.is_fitted = True
        return self
    
    def _train_autoencoder(self, X_scaled: np.ndarray):
        """Train the autoencoder model."""
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.config.autoencoder_device)
        
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.config.autoencoder_lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.autoencoder_batch_size, 
            shuffle=True
        )
        
        self.model_.train()
        for epoch in range(self.config.autoencoder_epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                
                reconstructed = self.model_(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Autoencoder Epoch {epoch+1}/{self.config.autoencoder_epochs}, Loss: {avg_loss:.6f}")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.use_autoencoder or self.model_ is None:
            return pd.DataFrame(index=X.index)
        
        try:
            # Prepare data same way as during training
            X_ae = X[self.numerical_cols_].fillna(X[self.numerical_cols_].median())
            X_scaled = self.scaler_.transform(X_ae)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.config.autoencoder_device)
            
            self.model_.eval()
            with torch.no_grad():
                latent_features = self.model_.encode(X_tensor).cpu().numpy()
            
            return pd.DataFrame(
                latent_features,
                columns=self.feature_names_,
                index=X.index
            )
            
        except Exception as e:
            warnings.warn(f"Autoencoder transform failed: {e}")
            return pd.DataFrame(index=X.index)


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder implementation."""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        hidden_dim = max(latent_dim * 2, min(input_dim * 2, 256))
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)




class FeatureSelector:
    """
    Enhanced feature selector using mutual information with performance optimizations.
    
    Key improvements:
    1. Uses optimized AdaptiveMI.score_pairwise() for batch processing
    2. Efficient data preprocessing and validation
    3. Fallback methods for robustness
    4. Better memory management
    5. Handles both classification and regression
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.mi_scores_: Optional[pd.Series] = None
        self.shap_scores_: Optional[pd.Series] = None
        self.selected_features_: List[str] = []
        
        # Initialize AdaptiveMI with optimized settings
        self.ami_scorer = AdaptiveMI(
            subsample=min(getattr(config, 'max_rows_score', 2000), 2000),
            spearman_gate=getattr(config, 'mi_spearman_gate', 0.05),
            min_overlap=getattr(config, 'mi_min_overlap', 50),
            ks=(3, 5, 10),
            n_bins=getattr(config, 'mi_bins', 16),
            random_state=getattr(config, 'random_state', 42),
        )
        
        # Cache for performance
        self._feature_cache: Dict[str, float] = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit feature selector using optimized MI scoring.
        """
        # Get numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) == 0:
            self.selected_features_ = []
            return self
        
        # Clean and prepare data efficiently
        X_clean, y_clean = self._prepare_data(X[numerical_cols], y)
        
        if len(X_clean) < getattr(self.config, 'min_samples', 10):
            warnings.warn(f"Insufficient samples ({len(X_clean)}) for reliable MI estimation")
            self.selected_features_ = numerical_cols
            return self
        
        try:
            # Use optimized batch scoring
            self.mi_scores_ = self._compute_mi_scores_optimized(X_clean, y_clean)
            
            # Select features above threshold
            mi_threshold = getattr(self.config, 'mi_threshold', 0.01)
            self.selected_features_ = self.mi_scores_[
                self.mi_scores_ > mi_threshold
            ].index.tolist()
            
            # Ensure minimum number of features
            min_features = getattr(self.config, 'min_features', 1)
            if len(self.selected_features_) < min_features:
                # Take top-k features even if below threshold
                self.selected_features_ = self.mi_scores_.head(min_features).index.tolist()
            
            # Respect maximum number of features
            max_features = getattr(self.config, 'max_features', len(numerical_cols))
            if len(self.selected_features_) > max_features:
                self.selected_features_ = self.selected_features_[:max_features]
        
        except Exception as e:
            warnings.warn(f"MI selection failed: {e}, using fallback method")
            self.selected_features_ = self._fallback_selection(X_clean, y_clean)
        
        return self
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Efficiently prepare and clean data for MI computation.
        """
        # Align indices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Remove rows with missing target
        valid_target = y_aligned.notna() & np.isfinite(y_aligned.astype(float, errors='ignore'))
        
        if not valid_target.any():
            return pd.DataFrame(), pd.Series(dtype=float)
        
        X_clean = X_aligned[valid_target]
        y_clean = y_aligned[valid_target]
        
        # Handle classification labels efficiently
        if getattr(self.config, 'task', 'regression') == 'classification':
            if not pd.api.types.is_numeric_dtype(y_clean):
                # Fast label encoding
                le = LabelEncoder()
                y_clean = pd.Series(le.fit_transform(y_clean), index=y_clean.index)
            else:
                # Ensure integer labels for classification
                y_clean = y_clean.astype(int)
        else:
            # Ensure float labels for regression
            y_clean = y_clean.astype(float)
        
        # Convert to optimal dtypes for speed
        X_clean = X_clean.astype(np.float32, errors='ignore')
        
        return X_clean, y_clean
    
    def _compute_mi_scores_optimized(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Compute MI scores using optimized batch processing.
        """
        # Convert to numpy for batch processing
        X_values = X.to_numpy()
        y_values = y.to_numpy()
        
        # Use batch scoring for efficiency
        scores = self.ami_scorer.score_pairwise(X_values, y_values)
        
        # Create series with proper index
        mi_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        
        # Ensure all scores are finite and non-negative
        mi_scores = mi_scores.fillna(0.0).clip(lower=0.0)
        
        return mi_scores
    
    def _fallback_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Fallback feature selection using sklearn's univariate methods.
        """
        try:
            # Choose appropriate scoring function
            if getattr(self.config, 'task', 'regression') == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
            
            # Use SelectKBest as fallback
            k = min(getattr(self.config, 'max_features', 10), len(X.columns))
            selector = SelectKBest(score_func=score_func, k=k)
            
            # Handle any remaining NaN values
            X_filled = X.fillna(X.median())
            selector.fit(X_filled, y)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            return X.columns[selected_mask].tolist()
        
        except Exception as e:
            warnings.warn(f"Fallback selection also failed: {e}")
            # Ultimate fallback: return top features by variance
            variances = X.var().sort_values(ascending=False)
            max_features = min(getattr(self.config, 'max_features', 10), len(X.columns))
            return variances.head(max_features).index.tolist()
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features_.copy()
    
    def get_feature_scores(self) -> Optional[pd.Series]:
        """Get MI scores for all features."""
        return self.mi_scores_.copy() if self.mi_scores_ is not None else None
    
    def get_top_features(self, n: int = 10) -> List[str]:
        """Get top n features by MI score."""
        if self.mi_scores_ is None:
            return self.selected_features_[:n]
        return self.mi_scores_.head(n).index.tolist()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to include only selected features.
        """
        if not self.selected_features_:
            return pd.DataFrame(index=X.index)
        
        # Only return features that exist in X
        available_features = [f for f in self.selected_features_ if f in X.columns]
        
        if not available_features:
            warnings.warn("None of the selected features are available in the input data")
            return pd.DataFrame(index=X.index)
        
        return X[available_features].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the selector and transform the data in one step.
        """
        return self.fit(X, y).transform(X)


class CategoricalTransformer(BaseFeatureTransformer):
    """Handles categorical feature encoding with multiple strategies."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.encoders_ = {}
        self.rare_categories_ = {}
        self.categorical_cols_ = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalTransformer':
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not self.categorical_cols_:
            self.is_fitted = True
            return self
        
        for col in self.categorical_cols_:
            if col not in X.columns:
                continue
            
            # Handle rare categories
            value_counts = X[col].value_counts(normalize=True)
            rare_cats = value_counts[value_counts < self.config.rare_threshold].index.tolist()
            self.rare_categories_[col] = rare_cats
            
            # Choose encoding strategy based on cardinality
            n_unique = X[col].nunique()
            
            if n_unique > self.config.target_encode_threshold and y is not None:
                # Target encoding for high cardinality
                self.encoders_[col] = self._fit_target_encoder(X[col], y)
            elif n_unique <= 10:
                # One-hot encoding for low cardinality
                self.encoders_[col] = self._fit_onehot_encoder(X[col])
            else:
                # Frequency encoding for medium cardinality
                self.encoders_[col] = self._fit_frequency_encoder(X[col])
        
        self.is_fitted = True
        return self
    
    def _fit_target_encoder(self, series: pd.Series, y: pd.Series) -> Dict:
        """Fit target encoder with regularization."""
        global_mean = y.mean()
        category_stats = pd.DataFrame({
            'count': series.value_counts(),
            'target_mean': y.groupby(series).mean()
        }).fillna(global_mean)
        
        # Regularization
        alpha = 10
        regularized_means = (
            category_stats['count'] * category_stats['target_mean'] + alpha * global_mean
        ) / (category_stats['count'] + alpha)
        
        return {
            'type': 'target',
            'mapping': regularized_means.to_dict(),
            'global_mean': global_mean
        }
    
    def _fit_onehot_encoder(self, series: pd.Series) -> Dict:
        """Fit one-hot encoder."""
        from sklearn.preprocessing import OneHotEncoder
        
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary')
        encoder.fit(series.values.reshape(-1, 1))
        
        return {
            'type': 'onehot',
            'encoder': encoder,
            'feature_names': encoder.get_feature_names_out([series.name])
        }
    
    def _fit_frequency_encoder(self, series: pd.Series) -> Dict:
        """Fit frequency encoder."""
        freq_map = series.value_counts(normalize=True).to_dict()
        return {
            'type': 'frequency',
            'mapping': defaultdict(lambda: 0, freq_map)
        }
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_cols_:
            return pd.DataFrame(index=X.index)
        
        result_dfs = []
        
        for col in self.categorical_cols_:
            if col not in X.columns or col not in self.encoders_:
                continue
            
            # Handle rare categories and missing values
            series = X[col].fillna('MISSING')
            if col in self.rare_categories_:
                series = series.replace(self.rare_categories_[col], 'OTHER')
            
            encoder_info = self.encoders_[col]
            
            if encoder_info['type'] == 'target':
                encoded = series.map(encoder_info['mapping']).fillna(encoder_info['global_mean'])
                result_dfs.append(pd.DataFrame({f"{col}_target_enc": encoded}, index=X.index))
            
            elif encoder_info['type'] == 'onehot':
                try:
                    encoded = encoder_info['encoder'].transform(series.values.reshape(-1, 1))
                    feature_names = encoder_info['feature_names']
                    result_dfs.append(pd.DataFrame(encoded, columns=feature_names, index=X.index))
                except Exception as e:
                    warnings.warn(f"OneHot encoding failed for {col}: {e}")
            
            elif encoder_info['type'] == 'frequency':
                encoded = series.map(encoder_info['mapping'])
                result_dfs.append(pd.DataFrame({f"{col}_freq": encoded}, index=X.index))
        
        return pd.concat(result_dfs, axis=1) if result_dfs else pd.DataFrame(index=X.index)


class BinningTransformer(BaseFeatureTransformer):
    """Creates adaptive binning features for numerical data."""
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BinningTransformer':
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.binning_transformers_ = {}
        
        if not self.config.create_binning:
            self.is_fitted = True
            return self
        
        for col in self.numerical_cols_:
            if col not in X.columns:
                continue
            
            data = X[[col]].dropna()
            
            # Skip if insufficient data or too constant
            if (len(data) < self.config.n_bins * 2 or 
                data[col].nunique() <= 1 or 
                (data[col].max() - data[col].min()) < 1e-6):
                continue
            
            try:
                discretizer = KBinsDiscretizer(
                    n_bins=self.config.n_bins,
                    encode='ordinal',
                    strategy='quantile',
                    subsample=min(10000, len(data))
                )
                discretizer.fit(data)
                self.binning_transformers_[col] = discretizer
            except Exception as e:
                warnings.warn(f"Binning failed for {col}: {e}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.create_binning or not self.binning_transformers_:
            return pd.DataFrame(index=X.index)
        
        features = {}
        for col, transformer in self.binning_transformers_.items():
            if col not in X.columns:
                continue
            
            try:
                X_col = X[[col]].fillna(X[col].median())
                binned = transformer.transform(X_col)
                features[f"{col}_bin"] = binned.flatten()
            except Exception as e:
                warnings.warn(f"Binning transform failed for {col}: {e}")
        
        return pd.DataFrame(features, index=X.index)


class FourierTransformer(BaseFeatureTransformer):
    """Creates Fourier features for periodic patterns."""
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FourierTransformer':
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.fourier_configs_ = {}
        
        if not self.config.create_fourier:
            self.is_fitted = True
            return self
        
        for col in self.numerical_cols_:
            if col not in X.columns:
                continue
            
            data = X[col].fillna(X[col].median())
            
            # Skip if insufficient variation
            if data.var() < 1e-6:
                continue
            
            try:
                # Normalize and compute FFT
                data_norm = (data - data.mean()) / (data.std() + 1e-8)
                fft_vals = fft.fft(data_norm.values)
                freqs = fft.fftfreq(len(data_norm))
                
                # Extract dominant frequencies
                magnitude = np.abs(fft_vals)
                top_freq_idx = np.argsort(magnitude)[-(self.config.n_fourier_terms + 1):-1]
                
                valid_frequencies = [freqs[idx] for idx in top_freq_idx if freqs[idx] != 0]
                
                if valid_frequencies:
                    self.fourier_configs_[col] = {
                        'frequencies': valid_frequencies,
                        'mean': data.mean(),
                        'std': data.std()
                    }
            except Exception as e:
                warnings.warn(f"Fourier analysis failed for {col}: {e}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.create_fourier or not self.fourier_configs_:
            return pd.DataFrame(index=X.index)
        
        features = {}
        for col, config in self.fourier_configs_.items():
            if col not in X.columns:
                continue
            
            data = X[col].fillna(config['mean'])
            
            for i, freq in enumerate(config['frequencies']):
                features[f"{col}_fourier_cos_{i}"] = np.cos(2 * np.pi * freq * np.arange(len(data)))
                features[f"{col}_fourier_sin_{i}"] = np.sin(2 * np.pi * freq * np.arange(len(data)))
        
        return pd.DataFrame(features, index=X.index)


class CorrelationFilter:
    """Removes highly correlated features."""
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.features_to_drop_ = []
    
    def fit(self, X: pd.DataFrame) -> 'CorrelationFilter':
        """Fit correlation filter."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            self.features_to_drop_ = []
            return self
        
        try:
            corr_matrix = X[numerical_cols].corr().abs()
            upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            high_corr_pairs = np.where((corr_matrix > self.threshold) & upper_tri)
            
            for i, j in zip(*high_corr_pairs):
                col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                # Drop feature with lower variance
                var1, var2 = X[col1].var(), X[col2].var()
                drop_col = col1 if var1 < var2 else col2
                if drop_col not in self.features_to_drop_:
                    self.features_to_drop_.append(drop_col)
        except Exception as e:
            warnings.warn(f"Correlation analysis failed: {e}")
            self.features_to_drop_ = []
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove correlated features."""
        return X.drop(columns=self.features_to_drop_, errors='ignore')


# Enhanced SOTA Feature Engineer with all transformers
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive state-of-the-art feature engineering pipeline.
    
    Features:
    - Modular transformer architecture
    - Configuration-driven approach
    - Comprehensive feature creation and selection
    - Robust error handling
    - Memory efficient processing
    - Extensive logging and monitoring
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.transformers_ = {}
        self.correlation_filter_ = None
        self.selector_ = None
        self.final_scaler_ = None
        self.fitted_ = False
        
        # Track feature creation statistics
        self.feature_stats_ = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fit comprehensive feature engineering pipeline."""
        X = X.copy()
        if y is not None:
            y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        
        print(f"🚀 Fitting SOTA Feature Engineer on {X.shape[0]} rows, {X.shape[1]} columns")
        
        # Initialize all transformers
        self.transformers_ = {
            'datetime': DateTimeTransformer(self.config),
            'mathematical': MathematicalTransformer(self.config),
            'interactions': InteractionTransformer(self.config),
            'categorical': CategoricalTransformer(self.config),
            'binning': BinningTransformer(self.config),
            'clustering': ClusteringTransformer(self.config),
            'statistical': StatisticalTransformer(self.config),
            'fourier': FourierTransformer(self.config),
            #'autoencoder': AutoencoderTransformer(self.config),
        }
        
        # Fit transformers sequentially
        current_X = X.copy()
        for name, transformer in self.transformers_.items():
            print(f"🔧 Fitting {name} transformer...")
            # try:
            transformer.fit(current_X, y)
            # Update current_X with new features for next transformer
            transformed = transformer.transform(current_X)
            if not transformed.empty:
                current_X = pd.concat([current_X, transformed], axis=1)
                self.feature_stats_[name] = transformed.shape[1]
            else:
                self.feature_stats_[name] = 0
        # except Exception as e:
        #     warnings.warn(f"❌ {name.title()} transformer failed: {e}")
        #     self.feature_stats_[name] = 0
        
        # Apply correlation filtering
        print("🔍 Applying correlation filtering...")
        self.correlation_filter_ = CorrelationFilter(self.config.corr_threshold)
        self.correlation_filter_.fit(current_X)
        current_X = self.correlation_filter_.transform(current_X)
        
        # Feature selection
        if y is not None and not current_X.empty:
            print("🎯 Performing feature selection...")
            self.selector_ = FeatureSelector(self.config)
            self.selector_.fit(current_X, y)
            selected_features = self.selector_.get_selected_features()
            
            if selected_features:
                current_X = current_X[selected_features]
                
                # Fit final scaler
                if self.config.use_quantile_transform:
                    print("📊 Fitting final quantile transformer...")
                    self.final_scaler_ = QuantileTransformer(
                        n_quantiles=min(1000, max(10, len(current_X) // 2)),
                        output_distribution='normal',
                        random_state=self.config.random_state,
                    )
                    X_filled = current_X.fillna(current_X.median())
                    self.final_scaler_.fit(X_filled)
                
                print(f"✅ Selected {len(selected_features)} features after filtering")
            else:
                print("⚠️  No features selected - pipeline may need tuning")
        
        # Print feature creation summary
        self._print_feature_summary()
        
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted pipeline."""
        check_is_fitted(self, 'fitted_')
        
        current_X = X.copy()
        
        # Apply all transformers
        for name, transformer in self.transformers_.items():
            try:
                transformed = transformer.transform(current_X)
                if not transformed.empty:
                    current_X = pd.concat([current_X, transformed], axis=1)
            except Exception as e:
                warnings.warn(f"Transform failed for {name}: {e}")
        
        # Apply correlation filtering
        if self.correlation_filter_ is not None:
            current_X = self.correlation_filter_.transform(current_X)
        
        # Apply feature selection
        if self.selector_ is not None:
            selected_features = self.selector_.get_selected_features()
            available_features = [col for col in selected_features if col in current_X.columns]
            
            if available_features:
                current_X = current_X[available_features]
                
                # Apply final scaling
                if self.final_scaler_ is not None:
                    X_filled = current_X.fillna(current_X.median())
                    X_scaled = self.final_scaler_.transform(X_filled)
                    return pd.DataFrame(X_scaled, columns=current_X.columns, index=X.index)
        
        return current_X if not current_X.empty else pd.DataFrame(index=X.index)
    
    def _print_feature_summary(self):
        """Print summary of feature creation."""
        print("\n📈 Feature Creation Summary:")
        print("-" * 40)
        total_features = 0
        for name, count in self.feature_stats_.items():
            print(f"{name.title():<15}: {count:>4} features")
            total_features += count
        print("-" * 40)
        print(f"{'Total':<15}: {total_features:>4} features")
        
        if self.correlation_filter_:
            dropped = len(self.correlation_filter_.features_to_drop_)
            print(f"{'Corr. Dropped':<15}: {dropped:>4} features")
        
        if self.selector_:
            selected = len(self.selector_.get_selected_features())
            print(f"{'Final Selected':<15}: {selected:>4} features")
        print()
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from mutual information analysis."""
        if self.selector_ and hasattr(self.selector_, 'mi_scores_'):
            return self.selector_.mi_scores_
        return None
    
    def plot_feature_importance(self, top_k: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """Plot feature importance with enhanced visualization."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        importance_scores = self.get_feature_importance()
        if importance_scores is None:
            print("❌ No feature importance scores available.")
            return
        
        top_scores = importance_scores.head(top_k)
        
        plt.figure(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_scores)))
        bars = plt.barh(range(len(top_scores)), top_scores.values, color=colors)
        
        plt.title(f'🎯 Top {top_k} Features by Mutual Information', fontsize=16, fontweight='bold')
        plt.xlabel('Mutual Information Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.yticks(range(len(top_scores)), top_scores.index)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_scores.values)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.grid(axis='x', alpha=0.3)
        plt.show()
    
    def get_transformation_report(self) -> Dict[str, Any]:
        """Get detailed report of transformations applied."""
        report = {
            'feature_stats': self.feature_stats_.copy(),
            'config': self.config.__dict__.copy(),
            'correlation_dropped': len(self.correlation_filter_.features_to_drop_) if self.correlation_filter_ else 0,
            'selected_features': len(self.selector_.get_selected_features()) if self.selector_ else 0,
        }
        
        if self.selector_ and hasattr(self.selector_, 'mi_scores_'):
            report['top_features'] = self.selector_.mi_scores_.head(10).to_dict()
        
        return report


