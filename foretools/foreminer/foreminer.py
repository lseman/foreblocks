import os
import sys
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import mahalanobis, pdist, squareform
from scipy.stats import (
    anderson,
    entropy,
    jarque_bera,
    ks_2samp,
    kurtosis,
    linregress,
    mode,
    normaltest,
    shapiro,
    skew,
    wasserstein_distance,
)
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance, MinCovDet
from sklearn.decomposition import NMF, PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from .foreminer_aux import *
from .foreminer_aux import _run_analysis_worker

# Suppress known noise warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)


# ============================================================================
# COMPREHENSIVE ANALYSIS STRATEGIES
# ============================================================================


class DistributionAnalyzer(AnalysisStrategy):
    """Comprehensive distribution analysis with advanced statistical features"""

    @property
    def name(self) -> str:
        return "distributions"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        summary_data = []

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 8:
                continue

            try:
                stats_dict = self._compute_comprehensive_stats(col_data, col, config)
                summary_data.append(stats_dict)
            except Exception as e:
                print(f"Distribution analysis failed for {col}: {e}")

        return {"summary": pd.DataFrame(summary_data)}

    def _compute_comprehensive_stats(
        self, col_data: pd.Series, col_name: str, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Compute comprehensive distribution statistics"""
        mean, std = col_data.mean(), col_data.std()
        min_val, max_val = col_data.min(), col_data.max()
        range_val = max_val - min_val
        cv = std / abs(mean) if mean != 0 else np.inf
        skew_val = skew(col_data)
        kurt_val = kurtosis(col_data, fisher=False)

        # Histogram for entropy
        hist, _ = np.histogram(col_data, bins=30, density=True)
        hist += 1e-10
        entr = entropy(hist, base=2)

        # Quartiles and IQR
        q1, q2, q3 = col_data.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        # Normality tests
        _, p_norm = normaltest(col_data)
        sample_size = min(5000, len(col_data))
        _, p_shapiro = shapiro(
            col_data.sample(sample_size, random_state=config.random_state)
        )

        # Advanced distribution features
        bimodality_coeff = (skew_val**2 + 1) / kurt_val if kurt_val != 0 else 0
        lower_5 = col_data.quantile(0.05)
        upper_95 = col_data.quantile(0.95)
        tail_ratio = (
            (max_val - upper_95) / (lower_5 - min_val + 1e-6)
            if (lower_5 > min_val)
            else np.nan
        )

        return {
            "feature": col_name,
            "count": len(col_data),
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "cv": cv,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "excess_kurtosis": kurt_val - 3,
            "entropy": entr,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "normaltest_p": p_norm,
            "shapiro_p": p_shapiro,
            "is_gaussian": p_norm > config.confidence_level and abs(skew_val) < 1,
            "is_skewed": abs(skew_val) > 1,
            "is_heavy_tailed": abs(kurt_val) > 3,
            "bimodality_coeff": bimodality_coeff,
            "tail_ratio": tail_ratio,
        }


class CorrelationAnalyzer(AnalysisStrategy):
    """Advanced correlation analysis with multiple methods"""

    @property
    def name(self) -> str:
        return "correlations"

    def __init__(self):
        self.strategies = {
            "pearson": PearsonCorrelation(),
            "spearman": SpearmanCorrelation(),
            "mutual_info": MutualInfoCorrelation(),
            "distance": DistanceCorrelation(),
        }
        if OPTIONAL_IMPORTS["phik"]:
            self.strategies["phik"] = PhiKCorrelation()

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        if numeric_data.empty:
            return {}

        results = {}
        for method, strategy in self.strategies.items():
            try:
                results[method] = strategy.compute(numeric_data)
            except Exception as e:
                print(f"Failed to compute {method} correlation: {e}")

        return results
warnings.filterwarnings('ignore')

try:
    from pyod.models.cblof import CBLOF
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.hbos import HBOS
    from pyod.models.knn import KNN
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

warnings.filterwarnings('ignore')

try:
    from pyod.models.cblof import CBLOF
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.hbos import HBOS
    from pyod.models.knn import KNN
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

warnings.filterwarnings('ignore')

try:
    from pyod.models.cblof import CBLOF
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.hbos import HBOS
    from pyod.models.knn import KNN
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


class OutlierAnalyzer(AnalysisStrategy):
    """State-of-the-art outlier detection with ensemble methods and adaptive thresholding"""

    @property
    def name(self) -> str:
        return "outliers"

    def _adaptive_preprocessing(self, data: pd.DataFrame, config: AnalysisConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Enhanced preprocessing with missing value handling and scaling selection"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric data available for outlier detection")

        preprocessing_info = {}
        
        # Analyze missing data pattern
        na_mask = numeric_data.isna().any(axis=1)
        missing_percentage = na_mask.sum() / len(data) * 100
        preprocessing_info['missing_data_percentage'] = float(missing_percentage)
        preprocessing_info['samples_with_missing'] = int(na_mask.sum())
        
        if na_mask.all():
            raise ValueError("All samples have missing values")

        # Handle missing values intelligently
        if missing_percentage > 50:
            # High missing data - use only complete cases
            clean_data = numeric_data.dropna()
            # Get integer positions, not pandas index values
            complete_indices = np.where(~na_mask)[0]  # Convert to integer positions
            preprocessing_info['handling_strategy'] = 'complete_cases_only'
        elif missing_percentage > 10:
            # Moderate missing data - impute with robust methods
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            clean_data = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_data.columns,
                index=numeric_data.index
            )
            complete_indices = np.arange(len(clean_data))  # All indices since we imputed
            preprocessing_info['handling_strategy'] = 'median_imputation'
        else:
            # Low missing data - use complete cases
            clean_data = numeric_data.dropna()
            # Get integer positions, not pandas index values
            complete_indices = np.where(~na_mask)[0]  # Convert to integer positions
            preprocessing_info['handling_strategy'] = 'complete_cases_only'

        # Data quality assessment
        preprocessing_info['final_sample_size'] = len(clean_data)
        preprocessing_info['final_feature_count'] = clean_data.shape[1]
        
        # Detect data characteristics for optimal scaling
        skewness = clean_data.skew().abs().mean()
        kurtosis = clean_data.kurtosis().abs().mean()
        preprocessing_info['data_skewness'] = float(skewness)
        preprocessing_info['data_kurtosis'] = float(kurtosis)
        
        # Adaptive scaling selection
        scaling_methods = []
        
        if skewness > 3 or kurtosis > 10:
            # Highly skewed/heavy-tailed data
            scaling_methods = [
                ('power_transform', PowerTransformer(method='yeo-johnson', standardize=True)),
                ('robust', RobustScaler()),
                ('standard', StandardScaler())
            ]
        elif skewness > 1.5:
            # Moderately skewed data
            scaling_methods = [
                ('robust', RobustScaler()),
                ('power_transform', PowerTransformer(method='yeo-johnson', standardize=True)),
                ('standard', StandardScaler())
            ]
        else:
            # Well-behaved data
            scaling_methods = [
                ('standard', StandardScaler()),
                ('robust', RobustScaler())
            ]

        # Select best scaling method based on condition number
        best_scaler = None
        best_score = float('inf')
        
        for name, scaler in scaling_methods:
            try:
                scaled = scaler.fit_transform(clean_data)
                # Evaluate scaling quality
                cond_number = np.linalg.cond(np.cov(scaled.T))
                if cond_number < best_score:
                    best_score = cond_number
                    best_scaler = scaler
                    preprocessing_info['scaling_method'] = name
            except Exception:
                continue

        if best_scaler is None:
            best_scaler = RobustScaler()
            preprocessing_info['scaling_method'] = 'robust_fallback'

        scaled_data = best_scaler.fit_transform(clean_data)
        preprocessing_info['condition_number'] = float(best_score) if best_score != float('inf') else None
        
        return scaled_data, complete_indices, preprocessing_info

    def _statistical_outlier_detection(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Statistical outlier detection methods"""
        results = {}
        n_samples, n_features = data.shape
        
        # Z-score based detection (multivariate)
        try:
            z_scores = np.abs(stats.zscore(data, axis=0))
            z_threshold = 3.0  # Standard 3-sigma rule
            z_outliers = (z_scores > z_threshold).any(axis=1)
            
            results['z_score'] = {
                'outliers': z_outliers,
                'scores': np.max(z_scores, axis=1),
                'threshold': z_threshold,
                'method_type': 'statistical'
            }
        except Exception as e:
            print(f"Z-score detection failed: {e}")

        # Modified Z-score (using median)
        try:
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            modified_z_scores = 0.6745 * (data - median) / (mad + 1e-10)  # Avoid division by zero
            modified_z_scores = np.abs(modified_z_scores)
            mz_threshold = 3.5
            mz_outliers = (modified_z_scores > mz_threshold).any(axis=1)
            
            results['modified_z_score'] = {
                'outliers': mz_outliers,
                'scores': np.max(modified_z_scores, axis=1),
                'threshold': mz_threshold,
                'method_type': 'statistical'
            }
        except Exception as e:
            print(f"Modified Z-score detection failed: {e}")

        # IQR method (univariate applied to each feature)
        try:
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            iqr_lower = Q1 - 1.5 * IQR
            iqr_upper = Q3 + 1.5 * IQR
            iqr_outliers = ((data < iqr_lower) | (data > iqr_upper)).any(axis=1)
            
            results['iqr'] = {
                'outliers': iqr_outliers,
                'lower_bounds': iqr_lower,
                'upper_bounds': iqr_upper,
                'method_type': 'statistical'
            }
        except Exception as e:
            print(f"IQR detection failed: {e}")

        # Grubbs test for univariate outliers (applied to principal component)
        try:
            if n_features > 1:
                # Apply PCA and use first component
                pca = PCA(n_components=1, random_state=42)
                pc1 = pca.fit_transform(data).flatten()
            else:
                pc1 = data.flatten()
            
            # Grubbs test statistic
            mean_pc = np.mean(pc1)
            std_pc = np.std(pc1)
            z_scores = np.abs((pc1 - mean_pc) / std_pc)
            
            # Critical value for Grubbs test (approximate)
            alpha = 0.05
            n = len(pc1)
            t_critical = stats.t.ppf(1 - alpha/(2*n), n-2)
            grubbs_critical = ((n-1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n-2 + t_critical**2))
            
            grubbs_outliers = z_scores > grubbs_critical
            
            results['grubbs'] = {
                'outliers': grubbs_outliers,
                'scores': z_scores,
                'threshold': grubbs_critical,
                'method_type': 'statistical'
            }
        except Exception as e:
            print(f"Grubbs test failed: {e}")

        return results

    def _distance_based_detection(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Distance-based outlier detection methods"""
        results = {}
        n_samples = len(data)
        
        # Mahalanobis distance with robust covariance estimation
        try:
            # Use Minimum Covariance Determinant for robustness
            robust_cov = MinCovDet(random_state=config.random_state).fit(data)
            maha_distances = robust_cov.mahalanobis(data)
            
            # Threshold using chi-square distribution
            from scipy.stats import chi2
            threshold = chi2.ppf(1 - config.outlier_contamination, data.shape[1])
            maha_outliers = maha_distances > threshold
            
            results['mahalanobis_robust'] = {
                'outliers': maha_outliers,
                'distances': maha_distances,
                'threshold': threshold,
                'method_type': 'distance_based'
            }
        except Exception as e:
            print(f"Robust Mahalanobis detection failed: {e}")

        # K-nearest neighbors distance
        try:
            k = min(20, n_samples // 10, n_samples - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)  # +1 because point is its own neighbor
            distances, indices = nbrs.kneighbors(data)
            
            # Use mean distance to k-th nearest neighbors (excluding self)
            knn_distances = np.mean(distances[:, 1:], axis=1)
            knn_threshold = np.percentile(knn_distances, 100 * (1 - config.outlier_contamination))
            knn_outliers = knn_distances > knn_threshold
            
            results['knn_distance'] = {
                'outliers': knn_outliers,
                'distances': knn_distances,
                'threshold': knn_threshold,
                'k': k,
                'method_type': 'distance_based'
            }
        except Exception as e:
            print(f"KNN distance detection failed: {e}")

        # DBSCAN-based outlier detection
        try:
            # Estimate eps using k-distance graph
            k = min(4, n_samples // 20)
            nbrs = NearestNeighbors(n_neighbors=k).fit(data)
            distances, _ = nbrs.kneighbors(data)
            eps = np.percentile(distances[:, -1], 90)  # 90th percentile of k-distances
            
            dbscan = DBSCAN(eps=eps, min_samples=max(2, k))
            labels = dbscan.fit_predict(data)
            dbscan_outliers = labels == -1
            
            results['dbscan'] = {
                'outliers': dbscan_outliers,
                'labels': labels,
                'eps': eps,
                'min_samples': max(2, k),
                'method_type': 'density_based'
            }
        except Exception as e:
            print(f"DBSCAN outlier detection failed: {e}")

        return results

    def _machine_learning_detection(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Machine learning based outlier detection"""
        results = {}
        
        # Enhanced Isolation Forest with multiple configurations
        try:
            # Try different configurations and pick the most stable one
            forest_configs = [
                {'n_estimators': 200, 'max_samples': 'auto', 'contamination': config.outlier_contamination},
                {'n_estimators': 150, 'max_samples': min(256, len(data)), 'contamination': config.outlier_contamination},
                {'n_estimators': 100, 'max_samples': 0.8, 'contamination': config.outlier_contamination}
            ]
            
            best_forest = None
            best_consistency = 0
            
            for forest_config in forest_configs:
                try:
                    # Run multiple times to check consistency
                    predictions = []
                    for seed in range(3):
                        forest = IsolationForest(random_state=config.random_state + seed, **forest_config)
                        pred = forest.fit_predict(data)
                        predictions.append(pred)
                    
                    # Calculate consistency (agreement between runs)
                    consistency = np.mean([np.mean(p1 == p2) for p1 in predictions for p2 in predictions if not np.array_equal(p1, p2)])
                    
                    if consistency > best_consistency:
                        best_consistency = consistency
                        best_forest = IsolationForest(random_state=config.random_state, **forest_config)
                
                except Exception:
                    continue
            
            if best_forest is not None:
                best_forest.fit(data)
                iso_predictions = best_forest.predict(data) == -1
                iso_scores = best_forest.decision_function(data)
                
                results['isolation_forest'] = {
                    'outliers': iso_predictions,
                    'scores': -iso_scores,  # Negative because lower scores = more anomalous
                    'model': best_forest,
                    'method_type': 'ensemble'
                }
        except Exception as e:
            print(f"Isolation Forest detection failed: {e}")

        # Enhanced Local Outlier Factor
        try:
            # Adaptive neighborhood size
            n_samples = len(data)
            neighbor_sizes = [min(20, max(5, n_samples // 20)), min(50, max(10, n_samples // 10))]
            
            best_lof = None
            best_score = -float('inf')
            
            for n_neighbors in neighbor_sizes:
                try:
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=config.outlier_contamination)
                    lof_predictions = lof.fit_predict(data) == -1
                    
                    # Evaluate using silhouette-like metric
                    if lof_predictions.sum() > 0 and (~lof_predictions).sum() > 0:
                        from sklearn.metrics import silhouette_score
                        score = silhouette_score(data, ~lof_predictions)  # Invert for outlier/inlier
                        
                        if score > best_score:
                            best_score = score
                            best_lof = lof
                
                except Exception:
                    continue
            
            if best_lof is not None:
                lof_predictions = best_lof.fit_predict(data) == -1
                lof_scores = -best_lof.negative_outlier_factor_
                
                results['local_outlier_factor'] = {
                    'outliers': lof_predictions,
                    'scores': lof_scores,
                    'n_neighbors': best_lof.n_neighbors,
                    'method_type': 'density_based'
                }
        except Exception as e:
            print(f"LOF detection failed: {e}")

        # One-Class SVM with multiple kernels
        try:
            kernels = ['rbf', 'sigmoid']
            best_svm = None
            best_score = -float('inf')
            
            for kernel in kernels:
                try:
                    svm = OneClassSVM(kernel=kernel, gamma='scale', nu=config.outlier_contamination)
                    svm.fit(data)
                    svm_predictions = svm.predict(data) == -1
                    
                    # Simple evaluation based on decision function spread
                    decision_scores = svm.decision_function(data)
                    score = np.std(decision_scores)  # Higher std = better separation
                    
                    if score > best_score:
                        best_score = score
                        best_svm = svm
                
                except Exception:
                    continue
            
            if best_svm is not None:
                svm_predictions = best_svm.predict(data) == -1
                svm_scores = -best_svm.decision_function(data)
                
                results['one_class_svm'] = {
                    'outliers': svm_predictions,
                    'scores': svm_scores,
                    'kernel': best_svm.kernel,
                    'method_type': 'boundary_based'
                }
        except Exception as e:
            print(f"One-Class SVM detection failed: {e}")

        # Elliptic Envelope
        try:
            elliptic = EllipticEnvelope(
                contamination=config.outlier_contamination,
                random_state=config.random_state
            )
            elliptic.fit(data)
            elliptic_predictions = elliptic.predict(data) == -1
            elliptic_scores = elliptic.decision_function(data)
            
            results['elliptic_envelope'] = {
                'outliers': elliptic_predictions,
                'scores': -elliptic_scores,
                'method_type': 'covariance_based'
            }
        except Exception as e:
            print(f"Elliptic Envelope detection failed: {e}")

        return results

    def _advanced_detection_methods(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Advanced outlier detection methods using external libraries"""
        results = {}
        
        # PyOD methods (if available)
        if HAS_PYOD:
            try:
                # Histogram-based Outlier Score
                hbos = HBOS(contamination=config.outlier_contamination)
                hbos.fit(data)
                hbos_predictions = hbos.predict(data) == 1  # PyOD uses 1 for outliers
                hbos_scores = hbos.decision_scores_
                
                results['hbos'] = {
                    'outliers': hbos_predictions,
                    'scores': hbos_scores,
                    'method_type': 'histogram_based'
                }
            except Exception as e:
                print(f"HBOS detection failed: {e}")

            try:
                # Feature Bagging
                feature_bagging = FeatureBagging(contamination=config.outlier_contamination, random_state=config.random_state)
                feature_bagging.fit(data)
                fb_predictions = feature_bagging.predict(data) == 1
                fb_scores = feature_bagging.decision_scores_
                
                results['feature_bagging'] = {
                    'outliers': fb_predictions,
                    'scores': fb_scores,
                    'method_type': 'ensemble'
                }
            except Exception as e:
                print(f"Feature Bagging detection failed: {e}")

            try:
                # Cluster-based Local Outlier Factor
                if len(data) >= 20:  # CBLOF needs sufficient data
                    cblof = CBLOF(contamination=config.outlier_contamination, random_state=config.random_state)
                    cblof.fit(data)
                    cblof_predictions = cblof.predict(data) == 1
                    cblof_scores = cblof.decision_scores_
                    
                    results['cblof'] = {
                        'outliers': cblof_predictions,
                        'scores': cblof_scores,
                        'method_type': 'cluster_based'
                    }
            except Exception as e:
                print(f"CBLOF detection failed: {e}")

        # HDBSCAN-based outlier detection
        if HAS_HDBSCAN:
            try:
                min_cluster_size = max(5, len(data) // 50)
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(1, min_cluster_size // 2)
                )
                cluster_labels = clusterer.fit_predict(data)
                hdbscan_outliers = cluster_labels == -1
                
                # Use outlier scores if available
                outlier_scores = getattr(clusterer, 'outlier_scores_', None)
                
                results['hdbscan'] = {
                    'outliers': hdbscan_outliers,
                    'scores': outlier_scores if outlier_scores is not None else np.zeros(len(data)),
                    'min_cluster_size': min_cluster_size,
                    'method_type': 'density_based'
                }
            except Exception as e:
                print(f"HDBSCAN outlier detection failed: {e}")

        return results

    def _ensemble_outlier_detection(self, all_results: Dict[str, Any], data: np.ndarray, 
                                  config: AnalysisConfig) -> Dict[str, Any]:
        """Ensemble outlier detection combining multiple methods"""
        if len(all_results) < 2:
            return {}

        try:
            # Collect predictions and scores from all methods
            predictions = []
            scores_list = []
            method_weights = {}
            
            for method_name, result in all_results.items():
                if 'outliers' in result and isinstance(result['outliers'], np.ndarray):
                    predictions.append(result['outliers'])
                    
                    # Normalize scores if available
                    if 'scores' in result and result['scores'] is not None:
                        scores = np.array(result['scores'])
                        # Normalize scores to [0, 1]
                        if scores.std() > 0:
                            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
                        else:
                            normalized_scores = np.zeros_like(scores)
                        scores_list.append(normalized_scores)
                    else:
                        # Use binary predictions as scores
                        scores_list.append(result['outliers'].astype(float))
                    
                    # Weight methods - could be enhanced with validation
                    method_weights[method_name] = 1.0

            if not predictions:
                return {}

            # Voting-based ensemble
            prediction_matrix = np.column_stack(predictions)
            vote_scores = np.mean(prediction_matrix.astype(float), axis=1)
            
            # Score-based ensemble
            if scores_list:
                score_matrix = np.column_stack(scores_list)
                ensemble_scores = np.mean(score_matrix, axis=1)
            else:
                ensemble_scores = vote_scores

            # Determine outliers using adaptive threshold
            # Use contamination rate but allow for some flexibility
            base_threshold = 1 - config.outlier_contamination
            
            # Adjust threshold based on score distribution
            if ensemble_scores.std() > 0:
                # If scores have good spread, use percentile-based threshold
                threshold = np.percentile(ensemble_scores, base_threshold * 100)
            else:
                # If scores are uniform, use majority voting
                threshold = 0.5

            ensemble_outliers = ensemble_scores > threshold

            # Consensus strength (how much methods agree)
            consensus_strength = np.std(prediction_matrix.astype(float), axis=1)
            
            return {
                'outliers': ensemble_outliers,
                'scores': ensemble_scores,
                'vote_scores': vote_scores,
                'consensus_strength': consensus_strength,
                'threshold': threshold,
                'participating_methods': list(all_results.keys()),
                'method_type': 'ensemble'
            }

        except Exception as e:
            print(f"Ensemble outlier detection failed: {e}")
            return {}

    def _evaluate_outlier_detection(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Evaluate outlier detection results"""
        evaluations = {}
        
        for method_name, result in all_results.items():
            if 'outliers' not in result:
                continue
            
            evaluation = {}
            outliers = result['outliers']
            
            try:
                # Basic statistics
                n_outliers = int(outliers.sum())
                outlier_rate = float(n_outliers / len(outliers))
                
                evaluation.update({
                    'n_outliers': n_outliers,
                    'outlier_rate': outlier_rate,
                    'inlier_rate': 1 - outlier_rate
                })

                # Separation quality (if scores available)
                if 'scores' in result and result['scores'] is not None:
                    scores = np.array(result['scores'])
                    outlier_scores = scores[outliers]
                    inlier_scores = scores[~outliers]
                    
                    if len(outlier_scores) > 0 and len(inlier_scores) > 0:
                        # Score separation
                        separation = np.mean(outlier_scores) - np.mean(inlier_scores)
                        evaluation['score_separation'] = float(separation)
                        
                        # Score overlap (using 95th percentile of inliers vs 5th percentile of outliers)
                        if len(outlier_scores) > 1 and len(inlier_scores) > 1:
                            inlier_95 = np.percentile(inlier_scores, 95)
                            outlier_5 = np.percentile(outlier_scores, 5)
                            overlap = max(0, inlier_95 - outlier_5)
                            evaluation['score_overlap'] = float(overlap)

                # Spatial distribution analysis
                if n_outliers > 0 and len(data) > n_outliers:
                    outlier_data = data[outliers]
                    inlier_data = data[~outliers]
                    
                    # Average distance from outliers to nearest inlier
                    from scipy.spatial.distance import cdist
                    if len(inlier_data) > 0:
                        distances = cdist(outlier_data, inlier_data)
                        min_distances = np.min(distances, axis=1)
                        evaluation['avg_distance_to_inliers'] = float(np.mean(min_distances))
                        evaluation['isolation_score'] = float(np.median(min_distances))

                # Method-specific evaluations
                method_type = result.get('method_type', 'unknown')
                evaluation['method_type'] = method_type
                
                if method_type == 'ensemble':
                    if 'consensus_strength' in result:
                        consensus = result['consensus_strength']
                        evaluation['avg_consensus'] = float(np.mean(consensus))
                        evaluation['consensus_std'] = float(np.std(consensus))

            except Exception as e:
                print(f"Evaluation failed for {method_name}: {e}")
                evaluation['error'] = str(e)
            
            evaluations[method_name] = evaluation

        return evaluations

    def _generate_outlier_recommendations(self, all_results: Dict[str, Any], 
                                        evaluations: Dict[str, Any], 
                                        preprocessing_info: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations for outlier detection"""
        recommendations = []

        # Find best method based on multiple criteria
        method_scores = {}
        for method_name, eval_metrics in evaluations.items():
            if 'error' in eval_metrics:
                continue
                
            score = 0
            
            # Score based on separation quality
            if 'score_separation' in eval_metrics:
                separation = eval_metrics['score_separation']
                score += min(separation / 2.0, 1.0) * 0.4  # Normalize and weight
            
            # Penalize excessive overlap
            if 'score_overlap' in eval_metrics:
                overlap = eval_metrics['score_overlap']
                score -= min(overlap, 1.0) * 0.2
            
            # Reward good isolation
            if 'isolation_score' in eval_metrics:
                isolation = eval_metrics['isolation_score']
                score += min(isolation / 5.0, 1.0) * 0.3  # Normalize to reasonable range
            
            # Consider outlier rate reasonableness (not too high or too low)
            outlier_rate = eval_metrics.get('outlier_rate', 0)
            if 0.01 <= outlier_rate <= 0.2:  # Reasonable range
                score += 0.1
            elif outlier_rate > 0.5:  # Too many outliers
                score -= 0.3
            
            method_scores[method_name] = max(0, score)

        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            best_eval = evaluations[best_method]
            
            recommendations.append(f"🏆 Best method: {best_method.upper()} (quality score: {best_score:.3f})")
            
            # Method-specific recommendations
            outlier_rate = best_eval.get('outlier_rate', 0)
            if outlier_rate > 0.3:
                recommendations.append("⚠️ High outlier rate detected - consider reviewing contamination parameter")
            elif outlier_rate < 0.01:
                recommendations.append("🔍 Very few outliers found - data may be very clean or threshold too strict")
            else:
                recommendations.append(f"✅ Reasonable outlier rate: {outlier_rate:.1%}")
            
            # Separation quality feedback
            if 'score_separation' in best_eval:
                separation = best_eval['score_separation']
                if separation > 1.0:
                    recommendations.append("📊 Excellent outlier-inlier separation detected")
                elif separation > 0.5:
                    recommendations.append("👍 Good separation between outliers and inliers")
                else:
                    recommendations.append("🤔 Moderate separation - outliers may be subtle")

        # Data-specific recommendations
        data_characteristics = preprocessing_info
        
        if data_characteristics.get('missing_data_percentage', 0) > 20:
            recommendations.append("📝 High missing data rate may affect outlier detection accuracy")
            
        if data_characteristics.get('data_skewness', 0) > 2:
            recommendations.append("📈 Highly skewed data - robust methods recommended")
            
        condition_number = data_characteristics.get('condition_number')
        if condition_number and condition_number > 1000:
            recommendations.append("🔧 Poor data conditioning - consider dimensionality reduction")

        # Method diversity recommendations
        successful_methods = len([e for e in evaluations.values() if 'error' not in e])
        if successful_methods <= 2:
            recommendations.append("🔄 Consider additional detection methods for robust analysis")
        elif successful_methods >= 5:
            recommendations.append("🤝 Multiple methods available - ensemble approach recommended")

        # Ensemble-specific recommendations
        if 'ensemble' in evaluations:
            ensemble_eval = evaluations['ensemble']
            avg_consensus = ensemble_eval.get('avg_consensus', 0)
            if avg_consensus < 0.3:
                recommendations.append("🎯 High consensus among methods - reliable outliers identified")
            else:
                recommendations.append("🤷 Low consensus among methods - results may vary")

        return recommendations[:5]  # Limit to top 5

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Main outlier detection analysis with comprehensive methods"""
        try:
            # Enhanced preprocessing
            scaled_data, complete_indices, preprocessing_info = self._adaptive_preprocessing(data, config)
            
            # Apply different detection methods
            all_results = {}
            
            # 1. Statistical methods
            try:
                statistical_results = self._statistical_outlier_detection(scaled_data, config)
                all_results.update(statistical_results)
            except Exception as e:
                print(f"Statistical outlier detection failed: {e}")
            
            # 2. Distance-based methods
            try:
                distance_results = self._distance_based_detection(scaled_data, config)
                all_results.update(distance_results)
            except Exception as e:
                print(f"Distance-based outlier detection failed: {e}")
            
            # 3. Machine learning methods
            try:
                ml_results = self._machine_learning_detection(scaled_data, config)
                all_results.update(ml_results)
            except Exception as e:
                print(f"ML-based outlier detection failed: {e}")
            
            # 4. Advanced methods (if libraries available)
            try:
                advanced_results = self._advanced_detection_methods(scaled_data, config)
                all_results.update(advanced_results)
            except Exception as e:
                print(f"Advanced outlier detection failed: {e}")
            
            # Map results back to original data indices
            final_results = {}
            full_length = len(data)
            
            for method_name, result in all_results.items():
                if 'outliers' in result:
                    # Create full-length mask
                    full_mask = np.full(full_length, False)
                    if len(complete_indices) > 0:
                        # Ensure we have a proper boolean array and valid indices
                        outlier_mask = np.asarray(result['outliers'], dtype=bool)
                        valid_indices = np.asarray(complete_indices, dtype=int)
                        
                        # Only map if dimensions match
                        if len(outlier_mask) == len(valid_indices):
                            full_mask[valid_indices] = outlier_mask
                    
                    mapped_result = {
                        'outliers': full_mask,
                        'count': int(full_mask.sum()),
                        'percentage': 100.0 * full_mask.sum() / full_length,
                        'method_type': result.get('method_type', 'unknown')
                    }
                    
                    # Map scores if available
                    if 'scores' in result and result['scores'] is not None:
                        full_scores = np.full(full_length, 0.0)
                        if len(complete_indices) > 0:
                            scores_array = np.asarray(result['scores'], dtype=float)
                            valid_indices = np.asarray(complete_indices, dtype=int)
                            
                            # Only map if dimensions match
                            if len(scores_array) == len(valid_indices):
                                full_scores[valid_indices] = scores_array
                        mapped_result['scores'] = full_scores
                    
                    # Copy other relevant information
                    for key in ['threshold', 'distances', 'k', 'eps', 'kernel', 'n_neighbors']:
                        if key in result:
                            mapped_result[key] = result[key]
                    
                    final_results[method_name] = mapped_result
            
            # 5. Ensemble method (using original scaled results for better accuracy)
            ensemble_result = self._ensemble_outlier_detection(all_results, scaled_data, config)
            if ensemble_result:
                # Map ensemble results to full data
                full_mask = np.full(full_length, False)
                full_scores = np.full(full_length, 0.0)
                
                if len(complete_indices) > 0:
                    ensemble_outliers = np.asarray(ensemble_result['outliers'], dtype=bool)
                    ensemble_scores = np.asarray(ensemble_result['scores'], dtype=float)
                    valid_indices = np.asarray(complete_indices, dtype=int)
                    
                    # Only map if dimensions match
                    if len(ensemble_outliers) == len(valid_indices):
                        full_mask[valid_indices] = ensemble_outliers
                    if len(ensemble_scores) == len(valid_indices):
                        full_scores[valid_indices] = ensemble_scores
                
                final_results['ensemble'] = {
                    'outliers': full_mask,
                    'count': int(full_mask.sum()),
                    'percentage': 100.0 * full_mask.sum() / full_length,
                    'scores': full_scores,
                    'method_type': 'ensemble',
                    'participating_methods': ensemble_result['participating_methods'],
                    'consensus_strength': ensemble_result.get('consensus_strength', np.array([]))
                }
            
            # Evaluate results
            evaluations = self._evaluate_outlier_detection(all_results, scaled_data)
            
            # Generate recommendations
            recommendations = self._generate_outlier_recommendations(
                final_results, evaluations, preprocessing_info
            )
            
            # Compile comprehensive results
            analysis_results = {
                'outlier_results': final_results,
                'evaluations': evaluations,
                'preprocessing_info': preprocessing_info,
                'data_characteristics': {
                    'total_samples': full_length,
                    'analyzed_samples': len(scaled_data),
                    'missing_samples': full_length - len(scaled_data),
                    'n_features': scaled_data.shape[1],
                    'contamination_rate': config.outlier_contamination
                },
                'recommendations': recommendations,
                'summary': {
                    'methods_attempted': len(all_results),
                    'successful_methods': len(final_results),
                    'best_method': max(evaluations.keys(), 
                                     key=lambda x: evaluations[x].get('score_separation', 0)) if evaluations else None,
                    'overall_outlier_rate': np.mean([r['percentage'] for r in final_results.values()]) / 100
                }
            }
            
            return analysis_results
            
        except Exception as e:
            return {'error': f'Outlier analysis failed: {str(e)}'}


# Utility functions for external use
def compare_outlier_methods(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a comparison table of outlier detection methods"""
    if 'evaluations' not in results:
        return pd.DataFrame()
    
    comparison_data = []
    outlier_results = results.get('outlier_results', {})
    
    for method, evaluation in results['evaluations'].items():
        if method in outlier_results:
            result_data = outlier_results[method]
            row = {
                'Method': method,
                'Outliers_Found': result_data.get('count', 0),
                'Outlier_Rate': result_data.get('percentage', 0),
                'Score_Separation': evaluation.get('score_separation', 0),
                'Score_Overlap': evaluation.get('score_overlap', 0),
                'Isolation_Score': evaluation.get('isolation_score', 0),
                'Method_Type': evaluation.get('method_type', 'unknown')
            }
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df.sort_values('Score_Separation', ascending=False) if not df.empty else df


def extract_outlier_profiles(data: pd.DataFrame, outlier_mask: np.ndarray, 
                           method_name: str = 'outlier_detection') -> Dict[str, Any]:
    """Extract statistical profiles of detected outliers vs inliers"""
    numeric_data = data.select_dtypes(include=[np.number])
    
    outliers = numeric_data[outlier_mask]
    inliers = numeric_data[~outlier_mask]
    
    profile = {
        'method': method_name,
        'n_outliers': int(outlier_mask.sum()),
        'n_inliers': int((~outlier_mask).sum()),
        'outlier_percentage': float(outlier_mask.sum() / len(outlier_mask) * 100),
        'feature_profiles': {}
    }
    
    # Compare feature distributions
    for col in numeric_data.columns:
        if len(outliers) > 0 and len(inliers) > 0:
            outlier_stats = {
                'mean': float(outliers[col].mean()),
                'std': float(outliers[col].std()),
                'median': float(outliers[col].median()),
                'min': float(outliers[col].min()),
                'max': float(outliers[col].max())
            }
            
            inlier_stats = {
                'mean': float(inliers[col].mean()),
                'std': float(inliers[col].std()),
                'median': float(inliers[col].median()),
                'min': float(inliers[col].min()),
                'max': float(inliers[col].max())
            }
            
            # Calculate difference metrics
            mean_diff = abs(outlier_stats['mean'] - inlier_stats['mean'])
            std_ratio = outlier_stats['std'] / inlier_stats['std'] if inlier_stats['std'] > 0 else 1
            
            profile['feature_profiles'][col] = {
                'outlier_stats': outlier_stats,
                'inlier_stats': inlier_stats,
                'mean_difference': float(mean_diff),
                'std_ratio': float(std_ratio)
            }
    
    return profile


def get_outlier_consensus(results: Dict[str, Any], min_methods: int = 2) -> np.ndarray:
    """Get consensus outliers that are detected by multiple methods"""
    if 'outlier_results' not in results:
        return np.array([])
    
    outlier_results = results['outlier_results']
    method_predictions = []
    
    for method_name, result in outlier_results.items():
        if 'outliers' in result:
            method_predictions.append(result['outliers'])
    
    if len(method_predictions) < min_methods:
        return np.array([])
    
    # Stack predictions and count votes
    prediction_matrix = np.column_stack(method_predictions)
    vote_counts = np.sum(prediction_matrix, axis=1)
    
    # Return outliers detected by at least min_methods
    consensus_outliers = vote_counts >= min_methods
    return consensus_outliers


warnings.filterwarnings('ignore')

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

try:
    from pyclustering.cluster.optics import optics
    from pyclustering.cluster.xmeans import xmeans
    HAS_PYCLUSTERING = True
except ImportError:
    HAS_PYCLUSTERING = False
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

try:
    from pyclustering.cluster.optics import optics
    from pyclustering.cluster.xmeans import xmeans
    HAS_PYCLUSTERING = True
except ImportError:
    HAS_PYCLUSTERING = False


warnings.filterwarnings('ignore')

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

try:
    from pyclustering.cluster.optics import optics
    from pyclustering.cluster.xmeans import xmeans
    HAS_PYCLUSTERING = True
except ImportError:
    HAS_PYCLUSTERING = False

warnings.filterwarnings('ignore')

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

try:
    from pyclustering.cluster.optics import optics
    from pyclustering.cluster.xmeans import xmeans
    HAS_PYCLUSTERING = True
except ImportError:
    HAS_PYCLUSTERING = False


class ClusterAnalyzer(AnalysisStrategy):
    """State-of-the-art clustering analysis with comprehensive evaluation and stability assessment"""

    @property
    def name(self) -> str:
        return "clusters"

    def _adaptive_preprocessing(self, data: pd.DataFrame, config: AnalysisConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced preprocessing with outlier detection and feature scaling"""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        if numeric_data.empty:
            raise ValueError("No numeric data available for clustering")

        preprocessing_info = {}
        
        # Outlier detection using IQR method
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
        n_outliers = outlier_mask.sum()
        preprocessing_info['outliers_detected'] = int(n_outliers)
        preprocessing_info['outlier_percentage'] = float(n_outliers / len(numeric_data) * 100)

        # Adaptive scaling based on data characteristics
        skewness = numeric_data.skew().abs().mean()
        outlier_fraction = n_outliers / len(numeric_data)
        
        if skewness > 2 or outlier_fraction > 0.1:
            try:
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                scaled_data = scaler.fit_transform(numeric_data)
                preprocessing_info['scaling_method'] = 'power_transform'
            except:
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                preprocessing_info['scaling_method'] = 'robust'
        else:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            preprocessing_info['scaling_method'] = 'standard'

        # Dimensionality assessment
        n_samples, n_features = scaled_data.shape
        preprocessing_info['curse_of_dimensionality_risk'] = n_features > n_samples / 3
        
        # PCA for high-dimensional data
        if n_features > 50 and n_samples > n_features:
            pca = PCA(n_components=min(50, n_samples // 2), random_state=config.random_state)
            scaled_data = pca.fit_transform(scaled_data)
            preprocessing_info['pca_applied'] = True
            preprocessing_info['pca_variance_explained'] = float(pca.explained_variance_ratio_.sum())
            preprocessing_info['final_dimensions'] = scaled_data.shape[1]
        else:
            preprocessing_info['pca_applied'] = False

        preprocessing_info['final_shape'] = scaled_data.shape
        return scaled_data, preprocessing_info

    def _estimate_optimal_clusters(self, data: np.ndarray, max_k: int = 15) -> Dict[str, Any]:
        """Multi-method optimal cluster number estimation"""
        n_samples = len(data)
        max_k = min(max_k, n_samples // 3)
        
        if max_k < 2:
            return {'optimal_k': 2, 'methods': {}, 'confidence': 'low'}

        methods_results = {}
        
        # 1. Elbow Method (Within-cluster sum of squares)
        wcss = []
        k_range = range(1, max_k + 1)
        for k in k_range:
            if k == 1:
                wcss.append(np.sum(np.var(data, axis=0)) * len(data))
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
        
        # Find elbow using second derivative
        if len(wcss) >= 3:
            second_diff = np.diff(np.diff(wcss))
            elbow_k = np.argmax(second_diff) + 2  # +2 because of double diff
            methods_results['elbow'] = min(max_k, max(2, elbow_k))

        # 2. Silhouette Analysis
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append((k, score))
        
        if silhouette_scores:
            best_sil_k = max(silhouette_scores, key=lambda x: x[1])[0]
            methods_results['silhouette'] = best_sil_k

        # 3. Calinski-Harabasz Index
        ch_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = calinski_harabasz_score(data, labels)
            ch_scores.append((k, score))
        
        if ch_scores:
            best_ch_k = max(ch_scores, key=lambda x: x[1])[0]
            methods_results['calinski_harabasz'] = best_ch_k

        # 4. Davies-Bouldin Index (lower is better)
        db_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = davies_bouldin_score(data, labels)
            db_scores.append((k, score))
        
        if db_scores:
            best_db_k = min(db_scores, key=lambda x: x[1])[0]
            methods_results['davies_bouldin'] = best_db_k

        # 5. Gap Statistic (simplified version)
        try:
            gap_stats = []
            for k in range(1, max_k + 1):
                if k == 1:
                    intra_disp = np.sum(np.var(data, axis=0)) * len(data)
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(data)
                    intra_disp = kmeans.inertia_

                # Reference distribution (uniform random)
                ref_disps = []
                for _ in range(5):  # Reduced iterations for speed
                    random_data = np.random.uniform(
                        data.min(axis=0), data.max(axis=0), size=data.shape
                    )
                    if k == 1:
                        ref_disp = np.sum(np.var(random_data, axis=0)) * len(random_data)
                    else:
                        ref_kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                        ref_kmeans.fit(random_data)
                        ref_disp = ref_kmeans.inertia_
                    ref_disps.append(ref_disp)

                gap = np.log(np.mean(ref_disps)) - np.log(intra_disp)
                gap_stats.append((k, gap))

            # Find k where gap(k) >= gap(k+1) - std(gap(k+1))
            for i in range(len(gap_stats) - 1):
                k, gap_k = gap_stats[i]
                k_plus_1, gap_k_plus_1 = gap_stats[i + 1]
                if gap_k >= gap_k_plus_1 - 0.1:  # Simplified std approximation
                    methods_results['gap_statistic'] = k
                    break

        except Exception as e:
            print(f"Gap statistic calculation failed: {e}")

        # Consensus estimation
        if methods_results:
            method_values = list(methods_results.values())
            # Weight different methods
            weights = {
                'silhouette': 0.3,
                'calinski_harabasz': 0.25,
                'davies_bouldin': 0.2,
                'elbow': 0.15,
                'gap_statistic': 0.1
            }
            
            weighted_sum = sum(weights.get(method, 0.2) * k 
                             for method, k in methods_results.items())
            optimal_k = max(2, min(max_k, int(np.round(weighted_sum))))
            
            # Calculate confidence based on agreement
            variance = np.var(method_values)
            if variance < 0.5:
                confidence = 'high'
            elif variance < 2.0:
                confidence = 'medium'
            else:
                confidence = 'low'
        else:
            optimal_k = min(3, max_k)
            confidence = 'low'

        return {
            'optimal_k': optimal_k,
            'methods': methods_results,
            'confidence': confidence,
            'method_agreement': len(set(method_values)) if methods_results else 0
        }

    def _enhanced_kmeans_analysis(self, data: np.ndarray, config: AnalysisConfig, 
                                optimal_k_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced K-means with multiple initializations and stability analysis"""
        max_k = min(config.max_clusters, len(data) // 3)
        k_range = range(2, max_k + 1)
        
        all_results = []
        best_model = None
        best_k = optimal_k_info['optimal_k']
        best_score = -1

        for k in k_range:
            # Multiple runs with different initializations
            stability_scores = []
            models = []
            
            for init_method in ['k-means++', 'random']:
                try:
                    kmeans = KMeans(
                        n_clusters=k, 
                        init=init_method,
                        n_init=20,
                        max_iter=500,
                        random_state=config.random_state,
                        algorithm='lloyd'
                    )
                    labels = kmeans.fit_predict(data)
                    
                    # Stability check: run again and compare
                    kmeans_stable = KMeans(
                        n_clusters=k, 
                        init=init_method,
                        n_init=20,
                        max_iter=500,
                        random_state=config.random_state + 1
                    )
                    labels_stable = kmeans_stable.fit_predict(data)
                    stability = adjusted_rand_score(labels, labels_stable)
                    stability_scores.append(stability)
                    models.append((kmeans, labels))
                    
                except Exception as e:
                    print(f"K-means failed for k={k}, init={init_method}: {e}")
                    continue

            if not models:
                continue

            # Select best model for this k
            best_model_k = max(models, key=lambda x: silhouette_score(data, x[1]) if len(set(x[1])) > 1 else -1)
            kmeans, labels = best_model_k

            # Comprehensive scoring
            try:
                sil_score = silhouette_score(data, labels)
                ch_score = calinski_harabasz_score(data, labels)
                db_score = davies_bouldin_score(data, labels)
                avg_stability = np.mean(stability_scores) if stability_scores else 0
                
                # Cluster balance score (penalize very unbalanced clusters)
                cluster_sizes = np.bincount(labels)
                balance_score = 1 - np.std(cluster_sizes) / np.mean(cluster_sizes)
                
                result = {
                    'k': k,
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'inertia': kmeans.inertia_,
                    'stability': avg_stability,
                    'balance': balance_score,
                    'cluster_sizes': dict(Counter(labels)),
                    'model': kmeans,
                    'labels': labels
                }
                
                all_results.append(result)
                
                # Update best model based on composite score
                composite_score = (sil_score * 0.4 + avg_stability * 0.3 + 
                                 balance_score * 0.2 + (1 - db_score/10) * 0.1)
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_k = k
                    best_model = kmeans

            except Exception as e:
                print(f"Scoring failed for k={k}: {e}")
                continue

        final_labels = best_model.fit_predict(data) if best_model else np.zeros(len(data))
        
        return {
            'labels': final_labels,
            'model': best_model,
            'scores': all_results,
            'best_k': best_k,
            'centers': best_model.cluster_centers_ if best_model else None,
            'cluster_sizes': dict(Counter(final_labels)),
            'method_type': 'centroid_based'
        }

    def _hierarchical_clustering_analysis(self, data: np.ndarray, config: AnalysisConfig,
                                        optimal_k_info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive hierarchical clustering with multiple linkage methods"""
        results = {}
        n_samples = len(data)
        
        if n_samples > 1000:
            # For large datasets, use a sample for linkage computation
            sample_idx = np.random.choice(n_samples, 1000, replace=False)
            sample_data = data[sample_idx]
        else:
            sample_data = data
            sample_idx = np.arange(n_samples)

        linkage_methods = ['ward', 'complete', 'average', 'single']
        best_linkage = None
        best_score = -1
        
        for method in linkage_methods:
            try:
                # Compute linkage matrix
                if method == 'ward':
                    distance_matrix = pdist(sample_data, metric='euclidean')
                else:
                    distance_matrix = pdist(sample_data, metric='euclidean')
                
                linkage_matrix = linkage(distance_matrix, method=method)
                
                # Get clusters for optimal k
                optimal_k = optimal_k_info['optimal_k']
                cluster_labels_sample = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
                
                # Apply to full dataset using AgglomerativeClustering
                agg_clustering = AgglomerativeClustering(
                    n_clusters=optimal_k,
                    linkage=method,
                    compute_distances=True
                )
                full_labels = agg_clustering.fit_predict(data)
                
                # Evaluate clustering quality
                sil_score = silhouette_score(data, full_labels)
                ch_score = calinski_harabasz_score(data, full_labels)
                db_score = davies_bouldin_score(data, full_labels)
                
                result = {
                    'linkage_method': method,
                    'labels': full_labels,
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'cluster_sizes': dict(Counter(full_labels)),
                    'model': agg_clustering,
                    'linkage_matrix': linkage_matrix,
                    'method_type': 'hierarchical'
                }
                
                results[f'hierarchical_{method}'] = result
                
                if sil_score > best_score:
                    best_score = sil_score
                    best_linkage = method
                    
            except Exception as e:
                print(f"Hierarchical clustering with {method} failed: {e}")
                continue

        # Return best hierarchical clustering result
        best_result = results.get(f'hierarchical_{best_linkage}', {})
        if best_result:
            best_result['best_linkage_method'] = best_linkage
            
        return best_result

    def _density_based_clustering(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Enhanced density-based clustering with parameter optimization"""
        results = {}
        
        # DBSCAN with automated parameter selection
        try:
            # Estimate eps using k-distance graph
            k = min(4, len(data) // 10)
            nbrs = NearestNeighbors(n_neighbors=k).fit(data)
            distances, indices = nbrs.kneighbors(data)
            distances = np.sort(distances[:, k-1], axis=0)
            
            # Use knee/elbow detection for eps
            diff = np.diff(distances)
            if len(diff) > 0:
                eps = distances[np.argmax(diff)]
            else:
                eps = np.median(distances)
            
            min_samples = max(2, k)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = int((labels == -1).sum())
            
            result = {
                'labels': labels,
                'model': dbscan,
                'n_clusters': n_clusters,
                'noise_points': noise_points,
                'eps': eps,
                'min_samples': min_samples,
                'cluster_sizes': dict(Counter(labels[labels != -1])),
                'method_type': 'density_based'
            }
            
            # Calculate silhouette only for non-noise points
            if n_clusters > 1:
                valid_mask = labels != -1
                if valid_mask.sum() > 1:
                    valid_data = data[valid_mask]
                    valid_labels = labels[valid_mask]
                    result['silhouette'] = silhouette_score(valid_data, valid_labels)
            
            results['dbscan'] = result
            
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")

        # HDBSCAN if available
        if HAS_HDBSCAN:
            try:
                min_cluster_size = max(5, len(data) // 50)
                min_samples = max(1, min_cluster_size // 2)
                
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=0.0,
                    cluster_selection_method='eom'
                )
                labels = hdbscan_model.fit_predict(data)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_points = int((labels == -1).sum())
                
                result = {
                    'labels': labels,
                    'model': hdbscan_model,
                    'n_clusters': n_clusters,
                    'noise_points': noise_points,
                    'min_cluster_size': min_cluster_size,
                    'cluster_sizes': dict(Counter(labels[labels != -1])),
                    'method_type': 'density_based',
                    'probabilities': hdbscan_model.probabilities_.tolist() if hasattr(hdbscan_model, 'probabilities_') else None
                }
                
                if n_clusters > 1:
                    valid_mask = labels != -1
                    if valid_mask.sum() > 1:
                        valid_data = data[valid_mask]
                        valid_labels = labels[valid_mask]
                        result['silhouette'] = silhouette_score(valid_data, valid_labels)
                
                results['hdbscan'] = result
                
            except Exception as e:
                print(f"HDBSCAN clustering failed: {e}")

        return results

    def _probabilistic_clustering(self, data: np.ndarray, config: AnalysisConfig,
                                optimal_k_info: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced probabilistic clustering with model selection"""
        results = {}
        max_k = min(config.max_clusters, len(data) // 2)
        
        # Gaussian Mixture Models with comprehensive model selection
        try:
            covariance_types = ['full', 'tied', 'diag', 'spherical']
            best_bic = np.inf
            best_aic = np.inf
            best_gmm = None
            best_k = optimal_k_info['optimal_k']
            model_comparison = []
            
            for cov_type in covariance_types:
                for k in range(2, max_k + 1):
                    try:
                        gmm = GaussianMixture(
                            n_components=k,
                            covariance_type=cov_type,
                            max_iter=200,
                            n_init=3,
                            random_state=config.random_state
                        )
                        gmm.fit(data)
                        
                        bic = gmm.bic(data)
                        aic = gmm.aic(data)
                        log_likelihood = gmm.score(data)
                        
                        model_info = {
                            'k': k,
                            'covariance_type': cov_type,
                            'bic': bic,
                            'aic': aic,
                            'log_likelihood': log_likelihood,
                            'converged': gmm.converged_
                        }
                        model_comparison.append(model_info)
                        
                        if bic < best_bic and gmm.converged_:
                            best_bic = bic
                            best_gmm = gmm
                            best_k = k
                            
                    except Exception:
                        continue
            
            if best_gmm is not None:
                labels = best_gmm.predict(data)
                probs = best_gmm.predict_proba(data)
                
                result = {
                    'labels': labels,
                    'probabilities': probs.tolist(),
                    'model': best_gmm,
                    'best_k': best_k,
                    'best_bic': best_bic,
                    'model_comparison': model_comparison,
                    'cluster_sizes': dict(Counter(labels)),
                    'silhouette': silhouette_score(data, labels),
                    'method_type': 'probabilistic',
                    'covariance_type': best_gmm.covariance_type,
                    'means': best_gmm.means_.tolist(),
                    'covariances': best_gmm.covariances_.tolist()
                }
                results['gmm'] = result
                
        except Exception as e:
            print(f"GMM clustering failed: {e}")

        # Bayesian Gaussian Mixture (if computational resources allow)
        try:
            if len(data) <= 5000:  # Only for smaller datasets due to computational cost
                bgmm = BayesianGaussianMixture(
                    n_components=min(10, max_k),
                    covariance_type='full',
                    max_iter=200,
                    random_state=config.random_state
                )
                bgmm.fit(data)
                labels = bgmm.predict(data)
                
                # Count effective components (with significant weight)
                effective_components = np.sum(bgmm.weights_ > 0.01)
                
                result = {
                    'labels': labels,
                    'probabilities': bgmm.predict_proba(data).tolist(),
                    'model': bgmm,
                    'effective_components': int(effective_components),
                    'weights': bgmm.weights_.tolist(),
                    'cluster_sizes': dict(Counter(labels)),
                    'silhouette': silhouette_score(data, labels),
                    'method_type': 'bayesian_probabilistic'
                }
                results['bayesian_gmm'] = result
                
        except Exception as e:
            print(f"Bayesian GMM clustering failed: {e}")

        return results

    def _ensemble_clustering(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Ensemble clustering using consensus from multiple methods"""
        if not all_results:
            return {}

        try:
            # Collect all label assignments
            label_sets = []
            method_weights = {}
            
            for method_name, result in all_results.items():
                if 'labels' in result and len(result['labels']) == len(data):
                    labels = np.array(result['labels'])
                    label_sets.append(labels)
                    
                    # Weight methods based on their silhouette scores
                    sil_score = result.get('silhouette', 0)
                    method_weights[method_name] = max(0, sil_score)

            if len(label_sets) < 2:
                return {}

            # Create co-association matrix
            n_samples = len(data)
            co_assoc = np.zeros((n_samples, n_samples))
            
            for labels in label_sets:
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        if labels[i] == labels[j] and labels[i] != -1:  # Ignore noise points
                            co_assoc[i, j] += 1
                            co_assoc[j, i] += 1

            # Normalize co-association matrix
            co_assoc /= len(label_sets)

            # Apply hierarchical clustering on co-association matrix
            # Convert co-association to distance matrix (1 - similarity)
            distance_matrix = 1 - co_assoc
            
            # Ensure diagonal is exactly zero (fix floating point precision issues)
            np.fill_diagonal(distance_matrix, 0)
            
            # Ensure matrix is symmetric
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            
            # Convert to condensed form for linkage
            condensed_distance = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distance, method='average')
            
            # Determine optimal number of clusters for consensus
            cluster_counts = []
            for labels in label_sets:
                # Convert to numpy array for consistent handling
                labels_array = np.array(labels)
                # Get unique non-noise labels
                unique_labels = np.unique(labels_array[labels_array != -1])
                n_clusters = len(unique_labels)
                if n_clusters > 1:  # Only count valid clusterings
                    cluster_counts.append(n_clusters)
            
            if not cluster_counts:
                # Fallback: use 3 clusters if no valid clusterings
                optimal_k = 3
            else:
                # Use most common cluster count
                from collections import Counter
                cluster_counter = Counter(cluster_counts)
                optimal_k = cluster_counter.most_common(1)[0][0]
            
            # Get consensus clustering
            consensus_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust') - 1
            
            # Evaluate consensus clustering
            consensus_result = {
                'labels': consensus_labels,
                'n_methods': len(label_sets),
                'co_association_matrix': co_assoc.tolist(),
                'consensus_k': optimal_k,
                'cluster_sizes': dict(Counter(consensus_labels)),
                'silhouette': silhouette_score(data, consensus_labels),
                'method_type': 'ensemble',
                'participating_methods': list(all_results.keys())
            }
            
            return consensus_result
            
        except Exception as e:
            print(f"Ensemble clustering failed: {e}")
            # Return empty dict instead of error to avoid breaking the analysis
            return {}

    def _evaluate_clustering_results(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation of clustering results"""
        evaluations = {}
        
        for method_name, result in all_results.items():
            if 'labels' not in result:
                continue
                
            labels = np.array(result['labels'])
            evaluation = {}
            
            try:
                # Basic metrics
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = int((labels == -1).sum())
                
                evaluation.update({
                    'n_clusters': n_clusters,
                    'n_noise_points': n_noise,
                    'noise_ratio': n_noise / len(labels)
                })
                
                # Silhouette analysis (skip if already computed)
                if 'silhouette' not in result and n_clusters > 1:
                    valid_mask = labels != -1
                    if valid_mask.sum() > 1:
                        valid_data = data[valid_mask]
                        valid_labels = labels[valid_mask]
                        evaluation['silhouette_score'] = silhouette_score(valid_data, valid_labels)
                else:
                    evaluation['silhouette_score'] = result.get('silhouette', 0)

                # Additional metrics for non-noise clusters
                if n_clusters > 1:
                    valid_mask = labels != -1
                    valid_data = data[valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    if len(valid_data) > 0:
                        evaluation.update({
                            'calinski_harabasz_score': calinski_harabasz_score(valid_data, valid_labels),
                            'davies_bouldin_score': davies_bouldin_score(valid_data, valid_labels)
                        })
                        
                        # Cluster balance and separation
                        cluster_sizes = np.bincount(valid_labels)
                        balance_score = 1 - np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
                        evaluation['cluster_balance'] = balance_score
                        
                        # Intra-cluster vs inter-cluster distance ratio
                        unique_valid_labels = np.unique(valid_labels)
                        if len(unique_valid_labels) >= 2:
                            intra_distances = []
                            inter_distances = []
                            
                            for label in unique_valid_labels:
                                cluster_mask = valid_labels == label
                                cluster_data = valid_data[cluster_mask]
                                
                                if len(cluster_data) > 1:
                                    # Intra-cluster distances
                                    intra_dist = pdist(cluster_data)
                                    intra_distances.extend(intra_dist)
                                    
                                    # Inter-cluster distances
                                    other_data = valid_data[~cluster_mask]
                                    if len(other_data) > 0:
                                        for point in cluster_data:
                                            distances_to_others = np.linalg.norm(other_data - point, axis=1)
                                            inter_distances.extend(distances_to_others)
                            
                            if intra_distances and inter_distances:
                                mean_intra = np.mean(intra_distances)
                                mean_inter = np.mean(inter_distances)
                                separation_ratio = mean_inter / mean_intra if mean_intra > 0 else 0
                                evaluation['separation_ratio'] = separation_ratio
                
                # Stability metrics (if multiple runs were performed)
                if 'stability' in result:
                    evaluation['stability_score'] = result['stability']
                
                # Method-specific metrics
                method_type = result.get('method_type', 'unknown')
                evaluation['method_type'] = method_type
                
                if method_type == 'probabilistic' and 'probabilities' in result:
                    # Entropy of cluster assignments (uncertainty)
                    probs = np.array(result['probabilities'])
                    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    evaluation['mean_assignment_entropy'] = float(np.mean(entropy))
                    evaluation['assignment_uncertainty'] = float(np.std(entropy))
                
            except Exception as e:
                print(f"Evaluation failed for {method_name}: {e}")
                evaluation['error'] = str(e)
            
            evaluations[method_name] = evaluation
            
        return evaluations

    def _generate_clustering_recommendations(self, all_results: Dict[str, Any], 
                                           evaluations: Dict[str, Any],
                                           optimal_k_info: Dict[str, Any],
                                           preprocessing_info: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on clustering results"""
        recommendations = []
        
        # Find best performing method
        method_scores = {}
        for method_name, eval_metrics in evaluations.items():
            if 'error' in eval_metrics:
                continue
                
            score = 0
            weight_sum = 0
            
            # Weighted scoring
            if 'silhouette_score' in eval_metrics:
                score += eval_metrics['silhouette_score'] * 0.3
                weight_sum += 0.3
            
            if 'cluster_balance' in eval_metrics:
                score += eval_metrics['cluster_balance'] * 0.2
                weight_sum += 0.2
                
            if 'separation_ratio' in eval_metrics:
                # Normalize separation ratio (higher is better, but cap at reasonable value)
                norm_sep = min(eval_metrics['separation_ratio'] / 5.0, 1.0)
                score += norm_sep * 0.2
                weight_sum += 0.2
                
            if 'stability_score' in eval_metrics:
                score += eval_metrics['stability_score'] * 0.2
                weight_sum += 0.2
                
            # Penalize excessive noise
            noise_penalty = eval_metrics.get('noise_ratio', 0)
            score -= noise_penalty * 0.1
            weight_sum += 0.1
            
            if weight_sum > 0:
                method_scores[method_name] = score / weight_sum
        
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            recommendations.append(f"🏆 Best method: {best_method.upper()} (score: {best_score:.3f})")
            
            # Method-specific recommendations
            best_result = all_results.get(best_method, {})
            method_type = best_result.get('method_type', '')
            
            if 'density' in method_type:
                noise_ratio = evaluations[best_method].get('noise_ratio', 0)
                if noise_ratio > 0.1:
                    recommendations.append(f"⚠️ High noise ratio ({noise_ratio:.1%}) - consider adjusting density parameters")
                else:
                    recommendations.append("✅ Density-based clustering effectively identified cluster structure")
            
            elif method_type == 'probabilistic':
                uncertainty = evaluations[best_method].get('assignment_uncertainty', 0)
                if uncertainty > 0.5:
                    recommendations.append("🔀 High assignment uncertainty suggests overlapping clusters")
                else:
                    recommendations.append("📊 Probabilistic clustering shows confident cluster assignments")
                    
            elif method_type == 'hierarchical':
                linkage_method = best_result.get('best_linkage_method', 'unknown')
                recommendations.append(f"🌳 Hierarchical clustering with {linkage_method} linkage worked best")
                
            elif method_type == 'ensemble':
                n_methods = best_result.get('n_methods', 0)
                recommendations.append(f"🤝 Ensemble of {n_methods} methods achieved robust consensus")
        
        # Data-specific recommendations
        if preprocessing_info.get('curse_of_dimensionality_risk', False):
            recommendations.append("📏 High-dimensional data detected - consider dimensionality reduction")
            
        if preprocessing_info.get('outlier_percentage', 0) > 10:
            recommendations.append("🎯 Many outliers detected - density-based methods recommended")
            
        # Optimal k analysis
        k_confidence = optimal_k_info.get('confidence', 'low')
        optimal_k = optimal_k_info.get('optimal_k', 2)
        
        if k_confidence == 'high':
            recommendations.append(f"✨ Strong evidence for {optimal_k} clusters")
        elif k_confidence == 'medium':
            recommendations.append(f"🤔 Moderate evidence for {optimal_k} clusters - consider range {optimal_k-1}-{optimal_k+1}")
        else:
            recommendations.append("❓ Unclear optimal cluster number - try multiple values")
            
        # Performance recommendations
        successful_methods = len([r for r in all_results.values() if 'labels' in r])
        if successful_methods <= 2:
            recommendations.append("⚡ Consider trying additional clustering algorithms for comparison")
            
        # Cluster interpretability
        if best_method in evaluations:
            n_clusters = evaluations[best_method].get('n_clusters', 0)
            if n_clusters > 10:
                recommendations.append("📊 Many clusters found - consider hierarchical visualization")
            elif n_clusters < 2:
                recommendations.append("🔍 No clear clusters found - data may not have cluster structure")
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Main clustering analysis with comprehensive methods and evaluation"""
        try:
            # Enhanced preprocessing
            scaled_data, preprocessing_info = self._adaptive_preprocessing(data, config)
            
            # Estimate optimal number of clusters
            optimal_k_info = self._estimate_optimal_clusters(scaled_data, config.max_clusters)
            
            # Apply different clustering methods
            all_results = {}
            
            # 1. Enhanced K-means
            try:
                kmeans_result = self._enhanced_kmeans_analysis(scaled_data, config, optimal_k_info)
                if kmeans_result:
                    all_results['kmeans'] = kmeans_result
            except Exception as e:
                print(f"Enhanced K-means failed: {e}")
            
            # 2. Hierarchical clustering
            try:
                hier_result = self._hierarchical_clustering_analysis(scaled_data, config, optimal_k_info)
                if hier_result:
                    all_results['hierarchical'] = hier_result
            except Exception as e:
                print(f"Hierarchical clustering failed: {e}")
            
            # 3. Density-based clustering
            try:
                density_results = self._density_based_clustering(scaled_data, config)
                all_results.update(density_results)
            except Exception as e:
                print(f"Density-based clustering failed: {e}")
            
            # 4. Probabilistic clustering
            try:
                prob_results = self._probabilistic_clustering(scaled_data, config, optimal_k_info)
                all_results.update(prob_results)
            except Exception as e:
                print(f"Probabilistic clustering failed: {e}")
            
            # 5. Additional methods if available
            # Spectral clustering for non-convex clusters
            try:
                if len(scaled_data) <= 2000:  # Computationally expensive
                    spectral = SpectralClustering(
                        n_clusters=optimal_k_info['optimal_k'],
                        random_state=config.random_state,
                        affinity='rbf',
                        gamma=1.0
                    )
                    spectral_labels = spectral.fit_predict(scaled_data)
                    
                    all_results['spectral'] = {
                        'labels': spectral_labels,
                        'model': spectral,
                        'n_clusters': optimal_k_info['optimal_k'],
                        'cluster_sizes': dict(Counter(spectral_labels)),
                        'silhouette': silhouette_score(scaled_data, spectral_labels),
                        'method_type': 'spectral'
                    }
            except Exception as e:
                print(f"Spectral clustering failed: {e}")
            
            # Mean Shift for automatic cluster detection
            try:
                if len(scaled_data) <= 1000:  # Computationally expensive
                    mean_shift = MeanShift()
                    ms_labels = mean_shift.fit_predict(scaled_data)
                    n_clusters_ms = len(set(ms_labels))
                    
                    if n_clusters_ms > 1:
                        all_results['mean_shift'] = {
                            'labels': ms_labels,
                            'model': mean_shift,
                            'n_clusters': n_clusters_ms,
                            'cluster_sizes': dict(Counter(ms_labels)),
                            'silhouette': silhouette_score(scaled_data, ms_labels),
                            'method_type': 'mean_shift'
                        }
            except Exception as e:
                print(f"Mean Shift clustering failed: {e}")
            
            # 6. Ensemble clustering
            ensemble_result = self._ensemble_clustering(all_results, scaled_data)
            if ensemble_result:
                all_results['ensemble'] = ensemble_result
            
            # Comprehensive evaluation
            evaluations = self._evaluate_clustering_results(all_results, scaled_data)
            
            # Generate recommendations
            recommendations = self._generate_clustering_recommendations(
                all_results, evaluations, optimal_k_info, preprocessing_info
            )
            
            # Compile final results
            final_results = {
                'clustering_results': all_results,
                'evaluations': evaluations,
                'optimal_k_analysis': optimal_k_info,
                'preprocessing_info': preprocessing_info,
                'data_characteristics': {
                    'n_samples': scaled_data.shape[0],
                    'n_features': scaled_data.shape[1],
                    'data_variance': float(np.var(scaled_data)),
                    'data_spread': float(np.ptp(scaled_data))
                },
                'recommendations': recommendations,
                'summary': {
                    'methods_attempted': len(all_results),
                    'successful_methods': len([r for r in all_results.values() if 'labels' in r]),
                    'best_method': max(evaluations.keys(), 
                                     key=lambda x: evaluations[x].get('silhouette_score', -1)) if evaluations else None
                }
            }
            
            return final_results
            
        except Exception as e:
            return {'error': f'Clustering analysis failed: {str(e)}'}

warnings.filterwarnings('ignore')

try:
    import umap.umap_ as UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from openTSNE import TSNE as OpenTSNE
    HAS_OPENTSNE = True
except ImportError:
    HAS_OPENTSNE = False

try:
    import trimap
    HAS_TRIMAP = True
except ImportError:
    HAS_TRIMAP = False

try:
    from MulticoreTSNE import MulticoreTSNE
    HAS_MULTICORE_TSNE = True
except ImportError:
    HAS_MULTICORE_TSNE = False


class DimensionalityAnalyzer(AnalysisStrategy):
    """State-of-the-art dimensionality reduction with adaptive preprocessing and ensemble methods"""

    @property
    def name(self) -> str:
        return "dimensionality"

    def _adaptive_preprocessing(self, data: pd.DataFrame, config: AnalysisConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advanced preprocessing with automatic method selection"""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        if numeric_data.empty or numeric_data.shape[1] < 2:
            raise ValueError("Insufficient numeric data for dimensionality reduction")

        preprocessing_info = {}
        
        # Remove low-variance features
        var_threshold = VarianceThreshold(threshold=0.01)
        numeric_data = pd.DataFrame(
            var_threshold.fit_transform(numeric_data),
            columns=numeric_data.columns[var_threshold.get_support()],
            index=numeric_data.index
        )
        preprocessing_info['features_removed'] = (~var_threshold.get_support()).sum()

        # Adaptive sampling with stratification if possible
        if len(numeric_data) > config.sample_size_threshold:
            # Try to preserve data distribution
            try:
                # Simple clustering-based stratified sampling
                kmeans = KMeans(n_clusters=min(10, len(numeric_data)//100), random_state=config.random_state)
                clusters = kmeans.fit_predict(StandardScaler().fit_transform(numeric_data))
                
                sample_indices = []
                for cluster_id in np.unique(clusters):
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    n_samples = max(1, int(config.sample_size_threshold * len(cluster_indices) / len(numeric_data)))
                    sample_indices.extend(
                        np.random.choice(cluster_indices, min(n_samples, len(cluster_indices)), replace=False)
                    )
                
                numeric_data = numeric_data.iloc[sample_indices[:config.sample_size_threshold]]
                preprocessing_info['sampling_method'] = 'stratified'
            except:
                # Fallback to random sampling
                sample_idx = np.random.choice(
                    len(numeric_data), config.sample_size_threshold, replace=False
                )
                numeric_data = numeric_data.iloc[sample_idx]
                preprocessing_info['sampling_method'] = 'random'

        # Intelligent scaling method selection
        scaling_methods = []
        
        # Check for normality and outliers
        skewness = numeric_data.skew().abs().mean()
        outlier_fraction = ((numeric_data - numeric_data.mean()).abs() > 3 * numeric_data.std()).sum().sum() / numeric_data.size
        
        if skewness > 2 or outlier_fraction > 0.1:
            # Heavy tails or many outliers - use robust methods
            scaling_methods = [
                ('power_transform', PowerTransformer(method='yeo-johnson', standardize=True)),
                ('robust', RobustScaler()),
                ('standard', StandardScaler())
            ]
        else:
            # Well-behaved data
            scaling_methods = [
                ('standard', StandardScaler()),
                ('robust', RobustScaler())
            ]

        # Try scaling methods and pick the best one
        best_scaler = None
        best_score = -np.inf
        
        for name, scaler in scaling_methods:
            try:
                scaled = scaler.fit_transform(numeric_data)
                # Score based on condition number and variance homogeneity
                cond_number = np.linalg.cond(np.cov(scaled.T))
                var_ratio = np.var(scaled, axis=0).max() / np.var(scaled, axis=0).min()
                score = -np.log(cond_number) - np.log(var_ratio)
                
                if score > best_score:
                    best_score = score
                    best_scaler = scaler
                    preprocessing_info['scaling_method'] = name
            except:
                continue

        if best_scaler is None:
            best_scaler = StandardScaler()
            preprocessing_info['scaling_method'] = 'standard_fallback'

        scaled_data = best_scaler.fit_transform(numeric_data)
        preprocessing_info['final_shape'] = scaled_data.shape
        
        return scaled_data, preprocessing_info

    def _compute_optimal_components(self, scaled_data: np.ndarray, max_components: int = 10) -> int:
        """Determine optimal number of components using multiple criteria"""
        n_samples, n_features = scaled_data.shape
        max_comp = min(max_components, n_features, n_samples // 2)
        
        if max_comp < 2:
            return min(2, n_features)

        # PCA explained variance analysis
        pca_full = PCA().fit(scaled_data)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Multiple criteria for component selection
        criteria_results = {}
        
        # 1. Explained variance threshold (95%)
        criteria_results['var_95'] = np.argmax(cumvar >= 0.95) + 1
        
        # 2. Elbow method on eigenvalues
        eigenvals = pca_full.explained_variance_
        if len(eigenvals) >= 3:
            diffs = np.diff(eigenvals)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                criteria_results['elbow'] = np.argmax(second_diffs) + 2
        
        # 3. Kaiser criterion (eigenvalues > 1)
        criteria_results['kaiser'] = np.sum(eigenvals > 1)
        
        # 4. Parallel analysis approximation
        random_eigenvals = []
        for _ in range(5):  # Reduced iterations for speed
            random_data = np.random.randn(*scaled_data.shape)
            random_pca = PCA().fit(random_data)
            random_eigenvals.append(random_pca.explained_variance_)
        
        mean_random_eigenvals = np.mean(random_eigenvals, axis=0)
        criteria_results['parallel'] = np.sum(eigenvals > mean_random_eigenvals[:len(eigenvals)])
        
        # Combine criteria with weights
        valid_results = {k: v for k, v in criteria_results.items() 
                        if v > 0 and v <= max_comp}
        
        if valid_results:
            # Weighted average, favoring conservative estimates
            weights = {'var_95': 0.3, 'elbow': 0.2, 'kaiser': 0.2, 'parallel': 0.3}
            weighted_sum = sum(weights.get(k, 0.25) * v for k, v in valid_results.items())
            optimal = max(2, min(max_comp, int(np.round(weighted_sum))))
        else:
            optimal = min(3, max_comp)
        
        return optimal

    def _enhanced_dimensionality_reduction(self, scaled_data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Apply state-of-the-art dimensionality reduction techniques"""
        results = {}
        n_samples, n_features = scaled_data.shape
        optimal_components = self._compute_optimal_components(scaled_data)
        
        # Linear methods
        linear_methods = {
            'pca': PCA(n_components=optimal_components, random_state=config.random_state),
            'ica': FastICA(n_components=optimal_components, random_state=config.random_state, 
                          max_iter=1000, tol=1e-4),
            'factor_analysis': FactorAnalysis(n_components=optimal_components, random_state=config.random_state),
        }
        
        # Add NMF for non-negative data
        if (scaled_data >= 0).all():
            linear_methods['nmf'] = NMF(n_components=optimal_components, random_state=config.random_state,
                                       max_iter=1000, tol=1e-4)
        
        # SVD for high-dimensional sparse data
        if n_features > 100:
            linear_methods['truncated_svd'] = TruncatedSVD(n_components=optimal_components, 
                                                          random_state=config.random_state)

        for name, method in linear_methods.items():
            try:
                embedding = method.fit_transform(scaled_data)
                results[name] = {
                    'embedding': embedding,
                    'method_type': 'linear',
                    'n_components': optimal_components
                }
                
                # Add explained variance for methods that support it
                if hasattr(method, 'explained_variance_ratio_'):
                    results[name]['explained_variance_ratio'] = method.explained_variance_ratio_
                    results[name]['total_variance_explained'] = method.explained_variance_ratio_.sum()
                    
            except Exception as e:
                print(f"Linear method {name} failed: {e}")

        # Non-linear methods with adaptive parameters
        target_dim = 2  # For visualization
        perplexity = min(30, max(5, n_samples // 4))
        n_neighbors = min(15, max(3, n_samples // 10))

        # Standard t-SNE
        tsne_params = {
            'n_components': target_dim,
            'random_state': config.random_state,
            'perplexity': perplexity,
            'init': 'pca',
            'learning_rate': 'auto',
            'n_iter': 1000,
            'early_exaggeration': 12,
            'min_grad_norm': 1e-7
        }

        # Use best available t-SNE implementation
        if HAS_OPENTSNE and n_samples > 1000:
            try:
                tsne = OpenTSNE(**tsne_params, n_jobs=4, negative_sample_rate=5)
                results['tsne_openai'] = {
                    'embedding': tsne.fit(scaled_data),
                    'method_type': 'nonlinear',
                    'perplexity': perplexity
                }
            except Exception as e:
                print(f"OpenTSNE failed: {e}")
        
        elif HAS_MULTICORE_TSNE:
            try:
                tsne = MulticoreTSNE(n_jobs=4, **tsne_params)
                results['tsne_multicore'] = {
                    'embedding': tsne.fit_transform(scaled_data),
                    'method_type': 'nonlinear',
                    'perplexity': perplexity
                }
            except Exception as e:
                print(f"MulticoreTSNE failed: {e}")
        
        # Fallback to standard t-SNE
        if not any('tsne' in k for k in results.keys()):
            try:
                tsne = TSNE(**tsne_params)
                results['tsne'] = {
                    'embedding': tsne.fit_transform(scaled_data),
                    'method_type': 'nonlinear',
                    'perplexity': perplexity
                }
            except Exception as e:
                print(f"Standard t-SNE failed: {e}")

        # UMAP with optimized parameters
        if HAS_UMAP:
            try:
                umap_params = {
                    'n_components': target_dim,
                    'random_state': config.random_state,
                    'n_neighbors': n_neighbors,
                    'min_dist': 0.1,
                    'metric': 'euclidean',
                    'spread': 1.0,
                    'low_memory': n_samples > 10000,
                    'n_epochs': None,  # Auto-determine
                    'learning_rate': 1.0,
                    'repulsion_strength': 1.0
                }
                
                umap_reducer = UMAP(**umap_params)
                results['umap'] = {
                    'embedding': umap_reducer.fit_transform(scaled_data),
                    'method_type': 'nonlinear',
                    'n_neighbors': n_neighbors
                }
            except Exception as e:
                print(f"UMAP failed: {e}")

        # TriMap (if available)
        if HAS_TRIMAP and n_samples >= 100:
            try:
                trimap_embedding = trimap.TRIMAP(
                    n_dims=target_dim,
                    n_inliers=max(3, n_neighbors // 2),
                    n_outliers=max(1, n_neighbors // 6),
                    n_random=max(1, n_neighbors // 6),
                    lr=1000.0,
                    n_iters=1200
                ).fit_transform(scaled_data)
                
                results['trimap'] = {
                    'embedding': trimap_embedding,
                    'method_type': 'nonlinear'
                }
            except Exception as e:
                print(f"TriMap failed: {e}")

        # Other manifold learning methods
        manifold_methods = {
            'isomap': Isomap(n_components=target_dim, n_neighbors=n_neighbors),
            'lle': LocallyLinearEmbedding(n_components=target_dim, n_neighbors=n_neighbors, 
                                        random_state=config.random_state, method='standard'),
            'spectral': SpectralEmbedding(n_components=target_dim, random_state=config.random_state,
                                        n_neighbors=n_neighbors)
        }

        for name, method in manifold_methods.items():
            try:
                embedding = method.fit_transform(scaled_data)
                results[name] = {
                    'embedding': embedding,
                    'method_type': 'nonlinear',
                    'n_neighbors': getattr(method, 'n_neighbors', None)
                }
            except Exception as e:
                print(f"Manifold method {name} failed: {e}")

        return results

    def _evaluate_embeddings(self, embeddings: Dict[str, Any], scaled_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate embedding quality using multiple metrics"""
        evaluation_results = {}
        
        for name, result in embeddings.items():
            if 'embedding' not in result:
                continue
                
            embedding = result['embedding']
            if embedding.shape[0] < 10:  # Need minimum samples for evaluation
                continue
                
            metrics = {}
            
            try:
                # Silhouette score using k-means clustering
                n_clusters = min(8, max(2, embedding.shape[0] // 10))
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embedding)
                    if len(np.unique(cluster_labels)) > 1:
                        metrics['silhouette_score'] = silhouette_score(embedding, cluster_labels)
            except:
                pass

            try:
                # Neighborhood preservation (for nonlinear methods)
                if result.get('method_type') == 'nonlinear' and scaled_data.shape[0] <= 1000:
                    from scipy.spatial.distance import pdist, squareform
                    from scipy.stats import spearmanr

                    # Sample for computational efficiency
                    n_sample = min(200, embedding.shape[0])
                    indices = np.random.choice(embedding.shape[0], n_sample, replace=False)
                    
                    orig_dist = squareform(pdist(scaled_data[indices]))
                    embed_dist = squareform(pdist(embedding[indices]))
                    
                    # Spearman correlation between distance matrices
                    correlation, _ = spearmanr(orig_dist.flatten(), embed_dist.flatten())
                    metrics['neighborhood_preservation'] = correlation
            except:
                pass

            try:
                # Local continuity
                embedding_std = np.std(embedding, axis=0)
                metrics['embedding_stability'] = 1.0 / (1.0 + np.mean(embedding_std))
            except:
                pass

            evaluation_results[name] = metrics

        return evaluation_results

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Main analysis method with comprehensive dimensionality reduction"""
        try:
            # Advanced preprocessing
            scaled_data, preprocessing_info = self._adaptive_preprocessing(data, config)
            
            # Enhanced dimensionality reduction
            embeddings = self._enhanced_dimensionality_reduction(scaled_data, config)
            
            if not embeddings:
                return {'error': 'All dimensionality reduction methods failed'}
            
            # Evaluate embedding quality
            evaluation_results = self._evaluate_embeddings(embeddings, scaled_data)
            
            # Compile final results
            results = {
                'embeddings': embeddings,
                'evaluation': evaluation_results,
                'preprocessing_info': preprocessing_info,
                'data_characteristics': {
                    'n_samples': scaled_data.shape[0],
                    'n_features': scaled_data.shape[1],
                    'condition_number': np.linalg.cond(scaled_data.T @ scaled_data),
                    'effective_rank': np.linalg.matrix_rank(scaled_data)
                },
                'recommendations': self._generate_recommendations(embeddings, evaluation_results, preprocessing_info)
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Dimensionality analysis failed: {str(e)}'}

    def _generate_recommendations(self, embeddings: Dict[str, Any], 
                                evaluations: Dict[str, Any], 
                                preprocessing_info: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on results"""
        recommendations = []
        
        # Find best performing methods
        method_scores = {}
        for name, metrics in evaluations.items():
            score = 0
            if 'silhouette_score' in metrics:
                score += metrics['silhouette_score'] * 0.4
            if 'neighborhood_preservation' in metrics:
                score += metrics['neighborhood_preservation'] * 0.4
            if 'embedding_stability' in metrics:
                score += metrics['embedding_stability'] * 0.2
            method_scores[name] = score

        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            recommendations.append(f"Best performing method: {best_method}")
            
            # Method-specific recommendations
            if 'umap' in best_method:
                recommendations.append("UMAP preserves both local and global structure well")
            elif 'tsne' in best_method:
                recommendations.append("t-SNE excels at revealing local cluster structure")
            elif best_method == 'pca':
                recommendations.append("PCA indicates linear relationships dominate the data")

        # Data-specific recommendations
        if preprocessing_info.get('features_removed', 0) > 0:
            recommendations.append(f"Removed {preprocessing_info['features_removed']} low-variance features")
            
        if preprocessing_info.get('scaling_method') == 'power_transform':
            recommendations.append("Applied power transformation due to skewed data distribution")
            
        if len([k for k in embeddings if embeddings[k].get('method_type') == 'linear']) > len([k for k in embeddings if embeddings[k].get('method_type') == 'nonlinear']):
            recommendations.append("Consider nonlinear methods for more complex pattern discovery")

        return recommendations


class TimeSeriesAnalyzer(AnalysisStrategy):
    """SOTA comprehensive time series analysis with advanced ML techniques"""

    @property
    def name(self) -> str:
        return "timeseries"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove time and target columns from analysis
        time_col = getattr(config, "time_col", None)
        target_col = getattr(config, "target", None)
        if time_col in numeric_cols: numeric_cols.remove(time_col)
        if target_col in numeric_cols: numeric_cols.remove(target_col)
        
        results = {
            "stationarity": self._analyze_stationarity(data, numeric_cols, config),
            "temporal_patterns": self._analyze_temporal_patterns(data, numeric_cols, config),
            "lag_suggestions": self._suggest_lag_features(data, numeric_cols, config),
            "seasonality_tests": self._advanced_seasonality_tests(data, numeric_cols, config),
            "change_point_detection": self._detect_change_points(data, numeric_cols, config),
            "regime_switching": self._detect_regime_switching(data, numeric_cols, config),
            "forecasting_readiness": self._assess_forecasting_readiness(data, numeric_cols, config),
            "volatility_analysis": self._analyze_volatility(data, numeric_cols, config),
            "cyclical_patterns": self._detect_cyclical_patterns(data, numeric_cols, config),
            "causality_analysis": self._granger_causality_analysis(data, numeric_cols, config)
        }

        return results

    def _analyze_stationarity(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> pd.DataFrame:
        """Enhanced stationarity testing with multiple tests and differencing suggestions"""
        from statsmodels.stats.diagnostic import breaks_hansen
        from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
        
        results = []

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 20:
                continue

            try:
                # ADF test (null: unit root present, non-stationary)
                adf_result = adfuller(col_data, autolag="AIC")
                
                # KPSS test (null: stationary)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    kpss_result = kpss(col_data, regression="c", nlags="auto")
                
                # Zivot-Andrews test (structural break unit root test)
                za_stat = za_pvalue = None
                try:
                    if len(col_data) > 50:  # Minimum required for ZA test
                        za_result = zivot_andrews(col_data, model="c")
                        za_stat, za_pvalue = za_result[0], za_result[1]
                except: pass
                
                # Variance ratio test for random walk
                variance_ratio = self._variance_ratio_test(col_data)
                
                # Determine stationarity consensus
                adf_stationary = adf_result[1] < config.confidence_level
                kpss_stationary = kpss_result[1] > config.confidence_level
                
                # Enhanced stationarity classification
                if adf_stationary and kpss_stationary:
                    stationarity_type = "stationary"
                elif not adf_stationary and not kpss_stationary:
                    stationarity_type = "non_stationary"
                elif adf_stationary and not kpss_stationary:
                    stationarity_type = "trend_stationary"
                else:
                    stationarity_type = "difference_stationary"
                
                # Test first difference if non-stationary
                diff_results = {}
                if not (adf_stationary and kpss_stationary):
                    diff_series = col_data.diff().dropna()
                    if len(diff_series) > 10:
                        try:
                            diff_adf = adfuller(diff_series, autolag="AIC")
                            diff_results = {
                                "diff_adf_stat": diff_adf[0],
                                "diff_adf_pvalue": diff_adf[1],
                                "diff_stationary": diff_adf[1] < config.confidence_level
                            }
                        except: pass
                
                # Seasonal differencing test
                seasonal_diff_results = {}
                if len(col_data) > 24:
                    try:
                        seasonal_diff = col_data.diff(12).dropna()  # 12-period seasonal diff
                        if len(seasonal_diff) > 10:
                            seas_adf = adfuller(seasonal_diff, autolag="AIC")
                            seasonal_diff_results = {
                                "seasonal_diff_adf_stat": seas_adf[0],
                                "seasonal_diff_adf_pvalue": seas_adf[1],
                                "seasonal_diff_stationary": seas_adf[1] < config.confidence_level
                            }
                    except: pass

                result_dict = {
                    "feature": col,
                    "adf_statistic": adf_result[0],
                    "adf_pvalue": adf_result[1],
                    "adf_critical_1pct": adf_result[4]['1%'],
                    "adf_critical_5pct": adf_result[4]['5%'],
                    "kpss_statistic": kpss_result[0],
                    "kpss_pvalue": kpss_result[1],
                    "kpss_critical_1pct": kpss_result[3]['1%'],
                    "kpss_critical_5pct": kpss_result[3]['5%'],
                    "za_statistic": za_stat,
                    "za_pvalue": za_pvalue,
                    "variance_ratio": variance_ratio,
                    "is_stationary_adf": adf_stationary,
                    "is_stationary_kpss": kpss_stationary,
                    "stationarity_type": stationarity_type,
                    "consensus_stationary": adf_stationary and kpss_stationary,
                    "is_stationary": adf_stationary and kpss_stationary,
                    **diff_results,
                    **seasonal_diff_results
                }
                
                results.append(result_dict)
                
            except Exception as e:
                print(f"Enhanced stationarity test failed for {col}: {e}")

        return pd.DataFrame(results)

    def _variance_ratio_test(self, series: pd.Series, lags: int = 4) -> float:
        """Variance ratio test for random walk hypothesis"""
        try:
            n = len(series)
            if n < lags * 4:
                return np.nan
            
            # Calculate variance ratio
            returns = series.pct_change().dropna()
            var_1 = np.var(returns)
            
            # k-period variance
            k_returns = returns.rolling(window=lags).sum().dropna()
            var_k = np.var(k_returns) / lags
            
            variance_ratio = var_k / var_1 if var_1 > 0 else np.nan
            return variance_ratio
            
        except:
            return np.nan

    def _analyze_temporal_patterns(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Enhanced temporal pattern analysis with multiple decomposition methods"""
        from scipy.signal import find_peaks, periodogram
        from scipy.stats import linregress
        from statsmodels.tsa.seasonal import STL, seasonal_decompose
        from statsmodels.tsa.x13 import x13_arima_analysis
        
        patterns = {}

        for col in numeric_cols[:8]:  # Limit for performance
            series = data[col].dropna()
            if len(series) < 24:
                continue

            try:
                # Enhanced trend analysis with multiple methods
                x = np.arange(len(series))
                
                # Linear trend
                slope, intercept, r_value, p_value, stderr = linregress(x, series)
                
                # Polynomial trend (degree 2)
                poly_coeffs = np.polyfit(x, series, deg=2)
                poly_trend = np.polyval(poly_coeffs, x)
                poly_r2 = np.corrcoef(series, poly_trend)[0, 1] ** 2
                
                # Hodrick-Prescott filter trend
                try:
                    from statsmodels.tsa.filters.hp_filter import hpfilter
                    hp_cycle, hp_trend = hpfilter(series, lamb=1600)
                    hp_trend_strength = np.var(hp_trend) / np.var(series)
                except:
                    hp_trend_strength = None
                
                trend_classification = self._classify_trend(slope, p_value, series.std())
                
                patterns[f"{col}_trend"] = {
                    "linear_slope": slope,
                    "linear_r_squared": r_value**2,
                    "linear_p_value": p_value,
                    "linear_significant": p_value < 0.05,
                    "trend_direction": trend_classification,
                    "slope_normalized": slope / series.std() if series.std() > 0 else 0,
                    "polynomial_r2": poly_r2,
                    "hp_trend_strength": hp_trend_strength,
                    "trend_strength_category": self._categorize_trend_strength(abs(slope), series.std())
                }

                # Advanced seasonality detection with multiple methods
                seasonality_results = {}
                
                # STL decomposition
                try:
                    # Auto-detect period or use default
                    period = self._estimate_period(series)
                    stl = STL(series, seasonal=min(period, len(series)//3), period=period)
                    stl_decomp = stl.fit()
                    
                    seasonal_var = np.var(stl_decomp.seasonal)
                    total_var = np.var(series)
                    seasonal_strength = seasonal_var / total_var if total_var > 0 else 0
                    
                    seasonality_results.update({
                        "stl_seasonal_strength": seasonal_strength,
                        "stl_period": period,
                        "stl_classification": self._classify_seasonality(seasonal_strength),
                        "trend_strength": np.var(stl_decomp.trend.dropna()) / total_var if total_var > 0 else 0
                    })
                    
                except Exception as e:
                    seasonality_results["stl_error"] = str(e)
                
                # X-13ARIMA-SEATS decomposition (if available)
                try:
                    if len(series) >= 36:  # Minimum for X-13
                        x13_result = x13_arima_analysis(series)
                        seasonality_results["x13_seasonal_strength"] = "computed"
                except:
                    pass
                
                # Periodogram analysis for dominant frequencies
                try:
                    freqs, psd = periodogram(series, scaling='density')
                    dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
                    dominant_period = 1 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] > 0 else None
                    seasonality_results.update({
                        "dominant_period_periodogram": dominant_period,
                        "spectral_peak_power": psd[dominant_freq_idx]
                    })
                except: pass

                patterns[f"{col}_seasonality"] = seasonality_results

                # Enhanced volatility clustering analysis
                volatility_results = {}
                returns = series.pct_change().dropna()
                if len(returns) > 20:
                    # Multiple volatility measures
                    rolling_vol = returns.rolling(window=min(10, len(returns)//3)).std()
                    
                    # ARCH effects test
                    squared_returns = returns ** 2
                    arch_lm_stat = self._arch_lm_test(returns)
                    
                    # Volatility persistence
                    vol_autocorr = rolling_vol.dropna().autocorr(lag=1) if len(rolling_vol.dropna()) > 1 else 0
                    
                    # GARCH-like volatility clustering
                    high_vol_periods = (rolling_vol > rolling_vol.quantile(0.8)).sum() if len(rolling_vol.dropna()) > 0 else 0
                    vol_clustering_score = high_vol_periods / len(rolling_vol.dropna()) if len(rolling_vol.dropna()) > 0 else 0
                    
                    volatility_results = {
                        "returns_volatility": returns.std(),
                        "rolling_vol_mean": rolling_vol.mean(),
                        "vol_autocorr": vol_autocorr,
                        "arch_lm_statistic": arch_lm_stat,
                        "vol_clustering_score": vol_clustering_score,
                        "volatility_persistent": abs(vol_autocorr) > 0.3 if not np.isnan(vol_autocorr) else False
                    }
                
                patterns[f"{col}_volatility"] = volatility_results

                # Structural break detection
                break_results = {}
                if len(series) > 50:
                    try:
                        # Simple CUSUM test for structural breaks
                        cumsum = np.cumsum(series - series.mean())
                        max_cusum = np.max(np.abs(cumsum))
                        cusum_stat = max_cusum / (series.std() * np.sqrt(len(series)))
                        
                        # Find potential break points
                        break_point_idx = np.argmax(np.abs(cumsum))
                        break_point_pct = break_point_idx / len(series)
                        
                        break_results = {
                            "cusum_statistic": cusum_stat,
                            "potential_break_point": break_point_pct,
                            "break_significant": cusum_stat > 1.5,  # Rough threshold
                            "pre_break_mean": series.iloc[:break_point_idx].mean() if break_point_idx > 10 else None,
                            "post_break_mean": series.iloc[break_point_idx:].mean() if len(series) - break_point_idx > 10 else None
                        }
                    except: pass
                
                patterns[f"{col}_structural_breaks"] = break_results

            except Exception as e:
                print(f"Enhanced temporal pattern analysis failed for {col}: {e}")

        return patterns

    def _classify_trend(self, slope: float, p_value: float, std: float) -> str:
        """Classify trend direction and strength"""
        if p_value > 0.05:
            return "no_trend"
        
        threshold = std * 0.01  # 1% of standard deviation
        if slope > threshold:
            return "increasing" if slope > 2 * threshold else "weakly_increasing"
        elif slope < -threshold:
            return "decreasing" if slope < -2 * threshold else "weakly_decreasing"
        else:
            return "stable"

    def _classify_seasonality(self, strength: float) -> str:
        """Classify seasonality strength"""
        if strength > 0.4:
            return "strong"
        elif strength > 0.2:
            return "moderate"
        elif strength > 0.05:
            return "weak"
        else:
            return "none"

    def _categorize_trend_strength(self, abs_slope: float, std: float) -> str:
        """Categorize trend strength"""
        normalized_slope = abs_slope / (std + 1e-8)
        if normalized_slope > 0.1:
            return "strong"
        elif normalized_slope > 0.05:
            return "moderate"
        elif normalized_slope > 0.01:
            return "weak"
        else:
            return "none"

    def _estimate_period(self, series: pd.Series) -> int:
        """Estimate dominant period using autocorrelation"""
        from statsmodels.tsa.stattools import acf
        
        try:
            max_lags = min(len(series) // 3, 100)
            acf_vals = acf(series, nlags=max_lags, fft=True)
            
            # Find peaks in ACF
            peaks, _ = find_peaks(acf_vals[1:], height=0.1)
            if len(peaks) > 0:
                dominant_lag = peaks[np.argmax(acf_vals[peaks + 1])] + 1
                return max(2, min(dominant_lag, 52))  # Reasonable bounds
            else:
                # Default periods based on series length
                if len(series) >= 365:
                    return 365  # Daily data -> yearly seasonality
                elif len(series) >= 52:
                    return 52   # Weekly data -> yearly seasonality
                elif len(series) >= 12:
                    return 12   # Monthly data -> yearly seasonality
                else:
                    return max(2, len(series) // 4)
        except:
            return max(2, min(12, len(series) // 4))

    def _arch_lm_test(self, returns: pd.Series, lags: int = 5) -> float:
        """ARCH LM test for volatility clustering"""
        try:
            from statsmodels.stats.diagnostic import het_arch
            if len(returns) > lags + 10:
                lm_stat, p_val, _, _ = het_arch(returns, nlags=lags)
                return p_val
        except:
            pass
        return np.nan

    def _suggest_lag_features(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Dict[str, Any]]:
        """Enhanced lag feature suggestions using multiple methods"""
        from statsmodels.tsa.ar_model import AutoReg
        from statsmodels.tsa.stattools import acf, ccf, pacf
        
        suggestions = {}
        max_lags = min(24, len(data) // 10)  # Adaptive max lags
        
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) < max_lags + 20:
                continue

            try:
                # Enhanced autocorrelation analysis
                pacf_vals = pacf(series, nlags=max_lags, method="ols")[1:]
                acf_vals = acf(series, nlags=max_lags, fft=True)[1:]
                n = len(series)

                # Dynamic confidence intervals
                pacf_ci = 1.96 / np.sqrt(n)
                acf_ci = 1.96 / np.sqrt(n)

                # Score lags using multiple criteria
                lag_scores = []
                for lag in range(1, max_lags + 1):
                    p_val = pacf_vals[lag - 1]
                    a_val = acf_vals[lag - 1]
                    
                    # Multi-criteria scoring
                    significance_score = 0
                    if abs(p_val) > pacf_ci:
                        significance_score += 0.6 * abs(p_val)
                    if abs(a_val) > acf_ci:
                        significance_score += 0.4 * abs(a_val)
                    
                    # Seasonal lag bonus (12, 24, etc.)
                    if lag in [12, 24, 52] and lag < len(series) // 3:
                        significance_score *= 1.2
                    
                    # Penalize very long lags
                    if lag > len(series) // 5:
                        significance_score *= 0.8
                    
                    if significance_score > 0.1:  # Threshold
                        lag_scores.append((lag, significance_score))

                # Auto-regression model order selection
                ar_order = None
                try:
                    # Use AIC to select optimal AR order
                    aic_scores = []
                    max_ar_order = min(10, len(series) // 10)
                    for order in range(1, max_ar_order + 1):
                        try:
                            ar_model = AutoReg(series, lags=order).fit()
                            aic_scores.append((order, ar_model.aic))
                        except:
                            continue
                    
                    if aic_scores:
                        ar_order = min(aic_scores, key=lambda x: x[1])[0]
                        
                except: pass

                # Information criteria-based lag selection
                information_criteria = {}
                try:
                    from statsmodels.tsa.vector_ar.var_model import VAR

                    # Single variable VAR for lag selection
                    model_data = series.values.reshape(-1, 1)
                    var_model = VAR(model_data)
                    lag_order_results = var_model.select_order(maxlags=min(12, len(series)//10))
                    information_criteria = {
                        "aic_optimal": lag_order_results.aic,
                        "bic_optimal": lag_order_results.bic,
                        "hqic_optimal": lag_order_results.hqic
                    }
                except: pass

                # Cross-correlation with other features (if multivariate)
                ccf_lags = {}
                if len(numeric_cols) > 1:
                    for other_col in numeric_cols[:5]:  # Limit to avoid explosion
                        if other_col != col and other_col in data.columns:
                            try:
                                other_series = data[other_col].dropna()
                                # Align series
                                min_len = min(len(series), len(other_series))
                                if min_len > max_lags + 10:
                                    s1 = series.iloc[-min_len:]
                                    s2 = other_series.iloc[-min_len:]
                                    cross_corr = ccf(s1, s2, adjusted=False)
                                    
                                    # Find significant cross-correlations
                                    significant_ccf_lags = []
                                    cc_ci = 1.96 / np.sqrt(min_len)
                                    for lag_idx, cc_val in enumerate(cross_corr):
                                        if abs(cc_val) > cc_ci:
                                            actual_lag = lag_idx - len(cross_corr) // 2
                                            if actual_lag > 0:  # Only positive lags
                                                significant_ccf_lags.append((actual_lag, cc_val))
                                    
                                    if significant_ccf_lags:
                                        ccf_lags[other_col] = significant_ccf_lags[:3]  # Top 3
                            except: pass

                # Sort and select top lags
                lag_scores.sort(key=lambda x: x[1], reverse=True)
                top_lags = [lag for lag, _ in lag_scores[:5]]  # Top 5 lags

                if top_lags or ar_order or information_criteria or ccf_lags:
                    suggestions[col] = {
                        "autocorr_lags": top_lags,
                        "ar_optimal_order": ar_order,
                        "information_criteria": information_criteria,
                        "cross_correlation_lags": ccf_lags,
                        "recommended_lags": top_lags[:3] if top_lags else ([ar_order] if ar_order else []),
                        "lag_selection_method": "multi_criteria",
                        "seasonal_lags": [lag for lag in top_lags if lag in [12, 24, 52]]
                    }

            except Exception as e:
                print(f"Enhanced lag suggestion failed for {col}: {e}")

        return suggestions

    def _advanced_seasonality_tests(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Advanced seasonality tests using multiple statistical methods"""
        from scipy.stats import friedmanchisquare, kruskal
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        seasonality_results = {}
        time_col = getattr(config, "time_col", None)
        
        if not time_col or time_col not in data.columns:
            return seasonality_results
        
        for col in numeric_cols[:5]:  # Limit for performance
            series_data = data[[time_col, col]].dropna()
            if len(series_data) < 24:
                continue
            
            try:
                # Ensure datetime index
                series_data[time_col] = pd.to_datetime(series_data[time_col])
                series_data.set_index(time_col, inplace=True)
                series = series_data[col]
                
                tests_results = {}
                
                # Friedman test for seasonal patterns
                if len(series) >= 36:  # At least 3 years of monthly data
                    try:
                        # Group by month
                        monthly_groups = []
                        for month in range(1, 13):
                            month_data = series[series.index.month == month]
                            if len(month_data) >= 3:  # At least 3 observations per month
                                monthly_groups.append(month_data.values)
                        
                        if len(monthly_groups) >= 6:  # At least 6 months with data
                            # Make groups equal length by truncating to minimum
                            min_length = min(len(group) for group in monthly_groups)
                            equal_groups = [group[:min_length] for group in monthly_groups]
                            
                            if min_length >= 3:
                                stat, p_val = friedmanchisquare(*equal_groups)
                                tests_results["friedman_test"] = {
                                    "statistic": stat,
                                    "p_value": p_val,
                                    "seasonal_significant": p_val < 0.05
                                }
                    except Exception as e:
                        tests_results["friedman_test"] = {"error": str(e)}
                
                # Kruskal-Wallis test for day-of-week effects
                try:
                    dow_groups = [series[series.index.dayofweek == dow] for dow in range(7)]
                    dow_groups = [group for group in dow_groups if len(group) >= 5]
                    
                    if len(dow_groups) >= 5:  # At least 5 days with sufficient data
                        stat, p_val = kruskal(*dow_groups)
                        tests_results["kruskal_dow"] = {
                            "statistic": stat,
                            "p_value": p_val,
                            "dow_effect_significant": p_val < 0.05
                        }
                except Exception as e:
                    tests_results["kruskal_dow"] = {"error": str(e)}
                
                # Ljung-Box test for serial correlation at seasonal lags
                seasonal_lags = [12, 24, 52] if len(series) > 104 else [12] if len(series) > 24 else []
                for lag in seasonal_lags:
                    if len(series) > lag + 10:
                        try:
                            lb_result = acorr_ljungbox(series, lags=[lag], return_df=True)
                            tests_results[f"ljungbox_lag_{lag}"] = {
                                "statistic": lb_result["lb_stat"].iloc[0],
                                "p_value": lb_result["lb_pvalue"].iloc[0],
                                "seasonal_correlation": lb_result["lb_pvalue"].iloc[0] < 0.05
                            }
                        except: pass
                
                # Spectral analysis for periodic components
                try:
                    from scipy.signal import welch
                    frequencies, psd = welch(series, nperseg=min(len(series)//4, 256))
                    
                    # Find dominant frequencies
                    peak_indices = find_peaks(psd, height=np.percentile(psd, 90))[0]
                    dominant_periods = []
                    
                    for peak_idx in peak_indices[:5]:  # Top 5 peaks
                        freq = frequencies[peak_idx]
                        if freq > 0:
                            period = 1 / freq
                            power = psd[peak_idx]
                            dominant_periods.append({
                                "period": period,
                                "frequency": freq,
                                "power": power
                            })
                    
                    tests_results["spectral_analysis"] = {
                        "dominant_periods": sorted(dominant_periods, key=lambda x: x["power"], reverse=True)
                    }
                except: pass
                
                # QS (Quarterly Seasonal) test simulation
                try:
                    if len(series) >= 48:  # At least 4 years of quarterly data
                        quarterly_means = []
                        for quarter in range(1, 5):
                            q_data = series[series.index.quarter == quarter]
                            if len(q_data) >= 4:
                                quarterly_means.append(q_data.mean())
                        
                        if len(quarterly_means) == 4:
                            # Simple seasonal variance test
                            overall_mean = series.mean()
                            seasonal_variance = np.var(quarterly_means)
                            total_variance = series.var()
                            seasonal_ratio = seasonal_variance / (total_variance + 1e-8)
                            
                            tests_results["quarterly_seasonality"] = {
                                "seasonal_variance_ratio": seasonal_ratio,
                                "significant": seasonal_ratio > 0.1  # Heuristic threshold
                            }
                except: pass
                
                seasonality_results[col] = tests_results
                
            except Exception as e:
                print(f"Advanced seasonality test failed for {col}: {e}")
        
        return seasonality_results

    def _detect_change_points(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Detect structural change points using multiple algorithms"""
        time_col = getattr(config, "time_col", None)
        if not time_col or time_col not in data.columns:
            return {}
        
        change_points = {}
        
        for col in numeric_cols[:5]:
            series_data = data[[time_col, col]].dropna()
            if len(series_data) < 50:
                continue
                
            try:
                # Prepare time series
                series_data[time_col] = pd.to_datetime(series_data[time_col])
                series_data.set_index(time_col, inplace=True)
                series = series_data[col]
                
                detected_changes = {}
                
                # CUSUM-based change point detection
                try:
                    mean_val = series.mean()
                    cumsum = np.cumsum(series - mean_val)
                    
                    # Find maximum deviation points
                    abs_cumsum = np.abs(cumsum)
                    change_idx = np.argmax(abs_cumsum)
                    max_cusum = abs_cumsum[change_idx]
                    
                    # Standardize CUSUM statistic
                    cusum_stat = max_cusum / (series.std() * np.sqrt(len(series)))
                    
                    if cusum_stat > 1.5:  # Significant change threshold
                        change_timestamp = series.index[change_idx]
                        pre_mean = series.iloc[:change_idx].mean()
                        post_mean = series.iloc[change_idx:].mean()
                        
                        detected_changes["cusum"] = {
                            "change_point": change_timestamp,
                            "statistic": cusum_stat,
                            "pre_change_mean": pre_mean,
                            "post_change_mean": post_mean,
                            "magnitude": abs(post_mean - pre_mean),
                            "significant": True
                        }
                except: pass
                
                # Bayesian change point detection (simplified)
                try:
                    # Rolling window variance change detection
                    window_size = max(10, len(series) // 20)
                    rolling_mean = series.rolling(window=window_size).mean()
                    rolling_var = series.rolling(window=window_size).var()
                    
                    # Detect mean shifts
                    mean_changes = []
                    for i in range(window_size, len(rolling_mean) - window_size):
                        before_mean = rolling_mean.iloc[i-window_size:i].mean()
                        after_mean = rolling_mean.iloc[i:i+window_size].mean()
                        
                        if not (pd.isna(before_mean) or pd.isna(after_mean)):
                            change_magnitude = abs(after_mean - before_mean)
                            if change_magnitude > series.std():
                                mean_changes.append({
                                    "timestamp": series.index[i],
                                    "magnitude": change_magnitude,
                                    "before": before_mean,
                                    "after": after_mean
                                })
                    
                    # Detect variance changes
                    variance_changes = []
                    for i in range(window_size, len(rolling_var) - window_size):
                        before_var = rolling_var.iloc[i-window_size:i].mean()
                        after_var = rolling_var.iloc[i:i+window_size].mean()
                        
                        if not (pd.isna(before_var) or pd.isna(after_var)) and before_var > 0:
                            var_ratio = max(after_var, before_var) / min(after_var, before_var)
                            if var_ratio > 2.0:  # 100% variance change
                                variance_changes.append({
                                    "timestamp": series.index[i],
                                    "ratio": var_ratio,
                                    "before_var": before_var,
                                    "after_var": after_var
                                })
                    
                    if mean_changes or variance_changes:
                        detected_changes["rolling_window"] = {
                            "mean_changes": sorted(mean_changes, key=lambda x: x["magnitude"], reverse=True)[:3],
                            "variance_changes": sorted(variance_changes, key=lambda x: x["ratio"], reverse=True)[:3]
                        }
                        
                except: pass
                
                # Page-Hinkley test for online change detection
                try:
                    # Simplified Page-Hinkley implementation
                    delta = series.std() * 0.5  # Detection threshold
                    lambda_param = series.std() * 0.1  # Drift parameter
                    
                    cumulative_sum = 0
                    min_sum = 0
                    change_points_ph = []
                    
                    for i, value in enumerate(series):
                        cumulative_sum += (value - series.mean() - lambda_param)
                        min_sum = min(min_sum, cumulative_sum)
                        
                        if cumulative_sum - min_sum > delta:
                            change_points_ph.append({
                                "index": i,
                                "timestamp": series.index[i],
                                "cumulative_sum": cumulative_sum - min_sum
                            })
                            cumulative_sum = 0
                            min_sum = 0
                    
                    if change_points_ph:
                        detected_changes["page_hinkley"] = {
                            "change_points": change_points_ph[:5],  # Top 5
                            "threshold": delta
                        }
                        
                except: pass
                
                # Multiple change points using binary segmentation approach
                try:
                    def detect_single_change(subseries):
                        """Detect single change point in subseries"""
                        n = len(subseries)
                        if n < 20:  # Minimum segment size
                            return None
                        
                        best_change = None
                        best_stat = 0
                        
                        for k in range(10, n-10):  # Leave margins
                            left_mean = subseries.iloc[:k].mean()
                            right_mean = subseries.iloc[k:].mean()
                            
                            # Calculate test statistic
                            left_var = subseries.iloc[:k].var()
                            right_var = subseries.iloc[k:].var()
                            
                            if left_var > 0 and right_var > 0:
                                pooled_var = ((k-1)*left_var + (n-k-1)*right_var) / (n-2)
                                t_stat = abs(left_mean - right_mean) / np.sqrt(pooled_var * (1/k + 1/(n-k)))
                                
                                if t_stat > best_stat:
                                    best_stat = t_stat
                                    best_change = k
                        
                        return (best_change, best_stat) if best_stat > 2.0 else None  # t-stat threshold
                    
                    # Apply binary segmentation
                    segments_to_process = [(0, len(series), series)]
                    multiple_changes = []
                    
                    while segments_to_process and len(multiple_changes) < 5:  # Max 5 change points
                        start, end, segment = segments_to_process.pop(0)
                        
                        change_result = detect_single_change(segment)
                        if change_result:
                            rel_change_idx, stat = change_result
                            abs_change_idx = start + rel_change_idx
                            
                            multiple_changes.append({
                                "index": abs_change_idx,
                                "timestamp": series.index[abs_change_idx],
                                "statistic": stat,
                                "segment_start": start,
                                "segment_end": end
                            })
                            
                            # Add new segments to process
                            left_segment = series.iloc[start:start + rel_change_idx]
                            right_segment = series.iloc[start + rel_change_idx:end]
                            
                            if len(left_segment) >= 20:
                                segments_to_process.append((start, start + rel_change_idx, left_segment))
                            if len(right_segment) >= 20:
                                segments_to_process.append((start + rel_change_idx, end, right_segment))
                    
                    if multiple_changes:
                        detected_changes["binary_segmentation"] = {
                            "change_points": sorted(multiple_changes, key=lambda x: x["statistic"], reverse=True)
                        }
                
                except: pass
                
                if detected_changes:
                    change_points[col] = detected_changes
                    
            except Exception as e:
                print(f"Change point detection failed for {col}: {e}")
        
        return change_points

    def _detect_regime_switching(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Detect regime switching patterns using Markov switching models and other methods"""
        time_col = getattr(config, "time_col", None)
        if not time_col or time_col not in data.columns:
            return {}
        
        regime_results = {}
        
        for col in numeric_cols[:3]:  # Limit for computational efficiency
            series_data = data[[time_col, col]].dropna()
            if len(series_data) < 100:  # Need sufficient data for regime detection
                continue
                
            try:
                series_data[time_col] = pd.to_datetime(series_data[time_col])
                series_data.set_index(time_col, inplace=True)
                series = series_data[col]
                
                regime_analysis = {}
                
                # Hidden Markov Model approach (simplified)
                try:
                    from sklearn.mixture import GaussianMixture

                    # Prepare data for regime detection
                    returns = series.pct_change().dropna()
                    features = np.column_stack([
                        returns.values,
                        returns.rolling(5).mean().dropna().values,
                        returns.rolling(5).std().dropna().values
                    ])
                    
                    # Remove any rows with NaN
                    features = features[~np.isnan(features).any(axis=1)]
                    
                    if len(features) > 50:
                        # Fit GMM with different numbers of regimes
                        best_n_regimes = 2
                        best_score = -np.inf
                        
                        for n_regimes in range(2, 5):  # Test 2-4 regimes
                            try:
                                gmm = GaussianMixture(n_components=n_regimes, random_state=config.random_state)
                                gmm.fit(features)
                                score = gmm.score(features)
                                
                                if score > best_score:
                                    best_score = score
                                    best_n_regimes = n_regimes
                            except:
                                continue
                        
                        # Fit best model
                        final_gmm = GaussianMixture(n_components=best_n_regimes, random_state=config.random_state)
                        final_gmm.fit(features)
                        regime_labels = final_gmm.predict(features)
                        regime_probs = final_gmm.predict_proba(features)
                        
                        # Analyze regime characteristics
                        regime_stats = {}
                        for regime in range(best_n_regimes):
                            regime_mask = regime_labels == regime
                            regime_returns = returns.iloc[-len(regime_mask):][regime_mask]
                            
                            regime_stats[f"regime_{regime}"] = {
                                "mean_return": regime_returns.mean(),
                                "volatility": regime_returns.std(),
                                "duration_pct": regime_mask.sum() / len(regime_mask) * 100,
                                "persistence": self._calculate_regime_persistence(regime_labels, regime)
                            }
                        
                        # Regime transitions
                        transitions = []
                        for i in range(1, len(regime_labels)):
                            if regime_labels[i] != regime_labels[i-1]:
                                transitions.append({
                                    "from_regime": int(regime_labels[i-1]),
                                    "to_regime": int(regime_labels[i]),
                                    "timestamp": returns.index[i] if i < len(returns.index) else None
                                })
                        
                        regime_analysis["markov_switching"] = {
                            "n_regimes": best_n_regimes,
                            "regime_statistics": regime_stats,
                            "transitions": transitions[-10:],  # Last 10 transitions
                            "current_regime": int(regime_labels[-1]),
                            "current_regime_probability": float(np.max(regime_probs[-1]))
                        }
                        
                except: pass
                
                # Threshold autoregressive model detection
                try:
                    # Simple TAR model: different behavior above/below threshold
                    median_val = series.median()
                    above_threshold = series > median_val
                    
                    # Calculate statistics for each regime
                    high_regime_stats = {
                        "mean": series[above_threshold].mean(),
                        "std": series[above_threshold].std(),
                        "autocorr": series[above_threshold].autocorr() if len(series[above_threshold]) > 1 else 0,
                        "duration_pct": above_threshold.sum() / len(series) * 100
                    }
                    
                    low_regime_stats = {
                        "mean": series[~above_threshold].mean(),
                        "std": series[~above_threshold].std(),
                        "autocorr": series[~above_threshold].autocorr() if len(series[~above_threshold]) > 1 else 0,
                        "duration_pct": (~above_threshold).sum() / len(series) * 100
                    }
                    
                    # Test for regime differences
                    from scipy.stats import ttest_ind
                    t_stat, p_val = ttest_ind(series[above_threshold], series[~above_threshold])
                    
                    regime_analysis["threshold_autoregressive"] = {
                        "threshold": median_val,
                        "high_regime": high_regime_stats,
                        "low_regime": low_regime_stats,
                        "regime_difference_significant": p_val < 0.05,
                        "t_statistic": t_stat,
                        "p_value": p_val
                    }
                    
                except: pass
                
                # Volatility regime switching
                try:
                    returns = series.pct_change().dropna()
                    if len(returns) > 50:
                        # Rolling volatility
                        vol_window = min(20, len(returns) // 5)
                        rolling_vol = returns.rolling(vol_window).std()
                        
                        # High/low volatility regimes
                        vol_threshold = rolling_vol.quantile(0.7)  # 70th percentile
                        high_vol_periods = rolling_vol > vol_threshold
                        
                        # Volatility regime persistence
                        vol_regime_changes = (high_vol_periods != high_vol_periods.shift(1)).sum()
                        avg_regime_length = len(high_vol_periods) / (vol_regime_changes + 1)
                        
                        regime_analysis["volatility_switching"] = {
                            "vol_threshold": vol_threshold,
                            "high_vol_pct": high_vol_periods.sum() / len(high_vol_periods) * 100,
                            "avg_regime_length": avg_regime_length,
                            "regime_switches": vol_regime_changes,
                            "current_regime": "high" if high_vol_periods.iloc[-1] else "low"
                        }
                        
                except: pass
                
                if regime_analysis:
                    regime_results[col] = regime_analysis
                    
            except Exception as e:
                print(f"Regime switching detection failed for {col}: {e}")
        
        return regime_results

    def _calculate_regime_persistence(self, regime_labels: np.ndarray, regime: int) -> float:
        """Calculate average persistence (duration) of a regime"""
        regime_runs = []
        current_run = 0
        
        for label in regime_labels:
            if label == regime:
                current_run += 1
            else:
                if current_run > 0:
                    regime_runs.append(current_run)
                    current_run = 0
        
        if current_run > 0:
            regime_runs.append(current_run)
        
        return np.mean(regime_runs) if regime_runs else 0

    def _assess_forecasting_readiness(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Assess how ready each time series is for forecasting"""
        readiness_scores = {}
        
        for col in numeric_cols[:5]:
            series = data[col].dropna()
            if len(series) < 20:
                continue
                
            try:
                score_components = {}
                
                # Data sufficiency (more data = better)
                data_score = min(len(series) / 100, 1.0)  # Normalize to [0,1], optimal at 100+ points
                score_components["data_sufficiency"] = data_score
                
                # Missing data penalty
                missing_pct = data[col].isnull().mean()
                missing_score = 1.0 - missing_pct
                score_components["completeness"] = missing_score
                
                # Stationarity assessment
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(series)
                    stationarity_score = 1.0 if adf_result[1] < 0.05 else 0.5
                except:
                    stationarity_score = 0.5
                score_components["stationarity"] = stationarity_score
                
                # Trend strength (moderate trend is good for forecasting)
                from scipy.stats import linregress
                x = np.arange(len(series))
                slope, _, r_value, p_value, _ = linregress(x, series)
                trend_strength = abs(r_value) if p_value < 0.05 else 0
                trend_score = min(trend_strength * 2, 1.0)  # Cap at 1.0
                score_components["trend_strength"] = trend_score
                
                # Seasonality (detectable seasonality helps forecasting)
                seasonality_score = 0
                try:
                    from statsmodels.tsa.seasonal import STL
                    if len(series) >= 24:
                        period = min(12, len(series) // 3)
                        stl = STL(series, seasonal=period).fit()
                        seasonal_var = np.var(stl.seasonal)
                        total_var = np.var(series)
                        seasonality_strength = seasonal_var / total_var if total_var > 0 else 0
                        seasonality_score = min(seasonality_strength * 2, 1.0)
                except:
                    pass
                score_components["seasonality"] = seasonality_score
                
                # Noise level (lower noise = better for forecasting)
                returns = series.pct_change().dropna()
                if len(returns) > 1:
                    noise_level = returns.std()
                    # Normalize noise score (lower noise = higher score)
                    noise_score = max(0, 1.0 - min(noise_level, 1.0))
                else:
                    noise_score = 0.5
                score_components["signal_to_noise"] = noise_score
                
                # Autocorrelation (predictable patterns)
                autocorr_score = 0
                try:
                    from statsmodels.tsa.stattools import acf
                    acf_vals = acf(series, nlags=min(20, len(series)//2), fft=True)
                    significant_lags = (np.abs(acf_vals[1:]) > 1.96/np.sqrt(len(series))).sum()
                    autocorr_score = min(significant_lags / 10, 1.0)
                except:
                    pass
                score_components["autocorrelation"] = autocorr_score
                
                # Outlier penalty
                q75, q25 = np.percentile(series, [75, 25])
                iqr = q75 - q25
                outliers = ((series < (q25 - 1.5 * iqr)) | (series > (q75 + 1.5 * iqr))).sum()
                outlier_pct = outliers / len(series)
                outlier_score = max(0, 1.0 - outlier_pct * 5)  # Heavy penalty for outliers
                score_components["outlier_robustness"] = outlier_score
                
                # Overall readiness score (weighted average)
                weights = {
                    "data_sufficiency": 0.20,
                    "completeness": 0.15,
                    "stationarity": 0.15,
                    "trend_strength": 0.10,
                    "seasonality": 0.10,
                    "signal_to_noise": 0.15,
                    "autocorrelation": 0.10,
                    "outlier_robustness": 0.05
                }
                
                overall_score = sum(score_components[component] * weights[component] 
                                  for component in score_components)
                
                # Readiness classification
                if overall_score >= 0.8:
                    readiness_level = "excellent"
                elif overall_score >= 0.6:
                    readiness_level = "good"
                elif overall_score >= 0.4:
                    readiness_level = "fair"
                else:
                    readiness_level = "poor"
                
                # Recommendations
                recommendations = []
                if score_components["data_sufficiency"] < 0.5:
                    recommendations.append("Collect more historical data")
                if score_components["completeness"] < 0.8:
                    recommendations.append("Address missing values")
                if score_components["stationarity"] < 0.7:
                    recommendations.append("Apply differencing or transformation")
                if score_components["outlier_robustness"] < 0.7:
                    recommendations.append("Clean outliers")
                if score_components["signal_to_noise"] < 0.5:
                    recommendations.append("Apply smoothing techniques")
                
                readiness_scores[col] = {
                    "overall_score": overall_score,
                    "readiness_level": readiness_level,
                    "component_scores": score_components,
                    "recommendations": recommendations
                }
                
            except Exception as e:
                print(f"Forecasting readiness assessment failed for {col}: {e}")
        
        return readiness_scores

    def _analyze_volatility(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Advanced volatility analysis including GARCH effects and volatility clustering"""
        volatility_results = {}
        
        for col in numeric_cols[:5]:
            series = data[col].dropna()
            if len(series) < 50:
                continue
                
            try:
                volatility_analysis = {}
                
                # Calculate returns
                returns = series.pct_change().dropna()
                if len(returns) < 20:
                    continue
                
                # Basic volatility measures
                volatility_analysis["basic_measures"] = {
                    "returns_mean": returns.mean(),
                    "returns_std": returns.std(),
                    "annualized_volatility": returns.std() * np.sqrt(252),  # Assuming daily data
                    "skewness": returns.skew(),
                    "kurtosis": returns.kurtosis(),
                    "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0
                }
                
                # Volatility clustering tests
                try:
                    from statsmodels.stats.diagnostic import het_arch

                    # ARCH test
                    if len(returns) > 10:
                        arch_lm_stat, arch_p_val, _, _ = het_arch(returns, nlags=5)
                        volatility_analysis["arch_test"] = {
                            "statistic": arch_lm_stat,
                            "p_value": arch_p_val,
                            "volatility_clustering": arch_p_val < 0.05
                        }
                except: pass
                
                # Rolling volatility analysis
                try:
                    window_size = min(20, len(returns) // 5)
                    rolling_vol = returns.rolling(window_size).std()
                    
                    volatility_analysis["rolling_volatility"] = {
                        "mean": rolling_vol.mean(),
                        "std": rolling_vol.std(),
                        "min": rolling_vol.min(),
                        "max": rolling_vol.max(),
                        "volatility_of_volatility": rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() > 0 else 0
                    }
                    
                    # Volatility regime identification
                    vol_threshold_high = rolling_vol.quantile(0.75)
                    vol_threshold_low = rolling_vol.quantile(0.25)
                    
                    high_vol_periods = (rolling_vol > vol_threshold_high).sum()
                    low_vol_periods = (rolling_vol < vol_threshold_low).sum()
                    
                    volatility_analysis["volatility_regimes"] = {
                        "high_vol_periods": high_vol_periods,
                        "low_vol_periods": low_vol_periods,
                        "high_vol_pct": high_vol_periods / len(rolling_vol.dropna()) * 100,
                        "low_vol_pct": low_vol_periods / len(rolling_vol.dropna()) * 100
                    }
                    
                except: pass
                
                # GARCH modeling readiness
                try:
                    # Test for GARCH effects
                    squared_returns = returns ** 2
                    from statsmodels.tsa.stattools import acf

                    # Autocorrelation in squared returns
                    acf_sq = acf(squared_returns, nlags=min(10, len(squared_returns)//4))
                    significant_acf = (np.abs(acf_sq[1:]) > 1.96/np.sqrt(len(squared_returns))).sum()
                    
                    volatility_analysis["garch_effects"] = {
                        "squared_returns_acf": acf_sq[1:5].tolist() if len(acf_sq) > 5 else acf_sq[1:].tolist(),
                        "significant_lags": int(significant_acf),
                        "garch_suitable": significant_acf > 0
                    }
                    
                except: pass
                
                # Value at Risk (VaR) estimates
                try:
                    var_95 = np.percentile(returns, 5)
                    var_99 = np.percentile(returns, 1)
                    
                    # Expected Shortfall (Conditional VaR)
                    es_95 = returns[returns <= var_95].mean()
                    es_99 = returns[returns <= var_99].mean()
                    
                    volatility_analysis["risk_measures"] = {
                        "var_95": var_95,
                        "var_99": var_99,
                        "expected_shortfall_95": es_95,
                        "expected_shortfall_99": es_99,
                        "max_drawdown": (returns.cumsum() - returns.cumsum().expanding().max()).min()
                    }
                    
                except: pass
                
                volatility_results[col] = volatility_analysis
                
            except Exception as e:
                print(f"Volatility analysis failed for {col}: {e}")
        
        return volatility_results

    def _detect_cyclical_patterns(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Detect cyclical patterns using advanced spectral analysis"""
        cyclical_results = {}
        
        for col in numeric_cols[:5]:
            series = data[col].dropna()
            if len(series) < 50:
                continue
                
            try:
                cyclical_analysis = {}
                
                # Fourier Transform Analysis
                try:
                    from scipy.fft import fft, fftfreq

                    # Detrend the series
                    detrended = series - series.rolling(window=min(12, len(series)//4)).mean()
                    detrended = detrended.dropna()
                    
                    if len(detrended) > 20:
                        # Compute FFT
                        fft_vals = fft(detrended.values)
                        freqs = fftfreq(len(detrended))
                        
                        # Get power spectrum
                        power_spectrum = np.abs(fft_vals)**2
                        
                        # Find dominant frequencies (excluding DC component)
                        positive_freqs = freqs[:len(freqs)//2][1:]  # Exclude DC
                        positive_power = power_spectrum[:len(power_spectrum)//2][1:]
                        
                        # Find peaks in power spectrum
                        peak_indices = find_peaks(positive_power, height=np.percentile(positive_power, 85))[0]
                        
                        dominant_cycles = []
                        for peak_idx in peak_indices[:5]:  # Top 5 peaks
                            freq = positive_freqs[peak_idx]
                            if freq > 0:
                                period = 1 / freq
                                power = positive_power[peak_idx]
                                dominant_cycles.append({
                                    "period": period,
                                    "frequency": freq,
                                    "power": power,
                                    "power_normalized": power / np.sum(positive_power)
                                })
                        
                        cyclical_analysis["fourier_analysis"] = {
                            "dominant_cycles": sorted(dominant_cycles, key=lambda x: x["power"], reverse=True),
                            "total_cycles_detected": len(dominant_cycles)
                        }
                        
                except: pass
                
                # Wavelet Analysis (simplified)
                try:
                    from scipy.signal import cwt, ricker

                    # Use Ricker wavelets for different scales
                    scales = np.arange(2, min(50, len(series)//4))
                    coefficients = cwt(series.values, ricker, scales)
                    
                    # Find scales with highest energy
                    energy_per_scale = np.sum(coefficients**2, axis=1)
                    dominant_scale_idx = np.argmax(energy_per_scale)
                    dominant_scale = scales[dominant_scale_idx]
                    
                    cyclical_analysis["wavelet_analysis"] = {
                        "dominant_scale": dominant_scale,
                        "energy_at_dominant_scale": energy_per_scale[dominant_scale_idx],
                        "scales_analyzed": len(scales)
                    }
                    
                except: pass
                
                # Business cycle detection (for economic data)
                try:
                    # Hodrick-Prescott filter for cycle extraction
                    from statsmodels.tsa.filters.hp_filter import hpfilter
                    
                    hp_cycle, hp_trend = hpfilter(series, lamb=1600)  # Standard lambda for quarterly data
                    
                    # Analyze cycle characteristics
                    cycle_peaks = find_peaks(hp_cycle)[0]
                    cycle_troughs = find_peaks(-hp_cycle)[0]
                    
                    # Calculate cycle durations
                    cycle_durations = []
                    if len(cycle_peaks) > 1:
                        cycle_durations = np.diff(cycle_peaks)
                    
                    cyclical_analysis["business_cycle"] = {
                        "cycle_component_std": hp_cycle.std(),
                        "trend_component_std": hp_trend.std(),
                        "n_peaks": len(cycle_peaks),
                        "n_troughs": len(cycle_troughs),
                        "avg_cycle_duration": np.mean(cycle_durations) if cycle_durations else None,
                        "cycle_amplitude": np.max(hp_cycle) - np.min(hp_cycle)
                    }
                    
                except: pass
                
                # Phase analysis
                try:
                    # Hilbert transform for instantaneous phase
                    from scipy.signal import hilbert
                    
                    analytic_signal = hilbert(detrended.values)
                    instantaneous_phase = np.angle(analytic_signal)
                    instantaneous_amplitude = np.abs(analytic_signal)
                    
                    # Phase consistency measure
                    phase_diff = np.diff(instantaneous_phase)
                    phase_consistency = 1 - np.std(phase_diff) / np.pi  # Normalized to [0,1]
                    
                    cyclical_analysis["phase_analysis"] = {
                        "phase_consistency": phase_consistency,
                        "mean_amplitude": np.mean(instantaneous_amplitude),
                        "amplitude_variability": np.std(instantaneous_amplitude) / np.mean(instantaneous_amplitude)
                    }
                    
                except: pass
                
                cyclical_results[col] = cyclical_analysis
                
            except Exception as e:
                print(f"Cyclical pattern detection failed for {col}: {e}")
        
        return cyclical_results

    def _granger_causality_analysis(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Granger causality analysis between time series"""
        if len(numeric_cols) < 2:
            return {}
        
        causality_results = {}
        
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # Test causality between all pairs of variables
            for i, var1 in enumerate(numeric_cols[:5]):  # Limit to first 5 for computational efficiency
                for var2 in enumerate(numeric_cols[:5]):
                    if var1 != var2 and var1 in data.columns and var2 in data.columns:
                        
                        # Prepare data
                        pair_data = data[[var1, var2]].dropna()
                        if len(pair_data) < 50:  # Need sufficient data
                            continue
                        
                        try:
                            # Test both directions
                            max_lag = min(12, len(pair_data) // 10)
                            
                            # var1 -> var2
                            try:
                                test_data_12 = pair_data[[var2, var1]].values  # Note: order matters in granger test
                                gc_result_12 = grangercausalitytests(test_data_12, maxlag=max_lag, verbose=False)
                                
                                # Extract p-values for different lags
                                p_values_12 = {}
                                for lag in range(1, max_lag + 1):
                                    if lag in gc_result_12:
                                        p_val = gc_result_12[lag][0]['ssr_ftest'][1]  # F-test p-value
                                        p_values_12[lag] = p_val
                                
                                # Find best lag (lowest p-value)
                                if p_values_12:
                                    best_lag_12 = min(p_values_12.keys(), key=lambda x: p_values_12[x])
                                    best_p_val_12 = p_values_12[best_lag_12]
                                    causality_12 = best_p_val_12 < 0.05
                                else:
                                    best_lag_12 = None
                                    best_p_val_12 = 1.0
                                    causality_12 = False
                                    
                            except:
                                best_lag_12 = None
                                best_p_val_12 = 1.0
                                causality_12 = False
                            
                            # var2 -> var1
                            try:
                                test_data_21 = pair_data[[var1, var2]].values
                                gc_result_21 = grangercausalitytests(test_data_21, maxlag=max_lag, verbose=False)
                                
                                p_values_21 = {}
                                for lag in range(1, max_lag + 1):
                                    if lag in gc_result_21:
                                        p_val = gc_result_21[lag][0]['ssr_ftest'][1]
                                        p_values_21[lag] = p_val
                                
                                if p_values_21:
                                    best_lag_21 = min(p_values_21.keys(), key=lambda x: p_values_21[x])
                                    best_p_val_21 = p_values_21[best_lag_21]
                                    causality_21 = best_p_val_21 < 0.05
                                else:
                                    best_lag_21 = None
                                    best_p_val_21 = 1.0
                                    causality_21 = False
                                    
                            except:
                                best_lag_21 = None
                                best_p_val_21 = 1.0
                                causality_21 = False
                            
                            # Store results if any causality detected
                            if causality_12 or causality_21:
                                pair_key = f"{var1}_vs_{var2}"
                                causality_results[pair_key] = {
                                    "var1_causes_var2": {
                                        "significant": causality_12,
                                        "best_lag": best_lag_12,
                                        "p_value": best_p_val_12
                                    },
                                    "var2_causes_var1": {
                                        "significant": causality_21,
                                        "best_lag": best_lag_21,
                                        "p_value": best_p_val_21
                                    },
                                    "bidirectional": causality_12 and causality_21,
                                    "sample_size": len(pair_data)
                                }
                                
                        except Exception as e:
                            print(f"Granger causality test failed for {var1} vs {var2}: {e}")
                            continue
            
        except ImportError:
            print("statsmodels not available for Granger causality tests")
        except Exception as e:
            print(f"Granger causality analysis failed: {e}")
        
        return causality_results
    
class PatternDetector(AnalysisStrategy):
    """SOTA Pattern Detection with Advanced ML Techniques"""

    @property
    def name(self) -> str:
        return "patterns"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        return {
            "feature_types": self._classify_features(data, config),
            "relationships": self._analyze_relationships(data, config),
            "distributions": self._fit_distributions(data, config),
        }

    def _classify_features(self, data, config) -> Dict[str, List[str]]:
        from scipy.stats import anderson, jarque_bera, shapiro
        from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
        from sklearn.preprocessing import PowerTransformer
        
        dist_summary = DistributionAnalyzer().analyze(data, config).get("summary", pd.DataFrame())
        classification = {
            "gaussian": [], "log_normal_candidates": [], "bounded": [], "count_like": [],
            "continuous": [], "highly_skewed": [], "heavy_tailed": [], "mixture_model": [],
            "seasonal": [], "power_law": [], "bimodal": [], "uniform_like": [],
            "zero_inflated": [], "transformable_to_normal": []
        }

        time_col = getattr(config, "time_col", None)

        for _, row in dist_summary.iterrows():
            feature = row["feature"]
            skew, kurt, std, mean = row["skewness"], row["kurtosis"], row["std"], row["mean"]
            col_data = data[feature].dropna()
            if len(col_data) < 10: continue

            # Enhanced normality testing
            is_normal = False
            if len(col_data) <= 5000:
                try:
                    _, jb_p = jarque_bera(col_data)
                    _, sw_p = shapiro(col_data.sample(min(5000, len(col_data))))
                    anderson_stat, critical_vals, _ = anderson(col_data, dist='norm')
                    is_normal = jb_p > 0.05 and sw_p > 0.05 and anderson_stat < critical_vals[2]
                except: pass

            if is_normal or row.get("is_gaussian", False):
                classification["gaussian"].append(feature)

            # Advanced distribution classification
            if row.get("is_heavy_tailed", False) or kurt > 5:
                classification["heavy_tailed"].append(feature)
            
            if abs(skew) > 2:
                classification["highly_skewed"].append(feature)
            
            # Log-normal detection with better heuristics
            if (mean > 0 and skew > 1 and std > 0 and 
                col_data.min() > 0 and np.log(col_data).std() < col_data.std()):
                classification["log_normal_candidates"].append(feature)
            
            # Zero-inflated detection
            zero_pct = (col_data == 0).mean()
            if zero_pct > 0.1 and zero_pct < 0.9:
                classification["zero_inflated"].append(feature)
            
            # Bounded/discrete detection
            if col_data.min() >= 0 and col_data.max() <= 1:
                classification["bounded"].append(feature)
            elif np.all(np.isclose(col_data % 1, 0)) and col_data.var() / (col_data.mean() + 1e-8) < 3:
                classification["count_like"].append(feature)
            
            # Power law detection
            if col_data.min() > 0:
                log_data = np.log(col_data)
                if log_data.std() > 1 and skew > 2:
                    # Simple power law test
                    x_log = np.log(np.sort(col_data)[::-1])
                    y_log = np.log(np.arange(1, len(x_log) + 1))
                    if len(x_log) > 10:
                        corr = np.corrcoef(x_log[:len(x_log)//2], y_log[:len(x_log)//2])[0,1]
                        if abs(corr) > 0.8:
                            classification["power_law"].append(feature)
            
            # Uniform-like detection
            if abs(skew) < 0.5 and abs(kurt) < 1.5:
                classification["uniform_like"].append(feature)
            
            # Mixture model detection with Bayesian approach
            if kurt > 1 or len(np.unique(col_data)) > 20:
                try:
                    # Try both regular and Bayesian GMM
                    gmm = GaussianMixture(n_components=2, random_state=config.random_state)
                    bgmm = BayesianGaussianMixture(n_components=3, random_state=config.random_state)
                    
                    gmm.fit(col_data.values.reshape(-1, 1))
                    bgmm.fit(col_data.values.reshape(-1, 1))
                    
                    # Use AIC/BIC for model selection
                    single_aic = 2 * 2 - 2 * np.sum(np.log(col_data.std() * np.sqrt(2 * np.pi)) - 
                                                   0.5 * ((col_data - col_data.mean()) / col_data.std()) ** 2)
                    
                    if gmm.converged_ and gmm.aic_ < single_aic:
                        classification["mixture_model"].append(feature)
                    elif bgmm.converged_ and hasattr(bgmm, 'lower_bound_') and bgmm.n_components_ > 1:
                        classification["bimodal"].append(feature)
                except: pass
            
            # Test transformability to normal
            if not is_normal and col_data.min() >= 0:
                try:
                    pt = PowerTransformer(method='yeo-johnson')
                    transformed = pt.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                    _, trans_p = shapiro(transformed.sample(min(5000, len(transformed))))
                    if trans_p > 0.05:
                        classification["transformable_to_normal"].append(feature)
                except: pass
            
            # Default continuous classification
            if (feature not in classification["count_like"] + classification["gaussian"] + 
                classification["bounded"] + classification["zero_inflated"]):
                classification["continuous"].append(feature)

            # Seasonality detection
            if time_col and pd.api.types.is_datetime64_any_dtype(data[time_col]):
                try:
                    from statsmodels.tsa.seasonal import STL
                    from statsmodels.tsa.stattools import acf
                    
                    series = data.set_index(time_col)[feature].dropna()
                    if len(series) >= 24:
                        # STL decomposition
                        stl = STL(series, seasonal=min(13, len(series)//3)).fit()
                        seasonal_strength = np.var(stl.seasonal) / (np.var(series) + 1e-8)
                        
                        # Autocorrelation analysis
                        autocorr = acf(series, nlags=min(40, len(series)//4), fft=True)
                        
                        if seasonal_strength > 0.15 or np.max(np.abs(autocorr[12:])) > 0.3:
                            classification["seasonal"].append(feature)
                except: pass

        return classification

    def _analyze_relationships(self, data, config) -> Dict[str, Any]:
        try:
            from dcor import distance_correlation
            DCOR_AVAILABLE = True
        except ImportError:
            DCOR_AVAILABLE = False
        
        from scipy.stats import kendalltau, spearmanr
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.metrics import mutual_info_score
        
        correlations = CorrelationAnalyzer().analyze(data, config)
        pearson = correlations.get("pearson")
        if pearson is None:
            return {}

        patterns = {
            "nonlinear": [], "complex": [], "distance_corr": [],
            "monotonic": [], "tail_dependence": [], "conditional": []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                f1, f2 = numeric_cols[i], numeric_cols[j]
                
                # Skip if too many missing values
                valid_data = data[[f1, f2]].dropna()
                if len(valid_data) < 10:
                    continue
                    
                x, y = valid_data[f1], valid_data[f2]
                
                # Standard correlations
                pc = abs(pearson.loc[f1, f2]) if f1 in pearson.index and f2 in pearson.columns else 0
                
                try:
                    sc, _ = spearmanr(x, y)
                    sc = abs(sc)
                    tau, _ = kendalltau(x, y)
                    tau = abs(tau)
                except:
                    sc = tau = 0
                
                # Mutual information
                try:
                    if len(x) > 50:
                        mi_val = mutual_info_regression(x.values.reshape(-1, 1), y, 
                                                      random_state=config.random_state)[0]
                    else:
                        mi_val = 0
                except:
                    mi_val = 0
                
                # Distance correlation
                dc = 0
                if DCOR_AVAILABLE and len(x) <= 1000:  # Limit for performance
                    try:
                        dc = distance_correlation(x, y)
                    except: pass
                
                # Nonlinear relationships
                if sc > 0.4 and sc - pc > 0.2:
                    patterns["nonlinear"].append({
                        "feature1": f1, "feature2": f2,
                        "pearson": pc, "spearman": sc, "kendall": tau,
                        "nonlinearity_score": sc - pc
                    })
                
                # Complex relationships via mutual information
                if mi_val > 0.2 and pc < 0.4:
                    patterns["complex"].append({
                        "feature1": f1, "feature2": f2,
                        "mutual_info": mi_val, "pearson": pc,
                        "complexity_score": mi_val - pc
                    })
                
                # Distance correlation patterns
                if dc > 0.4 and pc < 0.4:
                    patterns["distance_corr"].append({
                        "feature1": f1, "feature2": f2, 
                        "distance_corr": dc, "pearson": pc
                    })
                
                # Monotonic relationships
                if tau > 0.5:
                    patterns["monotonic"].append({
                        "feature1": f1, "feature2": f2,
                        "kendall_tau": tau, "relationship_type": "monotonic"
                    })
                
                # Tail dependence (simplified)
                if len(x) > 100:
                    try:
                        # Upper tail dependence
                        q95_x, q95_y = np.percentile(x, 95), np.percentile(y, 95)
                        upper_tail = ((x >= q95_x) & (y >= q95_y)).mean()
                        
                        if upper_tail > 0.05:  # 5% threshold
                            patterns["tail_dependence"].append({
                                "feature1": f1, "feature2": f2,
                                "upper_tail_dep": upper_tail,
                                "type": "upper_tail"
                            })
                    except: pass

        # Sort and limit results
        for k in patterns:
            if patterns[k]:
                sort_key = lambda x: list(x.values())[-1] if isinstance(list(x.values())[-1], (int, float)) else 0
                patterns[k] = sorted(patterns[k], key=sort_key, reverse=True)[:10]

        return patterns

    def _fit_distributions(self, data, config) -> Dict[str, Dict[str, Any]]:
        import warnings

        from scipy import stats
        warnings.filterwarnings('ignore')
        
        # Extended list of distributions
        distributions = [
            ("normal", stats.norm),
            ("lognormal", stats.lognorm),
            ("exponential", stats.expon),
            ("gamma", stats.gamma),
            ("beta", stats.beta),
            ("weibull_min", stats.weibull_min),
            ("pareto", stats.pareto),
            ("gumbel_r", stats.gumbel_r),
            ("uniform", stats.uniform),
            ("chi2", stats.chi2),
            ("t", stats.t),
            ("f", stats.f),
            ("logistic", stats.logistic),
            ("laplace", stats.laplace),
            ("rayleigh", stats.rayleigh)
        ]

        fits = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:15]:  # Limit for performance
            x = data[col].dropna()
            if len(x) < 30: continue
            
            # Sample for large datasets
            if len(x) > 2000:
                x = x.sample(2000, random_state=config.random_state)
            
            candidate_fits = []
            
            for name, dist in distributions:
                try:
                    # Fit distribution
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        params = dist.fit(x)
                    
                    # Calculate goodness of fit metrics
                    log_likelihood = np.sum(dist.logpdf(x, *params))
                    k = len(params)  # number of parameters
                    n = len(x)       # sample size
                    
                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.kstest(x, dist.cdf, args=params)
                    
                    # Anderson-Darling test (where applicable)
                    ad_stat = ad_p = None
                    if name in ['normal', 'exponential', 'logistic', 'gumbel_r']:
                        try:
                            ad_result = stats.anderson(x, dist=name.replace('_r', ''))
                            ad_stat = ad_result.statistic
                            # Approximate p-value based on critical values
                            if ad_stat < ad_result.critical_values[2]:
                                ad_p = 0.05  # Rough approximation
                            else:
                                ad_p = 0.01
                        except: pass
                    
                    candidate_fits.append({
                        "distribution": name,
                        "params": params,
                        "aic": aic,
                        "bic": bic,
                        "log_likelihood": log_likelihood,
                        "ks_statistic": ks_stat,
                        "ks_pvalue": ks_p,
                        "ad_statistic": ad_stat,
                        "ad_pvalue": ad_p,
                        "fit_quality": ks_p if ks_p > 0 else 1e-10  # For sorting
                    })
                    
                except Exception as e:
                    continue
            
            if candidate_fits:
                # Sort by multiple criteria: AIC, then KS p-value
                candidate_fits.sort(key=lambda x: (x["aic"], -x["fit_quality"]))
                
                # Best fit for backward compatibility (flat structure)
                best_fit = candidate_fits[0]
                fits[col] = {
                    "distribution": best_fit["distribution"],
                    "params": best_fit["params"],
                    "aic": best_fit["aic"],
                    "bic": best_fit["bic"],
                    "ks_stat": best_fit["ks_statistic"],
                    "ks_pvalue": best_fit["ks_pvalue"]
                }
                
                # Store detailed results for advanced users
                fits[col]["_detailed"] = {
                    "best_fit": best_fit,
                    "alternatives": candidate_fits[1:3],  # Top 3 alternatives
                    "total_tested": len(candidate_fits)
                }

        return fits
    
class MissingnessAnalyzer(AnalysisStrategy):
    """SOTA analysis of missingness patterns, correlations, and MNAR behavior"""

    @property
    def name(self) -> str:
        return "missingness"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        results = {}
        missing_stats = data.isnull().mean()

        # Basic missing summary
        results["missing_rate"] = missing_stats[missing_stats > 0].sort_values(ascending=False)

        # Missingness correlation (standard + Jaccard)
        missing_mask = data.isnull().astype(int)
        results["missingness_correlation"] = missing_mask.corr()

        # Pairwise Jaccard similarity
        jaccard_matrix = missing_mask.T.dot(missing_mask)
        counts = missing_mask.sum(axis=0)
        jaccard_matrix = jaccard_matrix.div(counts, axis=0).div(counts, axis=1)
        results["missingness_jaccard"] = jaccard_matrix

        # Optionally cluster missingness patterns
        if missing_mask.shape[1] >= 2:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(missing_mask.T, metric="hamming")
            clustering = AgglomerativeClustering(n_clusters=min(5, len(missing_mask.columns)), linkage='average')
            results["missingness_clusters"] = dict(zip(missing_mask.columns, clustering.fit_predict(dists)))

        # MNAR analysis using both numeric and categorical targets
        target_col = getattr(config, "target", None)
        if target_col and target_col in data.columns:
            results["missing_vs_target"] = self._analyze_mnar(data, target_col, config)

        return results

    def _analyze_mnar(self, data: pd.DataFrame, target_col: str, config: AnalysisConfig) -> Dict[str, Any]:
        import warnings

        from scipy.stats import chi2_contingency, ks_2samp, ttest_ind
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import mutual_info_score, roc_auc_score

        warnings.filterwarnings("ignore")
        target = data[target_col]
        insights = {}

        for col in data.columns:
            if col == target_col or data[col].isnull().sum() == 0:
                continue

            miss_indicator = data[col].isnull().astype(int)

            try:
                if target.dtype.kind in "ifc":  # Continuous target
                    group1 = target[miss_indicator == 1].dropna()
                    group2 = target[miss_indicator == 0].dropna()
                    if len(group1) > 5 and len(group2) > 5:
                        _, pval_t = ttest_ind(group1, group2, equal_var=False)
                        _, pval_ks = ks_2samp(group1, group2)

                        # Predicting missingness with logistic regression
                        if len(set(miss_indicator)) == 2:
                            lr_model = LogisticRegression(solver="liblinear")
                            x_input = target.values.reshape(-1, 1)
                            lr_model.fit(x_input, miss_indicator)
                            auc = roc_auc_score(miss_indicator, lr_model.predict_proba(x_input)[:, 1])
                        else:
                            auc = 0

                        insights[col] = {
                            "ttest_p": pval_t,
                            "ks_p": pval_ks,
                            "auc": auc,
                            "suggested_mnar": (pval_t < 0.05 or pval_ks < 0.05 or auc > 0.7),
                        }

                elif target.dtype.name == "category" or target.nunique() < 15:
                    contingency = pd.crosstab(miss_indicator, target)
                    if contingency.shape[0] == 2 and contingency.shape[1] >= 2:
                        _, pval, _, _ = chi2_contingency(contingency)
                        cramer_v = np.sqrt(pval * min(contingency.shape) / len(target))
                        mi = mutual_info_score(miss_indicator, target)
                        insights[col] = {
                            "chi2_p": pval,
                            "cramers_v": cramer_v,
                            "mutual_info": mi,
                            "suggested_mnar": pval < 0.05 or cramer_v > 0.1 or mi > 0.05,
                        }

            except Exception as e:
                if config.verbose:
                    print(f"MNAR analysis failed for {col}: {e}")

        return insights

class FeatureEngineeringAnalyzer(AnalysisStrategy):
    """SOTA intelligent feature engineering with advanced ML techniques"""

    @property
    def name(self) -> str:
        return "feature_engineering"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        from itertools import combinations, permutations

        import numpy as np
        import pandas as pd
        import scipy.stats as stats
        from scipy.stats import anderson, jarque_bera, shapiro
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.feature_selection import (
            SelectKBest,
            f_regression,
            mutual_info_regression,
        )
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import (
            PowerTransformer,
            QuantileTransformer,
            RobustScaler,
        )
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tsa.seasonal import STL
        try:
            import shap
            SHAP_AVAILABLE = True
        except ImportError:
            SHAP_AVAILABLE = False
        try:
            from category_encoders import CatBoostEncoder, TargetEncoder, WOEEncoder
            CATEGORY_ENCODERS_AVAILABLE = True
        except ImportError:
            CATEGORY_ENCODERS_AVAILABLE = False

        # Initialize both structures properly
        suggestions = {}
        detailed_results = {
            "transformations": {}, 
            "interactions": [], 
            "encodings": {}, 
            "feature_ranking": {}, 
            "dimensionality_reduction": [],
            "time_series_features": {}, 
            "advanced_features": []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Remove time and target columns from feature lists
        time_col = getattr(config, "time_col", None)
        target_col = getattr(config, "target", None)
        if time_col in numeric_cols: numeric_cols.remove(time_col)
        if target_col in numeric_cols: numeric_cols.remove(target_col)
        if time_col in categorical_cols: categorical_cols.remove(time_col)
        if target_col in categorical_cols: categorical_cols.remove(target_col)

        # === Advanced Distribution Analysis & Transformations ===
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) < 10: continue
            
            transforms = []
            col_info = {
                "skewness": stats.skew(series),
                "kurtosis": stats.kurtosis(series),
                "cv": series.std() / (abs(series.mean()) + 1e-8),
                "outliers_pct": ((series < (series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25)))) | 
                                (series > (series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25))))).mean() * 100
            }
            
            # Normality tests
            try:
                jb_stat, jb_p = jarque_bera(series.sample(min(5000, len(series))))
                sw_stat, sw_p = shapiro(series.sample(min(5000, len(series))))
                col_info["normality_jb_p"] = jb_p
                col_info["normality_sw_p"] = sw_p
            except: pass
            
            # Smart transformation suggestions based on distribution properties
            if abs(col_info["skewness"]) > 2:
                if series.min() > 0:
                    transforms.extend([f"np.log1p({col})", f"np.sqrt({col})", f"1/({col}+1e-8)"])
                else:
                    transforms.extend([f"np.sign({col}) * np.log1p(np.abs({col}))", f"scipy.stats.yeojohnson({col})[0]"])
            
            if col_info["kurtosis"] > 3:  # Heavy tails
                transforms.extend([f"np.tanh({col}/series.std())", f"scipy.stats.rankdata({col})/len({col})"])
            
            if col_info["outliers_pct"] > 10:
                transforms.extend([f"scipy.stats.mstats.winsorize({col}, limits=(0.05, 0.05))",
                                f"RobustScaler().fit_transform({col}.values.reshape(-1,1)).flatten()"])
            
            if col_info["cv"] > 3:  # High variability
                transforms.extend([f"pd.qcut({col}, q=10, labels=False, duplicates='drop')",
                                f"PowerTransformer(method='yeo-johnson').fit_transform({col}.values.reshape(-1,1)).flatten()"])
            
            # Advanced statistical transformations
            if series.min() >= 0:
                transforms.extend([f"np.power({col}, 1/3)", f"np.exp(-{col}/series.mean())"])
            
            transforms.extend([f"({col} - series.mean()) / series.std()",  # Z-score
                             f"({col} - series.median()) / series.mad()"])  # Robust standardization
            
            # Store in detailed results
            detailed_results["transformations"][col] = {
                "stats": col_info,
                "recommended": transforms[:8]  # Top 8 transforms
            }
            
            # Backward compatibility - flat list for existing interface
            suggestions[col] = transforms[:4]

        # === Ensemble-based Feature Importance & Selection ===
        if target_col and target_col in data.columns:
            X = data[numeric_cols].fillna(data[numeric_cols].median())
            y = data[target_col].fillna(data[target_col].median())
            
            if len(X) > 50 and X.shape[1] > 1:
                # Multiple model ensemble for robust importance
                models = [
                    RandomForestRegressor(n_estimators=100, random_state=42),
                    GradientBoostingRegressor(n_estimators=100, random_state=42),
                    ExtraTreesRegressor(n_estimators=100, random_state=42)
                ]
                
                importance_scores = {}
                for model in models:
                    try:
                        model.fit(X, y)
                        for i, col in enumerate(numeric_cols):
                            importance_scores.setdefault(col, []).append(model.feature_importances_[i])
                    except: continue
                
                # Aggregate importance scores
                avg_importance = {col: np.mean(scores) for col, scores in importance_scores.items()}
                detailed_results["feature_ranking"] = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
                
                # SHAP analysis if available
                if SHAP_AVAILABLE and len(X) < 1000:  # Limit for performance
                    try:
                        model = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X, y)
                        explainer = shap.Explainer(model, X.sample(min(100, len(X))))
                        shap_values = explainer(X.sample(min(200, len(X))))
                        shap_importance = np.abs(shap_values.values).mean(axis=0)
                        detailed_results["feature_ranking"]["shap_importance"] = dict(zip(numeric_cols, shap_importance))
                    except: pass

        # === Intelligent Feature Interactions ===
        if len(numeric_cols) >= 2:
            top_features = list(detailed_results.get("feature_ranking", {}).keys())[:10] or numeric_cols[:10]
            
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    if feat1 in data.columns and feat2 in data.columns:
                        # Calculate interaction strength
                        try:
                            corr = data[[feat1, feat2]].corr().iloc[0, 1]
                            if abs(corr) < 0.9:  # Avoid highly correlated pairs
                                interactions = [
                                    f"{feat1} * {feat2}",
                                    f"{feat1} / ({feat2} + 1e-8)",
                                    f"np.sqrt({feat1}**2 + {feat2}**2)",
                                    f"np.maximum({feat1}, {feat2})",
                                    f"np.minimum({feat1}, {feat2})",
                                    f"({feat1} + {feat2}) / 2",
                                    f"abs({feat1} - {feat2})",
                                    f"np.where({feat1} > {feat2}, 1, 0)"
                                ]
                                detailed_results["interactions"].extend(interactions)
                        except: continue
            
            # Backward compatibility - add interactions to main suggestions
            suggestions["interactions"] = detailed_results["interactions"][:20]

        # === Advanced Categorical Encoding ===
        for col in categorical_cols:
            nunique = data[col].nunique()
            null_pct = data[col].isnull().mean() * 100
            
            encoding_strategies = []
            
            if 2 <= nunique <= 5:
                encoding_strategies.append("pd.get_dummies(drop_first=True)")
            elif nunique <= 20:
                encoding_strategies.extend(["LabelEncoder", "OrdinalEncoder"])
                if target_col and CATEGORY_ENCODERS_AVAILABLE:
                    encoding_strategies.extend(["TargetEncoder", "WOEEncoder"])
            else:
                encoding_strategies.extend(["HashingEncoder", "FrequencyEncoder"])
                if target_col and CATEGORY_ENCODERS_AVAILABLE:
                    encoding_strategies.append("CatBoostEncoder")
            
            # Add rare category handling for high cardinality
            if nunique > 10:
                encoding_strategies.append(f"RareCategoryEncoder(threshold=0.01)")
            
            # Add encoding suggestions to main dict for backward compatibility  
            suggestions[col] = encoding_strategies[:3]  # Top 3 strategies
            detailed_results["encodings"][col] = {
                "cardinality": nunique,
                "null_percentage": null_pct,
                "strategies": encoding_strategies
            }

        # === Multicollinearity Detection (VIF) ===
        if len(numeric_cols) > 1 and len(numeric_cols) < 50:
            try:
                X_vif = data[numeric_cols].fillna(data[numeric_cols].median())
                vif_data = pd.DataFrame()
                vif_data["Feature"] = numeric_cols
                vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
                
                high_vif = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
                if high_vif:
                    detailed_results["dimensionality_reduction"].append(f"High VIF features (>10): {high_vif}")
                    detailed_results["dimensionality_reduction"].append("Consider PCA or feature selection")
            except: pass

        # === Time Series Feature Engineering ===
        if time_col and time_col in data.columns:
            time_series_features = []
            
            # Temporal features
            time_series_features.extend([
                f"df['{time_col}'].dt.hour",
                f"df['{time_col}'].dt.dayofweek", 
                f"df['{time_col}'].dt.month",
                f"df['{time_col}'].dt.quarter",
                f"df['{time_col}'].dt.is_weekend",
                f"df['{time_col}'].dt.dayofyear"
            ])
            
            # Cyclical encoding
            time_series_features.extend([
                f"np.sin(2 * np.pi * df['{time_col}'].dt.hour / 24)",
                f"np.cos(2 * np.pi * df['{time_col}'].dt.hour / 24)",
                f"np.sin(2 * np.pi * df['{time_col}'].dt.dayofyear / 365)",
                f"np.cos(2 * np.pi * df['{time_col}'].dt.dayofyear / 365)"
            ])
            
            # Lag and window features for numeric columns
            for col in numeric_cols[:5]:  # Limit to top 5 for performance
                try:
                    series = data.set_index(time_col)[col].dropna()
                    if len(series) > 24:
                        # Decomposition analysis
                        stl = STL(series, seasonal=min(13, len(series)//3)).fit()
                        trend_strength = np.var(stl.trend.dropna()) / (np.var(series) + 1e-8)
                        seasonal_strength = np.var(stl.seasonal) / (np.var(series) + 1e-8)
                        
                        lag_features = [
                            f"df['{col}'].shift(1)",
                            f"df['{col}'].shift(7)", 
                            f"df['{col}'].rolling(window=3).mean()",
                            f"df['{col}'].rolling(window=7).mean()",
                            f"df['{col}'].rolling(window=3).std()",
                            f"df['{col}'].expanding().mean()",
                            f"df['{col}'].pct_change()"
                        ]
                        
                        if trend_strength > 0.3:
                            lag_features.append(f"STL_trend_component")
                        if seasonal_strength > 0.3:
                            lag_features.append(f"STL_seasonal_component")
                        
                        detailed_results["time_series_features"][col] = lag_features
                except: continue
            
            detailed_results["time_series_features"]["temporal"] = time_series_features

        # === Advanced Feature Creation Patterns ===
        advanced_patterns = [
            "Polynomial features (degree 2-3) for top predictors",
            "Binning continuous variables into quantiles",
            "Distance/similarity features between data points",
            "Aggregated features by categorical groupings",
            "Ratio features between related numeric columns",
            "Deviation from group means/medians",
            "Percentile rank transformations",
            "Fourier transform features for periodic patterns"
        ]
        detailed_results["advanced_features"] = advanced_patterns

        # === Feature Selection Recommendations ===
        selection_methods = [
            "Recursive Feature Elimination (RFE)",
            "SelectKBest with mutual information",
            "LASSO regularization for sparse selection", 
            "Boruta algorithm for all-relevant features",
            "Permutation importance ranking"
        ]
        detailed_results["feature_selection_methods"] = selection_methods
        
        # Store detailed results for advanced users
        suggestions["_detailed"] = detailed_results

        return suggestions

class SHAPAnalyzer(AnalysisStrategy):
    """SHAP-based feature explanation"""

    @property
    def name(self) -> str:
        return "shap_explanation"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        # This would require a fitted model to be passed in
        # For now, return placeholder
        return {"explanation": "SHAP analysis requires a fitted model"}



# ============================================================================
# MAIN ANALYZER CLASS (Clean and comprehensive)
# ============================================================================


class DatasetAnalyzer:
    """Clean, extensible dataset analyzer with all advanced functionality preserved"""

    def __init__(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        config: Optional[AnalysisConfig] = None,
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.time_col = time_col
        self.config = config or AnalysisConfig()
        self.verbose = verbose

        # Initialize components
        self.hooks = AnalysisHooks()
        self.plot_helper = PlotHelper()
        self._strategies: Dict[str, AnalysisStrategy] = {}
        self._results_cache: Dict[str, Any] = {}

        # Cache frequently used data
        self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Register default strategies and plotters
        self._register_default_strategies()
        self._register_default_plotters()

        # Setup
        self._setup()

    def _setup(self):
        """Initialize analyzer setup"""
        PlotHelper.setup_style(self.config)
        self._setup_time_index()

        if self.verbose:
            print(
                f"🔍 Initialized analyzer with {self.df.shape[0]:,} rows × {self.df.shape[1]} columns"
            )
            print(f"   • Numeric features: {len(self._numeric_cols)}")
            print(f"   • Categorical features: {len(self._categorical_cols)}")

    def _setup_time_index(self):
        """Setup time index if provided"""
        if self.time_col and self.time_col in self.df.columns:
            self.df.set_index(self.time_col, inplace=True)
            self.df.index = pd.to_datetime(self.df.index)

    def _register_default_strategies(self):
        """Register all available analysis strategies"""
        strategies = [
            DistributionAnalyzer(),
            CorrelationAnalyzer(),
            OutlierAnalyzer(),
            ClusterAnalyzer(),
            DimensionalityAnalyzer(),
            PatternDetector(),
            MissingnessAnalyzer(),
            FeatureEngineeringAnalyzer(),
            SHAPAnalyzer(),
        ]

        # Add time series analyzer if time column is provided
        if self.time_col:
            strategies.append(TimeSeriesAnalyzer())

        for strategy in strategies:
            self._strategies[strategy.name] = strategy

    def _register_default_plotters(self):
        """Register all available plotters"""
        plotters = [
            ("distributions", PlotHelper.plot_distributions),
            ("correlations", PlotHelper.plot_correlations),
            ("outliers", PlotHelper.plot_outliers_pca),
            ("clusters", PlotHelper.plot_clusters),
            ("dimensionality", PlotHelper.plot_dimensionality),
            ("missingness", PlotHelper.plot_missingness_analysis),
        ]

        if OPTIONAL_IMPORTS["networkx"]:
            plotters.append(
                ("correlation_network", PlotHelper.plot_correlation_network)
            )

        for analysis_type, plotter in plotters:
            self.hooks.register_plotter(analysis_type, plotter)

    def _log(self, message: str):
        if self.verbose:
            print(f"🔍 {message}")

    @lru_cache(maxsize=32)
    def _get_clean_numeric_data(self) -> pd.DataFrame:
        """Get cached clean numeric data"""
        return self.df[self._numeric_cols].dropna()

    # ========================================================================
    # PUBLIC API (Clean and comprehensive)
    # ========================================================================
    def analyze(self, analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run specified analyses in parallel with pre/post hooks"""
        analysis_types = analysis_types or list(self._strategies.keys())
        results = {}

        tasks = []
        with ThreadPoolExecutor() as executor:
            for analysis_type in analysis_types:
                if analysis_type not in self._strategies:
                    continue

                self._log(f"Running {analysis_type} analysis...")
                context = {
                    "data": self.df,
                    "config": self.config,
                    "type": analysis_type,
                }
                self.hooks.trigger(f"pre_{analysis_type}", context)

                strategy = self._strategies[analysis_type]
                tasks.append(executor.submit(_run_analysis_worker, analysis_type, strategy, self.df, self.config))

            for future in as_completed(tasks):
                strategy_name, result, error = future.result()
                if error:
                    print(f"❌ {strategy_name} analysis failed:\n{error}")
                    continue

                results[strategy_name] = result
                self._results_cache[strategy_name] = result
                self.hooks.trigger(f"post_{strategy_name}", {
                    "data": self.df,
                    "config": self.config,
                    "type": strategy_name,
                    "result": result,
                })

        return results


    def plot(self, analysis_types: Optional[List[str]] = None):
        """Generate comprehensive plots for analysis results"""
        analysis_types = analysis_types or list(self._results_cache.keys())

        for analysis_type in analysis_types:
            if analysis_type in self._results_cache:
                self._log(f"Plotting {analysis_type}...")
                self.hooks.plot(
                    analysis_type,
                    self._results_cache[analysis_type],
                    self.df,
                    self.config,
                )

    def analyze_and_plot(self, analysis_types: Optional[List[str]] = None):
        """Run comprehensive analysis and generate all plots"""
        results = self.analyze(analysis_types)
        self.plot(analysis_types)
        return results

    def analyze_everything(self):
        """Run the complete comprehensive analysis workflow"""
        self._log("🚀 Starting comprehensive dataset analysis...")

        # Run all analyses
        results = self.analyze()

        # Generate comprehensive insights report
        self.print_detailed_insights()

        # Generate all plots
        self.plot()

        # Special network plot if available
        if "correlations" in results and OPTIONAL_IMPORTS["networkx"]:
            try:
                self.plot_correlation_network()
            except Exception as e:
                print(f"Network plot failed: {e}")

        self._log("🎉 Comprehensive analysis complete!")
        return results

    # ========================================================================
    # SPECIALIZED METHODS (Preserved from original)
    # ========================================================================

    def decompose_series(self, column: str, period: Optional[int] = None):
        """STL decomposition for time series"""
        if column not in self._numeric_cols:
            raise ValueError(f"Column {column} not found in numeric columns")

        series = self.df[column].dropna()
        if len(series) < 24:
            raise ValueError("Series too short for decomposition (minimum 24 points)")

        if period is None:
            period = min(max(2, len(series) // 10), 24)

        stl = STL(series, period=period, seasonal=7)
        decomposition = stl.fit()

        # Enhanced plotting
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        decomposition.observed.plot(ax=axes[0], title=f"Original Series: {column}")
        decomposition.trend.plot(ax=axes[1], title="Trend", color="orange")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonal", color="green")
        decomposition.resid.plot(ax=axes[3], title="Residual", color="red")

        plt.tight_layout()
        plt.show()

        return decomposition

    def plot_autocorrelations(self, column: str, lags: int = 40):
        """Enhanced ACF and PACF plots"""
        if column not in self._numeric_cols:
            raise ValueError(f"Column {column} not found in numeric columns")

        series = self.df[column].dropna()
        if len(series) < lags + 10:
            lags = max(10, len(series) // 3)

        # Compute ACF and PACF
        acf_vals = acf(series, nlags=lags, fft=True)
        pacf_vals = pacf(series, nlags=lags, method="ols")

        # Confidence intervals
        n = len(series)
        ci = 1.96 / np.sqrt(n)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # ACF plot
        ax1.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax1.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
        ax1.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
        ax1.fill_between(range(len(acf_vals)), -ci, ci, alpha=0.2, color="red")
        ax1.set_title(f"Autocorrelation Function - {column}")
        ax1.set_xlabel("Lags")
        ax1.set_ylabel("ACF")

        # PACF plot
        ax2.stem(range(len(pacf_vals)), pacf_vals, basefmt=" ")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
        ax2.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
        ax2.fill_between(range(len(pacf_vals)), -ci, ci, alpha=0.2, color="red")
        ax2.set_title(f"Partial Autocorrelation Function - {column}")
        ax2.set_xlabel("Lags")
        ax2.set_ylabel("PACF")

        plt.tight_layout()
        plt.show()

    def plot_correlation_network(
        self, method: str = "pearson", threshold: Optional[float] = None
    ):
        """Create correlation network graph"""
        if "correlations" not in self._results_cache:
            self._log("Running correlation analysis first...")
            self.analyze(["correlations"])

        correlations = self._results_cache["correlations"]
        PlotHelper.plot_correlation_network(correlations, self.df, self.config)

    @requires_library("shap")
    def explain_features(
        self, model, X: Optional[pd.DataFrame] = None, max_display: int = 10
    ):
        """SHAP feature explanation"""
        import shap

        X = X or self._get_clean_numeric_data()
        if X.empty:
            self._log("No data available for SHAP explanation.")
            return {}

        try:
            if hasattr(model, "predict_proba") and "Tree" in model.__class__.__name__:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                self._log("Using KernelExplainer (model-agnostic).")
                f = model.predict if hasattr(model, "predict") else model.predict_proba
                background = X.sample(
                    min(100, len(X)), random_state=self.config.random_state
                )
                explainer = shap.KernelExplainer(f, background)
                shap_values = explainer.shap_values(X.sample(min(500, len(X))))

        except Exception as e:
            self._log(f"SHAP explanation failed: {e}")
            return {}

        # Generate summary plot
        plt.figure(figsize=(10, 5))
        try:
            shap.summary_plot(shap_values, X, show=False, max_display=max_display)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
        except Exception as e:
            self._log(f"Failed to plot SHAP summary: {e}")

        return {
            "shap_values": shap_values,
            "expected_value": getattr(explainer, "expected_value", None),
        }

    # ========================================================================
    # COMPREHENSIVE INSIGHTS AND REPORTING
    # ========================================================================

    def print_detailed_insights(self):
        """Print comprehensive detailed insights with specific recommendations"""
        print("=" * 100)
        print("🔬 DATASET ANALYSIS")
        print("=" * 100)

        # Dataset Overview
        print(f"\n📊 DATASET OVERVIEW")
        print("-" * 40)
        print(f"   Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"   Numeric features: {len(self._numeric_cols)}")
        print(f"   Categorical features: {len(self._categorical_cols)}")
        print(
            f"   Memory usage: {self.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
        )

        missing_pct = (
            self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])
        ) * 100
        print(f"   Missing values: {missing_pct:.2f}% of total dataset")

        # Run analyses if not already cached
        required_analyses = ["distributions", "correlations", "outliers", "patterns"]
        for analysis in required_analyses:
            if analysis not in self._results_cache:
                try:
                    result = self._strategies[analysis].analyze(self.df, self.config)
                    self._results_cache[analysis] = result
                except Exception as e:
                    print(f"Failed to run {analysis}: {e}")

        # Distribution Analysis Insights
        if "distributions" in self._results_cache:
            dist_summary = self._results_cache["distributions"].get(
                "summary", pd.DataFrame()
            )
            if not dist_summary.empty:
                print(f"\n📈 DETAILED DISTRIBUTION ANALYSIS")
                print("-" * 40)

                gaussian_count = dist_summary["is_gaussian"].sum()
                skewed_count = dist_summary["is_skewed"].sum()
                heavy_tailed_count = dist_summary["is_heavy_tailed"].sum()

                print(
                    f"   ✓ Gaussian features: {gaussian_count}/{len(dist_summary)} ({gaussian_count / len(dist_summary) * 100:.1f}%)"
                )
                if gaussian_count > 0:
                    gaussian_features = dist_summary[dist_summary["is_gaussian"]][
                        "feature"
                    ].tolist()
                    print(f"     → {', '.join(gaussian_features[:5])}")

                print(
                    f"   ⚠️  Skewed features: {skewed_count}/{len(dist_summary)} ({skewed_count / len(dist_summary) * 100:.1f}%)"
                )
                if skewed_count > 0:
                    skewed_features = dist_summary[dist_summary["is_skewed"]][
                        "feature"
                    ].tolist()
                    print(f"     → {', '.join(skewed_features[:5])}")

                print(
                    f"   📏 Heavy-tailed features: {heavy_tailed_count}/{len(dist_summary)} ({heavy_tailed_count / len(dist_summary) * 100:.1f}%)"
                )

        # Correlation Insights
        if "correlations" in self._results_cache:
            correlations = self._results_cache["correlations"]
            if "pearson" in correlations:
                print(f"\n🔗 DETAILED CORRELATION ANALYSIS")
                print("-" * 40)

                corr_matrix = correlations["pearson"]
                strong_pos_pairs = []
                strong_neg_pairs = []

                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]

                        if corr_val > 0.7:
                            strong_pos_pairs.append((feat1, feat2, corr_val))
                        elif corr_val < -0.7:
                            strong_neg_pairs.append((feat1, feat2, corr_val))

                if strong_pos_pairs:
                    print(f"   💚 STRONG POSITIVE CORRELATIONS (r > 0.7):")
                    for feat1, feat2, corr in strong_pos_pairs[:5]:
                        print(f"     → {feat1} ↔ {feat2}: {corr:.3f}")

                if strong_neg_pairs:
                    print(f"   💔 STRONG NEGATIVE CORRELATIONS (r < -0.7):")
                    for feat1, feat2, corr in strong_neg_pairs[:5]:
                        print(f"     → {feat1} ↔ {feat2}: {corr:.3f}")
        # Pattern Detection Results
        if "patterns" in self._results_cache:
            patterns = self._results_cache["patterns"]
            print(f"\n🔍 ADVANCED PATTERN DETECTION")
            print("-" * 40)

            # Feature types
            if "feature_types" in patterns:
                feature_types = patterns["feature_types"]
                print("   📋 FEATURE TYPE CLASSIFICATION:")
                for ftype, features in feature_types.items():
                    if features:
                        type_name = ftype.replace("_", " ").title()
                        print(f"     → {type_name}: {len(features)} features")
                        print(f"       ∘ Top: {', '.join(features[:5])}")

                if feature_types.get("seasonal"):
                    print(f"\n   ⏳ SEASONAL FEATURES:")
                    print(f"     → {', '.join(feature_types['seasonal'][:5])}")

                if feature_types.get("transformable_to_normal"):
                    print(f"\n   🔄 TRANSFORMABLE TO GAUSSIAN:")
                    print(f"     → {', '.join(feature_types['transformable_to_normal'][:5])}")

                if feature_types.get("bimodal") or feature_types.get("mixture_model"):
                    print(f"\n   🔀 MIXTURE DISTRIBUTIONS:")
                    if feature_types.get("bimodal"):
                        print(f"     → Bimodal: {', '.join(feature_types['bimodal'][:5])}")
                    if feature_types.get("mixture_model"):
                        print(f"     → Mixture Model: {', '.join(feature_types['mixture_model'][:5])}")

            # Relationships
            if "relationships" in patterns:
                rel_patterns = patterns["relationships"]

                if rel_patterns.get("nonlinear"):
                    print(f"\n   🌀 NON-LINEAR RELATIONSHIPS DETECTED:")
                    for rel in rel_patterns["nonlinear"][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(f"       ∘ Nonlinearity score: {rel['nonlinearity_score']:.3f}")

                if rel_patterns.get("complex"):
                    print(f"\n   🧬 COMPLEX RELATIONSHIPS (High MI, Low Linear):")
                    for rel in rel_patterns["complex"][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(f"       ∘ Mutual Info: {rel['mutual_info']:.3f}")

                if rel_patterns.get("distance_corr"):
                    print(f"\n   📐 DISTANCE CORRELATION PATTERNS:")
                    for rel in rel_patterns["distance_corr"][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(f"       ∘ DCor: {rel['distance_corr']:.3f}, Pearson: {rel['pearson']:.3f}")

                if rel_patterns.get("tail_dependence"):
                    print(f"\n   🧪 TAIL DEPENDENCE DETECTED:")
                    for rel in rel_patterns["tail_dependence"][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(f"       ∘ Upper Tail Dependency: {rel['upper_tail_dep']:.2%}")

            # Best-fit distributions
            if "distributions" in patterns and patterns["distributions"]:
                print(f"\n   📊 BEST-FIT DISTRIBUTIONS:")
                for feature, fit_info in list(patterns["distributions"].items())[:5]:
                    print(f"     → {feature}: {fit_info['distribution'].title()}")
                    print(f"       ∘ AIC: {fit_info['aic']:.2f}, KS p-value: {fit_info['ks_pvalue']:.3f}")
                    if "_detailed" in fit_info and fit_info["_detailed"].get("alternatives"):
                        alt_names = [alt["distribution"] for alt in fit_info["_detailed"]["alternatives"]]
                        print(f"       ∘ Alt fits: {', '.join(alt_names)}")

        # Feature Engineering Suggestions
        if "feature_engineering" in self._results_cache:
            suggestions = self._results_cache["feature_engineering"]
            details = suggestions.get("_detailed", {})
            print(f"\n🛠️  FEATURE ENGINEERING RECOMMENDATIONS")
            print("-" * 45)

            # Transformations
            feature_suggestions = {
                k: v for k, v in suggestions.items() if k != "interactions" and not k.startswith("_")
            }
            if feature_suggestions:
                print("   🔄 RECOMMENDED TRANSFORMATIONS:")
                for feature, transforms in list(feature_suggestions.items())[:5]:
                    print(f"     → {feature}")
                    stats_info = details.get("transformations", {}).get(feature, {}).get("stats", {})
                    if stats_info:
                        print(f"       ∘ Skew: {stats_info.get('skewness', 0):.2f} | "
                              f"Kurtosis: {stats_info.get('kurtosis', 0):.2f} | "
                              f"Outliers: {stats_info.get('outliers_pct', 0):.1f}%")
                    for i, transform in enumerate(transforms[:3], 1):
                        print(f"       {i}. {transform}")

            # Feature Importance
            if "feature_ranking" in details:
                ranking = details["feature_ranking"]
                if ranking:
                    print("\n   🏆 TOP RANKED FEATURES:")
                    for name, score in list(ranking.items())[:5]:
                        if name != "shap_importance":
                            print(f"     → {name}: importance = {score:.4f}")
                    if "shap_importance" in ranking:
                        print("     ∘ SHAP importance available.")

            # Interactions
            if "interactions" in suggestions:
                interactions = suggestions["interactions"]
                print(f"\n   🤝 INTERACTION FEATURES: {len(interactions)} suggested")
                for i, interaction in enumerate(interactions[:10], 1):
                    print(f"     {i}. {interaction}")
                if len(interactions) > 10:
                    print(f"     ...and {len(interactions)-10} more")

                    
        # Time Series Insights (if applicable)
        if self.time_col and "timeseries" in self._results_cache:
            ts_results = self._results_cache["timeseries"]
            print(f"\n⏰ TIME SERIES ANALYSIS")
            print("-" * 50)

            # STATIONARITY
            if "stationarity" in ts_results and not ts_results["stationarity"].empty:
                stationarity = ts_results["stationarity"]
                if "is_stationary" in stationarity.columns:
                    stationary_mask = stationarity["is_stationary"]
                elif "consensus_stationary" in stationarity.columns:
                    stationary_mask = stationarity["consensus_stationary"]
                else:
                    stationary_mask = stationarity["is_stationary_adf"] & stationarity["is_stationary_kpss"]

                stationary_count = stationary_mask.sum()
                total_features = len(stationarity)

                print(f"📈 Stationarity:")
                print(f"   → {stationary_count}/{total_features} features are stationary")

                # Breakdown by type
                if "stationarity_type" in stationarity.columns:
                    type_counts = stationarity["stationarity_type"].value_counts()
                    for typ, count in type_counts.items():
                        print(f"     ∘ {typ.replace('_', ' ').capitalize()}: {count}")

                # Highlight non-stationary features
                non_stationary = stationarity[~stationary_mask]
                if not non_stationary.empty:
                    print(f"\n   ⚠️ Features that may require differencing or transformation:")
                    for _, row in non_stationary.iterrows():
                        print(
                            f"     - {row['feature']} (ADF p={row.get('adf_pvalue', np.nan):.4f}, "
                            f"KPSS p={row.get('kpss_pvalue', np.nan):.4f})"
                        )

            # LAG SUGGESTIONS
            if "lag_suggestions" in ts_results and ts_results["lag_suggestions"]:
                print(f"\n🔁 Lag Feature Suggestions:")
                for feature, suggestion in list(ts_results["lag_suggestions"].items())[:5]:
                    rec_lags = suggestion.get("recommended_lags", [])
                    seasonal = suggestion.get("seasonal_lags", [])
                    method = suggestion.get("lag_selection_method", "unknown")
                    print(
                        f"   → {feature}: Lags {rec_lags} "
                        f"{'(seasonal: ' + str(seasonal) + ')' if seasonal else ''} "
                        f"[{method}]"
                    )

            if "change_point_detection" in ts_results and len(ts_results["change_point_detection"]) > 0:
                print(f"\n🪓 Change Point Detection:")
                for col, methods in ts_results["change_point_detection"].items():
                    print(f"   → {col}:")
                    for method, result in methods.items():
                        if method == "cusum" and result.get("significant"):
                            ts = result["change_point"]
                            print(f"     ∘ CUSUM break at {ts} (Δ={result['magnitude']:.3f})")
                        if method == "page_hinkley":
                            for cp in result["change_points"][:2]:
                                print(f"     ∘ Page-Hinkley shift at {cp['timestamp']}")

            if "regime_switching" in ts_results and len(ts_results["regime_switching"]) > 0:
                print(f"\n🔁 Regime Switching:")
                for col, result in ts_results["regime_switching"].items():
                    rs = result.get("markov_switching")
                    if rs:
                        print(f"   → {col}: {rs['n_regimes']} regimes detected (current: {rs['current_regime']})")
                    vs = result.get("volatility_switching")
                    if vs:
                        print(f"     ∘ Volatility regime: {vs['current_regime']} (switches: {vs['regime_switches']})")

            if "forecasting_readiness" in ts_results and len(ts_results["forecasting_readiness"]) > 0:
                print(f"\n📈 Forecasting Readiness:")
                fr_scores = ts_results["forecasting_readiness"]
                sorted_fr = sorted(fr_scores.items(), key=lambda x: -x[1]["overall_score"])
                for name, val in sorted_fr[:3]:
                    print(f"   → {name}: {val['readiness_level']} (score={val['overall_score']:.2f})")
                    if val["recommendations"]:
                        print(f"     ∘ Suggestions: {', '.join(val['recommendations'])}")

            if "causality_analysis" in ts_results and len(ts_results["causality_analysis"]) > 0:
                print(f"\n📣 Granger Causality:")
                for pair, result in ts_results["causality_analysis"].items():
                    if result["var1_causes_var2"]["significant"]:
                        print(f"   → {pair}: {pair.split('_vs_')[0]} causes {pair.split('_vs_')[1]} (p={result['var1_causes_var2']['p_value']:.4f})")
                    if result["var2_causes_var1"]["significant"]:
                        print(f"   → {pair}: {pair.split('_vs_')[1]} causes {pair.split('_vs_')[0]} (p={result['var2_causes_var1']['p_value']:.4f})")
        
        if "clusters" in self._results_cache:
            cluster_data = self._results_cache["clusters"]
            print(f"\n🧩 CLUSTERING INSIGHTS")
            print("-" * 40)

            if not cluster_data or 'error' in cluster_data:
                error_msg = cluster_data.get('error', 'Unknown error') if isinstance(cluster_data, dict) else "No clustering results available"
                print(f"   ⚠️  {error_msg}")
            else:
                # Summary statistics
                if 'summary' in cluster_data:
                    summary = cluster_data['summary']
                    print(f"   📊 ANALYSIS SUMMARY:")
                    print(f"     → Methods attempted: {summary.get('methods_attempted', 'N/A')}")
                    print(f"     → Successful methods: {summary.get('successful_methods', 'N/A')}")
                    if summary.get('best_method'):
                        print(f"     ⭐ Best method: {summary['best_method'].upper()}")

                # Data characteristics
                if 'data_characteristics' in cluster_data:
                    chars = cluster_data['data_characteristics']
                    print(f"\n   📈 DATA CHARACTERISTICS:")
                    print(f"     → Samples: {chars.get('n_samples', 'N/A'):,}")
                    print(f"     → Features: {chars.get('n_features', 'N/A'):,}")
                    
                    # Data quality indicators
                    variance = chars.get('data_variance', 0)
                    spread = chars.get('data_spread', 0)
                    if variance > 0:
                        print(f"     → Data variance: {variance:.3f}")
                    if spread > 0:
                        print(f"     → Data range: {spread:.3f}")

                # Optimal k analysis
                if 'optimal_k_analysis' in cluster_data:
                    k_info = cluster_data['optimal_k_analysis']
                    print(f"\n   🎯 OPTIMAL CLUSTERS ANALYSIS:")
                    print(f"     → Estimated optimal k: {k_info.get('optimal_k', 'N/A')}")
                    print(f"     → Confidence level: {k_info.get('confidence', 'N/A').title()}")
                    
                    if 'methods' in k_info and k_info['methods']:
                        print(f"     → Method agreement: {k_info.get('method_agreement', 'N/A')} different values")
                        method_results = k_info['methods']
                        print(f"     → Individual estimates:")
                        for method, k_val in method_results.items():
                            print(f"        ∘ {method.title()}: {k_val}")

                # Preprocessing information
                if 'preprocessing_info' in cluster_data:
                    prep = cluster_data['preprocessing_info']
                    print(f"\n   🔧 PREPROCESSING APPLIED:")
                    print(f"     → Scaling method: {prep.get('scaling_method', 'unknown').replace('_', ' ').title()}")
                    
                    outlier_pct = prep.get('outlier_percentage', 0)
                    if outlier_pct > 0:
                        print(f"     → Outliers detected: {prep.get('outliers_detected', 0)} ({outlier_pct:.1f}%)")
                    
                    if prep.get('pca_applied', False):
                        var_exp = prep.get('pca_variance_explained', 0)
                        final_dims = prep.get('final_dimensions', 'N/A')
                        print(f"     → PCA dimensionality reduction: {final_dims} components ({var_exp:.1%} variance)")
                    
                    if prep.get('curse_of_dimensionality_risk', False):
                        print(f"     ⚠️ High dimensionality risk detected")

                # Individual clustering results
                if 'clustering_results' in cluster_data:
                    clustering_results = cluster_data['clustering_results']
                    evaluations = cluster_data.get('evaluations', {})
                    
                    print(f"\n   🔍 CLUSTERING METHODS RESULTS:")
                    
                    # Group methods by type
                    method_groups = {
                        'Centroid-based': [],
                        'Hierarchical': [],
                        'Density-based': [],
                        'Probabilistic': [],
                        'Spectral & Others': [],
                        'Ensemble': []
                    }
                    
                    for method_name, result in clustering_results.items():
                        method_type = result.get('method_type', 'unknown')
                        if 'centroid' in method_type or method_name == 'kmeans':
                            method_groups['Centroid-based'].append((method_name, result))
                        elif 'hierarchical' in method_type:
                            method_groups['Hierarchical'].append((method_name, result))
                        elif 'density' in method_type:
                            method_groups['Density-based'].append((method_name, result))
                        elif 'probabilistic' in method_type or 'bayesian' in method_type:
                            method_groups['Probabilistic'].append((method_name, result))
                        elif method_type == 'ensemble':
                            method_groups['Ensemble'].append((method_name, result))
                        else:
                            method_groups['Spectral & Others'].append((method_name, result))
                    
                    # Display results by group
                    for group_name, methods in method_groups.items():
                        if not methods:
                            continue
                            
                        print(f"\n     📂 {group_name}:")
                        for method_name, result in methods:
                            print(f"       🔹 {method_name.upper()}:")
                            
                            # Basic cluster information
                            n_clusters = result.get('n_clusters', result.get('best_k', 0))
                            if n_clusters > 0:
                                print(f"         → Clusters found: {n_clusters}")
                            
                            # Noise points for density-based methods
                            if 'noise_points' in result:
                                noise_count = result['noise_points']
                                noise_pct = evaluations.get(method_name, {}).get('noise_ratio', 0) * 100
                                print(f"         → Noise points: {noise_count} ({noise_pct:.1f}%)")
                            
                            # Quality scores
                            eval_data = evaluations.get(method_name, {})
                            if 'silhouette_score' in eval_data:
                                sil_score = eval_data['silhouette_score']
                                if sil_score > 0.7:
                                    quality_indicator = "Excellent 🟢"
                                elif sil_score > 0.5:
                                    quality_indicator = "Good 🟡"
                                elif sil_score > 0.25:
                                    quality_indicator = "Fair 🟠"
                                else:
                                    quality_indicator = "Poor 🔴"
                                print(f"         → Silhouette: {sil_score:.3f} ({quality_indicator})")
                            
                            # Additional quality metrics
                            if 'cluster_balance' in eval_data:
                                balance = eval_data['cluster_balance']
                                print(f"         → Cluster balance: {balance:.3f}")
                            
                            if 'separation_ratio' in eval_data:
                                separation = eval_data['separation_ratio']
                                print(f"         → Separation ratio: {separation:.2f}")
                            
                            # Method-specific information
                            if method_name == 'kmeans' and 'stability_score' in eval_data:
                                stability = eval_data['stability_score']
                                print(f"         → Stability: {stability:.3f}")
                            
                            if method_name == 'hierarchical' and 'best_linkage_method' in result:
                                linkage_method = result['best_linkage_method']
                                print(f"         → Best linkage: {linkage_method}")
                            
                            if 'gmm' in method_name and 'model_comparison' in result:
                                best_bic = result.get('best_bic', 0)
                                cov_type = result.get('covariance_type', 'unknown')
                                print(f"         → Best BIC: {best_bic:.2f} ({cov_type} covariance)")
                            
                            if 'bayesian' in method_name and 'effective_components' in result:
                                eff_comp = result['effective_components']
                                print(f"         → Effective components: {eff_comp}")
                            
                            if method_name == 'ensemble' and 'participating_methods' in result:
                                n_methods = len(result['participating_methods'])
                                print(f"         → Consensus from {n_methods} methods")
                            
                            # Cluster size distribution
                            if 'cluster_sizes' in result and result['cluster_sizes']:
                                sizes = result['cluster_sizes']
                                if isinstance(sizes, dict):
                                    total_points = sum(sizes.values())
                                    largest_cluster = max(sizes.values())
                                    smallest_cluster = min(sizes.values())
                                    print(f"         → Size range: {smallest_cluster}-{largest_cluster} points")
                                    
                                    # Show distribution for small number of clusters
                                    if len(sizes) <= 5:
                                        size_str = ", ".join([f"C{k}:{v}" for k, v in sorted(sizes.items())])
                                        print(f"         → Distribution: {size_str}")
                            
                            # Uncertainty for probabilistic methods
                            if 'mean_assignment_entropy' in eval_data:
                                entropy = eval_data['mean_assignment_entropy']
                                uncertainty = eval_data.get('assignment_uncertainty', 0)
                                print(f"         → Assignment entropy: {entropy:.3f} ±{uncertainty:.3f}")

                # Top recommendations
                if 'recommendations' in cluster_data and cluster_data['recommendations']:
                    print(f"\n   💡 KEY RECOMMENDATIONS:")
                    for i, rec in enumerate(cluster_data['recommendations'][:4], 1):  # Show top 4
                        # Clean up recommendation formatting
                        clean_rec = rec.replace('🏆', '').replace('⚠️', '').replace('✅', '').replace('📊', '').replace('🌳', '').replace('🤝', '').strip()
                        if clean_rec.startswith('Best method:'):
                            print(f"     {i}. 🏆 {clean_rec}")
                        elif 'noise' in clean_rec.lower() or 'uncertainty' in clean_rec.lower():
                            print(f"     {i}. ⚠️ {clean_rec}")
                        elif 'evidence' in clean_rec.lower() or 'confident' in clean_rec.lower():
                            print(f"     {i}. ✅ {clean_rec}")
                        else:
                            print(f"     {i}. 💭 {clean_rec}")

                # Performance summary
                if 'evaluations' in cluster_data:
                    successful_methods = len([e for e in cluster_data['evaluations'].values() if 'silhouette_score' in e])
                    total_methods = len(cluster_data.get('clustering_results', {}))
                    
                    # Calculate average silhouette across all methods
                    silhouette_scores = [e['silhouette_score'] for e in cluster_data['evaluations'].values() 
                                       if 'silhouette_score' in e and e['silhouette_score'] > 0]
                    
                    print(f"\n   📊 PERFORMANCE SUMMARY:")
                    print(f"     → Methods evaluated: {successful_methods}/{total_methods}")
                    
                    if silhouette_scores:
                        avg_sil = np.mean(silhouette_scores)
                        best_sil = max(silhouette_scores)
                        print(f"     → Average silhouette: {avg_sil:.3f}")
                        print(f"     → Best silhouette: {best_sil:.3f}")
                        
                        # Overall clustering quality assessment
                        if best_sil > 0.7:
                            overall_quality = "Excellent clustering structure detected 🎯"
                        elif best_sil > 0.5:
                            overall_quality = "Good clustering structure found 👍"
                        elif best_sil > 0.25:
                            overall_quality = "Moderate clustering structure present 🤔"
                        else:
                            overall_quality = "Weak clustering structure detected 😐"
                        
                        print(f"     → Overall assessment: {overall_quality}")

        # Outlier Insights
        if "outliers" in self._results_cache:
            outlier_data = self._results_cache["outliers"]
            print(f"\n🚨 OUTLIER DETECTION SUMMARY")
            print("-" * 40)

            if not outlier_data or 'error' in outlier_data:
                error_msg = outlier_data.get('error', 'Unknown error') if isinstance(outlier_data, dict) else "No outlier results available"
                print(f"   ⚠️  {error_msg}")
            else:
                # Analysis summary
                if 'summary' in outlier_data:
                    summary = outlier_data['summary']
                    print(f"   📊 ANALYSIS OVERVIEW:")
                    print(f"     → Methods attempted: {summary.get('methods_attempted', 'N/A')}")
                    print(f"     → Successful methods: {summary.get('successful_methods', 'N/A')}")
                    print(f"     → Overall outlier rate: {summary.get('overall_outlier_rate', 0):.1%}")
                    if summary.get('best_method'):
                        print(f"     ⭐ Best method: {summary['best_method'].upper()}")

                # Data characteristics
                if 'data_characteristics' in outlier_data:
                    chars = outlier_data['data_characteristics']
                    print(f"\n   📈 DATA CHARACTERISTICS:")
                    print(f"     → Total samples: {chars.get('total_samples', 'N/A'):,}")
                    print(f"     → Analyzed samples: {chars.get('analyzed_samples', 'N/A'):,}")
                    
                    missing_samples = chars.get('missing_samples', 0)
                    if missing_samples > 0:
                        missing_pct = missing_samples / chars.get('total_samples', 1) * 100
                        print(f"     → Missing data: {missing_samples:,} samples ({missing_pct:.1f}%)")
                    
                    print(f"     → Features analyzed: {chars.get('n_features', 'N/A')}")
                    print(f"     → Target contamination: {chars.get('contamination_rate', 0):.1%}")

                # Preprocessing information
                if 'preprocessing_info' in outlier_data:
                    prep = outlier_data['preprocessing_info']
                    print(f"\n   🔧 PREPROCESSING APPLIED:")
                    print(f"     → Scaling method: {prep.get('scaling_method', 'unknown').replace('_', ' ').title()}")
                    print(f"     → Missing data strategy: {prep.get('handling_strategy', 'unknown').replace('_', ' ').title()}")
                    
                    skewness = prep.get('data_skewness', 0)
                    if skewness > 0:
                        if skewness > 3:
                            skew_desc = "Highly skewed 🔴"
                        elif skewness > 1.5:
                            skew_desc = "Moderately skewed 🟡"
                        else:
                            skew_desc = "Low skewness 🟢"
                        print(f"     → Data skewness: {skewness:.2f} ({skew_desc})")
                    
                    cond_num = prep.get('condition_number')
                    if cond_num:
                        if cond_num < 100:
                            cond_desc = "Well-conditioned 🟢"
                        elif cond_num < 1000:
                            cond_desc = "Acceptable 🟡"
                        else:
                            cond_desc = "Poorly conditioned 🔴"
                        print(f"     → Data condition: {cond_num:.1e} ({cond_desc})")

                # Individual method results
                if 'outlier_results' in outlier_data:
                    outlier_results = outlier_data['outlier_results']
                    evaluations = outlier_data.get('evaluations', {})
                    
                    print(f"\n   🔍 DETECTION METHODS RESULTS:")
                    
                    # Group methods by type
                    method_groups = {
                        'Statistical': [],
                        'Distance-based': [],
                        'Machine Learning': [],
                        'Advanced': [],
                        'Ensemble': []
                    }
                    
                    for method_name, result in outlier_results.items():
                        method_type = result.get('method_type', 'unknown')
                        if method_type == 'statistical':
                            method_groups['Statistical'].append((method_name, result))
                        elif 'distance' in method_type or 'density' in method_type:
                            method_groups['Distance-based'].append((method_name, result))
                        elif method_type in ['ensemble', 'boundary_based', 'covariance_based']:
                            method_groups['Machine Learning'].append((method_name, result))
                        elif method_type in ['histogram_based', 'cluster_based']:
                            method_groups['Advanced'].append((method_name, result))
                        elif method_type == 'ensemble':
                            method_groups['Ensemble'].append((method_name, result))
                        else:
                            method_groups['Advanced'].append((method_name, result))
                    
                    # Track for consensus calculation
                    consensus_votes = np.zeros(outlier_data['data_characteristics'].get('total_samples', 0))
                    valid_methods = 0
                    
                    # Display results by group
                    for group_name, methods in method_groups.items():
                        if not methods:
                            continue
                            
                        print(f"\n     📂 {group_name}:")
                        for method_name, result in methods:
                            print(f"       🔹 {method_name.replace('_', ' ').title()}:")
                            
                            # Basic detection statistics
                            count = result.get('count', 0)
                            pct = result.get('percentage', 0)
                            print(f"         → Outliers found: {count:,} ({pct:.2f}%)")
                            
                            # Add to consensus
                            if 'outliers' in result:
                                consensus_votes += result['outliers'].astype(int)
                                valid_methods += 1
                            
                            # Quality metrics
                            eval_data = evaluations.get(method_name, {})
                            if 'score_separation' in eval_data:
                                separation = eval_data['score_separation']
                                if separation > 1.0:
                                    sep_desc = "Excellent 🟢"
                                elif separation > 0.5:
                                    sep_desc = "Good 🟡"
                                else:
                                    sep_desc = "Moderate 🟠"
                                print(f"         → Separation quality: {separation:.3f} ({sep_desc})")
                            
                            if 'isolation_score' in eval_data:
                                isolation = eval_data['isolation_score']
                                print(f"         → Isolation score: {isolation:.3f}")
                            
                            if 'score_overlap' in eval_data:
                                overlap = eval_data['score_overlap']
                                print(f"         → Score overlap: {overlap:.3f}")
                            
                            # Method-specific details
                            if method_name == 'knn_distance' and 'k' in result:
                                print(f"         → K neighbors: {result['k']}")
                            
                            if method_name == 'dbscan' and 'eps' in result:
                                print(f"         → Epsilon: {result['eps']:.3f}")
                            
                            if 'one_class_svm' in method_name and 'kernel' in result:
                                print(f"         → Kernel: {result['kernel']}")
                            
                            if method_name == 'ensemble' and 'participating_methods' in result:
                                n_methods = len(result['participating_methods'])
                                print(f"         → Combined {n_methods} methods")
                                
                                # Show consensus strength if available
                                if 'consensus_strength' in result and len(result['consensus_strength']) > 0:
                                    avg_consensus = np.mean(result['consensus_strength'])
                                    if avg_consensus < 0.3:
                                        consensus_desc = "High agreement 🎯"
                                    elif avg_consensus < 0.5:
                                        consensus_desc = "Moderate agreement 👍"
                                    else:
                                        consensus_desc = "Low agreement 🤔"
                                    print(f"         → Method consensus: {avg_consensus:.3f} ({consensus_desc})")

                # Consensus analysis
                if valid_methods >= 2:
                    print(f"\n   🎯 CONSENSUS ANALYSIS:")
                    
                    for threshold in [2, 3, max(2, valid_methods // 2)]:
                        if threshold <= valid_methods:
                            consensus_outliers = (consensus_votes >= threshold).sum()
                            consensus_pct = consensus_outliers / len(consensus_votes) * 100 if len(consensus_votes) > 0 else 0
                            
                            if threshold == 2:
                                confidence_desc = "Moderate confidence"
                            elif threshold == 3:
                                confidence_desc = "High confidence"
                            else:
                                confidence_desc = "Very high confidence"
                                
                            print(f"     → ≥{threshold} methods agree: {consensus_outliers:,} outliers ({consensus_pct:.2f}%) - {confidence_desc}")
                    
                    # Show most reliable outliers
                    max_votes = int(consensus_votes.max()) if len(consensus_votes) > 0 else 0
                    if max_votes > 0:
                        unanimous = (consensus_votes == max_votes).sum()
                        print(f"     → Unanimous detection: {unanimous:,} outliers ({unanimous/len(consensus_votes)*100:.2f}%)")

                # Top recommendations
                if 'recommendations' in outlier_data and outlier_data['recommendations']:
                    print(f"\n   💡 KEY RECOMMENDATIONS:")
                    for i, rec in enumerate(outlier_data['recommendations'][:4], 1):  # Show top 4
                        # Clean up recommendation formatting
                        clean_rec = rec.replace('🏆', '').replace('⚠️', '').replace('✅', '').replace('📊', '').replace('🔍', '').strip()
                        if clean_rec.startswith('Best method:'):
                            print(f"     {i}. 🏆 {clean_rec}")
                        elif 'high' in clean_rec.lower() and ('rate' in clean_rec.lower() or 'outlier' in clean_rec.lower()):
                            print(f"     {i}. ⚠️ {clean_rec}")
                        elif 'excellent' in clean_rec.lower() or 'good' in clean_rec.lower():
                            print(f"     {i}. ✅ {clean_rec}")
                        elif 'skewed' in clean_rec.lower() or 'robust' in clean_rec.lower():
                            print(f"     {i}. 🔧 {clean_rec}")
                        else:
                            print(f"     {i}. 💭 {clean_rec}")

                # Performance summary
                if 'evaluations' in outlier_data:
                    evaluations = outlier_data['evaluations']
                    successful_evals = len([e for e in evaluations.values() if 'error' not in e])
                    
                    # Calculate quality metrics across methods
                    separation_scores = [e.get('score_separation', 0) for e in evaluations.values() 
                                       if 'score_separation' in e and e['score_separation'] > 0]
                    
                    outlier_rates = [r.get('percentage', 0)/100 for r in outlier_results.values()]
                    
                    print(f"\n   📊 DETECTION QUALITY SUMMARY:")
                    print(f"     → Methods evaluated: {successful_evals}/{len(outlier_results)}")
                    
                    if separation_scores:
                        avg_separation = np.mean(separation_scores)
                        best_separation = max(separation_scores)
                        print(f"     → Average separation: {avg_separation:.3f}")
                        print(f"     → Best separation: {best_separation:.3f}")
                    
                    if outlier_rates:
                        rate_std = np.std(outlier_rates)
                        if rate_std < 0.02:
                            consistency_desc = "Very consistent 🎯"
                        elif rate_std < 0.05:
                            consistency_desc = "Consistent 👍"
                        else:
                            consistency_desc = "Variable 📊"
                        print(f"     → Rate consistency: σ={rate_std:.3f} ({consistency_desc})")
                    
                    # Overall assessment
                    if separation_scores and max(separation_scores) > 1.0:
                        overall_quality = "Excellent outlier detection quality 🌟"
                    elif separation_scores and max(separation_scores) > 0.5:
                        overall_quality = "Good outlier detection quality 👍"
                    elif successful_evals >= len(outlier_results) * 0.7:
                        overall_quality = "Satisfactory detection coverage 📊"
                    else:
                        overall_quality = "Limited detection reliability ⚠️"
                    
                    print(f"     → Overall assessment: {overall_quality}")
        # Missingness Analysis
        if "missingness" in self._results_cache:
            missing = self._results_cache["missingness"]
            print(f"\n📉 MISSINGNESS ANALYSIS")
            print("-" * 45)

            # Missing rate summary
            if "missing_rate" in missing and not missing["missing_rate"].empty:
                print("   ❗ MISSING VALUE RATES:")
                for col, rate in missing["missing_rate"].items():
                    print(f"     → {col}: {rate:.1%}")

            # MNAR diagnostics
            if "missing_vs_target" in missing and missing["missing_vs_target"]:
                print("\n   🧪 MNAR DEPENDENCE ON TARGET:")
                for col, stats in missing["missing_vs_target"].items():
                    if "ttest_p" in stats:
                        print(f"     → {col}: t-test p = {stats['ttest_p']:.4f}, KS p = {stats['ks_p']:.4f}, AUC = {stats.get('auc', 0):.2f}")
                    elif "chi2_p" in stats:
                        print(f"     → {col}: Chi² p = {stats['chi2_p']:.4f}, "
                              f"Cramér's V = {stats.get('cramers_v', 0):.2f}, "
                              f"MI = {stats.get('mutual_info', 0):.3f}")
                    if stats.get("suggested_mnar"):
                        print("       ∘ ⚠️ Likely MNAR (target-dependent missingness)")

            # Correlated missingness (Pearson)
            if "missingness_correlation" in missing:
                corr = missing["missingness_correlation"]
                upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
                corr_pairs = corr.where(upper).stack().sort_values(ascending=False)
                corr_pairs = corr_pairs[corr_pairs > 0.3]
                if not corr_pairs.empty:
                    print("\n   🔗 CORRELATED MISSINGNESS (Pearson > 0.3):")
                    for (col1, col2), val in corr_pairs[:5].items():
                        print(f"     → {col1} ↔ {col2}: corr = {val:.2f}")

            # Jaccard similarity (optional)
            if "missingness_jaccard" in missing:
                jac = missing["missingness_jaccard"].copy()
                upper = np.triu(np.ones_like(jac, dtype=bool), k=1)
                jac_pairs = jac.where(upper).stack().sort_values(ascending=False)
                jac_pairs = jac_pairs[jac_pairs > 0.1]
                if not jac_pairs.empty:
                    print("\n   🧬 JACCARD SIMILARITY (Missing Co-occurrence > 0.1):")
                    for (col1, col2), val in jac_pairs[:5].items():
                        print(f"     → {col1} ∩ {col2}: Jaccard = {val:.2f}")

            # Clustering results
            if "missingness_clusters" in missing:
                cluster_map = missing["missingness_clusters"]
                print("\n   🧩 MISSINGNESS CLUSTERS:")
                from collections import defaultdict
                cluster_dict = defaultdict(list)
                for col, cluster_id in cluster_map.items():
                    cluster_dict[cluster_id].append(col)
                for cid, members in sorted(cluster_dict.items()):
                    print(f"     • Cluster {cid}: {', '.join(members[:5])}" +
                          (f" (+{len(members) - 5} more)" if len(members) > 5 else ""))
        
        # Dimensionality Reduction Summary
        if "dimensionality" in self._results_cache:
            dimred = self._results_cache["dimensionality"]
            print(f"\n🧮 DIMENSIONALITY REDUCTION")
            print("-" * 40)

            if not dimred or 'error' in dimred:
                error_msg = dimred.get('error', 'Unknown error') if isinstance(dimred, dict) else "No dimensionality results available"
                print(f"   ⚠️  {error_msg}")
            else:
                # Data characteristics summary
                if 'data_characteristics' in dimred:
                    chars = dimred['data_characteristics']
                    print(f"   📊 DATA CHARACTERISTICS:")
                    print(f"     → Samples: {chars.get('n_samples', 'N/A'):,}")
                    print(f"     → Features: {chars.get('n_features', 'N/A'):,}")
                    print(f"     → Effective rank: {chars.get('effective_rank', 'N/A')}")
                    
                    # Data quality indicators
                    cond_num = chars.get('condition_number', 0)
                    if cond_num > 0:
                        if cond_num < 100:
                            quality = "Excellent 🟢"
                        elif cond_num < 1000:
                            quality = "Good 🟡"
                        else:
                            quality = "Poor (ill-conditioned) 🔴"
                        print(f"     → Data quality: {quality} (cond: {cond_num:.1e})")

                # Preprocessing info
                if 'preprocessing_info' in dimred:
                    prep = dimred['preprocessing_info']
                    print(f"\n   🔧 PREPROCESSING APPLIED:")
                    print(f"     → Scaling method: {prep.get('scaling_method', 'unknown').replace('_', ' ').title()}")
                    if prep.get('features_removed', 0) > 0:
                        print(f"     → Features removed (low variance): {prep['features_removed']}")
                    if 'sampling_method' in prep:
                        print(f"     → Sampling: {prep['sampling_method'].title()}")

                # Method results
                if 'embeddings' in dimred:
                    embeddings = dimred['embeddings']
                    print(f"\n   🔍 REDUCTION METHODS SUCCESSFULLY APPLIED:")
                    
                    # Separate linear and nonlinear methods
                    linear_methods = []
                    nonlinear_methods = []
                    
                    for method, result in embeddings.items():
                        if result.get('method_type') == 'linear':
                            linear_methods.append((method, result))
                        else:
                            nonlinear_methods.append((method, result))
                    
                    # Display linear methods
                    if linear_methods:
                        print("     📐 Linear Methods:")
                        for method, result in linear_methods:
                            emb = result['embedding']
                            print(f"       → {method.upper()}: {emb.shape}")
                            
                            # Show explained variance for PCA-like methods
                            if 'total_variance_explained' in result:
                                var_exp = result['total_variance_explained']
                                print(f"         ∘ Variance explained: {var_exp:.1%}")
                            
                            # Show sample values for 2D embeddings
                            if emb.shape[1] == 2 and emb.shape[0] > 0:
                                print(f"         ∘ Sample point: [{emb[0,0]:.2f}, {emb[0,1]:.2f}]")
                            else:
                                print(f"         ∘ Components: {emb.shape[1]}")
                    
                    # Display nonlinear methods
                    if nonlinear_methods:
                        print("     🌀 Nonlinear Methods:")
                        for method, result in nonlinear_methods:
                            emb = result['embedding']
                            print(f"       → {method.upper()}: {emb.shape}")
                            
                            # Show method-specific parameters
                            if 'perplexity' in result:
                                print(f"         ∘ Perplexity: {result['perplexity']}")
                            if 'n_neighbors' in result and result['n_neighbors']:
                                print(f"         ∘ Neighbors: {result['n_neighbors']}")
                            
                            # Show sample values for 2D embeddings
                            if emb.shape[1] == 2 and emb.shape[0] > 0:
                                print(f"         ∘ Sample point: [{emb[0,0]:.2f}, {emb[0,1]:.2f}]")

                # Quality evaluation
                if 'evaluation' in dimred and dimred['evaluation']:
                    print(f"\n   📈 QUALITY EVALUATION:")
                    evaluations = dimred['evaluation']
                    
                    # Find best method based on combined score
                    best_method = None
                    best_score = -float('inf')
                    
                    for method, metrics in evaluations.items():
                        if metrics:  # Only consider methods with evaluation metrics
                            score = 0
                            score_components = []
                            
                            if 'silhouette_score' in metrics:
                                sil_score = metrics['silhouette_score']
                                score += sil_score * 0.4
                                score_components.append(f"silhouette: {sil_score:.3f}")
                            
                            if 'neighborhood_preservation' in metrics:
                                neigh_score = metrics['neighborhood_preservation']
                                score += neigh_score * 0.4
                                score_components.append(f"preservation: {neigh_score:.3f}")
                            
                            if 'embedding_stability' in metrics:
                                stab_score = metrics['embedding_stability']
                                score += stab_score * 0.2
                                score_components.append(f"stability: {stab_score:.3f}")
                            
                            print(f"     → {method.upper()}: {', '.join(score_components)}")
                            
                            if score > best_score:
                                best_score = score
                                best_method = method
                    
                    if best_method:
                        print(f"     ⭐ Best performing: {best_method.upper()}")

                # Recommendations
                if 'recommendations' in dimred and dimred['recommendations']:
                    print(f"\n   💡 RECOMMENDATIONS:")
                    for i, rec in enumerate(dimred['recommendations'][:3], 1):  # Show top 3 recommendations
                        print(f"     {i}. {rec}")

                # Summary statistics
                total_methods = len(dimred.get('embeddings', {}))
                successful_evals = len([m for m in dimred.get('evaluation', {}).values() if m])
                print(f"\n   📝 SUMMARY: {total_methods} methods applied, {successful_evals} evaluated for quality")
                
        # Data Quality Assessment
        print(f"\n✅ DATA QUALITY ASSESSMENT")
        print("-" * 40)

        completeness = (
            1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])
        ) * 100
        uniqueness = len(self.df.drop_duplicates()) / len(self.df) * 100

        print(f"   📊 Completeness: {completeness:.1f}%")
        print(f"   🔍 Uniqueness: {uniqueness:.1f}%")

        if self._categorical_cols:
            print(f"   📋 Categorical Feature Cardinality:")
            for col in self._categorical_cols[:5]:
                unique_count = self.df[col].nunique()
                unique_pct = unique_count / len(self.df) * 100
                print(f"     → {col}: {unique_count} unique values ({unique_pct:.1f}%)")

        overall_quality = (completeness + uniqueness) / 2
        quality_status = (
            "Excellent"
            if overall_quality > 95
            else "Good"
            if overall_quality > 85
            else "Fair"
            if overall_quality > 70
            else "Poor"
        )
        quality_emoji = (
            "🟢"
            if overall_quality > 95
            else "🟡"
            if overall_quality > 85
            else "🟠"
            if overall_quality > 70
            else "🔴"
        )

        print(
            f"   {quality_emoji} Overall Quality Score: {overall_quality:.1f}% ({quality_status})"
        )

    # ========================================================================
    # EXTENSIBILITY METHODS
    # ========================================================================

    def register_strategy(self, strategy: AnalysisStrategy):
        """Register a custom analysis strategy"""
        self._strategies[strategy.name] = strategy

    def register_hook(self, event: str, callback: Callable):
        """Register a custom hook"""
        self.hooks.register_hook(event, callback)

    def register_plotter(self, analysis_type: str, plotter: Callable):
        """Register a custom plotter"""
        self.hooks.register_plotter(analysis_type, plotter)

    def get_results(self, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached results for an analysis type"""
        return self._results_cache.get(analysis_type)

    def get_available_analyses(self) -> List[str]:
        """Get list of available analysis types"""
        return list(self._strategies.keys())
