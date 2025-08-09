import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

from .foreminer_aux import *


class OutlierAnalyzer(AnalysisStrategy):
    """SOTA fast outlier detection with adaptive method selection and ensemble learning"""

    @property
    def name(self) -> str:
        return "outliers"

    def __init__(self):
        # Performance tiers based on data size
        self.fast_threshold = 1000      # Ultra-fast methods only
        self.medium_threshold = 5000    # Fast + some ML methods
        self.large_threshold = 20000    # All methods with subsampling
        self.max_sample_size = 3000     # Maximum for expensive methods

    # ---------------------- Enhanced Preprocessing ----------------------
    def _lightning_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Ultra-fast preprocessing with smart defaults"""
        numeric = data.select_dtypes(include=[np.number])
        
        if numeric.empty:
            raise ValueError("No numeric data available for outlier detection")

        info: Dict[str, Any] = {}
        n_samples, n_features = numeric.shape
        
        # Fast missing data handling
        na_counts = numeric.isna().sum()
        missing_pct = float(na_counts.sum() / (n_samples * n_features) * 100)
        info["missing_data_percentage"] = missing_pct
        
        if missing_pct > 80:
            raise ValueError("Too much missing data for reliable outlier detection")
        
        # Smart missing data strategy
        if missing_pct > 30:
            # Drop columns with >50% missing, median impute rest
            good_cols = na_counts < (n_samples * 0.5)
            numeric = numeric.loc[:, good_cols]
            if numeric.empty:
                raise ValueError("No columns with sufficient data")
            
            # Fast median imputation
            medians = numeric.median()
            clean = numeric.fillna(medians)
            complete_idx = np.arange(len(clean))
            info["handling_strategy"] = "column_filter_and_imputation"
        elif missing_pct > 5:
            # Simple median imputation
            medians = numeric.median()
            clean = numeric.fillna(medians)
            complete_idx = np.arange(len(clean))
            info["handling_strategy"] = "median_imputation"
        else:
            # Drop missing rows (fast)
            clean = numeric.dropna()
            complete_idx = np.where(~numeric.isna().any(axis=1))[0]
            info["handling_strategy"] = "complete_cases_only"

        info["final_sample_size"] = int(len(clean))
        info["final_feature_count"] = int(clean.shape[1])
        
        # Fast data characteristics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skewness = clean.skew().abs()
            kurtosis = clean.kurtosis().abs()
            
        info["data_skewness"] = float(skewness.mean())
        info["data_kurtosis"] = float(kurtosis.mean())
        
        # Smart scaling selection (much faster)
        high_skew = (skewness > 2).any()
        high_kurt = (kurtosis > 10).any()
        
        if high_skew and high_kurt:
            scaler = PowerTransformer(method="yeo-johnson", standardize=True)
            scaler_name = "power_transform"
        elif high_skew:
            scaler = RobustScaler()
            scaler_name = "robust"
        else:
            scaler = StandardScaler()
            scaler_name = "standard"
        
        try:
            X = scaler.fit_transform(clean)
            info["scaling_method"] = scaler_name
        except Exception:
            # Fallback
            scaler = RobustScaler()
            X = scaler.fit_transform(clean)
            info["scaling_method"] = "robust_fallback"
        
        return X, complete_idx, info

    # ---------------------- Ultra-Fast Statistical Methods ----------------------
    def _lightning_statistical(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Vectorized statistical outlier detection"""
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape
        
        # Vectorized Z-score (fastest)
        try:
            z_scores = np.abs(stats.zscore(data, axis=0, nan_policy="omit"))
            max_z = np.max(z_scores, axis=1)
            threshold = 3.0
            outliers = max_z > threshold
            
            results["z_score"] = {
                "outliers": outliers,
                "scores": max_z,
                "threshold": threshold,
                "method_type": "statistical"
            }
        except Exception:
            pass
        
        # Fast Modified Z-score using median
        try:
            medians = np.median(data, axis=0)
            mad = np.median(np.abs(data - medians), axis=0)
            modified_z = np.abs(0.6745 * (data - medians) / (mad + 1e-10))
            max_mz = np.max(modified_z, axis=1)
            threshold = 3.5
            outliers = max_mz > threshold
            
            results["modified_z_score"] = {
                "outliers": outliers,
                "scores": max_mz,
                "threshold": threshold,
                "method_type": "statistical"
            }
        except Exception:
            pass
        
        # Vectorized IQR
        try:
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower_bounds = Q1 - 1.5 * IQR
            upper_bounds = Q3 + 1.5 * IQR
            
            outliers = np.any((data < lower_bounds) | (data > upper_bounds), axis=1)
            
            results["iqr"] = {
                "outliers": outliers,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "method_type": "statistical"
            }
        except Exception:
            pass
        
        # Fast PCA-based outlier detection
        if n_features > 1:
            try:
                # Use first PC for univariate outlier detection
                pca = PCA(n_components=1, random_state=42)
                pc1 = pca.fit_transform(data).ravel()
                
                # Robust statistics on PC1
                median_pc1 = np.median(pc1)
                mad_pc1 = np.median(np.abs(pc1 - median_pc1))
                threshold = 3.0
                
                if mad_pc1 > 1e-10:
                    outliers = np.abs((pc1 - median_pc1) / (mad_pc1 * 1.4826)) > threshold
                    results["pca_outlier"] = {
                        "outliers": outliers,
                        "scores": np.abs((pc1 - median_pc1) / (mad_pc1 * 1.4826 + 1e-10)),
                        "threshold": threshold,
                        "explained_variance": float(pca.explained_variance_ratio_[0]),
                        "method_type": "dimensionality_reduction"
                    }
            except Exception:
                pass
        
        return results

    # ---------------------- Fast Distance-Based Methods ----------------------
    def _fast_distance_based(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Optimized distance-based methods with smart subsampling"""
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape
        
        # Smart subsampling for large datasets
        if n_samples > self.max_sample_size:
            indices = np.random.choice(n_samples, self.max_sample_size, replace=False)
            data_sample = data[indices]
            use_subsampling = True
        else:
            data_sample = data
            indices = np.arange(n_samples)
            use_subsampling = False
        
        # Fast KNN distance (optimized k selection)
        try:
            k = max(5, min(20, len(data_sample) // 20))
            
            if len(data_sample) > k:
                nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1)
                nn.fit(data_sample)
                
                if use_subsampling:
                    # Get distances for all data using fitted model
                    distances, _ = nn.kneighbors(data)
                else:
                    distances, _ = nn.kneighbors(data_sample)
                
                knn_distances = np.mean(distances[:, 1:], axis=1)
                threshold = np.percentile(knn_distances, 100 * (1 - config.outlier_contamination))
                outliers = knn_distances > threshold
                
                results["knn_distance"] = {
                    "outliers": outliers,
                    "distances": knn_distances,
                    "threshold": float(threshold),
                    "k": k,
                    "method_type": "distance_based"
                }
        except Exception:
            pass
        
        # Fast Robust Mahalanobis (with fallback)
        try:
            if n_features <= 50 and len(data_sample) >= n_features * 2:
                mcd = MinCovDet(random_state=getattr(config, "random_state", 42))
                mcd.fit(data_sample)
                
                if use_subsampling:
                    mahal_distances = mcd.mahalanobis(data)
                else:
                    mahal_distances = mcd.mahalanobis(data_sample)
                
                threshold = stats.chi2.ppf(1 - config.outlier_contamination, n_features)
                outliers = mahal_distances > threshold
                
                results["mahalanobis_robust"] = {
                    "outliers": outliers,
                    "distances": mahal_distances,
                    "threshold": float(threshold),
                    "method_type": "distance_based"
                }
        except Exception:
            pass
        
        # Fast DBSCAN with adaptive parameters
        try:
            if len(data_sample) >= 10:
                # Quick parameter estimation
                k = max(3, min(10, len(data_sample) // 50))
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(data_sample)
                distances, _ = nn.kneighbors(data_sample)
                eps = np.percentile(distances[:, -1], 75)  # 75th percentile for robustness
                
                if eps > 0:
                    dbscan = DBSCAN(eps=eps, min_samples=k, n_jobs=-1)
                    
                    if use_subsampling:
                        # Fit on sample, predict on all
                        labels_sample = dbscan.fit_predict(data_sample)
                        # For new points, use nearest neighbor to assign cluster
                        nn_full = NearestNeighbors(n_neighbors=1)
                        nn_full.fit(data_sample)
                        _, indices_nn = nn_full.kneighbors(data)
                        labels = labels_sample[indices_nn.ravel()]
                    else:
                        labels = dbscan.fit_predict(data_sample)
                    
                    outliers = labels == -1
                    
                    results["dbscan"] = {
                        "outliers": outliers,
                        "labels": labels,
                        "eps": eps,
                        "min_samples": k,
                        "method_type": "density_based"
                    }
        except Exception:
            pass
        
        return results

    # ---------------------- Smart ML-Based Detection ----------------------
    def _smart_ml_detection(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """ML methods with adaptive complexity based on data size"""
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape
        
        # Isolation Forest (always include - very fast)
        try:
            # Adaptive parameters based on data size
            if n_samples < 1000:
                n_estimators = 200
                max_samples = "auto"
            elif n_samples < 5000:
                n_estimators = 150
                max_samples = min(512, n_samples)
            else:
                n_estimators = 100
                max_samples = min(1024, n_samples)
            
            iso_forest = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=config.outlier_contamination,
                random_state=getattr(config, "random_state", 42),
                n_jobs=-1
            )
            
            outliers = iso_forest.fit_predict(data) == -1
            scores = -iso_forest.decision_function(data)
            
            results["isolation_forest"] = {
                "outliers": outliers,
                "scores": scores,
                "n_estimators": n_estimators,
                "method_type": "ensemble"
            }
        except Exception:
            pass
        
        # LOF (only for smaller datasets due to O(n²) complexity)
        if n_samples <= 5000:
            try:
                k = max(5, min(50, n_samples // 20))
                
                lof = LocalOutlierFactor(
                    n_neighbors=k,
                    contamination=config.outlier_contamination,
                    n_jobs=-1
                )
                
                outliers = lof.fit_predict(data) == -1
                scores = -lof.negative_outlier_factor_
                
                results["local_outlier_factor"] = {
                    "outliers": outliers,
                    "scores": scores,
                    "n_neighbors": k,
                    "method_type": "density_based"
                }
            except Exception:
                pass
        
        # One-Class SVM (only for smaller datasets)
        if n_samples <= 2000:
            try:
                nu = np.clip(config.outlier_contamination, 0.01, 0.5)
                
                oc_svm = OneClassSVM(
                    kernel="rbf",
                    gamma="scale",
                    nu=nu
                )
                
                outliers = oc_svm.fit_predict(data) == -1
                scores = -oc_svm.decision_function(data)
                
                results["one_class_svm"] = {
                    "outliers": outliers,
                    "scores": scores,
                    "kernel": "rbf",
                    "method_type": "boundary_based"
                }
            except Exception:
                pass
        
        # Elliptic Envelope (only for reasonable feature counts)
        if n_features <= 20 and n_samples >= n_features * 3:
            try:
                elliptic = EllipticEnvelope(
                    contamination=config.outlier_contamination,
                    random_state=getattr(config, "random_state", 42)
                )
                
                outliers = elliptic.fit_predict(data) == -1
                scores = -elliptic.decision_function(data)
                
                results["elliptic_envelope"] = {
                    "outliers": outliers,
                    "scores": scores,
                    "method_type": "covariance_based"
                }
            except Exception:
                pass
        
        return results

    # ---------------------- SOTA Advanced Methods ----------------------
    def _sota_advanced_methods(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Cutting-edge outlier detection methods"""
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape
        
        # ECOD (Empirical Cumulative Distribution Outlier Detection) - NEW SOTA
        try:
            # Fast implementation of ECOD principle
            # Transform features to empirical CDF space
            ecdf_data = np.zeros_like(data)
            for i in range(n_features):
                sorted_vals = np.sort(data[:, i])
                ecdf_data[:, i] = np.searchsorted(sorted_vals, data[:, i]) / n_samples
            
            # Compute tail probabilities
            tail_probs = np.minimum(ecdf_data, 1 - ecdf_data)
            min_tail_probs = np.min(tail_probs, axis=1)
            
            threshold = np.percentile(min_tail_probs, config.outlier_contamination * 100)
            outliers = min_tail_probs <= threshold
            scores = 1 - min_tail_probs  # Higher score = more outlying
            
            results["ecod"] = {
                "outliers": outliers,
                "scores": scores,
                "threshold": float(threshold),
                "method_type": "distribution_based"
            }
        except Exception:
            pass
        
        # Fast COPOD (Copula-based Outlier Detection)
        try:
            # Simplified COPOD using rank statistics
            ranks = np.zeros_like(data)
            for i in range(n_features):
                ranks[:, i] = stats.rankdata(data[:, i]) / n_samples
            
            # Empirical copula deviation from independence
            expected_rank_prod = np.prod(ranks, axis=1)
            
            # Compare with uniform distribution expectation
            copula_scores = -np.log(expected_rank_prod + 1e-10)
            threshold = np.percentile(copula_scores, 100 * (1 - config.outlier_contamination))
            outliers = copula_scores > threshold
            
            results["copod"] = {
                "outliers": outliers,
                "scores": copula_scores,
                "threshold": float(threshold),
                "method_type": "copula_based"
            }
        except Exception:
            pass
        
        # Angle-based Outlier Detection (ABOD) - simplified
        if n_samples <= 1000 and n_features <= 10:  # Only for small datasets
            try:
                # Simplified ABOD: compute variance of angles to other points
                angle_variances = np.zeros(n_samples)
                
                for i in range(n_samples):
                    # Sample subset of points for efficiency
                    sample_size = min(50, n_samples - 1)
                    other_indices = np.random.choice(
                        [j for j in range(n_samples) if j != i], 
                        size=sample_size, 
                        replace=False
                    )
                    
                    # Compute angles
                    angles = []
                    point_i = data[i]
                    
                    for j in range(len(other_indices) - 1):
                        for k in range(j + 1, len(other_indices)):
                            idx_j, idx_k = other_indices[j], other_indices[k]
                            vec_j = data[idx_j] - point_i
                            vec_k = data[idx_k] - point_i
                            
                            norm_j = np.linalg.norm(vec_j)
                            norm_k = np.linalg.norm(vec_k)
                            
                            if norm_j > 1e-10 and norm_k > 1e-10:
                                cos_angle = np.dot(vec_j, vec_k) / (norm_j * norm_k)
                                cos_angle = np.clip(cos_angle, -1, 1)
                                angle = np.arccos(cos_angle)
                                angles.append(angle)
                    
                    if len(angles) > 1:
                        angle_variances[i] = np.var(angles)
                
                # Lower variance = more outlying (points in "corners")
                scores = 1 / (angle_variances + 1e-10)
                threshold = np.percentile(scores, 100 * (1 - config.outlier_contamination))
                outliers = scores > threshold
                
                results["abod"] = {
                    "outliers": outliers,
                    "scores": scores,
                    "threshold": float(threshold),
                    "method_type": "angle_based"
                }
            except Exception:
                pass
        
        return results

    # ---------------------- Fast Smart Ensemble ----------------------
    def _fast_ensemble(self, all_results: Dict[str, Any], data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Fast ensemble with weighted voting based on method reliability"""
        if len(all_results) < 2:
            return {}
        
        try:
            method_weights = {
                "isolation_forest": 1.0,
                "z_score": 0.7,
                "modified_z_score": 0.8,
                "iqr": 0.6,
                "mahalanobis_robust": 1.2,
                "knn_distance": 1.0,
                "local_outlier_factor": 1.1,
                "dbscan": 0.9,
                "one_class_svm": 0.8,
                "elliptic_envelope": 1.0,
                "pca_outlier": 0.7,
                "ecod": 1.3,  # New SOTA method gets higher weight
                "copod": 1.2,
                "abod": 0.8,
            }
            
            outlier_votes = []
            score_arrays = []
            participating_methods = []
            weights = []
            
            for method_name, results in all_results.items():
                outliers = results.get("outliers")
                scores = results.get("scores")
                
                if isinstance(outliers, np.ndarray) and outliers.dtype == bool:
                    weight = method_weights.get(method_name, 0.5)
                    weights.append(weight)
                    outlier_votes.append(outliers.astype(float) * weight)
                    participating_methods.append(method_name)
                    
                    if scores is not None:
                        # Normalize scores to [0, 1]
                        scores = np.asarray(scores, dtype=float)
                        if scores.std() > 1e-10:
                            scores = (scores - scores.min()) / (scores.max() - scores.min())
                        score_arrays.append(scores * weight)
                    else:
                        score_arrays.append(outliers.astype(float) * weight)
            
            if not outlier_votes:
                return {}
            
            # Weighted ensemble
            total_weight = sum(weights)
            ensemble_votes = np.sum(outlier_votes, axis=0) / total_weight
            ensemble_scores = np.sum(score_arrays, axis=0) / total_weight
            
            # Adaptive threshold based on expected contamination
            threshold = np.percentile(ensemble_scores, 100 * (1 - config.outlier_contamination))
            outliers = ensemble_scores > threshold
            
            # Confidence measure: higher when methods agree
            vote_std = np.std(outlier_votes, axis=0)
            confidence = 1 - (vote_std / (np.mean(weights) + 1e-10))
            
            return {
                "outliers": outliers,
                "scores": ensemble_scores,
                "vote_scores": ensemble_votes,
                "confidence": confidence,
                "threshold": float(threshold),
                "participating_methods": participating_methods,
                "method_weights": dict(zip(participating_methods, weights)),
                "method_type": "ensemble"
            }
        
        except Exception:
            return {}

    # ---------------------- Lightning Evaluation ----------------------
    def _lightning_evaluation(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Fast evaluation with key metrics only"""
        evaluations = {}
        
        for method_name, results in all_results.items():
            outliers = results.get("outliers")
            if not isinstance(outliers, np.ndarray):
                continue
            
            try:
                n_outliers = int(outliers.sum())
                outlier_rate = float(n_outliers / len(outliers))
                
                eval_dict = {
                    "n_outliers": n_outliers,
                    "outlier_rate": outlier_rate,
                    "method_type": results.get("method_type", "unknown")
                }
                
                # Quick quality metrics
                scores = results.get("scores")
                if scores is not None and n_outliers > 0 and n_outliers < len(outliers):
                    scores = np.asarray(scores)
                    outlier_scores = scores[outliers]
                    inlier_scores = scores[~outliers]
                    
                    if len(outlier_scores) > 0 and len(inlier_scores) > 0:
                        separation = float(np.mean(outlier_scores) - np.mean(inlier_scores))
                        eval_dict["score_separation"] = separation
                
                # Add method-specific metrics
                if "confidence" in results:
                    eval_dict["avg_confidence"] = float(np.mean(results["confidence"]))
                
                evaluations[method_name] = eval_dict
                
            except Exception:
                evaluations[method_name] = {"error": "evaluation_failed"}
        
        return evaluations

    # ---------------------- Smart Recommendations ----------------------
    def _smart_recommendations(self, all_results: Dict[str, Any], evaluations: Dict[str, Any], 
                             preprocessing_info: Dict[str, Any], data_size: Tuple[int, int]) -> List[str]:
        """AI-powered recommendations based on results and data characteristics"""
        recs = []
        n_samples, n_features = data_size
        
        # Find best method
        method_scores = {}
        for method, eval_dict in evaluations.items():
            if "error" in eval_dict:
                continue
            
            score = 0.0
            outlier_rate = eval_dict.get("outlier_rate", 0)
            
            # Penalize extreme outlier rates
            if 0.01 <= outlier_rate <= 0.15:
                score += 1.0
            elif outlier_rate > 0.3:
                score -= 0.5
            
            # Reward good separation
            separation = eval_dict.get("score_separation", 0)
            if separation > 0:
                score += min(separation / 2.0, 1.0)
            
            # Bonus for ensemble methods
            if eval_dict.get("method_type") == "ensemble":
                score += 0.3
            
            # Bonus for SOTA methods
            if method in ["ecod", "copod", "isolation_forest"]:
                score += 0.2
            
            method_scores[method] = score
        
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            
            recs.append(f"🏆 Recommended method: {best_method.upper().replace('_', ' ')} (score: {best_score:.2f})")
            
            best_eval = evaluations[best_method]
            outlier_rate = best_eval.get("outlier_rate", 0)
            
            if outlier_rate > 0.2:
                recs.append("⚠️ High outlier rate - consider increasing contamination threshold")
            elif outlier_rate < 0.005:
                recs.append("🔍 Very few outliers - data appears very clean")
            else:
                recs.append(f"✅ Healthy outlier rate: {outlier_rate:.1%}")
        
        # Data-specific recommendations
        if n_samples > 10000:
            recs.append("🚀 Large dataset detected - fast methods prioritized")
        
        if n_features > 50:
            recs.append("📊 High-dimensional data - consider dimensionality reduction first")
        
        if preprocessing_info.get("data_skewness", 0) > 3:
            recs.append("📈 Highly skewed data - robust methods recommended")
        
        # Method availability recommendations
        successful_methods = len([e for e in evaluations.values() if "error" not in e])
        if successful_methods >= 5:
            recs.append("🎯 Multiple methods available - ensemble results highly reliable")
        elif successful_methods <= 2:
            recs.append("⚡ Limited methods applicable - consider data preprocessing")
        
        return recs[:4]  # Keep it concise

    # ---------------------- Adaptive Main Analysis ----------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Adaptive analysis with intelligent method selection"""
        try:
            # Lightning-fast preprocessing
            X, complete_indices, preprocessing_info = self._lightning_preprocessing(data, config)
            n_samples, n_features = X.shape
            
            # Adaptive method selection based on data size
            all_results = {}
            
            # Always run statistical methods (fastest)
            all_results.update(self._lightning_statistical(X, config))
            
            # Distance-based methods for reasonable sizes
            if n_samples <= 10000:
                all_results.update(self._fast_distance_based(X, config))
            
            # ML methods based on data size
            if n_samples <= 20000:
                all_results.update(self._smart_ml_detection(X, config))
            
            # SOTA methods for smaller datasets
            if n_samples <= 5000:
                all_results.update(self._sota_advanced_methods(X, config))
            
            # Map results back to original indices
            N = len(data)
            final_results = {}
            
            for method_name, method_results in all_results.items():
                outliers_subset = method_results.get("outliers")
                if outliers_subset is None:
                    continue
                
                # Map to full dataset
                full_outliers = np.zeros(N, dtype=bool)
                full_outliers[complete_indices] = outliers_subset
                
                mapped_result = {
                    "outliers": full_outliers,
                    "count": int(full_outliers.sum()),
                    "percentage": float(100.0 * full_outliers.mean()),
                    "method_type": method_results.get("method_type", "unknown")
                }
                
                # Map scores if available
                scores_subset = method_results.get("scores")
                if scores_subset is not None:
                    full_scores = np.zeros(N, dtype=float)
                    full_scores[complete_indices] = scores_subset
                    mapped_result["scores"] = full_scores
                
                # Copy other relevant metrics
                for key in ["threshold", "k", "eps", "n_neighbors", "kernel", "confidence"]:
                    if key in method_results:
                        mapped_result[key] = method_results[key]
                
                final_results[method_name] = mapped_result
            
            # Fast ensemble
            ensemble_result = self._fast_ensemble(all_results, X, config)
            if ensemble_result:
                # Map ensemble results
                full_ensemble_outliers = np.zeros(N, dtype=bool)
                full_ensemble_scores = np.zeros(N, dtype=float)
                full_confidence = np.zeros(N, dtype=float)
                
                full_ensemble_outliers[complete_indices] = ensemble_result["outliers"]
                full_ensemble_scores[complete_indices] = ensemble_result["scores"]
                
                if "confidence" in ensemble_result:
                    full_confidence[complete_indices] = ensemble_result["confidence"]
                
                final_results["ensemble"] = {
                    "outliers": full_ensemble_outliers,
                    "count": int(full_ensemble_outliers.sum()),
                    "percentage": float(100.0 * full_ensemble_outliers.mean()),
                    "scores": full_ensemble_scores,
                    "confidence": full_confidence,
                    "method_type": "ensemble",
                    "participating_methods": ensemble_result["participating_methods"],
                    "method_weights": ensemble_result.get("method_weights", {})
                }
            
            # Lightning evaluation
            evaluations = self._lightning_evaluation(all_results, X)
            
            # Smart recommendations
            recommendations = self._smart_recommendations(
                final_results, evaluations, preprocessing_info, (n_samples, n_features)
            )
            
            # Performance tier classification
            if n_samples < self.fast_threshold:
                perf_tier = "ultra_fast"
                tier_desc = "All methods available"
            elif n_samples < self.medium_threshold:
                perf_tier = "fast"
                tier_desc = "Most methods available"
            elif n_samples < self.large_threshold:
                perf_tier = "medium"
                tier_desc = "Core methods with subsampling"
            else:
                perf_tier = "large_scale"
                tier_desc = "Statistical and fast ML methods only"
            
            return {
                "outlier_results": final_results,
                "evaluations": evaluations,
                "preprocessing_info": preprocessing_info,
                "data_characteristics": {
                    "total_samples": int(N),
                    "analyzed_samples": int(n_samples),
                    "missing_samples": int(N - n_samples),
                    "n_features": int(n_features),
                    "contamination_rate": float(config.outlier_contamination),
                    "performance_tier": perf_tier,
                    "tier_description": tier_desc
                },
                "recommendations": recommendations,
                "summary": {
                    "methods_attempted": len(all_results),
                    "successful_methods": len(final_results),
                    "best_method": (max(evaluations.keys(), 
                                      key=lambda k: evaluations[k].get("score_separation", 0)) 
                                   if evaluations else None),
                    "overall_outlier_rate": float(np.mean([r["percentage"] for r in final_results.values()]) / 100.0) if final_results else 0.0,
                    "ensemble_available": "ensemble" in final_results,
                    "sota_methods_used": any(method in final_results for method in ["ecod", "copod", "abod"])
                },
                "performance_info": {
                    "adaptive_selection": True,
                    "subsampling_used": n_samples > self.max_sample_size,
                    "parallel_processing": True,
                    "optimization_level": "high"
                }
            }
            
        except Exception as e:
            return {
                "error": f"SOTA outlier analysis failed: {e}",
                "fallback_available": True,
                "recommendations": ["Consider data preprocessing", "Check for data quality issues"]
            }
