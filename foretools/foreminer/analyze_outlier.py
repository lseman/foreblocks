from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import (PowerTransformer, RobustScaler,
                                   StandardScaler)
from sklearn.svm import OneClassSVM

from .foreminer_aux import *


class OutlierAnalyzer(AnalysisStrategy):
    """State-of-the-art outlier detection with ensemble methods and adaptive thresholding"""

    @property
    def name(self) -> str:
        return "outliers"

    def _adaptive_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Enhanced preprocessing with missing value handling and scaling selection"""
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            raise ValueError("No numeric data available for outlier detection")

        preprocessing_info = {}

        # Analyze missing data pattern
        na_mask = numeric_data.isna().any(axis=1)
        missing_percentage = na_mask.sum() / len(data) * 100
        preprocessing_info["missing_data_percentage"] = float(missing_percentage)
        preprocessing_info["samples_with_missing"] = int(na_mask.sum())

        if na_mask.all():
            raise ValueError("All samples have missing values")

        # Handle missing values intelligently
        if missing_percentage > 50:
            # High missing data - use only complete cases
            clean_data = numeric_data.dropna()
            # Get integer positions, not pandas index values
            complete_indices = np.where(~na_mask)[0]  # Convert to integer positions
            preprocessing_info["handling_strategy"] = "complete_cases_only"
        elif missing_percentage > 10:
            # Moderate missing data - impute with robust methods
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            clean_data = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_data.columns,
                index=numeric_data.index,
            )
            complete_indices = np.arange(
                len(clean_data)
            )  # All indices since we imputed
            preprocessing_info["handling_strategy"] = "median_imputation"
        else:
            # Low missing data - use complete cases
            clean_data = numeric_data.dropna()
            # Get integer positions, not pandas index values
            complete_indices = np.where(~na_mask)[0]  # Convert to integer positions
            preprocessing_info["handling_strategy"] = "complete_cases_only"

        # Data quality assessment
        preprocessing_info["final_sample_size"] = len(clean_data)
        preprocessing_info["final_feature_count"] = clean_data.shape[1]

        # Detect data characteristics for optimal scaling
        skewness = clean_data.skew().abs().mean()
        kurtosis = clean_data.kurtosis().abs().mean()
        preprocessing_info["data_skewness"] = float(skewness)
        preprocessing_info["data_kurtosis"] = float(kurtosis)

        # Adaptive scaling selection
        scaling_methods = []

        if skewness > 3 or kurtosis > 10:
            # Highly skewed/heavy-tailed data
            scaling_methods = [
                (
                    "power_transform",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("robust", RobustScaler()),
                ("standard", StandardScaler()),
            ]
        elif skewness > 1.5:
            # Moderately skewed data
            scaling_methods = [
                ("robust", RobustScaler()),
                (
                    "power_transform",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("standard", StandardScaler()),
            ]
        else:
            # Well-behaved data
            scaling_methods = [
                ("standard", StandardScaler()),
                ("robust", RobustScaler()),
            ]

        # Select best scaling method based on condition number
        best_scaler = None
        best_score = float("inf")

        for name, scaler in scaling_methods:
            try:
                scaled = scaler.fit_transform(clean_data)
                # Evaluate scaling quality
                cond_number = np.linalg.cond(np.cov(scaled.T))
                if cond_number < best_score:
                    best_score = cond_number
                    best_scaler = scaler
                    preprocessing_info["scaling_method"] = name
            except Exception:
                continue

        if best_scaler is None:
            best_scaler = RobustScaler()
            preprocessing_info["scaling_method"] = "robust_fallback"

        scaled_data = best_scaler.fit_transform(clean_data)
        preprocessing_info["condition_number"] = (
            float(best_score) if best_score != float("inf") else None
        )

        return scaled_data, complete_indices, preprocessing_info

    def _statistical_outlier_detection(
        self, data: np.ndarray, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Statistical outlier detection methods"""
        results = {}
        n_samples, n_features = data.shape

        # Z-score based detection (multivariate)
        try:
            z_scores = np.abs(stats.zscore(data, axis=0))
            z_threshold = 3.0  # Standard 3-sigma rule
            z_outliers = (z_scores > z_threshold).any(axis=1)

            results["z_score"] = {
                "outliers": z_outliers,
                "scores": np.max(z_scores, axis=1),
                "threshold": z_threshold,
                "method_type": "statistical",
            }
        except Exception as e:
            print(f"Z-score detection failed: {e}")

        # Modified Z-score (using median)
        try:
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            modified_z_scores = (
                0.6745 * (data - median) / (mad + 1e-10)
            )  # Avoid division by zero
            modified_z_scores = np.abs(modified_z_scores)
            mz_threshold = 3.5
            mz_outliers = (modified_z_scores > mz_threshold).any(axis=1)

            results["modified_z_score"] = {
                "outliers": mz_outliers,
                "scores": np.max(modified_z_scores, axis=1),
                "threshold": mz_threshold,
                "method_type": "statistical",
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

            results["iqr"] = {
                "outliers": iqr_outliers,
                "lower_bounds": iqr_lower,
                "upper_bounds": iqr_upper,
                "method_type": "statistical",
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
            t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            grubbs_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(
                t_critical**2 / (n - 2 + t_critical**2)
            )

            grubbs_outliers = z_scores > grubbs_critical

            results["grubbs"] = {
                "outliers": grubbs_outliers,
                "scores": z_scores,
                "threshold": grubbs_critical,
                "method_type": "statistical",
            }
        except Exception as e:
            print(f"Grubbs test failed: {e}")

        return results

    def _distance_based_detection(
        self, data: np.ndarray, config: AnalysisConfig
    ) -> Dict[str, Any]:
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

            results["mahalanobis_robust"] = {
                "outliers": maha_outliers,
                "distances": maha_distances,
                "threshold": threshold,
                "method_type": "distance_based",
            }
        except Exception as e:
            print(f"Robust Mahalanobis detection failed: {e}")

        # K-nearest neighbors distance
        try:
            k = min(20, n_samples // 10, n_samples - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(
                data
            )  # +1 because point is its own neighbor
            distances, indices = nbrs.kneighbors(data)

            # Use mean distance to k-th nearest neighbors (excluding self)
            knn_distances = np.mean(distances[:, 1:], axis=1)
            knn_threshold = np.percentile(
                knn_distances, 100 * (1 - config.outlier_contamination)
            )
            knn_outliers = knn_distances > knn_threshold

            results["knn_distance"] = {
                "outliers": knn_outliers,
                "distances": knn_distances,
                "threshold": knn_threshold,
                "k": k,
                "method_type": "distance_based",
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

            results["dbscan"] = {
                "outliers": dbscan_outliers,
                "labels": labels,
                "eps": eps,
                "min_samples": max(2, k),
                "method_type": "density_based",
            }
        except Exception as e:
            print(f"DBSCAN outlier detection failed: {e}")

        return results

    def _machine_learning_detection(
        self, data: np.ndarray, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Machine learning based outlier detection"""
        results = {}

        # Enhanced Isolation Forest with multiple configurations
        try:
            # Try different configurations and pick the most stable one
            forest_configs = [
                {
                    "n_estimators": 200,
                    "max_samples": "auto",
                    "contamination": config.outlier_contamination,
                },
                {
                    "n_estimators": 150,
                    "max_samples": min(256, len(data)),
                    "contamination": config.outlier_contamination,
                },
                {
                    "n_estimators": 100,
                    "max_samples": 0.8,
                    "contamination": config.outlier_contamination,
                },
            ]

            best_forest = None
            best_consistency = 0

            for forest_config in forest_configs:
                try:
                    # Run multiple times to check consistency
                    predictions = []
                    for seed in range(3):
                        forest = IsolationForest(
                            random_state=config.random_state + seed, **forest_config
                        )
                        pred = forest.fit_predict(data)
                        predictions.append(pred)

                    # Calculate consistency (agreement between runs)
                    consistency = np.mean(
                        [
                            np.mean(p1 == p2)
                            for p1 in predictions
                            for p2 in predictions
                            if not np.array_equal(p1, p2)
                        ]
                    )

                    if consistency > best_consistency:
                        best_consistency = consistency
                        best_forest = IsolationForest(
                            random_state=config.random_state, **forest_config
                        )

                except Exception:
                    continue

            if best_forest is not None:
                best_forest.fit(data)
                iso_predictions = best_forest.predict(data) == -1
                iso_scores = best_forest.decision_function(data)

                results["isolation_forest"] = {
                    "outliers": iso_predictions,
                    "scores": -iso_scores,  # Negative because lower scores = more anomalous
                    "model": best_forest,
                    "method_type": "ensemble",
                }
        except Exception as e:
            print(f"Isolation Forest detection failed: {e}")

        # Enhanced Local Outlier Factor
        try:
            # Adaptive neighborhood size
            n_samples = len(data)
            neighbor_sizes = [
                min(20, max(5, n_samples // 20)),
                min(50, max(10, n_samples // 10)),
            ]

            best_lof = None
            best_score = -float("inf")

            for n_neighbors in neighbor_sizes:
                try:
                    lof = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        contamination=config.outlier_contamination,
                    )
                    lof_predictions = lof.fit_predict(data) == -1

                    # Evaluate using silhouette-like metric
                    if lof_predictions.sum() > 0 and (~lof_predictions).sum() > 0:
                        from sklearn.metrics import silhouette_score

                        score = silhouette_score(
                            data, ~lof_predictions
                        )  # Invert for outlier/inlier

                        if score > best_score:
                            best_score = score
                            best_lof = lof

                except Exception:
                    continue

            if best_lof is not None:
                lof_predictions = best_lof.fit_predict(data) == -1
                lof_scores = -best_lof.negative_outlier_factor_

                results["local_outlier_factor"] = {
                    "outliers": lof_predictions,
                    "scores": lof_scores,
                    "n_neighbors": best_lof.n_neighbors,
                    "method_type": "density_based",
                }
        except Exception as e:
            print(f"LOF detection failed: {e}")

        # One-Class SVM with multiple kernels
        try:
            kernels = ["rbf", "sigmoid"]
            best_svm = None
            best_score = -float("inf")

            for kernel in kernels:
                try:
                    svm = OneClassSVM(
                        kernel=kernel, gamma="scale", nu=config.outlier_contamination
                    )
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

                results["one_class_svm"] = {
                    "outliers": svm_predictions,
                    "scores": svm_scores,
                    "kernel": best_svm.kernel,
                    "method_type": "boundary_based",
                }
        except Exception as e:
            print(f"One-Class SVM detection failed: {e}")

        # Elliptic Envelope
        try:
            elliptic = EllipticEnvelope(
                contamination=config.outlier_contamination,
                random_state=config.random_state,
            )
            elliptic.fit(data)
            elliptic_predictions = elliptic.predict(data) == -1
            elliptic_scores = elliptic.decision_function(data)

            results["elliptic_envelope"] = {
                "outliers": elliptic_predictions,
                "scores": -elliptic_scores,
                "method_type": "covariance_based",
            }
        except Exception as e:
            print(f"Elliptic Envelope detection failed: {e}")

        return results

    def _advanced_detection_methods(
        self, data: np.ndarray, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Advanced outlier detection methods using external libraries"""
        results = {}

        # PyOD methods (if available)
        if HAS_PYOD:
            from pyod.models.cblof import CBLOF
            from pyod.models.feature_bagging import FeatureBagging
            from pyod.models.hbos import HBOS
            try:
                # Histogram-based Outlier Score
                hbos = HBOS(contamination=config.outlier_contamination)
                hbos.fit(data)
                hbos_predictions = hbos.predict(data) == 1  # PyOD uses 1 for outliers
                hbos_scores = hbos.decision_scores_

                results["hbos"] = {
                    "outliers": hbos_predictions,
                    "scores": hbos_scores,
                    "method_type": "histogram_based",
                }
            except Exception as e:
                print(f"HBOS detection failed: {e}")

            try:
                # Feature Bagging
                feature_bagging = FeatureBagging(
                    contamination=config.outlier_contamination,
                    random_state=config.random_state,
                )
                feature_bagging.fit(data)
                fb_predictions = feature_bagging.predict(data) == 1
                fb_scores = feature_bagging.decision_scores_

                results["feature_bagging"] = {
                    "outliers": fb_predictions,
                    "scores": fb_scores,
                    "method_type": "ensemble",
                }
            except Exception as e:
                print(f"Feature Bagging detection failed: {e}")

            try:
                # Cluster-based Local Outlier Factor
                if len(data) >= 20:  # CBLOF needs sufficient data
                    cblof = CBLOF(
                        contamination=config.outlier_contamination,
                        random_state=config.random_state,
                    )
                    cblof.fit(data)
                    cblof_predictions = cblof.predict(data) == 1
                    cblof_scores = cblof.decision_scores_

                    results["cblof"] = {
                        "outliers": cblof_predictions,
                        "scores": cblof_scores,
                        "method_type": "cluster_based",
                    }
            except Exception as e:
                print(f"CBLOF detection failed: {e}")

        # HDBSCAN-based outlier detection
        if HAS_HDBSCAN:
            try:
                min_cluster_size = max(5, len(data) // 50)
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(1, min_cluster_size // 2),
                )
                cluster_labels = clusterer.fit_predict(data)
                hdbscan_outliers = cluster_labels == -1

                # Use outlier scores if available
                outlier_scores = getattr(clusterer, "outlier_scores_", None)

                results["hdbscan"] = {
                    "outliers": hdbscan_outliers,
                    "scores": (
                        outlier_scores
                        if outlier_scores is not None
                        else np.zeros(len(data))
                    ),
                    "min_cluster_size": min_cluster_size,
                    "method_type": "density_based",
                }
            except Exception as e:
                print(f"HDBSCAN outlier detection failed: {e}")

        return results

    def _ensemble_outlier_detection(
        self, all_results: Dict[str, Any], data: np.ndarray, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Ensemble outlier detection combining multiple methods"""
        if len(all_results) < 2:
            return {}

        try:
            # Collect predictions and scores from all methods
            predictions = []
            scores_list = []
            method_weights = {}

            for method_name, result in all_results.items():
                if "outliers" in result and isinstance(result["outliers"], np.ndarray):
                    predictions.append(result["outliers"])

                    # Normalize scores if available
                    if "scores" in result and result["scores"] is not None:
                        scores = np.array(result["scores"])
                        # Normalize scores to [0, 1]
                        if scores.std() > 0:
                            normalized_scores = (scores - scores.min()) / (
                                scores.max() - scores.min()
                            )
                        else:
                            normalized_scores = np.zeros_like(scores)
                        scores_list.append(normalized_scores)
                    else:
                        # Use binary predictions as scores
                        scores_list.append(result["outliers"].astype(float))

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
                "outliers": ensemble_outliers,
                "scores": ensemble_scores,
                "vote_scores": vote_scores,
                "consensus_strength": consensus_strength,
                "threshold": threshold,
                "participating_methods": list(all_results.keys()),
                "method_type": "ensemble",
            }

        except Exception as e:
            print(f"Ensemble outlier detection failed: {e}")
            return {}

    def _evaluate_outlier_detection(
        self, all_results: Dict[str, Any], data: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate outlier detection results"""
        evaluations = {}

        for method_name, result in all_results.items():
            if "outliers" not in result:
                continue

            evaluation = {}
            outliers = result["outliers"]

            try:
                # Basic statistics
                n_outliers = int(outliers.sum())
                outlier_rate = float(n_outliers / len(outliers))

                evaluation.update(
                    {
                        "n_outliers": n_outliers,
                        "outlier_rate": outlier_rate,
                        "inlier_rate": 1 - outlier_rate,
                    }
                )

                # Separation quality (if scores available)
                if "scores" in result and result["scores"] is not None:
                    scores = np.array(result["scores"])
                    outlier_scores = scores[outliers]
                    inlier_scores = scores[~outliers]

                    if len(outlier_scores) > 0 and len(inlier_scores) > 0:
                        # Score separation
                        separation = np.mean(outlier_scores) - np.mean(inlier_scores)
                        evaluation["score_separation"] = float(separation)

                        # Score overlap (using 95th percentile of inliers vs 5th percentile of outliers)
                        if len(outlier_scores) > 1 and len(inlier_scores) > 1:
                            inlier_95 = np.percentile(inlier_scores, 95)
                            outlier_5 = np.percentile(outlier_scores, 5)
                            overlap = max(0, inlier_95 - outlier_5)
                            evaluation["score_overlap"] = float(overlap)

                # Spatial distribution analysis
                if n_outliers > 0 and len(data) > n_outliers:
                    outlier_data = data[outliers]
                    inlier_data = data[~outliers]

                    # Average distance from outliers to nearest inlier
                    from scipy.spatial.distance import cdist

                    if len(inlier_data) > 0:
                        distances = cdist(outlier_data, inlier_data)
                        min_distances = np.min(distances, axis=1)
                        evaluation["avg_distance_to_inliers"] = float(
                            np.mean(min_distances)
                        )
                        evaluation["isolation_score"] = float(np.median(min_distances))

                # Method-specific evaluations
                method_type = result.get("method_type", "unknown")
                evaluation["method_type"] = method_type

                if method_type == "ensemble":
                    if "consensus_strength" in result:
                        consensus = result["consensus_strength"]
                        evaluation["avg_consensus"] = float(np.mean(consensus))
                        evaluation["consensus_std"] = float(np.std(consensus))

            except Exception as e:
                print(f"Evaluation failed for {method_name}: {e}")
                evaluation["error"] = str(e)

            evaluations[method_name] = evaluation

        return evaluations

    def _generate_outlier_recommendations(
        self,
        all_results: Dict[str, Any],
        evaluations: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
    ) -> List[str]:
        """Generate intelligent recommendations for outlier detection"""
        recommendations = []

        # Find best method based on multiple criteria
        method_scores = {}
        for method_name, eval_metrics in evaluations.items():
            if "error" in eval_metrics:
                continue

            score = 0

            # Score based on separation quality
            if "score_separation" in eval_metrics:
                separation = eval_metrics["score_separation"]
                score += min(separation / 2.0, 1.0) * 0.4  # Normalize and weight

            # Penalize excessive overlap
            if "score_overlap" in eval_metrics:
                overlap = eval_metrics["score_overlap"]
                score -= min(overlap, 1.0) * 0.2

            # Reward good isolation
            if "isolation_score" in eval_metrics:
                isolation = eval_metrics["isolation_score"]
                score += (
                    min(isolation / 5.0, 1.0) * 0.3
                )  # Normalize to reasonable range

            # Consider outlier rate reasonableness (not too high or too low)
            outlier_rate = eval_metrics.get("outlier_rate", 0)
            if 0.01 <= outlier_rate <= 0.2:  # Reasonable range
                score += 0.1
            elif outlier_rate > 0.5:  # Too many outliers
                score -= 0.3

            method_scores[method_name] = max(0, score)

        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            best_eval = evaluations[best_method]

            recommendations.append(
                f"🏆 Best method: {best_method.upper()} (quality score: {best_score:.3f})"
            )

            # Method-specific recommendations
            outlier_rate = best_eval.get("outlier_rate", 0)
            if outlier_rate > 0.3:
                recommendations.append(
                    "⚠️ High outlier rate detected - consider reviewing contamination parameter"
                )
            elif outlier_rate < 0.01:
                recommendations.append(
                    "🔍 Very few outliers found - data may be very clean or threshold too strict"
                )
            else:
                recommendations.append(
                    f"✅ Reasonable outlier rate: {outlier_rate:.1%}"
                )

            # Separation quality feedback
            if "score_separation" in best_eval:
                separation = best_eval["score_separation"]
                if separation > 1.0:
                    recommendations.append(
                        "📊 Excellent outlier-inlier separation detected"
                    )
                elif separation > 0.5:
                    recommendations.append(
                        "👍 Good separation between outliers and inliers"
                    )
                else:
                    recommendations.append(
                        "🤔 Moderate separation - outliers may be subtle"
                    )

        # Data-specific recommendations
        data_characteristics = preprocessing_info

        if data_characteristics.get("missing_data_percentage", 0) > 20:
            recommendations.append(
                "📝 High missing data rate may affect outlier detection accuracy"
            )

        if data_characteristics.get("data_skewness", 0) > 2:
            recommendations.append("📈 Highly skewed data - robust methods recommended")

        condition_number = data_characteristics.get("condition_number")
        if condition_number and condition_number > 1000:
            recommendations.append(
                "🔧 Poor data conditioning - consider dimensionality reduction"
            )

        # Method diversity recommendations
        successful_methods = len([e for e in evaluations.values() if "error" not in e])
        if successful_methods <= 2:
            recommendations.append(
                "🔄 Consider additional detection methods for robust analysis"
            )
        elif successful_methods >= 5:
            recommendations.append(
                "🤝 Multiple methods available - ensemble approach recommended"
            )

        # Ensemble-specific recommendations
        if "ensemble" in evaluations:
            ensemble_eval = evaluations["ensemble"]
            avg_consensus = ensemble_eval.get("avg_consensus", 0)
            if avg_consensus < 0.3:
                recommendations.append(
                    "🎯 High consensus among methods - reliable outliers identified"
                )
            else:
                recommendations.append(
                    "🤷 Low consensus among methods - results may vary"
                )

        return recommendations[:5]  # Limit to top 5

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Main outlier detection analysis with comprehensive methods"""
        try:
            # Enhanced preprocessing
            scaled_data, complete_indices, preprocessing_info = (
                self._adaptive_preprocessing(data, config)
            )

            # Apply different detection methods
            all_results = {}

            # 1. Statistical methods
            try:
                statistical_results = self._statistical_outlier_detection(
                    scaled_data, config
                )
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
                if "outliers" in result:
                    # Create full-length mask
                    full_mask = np.full(full_length, False)
                    if len(complete_indices) > 0:
                        # Ensure we have a proper boolean array and valid indices
                        outlier_mask = np.asarray(result["outliers"], dtype=bool)
                        valid_indices = np.asarray(complete_indices, dtype=int)

                        # Only map if dimensions match
                        if len(outlier_mask) == len(valid_indices):
                            full_mask[valid_indices] = outlier_mask

                    mapped_result = {
                        "outliers": full_mask,
                        "count": int(full_mask.sum()),
                        "percentage": 100.0 * full_mask.sum() / full_length,
                        "method_type": result.get("method_type", "unknown"),
                    }

                    # Map scores if available
                    if "scores" in result and result["scores"] is not None:
                        full_scores = np.full(full_length, 0.0)
                        if len(complete_indices) > 0:
                            scores_array = np.asarray(result["scores"], dtype=float)
                            valid_indices = np.asarray(complete_indices, dtype=int)

                            # Only map if dimensions match
                            if len(scores_array) == len(valid_indices):
                                full_scores[valid_indices] = scores_array
                        mapped_result["scores"] = full_scores

                    # Copy other relevant information
                    for key in [
                        "threshold",
                        "distances",
                        "k",
                        "eps",
                        "kernel",
                        "n_neighbors",
                    ]:
                        if key in result:
                            mapped_result[key] = result[key]

                    final_results[method_name] = mapped_result

            # 5. Ensemble method (using original scaled results for better accuracy)
            ensemble_result = self._ensemble_outlier_detection(
                all_results, scaled_data, config
            )
            if ensemble_result:
                # Map ensemble results to full data
                full_mask = np.full(full_length, False)
                full_scores = np.full(full_length, 0.0)

                if len(complete_indices) > 0:
                    ensemble_outliers = np.asarray(
                        ensemble_result["outliers"], dtype=bool
                    )
                    ensemble_scores = np.asarray(ensemble_result["scores"], dtype=float)
                    valid_indices = np.asarray(complete_indices, dtype=int)

                    # Only map if dimensions match
                    if len(ensemble_outliers) == len(valid_indices):
                        full_mask[valid_indices] = ensemble_outliers
                    if len(ensemble_scores) == len(valid_indices):
                        full_scores[valid_indices] = ensemble_scores

                final_results["ensemble"] = {
                    "outliers": full_mask,
                    "count": int(full_mask.sum()),
                    "percentage": 100.0 * full_mask.sum() / full_length,
                    "scores": full_scores,
                    "method_type": "ensemble",
                    "participating_methods": ensemble_result["participating_methods"],
                    "consensus_strength": ensemble_result.get(
                        "consensus_strength", np.array([])
                    ),
                }

            # Evaluate results
            evaluations = self._evaluate_outlier_detection(all_results, scaled_data)

            # Generate recommendations
            recommendations = self._generate_outlier_recommendations(
                final_results, evaluations, preprocessing_info
            )

            # Compile comprehensive results
            analysis_results = {
                "outlier_results": final_results,
                "evaluations": evaluations,
                "preprocessing_info": preprocessing_info,
                "data_characteristics": {
                    "total_samples": full_length,
                    "analyzed_samples": len(scaled_data),
                    "missing_samples": full_length - len(scaled_data),
                    "n_features": scaled_data.shape[1],
                    "contamination_rate": config.outlier_contamination,
                },
                "recommendations": recommendations,
                "summary": {
                    "methods_attempted": len(all_results),
                    "successful_methods": len(final_results),
                    "best_method": (
                        max(
                            evaluations.keys(),
                            key=lambda x: evaluations[x].get("score_separation", 0),
                        )
                        if evaluations
                        else None
                    ),
                    "overall_outlier_rate": np.mean(
                        [r["percentage"] for r in final_results.values()]
                    )
                    / 100,
                },
            }

            return analysis_results

        except Exception as e:
            return {"error": f"Outlier analysis failed: {str(e)}"}

