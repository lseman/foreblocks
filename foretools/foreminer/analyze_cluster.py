
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import (DBSCAN, AgglomerativeClustering, KMeans,
                             MeanShift, SpectralClustering)
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import (PowerTransformer, RobustScaler,
                                   StandardScaler)

from .foreminer_aux import *


class ClusterAnalyzer(AnalysisStrategy):
    """State-of-the-art clustering analysis with comprehensive evaluation and stability assessment"""

    @property
    def name(self) -> str:
        return "clusters"

    def _adaptive_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced preprocessing with outlier detection and feature scaling"""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()

        if numeric_data.empty:
            raise ValueError("No numeric data available for clustering")

        preprocessing_info = {}

        # Outlier detection using IQR method
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (
            (numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))
        ).any(axis=1)
        n_outliers = outlier_mask.sum()
        preprocessing_info["outliers_detected"] = int(n_outliers)
        preprocessing_info["outlier_percentage"] = float(
            n_outliers / len(numeric_data) * 100
        )

        # Adaptive scaling based on data characteristics
        skewness = numeric_data.skew().abs().mean()
        outlier_fraction = n_outliers / len(numeric_data)

        if skewness > 2 or outlier_fraction > 0.1:
            try:
                scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                scaled_data = scaler.fit_transform(numeric_data)
                preprocessing_info["scaling_method"] = "power_transform"
            except:
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                preprocessing_info["scaling_method"] = "robust"
        else:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            preprocessing_info["scaling_method"] = "standard"

        # Dimensionality assessment
        n_samples, n_features = scaled_data.shape
        preprocessing_info["curse_of_dimensionality_risk"] = n_features > n_samples / 3

        # PCA for high-dimensional data
        if n_features > 50 and n_samples > n_features:
            pca = PCA(
                n_components=min(50, n_samples // 2), random_state=config.random_state
            )
            scaled_data = pca.fit_transform(scaled_data)
            preprocessing_info["pca_applied"] = True
            preprocessing_info["pca_variance_explained"] = float(
                pca.explained_variance_ratio_.sum()
            )
            preprocessing_info["final_dimensions"] = scaled_data.shape[1]
        else:
            preprocessing_info["pca_applied"] = False

        preprocessing_info["final_shape"] = scaled_data.shape
        return scaled_data, preprocessing_info

    def _estimate_optimal_clusters(
        self, data: np.ndarray, max_k: int = 15
    ) -> Dict[str, Any]:
        """Multi-method optimal cluster number estimation"""
        n_samples = len(data)
        max_k = min(max_k, n_samples // 3)

        if max_k < 2:
            return {"optimal_k": 2, "methods": {}, "confidence": "low"}

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
            methods_results["elbow"] = min(max_k, max(2, elbow_k))

        # 2. Silhouette Analysis
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append((k, score))

        if silhouette_scores:
            best_sil_k = max(silhouette_scores, key=lambda x: x[1])[0]
            methods_results["silhouette"] = best_sil_k

        # 3. Calinski-Harabasz Index
        ch_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = calinski_harabasz_score(data, labels)
            ch_scores.append((k, score))

        if ch_scores:
            best_ch_k = max(ch_scores, key=lambda x: x[1])[0]
            methods_results["calinski_harabasz"] = best_ch_k

        # 4. Davies-Bouldin Index (lower is better)
        db_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = davies_bouldin_score(data, labels)
            db_scores.append((k, score))

        if db_scores:
            best_db_k = min(db_scores, key=lambda x: x[1])[0]
            methods_results["davies_bouldin"] = best_db_k

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
                        ref_disp = np.sum(np.var(random_data, axis=0)) * len(
                            random_data
                        )
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
                    methods_results["gap_statistic"] = k
                    break

        except Exception as e:
            print(f"Gap statistic calculation failed: {e}")

        # Consensus estimation
        if methods_results:
            method_values = list(methods_results.values())
            # Weight different methods
            weights = {
                "silhouette": 0.3,
                "calinski_harabasz": 0.25,
                "davies_bouldin": 0.2,
                "elbow": 0.15,
                "gap_statistic": 0.1,
            }

            weighted_sum = sum(
                weights.get(method, 0.2) * k for method, k in methods_results.items()
            )
            optimal_k = max(2, min(max_k, int(np.round(weighted_sum))))

            # Calculate confidence based on agreement
            variance = np.var(method_values)
            if variance < 0.5:
                confidence = "high"
            elif variance < 2.0:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            optimal_k = min(3, max_k)
            confidence = "low"

        return {
            "optimal_k": optimal_k,
            "methods": methods_results,
            "confidence": confidence,
            "method_agreement": len(set(method_values)) if methods_results else 0,
        }

    def _enhanced_kmeans_analysis(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced K-means with multiple initializations and stability analysis"""
        max_k = min(config.max_clusters, len(data) // 3)
        k_range = range(2, max_k + 1)

        all_results = []
        best_model = None
        best_k = optimal_k_info["optimal_k"]
        best_score = -1

        for k in k_range:
            # Multiple runs with different initializations
            stability_scores = []
            models = []

            for init_method in ["k-means++", "random"]:
                try:
                    kmeans = KMeans(
                        n_clusters=k,
                        init=init_method,
                        n_init=20,
                        max_iter=500,
                        random_state=config.random_state,
                        algorithm="lloyd",
                    )
                    labels = kmeans.fit_predict(data)

                    # Stability check: run again and compare
                    kmeans_stable = KMeans(
                        n_clusters=k,
                        init=init_method,
                        n_init=20,
                        max_iter=500,
                        random_state=config.random_state + 1,
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
            best_model_k = max(
                models,
                key=lambda x: (
                    silhouette_score(data, x[1]) if len(set(x[1])) > 1 else -1
                ),
            )
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
                    "k": k,
                    "silhouette": sil_score,
                    "calinski_harabasz": ch_score,
                    "davies_bouldin": db_score,
                    "inertia": kmeans.inertia_,
                    "stability": avg_stability,
                    "balance": balance_score,
                    "cluster_sizes": dict(Counter(labels)),
                    "model": kmeans,
                    "labels": labels,
                }

                all_results.append(result)

                # Update best model based on composite score
                composite_score = (
                    sil_score * 0.4
                    + avg_stability * 0.3
                    + balance_score * 0.2
                    + (1 - db_score / 10) * 0.1
                )

                if composite_score > best_score:
                    best_score = composite_score
                    best_k = k
                    best_model = kmeans

            except Exception as e:
                print(f"Scoring failed for k={k}: {e}")
                continue

        final_labels = (
            best_model.fit_predict(data) if best_model else np.zeros(len(data))
        )

        return {
            "labels": final_labels,
            "model": best_model,
            "scores": all_results,
            "best_k": best_k,
            "centers": best_model.cluster_centers_ if best_model else None,
            "cluster_sizes": dict(Counter(final_labels)),
            "method_type": "centroid_based",
        }

    def _hierarchical_clustering_analysis(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
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

        linkage_methods = ["ward", "complete", "average", "single"]
        best_linkage = None
        best_score = -1

        for method in linkage_methods:
            try:
                # Compute linkage matrix
                if method == "ward":
                    distance_matrix = pdist(sample_data, metric="euclidean")
                else:
                    distance_matrix = pdist(sample_data, metric="euclidean")

                linkage_matrix = linkage(distance_matrix, method=method)

                # Get clusters for optimal k
                optimal_k = optimal_k_info["optimal_k"]
                cluster_labels_sample = fcluster(
                    linkage_matrix, optimal_k, criterion="maxclust"
                )

                # Apply to full dataset using AgglomerativeClustering
                agg_clustering = AgglomerativeClustering(
                    n_clusters=optimal_k, linkage=method, compute_distances=True
                )
                full_labels = agg_clustering.fit_predict(data)

                # Evaluate clustering quality
                sil_score = silhouette_score(data, full_labels)
                ch_score = calinski_harabasz_score(data, full_labels)
                db_score = davies_bouldin_score(data, full_labels)

                result = {
                    "linkage_method": method,
                    "labels": full_labels,
                    "silhouette": sil_score,
                    "calinski_harabasz": ch_score,
                    "davies_bouldin": db_score,
                    "cluster_sizes": dict(Counter(full_labels)),
                    "model": agg_clustering,
                    "linkage_matrix": linkage_matrix,
                    "method_type": "hierarchical",
                }

                results[f"hierarchical_{method}"] = result

                if sil_score > best_score:
                    best_score = sil_score
                    best_linkage = method

            except Exception as e:
                print(f"Hierarchical clustering with {method} failed: {e}")
                continue

        # Return best hierarchical clustering result
        best_result = results.get(f"hierarchical_{best_linkage}", {})
        if best_result:
            best_result["best_linkage_method"] = best_linkage

        return best_result

    def _density_based_clustering(
        self, data: np.ndarray, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Enhanced density-based clustering with parameter optimization"""
        results = {}

        # DBSCAN with automated parameter selection
        try:
            # Estimate eps using k-distance graph
            k = min(4, len(data) // 10)
            nbrs = NearestNeighbors(n_neighbors=k).fit(data)
            distances, indices = nbrs.kneighbors(data)
            distances = np.sort(distances[:, k - 1], axis=0)

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
                "labels": labels,
                "model": dbscan,
                "n_clusters": n_clusters,
                "noise_points": noise_points,
                "eps": eps,
                "min_samples": min_samples,
                "cluster_sizes": dict(Counter(labels[labels != -1])),
                "method_type": "density_based",
            }

            # Calculate silhouette only for non-noise points
            if n_clusters > 1:
                valid_mask = labels != -1
                if valid_mask.sum() > 1:
                    valid_data = data[valid_mask]
                    valid_labels = labels[valid_mask]
                    result["silhouette"] = silhouette_score(valid_data, valid_labels)

            results["dbscan"] = result

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
                    cluster_selection_method="eom",
                )
                labels = hdbscan_model.fit_predict(data)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_points = int((labels == -1).sum())

                result = {
                    "labels": labels,
                    "model": hdbscan_model,
                    "n_clusters": n_clusters,
                    "noise_points": noise_points,
                    "min_cluster_size": min_cluster_size,
                    "cluster_sizes": dict(Counter(labels[labels != -1])),
                    "method_type": "density_based",
                    "probabilities": (
                        hdbscan_model.probabilities_.tolist()
                        if hasattr(hdbscan_model, "probabilities_")
                        else None
                    ),
                }

                if n_clusters > 1:
                    valid_mask = labels != -1
                    if valid_mask.sum() > 1:
                        valid_data = data[valid_mask]
                        valid_labels = labels[valid_mask]
                        result["silhouette"] = silhouette_score(
                            valid_data, valid_labels
                        )

                results["hdbscan"] = result

            except Exception as e:
                print(f"HDBSCAN clustering failed: {e}")

        return results

    def _probabilistic_clustering(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Advanced probabilistic clustering with model selection"""
        results = {}
        max_k = min(config.max_clusters, len(data) // 2)

        # Gaussian Mixture Models with comprehensive model selection
        try:
            covariance_types = ["full", "tied", "diag", "spherical"]
            best_bic = np.inf
            np.inf
            best_gmm = None
            best_k = optimal_k_info["optimal_k"]
            model_comparison = []

            for cov_type in covariance_types:
                for k in range(2, max_k + 1):
                    try:
                        gmm = GaussianMixture(
                            n_components=k,
                            covariance_type=cov_type,
                            max_iter=200,
                            n_init=3,
                            random_state=config.random_state,
                        )
                        gmm.fit(data)

                        bic = gmm.bic(data)
                        aic = gmm.aic(data)
                        log_likelihood = gmm.score(data)

                        model_info = {
                            "k": k,
                            "covariance_type": cov_type,
                            "bic": bic,
                            "aic": aic,
                            "log_likelihood": log_likelihood,
                            "converged": gmm.converged_,
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
                    "labels": labels,
                    "probabilities": probs.tolist(),
                    "model": best_gmm,
                    "best_k": best_k,
                    "best_bic": best_bic,
                    "model_comparison": model_comparison,
                    "cluster_sizes": dict(Counter(labels)),
                    "silhouette": silhouette_score(data, labels),
                    "method_type": "probabilistic",
                    "covariance_type": best_gmm.covariance_type,
                    "means": best_gmm.means_.tolist(),
                    "covariances": best_gmm.covariances_.tolist(),
                }
                results["gmm"] = result

        except Exception as e:
            print(f"GMM clustering failed: {e}")

        # Bayesian Gaussian Mixture (if computational resources allow)
        try:
            if len(data) <= 5000:  # Only for smaller datasets due to computational cost
                bgmm = BayesianGaussianMixture(
                    n_components=min(10, max_k),
                    covariance_type="full",
                    max_iter=200,
                    random_state=config.random_state,
                )
                bgmm.fit(data)
                labels = bgmm.predict(data)

                # Count effective components (with significant weight)
                effective_components = np.sum(bgmm.weights_ > 0.01)

                result = {
                    "labels": labels,
                    "probabilities": bgmm.predict_proba(data).tolist(),
                    "model": bgmm,
                    "effective_components": int(effective_components),
                    "weights": bgmm.weights_.tolist(),
                    "cluster_sizes": dict(Counter(labels)),
                    "silhouette": silhouette_score(data, labels),
                    "method_type": "bayesian_probabilistic",
                }
                results["bayesian_gmm"] = result

        except Exception as e:
            print(f"Bayesian GMM clustering failed: {e}")

        return results

    def _ensemble_clustering(
        self, all_results: Dict[str, Any], data: np.ndarray
    ) -> Dict[str, Any]:
        """Ensemble clustering using consensus from multiple methods"""
        if not all_results:
            return {}

        try:
            # Collect all label assignments
            label_sets = []
            method_weights = {}

            for method_name, result in all_results.items():
                if "labels" in result and len(result["labels"]) == len(data):
                    labels = np.array(result["labels"])
                    label_sets.append(labels)

                    # Weight methods based on their silhouette scores
                    sil_score = result.get("silhouette", 0)
                    method_weights[method_name] = max(0, sil_score)

            if len(label_sets) < 2:
                return {}

            # Create co-association matrix
            n_samples = len(data)
            co_assoc = np.zeros((n_samples, n_samples))

            for labels in label_sets:
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        if (
                            labels[i] == labels[j] and labels[i] != -1
                        ):  # Ignore noise points
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
            linkage_matrix = linkage(condensed_distance, method="average")

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
            consensus_labels = (
                fcluster(linkage_matrix, optimal_k, criterion="maxclust") - 1
            )

            # Evaluate consensus clustering
            consensus_result = {
                "labels": consensus_labels,
                "n_methods": len(label_sets),
                "co_association_matrix": co_assoc.tolist(),
                "consensus_k": optimal_k,
                "cluster_sizes": dict(Counter(consensus_labels)),
                "silhouette": silhouette_score(data, consensus_labels),
                "method_type": "ensemble",
                "participating_methods": list(all_results.keys()),
            }

            return consensus_result

        except Exception as e:
            print(f"Ensemble clustering failed: {e}")
            # Return empty dict instead of error to avoid breaking the analysis
            return {}

    def _evaluate_clustering_results(
        self, all_results: Dict[str, Any], data: np.ndarray
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of clustering results"""
        evaluations = {}

        for method_name, result in all_results.items():
            if "labels" not in result:
                continue

            labels = np.array(result["labels"])
            evaluation = {}

            try:
                # Basic metrics
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = int((labels == -1).sum())

                evaluation.update(
                    {
                        "n_clusters": n_clusters,
                        "n_noise_points": n_noise,
                        "noise_ratio": n_noise / len(labels),
                    }
                )

                # Silhouette analysis (skip if already computed)
                if "silhouette" not in result and n_clusters > 1:
                    valid_mask = labels != -1
                    if valid_mask.sum() > 1:
                        valid_data = data[valid_mask]
                        valid_labels = labels[valid_mask]
                        evaluation["silhouette_score"] = silhouette_score(
                            valid_data, valid_labels
                        )
                else:
                    evaluation["silhouette_score"] = result.get("silhouette", 0)

                # Additional metrics for non-noise clusters
                if n_clusters > 1:
                    valid_mask = labels != -1
                    valid_data = data[valid_mask]
                    valid_labels = labels[valid_mask]

                    if len(valid_data) > 0:
                        evaluation.update(
                            {
                                "calinski_harabasz_score": calinski_harabasz_score(
                                    valid_data, valid_labels
                                ),
                                "davies_bouldin_score": davies_bouldin_score(
                                    valid_data, valid_labels
                                ),
                            }
                        )

                        # Cluster balance and separation
                        cluster_sizes = np.bincount(valid_labels)
                        balance_score = (
                            1 - np.std(cluster_sizes) / np.mean(cluster_sizes)
                            if np.mean(cluster_sizes) > 0
                            else 0
                        )
                        evaluation["cluster_balance"] = balance_score

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
                                            distances_to_others = np.linalg.norm(
                                                other_data - point, axis=1
                                            )
                                            inter_distances.extend(distances_to_others)

                            if intra_distances and inter_distances:
                                mean_intra = np.mean(intra_distances)
                                mean_inter = np.mean(inter_distances)
                                separation_ratio = (
                                    mean_inter / mean_intra if mean_intra > 0 else 0
                                )
                                evaluation["separation_ratio"] = separation_ratio

                # Stability metrics (if multiple runs were performed)
                if "stability" in result:
                    evaluation["stability_score"] = result["stability"]

                # Method-specific metrics
                method_type = result.get("method_type", "unknown")
                evaluation["method_type"] = method_type

                if method_type == "probabilistic" and "probabilities" in result:
                    # Entropy of cluster assignments (uncertainty)
                    probs = np.array(result["probabilities"])
                    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    evaluation["mean_assignment_entropy"] = float(np.mean(entropy))
                    evaluation["assignment_uncertainty"] = float(np.std(entropy))

            except Exception as e:
                print(f"Evaluation failed for {method_name}: {e}")
                evaluation["error"] = str(e)

            evaluations[method_name] = evaluation

        return evaluations

    def _generate_clustering_recommendations(
        self,
        all_results: Dict[str, Any],
        evaluations: Dict[str, Any],
        optimal_k_info: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
    ) -> List[str]:
        """Generate intelligent recommendations based on clustering results"""
        recommendations = []

        # Find best performing method
        method_scores = {}
        for method_name, eval_metrics in evaluations.items():
            if "error" in eval_metrics:
                continue

            score = 0
            weight_sum = 0

            # Weighted scoring
            if "silhouette_score" in eval_metrics:
                score += eval_metrics["silhouette_score"] * 0.3
                weight_sum += 0.3

            if "cluster_balance" in eval_metrics:
                score += eval_metrics["cluster_balance"] * 0.2
                weight_sum += 0.2

            if "separation_ratio" in eval_metrics:
                # Normalize separation ratio (higher is better, but cap at reasonable value)
                norm_sep = min(eval_metrics["separation_ratio"] / 5.0, 1.0)
                score += norm_sep * 0.2
                weight_sum += 0.2

            if "stability_score" in eval_metrics:
                score += eval_metrics["stability_score"] * 0.2
                weight_sum += 0.2

            # Penalize excessive noise
            noise_penalty = eval_metrics.get("noise_ratio", 0)
            score -= noise_penalty * 0.1
            weight_sum += 0.1

            if weight_sum > 0:
                method_scores[method_name] = score / weight_sum

        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            recommendations.append(
                f"🏆 Best method: {best_method.upper()} (score: {best_score:.3f})"
            )

            # Method-specific recommendations
            best_result = all_results.get(best_method, {})
            method_type = best_result.get("method_type", "")

            if "density" in method_type:
                noise_ratio = evaluations[best_method].get("noise_ratio", 0)
                if noise_ratio > 0.1:
                    recommendations.append(
                        f"⚠️ High noise ratio ({noise_ratio:.1%}) - consider adjusting density parameters"
                    )
                else:
                    recommendations.append(
                        "✅ Density-based clustering effectively identified cluster structure"
                    )

            elif method_type == "probabilistic":
                uncertainty = evaluations[best_method].get("assignment_uncertainty", 0)
                if uncertainty > 0.5:
                    recommendations.append(
                        "🔀 High assignment uncertainty suggests overlapping clusters"
                    )
                else:
                    recommendations.append(
                        "📊 Probabilistic clustering shows confident cluster assignments"
                    )

            elif method_type == "hierarchical":
                linkage_method = best_result.get("best_linkage_method", "unknown")
                recommendations.append(
                    f"🌳 Hierarchical clustering with {linkage_method} linkage worked best"
                )

            elif method_type == "ensemble":
                n_methods = best_result.get("n_methods", 0)
                recommendations.append(
                    f"🤝 Ensemble of {n_methods} methods achieved robust consensus"
                )

        # Data-specific recommendations
        if preprocessing_info.get("curse_of_dimensionality_risk", False):
            recommendations.append(
                "📏 High-dimensional data detected - consider dimensionality reduction"
            )

        if preprocessing_info.get("outlier_percentage", 0) > 10:
            recommendations.append(
                "🎯 Many outliers detected - density-based methods recommended"
            )

        # Optimal k analysis
        k_confidence = optimal_k_info.get("confidence", "low")
        optimal_k = optimal_k_info.get("optimal_k", 2)

        if k_confidence == "high":
            recommendations.append(f"✨ Strong evidence for {optimal_k} clusters")
        elif k_confidence == "medium":
            recommendations.append(
                f"🤔 Moderate evidence for {optimal_k} clusters - consider range {optimal_k-1}-{optimal_k+1}"
            )
        else:
            recommendations.append(
                "❓ Unclear optimal cluster number - try multiple values"
            )

        # Performance recommendations
        successful_methods = len([r for r in all_results.values() if "labels" in r])
        if successful_methods <= 2:
            recommendations.append(
                "⚡ Consider trying additional clustering algorithms for comparison"
            )

        # Cluster interpretability
        if best_method in evaluations:
            n_clusters = evaluations[best_method].get("n_clusters", 0)
            if n_clusters > 10:
                recommendations.append(
                    "📊 Many clusters found - consider hierarchical visualization"
                )
            elif n_clusters < 2:
                recommendations.append(
                    "🔍 No clear clusters found - data may not have cluster structure"
                )

        return recommendations[:5]  # Limit to top 5 recommendations

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Main clustering analysis with comprehensive methods and evaluation"""
        try:
            # Enhanced preprocessing
            scaled_data, preprocessing_info = self._adaptive_preprocessing(data, config)

            # Estimate optimal number of clusters
            optimal_k_info = self._estimate_optimal_clusters(
                scaled_data, config.max_clusters
            )

            # Apply different clustering methods
            all_results = {}

            # 1. Enhanced K-means
            try:
                kmeans_result = self._enhanced_kmeans_analysis(
                    scaled_data, config, optimal_k_info
                )
                if kmeans_result:
                    all_results["kmeans"] = kmeans_result
            except Exception as e:
                print(f"Enhanced K-means failed: {e}")

            # 2. Hierarchical clustering
            try:
                hier_result = self._hierarchical_clustering_analysis(
                    scaled_data, config, optimal_k_info
                )
                if hier_result:
                    all_results["hierarchical"] = hier_result
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
                prob_results = self._probabilistic_clustering(
                    scaled_data, config, optimal_k_info
                )
                all_results.update(prob_results)
            except Exception as e:
                print(f"Probabilistic clustering failed: {e}")

            # 5. Additional methods if available
            # Spectral clustering for non-convex clusters
            try:
                if len(scaled_data) <= 2000:  # Computationally expensive
                    spectral = SpectralClustering(
                        n_clusters=optimal_k_info["optimal_k"],
                        random_state=config.random_state,
                        affinity="rbf",
                        gamma=1.0,
                    )
                    spectral_labels = spectral.fit_predict(scaled_data)

                    all_results["spectral"] = {
                        "labels": spectral_labels,
                        "model": spectral,
                        "n_clusters": optimal_k_info["optimal_k"],
                        "cluster_sizes": dict(Counter(spectral_labels)),
                        "silhouette": silhouette_score(scaled_data, spectral_labels),
                        "method_type": "spectral",
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
                        all_results["mean_shift"] = {
                            "labels": ms_labels,
                            "model": mean_shift,
                            "n_clusters": n_clusters_ms,
                            "cluster_sizes": dict(Counter(ms_labels)),
                            "silhouette": silhouette_score(scaled_data, ms_labels),
                            "method_type": "mean_shift",
                        }
            except Exception as e:
                print(f"Mean Shift clustering failed: {e}")

            # 6. Ensemble clustering
            ensemble_result = self._ensemble_clustering(all_results, scaled_data)
            if ensemble_result:
                all_results["ensemble"] = ensemble_result

            # Comprehensive evaluation
            evaluations = self._evaluate_clustering_results(all_results, scaled_data)

            # Generate recommendations
            recommendations = self._generate_clustering_recommendations(
                all_results, evaluations, optimal_k_info, preprocessing_info
            )

            # Compile final results
            final_results = {
                "clustering_results": all_results,
                "evaluations": evaluations,
                "optimal_k_analysis": optimal_k_info,
                "preprocessing_info": preprocessing_info,
                "data_characteristics": {
                    "n_samples": scaled_data.shape[0],
                    "n_features": scaled_data.shape[1],
                    "data_variance": float(np.var(scaled_data)),
                    "data_spread": float(np.ptp(scaled_data)),
                },
                "recommendations": recommendations,
                "summary": {
                    "methods_attempted": len(all_results),
                    "successful_methods": len(
                        [r for r in all_results.values() if "labels" in r]
                    ),
                    "best_method": (
                        max(
                            evaluations.keys(),
                            key=lambda x: evaluations[x].get("silhouette_score", -1),
                        )
                        if evaluations
                        else None
                    ),
                },
            }

            return final_results

        except Exception as e:
            return {"error": f"Clustering analysis failed: {str(e)}"}
