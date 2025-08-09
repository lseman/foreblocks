import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from .foreminer_aux import *


class ClusterAnalyzer(AnalysisStrategy):
    """SOTA fast clustering analysis with adaptive method selection and advanced algorithms"""

    @property
    def name(self) -> str:
        return "clusters"

    def __init__(self):
        # Performance thresholds for adaptive method selection
        self.fast_threshold = 1000      # All methods
        self.medium_threshold = 5000    # Core + selective advanced
        self.large_threshold = 20000    # Fast methods only
        self.max_sample_size = 3000     # Subsampling threshold

    def _count_clusters(self, labels: np.ndarray) -> int:
        """Count clusters ignoring noise label -1."""
        unique_labels = set(labels)
        return int(len(unique_labels) - (1 if -1 in unique_labels else 0))

    # --------------------------- Lightning Preprocessing ---------------------------
    def _lightning_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ultra-fast preprocessing with smart defaults"""
        numeric = data.select_dtypes(include=[np.number]).dropna()
        if numeric.empty or numeric.shape[1] == 0:
            raise ValueError("No numeric data available for clustering")

        info: Dict[str, Any] = {}
        n_samples, n_features = numeric.shape

        # Fast outlier detection using IQR
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Q1 = numeric.quantile(0.25)
            Q3 = numeric.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ((numeric < (Q1 - 1.5 * IQR)) | (numeric > (Q3 + 1.5 * IQR))).any(axis=1)
        
        n_outliers = int(outlier_mask.sum())
        info["outliers_detected"] = n_outliers
        info["outlier_percentage"] = float(n_outliers / max(n_samples, 1) * 100.0)

        # Smart scaling selection (much faster than original)
        skewness = numeric.skew().abs().mean()
        outlier_fraction = n_outliers / max(n_samples, 1)
        
        try:
            if skewness > 2.0 or outlier_fraction > 0.10:
                scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                info["scaling_method"] = "power_transform"
            elif outlier_fraction > 0.05:
                scaler = RobustScaler()
                info["scaling_method"] = "robust"
            else:
                scaler = StandardScaler()
                info["scaling_method"] = "standard"
            
            X = scaler.fit_transform(numeric)
        except Exception:
            scaler = RobustScaler()
            X = scaler.fit_transform(numeric)
            info["scaling_method"] = "robust_fallback"

        # Smart dimensionality reduction
        info["curse_of_dimensionality_risk"] = bool(n_features > n_samples / 3)
        
        if n_features > 50 and n_samples > n_features * 2:
            try:
                # Keep 95% variance or max 50 components
                pca = PCA(n_components=min(50, n_samples // 2), random_state=getattr(config, "random_state", 42))
                X_pca = pca.fit_transform(X)
                
                # Only use PCA if it significantly reduces dimensions while keeping most variance
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                n_components_95 = np.argmax(cumvar >= 0.95) + 1
                
                if n_components_95 < n_features * 0.8:  # Significant reduction
                    X = X_pca[:, :n_components_95]
                    info["pca_applied"] = True
                    info["pca_variance_explained"] = float(cumvar[n_components_95 - 1])
                    info["final_dimensions"] = n_components_95
                else:
                    info["pca_applied"] = False
            except Exception:
                info["pca_applied"] = False
        else:
            info["pca_applied"] = False

        info["final_shape"] = tuple(X.shape)
        return X, info

    # --------------------------- Fast Optimal K Estimation ---------------------------
    def _fast_optimal_k(self, data: np.ndarray, max_k: int = 12) -> Dict[str, Any]:
        """Lightning-fast optimal cluster estimation with smart shortcuts"""
        n_samples = data.shape[0]
        max_k = int(min(max_k, max(3, n_samples // 5)))
        
        if max_k < 3:
            return {"optimal_k": 2, "methods": {}, "confidence": "low", "method_agreement": 0}

        methods: Dict[str, int] = {}
        
        # Sample for very large datasets
        if n_samples > 2000:
            indices = np.random.choice(n_samples, 2000, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data

        # 1) Fast Elbow Method using inertia differences
        try:
            inertias = []
            for k in range(1, max_k + 1):
                if k == 1:
                    inertias.append(float(np.sum(np.var(sample_data, axis=0)) * len(sample_data)))
                else:
                    # Use MiniBatchKMeans for speed
                    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=min(1000, len(sample_data)))
                    kmeans.fit(sample_data)
                    inertias.append(float(kmeans.inertia_))
            
            # Find elbow using second derivative
            if len(inertias) >= 4:
                second_derivatives = np.diff(np.diff(inertias))
                elbow_k = int(np.argmax(second_derivatives) + 2)
                methods["elbow"] = min(max(elbow_k, 2), max_k)
        except Exception:
            pass

        # 2) Fast Silhouette Analysis (reduced k range)
        try:
            silhouette_scores = []
            k_range = range(2, min(max_k + 1, 8))  # Limit range for speed
            
            for k in k_range:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=min(500, len(sample_data)))
                labels = kmeans.fit_predict(sample_data)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(sample_data, labels)
                    silhouette_scores.append((k, score))
            
            if silhouette_scores:
                best_k = max(silhouette_scores, key=lambda x: x[1])[0]
                methods["silhouette"] = best_k
        except Exception:
            pass

        # 3) Fast Calinski-Harabasz Index
        try:
            ch_scores = []
            for k in range(2, min(max_k + 1, 6)):  # Even more limited for speed
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=2)
                labels = kmeans.fit_predict(sample_data)
                if len(np.unique(labels)) > 1:
                    score = calinski_harabasz_score(sample_data, labels)
                    ch_scores.append((k, score))
            
            if ch_scores:
                best_k = max(ch_scores, key=lambda x: x[1])[0]
                methods["calinski_harabasz"] = best_k
        except Exception:
            pass

        # 4) Simple Gap Statistic (reduced iterations)
        try:
            gaps = []
            data_min, data_max = sample_data.min(axis=0), sample_data.max(axis=0)
            
            for k in range(1, min(max_k + 1, 6)):
                # Original data inertia
                if k == 1:
                    orig_inertia = np.sum(np.var(sample_data, axis=0)) * len(sample_data)
                else:
                    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=2)
                    kmeans.fit(sample_data)
                    orig_inertia = kmeans.inertia_

                # Random data inertia (fewer iterations for speed)
                random_inertias = []
                for _ in range(3):  # Reduced from 5 to 3
                    random_data = np.random.uniform(data_min, data_max, sample_data.shape)
                    if k == 1:
                        random_inertias.append(np.sum(np.var(random_data, axis=0)) * len(random_data))
                    else:
                        kmeans_random = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=1)
                        kmeans_random.fit(random_data)
                        random_inertias.append(kmeans_random.inertia_)
                
                gap = np.log(np.mean(random_inertias) + 1e-10) - np.log(orig_inertia + 1e-10)
                gaps.append((k, gap))
            
            # Find first k where gap(k) >= gap(k+1) - se(k+1)
            for i in range(len(gaps) - 1):
                k_current, gap_current = gaps[i]
                k_next, gap_next = gaps[i + 1]
                if gap_current >= gap_next - 0.1:  # Simple SE approximation
                    methods["gap_statistic"] = max(k_current, 2)
                    break
        except Exception:
            pass

        # 5) SOTA: Consensus-based estimation using multiple fast methods
        if len(methods) >= 2:
            # Weight methods by reliability
            weights = {
                "silhouette": 0.35,
                "calinski_harabasz": 0.25, 
                "elbow": 0.20,
                "gap_statistic": 0.20
            }
            
            weighted_sum = sum(weights.get(method, 0.2) * k for method, k in methods.items())
            total_weight = sum(weights.get(method, 0.2) for method in methods.keys())
            optimal_k = int(np.round(weighted_sum / total_weight))
            optimal_k = max(2, min(optimal_k, max_k))
            
            # Confidence based on agreement
            values = list(methods.values())
            variance = np.var(values) if len(values) > 1 else 0.0
            if variance < 0.5:
                confidence = "high"
            elif variance < 2.0:
                confidence = "medium"
            else:
                confidence = "low"
                
            agreement = len(set(values))
        else:
            optimal_k = min(3, max_k)
            confidence = "low"
            agreement = 0

        return {
            "optimal_k": optimal_k,
            "methods": methods,
            "confidence": confidence,
            "method_agreement": agreement
        }

    # --------------------------- SOTA KMeans with Stability ---------------------------
    def _sota_kmeans_analysis(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced KMeans with stability analysis and multiple initializations"""
        max_k = int(min(getattr(config, "max_clusters", 10), max(3, len(data) // 5)))
        optimal_k = optimal_k_info["optimal_k"]
        
        # Focus on range around optimal k for efficiency
        k_range = range(max(2, optimal_k - 2), min(max_k + 1, optimal_k + 3))
        
        best_result = None
        best_score = -1.0
        all_results = []

        for k in k_range:
            try:
                # Use appropriate KMeans variant based on data size
                if len(data) > 10000:
                    kmeans_class = MiniBatchKMeans
                    extra_params = {"batch_size": min(2000, len(data) // 5)}
                else:
                    kmeans_class = KMeans
                    extra_params = {}

                # Multiple initializations for stability
                models = []
                labels_list = []
                
                for init_method in ["k-means++", "random"]:
                    kmeans = kmeans_class(
                        n_clusters=k,
                        init=init_method,
                        n_init=10 if len(data) < 5000 else 5,
                        max_iter=300,
                        random_state=getattr(config, "random_state", 42),
                        **extra_params
                    )
                    labels = kmeans.fit_predict(data)
                    models.append((kmeans, labels))
                    labels_list.append(labels)

                # Select best model by silhouette score
                best_model, best_labels = max(models, key=lambda x: silhouette_score(data, x[1]) if len(np.unique(x[1])) > 1 else -1)

                # Stability analysis
                stability_scores = []
                for i in range(len(labels_list)):
                    for j in range(i + 1, len(labels_list)):
                        stability = adjusted_rand_score(labels_list[i], labels_list[j])
                        stability_scores.append(stability)

                avg_stability = float(np.mean(stability_scores)) if stability_scores else 0.0

                # Compute metrics
                silhouette = silhouette_score(data, best_labels) if len(np.unique(best_labels)) > 1 else -1
                calinski_harabasz = calinski_harabasz_score(data, best_labels) if len(np.unique(best_labels)) > 1 else 0
                davies_bouldin = davies_bouldin_score(data, best_labels) if len(np.unique(best_labels)) > 1 else np.inf

                # Cluster balance
                cluster_sizes = np.bincount(best_labels)
                balance = 1 - (np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-10))

                result = {
                    "k": k,
                    "labels": best_labels,
                    "model": best_model,
                    "silhouette": float(silhouette),
                    "calinski_harabasz": float(calinski_harabasz),
                    "davies_bouldin": float(davies_bouldin),
                    "stability": avg_stability,
                    "balance": float(balance),
                    "inertia": float(best_model.inertia_),
                    "cluster_sizes": dict(Counter(best_labels)),
                    "n_clusters": self._count_clusters(best_labels)
                }

                all_results.append(result)

                # Composite score for best model selection
                composite_score = (
                    silhouette * 0.35 +
                    avg_stability * 0.25 +
                    balance * 0.20 +
                    (1 - min(davies_bouldin / 10.0, 1.0)) * 0.20
                )

                if composite_score > best_score:
                    best_score = composite_score
                    best_result = result

            except Exception as e:
                print(f"KMeans failed for k={k}: {e}")
                continue

        if best_result is None:
            # Fallback
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(data)
            best_result = {
                "labels": labels,
                "model": kmeans,
                "n_clusters": self._count_clusters(labels),
                "cluster_sizes": dict(Counter(labels)),
                "method_type": "centroid_based"
            }

        best_result.update({
            "best_k": best_result.get("k", optimal_k),
            "centers": best_result["model"].cluster_centers_ if hasattr(best_result["model"], "cluster_centers_") else None,
            "method_type": "centroid_based",
            "all_k_results": all_results
        })

        return best_result

    # --------------------------- Fast Hierarchical ---------------------------
    def _fast_hierarchical_clustering(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fast hierarchical clustering with smart sampling"""
        n_samples = data.shape[0]
        optimal_k = optimal_k_info["optimal_k"]

        # Smart sampling for large datasets
        if n_samples > 1500:
            sample_size = min(1500, n_samples)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
            indices = np.arange(n_samples)

        # Test only the most effective linkage methods
        linkage_methods = ["ward", "complete", "average"]
        best_result = None
        best_silhouette = -1

        for method in linkage_methods:
            try:
                # Use AgglomerativeClustering for efficiency
                clusterer = AgglomerativeClustering(
                    n_clusters=optimal_k,
                    linkage=method,
                    metric="euclidean"
                )
                
                if n_samples > 1500:
                    # Fit on sample, then use fitted model concept
                    sample_labels = clusterer.fit_predict(sample_data)
                    
                    # For full dataset, use nearest neighbor assignment
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1)
                    nn.fit(sample_data)
                    _, nearest_indices = nn.kneighbors(data)
                    full_labels = sample_labels[nearest_indices.ravel()]
                else:
                    full_labels = clusterer.fit_predict(data)

                # Evaluate
                silhouette = silhouette_score(data, full_labels) if len(np.unique(full_labels)) > 1 else -1

                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_result = {
                        "labels": full_labels,
                        "model": clusterer,
                        "linkage_method": method,
                        "silhouette": float(silhouette),
                        "n_clusters": self._count_clusters(full_labels),
                        "cluster_sizes": dict(Counter(full_labels)),
                        "method_type": "hierarchical"
                    }

            except Exception as e:
                print(f"Hierarchical clustering with {method} failed: {e}")
                continue

        if best_result:
            # Add additional metrics
            try:
                labels = best_result["labels"]
                if len(np.unique(labels)) > 1:
                    best_result["calinski_harabasz"] = float(calinski_harabasz_score(data, labels))
                    best_result["davies_bouldin"] = float(davies_bouldin_score(data, labels))
            except Exception:
                pass

        return best_result if best_result else {}

    # --------------------------- Smart Density-Based ---------------------------
    def _smart_density_clustering(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Optimized density-based clustering with parameter auto-tuning"""
        results = {}
        n_samples = data.shape[0]

        # DBSCAN with smart parameter selection
        try:
            # Adaptive k based on dimensionality and sample size
            k = max(3, min(int(np.sqrt(data.shape[1]) * 2), n_samples // 20, 10))
            
            # Use sample for parameter estimation if dataset is large
            if n_samples > 3000:
                sample_indices = np.random.choice(n_samples, 3000, replace=False)
                sample_data = data[sample_indices]
            else:
                sample_data = data

            # Fast k-NN distance calculation
            nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
            nn.fit(sample_data)
            distances, _ = nn.kneighbors(sample_data)
            k_distances = np.sort(distances[:, k-1])

            # Find elbow point for eps
            diffs = np.diff(k_distances)
            eps_idx = np.argmax(diffs) if len(diffs) > 0 else len(k_distances) // 2
            eps = float(k_distances[eps_idx])
            
            if eps <= 0:
                eps = float(np.median(k_distances))

            min_samples = max(3, k)

            # Fit DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(data)
            
            n_clusters = self._count_clusters(labels)
            n_noise = int((labels == -1).sum())

            result = {
                "labels": labels,
                "model": dbscan,
                "n_clusters": n_clusters,
                "noise_points": n_noise,
                "noise_ratio": float(n_noise / len(labels)),
                "eps": eps,
                "min_samples": min_samples,
                "cluster_sizes": dict(Counter(labels[labels != -1])),
                "method_type": "density_based"
            }

            # Add silhouette score if we have clusters
            if n_clusters > 1 and n_noise < len(labels):
                mask = labels != -1
                if mask.sum() > 1:
                    result["silhouette"] = float(silhouette_score(data[mask], labels[mask]))

            results["dbscan"] = result

        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")

        # HDBSCAN (if available and dataset not too large)
        if hasattr(config, 'HAS_HDBSCAN') and getattr(config, 'HAS_HDBSCAN', False) and n_samples <= 10000:
            try:
                import hdbscan
                
                min_cluster_size = max(5, n_samples // 100)
                min_samples = max(1, min_cluster_size // 3)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method="eom",
                    core_dist_n_jobs=-1
                )
                
                labels = clusterer.fit_predict(data)
                n_clusters = self._count_clusters(labels)
                n_noise = int((labels == -1).sum())

                result = {
                    "labels": labels,
                    "model": clusterer,
                    "n_clusters": n_clusters,
                    "noise_points": n_noise,
                    "noise_ratio": float(n_noise / len(labels)),
                    "min_cluster_size": min_cluster_size,
                    "cluster_sizes": dict(Counter(labels[labels != -1])),
                    "method_type": "density_based"
                }

                if hasattr(clusterer, 'probabilities_'):
                    result["probabilities"] = clusterer.probabilities_.tolist()

                if n_clusters > 1 and n_noise < len(labels):
                    mask = labels != -1
                    if mask.sum() > 1:
                        result["silhouette"] = float(silhouette_score(data[mask], labels[mask]))

                results["hdbscan"] = result

            except Exception as e:
                print(f"HDBSCAN clustering failed: {e}")

        return results

    # --------------------------- Fast Probabilistic ---------------------------
    def _fast_probabilistic_clustering(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Efficient probabilistic clustering with smart model selection"""
        results = {}
        n_samples, n_features = data.shape
        optimal_k = optimal_k_info["optimal_k"]

        # Limit search space for efficiency
        max_k = min(getattr(config, "max_clusters", 10), 8, n_samples // 10)
        k_range = range(2, max_k + 1)

        # GMM with smart covariance type selection
        try:
            # Choose covariance types based on data characteristics
            if n_features > 20 or n_samples < n_features * 5:
                cov_types = ["diag", "spherical"]  # Simpler models for high-dim or small data
            else:
                cov_types = ["full", "diag"]  # More complex models for sufficient data

            best_gmm = None
            best_bic = np.inf
            best_labels = None

            for cov_type in cov_types:
                for k in k_range:
                    try:
                        gmm = GaussianMixture(
                            n_components=k,
                            covariance_type=cov_type,
                            max_iter=100,  # Reduced for speed
                            n_init=2,      # Reduced for speed
                            random_state=getattr(config, "random_state", 42)
                        )
                        
                        gmm.fit(data)
                        
                        if gmm.converged_:
                            bic = gmm.bic(data)
                            if bic < best_bic:
                                best_bic = bic
                                best_gmm = gmm
                                best_labels = gmm.predict(data)

                    except Exception:
                        continue

            if best_gmm is not None:
                probabilities = best_gmm.predict_proba(data)
                
                result = {
                    "labels": best_labels,
                    "probabilities": probabilities.tolist(),
                    "model": best_gmm,
                    "n_clusters": self._count_clusters(best_labels),
                    "best_bic": float(best_bic),
                    "cluster_sizes": dict(Counter(best_labels)),
                    "method_type": "probabilistic",
                    "covariance_type": best_gmm.covariance_type
                }

                # Add silhouette score
                if len(np.unique(best_labels)) > 1:
                    result["silhouette"] = float(silhouette_score(data, best_labels))

                # Assignment uncertainty
                max_probs = np.max(probabilities, axis=1)
                result["mean_certainty"] = float(np.mean(max_probs))
                result["assignment_uncertainty"] = float(np.std(max_probs))

                results["gmm"] = result

        except Exception as e:
            print(f"GMM clustering failed: {e}")

        # Bayesian GMM for smaller datasets
        if n_samples <= 3000:
            try:
                bgmm = BayesianGaussianMixture(
                    n_components=min(15, max_k * 2),
                    covariance_type="diag" if n_features > 10 else "full",
                    max_iter=100,
                    random_state=getattr(config, "random_state", 42)
                )
                
                bgmm.fit(data)
                labels = bgmm.predict(data)
                
                # Count effective components
                effective_components = int(np.sum(bgmm.weights_ > 0.01))
                
                result = {
                    "labels": labels,
                    "probabilities": bgmm.predict_proba(data).tolist(),
                    "model": bgmm,
                    "n_clusters": self._count_clusters(labels),
                    "effective_components": effective_components,
                    "weights": bgmm.weights_.tolist(),
                    "cluster_sizes": dict(Counter(labels)),
                    "method_type": "bayesian_probabilistic"
                }

                if len(np.unique(labels)) > 1:
                    result["silhouette"] = float(silhouette_score(data, labels))

                results["bayesian_gmm"] = result

            except Exception as e:
                print(f"Bayesian GMM clustering failed: {e}")

        return results

    # --------------------------- SOTA Advanced Methods ---------------------------
    def _sota_advanced_clustering(self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]) -> Dict[str, Any]:
        """Cutting-edge clustering methods for smaller datasets"""
        results = {}
        n_samples, n_features = data.shape
        optimal_k = optimal_k_info["optimal_k"]

        # Spectral Clustering (for non-convex clusters)
        if n_samples <= 2000:
            try:
                # Auto-select gamma for RBF kernel
                pairwise_dists = pdist(data[:min(500, n_samples)])
                gamma = 1.0 / np.median(pairwise_dists)**2 if len(pairwise_dists) > 0 else 1.0
                
                spectral = SpectralClustering(
                    n_clusters=optimal_k,
                    affinity="rbf",
                    gamma=gamma,
                    random_state=getattr(config, "random_state", 42),
                    assign_labels="kmeans",
                    n_jobs=-1
                )
                
                labels = spectral.fit_predict(data)
                
                result = {
                    "labels": labels,
                    "model": spectral,
                    "n_clusters": self._count_clusters(labels),
                    "cluster_sizes": dict(Counter(labels)),
                    "gamma": float(gamma),
                    "method_type": "spectral"
                }

                if len(np.unique(labels)) > 1:
                    result["silhouette"] = float(silhouette_score(data, labels))

                results["spectral"] = result

            except Exception as e:
                print(f"Spectral clustering failed: {e}")

        # Mean Shift (parameter-free clustering)
        if n_samples <= 1000:
            try:
                from sklearn.cluster import estimate_bandwidth

                # Estimate bandwidth
                bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=min(500, n_samples))
                
                if bandwidth > 0:
                    mean_shift = MeanShift(bandwidth=bandwidth, n_jobs=-1)
                    labels = mean_shift.fit_predict(data)
                    
                    n_clusters = self._count_clusters(labels)
                    
                    if n_clusters > 1:
                        result = {
                            "labels": labels,
                            "model": mean_shift,
                            "n_clusters": n_clusters,
                            "cluster_sizes": dict(Counter(labels)),
                            "bandwidth": float(bandwidth),
                            "n_centers": len(mean_shift.cluster_centers_),
                            "method_type": "mean_shift"
                        }

                        result["silhouette"] = float(silhouette_score(data, labels))
                        results["mean_shift"] = result

            except Exception as e:
                print(f"Mean Shift clustering failed: {e}")

        # Affinity Propagation (exemplar-based)
        if n_samples <= 800:
            try:
                from sklearn.cluster import AffinityPropagation

                # Use median of pairwise distances for preference
                sample_size = min(200, n_samples)
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                sample_data = data[sample_indices]
                
                distances = pdist(sample_data)
                preference = np.median(distances) if len(distances) > 0 else -1.0
                
                affinity_prop = AffinityPropagation(
                    preference=preference,
                    max_iter=200,
                    convergence_iter=15,
                    random_state=getattr(config, "random_state", 42)
                )
                
                labels = affinity_prop.fit_predict(data)
                n_clusters = self._count_clusters(labels)
                
                if n_clusters > 1:
                    result = {
                        "labels": labels,
                        "model": affinity_prop,
                        "n_clusters": n_clusters,
                        "cluster_sizes": dict(Counter(labels)),
                        "n_exemplars": len(affinity_prop.cluster_centers_indices_),
                        "preference": float(preference),
                        "method_type": "exemplar_based"
                    }

                    result["silhouette"] = float(silhouette_score(data, labels))
                    results["affinity_propagation"] = result

            except Exception as e:
                print(f"Affinity Propagation clustering failed: {e}")

        return results

    # --------------------------- Lightning Ensemble ---------------------------
    def _lightning_ensemble(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Fast consensus clustering with weighted voting"""
        if len(all_results) < 2:
            return {}

        try:
            # Collect labels and weights
            labels_list = []
            methods_used = []
            method_weights = {
                "kmeans": 1.0,
                "spectral": 1.2,
                "gmm": 1.1,
                "hierarchical": 0.9,
                "dbscan": 0.8,
                "hdbscan": 1.0,
                "bayesian_gmm": 0.9,
                "mean_shift": 0.7,
                "affinity_propagation": 0.8
            }

            for method_name, result in all_results.items():
                labels = result.get("labels")
                if labels is not None and len(labels) == len(data):
                    labels_array = np.asarray(labels)
                    
                    # Skip if too many noise points (for density methods)
                    noise_ratio = (labels_array == -1).mean()
                    if noise_ratio > 0.5:
                        continue
                    
                    labels_list.append(labels_array)
                    methods_used.append(method_name)

            if len(labels_list) < 2:
                return {}

            n_samples = len(data)
            
            # Fast co-association matrix computation
            co_occurrence = np.zeros((n_samples, n_samples), dtype=np.float32)
            total_weight = 0

            for i, (labels, method) in enumerate(zip(labels_list, methods_used)):
                weight = method_weights.get(method, 0.5)
                total_weight += weight
                
                # Only compute for non-noise points
                valid_mask = labels != -1
                valid_labels = labels[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                # Efficient co-occurrence computation
                for cluster_id in np.unique(valid_labels):
                    cluster_mask = valid_labels == cluster_id
                    cluster_indices = valid_indices[cluster_mask]
                    
                    if len(cluster_indices) > 1:
                        # Vectorized co-occurrence update
                        co_occurrence[np.ix_(cluster_indices, cluster_indices)] += weight

            # Normalize co-occurrence
            co_occurrence /= total_weight
            np.fill_diagonal(co_occurrence, 1.0)

            # Convert to distance and apply fast clustering
            distance_matrix = 1.0 - co_occurrence
            distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure symmetry
            np.fill_diagonal(distance_matrix, 0.0)

            # Determine consensus k using majority vote
            cluster_counts = []
            for labels in labels_list:
                unique_labels = np.unique(labels[labels != -1])
                if len(unique_labels) > 1:
                    cluster_counts.append(len(unique_labels))

            if cluster_counts:
                consensus_k = int(np.median(cluster_counts))
            else:
                consensus_k = 3

            # Fast hierarchical clustering on distance matrix
            try:
                condensed_distances = squareform(distance_matrix)
                linkage_matrix = linkage(condensed_distances, method="average")
                consensus_labels = fcluster(linkage_matrix, consensus_k, criterion="maxclust") - 1
            except Exception:
                # Fallback: use KMeans on co-occurrence matrix
                kmeans = KMeans(n_clusters=consensus_k, random_state=42, n_init=5)
                consensus_labels = kmeans.fit_predict(co_occurrence)

            # Compute consensus quality
            silhouette = float(silhouette_score(data, consensus_labels)) if len(np.unique(consensus_labels)) > 1 else -1

            # Compute consensus strength (how much methods agree)
            consensus_strength = np.mean(np.max(co_occurrence, axis=1))

            return {
                "labels": consensus_labels,
                "n_clusters": self._count_clusters(consensus_labels),
                "cluster_sizes": dict(Counter(consensus_labels)),
                "silhouette": silhouette,
                "consensus_k": consensus_k,
                "consensus_strength": float(consensus_strength),
                "n_methods": len(methods_used),
                "participating_methods": methods_used,
                "method_type": "ensemble"
            }

        except Exception as e:
            print(f"Ensemble clustering failed: {e}")
            return {}

    # --------------------------- Fast Evaluation ---------------------------
    def _fast_evaluation(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Lightning-fast evaluation with essential metrics only"""
        evaluations = {}

        for method_name, result in all_results.items():
            labels = result.get("labels")
            if labels is None or len(labels) != len(data):
                continue

            try:
                labels_array = np.asarray(labels)
                unique_labels = set(labels_array)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = int((labels_array == -1).sum())

                eval_dict = {
                    "n_clusters": n_clusters,
                    "n_noise_points": n_noise,
                    "noise_ratio": float(n_noise / len(labels_array)),
                    "method_type": result.get("method_type", "unknown")
                }

                # Use pre-computed silhouette if available, otherwise compute
                if "silhouette" in result:
                    eval_dict["silhouette_score"] = result["silhouette"]
                elif n_clusters > 1:
                    mask = labels_array != -1
                    if mask.sum() > 1 and len(np.unique(labels_array[mask])) > 1:
                        eval_dict["silhouette_score"] = float(silhouette_score(data[mask], labels_array[mask]))

                # Quick cluster balance
                if n_clusters > 1:
                    non_noise_labels = labels_array[labels_array != -1]
                    if len(non_noise_labels) > 0:
                        cluster_sizes = np.bincount(non_noise_labels)
                        balance = 1 - (np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-10))
                        eval_dict["cluster_balance"] = float(balance)

                # Add method-specific metrics
                if "stability" in result:
                    eval_dict["stability_score"] = result["stability"]
                
                if "consensus_strength" in result:
                    eval_dict["consensus_strength"] = result["consensus_strength"]

                evaluations[method_name] = eval_dict

            except Exception as e:
                evaluations[method_name] = {"error": str(e)}

        return evaluations

    # --------------------------- Smart Recommendations ---------------------------
    def _smart_recommendations(
        self, all_results: Dict[str, Any], evaluations: Dict[str, Any],
        optimal_k_info: Dict[str, Any], preprocessing_info: Dict[str, Any]
    ) -> List[str]:
        """AI-powered recommendations based on results and data characteristics"""
        recs = []

        # Find best method using composite scoring
        method_scores = {}
        for method_name, eval_dict in evaluations.items():
            if "error" in eval_dict:
                continue

            score = 0.0
            
            # Silhouette score (primary metric)
            silhouette = eval_dict.get("silhouette_score", -1)
            if silhouette > 0:
                score += silhouette * 0.4

            # Cluster balance
            balance = eval_dict.get("cluster_balance", 0)
            score += balance * 0.2

            # Stability (if available)
            stability = eval_dict.get("stability_score", 0)
            score += stability * 0.2

            # Penalize excessive noise
            noise_ratio = eval_dict.get("noise_ratio", 0)
            score -= noise_ratio * 0.3

            # Bonus for ensemble methods
            if eval_dict.get("method_type") == "ensemble":
                score += 0.1

            # Bonus for appropriate cluster count
            n_clusters = eval_dict.get("n_clusters", 0)
            optimal_k = optimal_k_info.get("optimal_k", 3)
            if abs(n_clusters - optimal_k) <= 1:
                score += 0.1

            method_scores[method_name] = max(0, score)

        # Provide recommendations
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            
            recs.append(f"🏆 Best method: {best_method.upper().replace('_', ' ')} (score: {best_score:.3f})")
            
            best_eval = evaluations[best_method]
            best_result = all_results.get(best_method, {})
            
            # Method-specific insights
            method_type = best_eval.get("method_type", "")
            
            if "density" in method_type:
                noise_ratio = best_eval.get("noise_ratio", 0)
                if noise_ratio > 0.15:
                    recs.append("⚠️ High noise ratio - consider parameter tuning or data preprocessing")
                else:
                    recs.append("✅ Density-based method effectively separated noise from clusters")
            
            elif method_type == "probabilistic":
                if "mean_certainty" in best_result:
                    certainty = best_result["mean_certainty"]
                    if certainty > 0.8:
                        recs.append("📊 High assignment certainty - well-separated clusters")
                    else:
                        recs.append("🔀 Moderate assignment certainty - some cluster overlap detected")
            
            elif method_type == "ensemble":
                consensus_strength = best_eval.get("consensus_strength", 0)
                if consensus_strength > 0.7:
                    recs.append("🎯 Strong consensus among methods - highly reliable results")
                else:
                    recs.append("🤔 Moderate consensus - consider investigating disagreements")
            
            elif method_type == "spectral":
                recs.append("🌀 Spectral clustering succeeded - non-convex cluster shapes detected")

        # Data-specific recommendations
        if preprocessing_info.get("curse_of_dimensionality_risk", False):
            recs.append("📏 High dimensionality detected - PCA preprocessing recommended")

        outlier_pct = preprocessing_info.get("outlier_percentage", 0)
        if outlier_pct > 15:
            recs.append("🎯 Many outliers present - density-based methods recommended")
        elif outlier_pct < 2:
            recs.append("✨ Clean data - centroid-based methods should work well")

        # Optimal k insights
        confidence = optimal_k_info.get("confidence", "low")
        optimal_k = optimal_k_info.get("optimal_k", 3)
        
        if confidence == "high":
            recs.append(f"🎯 Strong evidence for {optimal_k} clusters")
        elif confidence == "medium":
            recs.append(f"🤔 Moderate evidence for {optimal_k} clusters - try range {optimal_k-1} to {optimal_k+1}")
        else:
            recs.append("❓ Unclear cluster structure - consider if clustering is appropriate")

        return recs[:5]  # Keep concise

    # --------------------------- Adaptive Main Analysis ---------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Adaptive clustering analysis with intelligent method selection"""
        try:
            # Lightning preprocessing
            X, preprocessing_info = self._lightning_preprocessing(data, config)
            n_samples, n_features = X.shape

            # Fast optimal k estimation
            optimal_k_info = self._fast_optimal_k(X, getattr(config, "max_clusters", 12))

            # Adaptive method selection based on data size
            clustering_results = {}

            # Always run KMeans (fastest and most reliable)
            try:
                kmeans_result = self._sota_kmeans_analysis(X, config, optimal_k_info)
                if kmeans_result:
                    clustering_results["kmeans"] = kmeans_result
            except Exception as e:
                print(f"SOTA KMeans failed: {e}")

            # Hierarchical for medium-sized datasets
            if n_samples <= self.medium_threshold:
                try:
                    hierarchical_result = self._fast_hierarchical_clustering(X, config, optimal_k_info)
                    if hierarchical_result:
                        clustering_results["hierarchical"] = hierarchical_result
                except Exception as e:
                    print(f"Fast hierarchical clustering failed: {e}")

            # Density-based methods
            if n_samples <= self.large_threshold:
                try:
                    density_results = self._smart_density_clustering(X, config)
                    clustering_results.update(density_results)
                except Exception as e:
                    print(f"Smart density clustering failed: {e}")

            # Probabilistic methods for appropriate sizes
            if n_samples <= self.medium_threshold:
                try:
                    prob_results = self._fast_probabilistic_clustering(X, config, optimal_k_info)
                    clustering_results.update(prob_results)
                except Exception as e:
                    print(f"Fast probabilistic clustering failed: {e}")

            # Advanced methods for smaller datasets
            if n_samples <= self.fast_threshold:
                try:
                    advanced_results = self._sota_advanced_clustering(X, config, optimal_k_info)
                    clustering_results.update(advanced_results)
                except Exception as e:
                    print(f"SOTA advanced clustering failed: {e}")

            # Lightning ensemble
            ensemble_result = self._lightning_ensemble(clustering_results, X)
            if ensemble_result:
                clustering_results["ensemble"] = ensemble_result

            # Fast evaluation
            evaluations = self._fast_evaluation(clustering_results, X)

            # Smart recommendations
            recommendations = self._smart_recommendations(
                clustering_results, evaluations, optimal_k_info, preprocessing_info
            )

            # Performance tier classification
            if n_samples < self.fast_threshold:
                perf_tier = "comprehensive"
                tier_desc = "All clustering methods available"
            elif n_samples < self.medium_threshold:
                perf_tier = "standard"
                tier_desc = "Core and probabilistic methods"
            elif n_samples < self.large_threshold:
                perf_tier = "fast"
                tier_desc = "Fast methods with subsampling"
            else:
                perf_tier = "ultra_fast"
                tier_desc = "Essential methods only"

            return {
                "clustering_results": clustering_results,
                "evaluations": evaluations,
                "optimal_k_analysis": optimal_k_info,
                "preprocessing_info": preprocessing_info,
                "data_characteristics": {
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                    "performance_tier": perf_tier,
                    "tier_description": tier_desc,
                    "data_variance": float(np.var(X)),
                    "data_spread": float(np.ptp(X))
                },
                "recommendations": recommendations,
                "summary": {
                    "methods_attempted": len(clustering_results),
                    "successful_methods": len([r for r in clustering_results.values() if "labels" in r]),
                    "best_method": (max(evaluations.keys(), 
                                      key=lambda k: evaluations[k].get("silhouette_score", -1))
                                   if evaluations else None),
                    "ensemble_available": "ensemble" in clustering_results,
                    "adaptive_selection": True
                },
                "performance_info": {
                    "subsampling_used": n_samples > self.max_sample_size,
                    "pca_applied": preprocessing_info.get("pca_applied", False),
                    "parallel_processing": True,
                    "optimization_level": "high"
                }
            }

        except Exception as e:
            return {
                "error": f"SOTA clustering analysis failed: {e}",
                "fallback_available": True,
                "recommendations": ["Consider data preprocessing", "Check data quality"]
            }
