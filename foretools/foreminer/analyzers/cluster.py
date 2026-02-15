import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import check_random_state

from .analyzer_utils import drop_constant_numeric, get_numeric_frame, safe_call
from ..core import AnalysisConfig, AnalysisStrategy


class ClusterAnalyzer(AnalysisStrategy):
    """SOTA clustering analysis with modern methods and adaptive selection"""

    @property
    def name(self) -> str:
        return "clusters"

    def __init__(self):
        self.fast_threshold = 2000
        self.medium_threshold = 10000
        self.max_sample_size = 5000

    def _rng(self, config: AnalysisConfig) -> np.random.Generator:
        rs = getattr(config, "random_state", 42)
        return np.random.default_rng(rs if rs is not None else 42)

    def _robust_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Modern preprocessing pipeline with adaptive scaling"""
        numeric = get_numeric_frame(data)
        if numeric.empty:
            raise ValueError("No numeric data available for clustering")

        # Clean infinite values and missing data
        numeric = numeric.dropna()
        if numeric.empty:
            raise ValueError("No complete rows after cleaning")

        info = {"original_shape": data.shape}

        # Remove constant columns
        numeric, keep_cols = drop_constant_numeric(numeric)
        if not keep_cols:
            raise ValueError("All numeric columns are constant")

        info["dropped_constant_cols"] = n_features - len(keep_cols)

        X = numeric.values.astype(float)
        
        # Robust outlier handling using IQR
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Clip extreme outliers
        X = np.clip(X, lower_bound, upper_bound)
        info["outlier_clipping"] = True
        
        # Adaptive scaling based on data characteristics
        n_samples, n_features = X.shape
        if n_features > n_samples // 2:  # High-dimensional case
            scaler = RobustScaler()
            info["scaling_method"] = "robust"
        else:
            scaler = StandardScaler()
            info["scaling_method"] = "standard"
            
        X = scaler.fit_transform(X)
        
        # Dimensionality reduction for high-dimensional data
        if n_features > 50 and n_samples > 100:
            # Use PCA to find intrinsic dimensionality
            pca_temp = PCA(random_state=getattr(config, "random_state", 42))
            pca_temp.fit(X)
            
            # Find elbow in explained variance
            cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
            # Use 90% variance or significant drop, whichever is smaller
            var_90_idx = np.argmax(cumvar >= 0.90) + 1
            
            # Look for elbow using second derivative
            if len(cumvar) > 3:
                second_deriv = np.diff(cumvar, 2)
                elbow_idx = np.argmax(second_deriv) + 2
                optimal_components = min(var_90_idx, elbow_idx, n_features // 2)
            else:
                optimal_components = var_90_idx
                
            if optimal_components < n_features * 0.7 and optimal_components >= 3:
                pca = PCA(n_components=optimal_components, random_state=getattr(config, "random_state", 42))
                X = pca.fit_transform(X)
                info["pca_applied"] = True
                info["pca_components"] = optimal_components
                info["variance_explained"] = float(cumvar[optimal_components - 1])
            else:
                info["pca_applied"] = False
        else:
            info["pca_applied"] = False
            
        info["final_shape"] = X.shape
        return X, info

    def _modern_optimal_k(self, X: np.ndarray, max_k: int = 15) -> Dict[str, Any]:
        """Streamlined optimal k estimation using only SOTA methods"""
        n_samples = X.shape[0]
        max_k = min(max_k, max(2, n_samples // 10))
        
        if max_k < 3:
            return {"optimal_k": 2, "methods": {}, "confidence": "low"}
            
        # Sample for efficiency
        if n_samples > 3000:
            rng = self._rng(AnalysisConfig())
            idx = rng.choice(n_samples, 3000, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
            
        methods = {}
        
        # 1. Silhouette Analysis (most reliable for diverse cluster shapes)
        try:
            sil_k = self._silhouette_analysis(X_sample, max_k)
            methods["silhouette"] = sil_k
        except Exception:
            pass
            
        # 2. BIC Model Selection (best for probabilistic interpretation)
        try:
            bic_k = self._bic_model_selection(X_sample, max_k)
            methods["bic_selection"] = bic_k
        except Exception:
            pass
            
        # Simple consensus (just 2 strong methods)
        if len(methods) == 2:
            # Give slightly more weight to silhouette as it's more general
            optimal_k = int(np.round(0.6 * methods["silhouette"] + 0.4 * methods["bic_selection"]))
            optimal_k = np.clip(optimal_k, 2, max_k)
            
            # High confidence if methods agree closely
            diff = abs(methods["silhouette"] - methods["bic_selection"])
            confidence = "high" if diff <= 1 else "medium" if diff <= 2 else "low"
            
        elif len(methods) == 1:
            optimal_k = list(methods.values())[0]
            confidence = "medium"
        else:
            # Fallback: use simple heuristic
            optimal_k = max(2, min(int(np.sqrt(n_samples / 2)), max_k))
            confidence = "low"
            
        return {
            "optimal_k": optimal_k,
            "methods": methods,
            "confidence": confidence,
            "primary_method": "silhouette_bic_consensus"
        }

    def _silhouette_analysis(self, X: np.ndarray, max_k: int) -> int:
        """Enhanced silhouette analysis - the gold standard for cluster validation"""
        best_k, best_score = 2, -1
        
        for k in range(2, max_k + 1):
            scores = []
            # Multiple runs for stability
            for _ in range(3):
                kmeans = KMeans(n_clusters=k, random_state=np.random.randint(1000), n_init=5)
                labels = kmeans.fit_predict(X)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    scores.append(score)
                    
            if scores:
                avg_score = np.mean(scores)
                stability = 1 - np.std(scores)  # Penalize unstable solutions
                combined_score = avg_score * stability
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_k = k
                    
        return best_k

    def _bic_model_selection(self, X: np.ndarray, max_k: int) -> int:
        """BIC-based model selection using GMM - excellent for probabilistic clusters"""
        best_k, best_bic = 2, np.inf
        
        for k in range(2, max_k + 1):
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, max_iter=100)
                gmm.fit(X)
                if gmm.converged_:
                    bic = gmm.bic(X)
                    if bic < best_bic:
                        best_bic = bic
                        best_k = k
            except:
                continue
                
        return best_k

    def _find_knee(self, y: np.ndarray) -> int:
        """Find knee point in curve using perpendicular distance method"""
        if len(y) < 3:
            return len(y) // 2
            
        # Normalize
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
        x_norm = np.linspace(0, 1, len(y))
        
        # Find point with maximum distance from line connecting start and end
        line_y = np.linspace(y_norm[0], y_norm[-1], len(y_norm))
        distances = np.abs(y_norm - line_y)
        
        return np.argmax(distances)

    def _advanced_kmeans(self, X: np.ndarray, config: AnalysisConfig, optimal_k: int) -> Dict[str, Any]:
        """Advanced K-means with multiple initialization strategies and validation"""
        n_samples = X.shape[0]
        rng = self._rng(config)
        
        # Adaptive parameter selection
        if n_samples > 10000:
            algorithm = "auto"
            n_init = 5
        else:
            algorithm = "lloyd"
            n_init = 20
            
        best_result = None
        best_score = -np.inf
        
        # Test different k values around optimal
        k_range = range(max(2, optimal_k - 1), min(optimal_k + 3, n_samples // 5))
        
        for k in k_range:
            # Multiple runs with different initializations
            results = []
            
            for init_method in ["k-means++", "random"]:
                try:
                    kmeans = KMeans(
                        n_clusters=k,
                        init=init_method,
                        n_init=n_init,
                        max_iter=300,
                        algorithm=algorithm,
                        random_state=getattr(config, "random_state", 42)
                    )
                    labels = kmeans.fit_predict(X)
                    
                    # Comprehensive scoring
                    sil_score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
                    ch_score = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0
                    db_score = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.inf
                    
                    # Cluster balance
                    sizes = np.bincount(labels)
                    balance = 1 - (np.std(sizes) / (np.mean(sizes) + 1e-10))
                    
                    # Combined score
                    score = (sil_score * 0.4 + 
                            min(ch_score / 1000, 1) * 0.3 + 
                            (1 - min(db_score / 10, 1)) * 0.2 + 
                            balance * 0.1)
                    
                    results.append({
                        "model": kmeans,
                        "labels": labels,
                        "score": score,
                        "silhouette": sil_score,
                        "calinski_harabasz": ch_score,
                        "davies_bouldin": db_score,
                        "balance": balance,
                        "inertia": kmeans.inertia_,
                        "k": k
                    })
                    
                except Exception:
                    continue
                    
            # Select best from this k
            if results:
                best_for_k = max(results, key=lambda x: x["score"])
                if best_for_k["score"] > best_score:
                    best_score = best_for_k["score"]
                    best_result = best_for_k
                    
        if best_result is None:
            # Fallback
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(X)
            best_result = {
                "model": kmeans,
                "labels": labels,
                "k": 2,
                "silhouette": silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
            }
            
        # Add final metrics
        best_result.update({
            "n_clusters": len(np.unique(best_result["labels"])),
            "cluster_sizes": dict(Counter(best_result["labels"])),
            "centers": best_result["model"].cluster_centers_,
            "method_type": "centroid_based"
        })
        
        return best_result

    def _modern_spectral_clustering(self, X: np.ndarray, config: AnalysisConfig, optimal_k: int) -> Dict[str, Any]:
        """Advanced spectral clustering with adaptive affinity"""
        n_samples = X.shape[0]
        
        if n_samples > 2000:
            return {}  # Too expensive for large datasets
            
        try:
            # Adaptive affinity matrix construction
            if n_samples < 500:
                # Use RBF kernel with adaptive gamma
                from sklearn.metrics.pairwise import rbf_kernel

                # Estimate optimal gamma using median heuristic
                sample_size = min(200, n_samples)
                sample_idx = np.random.choice(n_samples, sample_size, replace=False)
                sample = X[sample_idx]
                
                distances = pdist(sample)
                if len(distances) > 0:
                    gamma = 1.0 / (2 * np.median(distances) ** 2)
                else:
                    gamma = 1.0
                    
                affinity = rbf_kernel(X, gamma=gamma)
            else:
                # Use k-NN graph for efficiency
                k_neighbors = max(10, min(30, n_samples // 20))
                affinity = kneighbors_graph(X, n_neighbors=k_neighbors, mode='connectivity')
                
            # Spectral clustering
            spectral = SpectralClustering(
                n_clusters=optimal_k,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=getattr(config, "random_state", 42)
            )
            
            labels = spectral.fit_predict(affinity)
            
            # Compute metrics
            n_clusters = len(np.unique(labels))
            if n_clusters > 1:
                sil_score = silhouette_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
                db_score = davies_bouldin_score(X, labels)
            else:
                sil_score = ch_score = 0
                db_score = np.inf
                
            return {
                "labels": labels,
                "model": spectral,
                "n_clusters": n_clusters,
                "silhouette": sil_score,
                "calinski_harabasz": ch_score,
                "davies_bouldin": db_score,
                "cluster_sizes": dict(Counter(labels)),
                "method_type": "spectral",
                "affinity_type": "rbf" if n_samples < 500 else "knn"
            }
            
        except Exception:
            return {}

    def _modern_density_clustering(self, X: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        """Advanced density-based clustering with parameter optimization"""
        results = {}
        n_samples, n_features = X.shape
        
        # DBSCAN with optimized parameters
        try:
            # Adaptive parameter selection
            k = max(4, min(2 * n_features, n_samples // 50, 20))
            
            # Use subset for parameter estimation if dataset is large
            if n_samples > 5000:
                sample_idx = np.random.choice(n_samples, 5000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
                
            # Find optimal eps using k-distance graph
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X_sample)
            distances, _ = nn.kneighbors(X_sample)
            
            k_distances = np.sort(distances[:, k-1])
            
            # Use knee detection for eps
            eps_idx = self._find_knee(k_distances)
            eps = k_distances[eps_idx]
            
            # Ensure eps is reasonable
            if eps <= 0 or not np.isfinite(eps):
                eps = np.percentile(k_distances, 95)
                
            min_samples = max(3, k // 2)
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            result = {
                "labels": labels,
                "model": dbscan,
                "n_clusters": n_clusters,
                "noise_points": n_noise,
                "noise_ratio": n_noise / len(labels),
                "eps": eps,
                "min_samples": min_samples,
                "cluster_sizes": dict(Counter([l for l in labels if l != -1])),
                "method_type": "density_based"
            }
            
            # Add silhouette score if meaningful clustering found
            if n_clusters > 1 and n_noise < len(labels) * 0.8:
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    result["silhouette"] = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                    
            results["dbscan"] = result
            
        except Exception:
            pass
            
        # HDBSCAN if available
        try:
            import hdbscan
            
            if n_samples <= 5000:  # Computational limit
                min_cluster_size = max(5, n_samples // 100)
                min_samples = max(1, min_cluster_size // 3)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method='eom',
                    core_dist_n_jobs=1
                )
                
                labels = clusterer.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                result = {
                    "labels": labels,
                    "model": clusterer,
                    "n_clusters": n_clusters,
                    "noise_points": n_noise,
                    "noise_ratio": n_noise / len(labels),
                    "min_cluster_size": min_cluster_size,
                    "cluster_sizes": dict(Counter([l for l in labels if l != -1])),
                    "method_type": "hierarchical_density"
                }
                
                if hasattr(clusterer, 'probabilities_'):
                    result["probabilities"] = clusterer.probabilities_
                    
                if n_clusters > 1 and n_noise < len(labels) * 0.8:
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        result["silhouette"] = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                        
                results["hdbscan"] = result
                
        except ImportError:
            pass
        except Exception:
            pass
            
        return results

    def _gaussian_mixture_clustering(self, X: np.ndarray, config: AnalysisConfig, optimal_k: int) -> Dict[str, Any]:
        """Advanced Gaussian Mixture Model clustering with model selection"""
        results = {}
        n_samples, n_features = X.shape
        
        # Limit complexity for high-dimensional or large datasets
        if n_features > 50 or n_samples > 10000:
            covariance_types = ["diag", "spherical"]
        else:
            covariance_types = ["full", "tied", "diag", "spherical"]
            
        max_components = min(15, n_samples // 10, optimal_k + 3)
        k_range = range(2, max_components + 1)
        
        best_gmm = None
        best_bic = np.inf
        best_labels = None
        best_config = None
        
        try:
            for covariance_type in covariance_types:
                for n_components in k_range:
                    try:
                        gmm = GaussianMixture(
                            n_components=n_components,
                            covariance_type=covariance_type,
                            max_iter=100,
                            n_init=3,
                            random_state=getattr(config, "random_state", 42),
                            reg_covar=1e-6
                        )
                        
                        gmm.fit(X)
                        
                        if gmm.converged_:
                            bic = gmm.bic(X)
                            aic = gmm.aic(X)
                            
                            # Prefer simpler models when BIC is close
                            penalized_bic = bic + (n_components - 2) * 10
                            
                            if penalized_bic < best_bic:
                                best_bic = penalized_bic
                                best_gmm = gmm
                                best_labels = gmm.predict(X)
                                best_config = (n_components, covariance_type, bic, aic)
                                
                    except Exception:
                        continue
                        
            if best_gmm is not None:
                n_components, cov_type, bic, aic = best_config
                probabilities = best_gmm.predict_proba(X)
                
                # Calculate uncertainty metrics
                max_probs = np.max(probabilities, axis=1)
                mean_certainty = np.mean(max_probs)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                mean_entropy = np.mean(entropy)
                
                result = {
                    "labels": best_labels,
                    "probabilities": probabilities,
                    "model": best_gmm,
                    "n_clusters": len(np.unique(best_labels)),
                    "bic": bic,
                    "aic": aic,
                    "covariance_type": cov_type,
                    "mean_certainty": mean_certainty,
                    "mean_entropy": mean_entropy,
                    "cluster_sizes": dict(Counter(best_labels)),
                    "method_type": "probabilistic"
                }
                
                if len(np.unique(best_labels)) > 1:
                    result["silhouette"] = silhouette_score(X, best_labels)
                    result["calinski_harabasz"] = calinski_harabasz_score(X, best_labels)
                    result["davies_bouldin"] = davies_bouldin_score(X, best_labels)
                    
                results["gaussian_mixture"] = result
                
        except Exception:
            pass
            
        return results

    def _ensemble_clustering(self, all_results: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
        """Modern ensemble clustering with consensus optimization"""
        if len(all_results) < 2:
            return {}
            
        # Filter valid results
        valid_results = {}
        for name, result in all_results.items():
            if isinstance(result, dict) and "labels" in result:
                labels = np.asarray(result["labels"])
                if len(labels) == len(X):
                    # Filter out degenerate clusterings
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    noise_ratio = np.sum(labels == -1) / len(labels) if -1 in labels else 0
                    
                    if n_clusters >= 2 and noise_ratio < 0.8:
                        valid_results[name] = labels
                        
        if len(valid_results) < 2:
            return {}
            
        # Consensus clustering using co-occurrence matrix
        n_samples = len(X)
        co_occurrence = np.zeros((n_samples, n_samples))
        
        for labels in valid_results.values():
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    # Count if points are in same cluster (excluding noise)
                    if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                        co_occurrence[i, j] += 1
                        co_occurrence[j, i] += 1
                        
        # Normalize co-occurrence matrix
        co_occurrence = co_occurrence / len(valid_results)
        
        # Use spectral clustering on co-occurrence matrix for final clustering
        try:
            # Estimate number of clusters from co-occurrence structure
            eigenvals = np.linalg.eigvals(co_occurrence)
            eigenvals = np.real(eigenvals[eigenvals > 1e-10])
            eigenvals = np.sort(eigenvals)[::-1]
            
            # Find eigengap
            if len(eigenvals) > 3:
                gaps = np.diff(eigenvals)
                n_clusters_est = np.argmax(gaps) + 1
                n_clusters_est = max(2, min(n_clusters_est, len(eigenvals) // 2))
            else:
                n_clusters_est = 2
                
            # Apply spectral clustering
            from sklearn.cluster import SpectralClustering
            spectral = SpectralClustering(
                n_clusters=n_clusters_est,
                affinity='precomputed',
                random_state=42
            )
            
            consensus_labels = spectral.fit_predict(co_occurrence)
            
            # Calculate consensus metrics
            consensus_scores = []
            for labels in valid_results.values():
                try:
                    ari = adjusted_rand_score(consensus_labels, labels)
                    consensus_scores.append(ari)
                except:
                    continue
                    
            mean_consensus = np.mean(consensus_scores) if consensus_scores else 0
            
            result = {
                "labels": consensus_labels,
                "n_clusters": len(np.unique(consensus_labels)),
                "cluster_sizes": dict(Counter(consensus_labels)),
                "consensus_strength": mean_consensus,
                "n_methods": len(valid_results),
                "participating_methods": list(valid_results.keys()),
                "method_type": "ensemble"
            }
            
            # Add standard metrics
            if len(np.unique(consensus_labels)) > 1:
                result["silhouette"] = silhouette_score(X, consensus_labels)
                result["calinski_harabasz"] = calinski_harabasz_score(X, consensus_labels)
                result["davies_bouldin"] = davies_bouldin_score(X, consensus_labels)
                
            return result
            
        except Exception:
            # Fallback to majority voting
            return self._majority_vote_ensemble(valid_results, X)

    def _majority_vote_ensemble(self, valid_results: Dict[str, np.ndarray], X: np.ndarray) -> Dict[str, Any]:
        """Fallback ensemble method using majority voting with label alignment"""
        if len(valid_results) < 2:
            return {}
            
        # Align all labelings to the first one
        methods = list(valid_results.keys())
        reference_labels = valid_results[methods[0]]
        aligned_results = {methods[0]: reference_labels}
        
        for method in methods[1:]:
            labels = valid_results[method]
            try:
                aligned_labels = self._align_labels_hungarian(labels, reference_labels)
                aligned_results[method] = aligned_labels
            except:
                continue
                
        if len(aligned_results) < 2:
            return {}
            
        # Majority voting
        n_samples = len(X)
        final_labels = np.full(n_samples, -1, dtype=int)
        
        for i in range(n_samples):
            votes = {}
            for labels in aligned_results.values():
                label = labels[i]
                if label != -1:  # Ignore noise votes
                    votes[label] = votes.get(label, 0) + 1
                    
            if votes:
                final_labels[i] = max(votes, key=votes.get)
                
        # Remove empty clusters and relabel
        unique_labels = np.unique(final_labels[final_labels != -1])
        label_map = {old: new for new, old in enumerate(unique_labels)}
        
        for i in range(n_samples):
            if final_labels[i] in label_map:
                final_labels[i] = label_map[final_labels[i]]
                
        if len(unique_labels) < 2:
            return {}
            
        result = {
            "labels": final_labels,
            "n_clusters": len(unique_labels),
            "cluster_sizes": dict(Counter(final_labels[final_labels != -1])),
            "n_methods": len(aligned_results),
            "participating_methods": list(aligned_results.keys()),
            "method_type": "ensemble"
        }
        
        if len(np.unique(final_labels)) > 1:
            result["silhouette"] = silhouette_score(X, final_labels)
            
        return result

    def _align_labels_hungarian(self, labels_to_align: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
        """Align cluster labels using Hungarian algorithm"""
        # Only consider non-noise points
        mask = (labels_to_align != -1) & (reference_labels != -1)
        if not np.any(mask):
            return labels_to_align.copy()
            
        la_clean = labels_to_align[mask]
        ref_clean = reference_labels[mask]
        
        la_unique = np.unique(la_clean)
        ref_unique = np.unique(ref_clean)
        
        if len(la_unique) == 0 or len(ref_unique) == 0:
            return labels_to_align.copy()
            
        # Build confusion matrix
        confusion = np.zeros((len(la_unique), len(ref_unique)))
        for i, la_label in enumerate(la_unique):
            for j, ref_label in enumerate(ref_unique):
                confusion[i, j] = np.sum((la_clean == la_label) & (ref_clean == ref_label))
                
        # Hungarian algorithm
        row_idx, col_idx = linear_sum_assignment(-confusion)
        
        # Create mapping
        mapping = {}
        for i, j in zip(row_idx, col_idx):
            mapping[la_unique[i]] = ref_unique[j]
            
        # Apply mapping
        aligned = labels_to_align.copy()
        for old_label, new_label in mapping.items():
            aligned[labels_to_align == old_label] = new_label
            
        return aligned

    def _evaluate_clustering(self, all_results: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation of all clustering results"""
        evaluations = {}
        
        for method, result in all_results.items():
            if not isinstance(result, dict) or "labels" not in result:
                continue
                
            try:
                labels = np.asarray(result["labels"])
                if len(labels) != len(X):
                    continue
                    
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = np.sum(labels == -1) if -1 in labels else 0
                
                eval_result = {
                    "n_clusters": n_clusters,
                    "n_noise_points": int(n_noise),
                    "noise_ratio": float(n_noise / len(labels)),
                    "method_type": result.get("method_type", "unknown")
                }
                
                # Internal validation metrics
                if n_clusters > 1:
                    # Handle noise points
                    if n_noise > 0:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1 and len(np.unique(labels[non_noise_mask])) > 1:
                            X_clean = X[non_noise_mask]
                            labels_clean = labels[non_noise_mask]
                            
                            eval_result["silhouette_score"] = float(silhouette_score(X_clean, labels_clean))
                            eval_result["calinski_harabasz_score"] = float(calinski_harabasz_score(X_clean, labels_clean))
                            eval_result["davies_bouldin_score"] = float(davies_bouldin_score(X_clean, labels_clean))
                    else:
                        eval_result["silhouette_score"] = float(silhouette_score(X, labels))
                        eval_result["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))
                        eval_result["davies_bouldin_score"] = float(davies_bouldin_score(X, labels))
                        
                # Cluster balance metric
                if n_clusters > 1:
                    non_noise_labels = labels[labels != -1]
                    if len(non_noise_labels) > 0:
                        cluster_sizes = np.bincount(non_noise_labels)
                        if len(cluster_sizes) > 1:
                            eval_result["cluster_balance"] = float(1 - (np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-10)))
                            
                # Method-specific metrics
                if "consensus_strength" in result:
                    eval_result["consensus_strength"] = float(result["consensus_strength"])
                    
                if "mean_certainty" in result:
                    eval_result["mean_certainty"] = float(result["mean_certainty"])
                    
                if "stability" in result:
                    eval_result["stability_score"] = float(result["stability"])
                    
                evaluations[method] = eval_result
                
            except Exception as e:
                evaluations[method] = {"error": str(e)}
                
        return evaluations

    def _generate_recommendations(self, 
                                 all_results: Dict[str, Any], 
                                 evaluations: Dict[str, Any],
                                 optimal_k_info: Dict[str, Any],
                                 preprocessing_info: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on results"""
        recommendations = []
        
        # Find best method based on composite score
        method_scores = {}
        for method, eval_data in evaluations.items():
            if "error" in eval_data:
                continue
                
            score = 0.0
            
            # Silhouette score (primary metric)
            sil = eval_data.get("silhouette_score", -1)
            if sil > 0:
                score += sil * 0.4
                
            # Cluster balance
            balance = eval_data.get("cluster_balance", 0)
            score += balance * 0.25
            
            # Low noise ratio is good
            noise_ratio = eval_data.get("noise_ratio", 1)
            score += (1 - noise_ratio) * 0.2
            
            # Consensus strength for ensemble methods
            consensus = eval_data.get("consensus_strength", 0)
            score += consensus * 0.15
            
            method_scores[method] = max(0, score)
            
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            
            recommendations.append(f"Best performing method: {best_method.upper().replace('_', ' ')} (score: {best_score:.3f})")
            
            # Method-specific insights
            best_eval = evaluations[best_method]
            method_type = best_eval.get("method_type", "")
            
            if "density" in method_type:
                noise_ratio = best_eval.get("noise_ratio", 0)
                if noise_ratio > 0.2:
                    recommendations.append("High noise detected - data may have outliers or natural noise structure")
                else:
                    recommendations.append("Density-based clustering found clear cluster boundaries")
                    
            elif method_type == "probabilistic":
                certainty = all_results.get(best_method, {}).get("mean_certainty", 0)
                if certainty > 0.8:
                    recommendations.append("High assignment certainty indicates well-separated clusters")
                elif certainty < 0.6:
                    recommendations.append("Low certainty suggests overlapping or fuzzy cluster boundaries")
                    
            elif method_type == "ensemble":
                consensus = best_eval.get("consensus_strength", 0)
                if consensus > 0.7:
                    recommendations.append("Strong consensus across methods validates cluster structure")
                else:
                    recommendations.append("Moderate consensus - consider inspecting individual method results")
                    
            elif method_type == "spectral":
                recommendations.append("Spectral clustering success suggests non-convex cluster shapes")
                
        # Data-specific recommendations
        optimal_k = optimal_k_info.get("optimal_k", 3)
        confidence = optimal_k_info.get("confidence", "low")
        
        if confidence == "high":
            recommendations.append(f"Strong evidence for {optimal_k} clusters across multiple methods")
        elif confidence == "medium":
            recommendations.append(f"Moderate evidence for {optimal_k} clusters - consider validation")
        else:
            recommendations.append("Unclear cluster structure - may need feature engineering or different approach")
            
        # Preprocessing insights
        if preprocessing_info.get("pca_applied", False):
            variance_explained = preprocessing_info.get("variance_explained", 0)
            recommendations.append(f"Dimensionality reduced via PCA ({variance_explained:.1%} variance retained)")
            
        if preprocessing_info.get("outlier_clipping", False):
            recommendations.append("Outliers detected and handled during preprocessing")
            
        return recommendations[:6]  # Limit to most important recommendations

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Main analysis pipeline with modern clustering methods"""
        try:
            # Preprocessing
            X, preprocessing_info = self._robust_preprocessing(data, config)
            n_samples, n_features = X.shape
            
            # Optimal k estimation
            optimal_k_info = self._modern_optimal_k(X, getattr(config, "max_clusters", 15))
            optimal_k = optimal_k_info["optimal_k"]
            
            clustering_results = {}
            method_errors: Dict[str, str] = {}
            
            # Core method: Advanced K-means (always run)
            kmeans_result, err = safe_call(self._advanced_kmeans, X, config, optimal_k)
            if kmeans_result:
                clustering_results["kmeans"] = kmeans_result
            elif err:
                method_errors["kmeans"] = err
                
            # Density-based clustering (for most datasets)
            if n_samples <= 20000:
                density_results, err = safe_call(
                    self._modern_density_clustering, X, config, default={}
                )
                if density_results:
                    clustering_results.update(density_results)
                elif err:
                    method_errors["density"] = err
                    
            # Gaussian Mixture Models (for medium datasets)
            if n_samples <= 10000:
                gmm_results, err = safe_call(
                    self._gaussian_mixture_clustering, X, config, optimal_k, default={}
                )
                if gmm_results:
                    clustering_results.update(gmm_results)
                elif err:
                    method_errors["gaussian_mixture"] = err
                    
            # Spectral clustering (for smaller datasets)
            if n_samples <= 2000:
                spectral_result, err = safe_call(
                    self._modern_spectral_clustering, X, config, optimal_k
                )
                if spectral_result:
                    clustering_results["spectral"] = spectral_result
                elif err:
                    method_errors["spectral"] = err
                    
            # Ensemble clustering (if multiple methods succeeded)
            if len(clustering_results) >= 2:
                ensemble_result, err = safe_call(
                    self._ensemble_clustering, clustering_results, X
                )
                if ensemble_result:
                    clustering_results["ensemble"] = ensemble_result
                elif err:
                    method_errors["ensemble"] = err
                    
            # Evaluation and recommendations
            evaluations = self._evaluate_clustering(clustering_results, X)
            recommendations = self._generate_recommendations(clustering_results, evaluations, optimal_k_info, preprocessing_info)
            
            # Determine performance tier
            if n_samples <= 2000:
                perf_tier = "comprehensive"
                tier_desc = "All advanced methods available"
            elif n_samples <= 10000:
                perf_tier = "standard"
                tier_desc = "Core methods with probabilistic clustering"
            else:
                perf_tier = "fast"
                tier_desc = "Scalable methods for large datasets"
                
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
                    "data_complexity": "high" if n_features > n_samples // 5 else "medium" if n_features > 10 else "low"
                },
                "recommendations": recommendations,
                "method_errors": method_errors,
                "summary": {
                    "methods_attempted": len(clustering_results),
                    "successful_methods": len([r for r in clustering_results.values() if "labels" in r]),
                    "best_method": (max(evaluations.keys(), 
                                      key=lambda k: evaluations[k].get("silhouette_score", -1)) 
                                   if evaluations else None),
                    "ensemble_available": "ensemble" in clustering_results,
                    "modern_methods": True
                }
            }
            
        except Exception as e:
            return {
                "error": f"Modern clustering analysis failed: {str(e)}",
                "fallback_available": False,
                "recommendations": [
                    "Verify data quality and format",
                    "Check for sufficient numeric features", 
                    "Consider data preprocessing",
                    "Ensure adequate sample size"
                ]
            }
