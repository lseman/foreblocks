from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import (NMF, PCA, FactorAnalysis, FastICA,
                                   TruncatedSVD)
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import (TSNE, Isomap, LocallyLinearEmbedding,
                              SpectralEmbedding)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import (PowerTransformer, RobustScaler,
                                   StandardScaler)

from .foreminer_aux import *


class DimensionalityAnalyzer(AnalysisStrategy):
    """State-of-the-art dimensionality reduction with adaptive preprocessing and ensemble methods"""

    @property
    def name(self) -> str:
        return "dimensionality"

    def _adaptive_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            index=numeric_data.index,
        )
        preprocessing_info["features_removed"] = (~var_threshold.get_support()).sum()

        # Adaptive sampling with stratification if possible
        if len(numeric_data) > config.sample_size_threshold:
            # Try to preserve data distribution
            try:
                # Simple clustering-based stratified sampling
                kmeans = KMeans(
                    n_clusters=min(10, len(numeric_data) // 100),
                    random_state=config.random_state,
                )
                clusters = kmeans.fit_predict(
                    StandardScaler().fit_transform(numeric_data)
                )

                sample_indices = []
                for cluster_id in np.unique(clusters):
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    n_samples = max(
                        1,
                        int(
                            config.sample_size_threshold
                            * len(cluster_indices)
                            / len(numeric_data)
                        ),
                    )
                    sample_indices.extend(
                        np.random.choice(
                            cluster_indices,
                            min(n_samples, len(cluster_indices)),
                            replace=False,
                        )
                    )

                numeric_data = numeric_data.iloc[
                    sample_indices[: config.sample_size_threshold]
                ]
                preprocessing_info["sampling_method"] = "stratified"
            except:
                # Fallback to random sampling
                sample_idx = np.random.choice(
                    len(numeric_data), config.sample_size_threshold, replace=False
                )
                numeric_data = numeric_data.iloc[sample_idx]
                preprocessing_info["sampling_method"] = "random"

        # Intelligent scaling method selection
        scaling_methods = []

        # Check for normality and outliers
        skewness = numeric_data.skew().abs().mean()
        outlier_fraction = (
            (numeric_data - numeric_data.mean()).abs() > 3 * numeric_data.std()
        ).sum().sum() / numeric_data.size

        if skewness > 2 or outlier_fraction > 0.1:
            # Heavy tails or many outliers - use robust methods
            scaling_methods = [
                (
                    "power_transform",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                ("robust", RobustScaler()),
                ("standard", StandardScaler()),
            ]
        else:
            # Well-behaved data
            scaling_methods = [
                ("standard", StandardScaler()),
                ("robust", RobustScaler()),
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
                    preprocessing_info["scaling_method"] = name
            except:
                continue

        if best_scaler is None:
            best_scaler = StandardScaler()
            preprocessing_info["scaling_method"] = "standard_fallback"

        scaled_data = best_scaler.fit_transform(numeric_data)
        preprocessing_info["final_shape"] = scaled_data.shape

        return scaled_data, preprocessing_info

    def _compute_optimal_components(
        self, scaled_data: np.ndarray, max_components: int = 10
    ) -> int:
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
        criteria_results["var_95"] = np.argmax(cumvar >= 0.95) + 1

        # 2. Elbow method on eigenvalues
        eigenvals = pca_full.explained_variance_
        if len(eigenvals) >= 3:
            diffs = np.diff(eigenvals)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                criteria_results["elbow"] = np.argmax(second_diffs) + 2

        # 3. Kaiser criterion (eigenvalues > 1)
        criteria_results["kaiser"] = np.sum(eigenvals > 1)

        # 4. Parallel analysis approximation
        random_eigenvals = []
        for _ in range(5):  # Reduced iterations for speed
            random_data = np.random.randn(*scaled_data.shape)
            random_pca = PCA().fit(random_data)
            random_eigenvals.append(random_pca.explained_variance_)

        mean_random_eigenvals = np.mean(random_eigenvals, axis=0)
        criteria_results["parallel"] = np.sum(
            eigenvals > mean_random_eigenvals[: len(eigenvals)]
        )

        # Combine criteria with weights
        valid_results = {
            k: v for k, v in criteria_results.items() if v > 0 and v <= max_comp
        }

        if valid_results:
            # Weighted average, favoring conservative estimates
            weights = {"var_95": 0.3, "elbow": 0.2, "kaiser": 0.2, "parallel": 0.3}
            weighted_sum = sum(
                weights.get(k, 0.25) * v for k, v in valid_results.items()
            )
            optimal = max(2, min(max_comp, int(np.round(weighted_sum))))
        else:
            optimal = min(3, max_comp)

        return optimal

    def _enhanced_dimensionality_reduction(
        self, scaled_data: np.ndarray, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Apply state-of-the-art dimensionality reduction techniques"""
        results = {}
        n_samples, n_features = scaled_data.shape
        optimal_components = self._compute_optimal_components(scaled_data)

        # Linear methods
        linear_methods = {
            "pca": PCA(
                n_components=optimal_components, random_state=config.random_state
            ),
            "ica": FastICA(
                n_components=optimal_components,
                random_state=config.random_state,
                max_iter=1000,
                tol=1e-4,
            ),
            "factor_analysis": FactorAnalysis(
                n_components=optimal_components, random_state=config.random_state
            ),
        }

        # Add NMF for non-negative data
        if (scaled_data >= 0).all():
            linear_methods["nmf"] = NMF(
                n_components=optimal_components,
                random_state=config.random_state,
                max_iter=1000,
                tol=1e-4,
            )

        # SVD for high-dimensional sparse data
        if n_features > 100:
            linear_methods["truncated_svd"] = TruncatedSVD(
                n_components=optimal_components, random_state=config.random_state
            )

        for name, method in linear_methods.items():
            try:
                embedding = method.fit_transform(scaled_data)
                results[name] = {
                    "embedding": embedding,
                    "method_type": "linear",
                    "n_components": optimal_components,
                }

                # Add explained variance for methods that support it
                if hasattr(method, "explained_variance_ratio_"):
                    results[name][
                        "explained_variance_ratio"
                    ] = method.explained_variance_ratio_
                    results[name][
                        "total_variance_explained"
                    ] = method.explained_variance_ratio_.sum()

            except Exception as e:
                print(f"Linear method {name} failed: {e}")

        # Non-linear methods with adaptive parameters
        target_dim = 2  # For visualization
        perplexity = min(30, max(5, n_samples // 4))
        n_neighbors = min(15, max(3, n_samples // 10))

        # Standard t-SNE
        tsne_params = {
            "n_components": target_dim,
            "random_state": config.random_state,
            "perplexity": perplexity,
            "init": "pca",
            "learning_rate": "auto",
            "n_iter": 1000,
            "early_exaggeration": 12,
            "min_grad_norm": 1e-7,
        }

        # Use best available t-SNE implementation
        if HAS_OPENTSNE and n_samples > 1000:
            try:
                tsne = OpenTSNE(**tsne_params, n_jobs=4, negative_sample_rate=5)
                results["tsne_openai"] = {
                    "embedding": tsne.fit(scaled_data),
                    "method_type": "nonlinear",
                    "perplexity": perplexity,
                }
            except Exception as e:
                print(f"OpenTSNE failed: {e}")

        elif HAS_MULTICORE_TSNE:
            try:
                tsne = MulticoreTSNE(n_jobs=4, **tsne_params)
                results["tsne_multicore"] = {
                    "embedding": tsne.fit_transform(scaled_data),
                    "method_type": "nonlinear",
                    "perplexity": perplexity,
                }
            except Exception as e:
                print(f"MulticoreTSNE failed: {e}")

        # Fallback to standard t-SNE
        if not any("tsne" in k for k in results.keys()):
            try:
                tsne = TSNE(**tsne_params)
                results["tsne"] = {
                    "embedding": tsne.fit_transform(scaled_data),
                    "method_type": "nonlinear",
                    "perplexity": perplexity,
                }
            except Exception as e:
                print(f"Standard t-SNE failed: {e}")

        # UMAP with optimized parameters
        if HAS_UMAP:
            try:
                umap_params = {
                    "n_components": target_dim,
                    "random_state": config.random_state,
                    "n_neighbors": n_neighbors,
                    "min_dist": 0.1,
                    "metric": "euclidean",
                    "spread": 1.0,
                    "low_memory": n_samples > 10000,
                    "n_epochs": None,  # Auto-determine
                    "learning_rate": 1.0,
                    "repulsion_strength": 1.0,
                }

                umap_reducer = UMAP(**umap_params)
                results["umap"] = {
                    "embedding": umap_reducer.fit_transform(scaled_data),
                    "method_type": "nonlinear",
                    "n_neighbors": n_neighbors,
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
                    n_iters=1200,
                ).fit_transform(scaled_data)

                results["trimap"] = {
                    "embedding": trimap_embedding,
                    "method_type": "nonlinear",
                }
            except Exception as e:
                print(f"TriMap failed: {e}")

        # Other manifold learning methods
        manifold_methods = {
            "isomap": Isomap(n_components=target_dim, n_neighbors=n_neighbors),
            "lle": LocallyLinearEmbedding(
                n_components=target_dim,
                n_neighbors=n_neighbors,
                random_state=config.random_state,
                method="standard",
            ),
            "spectral": SpectralEmbedding(
                n_components=target_dim,
                random_state=config.random_state,
                n_neighbors=n_neighbors,
            ),
        }

        for name, method in manifold_methods.items():
            try:
                embedding = method.fit_transform(scaled_data)
                results[name] = {
                    "embedding": embedding,
                    "method_type": "nonlinear",
                    "n_neighbors": getattr(method, "n_neighbors", None),
                }
            except Exception as e:
                print(f"Manifold method {name} failed: {e}")

        return results

    def _evaluate_embeddings(
        self, embeddings: Dict[str, Any], scaled_data: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate embedding quality using multiple metrics"""
        evaluation_results = {}

        for name, result in embeddings.items():
            if "embedding" not in result:
                continue

            embedding = result["embedding"]
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
                        metrics["silhouette_score"] = silhouette_score(
                            embedding, cluster_labels
                        )
            except:
                pass

            try:
                # Neighborhood preservation (for nonlinear methods)
                if (
                    result.get("method_type") == "nonlinear"
                    and scaled_data.shape[0] <= 1000
                ):
                    from scipy.spatial.distance import pdist, squareform
                    from scipy.stats import spearmanr

                    # Sample for computational efficiency
                    n_sample = min(200, embedding.shape[0])
                    indices = np.random.choice(
                        embedding.shape[0], n_sample, replace=False
                    )

                    orig_dist = squareform(pdist(scaled_data[indices]))
                    embed_dist = squareform(pdist(embedding[indices]))

                    # Spearman correlation between distance matrices
                    correlation, _ = spearmanr(
                        orig_dist.flatten(), embed_dist.flatten()
                    )
                    metrics["neighborhood_preservation"] = correlation
            except:
                pass

            try:
                # Local continuity
                embedding_std = np.std(embedding, axis=0)
                metrics["embedding_stability"] = 1.0 / (1.0 + np.mean(embedding_std))
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
                return {"error": "All dimensionality reduction methods failed"}

            # Evaluate embedding quality
            evaluation_results = self._evaluate_embeddings(embeddings, scaled_data)

            # Compile final results
            results = {
                "embeddings": embeddings,
                "evaluation": evaluation_results,
                "preprocessing_info": preprocessing_info,
                "data_characteristics": {
                    "n_samples": scaled_data.shape[0],
                    "n_features": scaled_data.shape[1],
                    "condition_number": np.linalg.cond(scaled_data.T @ scaled_data),
                    "effective_rank": np.linalg.matrix_rank(scaled_data),
                },
                "recommendations": self._generate_recommendations(
                    embeddings, evaluation_results, preprocessing_info
                ),
            }

            return results

        except Exception as e:
            return {"error": f"Dimensionality analysis failed: {str(e)}"}

    def _generate_recommendations(
        self,
        embeddings: Dict[str, Any],
        evaluations: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
    ) -> List[str]:
        """Generate intelligent recommendations based on results"""
        recommendations = []

        # Find best performing methods
        method_scores = {}
        for name, metrics in evaluations.items():
            score = 0
            if "silhouette_score" in metrics:
                score += metrics["silhouette_score"] * 0.4
            if "neighborhood_preservation" in metrics:
                score += metrics["neighborhood_preservation"] * 0.4
            if "embedding_stability" in metrics:
                score += metrics["embedding_stability"] * 0.2
            method_scores[name] = score

        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            recommendations.append(f"Best performing method: {best_method}")

            # Method-specific recommendations
            if "umap" in best_method:
                recommendations.append(
                    "UMAP preserves both local and global structure well"
                )
            elif "tsne" in best_method:
                recommendations.append(
                    "t-SNE excels at revealing local cluster structure"
                )
            elif best_method == "pca":
                recommendations.append(
                    "PCA indicates linear relationships dominate the data"
                )

        # Data-specific recommendations
        if preprocessing_info.get("features_removed", 0) > 0:
            recommendations.append(
                f"Removed {preprocessing_info['features_removed']} low-variance features"
            )

        if preprocessing_info.get("scaling_method") == "power_transform":
            recommendations.append(
                "Applied power transformation due to skewed data distribution"
            )

        if len(
            [k for k in embeddings if embeddings[k].get("method_type") == "linear"]
        ) > len(
            [k for k in embeddings if embeddings[k].get("method_type") == "nonlinear"]
        ):
            recommendations.append(
                "Consider nonlinear methods for more complex pattern discovery"
            )

        return recommendations
