from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from .foreminer_aux import *  # assumes HAS_UMAP, HAS_TRIMAP, HAS_OPENTSNE, HAS_MULTICORE_TSNE, etc.


class DimensionalityAnalyzer(AnalysisStrategy):
    """State-of-the-art dimensionality reduction with adaptive preprocessing and ensemble methods"""

    @property
    def name(self) -> str:
        return "dimensionality"

    # --------------------------- Public API ---------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        try:
            scaled_data, preprocessing_info = self._adaptive_preprocessing(data, config)
            embeddings = self._enhanced_dimensionality_reduction(scaled_data, config)
            if not embeddings:
                return {"error": "All dimensionality reduction methods failed"}

            evaluation_results = self._evaluate_embeddings(embeddings, scaled_data)

            # Stable condition number (regularize Gram)
            gram = scaled_data.T @ scaled_data
            eps = 1e-8 * np.trace(gram) / max(1, gram.shape[0])
            cond_num = np.linalg.cond(gram + eps * np.eye(gram.shape[0]))

            results = {
                "embeddings": embeddings,
                "evaluation": evaluation_results,
                "preprocessing_info": preprocessing_info,
                "data_characteristics": {
                    "n_samples": scaled_data.shape[0],
                    "n_features": scaled_data.shape[1],
                    "condition_number": cond_num,
                    "effective_rank": int(np.linalg.matrix_rank(scaled_data)),
                },
                "recommendations": self._generate_recommendations(
                    embeddings, evaluation_results, preprocessing_info
                ),
            }
            return results

        except Exception as e:
            return {"error": f"Dimensionality analysis failed: {e}"}

    # --------------------------- Preprocessing ---------------------------
    def _adaptive_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advanced preprocessing with automatic method selection."""
        rng = np.random.default_rng(getattr(config, "random_state", 42))

        numeric = data.select_dtypes(include=[np.number]).dropna()
        if numeric.empty or numeric.shape[1] < 2:
            raise ValueError("Insufficient numeric data for dimensionality reduction")

        info: Dict[str, Any] = {}

        # 1) Remove low-variance features
        vt = VarianceThreshold(threshold=0.01)
        X = pd.DataFrame(
            vt.fit_transform(numeric),
            columns=numeric.columns[vt.get_support()],
            index=numeric.index,
        )
        info["features_removed"] = int((~vt.get_support()).sum())

        # 2) Adaptive sampling (cluster-stratified if possible)
        sample_cap = int(getattr(config, "sample_size_threshold", 2000))
        if len(X) > sample_cap:
            try:
                n_clusters = int(np.clip(len(X) // 100, 2, 10))
                Z = StandardScaler().fit_transform(X)
                km = KMeans(n_clusters=n_clusters, random_state=getattr(config, "random_state", 42), n_init=10)
                clusters = km.fit_predict(Z)

                indices: List[int] = []
                for cid in np.unique(clusters):
                    idx = np.flatnonzero(clusters == cid)
                    n_take = max(1, int(sample_cap * len(idx) / len(X)))
                    indices.extend(rng.choice(idx, size=min(n_take, len(idx)), replace=False))
                X = X.iloc[indices[:sample_cap]]
                info["sampling_method"] = "stratified"
            except Exception:
                idx = rng.choice(len(X), size=sample_cap, replace=False)
                X = X.iloc[idx]
                info["sampling_method"] = "random"

        # 3) Intelligent scaling selection
        # Decide among Power(YJ), Robust, Standard by conditioning & variance homogeneity.
        scalers = []
        skewness = X.skew().abs().mean()
        outlier_frac = ((X - X.mean()).abs() > 3 * X.std(ddof=0)).sum().sum() / X.size

        if (skewness > 2) or (outlier_frac > 0.10):
            scalers = [
                ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
                ("robust", RobustScaler()),
                ("standard", StandardScaler()),
            ]
        else:
            scalers = [
                ("standard", StandardScaler()),
                ("robust", RobustScaler()),
            ]

        def _score_scaled(arr: np.ndarray) -> float:
            # higher is better; guard zeros
            v = np.var(arr, axis=0)
            vmin = max(np.min(v), 1e-12)
            vmax = max(np.max(v), 1e-12)
            var_ratio = vmax / vmin
            cov = np.cov(arr.T)
            # regularize for stability
            lam = 1e-8 * np.trace(cov) / max(1, cov.shape[0])
            cond = np.linalg.cond(cov + lam * np.eye(cov.shape[0]))
            return -np.log(cond + 1e-12) - np.log(var_ratio + 1e-12)

        best_name, best_scaler, best_score = None, None, -np.inf
        for name, scaler in scalers:
            try:
                scaled = scaler.fit_transform(X)
                s = _score_scaled(scaled)
                if s > best_score:
                    best_name, best_scaler, best_score = name, scaler, s
            except Exception:
                continue

        if best_scaler is None:
            best_name, best_scaler = "standard_fallback", StandardScaler()

        info["scaling_method"] = best_name
        scaled_X = best_scaler.fit_transform(X)
        info["final_shape"] = tuple(scaled_X.shape)
        return scaled_X, info

    # --------------------------- Component selection ---------------------------
    def _compute_optimal_components(self, scaled_data: np.ndarray, max_components: int = 10) -> int:
        """Determine optimal n_components using multiple criteria."""
        n_samples, n_features = scaled_data.shape
        max_comp = int(min(max_components, n_features, max(2, n_samples // 2)))
        if max_comp < 2:
            return min(2, n_features)

        pca_full = PCA(random_state=0).fit(scaled_data)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        eig = pca_full.explained_variance_

        crit: Dict[str, int] = {}
        crit["var_95"] = int(np.argmax(cumvar >= 0.95) + 1) if np.any(cumvar >= 0.95) else max_comp

        if len(eig) >= 3:
            d1 = np.diff(eig)
            d2 = np.diff(d1)
            if len(d2) > 0:
                crit["elbow"] = int(np.argmax(d2) + 2)

        crit["kaiser"] = int(np.sum(eig > 1.0))

        # Parallel analysis (approx.)
        rand_eigs = []
        for _ in range(5):
            rnd = np.random.randn(*scaled_data.shape)
            rand_eigs.append(PCA(random_state=0).fit(rnd).explained_variance_)
        mean_rand = np.mean(rand_eigs, axis=0)
        crit["parallel"] = int(np.sum(eig[: len(mean_rand)] > mean_rand[: len(eig)]))

        # Combine (weighted)
        weights = {"var_95": 0.3, "elbow": 0.2, "kaiser": 0.2, "parallel": 0.3}
        vals = {k: v for k, v in crit.items() if 0 < v <= max_comp}
        if vals:
            wsum = sum(weights.get(k, 0.25) * v for k, v in vals.items())
            return max(2, min(max_comp, int(np.round(wsum))))
        return min(3, max_comp)

    # --------------------------- DR Methods ---------------------------
    def _enhanced_dimensionality_reduction(
        self, scaled_data: np.ndarray, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Apply state-of-the-art dimensionality reduction techniques."""
        results: Dict[str, Any] = {}
        n_samples, n_features = scaled_data.shape
        n_comp = self._compute_optimal_components(scaled_data)
        rs = getattr(config, "random_state", 42)

        # Linear methods
        linear_methods = {
            "pca": PCA(n_components=n_comp, random_state=rs),
            "ica": FastICA(n_components=n_comp, random_state=rs, max_iter=1000, tol=1e-4),
            "factor_analysis": FactorAnalysis(n_components=n_comp, random_state=rs),
        }

        # NMF only if non-negative
        if np.all(scaled_data >= 0):
            linear_methods["nmf"] = NMF(n_components=n_comp, random_state=rs, max_iter=1000, tol=1e-4)

        if n_features > 100:
            linear_methods["truncated_svd"] = TruncatedSVD(n_components=n_comp, random_state=rs)

        for name, method in linear_methods.items():
            try:
                emb = method.fit_transform(scaled_data)
                entry = {"embedding": emb, "method_type": "linear", "n_components": n_comp}
                if hasattr(method, "explained_variance_ratio_"):
                    evr = method.explained_variance_ratio_
                    entry["explained_variance_ratio"] = evr
                    entry["total_variance_explained"] = float(np.sum(evr))
                results[name] = entry
            except Exception as e:
                print(f"Linear method {name} failed: {e}")

        # Nonlinear methods (2D for viz)
        target_dim = 2
        # Perplexity must be < n_samples/3 and >= 5
        perplexity = int(np.clip(n_samples // 3 - 1, 5, 50))
        n_neighbors = int(np.clip(n_samples // 10, 3, 50))

        tsne_params = {
            "n_components": target_dim,
            "random_state": rs,
            "perplexity": min(perplexity, max(5, n_samples // 4, 5), max(5, n_samples - 2)),
            "init": "pca",
            "learning_rate": "auto",
            "n_iter": 1000,
            "early_exaggeration": 12,
            "min_grad_norm": 1e-7,
        }

        # OpenTSNE / MulticoreTSNE / sklearn TSNE
        if 'HAS_OPENTSNE' in globals() and HAS_OPENTSNE and n_samples > 1000:
            try:
                from openTSNE import TSNE as OpenTSNE
                tsne = OpenTSNE(**tsne_params, n_jobs=4, negative_gradient_method="fft")
                results["tsne_opentsne"] = {
                    "embedding": tsne.fit(scaled_data),
                    "method_type": "nonlinear",
                    "perplexity": tsne_params["perplexity"],
                }
            except Exception as e:
                print(f"OpenTSNE failed: {e}")
        elif 'HAS_MULTICORE_TSNE' in globals() and HAS_MULTICORE_TSNE:
            try:
                from MulticoreTSNE import MulticoreTSNE
                tsne = MulticoreTSNE(n_jobs=4, **tsne_params)
                results["tsne_multicore"] = {
                    "embedding": tsne.fit_transform(scaled_data),
                    "method_type": "nonlinear",
                    "perplexity": tsne_params["perplexity"],
                }
            except Exception as e:
                print(f"MulticoreTSNE failed: {e}")

        if not any(k.startswith("tsne") for k in results):
            try:
                tsne = TSNE(**tsne_params)
                results["tsne"] = {
                    "embedding": tsne.fit_transform(scaled_data),
                    "method_type": "nonlinear",
                    "perplexity": tsne_params["perplexity"],
                }
            except Exception as e:
                print(f"Standard t-SNE failed: {e}")

        # UMAP
        if 'HAS_UMAP' in globals() and HAS_UMAP:
            try:
                from umap import UMAP
                umap_params = dict(
                    n_components=target_dim,
                    random_state=rs,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="euclidean",
                    spread=1.0,
                    low_memory=(n_samples > 10_000),
                    n_epochs=None,
                    learning_rate=1.0,
                    repulsion_strength=1.0,
                )
                umap_reducer = UMAP(**umap_params)
                results["umap"] = {
                    "embedding": umap_reducer.fit_transform(scaled_data),
                    "method_type": "nonlinear",
                    "n_neighbors": n_neighbors,
                }
            except Exception as e:
                print(f"UMAP failed: {e}")

        # TriMap
        if 'HAS_TRIMAP' in globals() and HAS_TRIMAP and n_samples >= 100:
            try:
                from trimap import trimap
                trimap_embedding = trimap.TRIMAP(
                    n_dims=target_dim,
                    n_inliers=max(3, n_neighbors // 2),
                    n_outliers=max(1, n_neighbors // 6),
                    n_random=max(1, n_neighbors // 6),
                    lr=1000.0,
                    n_iters=1200,
                ).fit_transform(scaled_data)
                results["trimap"] = {"embedding": trimap_embedding, "method_type": "nonlinear"}
            except Exception as e:
                print(f"TriMap failed: {e}")

        # Other manifold learners
        manifold = {
            "isomap": Isomap(n_components=target_dim, n_neighbors=n_neighbors),
            "lle": LocallyLinearEmbedding(
                n_components=target_dim, n_neighbors=n_neighbors, random_state=rs, method="standard"
            ),
            "spectral": SpectralEmbedding(n_components=target_dim, n_neighbors=n_neighbors, random_state=rs),
        }
        for name, method in manifold.items():
            try:
                emb = method.fit_transform(scaled_data)
                results[name] = {
                    "embedding": emb,
                    "method_type": "nonlinear",
                    "n_neighbors": getattr(method, "n_neighbors", None),
                }
            except Exception as e:
                print(f"Manifold method {name} failed: {e}")

        return results

    # --------------------------- Evaluation ---------------------------
    def _evaluate_embeddings(
        self, embeddings: Dict[str, Any], scaled_data: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate embedding quality using multiple metrics."""
        evals: Dict[str, Any] = {}
        rng = np.random.default_rng(42)

        for name, res in embeddings.items():
            emb = res.get("embedding")
            if emb is None or getattr(emb, "shape", (0, 0))[0] < 10:
                continue

            metrics: Dict[str, float] = {}

            # Silhouette with KMeans clusters
            try:
                n_clusters = int(np.clip(emb.shape[0] // 10, 2, 8))
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(emb)
                if len(np.unique(labels)) > 1:
                    metrics["silhouette_score"] = float(silhouette_score(emb, labels))
            except Exception:
                pass

            # Neighborhood preservation (sampled)
            try:
                if res.get("method_type") == "nonlinear" and scaled_data.shape[0] <= 1000:
                    from scipy.spatial.distance import pdist, squareform
                    from scipy.stats import spearmanr

                    n_sample = min(200, emb.shape[0])
                    idx = rng.choice(emb.shape[0], size=n_sample, replace=False)

                    D0 = squareform(pdist(scaled_data[idx]))
                    D1 = squareform(pdist(emb[idx]))
                    rho, _ = spearmanr(D0.ravel(), D1.ravel())
                    metrics["neighborhood_preservation"] = float(rho)
            except Exception:
                pass

            # Simple stability proxy
            try:
                metrics["embedding_stability"] = float(1.0 / (1.0 + np.mean(np.std(emb, axis=0))))
            except Exception:
                pass

            evals[name] = metrics

        return evals

    # --------------------------- Recommendations ---------------------------
    def _generate_recommendations(
        self,
        embeddings: Dict[str, Any],
        evaluations: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
    ) -> List[str]:
        recs: List[str] = []

        # Score combination (same idea, explicit weights)
        weights = {"silhouette_score": 0.4, "neighborhood_preservation": 0.4, "embedding_stability": 0.2}
        scores: Dict[str, float] = {}
        for name, mets in evaluations.items():
            score = 0.0
            for k, w in weights.items():
                if k in mets and np.isfinite(mets[k]):
                    score += w * mets[k]
            scores[name] = score

        if scores:
            best = max(scores, key=scores.get)
            recs.append(f"Best performing method: {best}")
            if "umap" in best:
                recs.append("UMAP preserves both local and global structure well")
            elif "tsne" in best:
                recs.append("t-SNE excels at revealing local cluster structure")
            elif best == "pca":
                recs.append("PCA indicates linear relationships dominate the data")

        # Data-specific notes
        if preprocessing_info.get("features_removed", 0) > 0:
            recs.append(f"Removed {preprocessing_info['features_removed']} low-variance features")

        if preprocessing_info.get("scaling_method") == "power_transform":
            recs.append("Applied power transformation due to skewed data distribution")

        n_linear = sum(1 for k in embeddings if embeddings[k].get("method_type") == "linear")
        n_nonlinear = sum(1 for k in embeddings if embeddings[k].get("method_type") == "nonlinear")
        if n_linear > n_nonlinear:
            recs.append("Consider nonlinear methods for more complex pattern discovery")

        return recs
