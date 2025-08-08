from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
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

from .foreminer_aux import *  # expects HAS_HDBSCAN and config attrs like max_clusters


class ClusterAnalyzer(AnalysisStrategy):
    """State-of-the-art clustering analysis with comprehensive evaluation and stability assessment"""

    @property
    def name(self) -> str:
        return "clusters"

    # --------------------------- Helpers ---------------------------
    def _count_clusters(self, labels: np.ndarray) -> int:
        """Count clusters ignoring noise label -1."""
        u = set(labels)
        return int(len(u) - (1 if -1 in u else 0))

    # --------------------------- Preprocessing ---------------------------
    def _adaptive_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced preprocessing with outlier detection and feature scaling"""
        numeric = data.select_dtypes(include=[np.number]).dropna()
        if numeric.empty or numeric.shape[1] == 0:
            raise ValueError("No numeric data available for clustering")

        info: Dict[str, Any] = {}

        # IQR outlier flag (row-level)
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = ((numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))).any(axis=1)
        n_out = int(outlier_mask.sum())
        info["outliers_detected"] = n_out
        info["outlier_percentage"] = float(n_out / max(len(numeric), 1) * 100.0)

        # Scaling choice
        skewness = float(numeric.skew().abs().mean())
        outlier_frac = n_out / max(len(numeric), 1)
        try:
            if (skewness > 2.0) or (outlier_frac > 0.10):
                scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                info["scaling_method"] = "power_transform"
            else:
                scaler = StandardScaler()
                info["scaling_method"] = "standard"
            X = scaler.fit_transform(numeric)
        except Exception:
            scaler = RobustScaler()
            X = scaler.fit_transform(numeric)
            info["scaling_method"] = "robust"

        n_samples, n_features = X.shape
        info["curse_of_dimensionality_risk"] = bool(n_features > n_samples / 3)

        # PCA for high-dim dense data
        if (n_features > 50) and (n_samples > n_features):
            try:
                pca = PCA(n_components=min(50, n_samples // 2), random_state=getattr(config, "random_state", 42))
                X = pca.fit_transform(X)
                info["pca_applied"] = True
                info["pca_variance_explained"] = float(pca.explained_variance_ratio_.sum())
                info["final_dimensions"] = int(X.shape[1])
            except Exception:
                info["pca_applied"] = False
        else:
            info["pca_applied"] = False

        info["final_shape"] = tuple(X.shape)
        return X, info

    # --------------------------- Estimate k ---------------------------
    def _estimate_optimal_clusters(self, data: np.ndarray, max_k: int = 15) -> Dict[str, Any]:
        """Multi-method optimal cluster number estimation"""
        n_samples = data.shape[0]
        max_k = int(min(max_k, max(2, n_samples // 3)))
        if max_k < 2:
            return {"optimal_k": 2, "methods": {}, "confidence": "low", "method_agreement": 0}

        methods: Dict[str, int] = {}

        # 1) Elbow on WCSS
        wcss = []
        for k in range(1, max_k + 1):
            if k == 1:
                wcss.append(float(np.var(data, axis=0, ddof=1).sum() * n_samples))
            else:
                km = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm="lloyd")
                km.fit(data)
                wcss.append(float(km.inertia_))
        if len(wcss) >= 3:
            d2 = np.diff(np.diff(wcss))
            elbow_k = int(np.argmax(d2) + 2)  # +2 for second diff offset
            methods["elbow"] = int(np.clip(elbow_k, 2, max_k))

        # 2) Silhouette / 3) Calinski–Harabasz / 4) Davies–Bouldin
        sil_best, ch_best, db_best = (None, -np.inf), (None, -np.inf), (None, np.inf)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm="lloyd")
            labels = km.fit_predict(data)
            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(data, labels)
            ch = calinski_harabasz_score(data, labels)
            db = davies_bouldin_score(data, labels)
            if sil > sil_best[1]: sil_best = (k, sil)
            if ch > ch_best[1]: ch_best = (k, ch)
            if db < db_best[1]: db_best = (k, db)

        if sil_best[0] is not None: methods["silhouette"] = sil_best[0]
        if ch_best[0] is not None: methods["calinski_harabasz"] = ch_best[0]
        if db_best[0] is not None: methods["davies_bouldin"] = db_best[0]

        # 5) Gap statistic (lightweight)
        try:
            gaps: List[Tuple[int, float]] = []
            mins, maxs = data.min(axis=0), data.max(axis=0)
            for k in range(1, max_k + 1):
                if k == 1:
                    intra = float(np.var(data, axis=0, ddof=1).sum() * n_samples)
                else:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(data)
                    intra = float(km.inertia_)

                ref = []
                for _ in range(5):
                    rand = np.random.uniform(mins, maxs, size=data.shape)
                    if k == 1:
                        ref.append(float(np.var(rand, axis=0, ddof=1).sum() * n_samples))
                    else:
                        kmr = KMeans(n_clusters=k, random_state=42, n_init=5)
                        kmr.fit(rand)
                        ref.append(float(kmr.inertia_))
                gap = float(np.log(np.mean(ref) + 1e-12) - np.log(intra + 1e-12))
                gaps.append((k, gap))
            for (k, gk), (_, gk1) in zip(gaps[:-1], gaps[1:]):
                if gk >= gk1 - 0.1:  # crude std proxy
                    methods["gap_statistic"] = k
                    break
        except Exception as e:
            print(f"Gap statistic calculation failed: {e}")

        if methods:
            weights = {"silhouette": 0.3, "calinski_harabasz": 0.25, "davies_bouldin": 0.2, "elbow": 0.15, "gap_statistic": 0.1}
            wsum = sum(weights.get(m, 0.2) * k for m, k in methods.items())
            opt_k = int(np.clip(np.round(wsum), 2, max_k))
            vals = list(methods.values())
            var = float(np.var(vals)) if len(vals) > 1 else 0.0
            conf = "high" if var < 0.5 else "medium" if var < 2.0 else "low"
            return {"optimal_k": opt_k, "methods": methods, "confidence": conf, "method_agreement": len(set(vals))}
        else:
            return {"optimal_k": min(3, max_k), "methods": {}, "confidence": "low", "method_agreement": 0}

    # --------------------------- KMeans (enhanced) ---------------------------
    def _enhanced_kmeans_analysis(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        max_k = int(min(getattr(config, "max_clusters", 10), max(2, len(data) // 3)))
        k_range = range(2, max_k + 1)

        all_results: List[Dict[str, Any]] = []
        best_model, best_k, best_score = None, int(optimal_k_info["optimal_k"]), -1.0

        for k in k_range:
            stability_scores = []
            models = []
            for init_method in ("k-means++", "random"):
                try:
                    km = KMeans(n_clusters=k, init=init_method, n_init=20, max_iter=500,
                                random_state=getattr(config, "random_state", 42), algorithm="lloyd")
                    labels = km.fit_predict(data)

                    km2 = KMeans(n_clusters=k, init=init_method, n_init=20, max_iter=500,
                                 random_state=getattr(config, "random_state", 42) + 1, algorithm="lloyd")
                    labels2 = km2.fit_predict(data)
                    stability_scores.append(adjusted_rand_score(labels, labels2))
                    models.append((km, labels))
                except Exception as e:
                    print(f"K-means failed for k={k}, init={init_method}: {e}")

            if not models:
                continue

            km, labels = max(models, key=lambda t: silhouette_score(data, t[1]) if len(set(t[1])) > 1 else -1)
            try:
                sil = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
                ch = calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else 0
                db = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else np.inf
                avg_stab = float(np.mean(stability_scores)) if stability_scores else 0.0

                counts = np.bincount(labels)
                balance = float(1 - np.std(counts) / (np.mean(counts) + 1e-12))

                entry = {
                    "k": k,
                    "silhouette": sil,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "inertia": float(km.inertia_),
                    "stability": avg_stab,
                    "balance": balance,
                    "cluster_sizes": dict(Counter(labels)),
                    "model": km,
                    "labels": labels,
                }
                all_results.append(entry)

                composite = sil * 0.4 + avg_stab * 0.3 + balance * 0.2 + (0.1 * (1 - min(db / 10.0, 1.0)))
                if composite > best_score:
                    best_score, best_k, best_model = composite, k, km
            except Exception as e:
                print(f"Scoring failed for k={k}: {e}")

        final_labels = best_model.fit_predict(data) if best_model is not None else np.zeros(len(data), dtype=int)

        return {
            "labels": final_labels,
            "model": best_model,
            "scores": all_results,
            "best_k": best_k,
            "n_clusters": self._count_clusters(final_labels),  # unified
            "centers": (best_model.cluster_centers_ if best_model is not None else None),
            "cluster_sizes": dict(Counter(final_labels)),
            "method_type": "centroid_based",
        }

    # --------------------------- Hierarchical ---------------------------
    def _hierarchical_clustering_analysis(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive hierarchical clustering with multiple linkage methods"""
        n = data.shape[0]
        idx = np.random.choice(n, size=min(n, 1000), replace=False)
        sample = data[idx]

        methods = ["ward", "complete", "average", "single"]
        results: Dict[str, Any] = {}
        best_link, best_sil = None, -np.inf

        for method in methods:
            try:
                # IMPORTANT: Ward expects observations directly and 'euclidean' metric.
                if method == "ward":
                    Z = linkage(sample, method=method, metric="euclidean")
                else:
                    d = pdist(sample, metric="euclidean")
                    Z = linkage(d, method=method)

                k = int(optimal_k_info["optimal_k"])
                # Use Agglomerative on full data for labels with same linkage
                agg = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=method,
                    metric="euclidean",          # never None (fixes ward error)
                    compute_distances=True,
                )

                full_labels = agg.fit_predict(data)

                sil = silhouette_score(data, full_labels) if len(np.unique(full_labels)) > 1 else -1
                ch = calinski_harabasz_score(data, full_labels) if sil > -1 else 0
                db = davies_bouldin_score(data, full_labels) if sil > -1 else np.inf

                result = {
                    "linkage_method": method,
                    "labels": full_labels,
                    "silhouette": sil,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "n_clusters": self._count_clusters(full_labels),  # unified
                    "cluster_sizes": dict(Counter(full_labels)),
                    "model": agg,
                    "linkage_matrix": Z,
                    "method_type": "hierarchical",
                }
                results[f"hierarchical_{method}"] = result
                if sil > best_sil:
                    best_sil, best_link = sil, method
            except Exception as e:
                print(f"Hierarchical clustering with {method} failed: {e}")

        best = results.get(f"hierarchical_{best_link}", {})
        if best:
            best["best_linkage_method"] = best_link
        return best

    # --------------------------- Density-based ---------------------------
    def _density_based_clustering(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # DBSCAN (eps knee)
        try:
            k = max(2, min(4, len(data) // 10))
            nn = NearestNeighbors(n_neighbors=k).fit(data)
            dist, _ = nn.kneighbors(data)
            kth = np.sort(dist[:, k - 1])
            diff = np.diff(kth)
            eps = float(kth[np.argmax(diff)]) if len(diff) else float(np.median(kth))
            min_samples = int(max(2, k))

            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(data)
            n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
            noise = int((labels == -1).sum())

            entry = {
                "labels": labels,
                "model": db,
                "n_clusters": n_clusters,
                "noise_points": noise,
                "eps": eps,
                "min_samples": min_samples,
                "cluster_sizes": dict(Counter(labels[labels != -1])),
                "method_type": "density_based",
            }
            if n_clusters > 1 and (labels != -1).sum() > 1:
                mask = labels != -1
                entry["silhouette"] = float(silhouette_score(data[mask], labels[mask]))
            results["dbscan"] = entry
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")

        # HDBSCAN (if available)
        if "HAS_HDBSCAN" in globals() and HAS_HDBSCAN:
            try:
                import hdbscan
                min_cluster_size = max(5, len(data) // 50)
                min_samples = max(1, min_cluster_size // 2)
                model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method="eom",
                )
                labels = model.fit_predict(data)
                n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
                noise = int((labels == -1).sum())

                entry = {
                    "labels": labels,
                    "model": model,
                    "n_clusters": n_clusters,
                    "noise_points": noise,
                    "min_cluster_size": int(min_cluster_size),
                    "cluster_sizes": dict(Counter(labels[labels != -1])),
                    "method_type": "density_based",
                    "probabilities": (model.probabilities_.tolist() if hasattr(model, "probabilities_") else None),
                }
                if n_clusters > 1 and (labels != -1).sum() > 1:
                    mask = labels != -1
                    entry["silhouette"] = float(silhouette_score(data[mask], labels[mask]))
                results["hdbscan"] = entry
            except Exception as e:
                print(f"HDBSCAN clustering failed: {e}")

        return results

    # --------------------------- Probabilistic ---------------------------
    def _probabilistic_clustering(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        max_k = int(min(getattr(config, "max_clusters", 10), max(2, len(data) // 2)))

        # GMM grid (k x covariance_type)
        try:
            cov_types = ("full", "tied", "diag", "spherical")
            best_bic, best_gmm, best_k = np.inf, None, int(optimal_k_info["optimal_k"])
            comparison: List[Dict[str, Any]] = []

            for cov in cov_types:
                for k in range(2, max_k + 1):
                    try:
                        gmm = GaussianMixture(
                            n_components=k, covariance_type=cov, max_iter=300, n_init=3,
                            random_state=getattr(config, "random_state", 42)
                        ).fit(data)
                        bic = float(gmm.bic(data))
                        aic = float(gmm.aic(data))
                        ll = float(gmm.score(data))
                        comparison.append({"k": k, "covariance_type": cov, "bic": bic, "aic": aic,
                                           "log_likelihood": ll, "converged": bool(gmm.converged_)})
                        if gmm.converged_ and bic < best_bic:
                            best_bic, best_gmm, best_k = bic, gmm, k
                    except Exception:
                        continue

            if best_gmm is not None:
                labels = best_gmm.predict(data)
                probs = best_gmm.predict_proba(data)
                entry = {
                    "labels": labels,
                    "probabilities": probs.tolist(),
                    "model": best_gmm,
                    "best_k": int(best_k),
                    "n_clusters": self._count_clusters(labels),  # unified
                    "best_bic": float(best_bic),
                    "model_comparison": comparison,
                    "cluster_sizes": dict(Counter(labels)),
                    "silhouette": float(silhouette_score(data, labels)) if len(np.unique(labels)) > 1 else -1,
                    "method_type": "probabilistic",
                    "covariance_type": best_gmm.covariance_type,
                    "means": best_gmm.means_.tolist(),
                    "covariances": best_gmm.covariances_.tolist(),
                }
                results["gmm"] = entry
        except Exception as e:
            print(f"GMM clustering failed: {e}")

        # Bayesian GMM (smaller datasets)
        try:
            if len(data) <= 5000:
                bgmm = BayesianGaussianMixture(
                    n_components=min(10, max_k), covariance_type="full", max_iter=300,
                    random_state=getattr(config, "random_state", 42)
                ).fit(data)
                labels = bgmm.predict(data)
                eff = int(np.sum(bgmm.weights_ > 0.01))
                entry = {
                    "labels": labels,
                    "probabilities": bgmm.predict_proba(data).tolist(),
                    "model": bgmm,
                    "effective_components": eff,
                    "n_clusters": self._count_clusters(labels),  # unified
                    "weights": bgmm.weights_.tolist(),
                    "cluster_sizes": dict(Counter(labels)),
                    "silhouette": float(silhouette_score(data, labels)) if len(np.unique(labels)) > 1 else -1,
                    "method_type": "bayesian_probabilistic",
                }
                results["bayesian_gmm"] = entry
        except Exception as e:
            print(f"Bayesian GMM clustering failed: {e}")

        return results

    # --------------------------- Ensemble ---------------------------
    def _ensemble_clustering(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Consensus clustering via co-association + average-linkage"""
        if not all_results:
            return {}
        try:
            label_sets = []
            for name, res in all_results.items():
                lab = res.get("labels")
                if lab is not None and len(lab) == len(data):
                    label_sets.append(np.asarray(lab))

            if len(label_sets) < 2:
                return {}

            n = len(data)
            co = np.zeros((n, n), dtype=float)

            # Build co-association (ignore noise)
            for lab in label_sets:
                valid = (lab != -1)
                for c in np.unique(lab[valid]):
                    inds = np.where(lab == c)[0]
                    if len(inds) > 1:
                        co[np.ix_(inds, inds)] += 1.0

            co /= len(label_sets)
            np.fill_diagonal(co, 1.0)
            dist = 1.0 - co
            dist = (dist + dist.T) / 2.0
            np.fill_diagonal(dist, 0.0)

            Z = linkage(squareform(dist), method="average")

            # Choose k by plurality of non-noise cluster counts among methods
            counts = []
            for lab in label_sets:
                u = np.unique(lab[lab != -1])
                if len(u) > 1:
                    counts.append(len(u))
            k = (Counter(counts).most_common(1)[0][0] if counts else 3)

            consensus = fcluster(Z, k, criterion="maxclust") - 1
            sil = float(silhouette_score(data, consensus)) if len(np.unique(consensus)) > 1 else -1

            return {
                "labels": consensus,
                "n_methods": len(label_sets),
                "co_association_matrix": co.tolist(),
                "consensus_k": int(k),
                "n_clusters": int(k),  # unified
                "cluster_sizes": dict(Counter(consensus)),
                "silhouette": sil,
                "method_type": "ensemble",
                "participating_methods": list(all_results.keys()),
            }
        except Exception as e:
            print(f"Ensemble clustering failed: {e}")
            return {}

    # --------------------------- Evaluation ---------------------------
    def _evaluate_clustering_results(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        evaluations: Dict[str, Any] = {}
        for name, res in all_results.items():
            lab = res.get("labels")
            if lab is None or len(lab) != len(data):
                continue

            evald: Dict[str, Any] = {}
            try:
                labels = np.asarray(lab)
                uniq = set(labels)
                n_clusters = len(uniq) - (1 if -1 in uniq else 0)
                n_noise = int((labels == -1).sum())

                evald.update({
                    "n_clusters": n_clusters,
                    "n_noise_points": n_noise,
                    "noise_ratio": float(n_noise / max(len(labels), 1)),
                })

                # Silhouette
                if "silhouette" in res:
                    evald["silhouette_score"] = res["silhouette"]
                elif n_clusters > 1:
                    mask = labels != -1
                    if mask.sum() > 1:
                        evald["silhouette_score"] = float(silhouette_score(data[mask], labels[mask]))

                # Extra metrics
                if n_clusters > 1:
                    mask = labels != -1
                    if mask.sum() > 1:
                        Xv, yv = data[mask], labels[mask]
                        evald["calinski_harabasz_score"] = float(calinski_harabasz_score(Xv, yv))
                        evald["davies_bouldin_score"] = float(davies_bouldin_score(Xv, yv))

                        sizes = np.bincount(yv)
                        balance = float(1 - np.std(sizes) / (np.mean(sizes) + 1e-12))
                        evald["cluster_balance"] = balance

                        # Separation ratio
                        uniqv = np.unique(yv)
                        if len(uniqv) >= 2:
                            intra, inter = [], []
                            for c in uniqv:
                                pts = Xv[yv == c]
                                if len(pts) > 1:
                                    intra.extend(pdist(pts))
                                    other = Xv[yv != c]
                                    if len(other) > 0:
                                        for p in pts:
                                            inter.extend(np.linalg.norm(other - p, axis=1))
                            if intra and inter:
                                mi = float(np.mean(intra))
                                Mo = float(np.mean(inter))
                                evald["separation_ratio"] = (Mo / mi) if mi > 0 else 0.0

                if "stability" in res:
                    evald["stability_score"] = res["stability"]

                evald["method_type"] = res.get("method_type", "unknown")

                if evald["method_type"] == "probabilistic" and "probabilities" in res:
                    probs = np.asarray(res["probabilities"])
                    ent = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    evald["mean_assignment_entropy"] = float(np.mean(ent))
                    evald["assignment_uncertainty"] = float(np.std(ent))
            except Exception as e:
                print(f"Evaluation failed for {name}: {e}")
                evald["error"] = str(e)

            evaluations[name] = evald
        return evaluations

    # --------------------------- Recommendations ---------------------------
    def _generate_clustering_recommendations(
        self,
        all_results: Dict[str, Any],
        evaluations: Dict[str, Any],
        optimal_k_info: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
    ) -> List[str]:
        recs: List[str] = []

        # Score & pick winner
        weights = {"silhouette_score": 0.3, "cluster_balance": 0.2, "separation_ratio": 0.2, "stability_score": 0.2}
        scores: Dict[str, float] = {}
        for name, mets in evaluations.items():
            if "error" in mets: continue
            s, wsum = 0.0, 0.0
            for k, w in weights.items():
                if k in mets and np.isfinite(mets[k]):
                    val = mets[k] if k != "separation_ratio" else min(mets[k] / 5.0, 1.0)
                    s += w * val
                    wsum += w
            s -= mets.get("noise_ratio", 0.0) * 0.1
            wsum += 0.1
            if wsum > 0:
                scores[name] = s / wsum

        best_method = max(scores, key=scores.get) if scores else None
        if best_method:
            recs.append(f"🏆 Best method: {best_method.upper()} (score: {scores[best_method]:.3f})")
            mtype = all_results.get(best_method, {}).get("method_type", "")
            if "density" in mtype:
                nr = evaluations[best_method].get("noise_ratio", 0.0)
                if nr > 0.10:
                    recs.append(f"⚠️ High noise ratio ({nr:.1%}) - consider tuning eps/min_samples")
                else:
                    recs.append("✅ Density-based clustering effectively identified cluster structure")
            elif mtype == "probabilistic":
                unct = evaluations[best_method].get("assignment_uncertainty", 0.0)
                recs.append("🔀 High assignment uncertainty suggests overlapping clusters" if unct > 0.5
                            else "📊 Probabilistic clustering shows confident cluster assignments")
            elif mtype == "hierarchical":
                recs.append(f"🌳 Hierarchical clustering with {all_results.get(best_method, {}).get('best_linkage_method', 'unknown')} linkage worked best")
            elif mtype == "ensemble":
                recs.append(f"🤝 Ensemble of {all_results.get(best_method, {}).get('n_methods', 0)} methods achieved robust consensus")

        if preprocessing_info.get("curse_of_dimensionality_risk", False):
            recs.append("📏 High-dimensional data detected - consider dimensionality reduction")
        if preprocessing_info.get("outlier_percentage", 0) > 10:
            recs.append("🎯 Many outliers detected - density-based methods recommended")

        k, conf = optimal_k_info.get("optimal_k", 2), optimal_k_info.get("confidence", "low")
        if conf == "high":
            recs.append(f"✨ Strong evidence for {k} clusters")
        elif conf == "medium":
            recs.append(f"🤔 Moderate evidence for {k} clusters - consider range {k-1}-{k+1}")
        else:
            recs.append("❓ Unclear optimal cluster number - try multiple values")

        succ = len([r for r in all_results.values() if "labels" in r])
        if succ <= 2:
            recs.append("⚡ Consider trying additional clustering algorithms for comparison")

        if best_method and evaluations.get(best_method, {}).get("n_clusters", 0) > 10:
            recs.append("📊 Many clusters found - consider hierarchical visualization")
        elif best_method and evaluations.get(best_method, {}).get("n_clusters", 0) < 2:
            recs.append("🔍 No clear clusters found - data may not have cluster structure")

        return recs[:5]

    # --------------------------- Main ---------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        try:
            X, prep = self._adaptive_preprocessing(data, config)
            optk = self._estimate_optimal_clusters(X, getattr(config, "max_clusters", 10))

            results: Dict[str, Any] = {}

            # K-means
            try:
                km = self._enhanced_kmeans_analysis(X, config, optk)
                if km: results["kmeans"] = km
            except Exception as e:
                print(f"Enhanced K-means failed: {e}")

            # Hierarchical
            try:
                h = self._hierarchical_clustering_analysis(X, config, optk)
                if h: results["hierarchical"] = h
            except Exception as e:
                print(f"Hierarchical clustering failed: {e}")

            # Density-based
            try:
                results.update(self._density_based_clustering(X, config))
            except Exception as e:
                print(f"Density-based clustering failed: {e}")

            # Probabilistic
            try:
                results.update(self._probabilistic_clustering(X, config, optk))
            except Exception as e:
                print(f"Probabilistic clustering failed: {e}")

            # Spectral (smaller datasets)
            try:
                if len(X) <= 2000:
                    spec = SpectralClustering(
                        n_clusters=int(optk["optimal_k"]),
                        random_state=getattr(config, "random_state", 42),
                        affinity="rbf",
                        gamma=1.0,
                        assign_labels="kmeans",
                    )
                    sl = spec.fit_predict(X)
                    results["spectral"] = {
                        "labels": sl,
                        "model": spec,
                        "n_clusters": self._count_clusters(sl),  # unified
                        "cluster_sizes": dict(Counter(sl)),
                        "silhouette": float(silhouette_score(X, sl)) if len(np.unique(sl)) > 1 else -1,
                        "method_type": "spectral",
                    }
            except Exception as e:
                print(f"Spectral clustering failed: {e}")

            # Mean Shift (small datasets)
            try:
                if len(X) <= 1000:
                    ms = MeanShift()
                    ms_labels = ms.fit_predict(X)
                    n_ms = self._count_clusters(ms_labels)
                    if n_ms > 1:
                        results["mean_shift"] = {
                            "labels": ms_labels,
                            "model": ms,
                            "n_clusters": n_ms,
                            "cluster_sizes": dict(Counter(ms_labels)),
                            "silhouette": float(silhouette_score(X, ms_labels)),
                            "method_type": "mean_shift",
                        }
            except Exception as e:
                print(f"Mean Shift clustering failed: {e}")

            # Ensemble consensus
            ens = self._ensemble_clustering(results, X)
            if ens:
                results["ensemble"] = ens

            evals = self._evaluate_clustering_results(results, X)
            recs = self._generate_clustering_recommendations(results, evals, optk, prep)

            return {
                "clustering_results": results,
                "evaluations": evals,
                "optimal_k_analysis": optk,
                "preprocessing_info": prep,
                "data_characteristics": {
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "data_variance": float(np.var(X)),
                    "data_spread": float(np.ptp(X)),
                },
                "recommendations": recs,
                "summary": {
                    "methods_attempted": len(results),
                    "successful_methods": len([r for r in results.values() if "labels" in r]),
                    "best_method": (max(evals.keys(), key=lambda k: evals[k].get("silhouette_score", -1)) if evals else None),
                },
            }
        except Exception as e:
            return {"error": f"Clustering analysis failed: {e}"}
