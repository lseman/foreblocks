import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
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


# --------------------------- utilities ---------------------------
def _rng(config: AnalysisConfig) -> np.random.Generator:
    rs = getattr(config, "random_state", 42)
    return np.random.default_rng(rs if rs is not None else 42)

def _is_constant(col: np.ndarray, tol: float = 1e-12) -> bool:
    # permit NaNs
    if len(col) == 0:
        return True
    cmin = np.nanmin(col)
    cmax = np.nanmax(col)
    return not np.isfinite(cmin) or not np.isfinite(cmax) or (cmax - cmin) <= tol

def _winsorize_inplace(X: np.ndarray, q: float = 0.001) -> None:
    """Robust winsorization with better edge case handling"""
    if X.size == 0:
        return
    
    lo = np.nanquantile(X, q, axis=0)
    hi = np.nanquantile(X, 1 - q, axis=0)
    
    # Handle edge cases where quantiles might be the same
    valid_mask = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    if np.any(valid_mask):
        np.clip(X[:, valid_mask], lo[valid_mask], hi[valid_mask], out=X[:, valid_mask])

def _safe_silhouette(X: np.ndarray, labels: np.ndarray, max_points: int = 4000) -> float:
    """Improved silhouette calculation with better sampling"""
    labs = np.asarray(labels)
    uniq = np.unique(labs[labs != -1])
    if uniq.size < 2:
        return -1.0
    
    n = len(labs)
    if n > max_points:
        # Better stratified sampling to maintain cluster proportions
        rng = np.random.default_rng(42)
        per_cluster = max(10, max_points // len(uniq))
        
        indices = []
        for cluster_id in uniq:
            cluster_indices = np.where(labs == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            # Sample from this cluster
            n_sample = min(per_cluster, len(cluster_indices))
            if n_sample > 0:
                sampled = rng.choice(cluster_indices, n_sample, replace=False)
                indices.extend(sampled)
        
        if len(indices) < 2:
            return -1.0
            
        idx = np.array(indices)
        return float(silhouette_score(X[idx], labs[idx]))
    
    return float(silhouette_score(X, labs))

def _median_heuristic_gamma(sample: np.ndarray) -> float:
    """More robust gamma calculation for RBF kernels"""
    if sample.shape[0] < 2:
        return 1.0
    
    # Subsample if too large
    if sample.shape[0] > 1000:
        rng = np.random.default_rng(42)
        idx = rng.choice(sample.shape[0], 1000, replace=False)
        sample = sample[idx]
    
    d = pdist(sample)
    if d.size == 0:
        return 1.0
        
    med = np.median(d)
    if med <= 0 or not np.isfinite(med):
        # Fallback to mean distance
        med = np.mean(d)
        if med <= 0 or not np.isfinite(med):
            return 1.0
    
    return 1.0 / (med ** 2)

def _kneedle_idx(y: np.ndarray) -> int:
    """Improved knee detection using normalized curve analysis"""
    if y.size < 3: 
        return int(np.argmax(y))
    
    # Normalize y values to [0, 1]
    y_min, y_max = np.min(y), np.max(y)
    if y_max - y_min < 1e-10:
        return len(y) // 2
    
    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, len(y))
    
    # Find point with maximum distance from line connecting endpoints
    line_y = np.linspace(y_norm[0], y_norm[-1], len(y_norm))
    distances = np.abs(y_norm - line_y)
    
    return int(np.argmax(distances))

def _better_gap_statistic(X: np.ndarray, max_k: int = 10, B: int = 5) -> int:
    """More robust gap statistic implementation"""
    n, d = X.shape
    if n < 10 or max_k < 2:
        return 2
    
    gaps = []
    s_k = []
    
    # Reference distribution bounds
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    
    # Ensure we have variation in each dimension
    valid_dims = (x_max - x_min) > 1e-10
    if not np.any(valid_dims):
        return 2
    
    for k in range(1, max_k + 1):
        # Observed within-cluster dispersion
        try:
            if k == 1:
                w_k = np.log(np.sum(np.var(X, axis=0)) * n + 1e-10)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
                labels = kmeans.fit_predict(X)
                w_k = 0
                for i in range(k):
                    cluster_points = X[labels == i]
                    if len(cluster_points) > 1:
                        w_k += np.sum(np.var(cluster_points, axis=0)) * len(cluster_points)
                w_k = np.log(w_k + 1e-10)
        except Exception:
            continue
        
        # Reference distribution
        w_kb_star = []
        for b in range(B):
            try:
                # Generate uniform reference data only for valid dimensions
                X_ref = X.copy()
                X_ref[:, valid_dims] = np.random.uniform(
                    x_min[valid_dims], x_max[valid_dims], (n, np.sum(valid_dims))
                )
                
                if k == 1:
                    w_ref = np.log(np.sum(np.var(X_ref, axis=0)) * n + 1e-10)
                else:
                    kmeans_ref = KMeans(n_clusters=k, random_state=42 + b, n_init=1, max_iter=50)
                    labels_ref = kmeans_ref.fit_predict(X_ref)
                    w_ref = 0
                    for i in range(k):
                        cluster_points = X_ref[labels_ref == i]
                        if len(cluster_points) > 1:
                            w_ref += np.sum(np.var(cluster_points, axis=0)) * len(cluster_points)
                    w_ref = np.log(w_ref + 1e-10)
                
                w_kb_star.append(w_ref)
            except Exception:
                continue
        
        if len(w_kb_star) == 0:
            continue
            
        gap = np.mean(w_kb_star) - w_k
        s_k.append(np.std(w_kb_star) * np.sqrt(1 + 1/B))
        gaps.append(gap)
    
    if len(gaps) < 2:
        return 2
    
    # Find optimal k using gap(k) >= gap(k+1) - s(k+1)
    for k in range(len(gaps) - 1):
        if gaps[k] >= gaps[k + 1] - s_k[k + 1]:
            return k + 1  # Convert to 1-based indexing
    
    return min(len(gaps), max_k)


class ClusterAnalyzer(AnalysisStrategy):
    """SOTA fast clustering analysis with adaptive method selection and advanced algorithms"""

    @property
    def name(self) -> str:
        return "clusters"

    def __init__(self):
        self.fast_threshold = 1000
        self.medium_threshold = 5000
        self.large_threshold = 20000
        self.max_sample_size = 3000

    def _count_clusters(self, labels: np.ndarray) -> int:
        unique_labels = set(labels)
        return int(len(unique_labels) - (1 if -1 in unique_labels else 0))

    # --------------------------- Lightning Preprocessing ---------------------------
    def _lightning_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        numeric = data.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError("No numeric data available for clustering")

        # inf -> NaN, then drop rows with any NaN (fast path)
        numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric = numeric.dropna()
        if numeric.empty:
            raise ValueError("No complete rows after NaN handling")

        info: Dict[str, Any] = {}
        n_samples, n_features = numeric.shape

        # drop constant columns
        keep = []
        for c in numeric.columns:
            col = numeric[c].to_numpy()
            if not _is_constant(col):
                keep.append(c)
        
        dropped = len(numeric.columns) - len(keep)
        if dropped > 0:
            numeric = numeric[keep]
        if numeric.shape[1] == 0:
            raise ValueError("All numeric columns are constant.")
        info["dropped_constant_cols"] = dropped

        X = numeric.to_numpy(dtype=float, copy=True)

        # light winsorization for heavy tails
        try:
            _winsorize_inplace(X, q=0.001)
            info["winsorization"] = True
        except Exception:
            info["winsorization"] = False

        # Calculate skewness more robustly
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skewness_values = []
                for col in range(X.shape[1]):
                    col_data = X[:, col]
                    if len(col_data) > 3:  # Need at least 3 points for skewness
                        skew_val = pd.Series(col_data).skew()
                        if np.isfinite(skew_val):
                            skewness_values.append(abs(skew_val))
                
                skew_mean = float(np.mean(skewness_values)) if skewness_values else 0.0
        except Exception:
            skew_mean = 0.0
        
        info["skewness_mean_abs"] = skew_mean

        # scaling choice with better error handling
        try:
            if skew_mean > 2.0 and X.shape[0] > 10:
                try:
                    scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                    X = scaler.fit_transform(X)
                    info["scaling_method"] = "power_transform"
                except Exception:
                    # Fallback to standard scaling
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    info["scaling_method"] = "standard_fallback"
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                info["scaling_method"] = "standard"
        except Exception:
            try:
                scaler = RobustScaler()
                X = scaler.fit_transform(X)
                info["scaling_method"] = "robust_fallback"
            except Exception:
                # Last resort - just normalize each feature
                X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
                info["scaling_method"] = "manual_normalize"

        # dimensionality guard with improved logic
        info["curse_of_dimensionality_risk"] = bool(X.shape[1] > X.shape[0] / 3)
        
        if X.shape[1] > 50 and X.shape[0] > X.shape[1] * 2:
            try:
                # More conservative PCA approach
                n_components = min(50, X.shape[0] // 3, X.shape[1])
                pca = PCA(n_components=n_components, random_state=getattr(config, "random_state", 42))
                Z = pca.fit_transform(X)
                
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                # Look for 95% variance or significant drop in eigenvalues
                k95 = int(np.argmax(cumvar >= 0.95) + 1)
                
                # Only apply PCA if it significantly reduces dimensionality
                if k95 < X.shape[1] * 0.7 and k95 >= 2:
                    X = Z[:, :k95]
                    info["pca_applied"] = True
                    info["pca_variance_explained"] = float(cumvar[k95 - 1])
                    info["final_dimensions"] = int(k95)
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
        n_samples = data.shape[0]
        max_k = int(min(max_k, max(3, n_samples // 5)))
        if max_k < 3:
            return {"optimal_k": 2, "methods": {}, "confidence": "low", "method_agreement": 0}

        rng = np.random.default_rng(42)
        if n_samples > 2000:
            idx = rng.choice(n_samples, 2000, replace=False)
            Xs = data[idx]
        else:
            Xs = data

        methods: Dict[str, int] = {}

        # 1) elbow via minibatch inertia
        try:
            inertias = []
            for k in range(1, max_k + 1):
                if k == 1:
                    inertias.append(float(np.sum(np.var(Xs, axis=0)) * len(Xs)))
                else:
                    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, 
                                       batch_size=min(1000, len(Xs)), max_iter=100)
                    km.fit(Xs)
                    inertias.append(float(km.inertia_))
            
            if len(inertias) >= 4:
                k_elbow = _kneedle_idx(np.array(inertias))
                methods["elbow"] = int(np.clip(k_elbow + 1, 2, max_k))
        except Exception:
            pass

        # 2) silhouette (sampled k range)
        try:
            ks = range(2, min(max_k + 1, 8))
            best = None
            for k in ks:
                km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, 
                                   batch_size=min(500, len(Xs)), max_iter=100)
                lab = km.fit_predict(Xs)
                if np.unique(lab).size > 1:
                    s = _safe_silhouette(Xs, lab, max_points=2000)
                    if best is None or s > best[1]:
                        best = (k, s)
            if best:
                methods["silhouette"] = int(best[0])
        except Exception:
            pass

        # 3) Calinski-Harabasz
        try:
            best = None
            for k in range(2, min(max_k + 1, 6)):
                km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=2, max_iter=100)
                lab = km.fit_predict(Xs)
                if np.unique(lab).size > 1:
                    s = calinski_harabasz_score(Xs, lab)
                    if best is None or s > best[1]:
                        best = (k, s)
            if best:
                methods["calinski_harabasz"] = int(best[0])
        except Exception:
            pass

        # 4) improved gap statistic
        try:
            gap_k = _better_gap_statistic(Xs, min(max_k, 8), B=3)
            methods["gap_statistic"] = max(int(gap_k), 2)
        except Exception:
            pass

        # 5) GMM-BIC vote
        try:
            best = None
            for k in range(2, min(max_k + 1, 8)):
                g = GaussianMixture(n_components=k, covariance_type="diag", n_init=1, 
                                  max_iter=100, random_state=42)
                g.fit(Xs)
                if g.converged_:
                    bic = g.bic(Xs)
                    if best is None or bic < best[1]:
                        best = (k, bic)
            if best:
                methods["gmm_bic"] = int(best[0])
        except Exception:
            pass

        if len(methods) >= 2:
            weights = {"silhouette": 0.35, "calinski_harabasz": 0.25, "elbow": 0.20, 
                      "gap_statistic": 0.10, "gmm_bic": 0.10}
            wsum = sum(weights.get(m, 0.15) * k for m, k in methods.items())
            wtot = sum(weights.get(m, 0.15) for m in methods)
            kopt = int(np.round(wsum / max(wtot, 1e-9)))
            kopt = int(np.clip(kopt, 2, max_k))
            vals = list(methods.values())
            var = np.var(vals) if len(vals) > 1 else 0.0
            conf = "high" if var < 0.5 else ("medium" if var < 2.0 else "low")
            agree = len(set(vals))
        else:
            kopt, conf, agree = min(3, max_k), "low", 0

        return {"optimal_k": kopt, "methods": methods, "confidence": conf, "method_agreement": agree}

    # --------------------------- SOTA KMeans with Stability ---------------------------
    def _sota_kmeans_analysis(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        max_k = int(min(getattr(config, "max_clusters", 10), max(3, len(data) // 5)))
        optimal_k = int(optimal_k_info["optimal_k"])
        k_range = range(max(2, optimal_k - 2), min(max_k + 1, optimal_k + 3))

        best_result, best_score = None, -1.0
        all_results = []
        rng = _rng(config)

        for k in k_range:
            try:
                if len(data) > 10000:
                    KM, extra = MiniBatchKMeans, {"batch_size": min(2000, len(data) // 5)}
                else:
                    KM, extra = KMeans, {}

                labels_runs = []
                models = []
                for init in ["k-means++", "random"]:
                    try:
                        km = KM(n_clusters=k, init=init, n_init=10 if len(data) < 5000 else 5,
                                max_iter=300, random_state=getattr(config, "random_state", 42), **extra)
                        lab = km.fit_predict(data)
                        labels_runs.append(lab)
                        models.append((km, lab))
                    except Exception:
                        continue

                if not models:
                    continue

                # bootstrap stability (cheap)
                stab_scores = []
                boot = 5 if len(data) <= 10000 else 3
                for _ in range(boot):
                    try:
                        idx = rng.choice(len(data), size=min(len(data), 2000), replace=False)
                        labs = []
                        for init in ["k-means++", "random"]:
                            try:
                                km = KM(n_clusters=k, init=init, n_init=1, max_iter=100,
                                        random_state=int(rng.integers(1_000_000)), **extra)
                                labs.append(km.fit_predict(data[idx]))
                            except Exception:
                                continue
                        if len(labs) == 2:
                            stab_scores.append(adjusted_rand_score(labs[0], labs[1]))
                    except Exception:
                        continue
                
                avg_stab = float(np.mean(stab_scores)) if stab_scores else 0.0

                # choose best by silhouette
                model, labels = max(models, key=lambda x: _safe_silhouette(data, x[1]) if np.unique(x[1]).size > 1 else -1)
                s = _safe_silhouette(data, labels)
                
                try:
                    ch = calinski_harabasz_score(data, labels) if np.unique(labels).size > 1 else 0.0
                    db = davies_bouldin_score(data, labels) if np.unique(labels).size > 1 else np.inf
                except Exception:
                    ch, db = 0.0, np.inf

                sizes = np.bincount(labels)
                balance = 1 - (np.std(sizes) / (np.mean(sizes) + 1e-10))
                comp = s * 0.35 + avg_stab * 0.25 + balance * 0.20 + (1 - min(db / 10.0, 1.0)) * 0.20

                res = {
                    "k": int(k),
                    "labels": labels,
                    "model": model,
                    "silhouette": float(s),
                    "calinski_harabasz": float(ch),
                    "davies_bouldin": float(db),
                    "stability": avg_stab,
                    "balance": float(balance),
                    "inertia": float(getattr(model, "inertia_", np.nan)),
                    "cluster_sizes": dict(Counter(labels)),
                    "n_clusters": self._count_clusters(labels),
                }
                all_results.append(res)
                if comp > best_score:
                    best_score, best_result = comp, res
            except Exception:
                continue

        if best_result is None:
            try:
                km = KMeans(n_clusters=2, random_state=42, max_iter=100)
                labels = km.fit_predict(data)
                best_result = {
                    "labels": labels,
                    "model": km,
                    "n_clusters": self._count_clusters(labels),
                    "cluster_sizes": dict(Counter(labels)),
                    "method_type": "centroid_based",
                }
            except Exception:
                return {}

        best_result.update({
            "best_k": int(best_result.get("k", optimal_k)),
            "centers": best_result["model"].cluster_centers_ if hasattr(best_result["model"], "cluster_centers_") else None,
            "method_type": "centroid_based",
            "all_k_results": all_results,
        })
        return best_result

    # --------------------------- Fast Hierarchical ---------------------------
    def _fast_hierarchical_clustering(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        n = data.shape[0]
        kopt = int(optimal_k_info["optimal_k"])
        if n > 1500:
            idx = np.random.choice(n, 1500, replace=False)
            Xs = data[idx]
        else:
            Xs, idx = data, np.arange(n)

        best, best_s = None, -1
        for method in ["ward", "complete", "average"]:
            try:
                agg = AgglomerativeClustering(n_clusters=kopt, linkage=method, metric="euclidean")
                if n > 1500:
                    lab_s = agg.fit_predict(Xs)
                    # Use nearest neighbor assignment for full dataset
                    try:
                        nn = NearestNeighbors(n_neighbors=1).fit(Xs)
                        _, nn_idx = nn.kneighbors(data)
                        lab = lab_s[nn_idx.ravel()]
                    except Exception:
                        # Fallback: assign based on closest centroid
                        centroids = np.array([Xs[lab_s == i].mean(axis=0) for i in np.unique(lab_s)])
                        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
                        lab = np.argmin(distances, axis=1)
                else:
                    lab = agg.fit_predict(data)
                
                s = _safe_silhouette(data, lab)
                if s > best_s:
                    best_s = s
                    best = {
                        "labels": lab,
                        "model": agg,
                        "linkage_method": method,
                        "silhouette": float(s),
                        "n_clusters": self._count_clusters(lab),
                        "cluster_sizes": dict(Counter(lab)),
                        "method_type": "hierarchical",
                    }
            except Exception:
                continue

        if best:
            try:
                L = best["labels"]
                if np.unique(L).size > 1:
                    best["calinski_harabasz"] = float(calinski_harabasz_score(data, L))
                    best["davies_bouldin"] = float(davies_bouldin_score(data, L))
            except Exception:
                pass
        return best if best else {}

    # --------------------------- Smart Density-Based ---------------------------
    def _smart_density_clustering(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results = {}
        n, d = data.shape
        
        # DBSCAN with improved parameter selection
        try:
            k = max(3, min(int(np.sqrt(d) * 2), n // 20, 10))
            if n > 3000:
                idx = np.random.choice(n, 3000, replace=False)
                Xs = data[idx]
            else:
                Xs = data
                
            nn = NearestNeighbors(n_neighbors=k).fit(Xs)
            dist, _ = nn.kneighbors(Xs)
            kdist = np.sort(dist[:, k - 1])
            
            # Better eps selection using knee detection
            eps_idx = _kneedle_idx(kdist)
            eps = float(kdist[eps_idx])
            
            if eps <= 0 or not np.isfinite(eps):
                eps = float(np.percentile(kdist, 90))  # Use 90th percentile as fallback
                
            min_samples = max(3, k)
            
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(data)
            nc = self._count_clusters(labels)
            n_noise = int((labels == -1).sum())
            
            res = {
                "labels": labels,
                "model": db,
                "n_clusters": nc,
                "noise_points": n_noise,
                "noise_ratio": float(n_noise / len(labels)),
                "eps": float(eps),
                "min_samples": int(min_samples),
                "cluster_sizes": dict(Counter(labels[labels != -1])),
                "method_type": "density_based",
            }
            
            if nc > 1 and n_noise < len(labels):
                mask = labels != -1
                if mask.sum() > 1 and np.unique(labels[mask]).size > 1:
                    res["silhouette"] = _safe_silhouette(data[mask], labels[mask])
            results["dbscan"] = res
        except Exception:
            pass

        # HDBSCAN optional (improved error handling)
        try:
            import hdbscan
            if n <= 10000:
                mcs = max(5, n // 100)
                ms = max(1, mcs // 3)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, 
                                          cluster_selection_method="eom", core_dist_n_jobs=1)
                lab = clusterer.fit_predict(data)
                nc = self._count_clusters(lab)
                n_noise = int((lab == -1).sum())
                
                res = {
                    "labels": lab,
                    "model": clusterer,
                    "n_clusters": nc,
                    "noise_points": n_noise,
                    "noise_ratio": float(n_noise / len(lab)),
                    "min_cluster_size": int(mcs),
                    "cluster_sizes": dict(Counter(lab[lab != -1])),
                    "method_type": "density_based",
                }
                if hasattr(clusterer, "probabilities_"):
                    res["probabilities"] = clusterer.probabilities_.tolist()
                if nc > 1 and n_noise < len(lab):
                    mask = lab != -1
                    if mask.sum() > 1 and np.unique(lab[mask]).size > 1:
                        res["silhouette"] = _safe_silhouette(data[mask], lab[mask])
                results["hdbscan"] = res
        except ImportError:
            pass  # HDBSCAN not available
        except Exception:
            pass

        return results

    # --------------------------- Fast Probabilistic ---------------------------
    def _fast_probabilistic_clustering(
        self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        results = {}
        n, d = data.shape
        kopt = int(optimal_k_info["optimal_k"])
        max_k = min(getattr(config, "max_clusters", 10), 8, max(2, n // 10))
        k_range = range(2, max_k + 1)

        # GMM (capped search) with improved error handling
        try:
            cov_types = ["diag", "spherical"] if (d > 20 or n < d * 5) else ["full", "diag"]
            best_gmm, best_bic, best_labels = None, np.inf, None
            
            for cov in cov_types:
                for k in k_range:
                    try:
                        g = GaussianMixture(n_components=k, covariance_type=cov, max_iter=100, n_init=2,
                                            random_state=getattr(config, "random_state", 42), reg_covar=1e-6)
                        g.fit(data)
                        if g.converged_:
                            bic = g.bic(data)
                            if np.isfinite(bic) and bic < best_bic:
                                best_bic, best_gmm = bic, g
                                best_labels = g.predict(data)
                    except Exception:
                        continue
                        
            if best_gmm is not None:
                try:
                    probs = best_gmm.predict_proba(data)
                    res = {
                        "labels": best_labels,
                        "probabilities": probs.tolist(),
                        "model": best_gmm,
                        "n_clusters": self._count_clusters(best_labels),
                        "best_bic": float(best_bic),
                        "cluster_sizes": dict(Counter(best_labels)),
                        "method_type": "probabilistic",
                        "covariance_type": best_gmm.covariance_type,
                    }
                    if np.unique(best_labels).size > 1:
                        res["silhouette"] = _safe_silhouette(data, best_labels)
                    mx = np.max(probs, axis=1)
                    res["mean_certainty"] = float(np.mean(mx))
                    res["assignment_uncertainty"] = float(np.std(mx))
                    results["gmm"] = res
                except Exception:
                    pass
        except Exception:
            pass

        # Bayesian GMM (small n) with improved handling
        if n <= 3000:
            try:
                bg = BayesianGaussianMixture(n_components=min(15, max_k * 2), 
                                             covariance_type="diag" if d > 10 else "full",
                                             max_iter=100, random_state=getattr(config, "random_state", 42),
                                             reg_covar=1e-6)
                bg.fit(data)
                lab = bg.predict(data)
                eff = int(np.sum(bg.weights_ > 0.01))
                
                res = {
                    "labels": lab,
                    "probabilities": bg.predict_proba(data).tolist(),
                    "model": bg,
                    "n_clusters": self._count_clusters(lab),
                    "effective_components": eff,
                    "weights": bg.weights_.tolist(),
                    "cluster_sizes": dict(Counter(lab)),
                    "method_type": "bayesian_probabilistic",
                }
                if np.unique(lab).size > 1:
                    res["silhouette"] = _safe_silhouette(data, lab)
                results["bayesian_gmm"] = res
            except Exception:
                pass

        return results

    # --------------------------- SOTA Advanced Methods ---------------------------
    def _sota_advanced_clustering(self, data: np.ndarray, config: AnalysisConfig, optimal_k_info: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        n = data.shape[0]
        kopt = int(optimal_k_info["optimal_k"])

        # Spectral clustering with improved parameter selection
        if n <= 2000:
            try:
                sample_size = min(500, n)
                sample = data[:sample_size] if n == sample_size else data[np.random.choice(n, sample_size, replace=False)]
                gamma = _median_heuristic_gamma(sample)
                
                sp = SpectralClustering(n_clusters=kopt, affinity="rbf", gamma=gamma,
                                        random_state=getattr(config, "random_state", 42), 
                                        assign_labels="kmeans", n_jobs=1)
                lab = sp.fit_predict(data)
                
                res = {
                    "labels": lab,
                    "model": sp,
                    "n_clusters": self._count_clusters(lab),
                    "cluster_sizes": dict(Counter(lab)),
                    "gamma": float(gamma),
                    "method_type": "spectral",
                }
                if np.unique(lab).size > 1:
                    res["silhouette"] = _safe_silhouette(data, lab)
                results["spectral"] = res
            except Exception:
                pass

        # Mean Shift with better bandwidth estimation
        if n <= 1000:
            try:
                from sklearn.cluster import estimate_bandwidth

                # Use a subset for bandwidth estimation if data is large
                sample_size = min(300, n)
                sample_data = data[:sample_size] if n == sample_size else data[np.random.choice(n, sample_size, replace=False)]
                
                bw = estimate_bandwidth(sample_data, quantile=0.2, n_samples=sample_size)
                if np.isfinite(bw) and bw > 0:
                    ms = MeanShift(bandwidth=bw, n_jobs=1)
                    lab = ms.fit_predict(data)
                    nc = self._count_clusters(lab)
                    if nc > 1 and nc < n // 2:  # Reasonable number of clusters
                        res = {
                            "labels": lab,
                            "model": ms,
                            "n_clusters": nc,
                            "cluster_sizes": dict(Counter(lab)),
                            "bandwidth": float(bw),
                            "n_centers": len(ms.cluster_centers_),
                            "method_type": "mean_shift",
                        }
                        res["silhouette"] = _safe_silhouette(data, lab)
                        results["mean_shift"] = res
            except Exception:
                pass

        # Affinity Propagation with improved preference setting
        if n <= 800:
            try:
                from sklearn.cluster import AffinityPropagation

                # Better preference estimation
                sample_size = min(200, n)
                sample_data = data[:sample_size] if n == sample_size else data[np.random.choice(n, sample_size, replace=False)]
                
                # Use negative median distance as preference
                if sample_size > 1:
                    distances = pdist(sample_data)
                    pref = -np.median(distances) if len(distances) > 0 else -1.0
                else:
                    pref = -1.0
                
                ap = AffinityPropagation(preference=pref, max_iter=200, convergence_iter=15,
                                         random_state=getattr(config, "random_state", 42))
                lab = ap.fit_predict(data)
                nc = self._count_clusters(lab)
                if nc > 1 and nc < n // 3:  # Reasonable number of clusters
                    res = {
                        "labels": lab,
                        "model": ap,
                        "n_clusters": nc,
                        "cluster_sizes": dict(Counter(lab)),
                        "n_exemplars": 0 if ap.cluster_centers_indices_ is None else int(len(ap.cluster_centers_indices_)),
                        "preference": float(pref),
                        "method_type": "exemplar_based",
                    }
                    res["silhouette"] = _safe_silhouette(data, lab)
                    results["affinity_propagation"] = res
            except Exception:
                pass

        return results

    # --------------------------- Improved Ensemble Method ---------------------------
    def _lightning_ensemble(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Simplified but robust ensemble voting method"""
        if len(all_results) < 2:
            return {}

        # Filter valid results
        valid_results = {}
        for name, res in all_results.items():
            if "labels" in res and len(res["labels"]) == len(data):
                labels = np.asarray(res["labels"])
                # Skip if too much noise or degenerate clustering
                if (labels == -1).mean() < 0.7 and len(np.unique(labels[labels != -1])) > 1:
                    valid_results[name] = labels

        if len(valid_results) < 2:
            return {}

        n = len(data)
        # Use simple majority voting with label alignment
        ref_name = next(iter(valid_results.keys()))
        ref_labels = valid_results[ref_name]
        
        # Simple ensemble: for each point, find the most common cluster assignment
        # First, align all labelings to the reference using Hungarian algorithm
        aligned_results = {ref_name: ref_labels}
        
        for name, labels in valid_results.items():
            if name == ref_name:
                continue
                
            try:
                # Align labels to reference
                aligned = self._align_labels(labels, ref_labels)
                aligned_results[name] = aligned
            except Exception:
                # Skip if alignment fails
                continue
        
        if len(aligned_results) < 2:
            return {}
        
        # Majority voting
        final_labels = np.full(n, -1, dtype=int)
        consensus_strength = np.zeros(n)
        
        for i in range(n):
            votes = {}
            total_votes = 0
            
            for labels in aligned_results.values():
                label = labels[i]
                if label != -1:  # Don't count noise votes
                    votes[label] = votes.get(label, 0) + 1
                    total_votes += 1
            
            if votes:
                best_label = max(votes, key=votes.get)
                final_labels[i] = best_label
                consensus_strength[i] = votes[best_label] / max(1, total_votes)
        
        # Calculate overall metrics
        nc = self._count_clusters(final_labels)
        if nc < 2:
            return {}
            
        try:
            sil = _safe_silhouette(data, final_labels)
        except Exception:
            sil = -1.0
        
        avg_consensus = float(np.mean(consensus_strength))
        
        return {
            "labels": final_labels,
            "n_clusters": nc,
            "cluster_sizes": dict(Counter(final_labels[final_labels != -1])),
            "silhouette": sil,
            "consensus_k": nc,
            "consensus_strength": avg_consensus,
            "n_methods": len(aligned_results),
            "participating_methods": list(aligned_results.keys()),
            "method_type": "ensemble",
        }
    
    def _align_labels(self, labels_to_align: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
        """Align cluster labels using Hungarian algorithm"""
        # Only consider non-noise points for alignment
        mask = (labels_to_align != -1) & (reference_labels != -1)
        if not np.any(mask):
            return labels_to_align.copy()
        
        la_clean = labels_to_align[mask]
        ref_clean = reference_labels[mask]
        
        # Get unique labels
        la_unique = np.unique(la_clean)
        ref_unique = np.unique(ref_clean)
        
        if len(la_unique) == 0 or len(ref_unique) == 0:
            return labels_to_align.copy()
        
        # Build confusion matrix
        confusion = np.zeros((len(la_unique), len(ref_unique)))
        for i, la_label in enumerate(la_unique):
            for j, ref_label in enumerate(ref_unique):
                confusion[i, j] = np.sum((la_clean == la_label) & (ref_clean == ref_label))
        
        # Hungarian algorithm to find optimal mapping
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

    # --------------------------- Fast Evaluation ---------------------------
    def _fast_evaluation(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        evaluations: Dict[str, Any] = {}
        for name, res in all_results.items():
            lab = res.get("labels")
            if lab is None or len(lab) != len(data):
                continue
            try:
                L = np.asarray(lab)
                uniq = set(L)
                n_clusters = len(uniq) - (1 if -1 in uniq else 0)
                n_noise = int((L == -1).sum())
                
                ev = {
                    "n_clusters": int(n_clusters),
                    "n_noise_points": n_noise,
                    "noise_ratio": float(n_noise / len(L)),
                    "method_type": res.get("method_type", "unknown"),
                }
                
                # Add silhouette score
                if "silhouette" in res:
                    ev["silhouette_score"] = float(res["silhouette"])
                elif n_clusters > 1:
                    mask = L != -1
                    if mask.sum() > 1 and np.unique(L[mask]).size > 1:
                        ev["silhouette_score"] = _safe_silhouette(data[mask], L[mask])
                
                # Add cluster balance metric
                if n_clusters > 1:
                    nz = L[L != -1]
                    if nz.size > 0:
                        cs = np.bincount(nz)
                        if len(cs) > 1:
                            ev["cluster_balance"] = float(1 - (np.std(cs) / (np.mean(cs) + 1e-10)))
                
                # Add stability if available
                if "stability" in res:
                    ev["stability_score"] = float(res["stability"])
                
                # Add consensus strength if available
                if "consensus_strength" in res:
                    ev["consensus_strength"] = float(res["consensus_strength"])
                    
                evaluations[name] = ev
            except Exception as e:
                evaluations[name] = {"error": str(e)}
                
        return evaluations

    # --------------------------- Smart Recommendations ---------------------------
    def _smart_recommendations(
        self, all_results: Dict[str, Any], evaluations: Dict[str, Any],
        optimal_k_info: Dict[str, Any], preprocessing_info: Dict[str, Any]
    ) -> List[str]:
        recs: List[str] = []
        scores = {}
        
        # Calculate composite scores for each method
        for m, ev in evaluations.items():
            if "error" in ev:
                continue
            
            sc = 0.0
            sil = ev.get("silhouette_score", -1)
            if sil > 0:
                sc += sil * 0.4
            
            sc += ev.get("cluster_balance", 0) * 0.2
            sc += ev.get("stability_score", 0) * 0.2
            sc -= ev.get("noise_ratio", 0) * 0.3
            
            # Bonus for ensemble methods
            if ev.get("method_type") == "ensemble":
                sc += 0.1
            
            # Bonus for optimal k agreement
            nk = ev.get("n_clusters", 0)
            ok = optimal_k_info.get("optimal_k", 3)
            if abs(nk - ok) <= 1:
                sc += 0.1
                
            scores[m] = max(0.0, sc)

        # Generate recommendations based on best method and data characteristics
        if scores:
            best = max(scores, key=scores.get)
            recs.append(f"Best method: {best.upper().replace('_', ' ')} (score: {scores[best]:.3f})")
            
            ev = evaluations[best]
            typ = ev.get("method_type", "")
            
            # Method-specific recommendations
            if "density" in typ:
                if ev.get("noise_ratio", 0) > 0.15:
                    recs.append("High noise ratio detected — consider tuning eps/min_samples or revisiting preprocessing.")
                else:
                    recs.append("Density clustering effectively separated noise from structure.")
            elif typ == "probabilistic":
                br = all_results.get(best, {})
                if "mean_certainty" in br:
                    c = br["mean_certainty"]
                    if c > 0.8:
                        recs.append("High assignment certainty indicates well-separated clusters.")
                    else:
                        recs.append("Moderate certainty suggests overlapping clusters.")
            elif typ == "ensemble":
                cs = ev.get("consensus_strength", 0)
                if cs > 0.7:
                    recs.append("Strong cross-method consensus validates clustering structure.")
                else:
                    recs.append("Moderate consensus — inspect method disagreements.")
            elif typ == "spectral":
                recs.append("Spectral success indicates non-convex cluster shapes.")

        # Data-specific recommendations
        if preprocessing_info.get("curse_of_dimensionality_risk", False):
            recs.append("High dimensionality detected — consider PCA or feature selection.")
        
        conf = optimal_k_info.get("confidence", "low")
        ok = optimal_k_info.get("optimal_k", 3)
        
        if conf == "high":
            recs.append(f"Strong evidence supports {ok} clusters.")
        elif conf == "medium":
            recs.append(f"Consider testing {ok-1} to {ok+1} clusters for validation.")
        else:
            recs.append("Unclear cluster structure — verify if clustering is appropriate for this data.")

        return recs[:5]

    # --------------------------- Adaptive Main Analysis ---------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        try:
            X, preprocessing_info = self._lightning_preprocessing(data, config)
            n_samples, n_features = X.shape

            optimal_k_info = self._fast_optimal_k(X, getattr(config, "max_clusters", 12))

            clustering_results: Dict[str, Any] = {}

            # Always try KMeans
            try:
                km_res = self._sota_kmeans_analysis(X, config, optimal_k_info)
                if km_res:
                    clustering_results["kmeans"] = km_res
            except Exception:
                pass

            # Hierarchical for medium-sized datasets
            if n_samples <= self.medium_threshold:
                try:
                    hier = self._fast_hierarchical_clustering(X, config, optimal_k_info)
                    if hier:
                        clustering_results["hierarchical"] = hier
                except Exception:
                    pass

            # Density-based for reasonable-sized datasets
            if n_samples <= self.large_threshold:
                try:
                    density = self._smart_density_clustering(X, config)
                    clustering_results.update(density)
                except Exception:
                    pass

            # Probabilistic for medium datasets
            if n_samples <= self.medium_threshold:
                try:
                    prob = self._fast_probabilistic_clustering(X, config, optimal_k_info)
                    clustering_results.update(prob)
                except Exception:
                    pass

            # Advanced methods for small datasets
            if n_samples <= self.fast_threshold:
                try:
                    adv = self._sota_advanced_clustering(X, config, optimal_k_info)
                    clustering_results.update(adv)
                except Exception:
                    pass

            # Ensemble if multiple methods succeeded
            if len(clustering_results) >= 2:
                try:
                    ens = self._lightning_ensemble(clustering_results, X)
                    if ens:
                        clustering_results["ensemble"] = ens
                except Exception:
                    pass

            evaluations = self._fast_evaluation(clustering_results, X)
            recommendations = self._smart_recommendations(clustering_results, evaluations, optimal_k_info, preprocessing_info)

            # Determine performance tier
            if n_samples < self.fast_threshold:
                perf_tier, tier_desc = "comprehensive", "All clustering methods available"
            elif n_samples < self.medium_threshold:
                perf_tier, tier_desc = "standard", "Core and probabilistic methods"
            elif n_samples < self.large_threshold:
                perf_tier, tier_desc = "fast", "Fast methods with subsampling"
            else:
                perf_tier, tier_desc = "ultra_fast", "Essential methods only"

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
                    "data_spread": float(np.ptp(X)),
                },
                "recommendations": recommendations,
                "summary": {
                    "methods_attempted": len([r for r in clustering_results.values() if isinstance(r, dict)]),
                    "successful_methods": len([r for r in clustering_results.values() if isinstance(r, dict) and "labels" in r]),
                    "best_method": (max(evaluations.keys(), key=lambda k: evaluations[k].get("silhouette_score", -1))
                                    if evaluations else None),
                    "ensemble_available": "ensemble" in clustering_results,
                    "adaptive_selection": True,
                },
                "performance_info": {
                    "subsampling_used": n_samples > self.max_sample_size,
                    "pca_applied": preprocessing_info.get("pca_applied", False),
                    "parallel_processing": False,  # Set to False for stability
                    "optimization_level": "high",
                },
            }

        except Exception as e:
            return {
                "error": f"Clustering analysis failed: {str(e)}",
                "fallback_available": True,
                "recommendations": ["Consider data preprocessing", "Check data quality", "Verify numeric data availability"],
            }
