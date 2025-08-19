import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import NMF, PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import (
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
    trustworthiness,
)
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd

# Your aux flags (HAS_UMAP, HAS_TRIMAP, HAS_OPENTSNE, etc.)
from .foreminer_aux import *  # noqa

# --------------------------- Small utilities ---------------------------

def _as_float32_if_big(X: np.ndarray, thr_bytes: int = 200_000_000) -> np.ndarray:
    """Cast to float32 when matrix is big to save time/memory."""
    if X.dtype == np.float32:
        return X
    bytes_ = X.size * 8
    return X.astype(np.float32, copy=False) if bytes_ > thr_bytes else X

def _set_default_rng(config) -> np.random.Generator:
    return np.random.default_rng(getattr(config, "random_state", 42))

def _gd_optimal_components(X: np.ndarray) -> int:
    """Gavish–Donoho optimal hard threshold for white noise; returns suggested rank."""
    m, n = X.shape
    beta = m / n if m <= n else n / m
    # omega(beta): optimal hard threshold factor ~ 2.858 * sqrt(beta) for square-ish;
    # we use the exact expression from Gavish-Donoho (approximated).
    # Practical simple rule: keep singular values >= median(s)*omega(beta).
    u, s, vt = randomized_svd(X, n_components=min(min(m, n), 128), random_state=0)
    med = np.median(s)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43  # smooth fit
    k = int(np.sum(s >= med * omega))
    return max(2, min(k, min(m, n)))

def _parallel_analysis_rank(X: np.ndarray, trials: int = 5, rs: int = 0) -> int:
    """Very light parallel analysis: compare eigenvalues to white-noise null."""
    r = min(X.shape)
    # Use randomized PCA for speed
    pca = PCA(n_components=min(128, r), svd_solver="randomized", random_state=rs).fit(X)
    eig = pca.explained_variance_
    rng = np.random.default_rng(rs)
    null_eigs = []
    for _ in range(trials):
        Z = rng.standard_normal(X.shape).astype(X.dtype, copy=False)
        e = PCA(n_components=len(eig), svd_solver="randomized", random_state=rs).fit(Z).explained_variance_
        null_eigs.append(e)
    thr = np.mean(null_eigs, axis=0)
    return int(np.sum(eig > thr[: len(eig)])) or 2

def _choose_rank(X: np.ndarray, cap: int = 64) -> int:
    # Blend of GD threshold and Parallel Analysis, with caps
    k1 = _gd_optimal_components(X)
    k2 = _parallel_analysis_rank(X)
    k = int(np.clip(round(0.5 * (k1 + k2)), 2, min(cap, min(X.shape) - 1)))
    return k

def _kcenter_greedy(X: np.ndarray, k: int, rs: int = 42) -> np.ndarray:
    """Fast approximate coreset sampling (k-center greedy on Euclidean)."""
    rng = np.random.default_rng(rs)
    n = X.shape[0]
    if k >= n:
        return np.arange(n)
    # Start with a random point
    centers = [rng.integers(0, n)]
    # Maintain min distance to any center
    d2 = np.sum((X - X[centers[0]])**2, axis=1)
    for _ in range(1, k):
        i = int(np.argmax(d2))
        centers.append(i)
        d2 = np.minimum(d2, np.sum((X - X[i])**2, axis=1))
    return np.array(centers, dtype=int)

def _procrustes_stability(emb: np.ndarray, boot: int = 3, frac: float = 0.7, rs: int = 42) -> float:
    """Rough stability estimate via Procrustes distance over bootstraps."""
    from scipy.spatial import procrustes
    rng = np.random.default_rng(rs)
    n = emb.shape[0]
    ref_idx = rng.choice(n, size=max(20, int(frac * n)), replace=False)
    ref = emb[ref_idx]
    dists = []
    for b in range(boot):
        idx = rng.choice(n, size=max(20, int(frac * n)), replace=False)
        X1, X2 = ref, emb[idx]
        # pad/trim to min length to compare
        m = min(len(X1), len(X2))
        _, _, d = procrustes(X1[:m], X2[:m])
        dists.append(d)
    return float(1.0 / (1.0 + np.mean(dists)))

def _qnx_knn_preservation(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10) -> float:
    """Fraction of preserved neighbors (Qnx)."""
    nbrs_h = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=-1).fit(X_high)
    nbrs_l = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=-1).fit(X_low)
    H = nbrs_h.kneighbors(return_distance=False)[:, 1:]
    L = nbrs_l.kneighbors(return_distance=False)[:, 1:]
    # Intersection per row
    inter = [len(set(H[i]).intersection(L[i])) / n_neighbors for i in range(len(H))]
    return float(np.mean(inter))

def _mrre(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10) -> float:
    """Mean Relative Rank Error (MRRE): lower is better; we invert for convenience."""
    nbrs_h = NearestNeighbors(n_neighbors=min(n_neighbors * 3 + 1, len(X_high)), n_jobs=-1).fit(X_high)
    nbrs_l = NearestNeighbors(n_neighbors=min(n_neighbors * 3 + 1, len(X_low)), n_jobs=-1).fit(X_low)
    H = nbrs_h.kneighbors(return_distance=False)[:, 1:]
    L = nbrs_l.kneighbors(return_distance=False)[:, 1:]
    inv_index_L = {}
    for i in range(L.shape[0]):
        inv = np.empty(L.shape[1], dtype=int)
        rank = {j: r for r, j in enumerate(L[i])}
        for r, j in enumerate(H[i]):
            inv[r] = rank.get(j, L.shape[1]-1)
        inv_index_L[i] = inv
    errs = []
    for i in range(len(H)):
        # relative rank error
        rr = (inv_index_L[i] - np.arange(len(inv_index_L[i]))) / max(1, L.shape[1])
        errs.append(np.mean(np.clip(rr, 0, 1)))
    mrre = float(np.mean(errs))
    return float(1.0 - mrre)  # higher is better now

# --------------------------- Main class ---------------------------

class DimensionalityAnalyzer(AnalysisStrategy):
    """SOTA-ish dimensionality analysis with shared kNN graph, robust metrics, and fast backends."""

    @property
    def name(self) -> str:
        return "dimensionality"

    # --------------------------- Public API ---------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        try:
            scaled_data, preprocessing_info = self._adaptive_preprocessing(data, config)

            # Pre-reduce + whiten before nonlinear methods for speed/stability
            pre_X, pre_info = self._prereduce_whiten(scaled_data, config)
            embeddings = self._dimensionality_reduction(scaled_data, pre_X, pre_info, config)
            if not embeddings:
                return {"error": "All dimensionality reduction methods failed"}

            evaluation_results = self._evaluate_embeddings(embeddings, scaled_data, pre_X)

            # Stable condition number
            u, s, _ = np.linalg.svd(scaled_data, full_matrices=False)
            cond_num = float(s.max() / max(s.min(), 1e-12))

            results = {
                "embeddings": embeddings,
                "evaluation": evaluation_results,
                "preprocessing_info": {**preprocessing_info, **pre_info},
                "data_characteristics": {
                    "n_samples": int(scaled_data.shape[0]),
                    "n_features": int(scaled_data.shape[1]),
                    "condition_number": cond_num,
                    "effective_rank": int(np.linalg.matrix_rank(scaled_data)),
                },
                "recommendations": self._generate_recommendations(
                    embeddings, evaluation_results, {**preprocessing_info, **pre_info}
                ),
            }
            return results

        except Exception as e:
            return {"error": f"Dimensionality analysis failed: {e}"}

    # --------------------------- Preprocessing ---------------------------
    def _adaptive_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        rng = _set_default_rng(config)
        numeric = data.select_dtypes(include=[np.number]).dropna()
        if numeric.empty or numeric.shape[1] < 2:
            raise ValueError("Insufficient numeric data for dimensionality reduction")

        info: Dict[str, Any] = {}

        # 1) Low variance filter
        vt = VarianceThreshold(threshold=0.01)
        X = pd.DataFrame(
            vt.fit_transform(numeric),
            columns=numeric.columns[vt.get_support()],
            index=numeric.index,
        )
        info["features_removed"] = int((~vt.get_support()).sum())

        # 2) Adaptive sampling (k-center coreset when very large; fallback to MiniBatchKMeans)
        sample_cap = int(getattr(config, "sample_size_threshold", 2000))
        if len(X) > sample_cap:
            try:
                Z = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
                idx = _kcenter_greedy(Z, k=sample_cap, rs=getattr(config, "random_state", 42))
                X = X.iloc[idx]
                info["sampling_method"] = "kcenter_coreset"
            except Exception:
                try:
                    n_clusters = int(np.clip(len(X) // 100, 2, 10))
                    Z = StandardScaler().fit_transform(X)
                    km = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        random_state=getattr(config, "random_state", 42),
                        batch_size=512,
                        n_init=5,
                    )
                    clusters = km.fit_predict(Z)
                    indices: List[int] = []
                    for cid in np.unique(clusters):
                        idx = np.flatnonzero(clusters == cid)
                        n_take = max(1, int(sample_cap * len(idx) / len(X)))
                        take = min(n_take, len(idx))
                        if take > 0:
                            indices.extend(rng.choice(idx, size=take, replace=False))
                    X = X.iloc[indices[:sample_cap]]
                    info["sampling_method"] = "stratified_minibatchkmeans"
                except Exception:
                    idx = rng.choice(len(X), size=sample_cap, replace=False)
                    X = X.iloc[idx]
                    info["sampling_method"] = "random"

        # 3) Scaling (score by condition no. + variance spread)
        skewness = X.skew().abs().mean()
        outlier_frac = ((X - X.mean()).abs() > 3 * X.std(ddof=0)).sum().sum() / X.size
        if (skewness > 2) or (outlier_frac > 0.10):
            scalers = [
                ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
                ("robust", RobustScaler()),
                ("standard", StandardScaler()),
            ]
        else:
            scalers = [("standard", StandardScaler()), ("robust", RobustScaler())]

        def _score_scaled(arr: np.ndarray) -> float:
            u, s, _ = np.linalg.svd(arr, full_matrices=False)
            cond = s.max() / max(s.min(), 1e-12)
            v = np.var(arr, axis=0)
            return -np.log(cond + 1e-12) - np.log((v.max() / max(v.min(), 1e-12)) + 1e-12)

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
        # Memory friendly dtype
        scaled_X = _as_float32_if_big(np.asarray(scaled_X))
        return scaled_X, info

    # --------------------------- Pre-reduction + whitening ---------------------------
    def _prereduce_whiten(
        self, X: np.ndarray, config: AnalysisConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        n, d = X.shape
        info: Dict[str, Any] = {}
        # Target dim for nonlinear pre-space
        k_cap = int(getattr(config, "prereduce_cap", 64))
        k = min(_choose_rank(X), k_cap)
        info["prereduce_dim"] = int(k)
        if d <= k:
            info["prereduce_method"] = "none"
            return X, info

        if d > 1000 and n < d:
            # very wide -> TruncatedSVD is great
            svd = TruncatedSVD(n_components=k, random_state=getattr(config, "random_state", 42))
            Z = svd.fit_transform(X)
            comps = svd.components_
            var = svd.explained_variance_
        else:
            pca = PCA(n_components=k, svd_solver="randomized", whiten=True,
                      random_state=getattr(config, "random_state", 42))
            Z = pca.fit_transform(X)
            comps = pca.components_
            var = pca.explained_variance_

        info["prereduce_method"] = "pca_whiten" if 'pca' in locals() else "tsvd"
        info["prereduce_total_variance"] = float(np.sum(var)) if var is not None else None
        Z = _as_float32_if_big(Z)
        return Z, info

    # --------------------------- Shared neighbor graph ---------------------------
    def _shared_knn(self, Z: np.ndarray, n_neighbors: int) -> NearestNeighbors:
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1)
        nn.fit(Z)
        return nn

    # --------------------------- Dimensionality Reduction ---------------------------
    def _dimensionality_reduction(
        self,
        scaled_data: np.ndarray,
        pre_X: np.ndarray,
        pre_info: Dict[str, Any],
        config: AnalysisConfig,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        n_samples, n_features = scaled_data.shape
        rs = getattr(config, "random_state", 42)
        target_dim = 2

        # Linear (on original scaled_data with optimal n_components)
        n_comp = min(_choose_rank(scaled_data), 64)
        linear_methods = {
            "pca": PCA(n_components=n_comp, random_state=rs, svd_solver="randomized"),
            "ica": FastICA(n_components=min(n_comp, n_features-1), random_state=rs, max_iter=500, tol=1e-3),
            "factor_analysis": FactorAnalysis(n_components=min(n_comp, n_features-1), random_state=rs),
        }
        if np.all(scaled_data >= 0):
            linear_methods["nmf"] = NMF(n_components=min(n_comp, n_features-1), random_state=rs, max_iter=500)
        if n_features > 100:
            linear_methods["truncated_svd"] = TruncatedSVD(n_components=n_comp, random_state=rs)

        for name, method in linear_methods.items():
            try:
                emb = method.fit_transform(scaled_data)
                entry = {"embedding": emb, "method_type": "linear", "n_components": int(n_comp)}
                if hasattr(method, "explained_variance_ratio_"):
                    evr = method.explained_variance_ratio_
                    entry["explained_variance_ratio"] = evr
                    entry["total_variance_explained"] = float(np.sum(evr))
                results[name] = entry
            except Exception as e:
                print(f"Linear method {name} failed: {e}")

        # Nonlinear 2D on pre_X (prereduced + whitened)
        n_neighbors = int(np.clip(pre_X.shape[0] // 10, 5, 64))
        knn = self._shared_knn(pre_X, n_neighbors=n_neighbors)

        # --- t-SNE: prefer openTSNE -> MulticoreTSNE -> sklearn
        tsne_done = False
        try:
            if 'HAS_OPENTSNE' in globals() and HAS_OPENTSNE:
                from openTSNE import TSNE as OTSNE
                ot = OTSNE(
                    n_components=target_dim, perplexity=min(30, max(5, pre_X.shape[0] // 100)),
                    negative_gradient_method="fft", n_jobs=-1, random_state=rs, initialization="pca",
                )
                results["tsne_opentsne"] = {"embedding": ot.fit(pre_X), "method_type": "nonlinear"}
                tsne_done = True
        except Exception as e:
            print(f"openTSNE failed: {e}")

        if not tsne_done:
            try:
                from MulticoreTSNE import MulticoreTSNE as MTSNE
                mt = MTSNE(
                    n_components=target_dim, perplexity=min(30, max(5, pre_X.shape[0] // 100))),
                emb = mt.fit_transform(pre_X)
                results["tsne_multicore"] = {"embedding": emb, "method_type": "nonlinear"}
                tsne_done = True
            except Exception:
                pass

        if not tsne_done:
            try:
                tsne = TSNE(
                    n_components=target_dim, random_state=rs,
                    perplexity=min(30, max(5, pre_X.shape[0] // 100)),
                    init="pca", learning_rate="auto",
                )
                results["tsne"] = {"embedding": tsne.fit_transform(pre_X), "method_type": "nonlinear"}
            except Exception as e:
                print(f"t-SNE failed: {e}")

        # --- UMAP (and densMAP if available)
        if 'HAS_UMAP' in globals() and HAS_UMAP:
            try:
                from umap import UMAP
                umap_reducer = UMAP(
                    n_components=target_dim, random_state=rs,
                    n_neighbors=n_neighbors, min_dist=0.1, metric="euclidean",
                )
                results["umap"] = {"embedding": umap_reducer.fit_transform(pre_X), "method_type": "nonlinear"}
                # small hyperparam sweep on a subsample for trustworthiness
                results.update(self._umap_micro_sweep(pre_X, rs))
            except Exception as e:
                print(f"UMAP failed: {e}")
            try:
                from umap import UMAP as _UMAP
                umap_dens = _UMAP(
                    n_components=target_dim, random_state=rs, densmap=True,
                    n_neighbors=n_neighbors, min_dist=0.1
                )
                results["densmap"] = {"embedding": umap_dens.fit_transform(pre_X), "method_type": "nonlinear"}
            except Exception:
                pass

        # --- PaCMAP
        try:
            import pacmap
            pac = pacmap.PaCMAP(n_components=target_dim, n_neighbors=n_neighbors, random_state=rs)
            results["pacmap"] = {"embedding": pac.fit_transform(pre_X), "method_type": "nonlinear"}
        except Exception:
            pass

        # --- PHATE
        try:
            import phate
            ph = phate.PHATE(n_components=target_dim, random_state=rs)
            results["phate"] = {"embedding": ph.fit_transform(pre_X), "method_type": "nonlinear"}
        except Exception:
            pass

        # --- Manifold learners using shared kNN
        manifold = {
            "isomap": Isomap(n_components=target_dim, n_neighbors=n_neighbors),
            "lle": LocallyLinearEmbedding(n_components=target_dim, n_neighbors=n_neighbors, random_state=rs),
            "spectral": SpectralEmbedding(n_components=target_dim, n_neighbors=n_neighbors, random_state=rs),
        }
        for name, method in manifold.items():
            try:
                emb = method.fit_transform(pre_X)
                results[name] = {"embedding": emb, "method_type": "nonlinear"}
            except Exception:
                pass

        return results

    def _umap_micro_sweep(self, pre_X: np.ndarray, rs: int) -> Dict[str, Any]:
        """Very small sweep for (n_neighbors, min_dist) on a subsample, pick best by trustworthiness."""
        out: Dict[str, Any] = {}
        if pre_X.shape[0] < 400 or not ('HAS_UMAP' in globals() and HAS_UMAP):
            return out
        try:
            from umap import UMAP
            rng = np.random.default_rng(rs)
            idx = rng.choice(pre_X.shape[0], size=400, replace=False)
            Xs = pre_X[idx]
            candidates = [(15, 0.1), (30, 0.1), (30, 0.01), (50, 0.1)]
            best = None
            best_score = -np.inf
            for nn, md in candidates:
                try:
                    em = UMAP(n_components=2, n_neighbors=nn, min_dist=md, random_state=rs).fit_transform(Xs)
                    sc = trustworthiness(Xs, em, n_neighbors=10)
                    if sc > best_score:
                        best, best_score = ("umap_nn{}_md{}".format(nn, md), sc)
                        out[best] = {"embedding": em, "method_type": "nonlinear", "note": "sweep_candidate"}
                except Exception:
                    pass
            return out
        except Exception:
            return out

    # --------------------------- Evaluation ---------------------------
    def _evaluate_embeddings(
        self,
        embeddings: Dict[str, Any],
        X_high: np.ndarray,
        pre_X: np.ndarray,
    ) -> Dict[str, Any]:
        evals: Dict[str, Any] = {}
        rs = 42

        # Decide if we need sampling for expensive metrics
        n = pre_X.shape[0]
        sample_idx: Optional[np.ndarray] = None
        if n > 3000:
            rng = np.random.default_rng(rs)
            sample_idx = rng.choice(n, size=2000, replace=False)
            X_eval = pre_X[sample_idx]
        else:
            X_eval = pre_X

        for name, res in embeddings.items():
            emb = res.get("embedding")
            if emb is None or getattr(emb, "shape", (0, 0))[0] < 10:
                continue

            # Align embedding samples if we sampled
            Y = emb if sample_idx is None else emb[sample_idx]

            metrics: Dict[str, float] = {}

            # Clustering quality (on Y)
            try:
                n_clusters = int(np.clip(Y.shape[0] // 10, 2, 8))
                km = KMeans(n_clusters=n_clusters, random_state=rs, n_init=5)
                labels = km.fit_predict(Y)
                if len(np.unique(labels)) > 1:
                    metrics["silhouette_score"] = float(silhouette_score(Y, labels))
            except Exception:
                pass

            # Trustworthiness / Continuity
            try:
                metrics["trustworthiness"] = float(trustworthiness(X_eval, Y, n_neighbors=10))
                metrics["continuity"] = float(trustworthiness(Y, X_eval, n_neighbors=10))
            except Exception:
                pass

            # Neighborhood preservation (Qnx) & MRRE
            try:
                metrics["knn_preservation_qnx"] = _qnx_knn_preservation(X_eval, Y, n_neighbors=10)
            except Exception:
                pass
            try:
                metrics["mrre_inverted"] = _mrre(X_eval, Y, n_neighbors=10)
            except Exception:
                pass

            # Stability via Procrustes bootstraps
            try:
                metrics["embedding_stability"] = _procrustes_stability(Y, boot=3, frac=0.7, rs=rs)
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
        # more informative weighting
        weights = {
            "trustworthiness": 0.35,
            "knn_preservation_qnx": 0.2,
            "mrre_inverted": 0.2,
            "continuity": 0.15,
            "embedding_stability": 0.05,
            "silhouette_score": 0.05,
        }

        scores: Dict[str, float] = {}
        for name, mets in evaluations.items():
            score = 0.0
            for k, w in weights.items():
                v = mets.get(k, np.nan)
                if np.isfinite(v):
                    score += w * float(v)
            scores[name] = score

        if scores:
            best = max(scores, key=scores.get)
            recs.append(f"Best performing method: {best}")
            if "umap" in best or "densmap" in best:
                recs.append("UMAP/densMAP preserved local neighborhoods well; consider these for clustering/visualization.")
            elif "tsne" in best:
                recs.append("t-SNE highlighted local clusters; use for cluster separation visuals.")
            elif "pacmap" in best:
                recs.append("PaCMAP balanced local/global structure; good general-purpose map.")
            elif "phate" in best:
                recs.append("PHATE captured continuous trajectories; useful for temporal/trajectory data.")
            elif best == "pca":
                recs.append("PCA dominance suggests largely linear structure; consider linear models or ICA for interpretability.")

        if preprocessing_info.get("features_removed", 0) > 0:
            recs.append(f"Removed {preprocessing_info['features_removed']} low-variance features.")
        if preprocessing_info.get("scaling_method") == "power_transform":
            recs.append("Applied power transformation due to skewness/outliers.")
        if preprocessing_info.get("prereduce_method") in ("pca_whiten", "tsvd"):
            recs.append(f"Pre-reduced to {preprocessing_info.get('prereduce_dim')} dims for stability and speed.")

        return recs
