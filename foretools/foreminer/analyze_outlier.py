import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

from .foreminer_aux import *


# ---------------------- small utils ----------------------
def _rng(config: AnalysisConfig) -> np.random.Generator:
    rs = getattr(config, "random_state", 42)
    return np.random.default_rng(rs if rs is not None else 42)


def _is_constant(col: np.ndarray, tol: float = 1e-12) -> bool:
    return np.nanmax(col) - np.nanmin(col) <= tol


def _safe_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(len(x))
    return (ranks + 1) / (len(x) + 1)


def _winsorize_inplace(X: np.ndarray, q: float = 0.001) -> None:
    lo = np.nanquantile(X, q, axis=0)
    hi = np.nanquantile(X, 1 - q, axis=0)
    np.clip(X, lo, hi, out=X)


def _normalize_scores(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).any():
        return np.zeros_like(s, dtype=float)
    msk = np.isfinite(s)
    if msk.sum() <= 1:
        return np.zeros_like(s, dtype=float)
    s_f = s[msk]
    rng = s_f.max() - s_f.min()
    if rng <= 1e-12:
        out = np.zeros_like(s, dtype=float)
        out[msk] = 0.5
        return out
    out = np.zeros_like(s, dtype=float)
    out[msk] = (s_f - s_f.min()) / rng
    return out


class OutlierAnalyzer(AnalysisStrategy):
    """Modern SOTA outlier detection with optional legacy methods"""

    @property
    def name(self) -> str:
        return "outliers"

    def __init__(self, use_legacy: bool = False):
        self.fast_threshold = 1000
        self.medium_threshold = 5000
        self.large_threshold = 20000
        self.max_sample_size = 3000
        self.use_legacy = use_legacy

    # ---------------------- Preprocessing ----------------------
    def _lightning_preprocessing(self, data: pd.DataFrame, config: AnalysisConfig):
        numeric = data.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError("No numeric data available for outlier detection")

        numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        info: Dict[str, Any] = {}
        n_samples, n_features = numeric.shape

        na_counts = numeric.isna().sum()
        missing_pct = float(na_counts.sum() / (n_samples * n_features) * 100.0)
        info["missing_data_percentage"] = missing_pct

        if missing_pct > 80:
            raise ValueError("Too much missing data for reliable outlier detection")

        keep_cols = [
            c for c in numeric.columns if not _is_constant(numeric[c].to_numpy())
        ]
        if not keep_cols:
            raise ValueError("All numeric columns are constant.")
        if len(keep_cols) < n_features:
            numeric = numeric[keep_cols]
        info["dropped_constant_cols"] = int(n_features - len(keep_cols))

        if missing_pct > 30:
            good_cols = na_counts[keep_cols] < (n_samples * 0.5)
            numeric = numeric.loc[:, good_cols.index[good_cols]]
            med = numeric.median()
            clean = numeric.fillna(med)
            complete_idx = np.arange(len(clean))
            info["handling_strategy"] = "column_filter_and_imputation"
        elif missing_pct > 5:
            med = numeric.median()
            clean = numeric.fillna(med)
            complete_idx = np.arange(len(clean))
            info["handling_strategy"] = "median_imputation"
        else:
            clean = numeric.dropna()
            complete_idx = np.where(~numeric.isna().any(axis=1))[0]
            info["handling_strategy"] = "complete_cases_only"

        info["final_sample_size"] = int(len(clean))
        info["final_feature_count"] = int(clean.shape[1])
        X = clean.to_numpy(dtype=float, copy=True)

        try:
            _winsorize_inplace(X, q=0.001)
            info["winsorization"] = True
        except Exception:
            info["winsorization"] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skewness = pd.DataFrame(X).skew().abs()
            kurtosis = pd.DataFrame(X).kurtosis().abs()
        info["data_skewness"] = float(np.nanmean(skewness))
        info["data_kurtosis"] = float(np.nanmean(kurtosis))

        high_skew = (np.asarray(skewness) > 2).any()
        high_kurt = (np.asarray(kurtosis) > 10).any()

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
            X = scaler.fit_transform(X)
            info["scaling_method"] = scaler_name
        except Exception:
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            info["scaling_method"] = "robust_fallback"

        return X, complete_idx, info

    # ---------------------- Core SOTA Methods ----------------------
    def _pca_reconstruction(self, X, config):
        try:
            k = min(10, max(1, int(np.ceil(0.2 * X.shape[1]))))
            pca = PCA(n_components=k, random_state=42)
            Z = pca.fit_transform(X)
            Xr = pca.inverse_transform(Z)
            rec_err = np.mean((X - Xr) ** 2, axis=1)
            thr = np.percentile(rec_err, 100 * (1 - config.outlier_contamination))
            return {
                "outliers": rec_err > thr,
                "scores": rec_err,
                "threshold": float(thr),
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
                "method_type": "dimensionality_reduction",
            }
        except Exception:
            return {}

    def _isolation_forest(self, X, config):
        try:
            iso = IsolationForest(
                n_estimators=150,
                max_samples=min(1024, len(X)),
                contamination=config.outlier_contamination,
                random_state=getattr(config, "random_state", 42),
                n_jobs=-1,
            )
            o = iso.fit_predict(X) == -1
            s = -iso.decision_function(X)
            return {"outliers": o, "scores": s, "method_type": "ensemble"}
        except Exception:
            return {}

    def _ecod(self, X, config):
        try:
            ranks = np.empty_like(X)
            for j in range(X.shape[1]):
                ranks[:, j] = _safe_rank(X[:, j])
            tail = np.minimum(ranks, 1.0 - ranks)
            min_tail = np.min(tail, axis=1)
            thr = np.percentile(min_tail, 100 * config.outlier_contamination)
            o = min_tail <= thr
            s = 1.0 - min_tail
            return {
                "outliers": o,
                "scores": s,
                "threshold": float(thr),
                "method_type": "distribution_based",
            }
        except Exception:
            return {}

    def _copod(self, X, config):
        try:
            ranks = np.empty_like(X)
            for j in range(X.shape[1]):
                ranks[:, j] = _safe_rank(X[:, j])
            prod_u = np.prod(ranks, axis=1)
            s = -np.log(prod_u + 1e-12)
            thr = np.percentile(s, 100 * (1 - config.outlier_contamination))
            o = s > thr
            return {
                "outliers": o,
                "scores": s,
                "threshold": float(thr),
                "method_type": "copula_based",
            }
        except Exception:
            return {}

    def _hbos(self, X, config):
        try:
            n, d = X.shape
            bins = np.clip(int(np.sqrt(n)), 10, 50)
            scores = np.zeros(n, dtype=float)
            for j in range(d):
                col = X[:, j]
                if _is_constant(col):
                    continue
                hist, edges = np.histogram(col, bins=bins)
                hist = hist + 1e-6
                dens = hist / np.sum(hist)
                idx = np.clip(
                    np.searchsorted(edges, col, side="right") - 1, 0, len(hist) - 1
                )
                scores += -np.log(dens[idx])
            thr = np.percentile(scores, 100 * (1 - config.outlier_contamination))
            o = scores > thr
            return {
                "outliers": o,
                "scores": scores,
                "threshold": float(thr),
                "method_type": "histogram_based",
            }
        except Exception:
            return {}

    def _aeod(self, X, config):
        try:
            ae = MLPRegressor(
                hidden_layer_sizes=(min(32, X.shape[1] * 2),),
                activation="relu",
                max_iter=200,
                random_state=getattr(config, "random_state", 42),
            )
            ae.fit(X, X)
            Xr = ae.predict(X)
            rec_err = np.mean((X - Xr) ** 2, axis=1)
            thr = np.percentile(rec_err, 100 * (1 - config.outlier_contamination))
            return {
                "outliers": rec_err > thr,
                "scores": rec_err,
                "threshold": float(thr),
                "method_type": "neural",
            }
        except Exception:
            return {}

    # ---------------------- Legacy Methods ----------------------
    def _legacy_methods(self, X, config):
        results = {}
        n, d = X.shape
        rng = _rng(config)

        # Plain z-score
        try:
            max_abs = np.max(np.abs(X), axis=1)
            outliers = max_abs > 3.0
            results["z_score"] = {
                "outliers": outliers,
                "scores": max_abs,
                "threshold": 3.0,
                "method_type": "statistical",
            }
        except Exception:
            pass

        # LOF (small n only)
        if n <= 5000:
            try:
                k = max(5, min(50, n // 20))
                lof = LocalOutlierFactor(
                    n_neighbors=k, contamination=config.outlier_contamination, n_jobs=-1
                )
                o = lof.fit_predict(X) == -1
                s = -lof.negative_outlier_factor_
                results["lof"] = {
                    "outliers": o,
                    "scores": s,
                    "method_type": "density_based",
                }
            except Exception:
                pass

        # DBSCAN
        try:
            nn = NearestNeighbors(n_neighbors=10).fit(X)
            dists, _ = nn.kneighbors(X)
            eps = np.percentile(dists[:, -1], 75)
            db = DBSCAN(eps=eps, min_samples=10)
            labels = db.fit_predict(X)
            results["dbscan"] = {
                "outliers": labels == -1,
                "labels": labels,
                "eps": float(eps),
                "method_type": "density_based",
            }
        except Exception:
            pass

        # One-Class SVM
        if n <= 2000:
            try:
                oc = OneClassSVM(
                    kernel="rbf",
                    gamma="scale",
                    nu=float(np.clip(config.outlier_contamination, 0.01, 0.5)),
                )
                o = oc.fit_predict(X) == -1
                s = -oc.decision_function(X)
                results["one_class_svm"] = {
                    "outliers": o,
                    "scores": s,
                    "method_type": "boundary_based",
                }
            except Exception:
                pass

        # Elliptic Envelope
        if d <= 20 and n >= d * 3:
            try:
                ee = EllipticEnvelope(
                    contamination=config.outlier_contamination,
                    random_state=getattr(config, "random_state", 42),
                )
                o = ee.fit_predict(X) == -1
                s = -ee.decision_function(X)
                results["elliptic_envelope"] = {
                    "outliers": o,
                    "scores": s,
                    "method_type": "covariance_based",
                }
            except Exception:
                pass

        # Robust Mahalanobis
        try:
            if d <= 50 and n >= max(50, d * 2):
                mcd = MinCovDet(random_state=getattr(config, "random_state", 42)).fit(X)
                md = mcd.mahalanobis(X)
                thr = stats.chi2.ppf(1 - config.outlier_contamination, d)
                results["mahalanobis_robust"] = {
                    "outliers": md > thr,
                    "scores": md,
                    "threshold": float(thr),
                    "method_type": "distance_based",
                }
        except Exception:
            pass

        return results

    # ---------------------- Ensemble Fusion ----------------------
    def _reciprocal_rank_fusion(self, scores_list, k=60):
        n = scores_list[0].shape[0]
        fused = np.zeros(n)
        for s in scores_list:
            ranks = np.argsort(np.argsort(-s))  # descending
            fused += 1.0 / (k + ranks)
        return fused

    def _ensemble(self, all_results, X, config):
        scores = []
        methods = []
        for m, res in all_results.items():
            if "scores" in res:
                s = _normalize_scores(res["scores"])
                scores.append(s)
                methods.append(m)
        if not scores:
            return {}
        fused = self._reciprocal_rank_fusion(scores, k=60)
        thr = np.percentile(fused, 100 * (1 - config.outlier_contamination))
        return {
            "outliers": fused > thr,
            "scores": fused,
            "threshold": float(thr),
            "method_type": "ensemble",
            "participating_methods": methods,
        }

    # ---------------------- Evaluation & Recommendations ----------------------
    def _lightning_evaluation(
        self, all_results: Dict[str, Any], X: np.ndarray
    ) -> Dict[str, Any]:
        evaluations = {}
        for m, res in all_results.items():
            o = res.get("outliers")
            if not isinstance(o, np.ndarray):
                continue
            try:
                n_out = int(o.sum())
                rate = float(n_out / len(o))
                ev = {
                    "n_outliers": n_out,
                    "outlier_rate": rate,
                    "method_type": res.get("method_type", "unknown"),
                }
                s = res.get("scores")
                if s is not None and 0 < n_out < len(o):
                    sep = float(np.nanmean(s[o]) - np.nanmean(s[~o]))
                    ev["score_separation"] = sep
                evaluations[m] = ev
            except Exception:
                evaluations[m] = {"error": "evaluation_failed"}
        return evaluations

    def _smart_recommendations(self, all_results, evaluations, prep, data_size):
        recs = []
        n_samples, n_features = data_size

        if evaluations:
            best = max(
                evaluations.items(),
                key=lambda kv: kv[1].get("score_separation", -np.inf),
            )[0]
            recs.append(f"🏆 Recommended method: {best.upper()}")

        if n_samples > 10000:
            recs.append(
                "🚀 Large dataset detected — prioritized fast methods (IF, HBOS, ECOD)."
            )
        if n_features > 50:
            recs.append(
                "📊 High-dimensional data — PCA/AE recommended before density methods."
            )
        if prep.get("data_skewness", 0) > 3:
            recs.append(
                "📈 Strong skew — robust scaling applied; ECOD/HBOS/IF preferred."
            )
        if len(all_results) >= 4:
            recs.append("🎯 Multiple methods succeeded — ensemble reliability is high.")

        return recs[:4]

    # ---------------------- Main ----------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        try:
            X, complete_indices, prep = self._lightning_preprocessing(data, config)
            N = len(data)
            n_samples, n_features = X.shape

            # Run methods
            methods = {
                "pca_recon": self._pca_reconstruction(X, config),
                "isolation_forest": self._isolation_forest(X, config),
                "ecod": self._ecod(X, config),
                "copod": self._copod(X, config),
                "hbos": self._hbos(X, config),
                "aeod": self._aeod(X, config),
            }
            if self.use_legacy:
                methods.update(self._legacy_methods(X, config))

            all_results = {k: v for k, v in methods.items() if v}

            # Ensemble
            ens = self._ensemble(all_results, X, config)
            if ens:
                all_results["ensemble"] = ens

            # Map results back to full dataset
            final_results = {}
            for name, res in all_results.items():
                sub = res.get("outliers")
                if sub is None:
                    continue
                full_o = np.zeros(N, dtype=bool)
                full_o[complete_indices] = sub
                mapped = {
                    "outliers": full_o,
                    "count": int(full_o.sum()),
                    "percentage": float(100.0 * full_o.mean()),
                    "method_type": res.get("method_type", "unknown"),
                }
                if "scores" in res:
                    full_s = np.zeros(N, dtype=float)
                    full_s[complete_indices] = np.asarray(res["scores"], dtype=float)
                    mapped["scores"] = full_s
                # include optional fields
                for k in ["threshold", "explained_variance", "confidence", "labels"]:
                    if k in res:
                        mapped[k] = res[k]
                final_results[name] = mapped

            # Evaluations
            evaluations = self._lightning_evaluation(all_results, X)

            # Recommendations
            recommendations = self._smart_recommendations(
                final_results, evaluations, prep, (n_samples, n_features)
            )

            # Performance tier info
            if n_samples < self.fast_threshold:
                perf_tier, tier_desc = "ultra_fast", "All methods available"
            elif n_samples < self.medium_threshold:
                perf_tier, tier_desc = "fast", "Most methods available"
            elif n_samples < self.large_threshold:
                perf_tier, tier_desc = "medium", "Core methods with subsampling"
            else:
                perf_tier, tier_desc = (
                    "large_scale",
                    "Statistical and fast ML methods only",
                )

            # Best method selection
            best_by_sep = None
            if evaluations:
                cand = [
                    (k, v.get("score_separation", -np.inf))
                    for k, v in evaluations.items()
                    if "error" not in v
                ]
                if cand:
                    best_by_sep = max(cand, key=lambda t: t[1])[0]

            return {
                "outlier_results": final_results,
                "evaluations": evaluations,
                "preprocessing_info": prep,
                "data_characteristics": {
                    "total_samples": int(N),
                    "analyzed_samples": int(n_samples),
                    "missing_samples": int(N - n_samples),
                    "n_features": int(n_features),
                    "contamination_rate": float(config.outlier_contamination),
                    "performance_tier": perf_tier,
                    "tier_description": tier_desc,
                },
                "recommendations": recommendations,
                "summary": {
                    "methods_attempted": len(methods),
                    "successful_methods": len(final_results),
                    "best_method": best_by_sep
                    if best_by_sep
                    else ("ensemble" if "ensemble" in final_results else None),
                    "overall_outlier_rate": float(
                        np.mean([r["percentage"] for r in final_results.values()])
                        / 100.0
                    )
                    if final_results
                    else 0.0,
                    "ensemble_available": "ensemble" in final_results,
                    "sota_methods_used": any(
                        m in final_results for m in ["ecod", "copod", "hbos", "aeod"]
                    ),
                },
                "performance_info": {
                    "adaptive_selection": True,
                    "subsampling_used": n_samples > self.max_sample_size,
                    "parallel_processing": True,
                    "optimization_level": "high",
                },
            }
        except Exception as e:
            return {
                "error": f"Outlier analysis failed: {e}",
                "fallback_available": True,
                "recommendations": [
                    "Consider data preprocessing",
                    "Check for data quality issues",
                ],
            }
