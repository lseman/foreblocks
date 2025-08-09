import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
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
    # Ranks in (0,1) using double argsort (faster than scipy.rankdata for large 1D)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(len(x))
    # average-rank tie handling (stable sort helps; this is an approximation but fine for ECDF/COP)
    return (ranks + 1) / (len(x) + 1)

def _winsorize_inplace(X: np.ndarray, q: float = 0.001) -> None:
    # Robust clipping to shrink extreme tails, in-place & vectorized
    lo = np.nanquantile(X, q, axis=0)
    hi = np.nanquantile(X, 1 - q, axis=0)
    np.clip(X, lo, hi, out=X)

def _normalize_scores(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).any():
        return np.zeros_like(s, dtype=float)
    # Robust min-max on finite values only
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
    """SOTA fast outlier detection with adaptive method selection and ensemble learning"""

    @property
    def name(self) -> str:
        return "outliers"

    def __init__(self):
        # Performance tiers based on data size
        self.fast_threshold = 1000
        self.medium_threshold = 5000
        self.large_threshold = 20000
        self.max_sample_size = 3000

    # ---------------------- Enhanced Preprocessing ----------------------
    def _lightning_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        numeric = data.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError("No numeric data available for outlier detection")

        # Replace inf with NaN (vectorized)
        numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

        info: Dict[str, Any] = {}
        n_samples, n_features = numeric.shape

        na_counts = numeric.isna().sum()
        missing_pct = float(na_counts.sum() / (n_samples * n_features) * 100.0)
        info["missing_data_percentage"] = missing_pct

        if missing_pct > 80:
            raise ValueError("Too much missing data for reliable outlier detection")

        # Drop near-constant columns (noise for distance/covariance methods)
        keep_cols = []
        for c in numeric.columns:
            col = numeric[c].to_numpy()
            if _is_constant(col):
                continue
            keep_cols.append(c)
        if not keep_cols:
            raise ValueError("All numeric columns are constant.")
        if len(keep_cols) < numeric.shape[1]:
            numeric = numeric[keep_cols]
        info["dropped_constant_cols"] = int(n_features - len(keep_cols))
        n_features = numeric.shape[1]

        # Missing strategy
        if missing_pct > 30:
            good_cols = na_counts[keep_cols] < (n_samples * 0.5)
            numeric = numeric.loc[:, good_cols.index[good_cols]]
            if numeric.empty:
                raise ValueError("No columns with sufficient data")
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

        # Light winsorization to stabilize heavy tails (very fast)
        try:
            _winsorize_inplace(X, q=0.001)
            info["winsorization"] = True
        except Exception:
            info["winsorization"] = False

        # Skew/kurt (silence warnings for small samples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skewness = pd.Series(X).astype(float) if X.ndim == 1 else pd.DataFrame(X).skew().abs()
            kurtosis = pd.Series(X).astype(float) if X.ndim == 1 else pd.DataFrame(X).kurtosis().abs()
        info["data_skewness"] = float(np.nanmean(np.asarray(skewness)))
        info["data_kurtosis"] = float(np.nanmean(np.asarray(kurtosis)))

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
            # Fallback
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            info["scaling_method"] = "robust_fallback"

        return X, complete_idx, info

    # ---------------------- Ultra-Fast Statistical Methods ----------------------
    def _lightning_statistical(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape

        # Z-score on already-scaled data
        try:
            max_abs = np.max(np.abs(data), axis=1)
            thr = 3.0
            outliers = max_abs > thr
            results["z_score"] = {
                "outliers": outliers,
                "scores": max_abs,
                "threshold": thr,
                "method_type": "statistical",
            }
        except Exception:
            pass

        # Modified Z-score (robust)
        try:
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            mad = np.where(mad < 1e-12, 1e-12, mad)
            mz = np.abs(0.6745 * (data - med) / mad)
            max_mz = np.max(mz, axis=1)
            thr = 3.5
            outliers = max_mz > thr
            results["modified_z_score"] = {
                "outliers": outliers,
                "scores": max_mz,
                "threshold": thr,
                "method_type": "statistical",
            }
        except Exception:
            pass

        # IQR (vectorized)
        try:
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = np.any((data < lower) | (data > upper), axis=1)
            results["iqr"] = {
                "outliers": outliers,
                "lower_bounds": lower,
                "upper_bounds": upper,
                "method_type": "statistical",
            }
        except Exception:
            pass

        # PCA reconstruction error (stronger than only PC1)
        try:
            k = min( min(10, n_features), max(1, int(np.ceil(0.2 * n_features))) )
            pca = PCA(n_components=k, random_state=42)
            Z = pca.fit_transform(data)
            Xr = pca.inverse_transform(Z)
            rec_err = np.mean((data - Xr) ** 2, axis=1)
            thr = np.percentile(rec_err, 100 * (1 - config.outlier_contamination))
            outliers = rec_err > thr
            results["pca_recon"] = {
                "outliers": outliers,
                "scores": rec_err,
                "threshold": float(thr),
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
                "method_type": "dimensionality_reduction",
            }
        except Exception:
            pass

        return results

    # ---------------------- Fast Distance/Density Methods ----------------------
    def _fast_distance_based(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape
        rng = _rng(config)

        # Deterministic subsample for expensive neighbors
        if n_samples > self.max_sample_size:
            idx = rng.choice(n_samples, self.max_sample_size, replace=False)
            data_sample = data[idx]
            use_subsample = True
        else:
            data_sample = data
            idx = np.arange(n_samples)
            use_subsample = False

        # Reusable NN model
        try:
            k = max(5, min(20, len(data_sample) // 20))
            if len(data_sample) > k:
                nn = NearestNeighbors(n_neighbors=k + 1)
                nn.fit(data_sample)

                # KNN score: mean distance to k neighbors (exclude self)
                if use_subsample:
                    dists, _ = nn.kneighbors(data)
                else:
                    dists, _ = nn.kneighbors(data_sample)
                knn_mean = np.mean(dists[:, 1:], axis=1)
                thr = np.percentile(knn_mean, 100 * (1 - config.outlier_contamination))
                outliers = knn_mean > thr
                results["knn_distance"] = {
                    "outliers": outliers,
                    "distances": knn_mean,
                    "threshold": float(thr),
                    "k": k,
                    "method_type": "distance_based",
                }

                # Adaptive eps for DBSCAN from kth-NN dist
                try:
                    eps = np.percentile(dists[:, -1], 75)
                    if eps > 0:
                        db = DBSCAN(eps=eps, min_samples=k)
                        if use_subsample:
                            labels_sub = db.fit_predict(data_sample)
                            nn1 = NearestNeighbors(n_neighbors=1).fit(data_sample)
                            _, nn_idx = nn1.kneighbors(data)
                            labels = labels_sub[nn_idx.ravel()]
                        else:
                            labels = db.fit_predict(data_sample)
                        outliers = labels == -1
                        results["dbscan"] = {
                            "outliers": outliers,
                            "labels": labels,
                            "eps": float(eps),
                            "min_samples": int(k),
                            "method_type": "density_based",
                        }
                except Exception:
                    pass
        except Exception:
            pass

        # Robust Mahalanobis with MCD
        try:
            if n_features <= 50 and len(data_sample) >= max(50, n_features * 2):
                mcd = MinCovDet(random_state=getattr(config, "random_state", 42))
                mcd.fit(data_sample)
                md = mcd.mahalanobis(data if use_subsample else data_sample)
                thr = stats.chi2.ppf(1 - config.outlier_contamination, n_features)
                outliers = md > thr
                results["mahalanobis_robust"] = {
                    "outliers": outliers,
                    "distances": md,
                    "threshold": float(thr),
                    "method_type": "distance_based",
                }
        except Exception:
            pass

        return results

    # ---------------------- Smart ML-Based Detection ----------------------
    def _smart_ml_detection(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape

        # Isolation Forest (very fast)
        try:
            if n_samples < 1000:
                n_estimators, max_samples = 200, "auto"
            elif n_samples < 5000:
                n_estimators, max_samples = 150, min(512, n_samples)
            else:
                n_estimators, max_samples = 100, min(1024, n_samples)

            iso = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=config.outlier_contamination,
                random_state=getattr(config, "random_state", 42),
                n_jobs=-1,
                bootstrap=False,
                warm_start=False,
            )
            o = iso.fit_predict(data) == -1
            s = -iso.decision_function(data)
            results["isolation_forest"] = {
                "outliers": o,
                "scores": s,
                "n_estimators": int(n_estimators),
                "method_type": "ensemble",
            }
        except Exception:
            pass

        # LOF (avoid huge n)
        if n_samples <= 5000:
            try:
                k = max(5, min(50, n_samples // 20))
                lof = LocalOutlierFactor(
                    n_neighbors=k,
                    contamination=config.outlier_contamination,
                    n_jobs=-1,
                    novelty=False,
                )
                o = lof.fit_predict(data) == -1
                s = -lof.negative_outlier_factor_
                results["local_outlier_factor"] = {
                    "outliers": o,
                    "scores": s,
                    "n_neighbors": int(k),
                    "method_type": "density_based",
                }
            except Exception:
                pass

        # One-Class SVM (small n only)
        if n_samples <= 2000:
            try:
                nu = float(np.clip(config.outlier_contamination, 0.01, 0.5))
                oc = OneClassSVM(kernel="rbf", gamma="scale", nu=nu, cache_size=256, shrinking=True)
                o = oc.fit_predict(data) == -1
                s = -oc.decision_function(data)
                results["one_class_svm"] = {
                    "outliers": o,
                    "scores": s,
                    "kernel": "rbf",
                    "method_type": "boundary_based",
                }
            except Exception:
                pass

        # Elliptic Envelope (reasonable dims)
        if n_features <= 20 and n_samples >= n_features * 3:
            try:
                ee = EllipticEnvelope(
                    contamination=config.outlier_contamination,
                    random_state=getattr(config, "random_state", 42),
                    support_fraction=None,
                )
                o = ee.fit_predict(data) == -1
                s = -ee.decision_function(data)
                results["elliptic_envelope"] = {
                    "outliers": o,
                    "scores": s,
                    "method_type": "covariance_based",
                }
            except Exception:
                pass

        return results

    # ---------------------- SOTA-ish, dependency-free additions ----------------------
    def _sota_advanced_methods(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape

        # ECOD: tail-prob min across features (vectorized)
        try:
            ranks = np.empty_like(data)
            for j in range(n_features):
                ranks[:, j] = _safe_rank(data[:, j])
            tail = np.minimum(ranks, 1.0 - ranks)
            min_tail = np.min(tail, axis=1)
            thr = np.percentile(min_tail, 100 * config.outlier_contamination)
            o = min_tail <= thr
            s = 1.0 - min_tail  # higher = more outlying
            results["ecod"] = {
                "outliers": o,
                "scores": s,
                "threshold": float(thr),
                "method_type": "distribution_based",
            }
        except Exception:
            pass

        # COPOD: simple copula product deviation
        try:
            ranks = np.empty_like(data)
            for j in range(n_features):
                ranks[:, j] = _safe_rank(data[:, j])
            prod_u = np.prod(ranks, axis=1)
            s = -np.log(prod_u + 1e-12)
            thr = np.percentile(s, 100 * (1 - config.outlier_contamination))
            o = s > thr
            results["copod"] = {
                "outliers": o,
                "scores": s,
                "threshold": float(thr),
                "method_type": "copula_based",
            }
        except Exception:
            pass

        # HBOS: histogram-based outlier score (fast, per-feature, no neighbor graphs)
        try:
            bins = np.clip(int(np.sqrt(n_samples)), 10, 50)
            scores = np.zeros(n_samples, dtype=float)
            for j in range(n_features):
                col = data[:, j]
                if _is_constant(col):
                    continue
                hist, edges = np.histogram(col, bins=bins)
                # avoid zeros
                hist = hist + 1e-6
                # density per bin
                dens = hist / np.trapz(hist, dx=1.0)
                # bin index for each point
                idx = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, len(hist) - 1)
                scores += -np.log(dens[idx])
            thr = np.percentile(scores, 100 * (1 - config.outlier_contamination))
            o = scores > thr
            results["hbos"] = {
                "outliers": o,
                "scores": scores,
                "threshold": float(thr),
                "method_type": "histogram_based",
            }
        except Exception:
            pass

        return results

    # ---------------------- Fast Smart Ensemble ----------------------
    def _fast_ensemble(self, all_results: Dict[str, Any], data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        if len(all_results) < 2:
            return {}

        # Method reliability weights
        method_weights = {
            "isolation_forest": 1.0,
            "z_score": 0.6,
            "modified_z_score": 0.8,
            "iqr": 0.5,
            "mahalanobis_robust": 1.2,
            "knn_distance": 1.0,
            "local_outlier_factor": 1.1,
            "dbscan": 0.8,
            "one_class_svm": 0.8,
            "elliptic_envelope": 1.0,
            "pca_recon": 0.9,
            "ecod": 1.3,
            "copod": 1.2,
            "hbos": 1.0,
        }

        votes = []
        scores = []
        methods = []
        weights = []

        for m, res in all_results.items():
            o = res.get("outliers")
            s = res.get("scores")
            if not isinstance(o, np.ndarray):
                continue
            w = method_weights.get(m, 0.7)
            methods.append(m)
            weights.append(w)
            # vote vector
            votes.append(o.astype(float) * w)
            # normalized scores (fallback to votes if None)
            if s is None:
                scores.append(o.astype(float) * w)
            else:
                scores.append(_normalize_scores(s) * w)

        if not votes:
            return {}

        W = np.sum(weights)
        vote_mat = np.vstack(votes)
        score_mat = np.vstack(scores)

        # Borda-style fusion on scores + vote stabilization
        fused = score_mat.sum(axis=0) / max(W, 1e-9)
        vote_disp = vote_mat.std(axis=0)
        confidence = 1 - _normalize_scores(vote_disp)  # higher when methods agree

        thr = np.percentile(fused, 100 * (1 - config.outlier_contamination))
        outliers = fused > thr

        return {
            "outliers": outliers,
            "scores": fused,
            "vote_scores": vote_mat.sum(axis=0) / max(W, 1e-9),
            "confidence": confidence,
            "threshold": float(thr),
            "participating_methods": methods,
            "method_weights": dict(zip(methods, weights)),
            "method_type": "ensemble",
        }

    # ---------------------- Lightning Evaluation ----------------------
    def _lightning_evaluation(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
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
                    s = np.asarray(s, dtype=float)
                    sep = float(np.nanmean(s[o]) - np.nanmean(s[~o]))
                    ev["score_separation"] = sep
                if "confidence" in res:
                    ev["avg_confidence"] = float(np.nanmean(res["confidence"]))
                evaluations[m] = ev
            except Exception:
                evaluations[m] = {"error": "evaluation_failed"}
        return evaluations

    # ---------------------- Smart Recommendations ----------------------
    def _smart_recommendations(
        self,
        all_results: Dict[str, Any],
        evaluations: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
        data_size: Tuple[int, int],
    ) -> List[str]:
        recs = []
        n_samples, n_features = data_size

        # Score methods
        scores = {}
        for m, ev in evaluations.items():
            if "error" in ev:
                continue
            sc = 0.0
            rate = ev.get("outlier_rate", 0)
            if 0.01 <= rate <= 0.15:
                sc += 1.0
            elif rate > 0.30:
                sc -= 0.5
            sep = ev.get("score_separation", 0.0)
            if sep > 0:
                sc += min(sep / 2.0, 1.0)
            if ev.get("method_type") == "ensemble":
                sc += 0.3
            if m in ["ecod", "copod", "isolation_forest", "hbos", "pca_recon"]:
                sc += 0.2
            scores[m] = sc

        if scores:
            best = max(scores, key=scores.get)
            recs.append(f"🏆 Recommended method: {best.upper().replace('_', ' ')} (score: {scores[best]:.2f})")
            rate = evaluations[best].get("outlier_rate", 0)
            if rate > 0.2:
                recs.append("⚠️ High outlier rate — consider raising contamination or reviewing preprocessing.")
            elif rate < 0.005:
                recs.append("🔍 Very few outliers — dataset looks clean; consider relaxing thresholds.")
            else:
                recs.append(f"✅ Healthy outlier rate: {rate:.1%}")

        if n_samples > 10000:
            recs.append("🚀 Large dataset detected — prioritized fast methods (IF, HBOS, ECOD, KNN).")
        if n_features > 50:
            recs.append("📊 High-dimensional data — PCA/feature selection before distance/covariance methods helps.")
        if preprocessing_info.get("data_skewness", 0) > 3:
            recs.append("📈 Strong skew — robust scaling/transform was applied; prefer ECOD/HBOS/IF.")
        if len([e for e in evaluations.values() if "error" not in e]) >= 5:
            recs.append("🎯 Multiple methods succeeded — ensemble reliability is high.")

        return recs[:4]

    # ---------------------- Adaptive Main Analysis ----------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        try:
            X, complete_indices, prep = self._lightning_preprocessing(data, config)
            n_samples, n_features = X.shape

            all_results: Dict[str, Any] = {}
            all_results.update(self._lightning_statistical(X, config))

            if n_samples <= 10000:
                all_results.update(self._fast_distance_based(X, config))
            if n_samples <= 20000:
                all_results.update(self._smart_ml_detection(X, config))
            if n_samples <= 5000:
                all_results.update(self._sota_advanced_methods(X, config))

            # Map results to full index
            N = len(data)
            final_results: Dict[str, Any] = {}
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
                for k in ["threshold", "k", "eps", "n_neighbors", "kernel", "confidence"]:
                    if k in res:
                        mapped[k] = res[k]
                final_results[name] = mapped

            # Ensemble on analyzed subset
            ens = self._fast_ensemble(all_results, X, config)
            if ens:
                full_o = np.zeros(N, dtype=bool)
                full_s = np.zeros(N, dtype=float)
                full_c = np.zeros(N, dtype=float)
                full_o[complete_indices] = ens["outliers"]
                full_s[complete_indices] = ens["scores"]
                if "confidence" in ens:
                    full_c[complete_indices] = ens["confidence"]
                final_results["ensemble"] = {
                    "outliers": full_o,
                    "count": int(full_o.sum()),
                    "percentage": float(100.0 * full_o.mean()),
                    "scores": full_s,
                    "confidence": full_c,
                    "method_type": "ensemble",
                    "participating_methods": ens["participating_methods"],
                    "method_weights": ens.get("method_weights", {}),
                }

            evaluations = self._lightning_evaluation(all_results, X)
            recommendations = self._smart_recommendations(final_results, evaluations, prep, (n_samples, n_features))

            if n_samples < self.fast_threshold:
                perf_tier, tier_desc = "ultra_fast", "All methods available"
            elif n_samples < self.medium_threshold:
                perf_tier, tier_desc = "fast", "Most methods available"
            elif n_samples < self.large_threshold:
                perf_tier, tier_desc = "medium", "Core methods with subsampling"
            else:
                perf_tier, tier_desc = "large_scale", "Statistical and fast ML methods only"

            # best_method by separation (fall back to ensemble if tied)
            best_by_sep = None
            if evaluations:
                cand = [(k, v.get("score_separation", -np.inf)) for k, v in evaluations.items() if "error" not in v]
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
                    "methods_attempted": len(all_results),
                    "successful_methods": len(final_results),
                    "best_method": best_by_sep if best_by_sep else ("ensemble" if "ensemble" in final_results else None),
                    "overall_outlier_rate": float(
                        np.mean([r["percentage"] for r in final_results.values()]) / 100.0
                    ) if final_results else 0.0,
                    "ensemble_available": "ensemble" in final_results,
                    "sota_methods_used": any(m in final_results for m in ["ecod", "copod", "hbos"]),
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
                "error": f"SOTA outlier analysis failed: {e}",
                "fallback_available": True,
                "recommendations": ["Consider data preprocessing", "Check for data quality issues"],
            }
