from typing import Any, Dict, List, Tuple

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


class OutlierAnalyzer(AnalysisStrategy):
    """State-of-the-art outlier detection with ensemble methods and adaptive thresholding"""

    @property
    def name(self) -> str:
        return "outliers"

    # ---------------------- Preprocessing ----------------------
    def _adaptive_preprocessing(
        self, data: pd.DataFrame, config: AnalysisConfig
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        numeric = data.select_dtypes(include=[np.number])

        if numeric.empty:
            raise ValueError("No numeric data available for outlier detection")

        info: Dict[str, Any] = {}

        na_mask = numeric.isna().any(axis=1)
        missing_pct = float(na_mask.mean() * 100)
        info["missing_data_percentage"] = missing_pct
        info["samples_with_missing"] = int(na_mask.sum())

        if na_mask.all():
            raise ValueError("All samples have missing values")

        # Handle missing rows
        if missing_pct > 50:
            clean = numeric.dropna()
            complete_idx = np.where(~na_mask)[0]
            info["handling_strategy"] = "complete_cases_only"
        elif missing_pct > 10:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            clean = pd.DataFrame(imputer.fit_transform(numeric), columns=numeric.columns, index=numeric.index)
            complete_idx = np.arange(len(clean))
            info["handling_strategy"] = "median_imputation"
        else:
            clean = numeric.dropna()
            complete_idx = np.where(~na_mask)[0]
            info["handling_strategy"] = "complete_cases_only"

        info["final_sample_size"] = int(len(clean))
        info["final_feature_count"] = int(clean.shape[1])

        skew = float(clean.skew().abs().mean())
        kurt = float(clean.kurtosis().abs().mean())
        info["data_skewness"] = skew
        info["data_kurtosis"] = kurt

        # Choose scaler; regularize covariance before cond()
        def _cond_score(arr: np.ndarray) -> float:
            try:
                C = np.cov(arr.T)
                lam = 1e-8 * np.trace(C) / max(1, C.shape[0])
                return float(np.linalg.cond(C + lam * np.eye(C.shape[0])))
            except Exception:
                return float("inf")

        candidates = (
            [("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
             ("robust", RobustScaler()), ("standard", StandardScaler())]
            if (skew > 3 or kurt > 10)
            else [("robust", RobustScaler()), ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
                  ("standard", StandardScaler())] if skew > 1.5
            else [("standard", StandardScaler()), ("robust", RobustScaler())]
        )

        best_name, best_scaler, best_cond = None, None, float("inf")
        for name, scaler in candidates:
            try:
                arr = scaler.fit_transform(clean)
                c = _cond_score(arr)
                if c < best_cond:
                    best_name, best_scaler, best_cond = name, scaler, c
            except Exception:
                continue

        if best_scaler is None:
            best_name, best_scaler = "robust_fallback", RobustScaler()

        X = best_scaler.fit_transform(clean)
        info["scaling_method"] = best_name
        info["condition_number"] = None if not np.isfinite(best_cond) else float(best_cond)
        return X, complete_idx, info

    # ---------------------- Statistical ----------------------
    def _statistical_outlier_detection(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        n_samples, n_features = data.shape

        # Z-score
        try:
            z = np.abs(stats.zscore(data, axis=0, nan_policy="omit"))
            thr = 3.0
            out = np.any(z > thr, axis=1)
            results["z_score"] = {"outliers": out, "scores": np.max(z, axis=1), "threshold": thr, "method_type": "statistical"}
        except Exception as e:
            print(f"Z-score detection failed: {e}")

        # Modified Z
        try:
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            mz = np.abs(0.6745 * (data - med) / (mad + 1e-10))
            thr = 3.5
            out = np.any(mz > thr, axis=1)
            results["modified_z_score"] = {"outliers": out, "scores": np.max(mz, axis=1), "threshold": thr, "method_type": "statistical"}
        except Exception as e:
            print(f"Modified Z-score detection failed: {e}")

        # IQR (per-feature)
        try:
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            out = np.any((data < low) | (data > high), axis=1)
            results["iqr"] = {"outliers": out, "lower_bounds": low, "upper_bounds": high, "method_type": "statistical"}
        except Exception as e:
            print(f"IQR detection failed: {e}")

        # Grubbs on first PC (guard sigma)
        try:
            pc1 = PCA(n_components=1, random_state=42).fit_transform(data).ravel() if n_features > 1 else data.ravel()
            mu, sd = float(np.mean(pc1)), float(np.std(pc1))
            if sd > 0:
                z = np.abs((pc1 - mu) / sd)
                alpha, n = 0.05, len(pc1)
                tcrit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
                gcrit = ((n - 1) / np.sqrt(n)) * np.sqrt((tcrit**2) / (n - 2 + tcrit**2))
                out = z > gcrit
                results["grubbs"] = {"outliers": out, "scores": z, "threshold": float(gcrit), "method_type": "statistical"}
        except Exception as e:
            print(f"Grubbs test failed: {e}")

        return results

    # ---------------------- Distance-based ----------------------
    def _distance_based_detection(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        n, p = data.shape

        # Robust Mahalanobis (squared distances vs chi2 threshold)
        try:
            mcd = MinCovDet(random_state=getattr(config, "random_state", 42)).fit(data)
            md2 = mcd.mahalanobis(data)  # squared
            from scipy.stats import chi2
            thr = float(chi2.ppf(1 - config.outlier_contamination, p))
            out = md2 > thr
            results["mahalanobis_robust"] = {"outliers": out, "distances": md2, "threshold": thr, "method_type": "distance_based"}
        except Exception as e:
            print(f"Robust Mahalanobis detection failed: {e}")

        # KNN distance
        try:
            k = int(max(1, min(20, n // 10, n - 1)))
            if k >= 1 and n > 1:
                nn = NearestNeighbors(n_neighbors=k + 1).fit(data)
                dists, _ = nn.kneighbors(data)
                knn_d = np.mean(dists[:, 1:], axis=1)
                thr = float(np.percentile(knn_d, 100 * (1 - config.outlier_contamination)))
                out = knn_d > thr
                results["knn_distance"] = {"outliers": out, "distances": knn_d, "threshold": thr, "k": k, "method_type": "distance_based"}
        except Exception as e:
            print(f"KNN distance detection failed: {e}")

        # DBSCAN
        try:
            k = int(max(2, min(4, n // 20)))
            if n >= 5:
                nn = NearestNeighbors(n_neighbors=k).fit(data)
                dists, _ = nn.kneighbors(data)
                eps = float(np.percentile(dists[:, -1], 90))
                if eps <= 0:
                    eps = float(np.median(dists[:, -1]))
                db = DBSCAN(eps=eps, min_samples=max(2, k))
                labels = db.fit_predict(data)
                out = labels == -1
                results["dbscan"] = {"outliers": out, "labels": labels, "eps": eps, "min_samples": max(2, k), "method_type": "density_based"}
        except Exception as e:
            print(f"DBSCAN outlier detection failed: {e}")

        return results

    # ---------------------- ML-based ----------------------
    def _machine_learning_detection(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # Isolation Forest (config sweep + consistency)
        try:
            cfgs = [
                {"n_estimators": 200, "max_samples": "auto", "contamination": config.outlier_contamination},
                {"n_estimators": 150, "max_samples": min(256, len(data)), "contamination": config.outlier_contamination},
                {"n_estimators": 100, "max_samples": 0.8, "contamination": config.outlier_contamination},
            ]
            best_forest, best_cons = None, -1.0
            for c in cfgs:
                preds = []
                for s in range(3):
                    clf = IsolationForest(random_state=getattr(config, "random_state", 42) + s, **c)
                    preds.append(clf.fit_predict(data))
                # agreement across runs
                agree = np.mean([np.mean(p1 == p2) for i, p1 in enumerate(preds) for j, p2 in enumerate(preds) if j > i])
                if agree > best_cons:
                    best_cons, best_forest = agree, IsolationForest(random_state=getattr(config, "random_state", 42), **c)
            if best_forest is not None:
                best_forest.fit(data)
                out = best_forest.predict(data) == -1
                scores = -best_forest.decision_function(data)
                results["isolation_forest"] = {"outliers": out, "scores": scores, "model": best_forest, "method_type": "ensemble"}
        except Exception as e:
            print(f"Isolation Forest detection failed: {e}")

        # LOF (adaptive k)
        try:
            n = len(data)
            ks = [min(20, max(5, n // 20)), min(50, max(10, n // 10))]
            best_lof, best_s = None, -np.inf
            for k in ks:
                if k >= 2 and n > k:
                    lof = LocalOutlierFactor(n_neighbors=k, contamination=config.outlier_contamination)
                    pred = lof.fit_predict(data) == -1
                    # quick heuristic: clusterability of inlier mask
                    if pred.sum() > 0 and (~pred).sum() > 0:
                        from sklearn.metrics import silhouette_score
                        s = silhouette_score(data, (~pred).astype(int))
                        if s > best_s:
                            best_s, best_lof = s, lof
            if best_lof is not None:
                pred = best_lof.fit_predict(data) == -1
                scores = -best_lof.negative_outlier_factor_
                results["local_outlier_factor"] = {"outliers": pred, "scores": scores, "n_neighbors": best_lof.n_neighbors, "method_type": "density_based"}
        except Exception as e:
            print(f"LOF detection failed: {e}")

        # One-Class SVM (clip nu to (0,1])
        try:
            nu = float(np.clip(getattr(config, "outlier_contamination", 0.05), 1e-3, 0.5))
            best_svm, best_spread = None, -np.inf
            for kernel in ("rbf", "sigmoid"):
                try:
                    svm = OneClassSVM(kernel=kernel, gamma="scale", nu=nu).fit(data)
                    dec = svm.decision_function(data)
                    spread = float(np.std(dec))
                    if spread > best_spread:
                        best_spread, best_svm = spread, svm
                except Exception:
                    continue
            if best_svm is not None:
                pred = best_svm.predict(data) == -1
                scores = -best_svm.decision_function(data)
                results["one_class_svm"] = {"outliers": pred, "scores": scores, "kernel": best_svm.kernel, "method_type": "boundary_based"}
        except Exception as e:
            print(f"One-Class SVM detection failed: {e}")

        # Elliptic Envelope
        try:
            ee = EllipticEnvelope(contamination=config.outlier_contamination, random_state=getattr(config, "random_state", 42))
            ee.fit(data)
            pred = ee.predict(data) == -1
            scores = -ee.decision_function(data)
            results["elliptic_envelope"] = {"outliers": pred, "scores": scores, "method_type": "covariance_based"}
        except Exception as e:
            print(f"Elliptic Envelope detection failed: {e}")

        return results

    # ---------------------- Advanced (optional libs) ----------------------
    def _advanced_detection_methods(self, data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if HAS_PYOD:
            try:
                from pyod.models.hbos import HBOS
                h = HBOS(contamination=config.outlier_contamination)
                h.fit(data)
                results["hbos"] = {"outliers": (h.predict(data) == 1), "scores": h.decision_scores_, "method_type": "histogram_based"}
            except Exception as e:
                print(f"HBOS detection failed: {e}")
            try:
                from pyod.models.feature_bagging import FeatureBagging
                fb = FeatureBagging(contamination=config.outlier_contamination, random_state=getattr(config, "random_state", 42))
                fb.fit(data)
                results["feature_bagging"] = {"outliers": (fb.predict(data) == 1), "scores": fb.decision_scores_, "method_type": "ensemble"}
            except Exception as e:
                print(f"Feature Bagging detection failed: {e}")
            try:
                if len(data) >= 20:
                    from pyod.models.cblof import CBLOF
                    cb = CBLOF(contamination=config.outlier_contamination, random_state=getattr(config, "random_state", 42))
                    cb.fit(data)
                    results["cblof"] = {"outliers": (cb.predict(data) == 1), "scores": cb.decision_scores_, "method_type": "cluster_based"}
            except Exception as e:
                print(f"CBLOF detection failed: {e}")

        if HAS_HDBSCAN:
            try:
                import hdbscan
                mcs = max(5, len(data) // 50)
                model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=max(1, mcs // 2))
                labs = model.fit_predict(data)
                out = labs == -1
                scores = getattr(model, "outlier_scores_", None)
                results["hdbscan"] = {"outliers": out, "scores": (scores if scores is not None else np.zeros(len(data))), "min_cluster_size": mcs, "method_type": "density_based"}
            except Exception as e:
                print(f"HDBSCAN outlier detection failed: {e}")

        return results

    # ---------------------- Ensemble ----------------------
    def _ensemble_outlier_detection(self, all_results: Dict[str, Any], data: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
        if len(all_results) < 2:
            return {}
        try:
            preds, scores = [], []
            for name, res in all_results.items():
                out = res.get("outliers")
                if isinstance(out, np.ndarray) and out.dtype == bool and out.shape[0] == len(data):
                    preds.append(out.astype(float))
                    s = res.get("scores")
                    if s is not None:
                        s = np.asarray(s, dtype=float)
                        if s.std() > 0:
                            s = (s - s.min()) / (s.max() - s.min())
                        else:
                            s = np.zeros_like(s)
                        scores.append(s)
                    else:
                        scores.append(out.astype(float))
            if not preds:
                return {}

            P = np.column_stack(preds)
            vote_scores = P.mean(axis=1)
            ensemble_scores = np.column_stack(scores).mean(axis=1)

            base_q = float(1 - config.outlier_contamination)
            thr = float(np.percentile(ensemble_scores, base_q * 100)) if ensemble_scores.std() > 0 else 0.5
            out = ensemble_scores > thr

            # std across methods is actually *disagreement*; smaller = higher consensus
            consensus_strength = P.std(axis=1)

            return {
                "outliers": out,
                "scores": ensemble_scores,
                "vote_scores": vote_scores,
                "consensus_strength": consensus_strength,
                "threshold": thr,
                "participating_methods": list(all_results.keys()),
                "method_type": "ensemble",
            }
        except Exception as e:
            print(f"Ensemble outlier detection failed: {e}")
            return {}

    # ---------------------- Evaluation ----------------------
    def _evaluate_outlier_detection(self, all_results: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        evaluations: Dict[str, Any] = {}
        for name, res in all_results.items():
            if "outliers" not in res:
                continue
            try:
                out = np.asarray(res["outliers"], dtype=bool)
                n_out = int(out.sum())
                rate = float(n_out / max(len(out), 1))
                ev: Dict[str, Any] = {"n_outliers": n_out, "outlier_rate": rate, "inlier_rate": float(1 - rate),
                                      "method_type": res.get("method_type", "unknown")}
                if "scores" in res and res["scores"] is not None:
                    s = np.asarray(res["scores"], dtype=float)
                    so = s[out]
                    si = s[~out]
                    if len(so) and len(si):
                        ev["score_separation"] = float(so.mean() - si.mean())
                        if len(so) > 1 and len(si) > 1:
                            ev["score_overlap"] = float(max(0.0, np.percentile(si, 95) - np.percentile(so, 5)))
                if n_out > 0 and len(data) > n_out:
                    from scipy.spatial.distance import cdist
                    dists = cdist(data[out], data[~out]) if (~out).any() else None
                    if dists is not None and dists.size:
                        mins = dists.min(axis=1)
                        ev["avg_distance_to_inliers"] = float(mins.mean())
                        ev["isolation_score"] = float(np.median(mins))
                if res.get("method_type") == "ensemble" and "consensus_strength" in res:
                    c = np.asarray(res["consensus_strength"], dtype=float)
                    ev["avg_consensus"] = float(c.mean()) if c.size else 0.0
                    ev["consensus_std"] = float(c.std()) if c.size else 0.0

                evaluations[name] = ev
            except Exception as e:
                print(f"Evaluation failed for {name}: {e}")
                evaluations[name] = {"error": str(e)}
        return evaluations

    # ---------------------- Recommendations ----------------------
    def _generate_outlier_recommendations(
        self, all_results: Dict[str, Any], evaluations: Dict[str, Any], preprocessing_info: Dict[str, Any]
    ) -> List[str]:
        recs: List[str] = []
        scores: Dict[str, float] = {}
        for name, ev in evaluations.items():
            if "error" in ev:
                continue
            s = 0.0
            if "score_separation" in ev:
                s += min(ev["score_separation"] / 2.0, 1.0) * 0.4
            if "score_overlap" in ev:
                s -= min(ev["score_overlap"], 1.0) * 0.2
            if "isolation_score" in ev:
                s += min(ev["isolation_score"] / 5.0, 1.0) * 0.3
            r = ev.get("outlier_rate", 0.0)
            s += 0.1 if 0.01 <= r <= 0.2 else (-0.3 if r > 0.5 else 0.0)
            scores[name] = max(0.0, s)

        if scores:
            best = max(scores, key=scores.get)
            recs.append(f"🏆 Best method: {best.upper()} (quality score: {scores[best]:.3f})")
            r = evaluations[best].get("outlier_rate", 0.0)
            if r > 0.3:
                recs.append("⚠️ High outlier rate detected - consider reviewing contamination parameter")
            elif r < 0.01:
                recs.append("🔍 Very few outliers found - data may be very clean or threshold too strict")
            else:
                recs.append(f"✅ Reasonable outlier rate: {r:.1%}")
            sep = evaluations[best].get("score_separation", None)
            if sep is not None:
                recs.append("📊 Excellent outlier-inlier separation detected" if sep > 1.0
                            else "👍 Good separation between outliers and inliers" if sep > 0.5
                            else "🤔 Moderate separation - outliers may be subtle")

        if preprocessing_info.get("missing_data_percentage", 0) > 20:
            recs.append("📝 High missing data rate may affect outlier detection accuracy")
        if preprocessing_info.get("data_skewness", 0) > 2:
            recs.append("📈 Highly skewed data - robust methods recommended")
        cn = preprocessing_info.get("condition_number")
        if cn and cn > 1000:
            recs.append("🔧 Poor data conditioning - consider dimensionality reduction")

        ok_methods = len([e for e in evaluations.values() if "error" not in e])
        if ok_methods <= 2:
            recs.append("🔄 Consider additional detection methods for robust analysis")
        elif ok_methods >= 5:
            recs.append("🤝 Multiple methods available - ensemble approach recommended")

        if "ensemble" in evaluations:
            avg_cons = evaluations["ensemble"].get("avg_consensus", 0.0)
            recs.append("🎯 High consensus among methods - reliable outliers identified" if avg_cons < 0.3
                        else "🤷 Low consensus among methods - results may vary")

        return recs[:5]

    # ---------------------- Main ----------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        try:
            X, idx, prep = self._adaptive_preprocessing(data, config)

            all_res: Dict[str, Any] = {}
            all_res.update(self._statistical_outlier_detection(X, config))
            all_res.update(self._distance_based_detection(X, config))
            all_res.update(self._machine_learning_detection(X, config))
            all_res.update(self._advanced_detection_methods(X, config))

            # Map to full length
            final: Dict[str, Any] = {}
            N = len(data)
            for name, res in all_res.items():
                if "outliers" not in res:
                    continue
                full_mask = np.zeros(N, dtype=bool)
                out = np.asarray(res["outliers"], dtype=bool)
                vidx = np.asarray(idx, dtype=int)
                if out.shape[0] == vidx.shape[0]:
                    full_mask[vidx] = out
                mapped = {
                    "outliers": full_mask,
                    "count": int(full_mask.sum()),
                    "percentage": float(100.0 * full_mask.mean()),
                    "method_type": res.get("method_type", "unknown"),
                }
                if "scores" in res and res["scores"] is not None:
                    full_scores = np.zeros(N, dtype=float)
                    s = np.asarray(res["scores"], dtype=float)
                    if s.shape[0] == vidx.shape[0]:
                        full_scores[vidx] = s
                    mapped["scores"] = full_scores
                for key in ("threshold", "distances", "k", "eps", "kernel", "n_neighbors"):
                    if key in res:
                        mapped[key] = res[key]
                final[name] = mapped

            # Ensemble on scaled X, then map
            ens = self._ensemble_outlier_detection(all_res, X, config)
            if ens:
                fmask = np.zeros(N, dtype=bool)
                fs = np.zeros(N, dtype=float)
                out = np.asarray(ens["outliers"], dtype=bool)
                sc = np.asarray(ens["scores"], dtype=float)
                vidx = np.asarray(idx, dtype=int)
                if out.shape[0] == vidx.shape[0]:
                    fmask[vidx] = out
                if sc.shape[0] == vidx.shape[0]:
                    fs[vidx] = sc
                final["ensemble"] = {
                    "outliers": fmask,
                    "count": int(fmask.sum()),
                    "percentage": float(100.0 * fmask.mean()),
                    "scores": fs,
                    "method_type": "ensemble",
                    "participating_methods": ens["participating_methods"],
                    "consensus_strength": ens.get("consensus_strength", np.array([])),
                }

            evals = self._evaluate_outlier_detection(all_res, X)
            recs = self._generate_outlier_recommendations(final, evals, prep)

            return {
                "outlier_results": final,
                "evaluations": evals,
                "preprocessing_info": prep,
                "data_characteristics": {
                    "total_samples": int(N),
                    "analyzed_samples": int(len(X)),
                    "missing_samples": int(N - len(X)),
                    "n_features": int(X.shape[1]),
                    "contamination_rate": float(config.outlier_contamination),
                },
                "recommendations": recs,
                "summary": {
                    "methods_attempted": len(all_res),
                    "successful_methods": len(final),
                    "best_method": (max(evals.keys(), key=lambda k: evals[k].get("score_separation", 0)) if evals else None),
                    "overall_outlier_rate": float(np.mean([r["percentage"] for r in final.values()]) / 100.0),
                },
            }
        except Exception as e:
            return {"error": f"Outlier analysis failed: {e}"}
