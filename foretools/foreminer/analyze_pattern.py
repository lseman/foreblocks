from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.stats as sps

from .analyze_correlation import CorrelationAnalyzer
from .analyze_distribution import DistributionAnalyzer
from .foreminer_aux import *


class PatternDetector(AnalysisStrategy):
    """SOTA Pattern Detection with Advanced ML Techniques"""

    @property
    def name(self) -> str:
        return "patterns"

    # ------------------------------ Public API ------------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        return {
            "feature_types": self._classify_features(data, config),
            "relationships": self._analyze_relationships(data, config),
            "distributions": self._fit_distributions(data, config),
        }

    # ------------------------------ Helpers (stats) ------------------------------
    @staticmethod
    def _safe_sample(series: pd.Series, n: int, rs: int) -> pd.Series:
        n = min(n, len(series))
        return series.sample(n=n, random_state=rs) if n > 0 else series

    @staticmethod
    def _is_integer_series(s: pd.Series) -> bool:
        s = s.dropna()
        if s.empty:
            return False
        # Treat near-integers as counts
        return bool(np.all(np.isclose(s % 1, 0)))

    @staticmethod
    def _zero_pct(s: pd.Series) -> float:
        s = s.dropna()
        return float((s == 0).mean()) if len(s) else 0.0

    @staticmethod
    def _try_normality(series: pd.Series, rs: int) -> bool:
        from scipy.stats import anderson, jarque_bera, shapiro

        x = series.dropna()
        if len(x) < 10:
            return False
        try:
            jb_p = jarque_bera(x)[1]
        except Exception:
            jb_p = np.nan
        try:
            xs = PatternDetector._safe_sample(x, 5000, rs)
            sh_p = shapiro(xs)[1]
        except Exception:
            sh_p = np.nan
        try:
            ad = anderson(x, dist="norm")
            ad_ok = ad.statistic < ad.critical_values[2]  # ~5%
        except Exception:
            ad_ok = False

        checks = [
            (jb_p, 0.05, True),
            (sh_p, 0.05, True),
        ]
        ok = []
        for p, thr, greater in checks:
            if np.isfinite(p):
                ok.append(p > thr if greater else p < thr)
        # Require at least two successful, AND Anderson below 5% cutoff
        return (sum(ok) >= 2) and ad_ok

    @staticmethod
    def _log_normal_candidate(x: pd.Series, mean: float, skew: float, std: float) -> bool:
        x = x.dropna()
        if len(x) < 10 or (x <= 0).any() or std <= 0 or mean <= 0 or skew <= 0:
            return False
        try:
            return np.log(x).std(ddof=1) < x.std(ddof=1)
        except Exception:
            return False

    @staticmethod
    def _power_law_candidate(x: pd.Series, skew: float) -> bool:
        """Quick tail linearity on log-log; needs positive values and notable skew."""
        x = x.dropna()
        if len(x) < 30 or (x <= 0).any() or skew <= 1.5:
            return False
        try:
            # Use top half of sorted tail
            xs = np.sort(x.values)[::-1]
            m = len(xs) // 2
            if m < 10:
                return False
            X = np.log(np.arange(1, m + 1))
            Y = np.log(xs[:m])
            A = np.vstack([X, np.ones_like(X)]).T
            coef, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
            yhat = A.dot(coef)
            # R^2
            ss_res = np.sum((Y - yhat) ** 2)
            ss_tot = np.sum((Y - Y.mean()) ** 2) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            return r2 > 0.64  # ~|corr|>0.8
        except Exception:
            return False

    @staticmethod
    def _mixture_bic_flags(x: pd.Series, rs: int) -> Dict[str, bool]:
        """Compare 1 vs 2 components via BIC; also probe BayesianGMM."""
        from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

        out = {"mixture_model": False, "bimodal": False}
        x = x.dropna().values.reshape(-1, 1)
        if len(x) < 50:
            return out
        try:
            gmm1 = GaussianMixture(n_components=1, random_state=rs).fit(x)
            gmm2 = GaussianMixture(n_components=2, random_state=rs).fit(x)
            if gmm2.bic(x) + 10 < gmm1.bic(x):  # margin to avoid noise
                out["mixture_model"] = True
        except Exception:
            pass
        try:
            bgmm = BayesianGaussianMixture(n_components=3, random_state=rs).fit(x)
            # If >1 active component with decent weight, flag bimodal
            if hasattr(bgmm, "weights_") and np.sum(bgmm.weights_ > 0.1) > 1:
                out["bimodal"] = True
        except Exception:
            pass
        return out

    @staticmethod
    def _seasonality_flags(df: pd.DataFrame, time_col: str, feature: str) -> bool:
        try:
            if not (time_col and pd.api.types.is_datetime64_any_dtype(df[time_col])):
                return False
            series = df.set_index(time_col)[feature].dropna()
            if len(series) < 24:
                return False
            from statsmodels.tsa.seasonal import STL
            from statsmodels.tsa.stattools import acf
            stl = STL(series, seasonal=min(13, max(7, len(series) // 6))).fit()
            s_strength = float(np.var(stl.seasonal) / (np.var(series) + 1e-8))
            ac = acf(series, nlags=min(40, len(series) // 4), fft=True)
            return (s_strength > 0.15) or (np.max(np.abs(ac[12:])) > 0.30)
        except Exception:
            return False

    # ------------------------------ Feature classification ------------------------------
    def _classify_features(self, data, config) -> Dict[str, List[str]]:
        dist_summary = DistributionAnalyzer().analyze(data, config).get("summary", pd.DataFrame())
        classification: Dict[str, List[str]] = {
            "gaussian": [],
            "log_normal_candidates": [],
            "bounded": [],
            "count_like": [],
            "continuous": [],
            "highly_skewed": [],
            "heavy_tailed": [],
            "mixture_model": [],
            "seasonal": [],
            "power_law": [],
            "bimodal": [],
            "uniform_like": [],
            "zero_inflated": [],
            "transformable_to_normal": [],
        }

        time_col = getattr(config, "time_col", None)
        rs = int(getattr(config, "random_state", 42))

        for _, row in dist_summary.iterrows():
            feature = row.get("feature")
            if feature not in data.columns:
                continue
            x = data[feature].dropna()
            if len(x) < 10:
                continue

            skew = float(row.get("skewness", x.skew()))
            
            # kurt = float(row.get("kurtosis", x.kurtosis(fisher=False)))
            kurt = float(row.get("kurtosis", sps.kurtosis(x, fisher=False, bias=False)))
            
            std = float(row.get("std", x.std(ddof=1)))
            mean = float(row.get("mean", x.mean()))

            # Normality
            is_normal = row.get("is_gaussian", False) or self._try_normality(x, rs)
            if is_normal:
                classification["gaussian"].append(feature)

            # Heavy tails / skew
            if row.get("is_heavy_tailed", False) or kurt > 3:
                classification["heavy_tailed"].append(feature)
            if abs(skew) > 2:
                classification["highly_skewed"].append(feature)

            # Candidate shapes
            if self._log_normal_candidate(x, mean, skew, std):
                classification["log_normal_candidates"].append(feature)

            z_pct = self._zero_pct(x)
            if 0.10 < z_pct < 0.90:
                classification["zero_inflated"].append(feature)

            if x.min() >= 0 and x.max() <= 1:
                classification["bounded"].append(feature)
            elif self._is_integer_series(x) and (x.var() / (x.mean() + 1e-8) < 3):
                classification["count_like"].append(feature)

            if self._power_law_candidate(x, skew):
                classification["power_law"].append(feature)

            if abs(skew) < 0.5 and abs(kurt - 3) < 1.5:
                classification["uniform_like"].append(feature)

            # Mixtures / bimodality
            mix = self._mixture_bic_flags(x, rs)
            if mix["mixture_model"]:
                classification["mixture_model"].append(feature)
            if mix["bimodal"]:
                classification["bimodal"].append(feature)

            # Transformable to normal (Yeo–Johnson)
            if not is_normal and x.min() >= 0:
                try:
                    from sklearn.preprocessing import PowerTransformer
                    pt = PowerTransformer(method="yeo-johnson")
                    xt = pd.Series(pt.fit_transform(x.values.reshape(-1, 1)).ravel(), index=x.index)
                    # Shapiro on sample
                    from scipy.stats import shapiro
                    p = shapiro(self._safe_sample(xt, 5000, rs))[1]
                    if np.isfinite(p) and p > 0.05:
                        classification["transformable_to_normal"].append(feature)
                except Exception:
                    pass

            # Default continuous (exclude the discrete/bounded/zero-inflated sets)
            if feature not in (
                classification["count_like"]
                + classification["gaussian"]
                + classification["bounded"]
                + classification["zero_inflated"]
            ):
                classification["continuous"].append(feature)

            # Seasonality
            if time_col and self._seasonality_flags(data, time_col, feature):
                classification["seasonal"].append(feature)

        return classification

    # ------------------------------ Relationships ------------------------------
    def _analyze_relationships(self, data, config) -> Dict[str, Any]:
        try:
            from dcor import distance_correlation
            DCOR_AVAILABLE = True
        except ImportError:
            DCOR_AVAILABLE = False

        from scipy.stats import kendalltau, spearmanr
        from sklearn.feature_selection import mutual_info_regression

        correlations = CorrelationAnalyzer().analyze(data, config)
        pearson = correlations.get("pearson")
        if pearson is None:
            return {}

        patterns: Dict[str, List[Dict[str, Any]]] = {
            "nonlinear": [],
            "complex": [],
            "distance_corr": [],
            "monotonic": [],
            "tail_dependence": [],
            "conditional": [],
        }

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        rs = int(getattr(config, "random_state", 42))

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                f1, f2 = numeric_cols[i], numeric_cols[j]
                df = data[[f1, f2]].dropna()
                if len(df) < 10:
                    continue
                x, y = df[f1], df[f2]

                pc = abs(pearson.loc[f1, f2]) if (f1 in pearson.index and f2 in pearson.columns) else 0.0

                # Spearman/Kendall
                try:
                    sc = abs(spearmanr(x, y).correlation)
                except Exception:
                    sc = 0.0
                try:
                    tau = abs(kendalltau(x, y).correlation)
                except Exception:
                    tau = 0.0

                # Mutual information (robust to constants)
                mi_val = 0.0
                try:
                    if x.nunique() > 1 and y.nunique() > 1 and len(x) > 50:
                        mi_val = float(mutual_info_regression(x.values.reshape(-1, 1), y.values, random_state=rs)[0])
                except Exception:
                    pass

                # Distance correlation (subset for speed)
                dc = 0.0
                if DCOR_AVAILABLE and len(x) <= 1000:
                    try:
                        dc = float(distance_correlation(x.values, y.values))
                    except Exception:
                        pass

                if sc > 0.4 and (sc - pc) > 0.2:
                    patterns["nonlinear"].append(
                        {"feature1": f1, "feature2": f2, "pearson": pc, "spearman": sc, "kendall": tau,
                         "nonlinearity_score": sc - pc}
                    )
                if mi_val > 0.2 and pc < 0.4:
                    patterns["complex"].append(
                        {"feature1": f1, "feature2": f2, "mutual_info": mi_val, "pearson": pc,
                         "complexity_score": mi_val - pc}
                    )
                if dc > 0.4 and pc < 0.4:
                    patterns["distance_corr"].append(
                        {"feature1": f1, "feature2": f2, "distance_corr": dc, "pearson": pc}
                    )
                if tau > 0.5:
                    patterns["monotonic"].append(
                        {"feature1": f1, "feature2": f2, "kendall_tau": tau, "relationship_type": "monotonic"}
                    )

                # Simple upper-tail dependence
                if len(x) > 100:
                    try:
                        q95_x, q95_y = np.percentile(x, 95), np.percentile(y, 95)
                        upper_tail = float(((x >= q95_x) & (y >= q95_y)).mean())
                        if upper_tail > 0.05:
                            patterns["tail_dependence"].append(
                                {"feature1": f1, "feature2": f2, "upper_tail_dep": upper_tail, "type": "upper_tail"}
                            )
                    except Exception:
                        pass

        # Sort & cap 10 per bucket
        for k, arr in patterns.items():
            if arr:
                def _last_num(v: Dict[str, Any]) -> float:
                    last = list(v.values())[-1]
                    return float(last) if isinstance(last, (int, float, np.floating)) else 0.0
                patterns[k] = sorted(arr, key=_last_num, reverse=True)[:10]

        return patterns

    # ------------------------------ Distribution fitting ------------------------------
    def _fit_distributions(self, data, config) -> Dict[str, Dict[str, Any]]:
        import warnings

        from scipy import stats

        warnings.filterwarnings("ignore")

        distributions = [
            ("normal", stats.norm),
            ("lognormal", stats.lognorm),
            ("exponential", stats.expon),
            ("gamma", stats.gamma),
            ("beta", stats.beta),
            ("weibull_min", stats.weibull_min),
            ("pareto", stats.pareto),
            ("gumbel_r", stats.gumbel_r),
            ("uniform", stats.uniform),
            ("chi2", stats.chi2),
            ("t", stats.t),
            ("f", stats.f),
            ("logistic", stats.logistic),
            ("laplace", stats.laplace),
            ("rayleigh", stats.rayleigh),
        ]

        fits: Dict[str, Dict[str, Any]] = {}
        rs = int(getattr(config, "random_state", 42))
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols[:15]:  # perf cap
            x = data[col].dropna()
            if len(x) < 30:
                continue
            if len(x) > 2000:
                x = self._safe_sample(x, 2000, rs)

            cand = []
            for name, dist in distributions:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        params = dist.fit(x)

                    ll = float(np.sum(dist.logpdf(x, *params)))
                    k = len(params)
                    n = len(x)
                    aic = 2 * k - 2 * ll
                    bic = k * np.log(n) - 2 * ll

                    ks_stat, ks_p = stats.kstest(x, dist.cdf, args=params)

                    ad_stat = ad_p = None
                    if name in {"normal", "exponential", "logistic", "gumbel_r"}:
                        try:
                            ad_res = stats.anderson(x, dist=name.replace("_r", ""))
                            ad_stat = float(ad_res.statistic)
                            # crude p approx via critical values
                            ad_p = 0.05 if ad_stat < ad_res.critical_values[2] else 0.01
                        except Exception:
                            pass

                    cand.append(
                        {
                            "distribution": name,
                            "params": params,
                            "aic": aic,
                            "bic": bic,
                            "log_likelihood": ll,
                            "ks_statistic": float(ks_stat),
                            "ks_pvalue": float(ks_p),
                            "ad_statistic": ad_stat,
                            "ad_pvalue": ad_p,
                            "fit_quality": float(max(ks_p, 1e-10)),
                        }
                    )
                except Exception:
                    continue

            if not cand:
                continue

            cand.sort(key=lambda z: (z["aic"], -z["fit_quality"]))
            best = cand[0]
            fits[col] = {
                "distribution": best["distribution"],
                "params": best["params"],
                "aic": best["aic"],
                "bic": best["bic"],
                "ks_stat": best["ks_statistic"],
                "ks_pvalue": best["ks_pvalue"],
            }
            fits[col]["_detailed"] = {
                "best_fit": best,
                "alternatives": cand[1:3],
                "total_tested": len(cand),
            }

        return fits
