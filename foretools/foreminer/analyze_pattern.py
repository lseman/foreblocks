from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.stats as sps

from ..aux.adaptive_mi import AdaptiveMI
from ..aux.distance_correlation import DistanceCorrelation
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
    def _log_normal_candidate(
        x: pd.Series, mean: float, skew: float, std: float
    ) -> bool:
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
        dist_summary = (
            DistributionAnalyzer().analyze(data, config).get("summary", pd.DataFrame())
        )
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
                    xt = pd.Series(
                        pt.fit_transform(x.values.reshape(-1, 1)).ravel(), index=x.index
                    )
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

    def _analyze_relationships(self, data: pd.DataFrame, config) -> Dict[str, Any]:
        """
        Relationship analysis with SOTA dependence detection.
        Normalized to match print_detailed_insights expectations.
        """

        # ---------- Optional deps ----------
        try:

            DCOR_AVAILABLE = True
        except Exception:
            DCOR_AVAILABLE = False

        # ---------- Imports ----------
        import numpy as np
        from scipy.optimize import curve_fit
        from scipy.stats import kendalltau, rankdata, spearmanr
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor

        # ---------- Config ----------
        rs = int(getattr(config, "random_state", 42))
        max_pairs = int(getattr(config, "rel_max_pairs", 20000))
        sample_n = int(getattr(config, "rel_sample_n", 2000))
        dcor_gate = float(getattr(config, "rel_dcor_gate", 0.35))
        mi_gate = float(getattr(config, "rel_mi_gate", 0.10))
        hsic_gate = float(getattr(config, "rel_hsic_gate", 0.0))
        nonlin_gap = float(getattr(config, "rel_nonlin_gap_gate", 0.15))
        tail_q = float(getattr(config, "rel_tail_q", 0.95))
        tail_gate = float(getattr(config, "rel_tail_gate", 0.05))
        rng = np.random.default_rng(rs)

        # ---------- Inputs ----------
        numeric = data.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return {}
        std = numeric.std(numeric_only=True)
        numeric = numeric.loc[:, std > 1e-10]
        n_samples, n_features = numeric.shape
        if n_features < 2:
            return {}
        pearson = numeric.corr(method="pearson")
        numeric_cols = numeric.columns.tolist()

        # ---------- Helpers ----------
        def _subsample_xy(x: np.ndarray, y: np.ndarray, n: int):
            nobs = len(x)
            if nobs <= n:
                return x, y
            idx = rng.choice(nobs, n, replace=False)
            return x[idx], y[idx]

        def _spearman_kendall(x: np.ndarray, y: np.ndarray):
            sc = tau = 0.0
            try:
                sc = float(abs(spearmanr(x, y).correlation))
            except Exception:
                pass
            try:
                tau = float(abs(kendalltau(x, y).correlation))
            except Exception:
                pass
            return sc, tau

        def _adaptive_mi(
            x: np.ndarray,
            y: np.ndarray,
            *,
            max_bins: int = 20,
            ks: tuple = (3, 5, 10),
            random_state: int = 42,
        ) -> float:
            """
            Adaptive, fast MI -> correlation-in-[0,1].
            Uses the same estimator selection as AdaptiveMI:
            - many ties -> quantile-binned (<= max_bins)
            - strong monotone -> copula-Gaussian
            - else -> KSG averaged over ks
            """
            # Pairwise finite-mask alignment
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 40:
                return 0.0
            x = x[m].astype(np.float64, copy=False)
            y = y[m].astype(np.float64, copy=False)

            # Lazy, tiny wrapper around AdaptiveMI internals (no dataframe needed)
            ami = AdaptiveMI(
                subsample=10**9,  # no extra subsampling here
                spearman_gate=0.0,  # no gate at pair level
                min_overlap=40,
                ks=ks,
                n_bins=min(max_bins, 32),  # cap bins defensively
                random_state=random_state,
            )

            # Selection logic mirrors AdaptiveMI.matrix():
            ties_x = ami._tie_fraction(x)
            ties_y = ami._tie_fraction(y)
            abs_rho = abs(ami._safe_spearman(x, y))

            if max(ties_x, ties_y) > 0.05:
                mi = ami._mi_binned_quantile(x, y, ami.n_bins)
            elif abs_rho >= 0.85:
                mi = ami._mi_copula_gaussian(x, y)
            else:
                mi = ami._mi_ksg_avg(x, y, ami.ks)

            return float(np.clip(ami._mi_to_coeff(mi), 0.0, 1.0))

        def _dcor(
            x: np.ndarray,
            y: np.ndarray,
            sample_n: int = None,
            random_state: int = 42,
            unbiased: bool = False,
        ) -> float:
            """
            Fast distance correlation using:
            - dcor backend if available
            - otherwise, DistanceCorrelation two-pass tiled fallback (no n×n allocations)
            Keeps optional pairwise subsampling via your _subsample_xy if provided.
            """
            # Pairwise finite-mask alignment
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 30:
                return 0.0
            x = x[m].astype(np.float64, copy=False)
            y = y[m].astype(np.float64, copy=False)

            # Fallback: optimized tiled algorithm
            dc = DistanceCorrelation(
                block_size=1024,
                unbiased=bool(unbiased),
                pearson_gate=0.0,  # no gate at pair level
                max_n=10**9,  # no extra subsample here
                random_state=random_state,
                use_coreset=False,
            )
            val = dc._dcorr_1d(x, y)
            return 0.0 if not np.isfinite(val) else float(np.clip(val, 0.0, 1.0))

        def _nonlinear_gap(x: np.ndarray, y: np.ndarray) -> float:
            if len(x) < 80:
                return 0.0
            lr = LinearRegression().fit(x.reshape(-1, 1), y)
            r2_lin = lr.score(x.reshape(-1, 1), y)
            tree = DecisionTreeRegressor(max_depth=3, random_state=rs).fit(
                x.reshape(-1, 1), y
            )
            r2_tree = tree.score(x.reshape(-1, 1), y)
            return max(0.0, r2_tree - r2_lin)

        def _tail_dep(x: np.ndarray, y: np.ndarray, q: float):
            if len(x) < 100:
                return {
                    "upper_tail_dep": 0,
                    "lower_tail_dep": 0,
                    "asymmetric_pos": 0,
                    "asymmetric_neg": 0,
                }
            u = rankdata(x) / (len(x) + 1.0)
            v = rankdata(y) / (len(y) + 1.0)
            return {
                "upper_tail_dep": float(np.mean((u > q) & (v > q))),
                "lower_tail_dep": float(np.mean((u < 1 - q) & (v < 1 - q))),
                "asymmetric_pos": float(np.mean((u > q) & (v < 1 - q))),
                "asymmetric_neg": float(np.mean((u < 1 - q) & (v > q))),
            }

        def _regime_split(x: np.ndarray, y: np.ndarray):
            if len(x) < 100:
                return {"has_regime": False}
            xm = np.median(x)
            qm1, qm3 = np.quantile(x, [0.25, 0.75])
            regimes = []
            for thr in [xm, qm1, qm3]:
                m1 = x <= thr
                m2 = ~m1
                if m1.sum() < 30 or m2.sum() < 30:
                    continue
                c1 = np.corrcoef(x[m1], y[m1])[0, 1]
                c2 = np.corrcoef(x[m2], y[m2])[0, 1]
                if np.isnan(c1) or np.isnan(c2):
                    continue
                diff = abs(c1 - c2)
                if diff > 0.3:
                    regimes.append((thr, c1, c2, diff))
            if not regimes:
                return {"has_regime": False}
            thr, c1, c2, diff = max(regimes, key=lambda r: r[3])
            return {
                "has_regime": True,
                "split_point": float(thr),
                "regime1_corr": float(c1),
                "regime2_corr": float(c2),
                "regime_diff": diff,
            }

        def _functional_forms(x: np.ndarray, y: np.ndarray):
            if len(x) < 50:
                return {"best_form": "none", "r2": 0.0, "all_forms": {}}
            forms = {}
            try:
                a = np.polyfit(x, y, 1)
                yp = np.polyval(a, x)
                forms["linear"] = 1 - np.var(y - yp) / (np.var(y) + 1e-12)
            except:
                forms["linear"] = 0.0
            try:
                a = np.polyfit(x, y, 2)
                yp = np.polyval(a, x)
                forms["quadratic"] = 1 - np.var(y - yp) / (np.var(y) + 1e-12)
            except:
                forms["quadratic"] = 0.0
            try:

                def logistic(xx, a, b):
                    return 1 / (1 + np.exp(-(a * xx + b)))

                p, _ = curve_fit(logistic, x, y, maxfev=2000)
                yp = logistic(x, *p)
                forms["logistic"] = 1 - np.var(y - yp) / (np.var(y) + 1e-12)
            except:
                forms["logistic"] = 0.0
            xm = np.median(x)
            x1 = (x <= xm).astype(float)
            x2 = (x > xm).astype(float)
            try:
                Z = np.column_stack([x, x1 * x, x2 * x])
                lr = LinearRegression().fit(Z, y)
                yp = lr.predict(Z)
                forms["hinge"] = 1 - np.var(y - yp) / (np.var(y) + 1e-12)
            except:
                forms["hinge"] = 0.0
            best = max(forms, key=forms.get)
            return {"best_form": best, "r2": float(forms[best]), "all_forms": forms}

        # ---------- Buckets ----------
        patterns = {
            k: []
            for k in [
                "nonlinear",
                "complex",
                "distance_corr",
                "monotonic",
                "tail_dependence",
                "regime_switching",
                "functional_forms",
                "ensemble_strong",
            ]
        }

        # ---------- Pair loop ----------
        pair_budget = max_pairs
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if pair_budget <= 0:
                    break
                f1, f2 = numeric_cols[i], numeric_cols[j]
                df = data[[f1, f2]].dropna()
                if len(df) < 40:
                    continue
                x = df[f1].to_numpy(float)
                y = df[f2].to_numpy(float)

                pc = float(abs(pearson.loc[f1, f2]))
                sc, tau = _spearman_kendall(x, y)
                mi_val = _adaptive_mi(x, y)
                dc = _dcor(x, y)
                nonlin = _nonlinear_gap(x, y)
                tail = _tail_dep(x, y, tail_q)
                regime = _regime_split(x, y)
                func = _functional_forms(x, y)

                # nonlinear bucket
                if nonlin >= nonlin_gap:
                    patterns["nonlinear"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "nonlinearity_score": nonlin,
                            "functional_form": func["best_form"],
                            "functional_r2": func["r2"],
                        }
                    )

                # complex dependencies
                if mi_val > mi_gate and pc < 0.4:
                    patterns["complex"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "mutual_info": mi_val,
                            "ensemble_score": np.mean([pc, sc, dc, mi_val]),
                            "copula_tau": 0.0,  # placeholder, not computed
                        }
                    )

                # distance corr
                if dc > dcor_gate and pc < 0.4:
                    patterns["distance_corr"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "distance_corr": dc,
                            "pearson": pc,
                            "strength": "moderate",
                        }
                    )

                # monotonic
                if tau > 0.5:
                    patterns["monotonic"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "kendall_tau": tau,
                            "spearman": sc,
                            "relationship_type": "monotonic",
                        }
                    )

                # tail dependence
                if (
                    tail["upper_tail_dep"] > tail_gate
                    or tail["lower_tail_dep"] > tail_gate
                ):
                    patterns["tail_dependence"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            **tail,
                            "tail_type": (
                                "upper"
                                if tail["upper_tail_dep"] > tail["lower_tail_dep"]
                                else "lower"
                            ),
                        }
                    )

                # regime switching
                if regime.get("has_regime", False):
                    patterns["regime_switching"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            **regime,
                            "regime_strength": (
                                "strong" if regime["regime_diff"] > 0.5 else "weak"
                            ),
                        }
                    )

                # functional forms
                if func["r2"] > 0.5 and func["best_form"] != "linear":
                    patterns["functional_forms"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "functional_form": func["best_form"],
                            "r2_score": func["r2"],  # <== normalized name
                            "all_forms": func["all_forms"],
                            "complexity": "moderate",
                        }
                    )

                # ensemble
                ens = np.mean([pc, sc, dc, mi_val])
                if ens > 0.4:
                    patterns["ensemble_strong"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "ensemble_score": ens,
                            "pearson": pc,
                            "spearman": sc,
                            "distance_corr": dc,
                            "mutual_info": mi_val,
                            "strength": "strong",
                        }
                    )

                pair_budget -= 1
            if pair_budget <= 0:
                break

        # ---------- Top 10 each ----------
        for k, v in patterns.items():
            if v:
                patterns[k] = sorted(
                    v,
                    key=lambda d: max(
                        [val for val in d.values() if isinstance(val, (int, float))],
                        default=0,
                    ),
                    reverse=True,
                )[:10]

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
