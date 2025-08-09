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

    def _analyze_relationships(self, data: pd.DataFrame, config) -> Dict[str, Any]:
        """
        Relationship analysis with advanced, but efficient, dependence detection.
        Keeps original buckets and adds optional richer ones. Safe fallbacks.
        """

        # ---------- Optional deps ----------
        try:
            from dcor import distance_correlation as _dcor_pkg
            DCOR_AVAILABLE = True
        except Exception:
            DCOR_AVAILABLE = False

        try:
            from minepy import MINE
            MINE_AVAILABLE = True
        except Exception:
            MINE_AVAILABLE = False

        # ---------- Always-available deps ----------
        from typing import Any, Dict, List

        import numpy as np
        from scipy.stats import kendalltau, rankdata, spearmanr
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        # ---------- Config (centralized knobs with defaults) ----------
        rs            = int(getattr(config, "random_state", 42))
        max_pairs     = int(getattr(config, "rel_max_pairs", 20000))   # global budget
        sample_n      = int(getattr(config, "rel_sample_n", 2000))     # heavy metrics sample
        dcor_gate     = float(getattr(config, "rel_dcor_gate", 0.35))
        mi_gate       = float(getattr(config, "rel_mi_gate", 0.10))
        mic_gate      = float(getattr(config, "rel_mic_gate", 0.20))
        hsic_gate     = float(getattr(config, "rel_hsic_gate", 0.0))   # 0 disables HSIC
        nonlin_gap    = float(getattr(config, "rel_nonlin_gap_gate", 0.15))
        iso_r2_gate   = float(getattr(config, "rel_iso_r2_gate", 0.50))
        tau_gate      = float(getattr(config, "rel_tau_gate", 0.50))
        tail_q        = float(getattr(config, "rel_tail_q", 0.95))
        tail_gate     = float(getattr(config, "rel_tail_gate", 0.05))
        cond_ctrl_k   = int(getattr(config, "rel_cond_ctrl_k", 1))
        ens_w         = tuple(getattr(config, "rel_ens_weights",
                            (0.30, 0.30, 0.20, 0.20)))  # (pearson, spearman, dcor, mi)

        rng = np.random.RandomState(rs)

        # ---------- Inputs ----------
        correlations = CorrelationAnalyzer().analyze(data, config)
        pearson = correlations.get("pearson")
        if pearson is None:
            return {}

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {}

        # ---------- Helpers ----------
        def _subsample_xy(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
            nobs = len(x)
            if nobs <= n:
                return x, y
            idx = rng.choice(nobs, n, replace=False)
            return x[idx], y[idx]

        def _spearman_kendall(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
            try:
                sc = float(abs(spearmanr(x, y).correlation))
            except Exception:
                sc = 0.0
            try:
                tau = float(abs(kendalltau(x, y).correlation))
            except Exception:
                tau = 0.0
            return sc, tau

        def _mi_cont(x: np.ndarray, y: np.ndarray) -> float:
            # Uses sklearn MI; robust to monotone transforms
            try:
                if np.unique(x).size < 2 or np.unique(y).size < 2 or len(x) < 50:
                    return 0.0
                val = mutual_info_regression(x.reshape(-1, 1), y, random_state=rs)[0]
                return float(max(val, 0.0))
            except Exception:
                return 0.0

        def _mic(xs: np.ndarray, ys: np.ndarray) -> float:
            if not MINE_AVAILABLE:
                return 0.0
            try:
                xss, yss = _subsample_xy(xs, ys, sample_n)
                mine = MINE(alpha=0.6, c=15)
                mine.compute_score(xss, yss)
                v = mine.mic()
                return float(v) if np.isfinite(v) else 0.0
            except Exception:
                return 0.0

        def _dcor_fast(xs: np.ndarray, ys: np.ndarray) -> float:
            # Computes distance correlation with fallback and subsample
            x, y = _subsample_xy(xs, ys, sample_n)
            if len(x) < 30:
                return 0.0
            if DCOR_AVAILABLE:
                try:
                    v = _dcor_pkg(x, y)
                    return float(v) if np.isfinite(v) else 0.0
                except Exception:
                    pass
            # Fallback O(n^2) centering, but on subsampled data
            n = len(x)
            X = np.abs(x[:, None] - x[None, :])
            Y = np.abs(y[:, None] - y[None, :])
            X = X - X.mean(0)[None, :] - X.mean(1)[:, None] + X.mean()
            Y = Y - Y.mean(0)[None, :] - Y.mean(1)[:, None] + Y.mean()
            dcov2 = (X * Y).sum() / (n * n)
            dvarx = (X * X).sum() / (n * n)
            dvary = (Y * Y).sum() / (n * n)
            denom = np.sqrt(max(dvarx, 1e-12) * max(dvary, 1e-12))
            return float(np.sqrt(max(dcov2, 0.0)) / (denom + 1e-12))

        def _hsic_gaussian(xs: np.ndarray, ys: np.ndarray) -> float:
            if hsic_gate <= 0:
                return 0.0
            x, y = _subsample_xy(xs, ys, min(sample_n, 1200))
            n = len(x)
            if n < 80:
                return 0.0
            xv = x.reshape(-1, 1); yv = y.reshape(-1, 1)
            Dx = np.abs(xv - xv.T); Dy = np.abs(yv - yv.T)
            sigx = np.median(Dx[Dx > 0]) if np.any(Dx > 0) else 1.0
            sigy = np.median(Dy[Dy > 0]) if np.any(Dy > 0) else 1.0
            Kx = np.exp(-(Dx**2) / (2 * (sigx + 1e-12) ** 2))
            Ky = np.exp(-(Dy**2) / (2 * (sigy + 1e-12) ** 2))
            H = np.eye(n) - np.ones((n, n)) / n
            Kx = H @ Kx @ H
            Ky = H @ Ky @ H
            hsic = (Kx * Ky).sum() / ((n - 1) ** 2)
            # normalize
            nx = np.linalg.norm(Kx, "fro"); ny = np.linalg.norm(Ky, "fro")
            return float(hsic / ((nx * ny / (n - 1) ** 2) + 1e-12))

        def _isotonic_r2(xs: np.ndarray, ys: np.ndarray) -> float:
            if len(xs) < 40:
                return 0.0
            sp = spearmanr(xs, ys).correlation
            sign = 1.0 if (np.isnan(sp) or sp >= 0) else -1.0
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            yf = iso.fit_transform(xs, sign * ys)
            ss_res = np.sum((sign * ys - yf) ** 2)
            ss_tot = np.sum((sign * ys - np.mean(sign * ys)) ** 2) + 1e-12
            return float(max(0.0, 1.0 - ss_res / ss_tot))

        def _tail_dep(xs: np.ndarray, ys: np.ndarray, q: float) -> Dict[str, float]:
            # Empirical copula tail dependence on both tails + asymmetry
            if len(xs) < 100:
                return {"upper": 0.0, "lower": 0.0, "asym_pos": 0.0, "asym_neg": 0.0}
            u = rankdata(xs) / (len(xs) + 1.0)
            v = rankdata(ys) / (len(ys) + 1.0)
            upper = float(np.mean((u > q) & (v > q)))
            lower = float(np.mean((u < 1 - q) & (v < 1 - q)))
            asym_pos = float(np.mean((u > q) & (v < 1 - q)))
            asym_neg = float(np.mean((u < 1 - q) & (v > q)))
            return {"upper": upper, "lower": lower, "asym_pos": asym_pos, "asym_neg": asym_neg}

        def _partial_corr_linear(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray) -> float:
            if Z.ndim == 1: Z = Z[:, None]
            if Z.shape[1] == 0:  # no controls
                c = np.corrcoef(xs, ys)[0, 1]
                return float(0.0 if np.isnan(c) else abs(c))
            lr = LinearRegression()
            try:
                lr.fit(Z, xs); rx = xs - lr.predict(Z)
                lr.fit(Z, ys); ry = ys - lr.predict(Z)
                c = np.corrcoef(rx, ry)[0, 1]
                return float(0.0 if np.isnan(c) else abs(c))
            except Exception:
                return 0.0

        def _regime_split(xs: np.ndarray, ys: np.ndarray) -> Dict[str, Any]:
            # Simple median split on x; robust and cheap
            if len(xs) < 100:
                return {"has_regime": False}
            xm = np.median(xs)
            m1 = xs <= xm
            m2 = ~m1
            if m1.sum() < 20 or m2.sum() < 20:
                return {"has_regime": False}
            c1 = np.corrcoef(xs[m1], ys[m1])[0, 1]
            c2 = np.corrcoef(xs[m2], ys[m2])[0, 1]
            if np.isnan(c1) or np.isnan(c2):
                return {"has_regime": False}
            diff = float(abs(c1 - c2))
            if diff <= 0.3:
                return {"has_regime": False}
            return {"has_regime": True, "regime1_corr": float(c1), "regime2_corr": float(c2),
                    "regime_diff": diff, "split_point": float(xm)}

        def _functional_forms(xs: np.ndarray, ys: np.ndarray) -> Dict[str, Any]:
            # Lightweight model selection among {linear, quadratic, power, exponential}
            if len(xs) < 50:
                return {"best_form": "none", "r2": 0.0, "all_forms": {}}
            x = (xs - xs.mean()) / (xs.std() + 1e-12)
            y = (ys - ys.mean()) / (ys.std() + 1e-12)

            forms = {}
            # linear
            try:
                a = np.polyfit(x, y, 1); yp = np.polyval(a, x)
                forms["linear"] = float(1 - np.var(y - yp) / (np.var(y) + 1e-12))
            except Exception:
                forms["linear"] = 0.0
            # quadratic
            try:
                a = np.polyfit(x, y, 2); yp = np.polyval(a, x)
                forms["quadratic"] = float(1 - np.var(y - yp) / (np.var(y) + 1e-12))
            except Exception:
                forms["quadratic"] = 0.0
            # exponential (y ~ exp(ax+b)) if y>0
            try:
                if (ys > 0).all() and np.var(np.log(ys)) > 1e-6:
                    a = np.polyfit(x, np.log(ys), 1)
                    yp = np.exp(np.polyval(a, x))
                    forms["exponential"] = float(1 - np.var(ys - yp) / (np.var(ys) + 1e-12))
                else:
                    forms["exponential"] = 0.0
            except Exception:
                forms["exponential"] = 0.0
            # power (y ~ x^a) if x,y>0
            try:
                if (xs > 0).all() and (ys > 0).all():
                    lx, ly = np.log(xs), np.log(ys)
                    if np.var(lx) > 1e-6 and np.var(ly) > 1e-6:
                        a = np.polyfit(lx, ly, 1)
                        yp = np.exp(a[1]) * np.power(xs, a[0])
                        forms["power"] = float(1 - np.var(ys - yp) / (np.var(ys) + 1e-12))
                    else:
                        forms["power"] = 0.0
                else:
                    forms["power"] = 0.0
            except Exception:
                forms["power"] = 0.0

            best = max(forms, key=forms.get)
            return {"best_form": best, "r2": float(max(0.0, forms[best])),
                    "all_forms": {k: float(max(0.0, v)) for k, v in forms.items()}}

        def _ensemble_score(xs: np.ndarray, ys: np.ndarray) -> float:
            # Weighted blend: Pearson, Spearman, dCor, MI (+ optional ML R2 small boost)
            s = []
            # Pearson
            pc = np.corrcoef(xs, ys)[0, 1]
            s.append(abs(pc) if np.isfinite(pc) else 0.0)
            # Spearman
            sc, _ = _spearman_kendall(xs, ys)
            s.append(sc)
            # dCor
            s.append(_dcor_fast(xs, ys))
            # MI
            s.append(_mi_cont(xs, ys))
            # Weighted mean
            w = np.array(ens_w, dtype=float)
            val = float((w * np.array(s[:4])).sum() / (w.sum() + 1e-12))
            # Optional tiny ML boost (very light RF)
            if len(xs) > 120:
                try:
                    rf = RandomForestRegressor(n_estimators=40, max_depth=5, random_state=rs, n_jobs=-1)
                    r2 = cross_val_score(rf, xs.reshape(-1, 1), ys, cv=3, scoring="r2")
                    val = float(max(val, np.mean(np.clip(r2, 0.0, 1.0))))
                except Exception:
                    pass
            return val

        # ---------- Buckets (kept + extended) ----------
        patterns: Dict[str, List[Dict[str, Any]]] = {
            "nonlinear": [],
            "complex": [],
            "distance_corr": [],
            "monotonic": [],
            "tail_dependence": [],
            "conditional": [],
            "regime_switching": [],
            "copula_dependence": [],
            "causal_hints": [],         # placeholder: can be filled by lag tests if you add time index
            "functional_forms": [],
            "ensemble_strong": [],
        }

        # ---------- Pair loop with budget ----------
        pair_budget = max_pairs
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if pair_budget <= 0:
                    break

                f1, f2 = numeric_cols[i], numeric_cols[j]
                df = data[[f1, f2]].dropna()
                if len(df) < 40:
                    continue

                x = df[f1].to_numpy(dtype=float)
                y = df[f2].to_numpy(dtype=float)
                n = len(x)

                pc = float(abs(pearson.loc[f1, f2])) if (f1 in pearson.index and f2 in pearson.columns) else 0.0
                sc, tau = _spearman_kendall(x, y)
                mi_val = _mi_cont(x, y)
                mic_val = _mic(x, y) if MINE_AVAILABLE else 0.0
                dc = _dcor_fast(x, y)
                hsic = _hsic_gaussian(x, y) if hsic_gate > 0 else 0.0
                iso_r2 = _isotonic_r2(x, y)
                tail = _tail_dep(x, y, tail_q)
                regime = _regime_split(x, y)
                func = _functional_forms(x, y)
                ens  = _ensemble_score(x, y)

                # -------- Bucketing rules (clear & minimal) --------
                if sc - pc > nonlin_gap and sc > 0.4:
                    patterns["nonlinear"].append({
                        "feature1": f1, "feature2": f2,
                        "pearson": pc, "spearman": sc, "kendall": tau,
                        "nonlinearity_gap": sc - pc,
                        "functional_form": func["best_form"],
                        "functional_r2": func["r2"]
                    })

                if ((mi_val > mi_gate) or (MINE_AVAILABLE and mic_val > mic_gate)) and pc < 0.4:
                    patterns["complex"].append({
                        "feature1": f1, "feature2": f2,
                        "mutual_info": mi_val,
                        "mic": mic_val if MINE_AVAILABLE else None,
                        "pearson": pc
                    })

                if dc > dcor_gate and pc < 0.4:
                    patterns["distance_corr"].append({
                        "feature1": f1, "feature2": f2,
                        "distance_corr": dc, "pearson": pc,
                        "hsic": hsic if hsic_gate > 0 else None
                    })

                if iso_r2 > iso_r2_gate or tau > tau_gate:
                    patterns["monotonic"].append({
                        "feature1": f1, "feature2": f2,
                        "isotonic_r2": iso_r2, "kendall_tau": tau
                    })

                if tail["upper"] > tail_gate or tail["lower"] > tail_gate:
                    patterns["tail_dependence"].append({
                        "feature1": f1, "feature2": f2,
                        "upper_tail_dep": tail["upper"], "lower_tail_dep": tail["lower"],
                        "asymmetric_pos": tail["asym_pos"], "asymmetric_neg": tail["asym_neg"],
                        "tail_type": "symmetric" if abs(tail["upper"] - tail["lower"]) < 0.02 else "asymmetric"
                    })

                if regime.get("has_regime", False):
                    patterns["regime_switching"].append({
                        "feature1": f1, "feature2": f2,
                        **regime,
                        "regime_strength": "strong" if regime["regime_diff"] > 0.5 else "moderate"
                    })

                # Copula body/tail summary via rank-invariant stats
                if tau > 0.3 or tail["upper"] > 0.1:
                    u = rankdata(x) / (n + 1.0)
                    v = rankdata(y) / (n + 1.0)
                    rho = float(np.corrcoef(u, v)[0, 1])
                    patterns["copula_dependence"].append({
                        "feature1": f1, "feature2": f2,
                        "kendall_tau": tau, "spearman_rho": abs(rho),
                        "tail_dep": tail["upper"],
                        "dependence_type": "tail" if tail["upper"] > 0.1 else "body"
                    })

                if ens > 0.4:
                    patterns["ensemble_strong"].append({
                        "feature1": f1, "feature2": f2,
                        "ensemble_score": ens,
                        "pearson": pc, "spearman": sc,
                        "distance_corr": dc, "mutual_info": mi_val,
                        "strength": "very_strong" if ens > 0.7 else "strong"
                    })

                if func["r2"] > 0.5 and func["best_form"] != "linear":
                    patterns["functional_forms"].append({
                        "feature1": f1, "feature2": f2,
                        "functional_form": func["best_form"],
                        "r2_score": func["r2"],
                        "all_forms": func["all_forms"],
                        "complexity": "high" if func["best_form"] in ("power", "exponential") else "moderate"
                    })

                # Conditional (linear) partial corr using top confounders by Pearson
                if cond_ctrl_k > 0 and len(numeric_cols) > 2:
                    others = [c for c in numeric_cols if c not in (f1, f2)]
                    if others:
                        # rank by max(|corr(o,f1)|,|corr(o,f2)|)
                        scs = []
                        for o in others:
                            if o in pearson.index and o in pearson.columns:
                                s1 = abs(float(pearson.loc[f1, o]))
                                s2 = abs(float(pearson.loc[f2, o]))
                                scs.append((o, max(s1, s2)))
                        scs.sort(key=lambda t: t[1], reverse=True)
                        Zcols = [t[0] for t in scs[:cond_ctrl_k]]
                        if Zcols:
                            Z = data[Zcols].loc[df.index].to_numpy(dtype=float)
                            pc_partial = _partial_corr_linear(x, y, Z)
                            if pc_partial > 0.2:
                                patterns["conditional"].append({
                                    "feature1": f1, "feature2": f2,
                                    "partial_corr": pc_partial, "controls": Zcols
                                })

                pair_budget -= 1
            if pair_budget <= 0:
                break

        # ---------- Sort & cap (≤10 per bucket) ----------
        def _score_last(d: Dict[str, Any]) -> float:
            for v in reversed(list(d.values())):
                if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                    return float(v)
            return 0.0

        sort_keys = {
            "nonlinear":        lambda x: x.get("nonlinearity_gap", 0.0),
            "complex":          lambda x: x.get("mutual_info", 0.0),
            "distance_corr":    lambda x: x.get("distance_corr", 0.0),
            "monotonic":        lambda x: max(x.get("isotonic_r2", 0.0), x.get("kendall_tau", 0.0)),
            "tail_dependence":  lambda x: max(x.get("upper_tail_dep", 0.0), x.get("lower_tail_dep", 0.0)),
            "conditional":      lambda x: x.get("partial_corr", 0.0),
            "regime_switching": lambda x: x.get("regime_diff", 0.0),
            "copula_dependence":lambda x: max(x.get("kendall_tau", 0.0), x.get("tail_dep", 0.0)),
            "functional_forms": lambda x: x.get("r2_score", 0.0),
            "ensemble_strong":  lambda x: x.get("ensemble_score", 0.0),
        }

        for k, arr in patterns.items():
            if not arr:
                continue
            key_fn = sort_keys.get(k, _score_last)
            try:
                patterns[k] = sorted(arr, key=key_fn, reverse=True)[:10]
            except Exception:
                patterns[k] = arr[:10]

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
