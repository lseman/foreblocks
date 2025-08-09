import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .foreminer_aux import *

# Optional deps (used if available)
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except Exception:
    HAS_RUPTURES = False

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

# ——— Lightweight jit (optional) ———
if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _cusum_stat_jit(x):
        n = x.shape[0]
        mu = 0.0
        for i in range(n):
            mu += x[i]
        mu /= n
        s = 0.0
        mx = 0.0
        for i in range(n):
            s += x[i] - mu
            v = s if s >= 0 else -s
            if v > mx:
                mx = v
        var = 0.0
        for i in range(n):
            d = x[i] - mu
            var += d * d
        var /= (n - 1) if n > 1 else 1
        sd = np.sqrt(var) if var > 0 else 1.0
        return mx / (sd * np.sqrt(n)) if sd > 0 else 0.0
else:
    def _cusum_stat_jit(x):
        # Pure NumPy fallback
        x = np.asarray(x)
        n = len(x)
        if n < 3:
            return 0.0
        mu = x.mean()
        s = np.cumsum(x - mu)
        mx = np.max(np.abs(s))
        sd = x.std(ddof=1) or 1.0
        return float(mx / (sd * np.sqrt(n)))

class TimeSeriesAnalyzer(AnalysisStrategy):
    """SOTA comprehensive time series analysis with advanced ML techniques (refactored)"""

    @property
    def name(self) -> str:
        return "timeseries"

    # -------------------- Public API --------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove time/target
        time_col = getattr(config, "time_col", None)
        target_col = getattr(config, "target", None)
        for c in (time_col, target_col):
            if c in numeric_cols:
                numeric_cols.remove(c)

        # Light preprocessing/caches shared across modules
        # Build a compact view: drop NaNs per series once
        series_map: Dict[str, pd.Series] = {}
        for c in numeric_cols:
            s = data[c].dropna()
            if len(s) >= 10:
                series_map[c] = s

        # Parallel workers (mild)
        workers = min(4, max(1, len(series_map) // 3 or 1))

        # Run sections (some reuse helpers and caches)
        results: Dict[str, Any] = {}
        results["stationarity"] = self._analyze_stationarity(data, list(series_map.keys()), config, series_map)
        results["temporal_patterns"] = self._analyze_temporal_patterns(data, list(series_map.keys()), config, series_map)
        results["lag_suggestions"] = self._suggest_lag_features(data, list(series_map.keys()), config, series_map)
        results["seasonality_tests"] = self._advanced_seasonality_tests(data, list(series_map.keys()), config, series_map)
        results["change_point_detection"] = self._detect_change_points(data, list(series_map.keys()), config, series_map)
        results["regime_switching"] = self._detect_regime_switching(data, list(series_map.keys()), config, series_map)
        results["forecasting_readiness"] = self._assess_forecasting_readiness(data, list(series_map.keys()), config, series_map)
        results["volatility_analysis"] = self._analyze_volatility(data, list(series_map.keys()), config, series_map)
        results["cyclical_patterns"] = self._detect_cyclical_patterns(data, list(series_map.keys()), config, series_map)
        results["causality_analysis"] = self._granger_causality_analysis(data, list(series_map.keys()), config)

        return results

    # -------------------- Helpers --------------------
    @staticmethod
    def _cap_max(items: List[str], cap: int) -> List[str]:
        return items[:cap] if cap and cap > 0 else items

    @staticmethod
    def _safe_std(x: pd.Series) -> float:
        v = float(np.std(x.values, ddof=1)) if len(x) > 1 else 0.0
        return v

    @staticmethod
    def _robust_iqr(x: pd.Series) -> Tuple[float, float, float]:
        q25, q75 = np.nanpercentile(x, [25, 75])
        iqr = q75 - q25
        return q25, q75, iqr

    @lru_cache(maxsize=256)
    def _dominant_period(self, key: str, values_hash: int, n: int) -> int:
        # Simple cache shim keyed by series identity
        from statsmodels.tsa.stattools import acf
        max_lags = min(n // 3, 100)
        acf_vals = acf(np.asarray(values_hash, dtype=np.float64), nlags=1)  # dummy safeguard
        # The cache key forces reuse but we don't compute here; see _estimate_period
        return 12

    # -------------------- Modules --------------------
    def _analyze_stationarity(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

        rows = []
        cols = self._cap_max(numeric_cols, cap=32)
        for col in cols:
            s = series_map.get(col)
            if s is None or len(s) < 20:
                continue
            try:
                adf_stat, adf_p, _, _, adf_crit, _ = adfuller(s, autolag="AIC")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    kpss_stat, kpss_p, _, kpss_crit = kpss(s, regression="c", nlags="auto")

                za_stat = za_p = None
                if len(s) > 50:
                    try:
                        za_stat, za_p, *_ = zivot_andrews(s, model="c")
                    except Exception:
                        pass

                variance_ratio = self._variance_ratio_test(s)

                adf_stationary = adf_p < config.confidence_level
                kpss_stationary = kpss_p > config.confidence_level
                if adf_stationary and kpss_stationary:
                    stype = "stationary"
                elif (not adf_stationary) and (not kpss_stationary):
                    stype = "non_stationary"
                elif adf_stationary and (not kpss_stationary):
                    stype = "trend_stationary"
                else:
                    stype = "difference_stationary"

                # One-step diff and seasonal diff (lightweight, guarded)
                diff_info = {}
                if stype != "stationary" and len(s) > 12:
                    try:
                        d1 = s.diff().dropna()
                        d1_adf_p = adfuller(d1, autolag="AIC")[1] if len(d1) > 10 else 1.0
                        diff_info = dict(diff_adf_pvalue=d1_adf_p, diff_stationary=bool(d1_adf_p < config.confidence_level))
                    except Exception:
                        pass
                    try:
                        seas = s.diff(12).dropna()
                        if len(seas) > 10:
                            seas_p = adfuller(seas, autolag="AIC")[1]
                            diff_info.update(seasonal_diff_adf_pvalue=seas_p, seasonal_diff_stationary=bool(seas_p < config.confidence_level))
                    except Exception:
                        pass

                rows.append({
                    "feature": col,
                    "adf_statistic": adf_stat,
                    "adf_pvalue": adf_p,
                    "adf_critical_1pct": adf_crit.get("1%", np.nan),
                    "adf_critical_5pct": adf_crit.get("5%", np.nan),
                    "kpss_statistic": kpss_stat,
                    "kpss_pvalue": kpss_p,
                    "kpss_critical_1pct": kpss_crit.get("1%", np.nan),
                    "kpss_critical_5pct": kpss_crit.get("5%", np.nan),
                    "za_statistic": za_stat,
                    "za_pvalue": za_p,
                    "variance_ratio": variance_ratio,
                    "is_stationary_adf": adf_stationary,
                    "is_stationary_kpss": kpss_stationary,
                    "stationarity_type": stype,
                    "consensus_stationary": bool(adf_stationary and kpss_stationary),
                    "is_stationary": bool(adf_stationary and kpss_stationary),
                    **diff_info,
                })
            except Exception as e:
                print(f"Stationarity failed for {col}: {e}")
        return pd.DataFrame(rows)

    def _variance_ratio_test(self, series: pd.Series, lags: int = 4) -> float:
        try:
            n = len(series)
            if n < lags * 4:
                return np.nan
            r = series.pct_change().to_numpy()[1:]
            if r.size < lags + 2:
                return np.nan
            var1 = np.var(r, ddof=1)
            if var1 <= 0:
                return np.nan
            # Sum of k consecutive returns; scale variance by k
            ksum = pd.Series(r).rolling(lags).sum().to_numpy()[lags - 1 :]
            if ksum.size < 2:
                return np.nan
            vark = np.var(ksum, ddof=1) / lags
            return float(vark / var1)
        except Exception:
            return np.nan

    def _analyze_temporal_patterns(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        from scipy.signal import find_peaks, periodogram
        from scipy.stats import linregress
        from statsmodels.tsa.seasonal import STL

        out: Dict[str, Any] = {}
        cols = self._cap_max(numeric_cols, cap=8)
        for col in cols:
            s = series_map.get(col)
            if s is None or len(s) < 24:
                continue
            try:
                x = np.arange(len(s))
                slope, _, r2, p, _ = linregress(x, s)
                std = self._safe_std(s)
                trend_info = {
                    "linear_slope": slope,
                    "linear_r_squared": r2**2 if np.ndim(r2) else r2,
                    "linear_p_value": p,
                    "linear_significant": bool(p < 0.05),
                    "trend_direction": self._classify_trend(slope, p, std),
                    "slope_normalized": float(slope / std) if std > 0 else 0.0,
                    "trend_strength_category": self._categorize_trend_strength(abs(slope), std),
                }

                seasonality = {}
                # Robust period guess using ACF peak; safe bounds
                period = self._estimate_period(s)
                seasonal = max(7, min(period, max(7, len(s)//6)))
                if seasonal % 2 == 0:
                    seasonal += 1
                try:
                    stl = STL(s, seasonal=seasonal, period=period, robust=True).fit()
                    tot = float(np.var(s))
                    seas_var = float(np.var(stl.seasonal))
                    seas_strength = (seas_var / tot) if tot > 0 else 0.0
                    seasonality.update(
                        stl_seasonal_strength=seas_strength,
                        stl_period=int(period),
                        stl_classification=self._classify_seasonality(seas_strength),
                        trend_strength=(float(np.var(stl.trend.dropna())) / tot if tot > 0 else 0.0),
                    )
                except Exception as e:
                    seasonality["stl_error"] = str(e)

                # Spectral dominant frequency
                try:
                    freqs, psd = periodogram(s, scaling="density")
                    if len(freqs) > 1:
                        idx = int(np.argmax(psd[1:]) + 1)
                        f = freqs[idx]
                        dom_period = (1 / f) if f > 0 else None
                        seasonality.update(dominant_period_periodogram=dom_period, spectral_peak_power=float(psd[idx]))
                except Exception:
                    pass

                out[f"{col}_trend"] = trend_info
                out[f"{col}_seasonality"] = seasonality

                # Volatility snapshot (reused elsewhere but cheap here)
                r = s.pct_change().dropna()
                if len(r) > 20:
                    rv = r.rolling(window=min(10, max(3, len(r)//10))).std()
                    vol_autocorr = float(rv.dropna().autocorr(lag=1)) if rv.notna().sum() > 1 else 0.0
                    out[f"{col}_volatility"] = dict(
                        returns_volatility=float(r.std()),
                        rolling_vol_mean=float(rv.mean()),
                        vol_autocorr=vol_autocorr,
                        volatility_persistent=bool(abs(vol_autocorr) > 0.3),
                    )

                # CUSUM break (fast)
                stat = _cusum_stat_jit(s.to_numpy(dtype=np.float64))
                idx = int(np.argmax(np.abs(np.cumsum(s - s.mean())))) if len(s) > 2 else 0
                out[f"{col}_structural_breaks"] = dict(
                    cusum_statistic=float(stat),
                    potential_break_point=float(idx / len(s)) if len(s) else None,
                    break_significant=bool(stat > 1.5),
                )
            except Exception as e:
                print(f"Temporal patterns failed for {col}: {e}")
        return out

    def _estimate_period(self, series: pd.Series) -> int:
        from scipy.signal import find_peaks
        from statsmodels.tsa.stattools import acf
        try:
            n = len(series)
            max_lags = min(n // 3, 100)
            if max_lags < 3:
                return max(2, n // 4)
            a = acf(series, nlags=max_lags, fft=True)
            peaks, _ = find_peaks(a[1:], height=0.15)
            if peaks.size:
                dom = int(peaks[np.argmax(a[peaks + 1])] + 1)
                return max(2, min(dom, max(12, n // 6)))
            # fallback by length
            if n >= 365: return 365
            if n >= 52:  return 52
            if n >= 24:  return 12
            return max(2, n // 4)
        except Exception:
            n = len(series)
            return max(2, min(12, n // 4))

    def _classify_trend(self, slope: float, p_value: float, std: float) -> str:
        if p_value > 0.05:
            return "no_trend"
        thr = std * 0.01
        if slope > thr:
            return "increasing" if slope > 2 * thr else "weakly_increasing"
        if slope < -thr:
            return "decreasing" if slope < -2 * thr else "weakly_decreasing"
        return "stable"

    def _classify_seasonality(self, strength: float) -> str:
        if strength > 0.4: return "strong"
        if strength > 0.2: return "moderate"
        if strength > 0.05: return "weak"
        return "none"

    def _categorize_trend_strength(self, abs_slope: float, std: float) -> str:
        n = abs_slope / (std + 1e-8)
        if n > 0.1: return "strong"
        if n > 0.05: return "moderate"
        if n > 0.01: return "weak"
        return "none"

    def _suggest_lag_features(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        from statsmodels.tsa.ar_model import AutoReg
        from statsmodels.tsa.stattools import acf, pacf

        out: Dict[str, Dict[str, Any]] = {}
        cols = self._cap_max(numeric_cols, cap=24)
        for col in cols:
            s = series_map.get(col)
            if s is None:
                continue
            n = len(s)
            max_lags = max(6, min(24, n // 10))
            if n < max_lags + 20:
                continue
            try:
                pvals = pacf(s, nlags=max_lags, method="ols")[1:]
                avals = acf(s, nlags=max_lags, fft=True)[1:]
                ci = 1.96 / np.sqrt(n)
                scored = []
                for lag in range(1, max_lags + 1):
                    ps, as_ = abs(pvals[lag - 1]), abs(avals[lag - 1])
                    sc = (0.6 * ps if ps > ci else 0.0) + (0.4 * as_ if as_ > ci else 0.0)
                    if lag in (12, 24, 52) and lag < n // 3:
                        sc *= 1.15
                    if lag > n // 5:
                        sc *= 0.85
                    if sc > 0.1:
                        scored.append((lag, sc))
                scored.sort(key=lambda x: x[1], reverse=True)
                top = [k for k, _ in scored[:5]]

                ar_order = None
                if n >= 40:
                    aics = []
                    max_ar = min(10, n // 10)
                    for p in range(1, max_ar + 1):
                        try:
                            aics.append((p, AutoReg(s, lags=p, old_names=False).fit().aic))
                        except Exception:
                            pass
                    if aics:
                        ar_order = min(aics, key=lambda x: x[1])[0]

                out[col] = {
                    "autocorr_lags": top,
                    "ar_optimal_order": ar_order,
                    "information_criteria": {},  # Kept for compatibility
                    "cross_correlation_lags": {},  # Kept; compute elsewhere if needed
                    "recommended_lags": (top[:3] if top else ([ar_order] if ar_order else [])),
                    "lag_selection_method": "multi_criteria",
                    "seasonal_lags": [k for k in top if k in (12, 24, 52)],
                }
            except Exception as e:
                print(f"Lag suggestion failed for {col}: {e}")
        return out

    def _advanced_seasonality_tests(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        from scipy.stats import friedmanchisquare, kruskal
        from statsmodels.stats.diagnostic import acorr_ljungbox

        time_col = getattr(config, "time_col", None)
        if not time_col or time_col not in data.columns:
            return {}

        out: Dict[str, Any] = {}
        cols = self._cap_max(numeric_cols, cap=10)
        for col in cols:
            try:
                df = data[[time_col, col]].dropna()
                if len(df) < 24:
                    continue
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col).sort_index()
                s = df[col]
                res: Dict[str, Any] = {}

                # Month seasonality via Friedman (needs enough yearly coverage)
                if len(s) >= 36:
                    groups = []
                    for m in range(1, 13):
                        g = s[s.index.month == m].values
                        if g.size >= 3:
                            groups.append(g)
                    if len(groups) >= 6:
                        L = min(len(g) for g in groups)
                        if L >= 3:
                            stat, p = friedmanchisquare(*[g[:L] for g in groups])
                            res["friedman_test"] = {"statistic": float(stat), "p_value": float(p), "seasonal_significant": bool(p < 0.05)}

                # Day-of-week Kruskal
                try:
                    dows = [s[s.index.dayofweek == d] for d in range(7)]
                    dows = [d for d in dows if len(d) >= 5]
                    if len(dows) >= 5:
                        st, p = kruskal(*dows)
                        res["kruskal_dow"] = {"statistic": float(st), "p_value": float(p), "dow_effect_significant": bool(p < 0.05)}
                except Exception:
                    pass

                # Seasonal Ljung-Box
                seas_lags = [lag for lag in (12, 24, 52) if len(s) > lag + 10]
                for lag in seas_lags:
                    try:
                        lb = acorr_ljungbox(s, lags=[lag], return_df=True)
                        res[f"ljungbox_lag_{lag}"] = {
                            "statistic": float(lb["lb_stat"].iloc[0]),
                            "p_value": float(lb["lb_pvalue"].iloc[0]),
                            "seasonal_correlation": bool(float(lb["lb_pvalue"].iloc[0]) < 0.05),
                        }
                    except Exception:
                        pass

                # Welch PSD peaks
                try:
                    from scipy.signal import find_peaks, welch
                    f, Pxx = welch(s, nperseg=min(len(s)//4, 256))
                    if len(Pxx):
                        thr = np.percentile(Pxx, 90)
                        idx = find_peaks(Pxx, height=thr)[0]
                        dom = []
                        for i in idx[:5]:
                            if f[i] > 0:
                                dom.append({"period": float(1.0/f[i]), "frequency": float(f[i]), "power": float(Pxx[i])})
                        res["spectral_analysis"] = {"dominant_periods": sorted(dom, key=lambda x: x["power"], reverse=True)}
                except Exception:
                    pass

                out[col] = res
            except Exception as e:
                print(f"Seasonality tests failed for {col}: {e}")
        return out

    def _detect_change_points(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        time_col = getattr(config, "time_col", None)
        if not time_col or time_col not in data.columns:
            return {}

        out: Dict[str, Any] = {}
        cols = self._cap_max(numeric_cols, cap=10)
        for col in cols:
            try:
                df = data[[time_col, col]].dropna()
                if len(df) < 50:
                    continue
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col).sort_index()
                s = df[col]
                found: Dict[str, Any] = {}

                # If ruptures is available, prefer a fast model (l2, binary segmentation)
                if HAS_RUPTURES and len(s) >= 80:
                    try:
                        algo = rpt.Binseg(model="l2").fit(s.values.astype(float))
                        bkps = algo.predict(n_bkps=min(3, max(1, len(s)//100)))  # up to 3 breakpoints
                        points = []
                        for bp in bkps[:-1]:
                            i = int(bp) - 1
                            if 10 <= i < len(s) - 10:
                                points.append({
                                    "index": i,
                                    "timestamp": s.index[i],
                                    "magnitude": float(abs(s.iloc[:i].mean() - s.iloc[i:].mean())),
                                })
                        if points:
                            found["ruptures_binseg"] = {"change_points": points}
                    except Exception:
                        pass

                # CUSUM fallback
                try:
                    stat = _cusum_stat_jit(s.to_numpy(dtype=np.float64))
                    if stat > 1.5:
                        i = int(np.argmax(np.abs(np.cumsum(s - s.mean()))))
                        found["cusum"] = {
                            "change_point": s.index[i],
                            "statistic": float(stat),
                            "pre_change_mean": float(s.iloc[:i].mean()) if i > 1 else None,
                            "post_change_mean": float(s.iloc[i:].mean()) if i < len(s) - 1 else None,
                        }
                except Exception:
                    pass

                if found:
                    out[col] = found
            except Exception as e:
                print(f"Change points failed for {col}: {e}")
        return out

    def _detect_regime_switching(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        cols = self._cap_max(numeric_cols, cap=3)
        for col in cols:
            try:
                s = series_map.get(col)
                if s is None or len(s) < 100:
                    continue
                r = s.pct_change().dropna()
                if len(r) < 60:
                    continue

                # Simple GMM on [ret, ma5, std5]
                try:
                    from sklearn.mixture import GaussianMixture
                    rv = r.values
                    ma5 = pd.Series(rv).rolling(5).mean().to_numpy()
                    sd5 = pd.Series(rv).rolling(5).std().to_numpy()
                    X = np.column_stack([rv, ma5, sd5])
                    X = X[~np.isnan(X).any(axis=1)]
                    if len(X) < 50:
                        continue

                    best_n, best_score = 2, -np.inf
                    for k in (2, 3, 4):
                        try:
                            gm = GaussianMixture(n_components=k, random_state=getattr(config, "random_state", 0))
                            gm.fit(X)
                            sc = gm.score(X)
                            if sc > best_score:
                                best_score, best_n = sc, k
                        except Exception:
                            pass
                    gm = GaussianMixture(n_components=best_n, random_state=getattr(config, "random_state", 0)).fit(X)
                    labels = gm.predict(X)
                    probs = gm.predict_proba(X)
                    stats = {}
                    for g in range(best_n):
                        mask = labels == g
                        if not np.any(mask):
                            continue
                        stats[f"regime_{g}"] = {
                            "mean_return": float(rv[-len(labels):][mask].mean()),
                            "volatility": float(rv[-len(labels):][mask].std()),
                            "duration_pct": float(mask.mean() * 100.0),
                            "persistence": float(self._calculate_regime_persistence(labels, g)),
                        }
                    # transitions
                    transitions = []
                    for i in range(1, len(labels)):
                        if labels[i] != labels[i-1]:
                            # align timestamps to last len(labels) of r.index
                            idx = r.index[-len(labels):][i] if len(r) >= len(labels) else None
                            transitions.append({"from_regime": int(labels[i-1]), "to_regime": int(labels[i]), "timestamp": idx})
                    out[col] = {
                        "markov_switching": {
                            "n_regimes": int(best_n),
                            "regime_statistics": stats,
                            "transitions": transitions[-10:],
                            "current_regime": int(labels[-1]),
                            "current_regime_probability": float(np.max(probs[-1])),
                        }
                    }
                except Exception:
                    pass

                # Vol regime snapshot
                try:
                    vol = r.rolling(min(20, max(5, len(r)//10))).std()
                    thr = vol.quantile(0.7)
                    mask = vol > thr
                    changes = (mask != mask.shift(1)).sum()
                    out.setdefault(col, {})
                    out[col]["volatility_switching"] = {
                        "vol_threshold": float(thr),
                        "high_vol_pct": float(mask.mean() * 100.0),
                        "avg_regime_length": float(len(mask) / (changes + 1)),
                        "regime_switches": int(changes),
                        "current_regime": "high" if bool(mask.iloc[-1]) else "low",
                    }
                except Exception:
                    pass
            except Exception as e:
                print(f"Regime switching failed for {col}: {e}")
        return out

    def _calculate_regime_persistence(self, labels: np.ndarray, regime: int) -> float:
        runs, cur = [], 0
        for v in labels:
            if v == regime:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur: runs.append(cur)
        return float(np.mean(runs)) if runs else 0.0

    def _assess_forecasting_readiness(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        cols = self._cap_max(numeric_cols, cap=10)
        for col in cols:
            s = series_map.get(col)
            if s is None or len(s) < 20:
                continue
            try:
                scores: Dict[str, float] = {}
                n = len(s)
                scores["data_sufficiency"] = float(min(n / 100.0, 1.0))
                scores["completeness"] = float(1.0 - data[col].isnull().mean())

                # Stationarity
                try:
                    from statsmodels.tsa.stattools import adfuller
                    p = adfuller(s)[1]
                    scores["stationarity"] = 1.0 if p < 0.05 else 0.5
                except Exception:
                    scores["stationarity"] = 0.5

                # Trend
                from scipy.stats import linregress
                x = np.arange(n)
                slope, _, r, p, _ = linregress(x, s)
                scores["trend_strength"] = float(min((abs(r) if p < 0.05 else 0.0) * 2, 1.0))

                # Seasonality
                seas = 0.0
                try:
                    from statsmodels.tsa.seasonal import STL
                    if n >= 24:
                        P = min(12, max(4, n // 6))
                        stl = STL(s, seasonal=(P if P % 2 else P + 1), period=P, robust=True).fit()
                        tvar = float(np.var(s)); svar = float(np.var(stl.seasonal))
                        seas = float(min((svar / tvar) * 2.0, 1.0)) if tvar > 0 else 0.0
                except Exception:
                    pass
                scores["seasonality"] = seas

                # Noise
                r = s.pct_change().dropna()
                noise = float(r.std()) if len(r) > 1 else 0.0
                scores["signal_to_noise"] = float(max(0.0, 1.0 - min(noise, 1.0)))

                # Autocorr
                ac = 0.0
                try:
                    from statsmodels.tsa.stattools import acf
                    a = acf(s, nlags=min(20, n // 2), fft=True)
                    sig = int((np.abs(a[1:]) > 1.96 / np.sqrt(n)).sum())
                    ac = float(min(sig / 10.0, 1.0))
                except Exception:
                    pass
                scores["autocorrelation"] = ac

                # Outliers
                q25, q75, iqr = self._robust_iqr(s)
                if iqr <= 0:
                    out_pct = 0.0
                else:
                    out_cnt = int(((s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))).sum())
                    out_pct = out_cnt / n
                scores["outlier_robustness"] = float(max(0.0, 1.0 - 5.0 * out_pct))

                weights = {
                    "data_sufficiency": 0.20,
                    "completeness": 0.15,
                    "stationarity": 0.15,
                    "trend_strength": 0.10,
                    "seasonality": 0.10,
                    "signal_to_noise": 0.15,
                    "autocorrelation": 0.10,
                    "outlier_robustness": 0.05,
                }
                overall = float(sum(scores[k] * weights[k] for k in scores))
                if overall >= 0.8: lvl = "excellent"
                elif overall >= 0.6: lvl = "good"
                elif overall >= 0.4: lvl = "fair"
                else: lvl = "poor"

                recs: List[str] = []
                if scores["data_sufficiency"] < 0.5: recs.append("Collect more historical data")
                if scores["completeness"] < 0.8: recs.append("Address missing values")
                if scores["stationarity"] < 0.7: recs.append("Apply differencing or transformation")
                if scores["outlier_robustness"] < 0.7: recs.append("Clean outliers")
                if scores["signal_to_noise"] < 0.5: recs.append("Apply smoothing techniques")

                out[col] = {
                    "overall_score": overall,
                    "readiness_level": lvl,
                    "component_scores": scores,
                    "recommendations": recs,
                }
            except Exception as e:
                print(f"Forecast readiness failed for {col}: {e}")
        return out

    def _analyze_volatility(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        cols = self._cap_max(numeric_cols, cap=10)
        for col in cols:
            s = series_map.get(col)
            if s is None or len(s) < 50:
                continue
            try:
                r = s.pct_change().dropna()
                if len(r) < 20:
                    continue
                basic = {
                    "returns_mean": float(r.mean()),
                    "returns_std": float(r.std()),
                    "annualized_volatility": float(r.std() * np.sqrt(252)),
                    "skewness": float(r.skew()),
                    "kurtosis": float(r.kurtosis()),
                    "sharpe_ratio": float((r.mean() / r.std()) if r.std() > 0 else 0.0),
                }
                vol: Dict[str, Any] = {"basic_measures": basic}

                # ARCH LM test on demand (quick)
                try:
                    from statsmodels.stats.diagnostic import het_arch
                    st, p, _, _ = het_arch(r, nlags=min(5, max(1, len(r)//20)))
                    vol["arch_test"] = {"statistic": float(st), "p_value": float(p), "volatility_clustering": bool(p < 0.05)}
                except Exception:
                    pass

                # Rolling vol
                try:
                    w = min(20, max(5, len(r)//10))
                    rv = r.rolling(w).std()
                    vol["rolling_volatility"] = {
                        "mean": float(rv.mean()),
                        "std": float(rv.std()),
                        "min": float(rv.min()),
                        "max": float(rv.max()),
                        "volatility_of_volatility": float((rv.std() / rv.mean()) if rv.mean() > 0 else 0.0),
                    }
                    qh, ql = float(rv.quantile(0.75)), float(rv.quantile(0.25))
                    hi = int((rv > qh).sum()); lo = int((rv < ql).sum()); denom = max(1, int(rv.notna().sum()))
                    vol["volatility_regimes"] = {
                        "high_vol_periods": hi,
                        "low_vol_periods": lo,
                        "high_vol_pct": float(hi / denom * 100.0),
                        "low_vol_pct": float(lo / denom * 100.0),
                    }
                except Exception:
                    pass

                # Squared-return acf as GARCH signal
                try:
                    from statsmodels.tsa.stattools import acf
                    sr = (r**2).dropna()
                    a = acf(sr, nlags=min(10, len(sr)//4))
                    sig = int((np.abs(a[1:]) > 1.96 / np.sqrt(len(sr))).sum())
                    vol["garch_effects"] = {
                        "squared_returns_acf": (a[1:5].tolist() if len(a) > 5 else a[1:].tolist()),
                        "significant_lags": int(sig),
                        "garch_suitable": bool(sig > 0),
                    }
                except Exception:
                    pass

                # Risk measures
                try:
                    var95 = float(np.percentile(r, 5))
                    var99 = float(np.percentile(r, 1))
                    es95 = float(r[r <= var95].mean()) if (r <= var95).any() else float("nan")
                    es99 = float(r[r <= var99].mean()) if (r <= var99).any() else float("nan")
                    cum = r.cumsum()
                    mdd = float((cum - cum.cummax()).min())
                    vol["risk_measures"] = {"var_95": var95, "var_99": var99, "expected_shortfall_95": es95, "expected_shortfall_99": es99, "max_drawdown": mdd}
                except Exception:
                    pass

                out[col] = vol
            except Exception as e:
                print(f"Volatility failed for {col}: {e}")
        return out

    def _detect_cyclical_patterns(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig, series_map: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        cols = self._cap_max(numeric_cols, cap=10)
        for col in cols:
            s = series_map.get(col)
            if s is None or len(s) < 50:
                continue
            try:
                cyc: Dict[str, Any] = {}
                # Fourier
                try:
                    from scipy.fft import fft, fftfreq
                    m = s.rolling(window=min(12, max(3, len(s)//10))).mean()
                    dt = (s - m).dropna()
                    if len(dt) > 20:
                        F = fft(dt.values)
                        freqs = fftfreq(len(dt))
                        P = (np.abs(F) ** 2)
                        pos = slice(1, len(P)//2)
                        power = P[pos]; fpos = freqs[pos]
                        from scipy.signal import find_peaks
                        idx = find_peaks(power, height=np.percentile(power, 85))[0]
                        dom = []
                        for i in idx[:5]:
                            f = fpos[i]
                            if f > 0:
                                dom.append({"period": float(1.0/f), "frequency": float(f), "power": float(power[i]), "power_normalized": float(power[i] / power.sum())})
                        cyc["fourier_analysis"] = {"dominant_cycles": sorted(dom, key=lambda x: x["power"], reverse=True), "total_cycles_detected": len(dom)}
                except Exception:
                    pass

                # Wavelet (cheap ricker)
                try:
                    from scipy.signal import cwt, ricker
                    scales = np.arange(2, min(50, max(10, len(s)//12)))
                    C = cwt(s.values, ricker, scales)
                    e = np.sum(C * C, axis=1)
                    j = int(np.argmax(e))
                    cyc["wavelet_analysis"] = {"dominant_scale": int(scales[j]), "energy_at_dominant_scale": float(e[j]), "scales_analyzed": int(len(scales))}
                except Exception:
                    pass

                # HP filter “business cycle”
                try:
                    from statsmodels.tsa.filters.hp_filter import hpfilter
                    cyc_comp, trend = hpfilter(s, lamb=1600)
                    from scipy.signal import find_peaks
                    pk = find_peaks(cyc_comp.values)[0]; tr = find_peaks(-cyc_comp.values)[0]
                    dur = np.diff(pk) if len(pk) > 1 else []
                    cyc["business_cycle"] = {
                        "cycle_component_std": float(cyc_comp.std()),
                        "trend_component_std": float(trend.std()),
                        "n_peaks": int(len(pk)), "n_troughs": int(len(tr)),
                        "avg_cycle_duration": (float(np.mean(dur)) if len(dur) else None),
                        "cycle_amplitude": float(cyc_comp.max() - cyc_comp.min()),
                    }
                except Exception:
                    pass

                # Phase via Hilbert
                try:
                    from scipy.signal import hilbert
                    m = s.rolling(window=min(12, max(3, len(s)//10))).mean()
                    dt = (s - m).dropna()
                    if len(dt) > 20:
                        z = hilbert(dt.values)
                        phase = np.angle(z); amp = np.abs(z)
                        dphi = np.diff(phase)
                        phi_cons = float(1.0 - (np.std(dphi) / np.pi))
                        cyc["phase_analysis"] = {
                            "phase_consistency": phi_cons,
                            "mean_amplitude": float(np.mean(amp)),
                            "amplitude_variability": float(np.std(amp) / (np.mean(amp) + 1e-12)),
                        }
                except Exception:
                    pass

                out[col] = cyc
            except Exception as e:
                print(f"Cyclical patterns failed for {col}: {e}")
        return out

    def _granger_causality_analysis(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        if len(numeric_cols) < 2:
            return {}
        out: Dict[str, Any] = {}
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            cols = self._cap_max(numeric_cols, cap=5)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    var1, var2 = cols[i], cols[j]
                    if var1 not in data.columns or var2 not in data.columns:
                        continue
                    df = data[[var1, var2]].dropna()
                    if len(df) < 50:
                        continue
                    max_lag = min(12, len(df) // 10)

                    key = f"{var1}_vs_{var2}"
                    res = {
                        "var1_causes_var2": {"significant": False, "best_lag": None, "p_value": 1.0},
                        "var2_causes_var1": {"significant": False, "best_lag": None, "p_value": 1.0},
                        "bidirectional": False,
                        "sample_size": int(len(df)),
                    }
                    try:
                        r12 = grangercausalitytests(df[[var2, var1]].values, maxlag=max_lag, verbose=False)
                        p12 = {lag: r12[lag][0]["ssr_ftest"][1] for lag in r12}
                        b12 = min(p12, key=p12.get); pv12 = float(p12[b12])
                        res["var1_causes_var2"] = {"significant": bool(pv12 < 0.05), "best_lag": int(b12), "p_value": pv12}
                    except Exception:
                        pass
                    try:
                        r21 = grangercausalitytests(df[[var1, var2]].values, maxlag=max_lag, verbose=False)
                        p21 = {lag: r21[lag][0]["ssr_ftest"][1] for lag in r21}
                        b21 = min(p21, key=p21.get); pv21 = float(p21[b21])
                        res["var2_causes_var1"] = {"significant": bool(pv21 < 0.05), "best_lag": int(b21), "p_value": pv21}
                    except Exception:
                        pass
                    res["bidirectional"] = bool(res["var1_causes_var2"]["significant"] and res["var2_causes_var1"]["significant"])
                    out[key] = res
        except ImportError:
            print("statsmodels not available for Granger causality tests")
        except Exception as e:
            print(f"Granger causality failed: {e}")
        return out
