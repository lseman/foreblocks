import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, periodogram, welch
from scipy.stats import linregress

from .foreminer_aux import *

# Optional dependencies
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Fast CUSUM implementation
if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _cusum_stat_jit(x):
        n = x.shape[0]
        if n < 3:
            return 0.0
        mu = np.mean(x)
        s = np.cumsum(x - mu)
        max_cusum = np.max(np.abs(s))
        std = np.std(x)
        return max_cusum / (std * np.sqrt(n)) if std > 0 else 0.0
else:
    def _cusum_stat_jit(x):
        x = np.asarray(x)
        if len(x) < 3:
            return 0.0
        mu = x.mean()
        s = np.cumsum(x - mu)
        max_cusum = np.max(np.abs(s))
        std = x.std()
        return float(max_cusum / (std * np.sqrt(len(x)))) if std > 0 else 0.0


class TimeSeriesAnalyzer(AnalysisStrategy):
    """Modern time series analysis with streamlined SOTA methods"""

    @property
    def name(self) -> str:
        return "timeseries"

    def __init__(self):
        self.min_series_length = 10
        self.max_series_count = 20  # Limit for performance

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Main analysis pipeline using modern time series methods"""
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove time/target columns
        time_col = getattr(config, "time_col", None)
        target_col = getattr(config, "target", None)
        for col in [time_col, target_col]:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        # Prepare clean series (remove short/invalid series)
        series_map = {}
        for col in numeric_cols[:self.max_series_count]:  # Limit for performance
            clean_series = data[col].dropna()
            if len(clean_series) >= self.min_series_length:
                series_map[col] = clean_series
        
        if not series_map:
            return {"error": "No valid time series found"}
        
        # Core analysis modules (streamlined)
        results = {
            "stationarity": self._analyze_stationarity(series_map, config),
            "temporal_patterns": self._analyze_temporal_patterns(series_map, config),
            "lag_suggestions": self._suggest_optimal_lags(series_map, config),
            "seasonality_tests": self._test_seasonality(data, series_map, config),
            "change_point_detection": self._detect_change_points(data, series_map, config),
            "forecasting_readiness": self._assess_forecasting_readiness(series_map, config),
            "volatility_analysis": self._analyze_volatility(series_map, config),
            "causality_analysis": self._granger_causality_analysis(data, list(series_map.keys()), config)
        }
        
        return results

    # --------------------------- SOTA Stationarity Testing ---------------------------
    def _analyze_stationarity(self, series_map: Dict[str, pd.Series], config: AnalysisConfig) -> pd.DataFrame:
        """Modern stationarity analysis using most reliable tests"""
        from statsmodels.tsa.stattools import adfuller, kpss
        
        results = []
        for col, series in series_map.items():
            try:
                # ADF test (most common)
                adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series, autolag="AIC")
                
                # KPSS test (complementary)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression="c", nlags="auto")
                
                # Interpretation logic
                adf_stationary = adf_p < 0.05
                kpss_stationary = kpss_p > 0.05
                
                if adf_stationary and kpss_stationary:
                    stationarity_type = "stationary"
                    consensus = True
                elif not adf_stationary and not kpss_stationary:
                    stationarity_type = "non_stationary"
                    consensus = True
                else:
                    stationarity_type = "conflicting_evidence"
                    consensus = False
                
                # Test first difference if non-stationary
                diff_info = {}
                if not consensus or stationarity_type == "non_stationary":
                    try:
                        diff_series = series.diff().dropna()
                        if len(diff_series) > 10:
                            diff_adf_stat, diff_adf_p = adfuller(diff_series, autolag="AIC")[:2]
                            diff_info = {
                                "diff_adf_pvalue": float(diff_adf_p),
                                "diff_stationary": diff_adf_p < 0.05
                            }
                    except Exception:
                        pass
                
                results.append({
                    "feature": col,
                    "adf_statistic": float(adf_stat),
                    "adf_pvalue": float(adf_p),
                    "adf_critical_5pct": float(adf_crit.get("5%", np.nan)),
                    "kpss_statistic": float(kpss_stat),
                    "kpss_pvalue": float(kpss_p),
                    "kpss_critical_5pct": float(kpss_crit.get("5%", np.nan)),
                    "is_stationary_adf": adf_stationary,
                    "is_stationary_kpss": kpss_stationary,
                    "stationarity_type": stationarity_type,
                    "consensus_stationary": consensus and stationarity_type == "stationary",
                    "is_stationary": consensus and stationarity_type == "stationary",
                    **diff_info
                })
                
            except Exception as e:
                results.append({
                    "feature": col,
                    "adf_statistic": np.nan,
                    "adf_pvalue": np.nan,
                    "adf_critical_5pct": np.nan,
                    "kpss_statistic": np.nan,
                    "kpss_pvalue": np.nan,
                    "kpss_critical_5pct": np.nan,
                    "is_stationary_adf": False,
                    "is_stationary_kpss": False,
                    "stationarity_type": "error",
                    "consensus_stationary": False,
                    "is_stationary": False,
                    "error": str(e)
                })
        
        return pd.DataFrame(results)

    # --------------------------- SOTA Temporal Pattern Analysis ---------------------------
    def _analyze_temporal_patterns(self, series_map: Dict[str, pd.Series], config: AnalysisConfig) -> Dict[str, Any]:
        """Modern temporal pattern analysis with trend and seasonality detection"""
        results = {}
        
        for col, series in series_map.items():
            if len(series) < 20:
                continue
                
            try:
                pattern_info = {}
                
                # Trend analysis using linear regression
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = linregress(x, series.values)
                
                series_std = series.std()
                trend_strength = abs(slope * len(series)) / series_std if series_std > 0 else 0
                
                pattern_info[f"{col}_trend"] = {
                    "linear_slope": float(slope),
                    "linear_r_squared": float(r_value ** 2),
                    "linear_p_value": float(p_value),
                    "linear_significant": p_value < 0.05,
                    "trend_direction": self._classify_trend(slope, p_value),
                    "trend_strength": float(trend_strength),
                    "trend_strength_category": self._categorize_trend_strength(trend_strength)
                }
                
                # Seasonality analysis using STL decomposition
                if len(series) >= 24:
                    seasonality_info = self._analyze_seasonality_stl(series)
                    pattern_info[f"{col}_seasonality"] = seasonality_info
                
                # Volatility clustering (for financial/economic data)
                if len(series) >= 30:
                    returns = series.pct_change().dropna()
                    if len(returns) >= 20:
                        vol_info = self._analyze_volatility_clustering(returns)
                        pattern_info[f"{col}_volatility"] = vol_info
                
                # Structural breaks using CUSUM
                cusum_stat = _cusum_stat_jit(series.values)
                cusum_breakpoint = None
                if cusum_stat > 1.5:  # Significant break threshold
                    cumulative_sum = np.cumsum(series - series.mean())
                    cusum_breakpoint = float(np.argmax(np.abs(cumulative_sum)) / len(series))
                
                pattern_info[f"{col}_structural_breaks"] = {
                    "cusum_statistic": float(cusum_stat),
                    "potential_break_point": cusum_breakpoint,
                    "break_significant": cusum_stat > 1.5
                }
                
                results.update(pattern_info)
                
            except Exception as e:
                results[f"{col}_error"] = str(e)
        
        return results

    def _classify_trend(self, slope: float, p_value: float) -> str:
        """Classify trend direction and significance"""
        if p_value > 0.05:
            return "no_trend"
        elif slope > 0:
            return "increasing"
        elif slope < 0:
            return "decreasing"
        else:
            return "stable"

    def _categorize_trend_strength(self, strength: float) -> str:
        """Categorize trend strength"""
        if strength > 2.0:
            return "strong"
        elif strength > 1.0:
            return "moderate"
        elif strength > 0.5:
            return "weak"
        else:
            return "none"

    def _analyze_seasonality_stl(self, series: pd.Series) -> Dict[str, Any]:
        """STL decomposition for seasonality analysis"""
        try:
            from statsmodels.tsa.seasonal import STL

            # Estimate period
            period = self._estimate_period(series)
            
            # STL decomposition
            seasonal_param = max(7, min(period, len(series) // 6))
            if seasonal_param % 2 == 0:
                seasonal_param += 1
                
            stl = STL(series, seasonal=seasonal_param, period=period, robust=True).fit()
            
            # Calculate seasonal strength
            total_var = float(np.var(series))
            seasonal_var = float(np.var(stl.seasonal))
            trend_var = float(np.var(stl.trend.dropna()))
            
            seasonal_strength = seasonal_var / total_var if total_var > 0 else 0
            trend_strength = trend_var / total_var if total_var > 0 else 0
            
            return {
                "stl_seasonal_strength": float(seasonal_strength),
                "stl_trend_strength": float(trend_strength),
                "stl_period": int(period),
                "stl_classification": self._classify_seasonality(seasonal_strength),
                "seasonal_component_detected": seasonal_strength > 0.1
            }
            
        except Exception as e:
            return {"stl_error": str(e)}

    def _estimate_period(self, series: pd.Series) -> int:
        """Estimate dominant period using autocorrelation"""
        try:
            from statsmodels.tsa.stattools import acf
            
            n = len(series)
            max_lags = min(n // 3, 100)
            
            if max_lags < 3:
                return max(2, n // 4)
            
            # Calculate autocorrelation
            autocorr = acf(series, nlags=max_lags, fft=True)
            
            # Find peaks in autocorrelation
            peaks, _ = find_peaks(autocorr[1:], height=0.1)
            
            if len(peaks) > 0:
                # Return the most prominent peak
                dominant_peak = peaks[np.argmax(autocorr[peaks + 1])] + 1
                return max(2, min(dominant_peak, n // 4))
            
            # Default periods based on series length
            if n >= 365:
                return 365  # Daily data, yearly seasonality
            elif n >= 52:
                return 52   # Weekly data, yearly seasonality
            elif n >= 24:
                return 12   # Monthly data, yearly seasonality
            else:
                return max(2, n // 4)
                
        except Exception:
            n = len(series)
            return max(2, min(12, n // 4))

    def _classify_seasonality(self, strength: float) -> str:
        """Classify seasonal strength"""
        if strength > 0.4:
            return "strong"
        elif strength > 0.2:
            return "moderate"
        elif strength > 0.1:
            return "weak"
        else:
            return "none"

    def _analyze_volatility_clustering(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze volatility clustering in returns"""
        try:
            vol_info = {
                "returns_volatility": float(returns.std()),
                "returns_skewness": float(returns.skew()),
                "returns_kurtosis": float(returns.kurtosis())
            }
            
            # Rolling volatility
            window = min(20, max(5, len(returns) // 10))
            rolling_vol = returns.rolling(window).std()
            
            if len(rolling_vol.dropna()) > 10:
                # Volatility of volatility
                vol_of_vol = rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() > 0 else 0
                
                # Volatility persistence (autocorrelation)
                vol_autocorr = rolling_vol.dropna().autocorr(lag=1)
                
                vol_info.update({
                    "rolling_vol_mean": float(rolling_vol.mean()),
                    "volatility_of_volatility": float(vol_of_vol),
                    "vol_autocorr": float(vol_autocorr) if not np.isnan(vol_autocorr) else 0.0,
                    "volatility_persistent": abs(vol_autocorr) > 0.3 if not np.isnan(vol_autocorr) else False
                })
            
            return vol_info
            
        except Exception as e:
            return {"volatility_error": str(e)}

    # --------------------------- SOTA Lag Analysis ---------------------------
    def _suggest_optimal_lags(self, series_map: Dict[str, pd.Series], config: AnalysisConfig) -> Dict[str, Dict[str, Any]]:
        """Modern lag selection using PACF and information criteria"""
        from statsmodels.tsa.ar_model import AutoReg
        from statsmodels.tsa.stattools import pacf
        
        results = {}
        
        for col, series in series_map.items():
            n = len(series)
            max_lags = min(24, n // 10)
            
            if n < max_lags + 20:
                continue
                
            try:
                # PACF-based lag selection
                pacf_values = pacf(series, nlags=max_lags, method="ols")
                
                # Significance threshold
                confidence_interval = 1.96 / np.sqrt(n)
                
                # Find significant lags
                significant_lags = []
                for lag in range(1, len(pacf_values)):
                    if abs(pacf_values[lag]) > confidence_interval:
                        score = abs(pacf_values[lag])
                        
                        # Boost seasonal lags
                        if lag in [12, 24, 52] and lag < n // 3:
                            score *= 1.2
                            
                        significant_lags.append((lag, score))
                
                # Sort by significance and take top 5
                significant_lags.sort(key=lambda x: x[1], reverse=True)
                recommended_lags = [lag for lag, _ in significant_lags[:5]]
                
                # AR model order selection using AIC
                optimal_ar_order = None
                if n >= 40:
                    max_ar_order = min(10, n // 10)
                    aic_scores = []
                    
                    for p in range(1, max_ar_order + 1):
                        try:
                            model = AutoReg(series, lags=p, old_names=False).fit()
                            aic_scores.append((p, model.aic))
                        except Exception:
                            continue
                    
                    if aic_scores:
                        optimal_ar_order = min(aic_scores, key=lambda x: x[1])[0]
                
                results[col] = {
                    "autocorr_lags": recommended_lags,
                    "ar_optimal_order": optimal_ar_order,
                    "recommended_lags": recommended_lags[:3],
                    "lag_selection_method": "pacf_with_ic",
                    "seasonal_lags": [lag for lag in recommended_lags if lag in [12, 24, 52]]
                }
                
            except Exception as e:
                results[col] = {"error": str(e)}
        
        return results

    # --------------------------- SOTA Seasonality Testing ---------------------------
    def _test_seasonality(self, data: pd.DataFrame, series_map: Dict[str, pd.Series], config: AnalysisConfig) -> Dict[str, Any]:
        """Modern seasonality testing using statistical tests and spectral analysis"""
        from scipy.stats import kruskal
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        time_col = getattr(config, "time_col", None)
        if not time_col or time_col not in data.columns:
            return {}
        
        results = {}
        
        for col in list(series_map.keys())[:10]:  # Limit for performance
            try:
                # Prepare time-indexed data
                df = data[[time_col, col]].dropna()
                if len(df) < 24:
                    continue
                    
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col).sort_index()
                series = df[col]
                
                col_results = {}
                
                # Kruskal-Wallis test for day-of-week effects
                try:
                    dow_groups = [series[series.index.dayofweek == d].values for d in range(7)]
                    dow_groups = [group for group in dow_groups if len(group) >= 5]
                    
                    if len(dow_groups) >= 5:
                        stat, p_value = kruskal(*dow_groups)
                        col_results["kruskal_dow"] = {
                            "statistic": float(stat),
                            "p_value": float(p_value),
                            "dow_effect_significant": p_value < 0.05
                        }
                except Exception:
                    pass
                
                # Ljung-Box test for seasonal autocorrelation
                for seasonal_lag in [12, 24, 52]:
                    if len(series) > seasonal_lag + 10:
                        try:
                            lb_result = acorr_ljungbox(series, lags=[seasonal_lag], return_df=True)
                            col_results[f"ljungbox_lag_{seasonal_lag}"] = {
                                "statistic": float(lb_result["lb_stat"].iloc[0]),
                                "p_value": float(lb_result["lb_pvalue"].iloc[0]),
                                "seasonal_correlation": float(lb_result["lb_pvalue"].iloc[0]) < 0.05
                            }
                        except Exception:
                            pass
                
                # Spectral analysis for periodic components
                try:
                    frequencies, power = welch(series.values, nperseg=min(len(series)//4, 256))
                    
                    # Find significant peaks
                    threshold = np.percentile(power, 90)
                    peak_indices = find_peaks(power, height=threshold)[0]
                    
                    dominant_periods = []
                    for idx in peak_indices[:5]:
                        if frequencies[idx] > 0:
                            period = 1.0 / frequencies[idx]
                            dominant_periods.append({
                                "period": float(period),
                                "frequency": float(frequencies[idx]),
                                "power": float(power[idx])
                            })
                    
                    dominant_periods.sort(key=lambda x: x["power"], reverse=True)
                    col_results["spectral_analysis"] = {
                        "dominant_periods": dominant_periods
                    }
                except Exception:
                    pass
                
                if col_results:
                    results[col] = col_results
                    
            except Exception as e:
                results[f"{col}_error"] = str(e)
        
        return results

    # --------------------------- SOTA Change Point Detection ---------------------------
    def _detect_change_points(self, data: pd.DataFrame, series_map: Dict[str, pd.Series], config: AnalysisConfig) -> Dict[str, Any]:
        """Modern change point detection using ruptures and CUSUM"""
        time_col = getattr(config, "time_col", None)
        if not time_col or time_col not in data.columns:
            return {}
        
        results = {}
        
        for col in list(series_map.keys())[:10]:  # Limit for performance
            try:
                # Prepare time-indexed data
                df = data[[time_col, col]].dropna()
                if len(df) < 50:
                    continue
                    
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col).sort_index()
                series = df[col]
                
                col_results = {}
                
                # Ruptures library (if available) - most modern approach
                if HAS_RUPTURES and len(series) >= 80:
                    try:
                        # Use Binary Segmentation with L2 cost
                        algo = rpt.Binseg(model="l2").fit(series.values)
                        n_bkps = min(3, max(1, len(series) // 100))
                        breakpoints = algo.predict(n_bkps=n_bkps)
                        
                        change_points = []
                        for bp in breakpoints[:-1]:  # Exclude the last point
                            if 10 <= bp < len(series) - 10:  # Ensure not at edges
                                pre_mean = series.iloc[:bp].mean()
                                post_mean = series.iloc[bp:].mean()
                                magnitude = abs(post_mean - pre_mean)
                                
                                change_points.append({
                                    "index": int(bp),
                                    "timestamp": series.index[bp],
                                    "magnitude": float(magnitude),
                                    "pre_change_mean": float(pre_mean),
                                    "post_change_mean": float(post_mean)
                                })
                        
                        if change_points:
                            col_results["ruptures_binseg"] = {
                                "change_points": change_points,
                                "method": "Binary Segmentation (L2)"
                            }
                    except Exception:
                        pass
                
                # CUSUM test (always available fallback)
                try:
                    cusum_stat = _cusum_stat_jit(series.values)
                    
                    if cusum_stat > 1.5:  # Significant change point
                        cumulative_sum = np.cumsum(series - series.mean())
                        bp_index = np.argmax(np.abs(cumulative_sum))
                        
                        if 5 < bp_index < len(series) - 5:
                            col_results["cusum"] = {
                                "change_point": series.index[bp_index],
                                "statistic": float(cusum_stat),
                                "pre_change_mean": float(series.iloc[:bp_index].mean()),
                                "post_change_mean": float(series.iloc[bp_index:].mean()),
                                "change_significant": True
                            }
                except Exception:
                    pass
                
                if col_results:
                    results[col] = col_results
                    
            except Exception as e:
                results[f"{col}_error"] = str(e)
        
        return results

    # --------------------------- SOTA Forecasting Readiness ---------------------------
    def _assess_forecasting_readiness(self, series_map: Dict[str, pd.Series], config: AnalysisConfig) -> Dict[str, Any]:
        """Modern forecasting readiness assessment"""
        results = {}
        
        for col, series in series_map.items():
            if len(series) < 20:
                continue
                
            try:
                scores = {}
                n = len(series)
                
                # Data sufficiency score
                scores["data_sufficiency"] = min(n / 100.0, 1.0)
                
                # Stationarity score
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_p = adfuller(series, autolag="AIC")[1]
                    scores["stationarity"] = 1.0 if adf_p < 0.05 else 0.3
                except Exception:
                    scores["stationarity"] = 0.5
                
                # Trend detection score
                x = np.arange(n)
                slope, _, r_value, p_value, _ = linregress(x, series.values)
                trend_strength = abs(r_value) if p_value < 0.05 else 0
                scores["trend_strength"] = min(trend_strength * 2, 1.0)
                
                # Seasonality score
                try:
                    seasonality_score = 0.0
                    if n >= 24:
                        from statsmodels.tsa.seasonal import STL
                        period = self._estimate_period(series)
                        seasonal_param = max(7, min(period, n // 6))
                        if seasonal_param % 2 == 0:
                            seasonal_param += 1
                            
                        stl = STL(series, seasonal=seasonal_param, period=period, robust=True).fit()
                        seasonal_var = np.var(stl.seasonal)
                        total_var = np.var(series)
                        seasonality_score = min((seasonal_var / total_var) * 2, 1.0) if total_var > 0 else 0
                    
                    scores["seasonality"] = seasonality_score
                except Exception:
                    scores["seasonality"] = 0.0
                
                # Noise level score (signal-to-noise ratio)
                returns = series.pct_change().dropna()
                noise_level = returns.std() if len(returns) > 1 else 0
                scores["signal_to_noise"] = max(0.0, 1.0 - min(noise_level, 1.0))
                
                # Autocorrelation score
                try:
                    from statsmodels.tsa.stattools import acf
                    autocorr = acf(series, nlags=min(20, n // 2), fft=True)
                    significant_lags = np.sum(np.abs(autocorr[1:]) > 1.96 / np.sqrt(n))
                    scores["autocorrelation"] = min(significant_lags / 10.0, 1.0)
                except Exception:
                    scores["autocorrelation"] = 0.5
                
                # Overall readiness score
                weights = {
                    "data_sufficiency": 0.25,
                    "stationarity": 0.20,
                    "trend_strength": 0.15,
                    "seasonality": 0.15,
                    "signal_to_noise": 0.15,
                    "autocorrelation": 0.10
                }
                
                overall_score = sum(scores[key] * weights[key] for key in scores)
                
                # Readiness level classification
                if overall_score >= 0.8:
                    readiness_level = "excellent"
                elif overall_score >= 0.6:
                    readiness_level = "good"
                elif overall_score >= 0.4:
                    readiness_level = "fair"
                else:
                    readiness_level = "poor"
                
                # Generate recommendations
                recommendations = []
                if scores["data_sufficiency"] < 0.5:
                    recommendations.append("Collect more historical data")
                if scores["stationarity"] < 0.6:
                    recommendations.append("Consider differencing or detrending")
                if scores["signal_to_noise"] < 0.5:
                    recommendations.append("Apply smoothing or outlier removal")
                if scores["autocorrelation"] < 0.3:
                    recommendations.append("Series may be too random for forecasting")
                
                results[col] = {
                    "overall_score": float(overall_score),
                    "readiness_level": readiness_level,
                    "component_scores": {k: float(v) for k, v in scores.items()},
                    "recommendations": recommendations
                }
                
            except Exception as e:
                results[col] = {"error": str(e)}
        
        return results

    # --------------------------- SOTA Volatility Analysis ---------------------------
    def _analyze_volatility(self, series_map: Dict[str, pd.Series], config: AnalysisConfig) -> Dict[str, Any]:
        """Modern volatility analysis with GARCH effects and risk measures"""
        results = {}
        
        for col, series in series_map.items():
            if len(series) < 50:
                continue
                
            try:
                returns = series.pct_change().dropna()
                if len(returns) < 20:
                    continue
                
                vol_results = {}
                
                # Basic volatility measures
                basic_measures = {
                    "returns_mean": float(returns.mean()),
                    "returns_std": float(returns.std()),
                    "annualized_volatility": float(returns.std() * np.sqrt(252)),
                    "skewness": float(returns.skew()),
                    "kurtosis": float(returns.kurtosis()),
                    "sharpe_ratio": float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0
                }
                vol_results["basic_measures"] = basic_measures
                
                # ARCH test for volatility clustering
                try:
                    from statsmodels.stats.diagnostic import het_arch
                    arch_stat, arch_p = het_arch(returns, nlags=min(5, len(returns)//20))[:2]
                    vol_results["arch_test"] = {
                        "statistic": float(arch_stat),
                        "p_value": float(arch_p),
                        "volatility_clustering": arch_p < 0.05
                    }
                except Exception:
                    pass
                
                # Rolling volatility analysis
                window = min(20, max(5, len(returns)//10))
                rolling_vol = returns.rolling(window).std()
                
                if len(rolling_vol.dropna()) > 10:
                    vol_results["rolling_volatility"] = {
                        "mean": float(rolling_vol.mean()),
                        "std": float(rolling_vol.std()),
                        "volatility_of_volatility": float(rolling_vol.std() / rolling_vol.mean()) if rolling_vol.mean() > 0 else 0.0
                    }
                
                # Risk measures
                var_95 = float(np.percentile(returns, 5))
                var_99 = float(np.percentile(returns, 1))
                
                # Expected Shortfall (Conditional VaR)
                es_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else np.nan
                es_99 = float(returns[returns <= var_99].mean()) if (returns <= var_99).any() else np.nan
                
                # Maximum Drawdown
                cumulative_returns = (1 + returns).cumprod()
                peak = cumulative_returns.cummax()
                drawdown = (cumulative_returns - peak) / peak
                max_drawdown = float(drawdown.min())
                
                vol_results["risk_measures"] = {
                    "var_95": var_95,
                    "var_99": var_99,
                    "expected_shortfall_95": es_95,
                    "expected_shortfall_99": es_99,
                    "max_drawdown": max_drawdown
                }
                
                results[col] = vol_results
                
            except Exception as e:
                results[col] = {"error": str(e)}
        
        return results

    # --------------------------- SOTA Granger Causality ---------------------------
    def _granger_causality_analysis(self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig) -> Dict[str, Any]:
        """Modern Granger causality analysis between time series"""
        if len(numeric_cols) < 2:
            return {}
        
        results = {}
        
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # Limit pairs for performance
            cols_subset = numeric_cols[:5]
            
            for i in range(len(cols_subset)):
                for j in range(i + 1, len(cols_subset)):
                    var1, var2 = cols_subset[i], cols_subset[j]
                    
                    if var1 not in data.columns or var2 not in data.columns:
                        continue
                    
                    # Prepare clean data
                    df = data[[var1, var2]].dropna()
                    if len(df) < 50:
                        continue
                    
                    max_lag = min(12, len(df) // 10)
                    pair_key = f"{var1}_vs_{var2}"
                    
                    pair_results = {
                        "var1_causes_var2": {"significant": False, "best_lag": None, "p_value": 1.0},
                        "var2_causes_var1": {"significant": False, "best_lag": None, "p_value": 1.0},
                        "bidirectional": False,
                        "sample_size": len(df)
                    }
                    
                    try:
                        # Test var1 -> var2
                        gc_result_12 = grangercausalitytests(
                            df[[var2, var1]].values, 
                            maxlag=max_lag, 
                            verbose=False
                        )
                        
                        # Extract best lag and p-value
                        p_values_12 = {lag: gc_result_12[lag][0]["ssr_ftest"][1] for lag in gc_result_12}
                        best_lag_12 = min(p_values_12, key=p_values_12.get)
                        best_p_12 = p_values_12[best_lag_12]
                        
                        pair_results["var1_causes_var2"] = {
                            "significant": best_p_12 < 0.05,
                            "best_lag": int(best_lag_12),
                            "p_value": float(best_p_12)
                        }
                        
                    except Exception:
                        pass
                    
                    try:
                        # Test var2 -> var1
                        gc_result_21 = grangercausalitytests(
                            df[[var1, var2]].values, 
                            maxlag=max_lag, 
                            verbose=False
                        )
                        
                        # Extract best lag and p-value
                        p_values_21 = {lag: gc_result_21[lag][0]["ssr_ftest"][1] for lag in gc_result_21}
                        best_lag_21 = min(p_values_21, key=p_values_21.get)
                        best_p_21 = p_values_21[best_lag_21]
                        
                        pair_results["var2_causes_var1"] = {
                            "significant": best_p_21 < 0.05,
                            "best_lag": int(best_lag_21),
                            "p_value": float(best_p_21)
                        }
                        
                    except Exception:
                        pass
                    
                    # Check for bidirectional causality
                    pair_results["bidirectional"] = (
                        pair_results["var1_causes_var2"]["significant"] and 
                        pair_results["var2_causes_var1"]["significant"]
                    )
                    
                    results[pair_key] = pair_results
                    
        except ImportError:
            results["error"] = "statsmodels not available for Granger causality tests"
        except Exception as e:
            results["error"] = str(e)
        
        return results
