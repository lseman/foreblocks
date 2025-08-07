
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .foreminer_aux import *


class TimeSeriesAnalyzer(AnalysisStrategy):
    """SOTA comprehensive time series analysis with advanced ML techniques"""

    @property
    def name(self) -> str:
        return "timeseries"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove time and target columns from analysis
        time_col = getattr(config, "time_col", None)
        target_col = getattr(config, "target", None)
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        results = {
            "stationarity": self._analyze_stationarity(data, numeric_cols, config),
            "temporal_patterns": self._analyze_temporal_patterns(
                data, numeric_cols, config
            ),
            "lag_suggestions": self._suggest_lag_features(data, numeric_cols, config),
            "seasonality_tests": self._advanced_seasonality_tests(
                data, numeric_cols, config
            ),
            "change_point_detection": self._detect_change_points(
                data, numeric_cols, config
            ),
            "regime_switching": self._detect_regime_switching(
                data, numeric_cols, config
            ),
            "forecasting_readiness": self._assess_forecasting_readiness(
                data, numeric_cols, config
            ),
            "volatility_analysis": self._analyze_volatility(data, numeric_cols, config),
            "cyclical_patterns": self._detect_cyclical_patterns(
                data, numeric_cols, config
            ),
            "causality_analysis": self._granger_causality_analysis(
                data, numeric_cols, config
            ),
        }

        return results

    def _analyze_stationarity(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> pd.DataFrame:
        """Enhanced stationarity testing with multiple tests and differencing suggestions"""
        from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

        results = []

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 20:
                continue

            try:
                # ADF test (null: unit root present, non-stationary)
                adf_result = adfuller(col_data, autolag="AIC")

                # KPSS test (null: stationary)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    kpss_result = kpss(col_data, regression="c", nlags="auto")

                # Zivot-Andrews test (structural break unit root test)
                za_stat = za_pvalue = None
                try:
                    if len(col_data) > 50:  # Minimum required for ZA test
                        za_result = zivot_andrews(col_data, model="c")
                        za_stat, za_pvalue = za_result[0], za_result[1]
                except:
                    pass

                # Variance ratio test for random walk
                variance_ratio = self._variance_ratio_test(col_data)

                # Determine stationarity consensus
                adf_stationary = adf_result[1] < config.confidence_level
                kpss_stationary = kpss_result[1] > config.confidence_level

                # Enhanced stationarity classification
                if adf_stationary and kpss_stationary:
                    stationarity_type = "stationary"
                elif not adf_stationary and not kpss_stationary:
                    stationarity_type = "non_stationary"
                elif adf_stationary and not kpss_stationary:
                    stationarity_type = "trend_stationary"
                else:
                    stationarity_type = "difference_stationary"

                # Test first difference if non-stationary
                diff_results = {}
                if not (adf_stationary and kpss_stationary):
                    diff_series = col_data.diff().dropna()
                    if len(diff_series) > 10:
                        try:
                            diff_adf = adfuller(diff_series, autolag="AIC")
                            diff_results = {
                                "diff_adf_stat": diff_adf[0],
                                "diff_adf_pvalue": diff_adf[1],
                                "diff_stationary": diff_adf[1]
                                < config.confidence_level,
                            }
                        except:
                            pass

                # Seasonal differencing test
                seasonal_diff_results = {}
                if len(col_data) > 24:
                    try:
                        seasonal_diff = col_data.diff(
                            12
                        ).dropna()  # 12-period seasonal diff
                        if len(seasonal_diff) > 10:
                            seas_adf = adfuller(seasonal_diff, autolag="AIC")
                            seasonal_diff_results = {
                                "seasonal_diff_adf_stat": seas_adf[0],
                                "seasonal_diff_adf_pvalue": seas_adf[1],
                                "seasonal_diff_stationary": seas_adf[1]
                                < config.confidence_level,
                            }
                    except:
                        pass

                result_dict = {
                    "feature": col,
                    "adf_statistic": adf_result[0],
                    "adf_pvalue": adf_result[1],
                    "adf_critical_1pct": adf_result[4]["1%"],
                    "adf_critical_5pct": adf_result[4]["5%"],
                    "kpss_statistic": kpss_result[0],
                    "kpss_pvalue": kpss_result[1],
                    "kpss_critical_1pct": kpss_result[3]["1%"],
                    "kpss_critical_5pct": kpss_result[3]["5%"],
                    "za_statistic": za_stat,
                    "za_pvalue": za_pvalue,
                    "variance_ratio": variance_ratio,
                    "is_stationary_adf": adf_stationary,
                    "is_stationary_kpss": kpss_stationary,
                    "stationarity_type": stationarity_type,
                    "consensus_stationary": adf_stationary and kpss_stationary,
                    "is_stationary": adf_stationary and kpss_stationary,
                    **diff_results,
                    **seasonal_diff_results,
                }

                results.append(result_dict)

            except Exception as e:
                print(f"Enhanced stationarity test failed for {col}: {e}")

        return pd.DataFrame(results)

    def _variance_ratio_test(self, series: pd.Series, lags: int = 4) -> float:
        """Variance ratio test for random walk hypothesis"""
        try:
            n = len(series)
            if n < lags * 4:
                return np.nan

            # Calculate variance ratio
            returns = series.pct_change().dropna()
            var_1 = np.var(returns)

            # k-period variance
            k_returns = returns.rolling(window=lags).sum().dropna()
            var_k = np.var(k_returns) / lags

            variance_ratio = var_k / var_1 if var_1 > 0 else np.nan
            return variance_ratio

        except:
            return np.nan

    def _analyze_temporal_patterns(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Enhanced temporal pattern analysis with multiple decomposition methods"""
        from scipy.signal import periodogram
        from scipy.stats import linregress
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.x13 import x13_arima_analysis

        patterns = {}

        for col in numeric_cols[:8]:  # Limit for performance
            series = data[col].dropna()
            if len(series) < 24:
                continue

            try:
                # Enhanced trend analysis with multiple methods
                x = np.arange(len(series))

                # Linear trend
                slope, intercept, r_value, p_value, stderr = linregress(x, series)

                # Polynomial trend (degree 2)
                poly_coeffs = np.polyfit(x, series, deg=2)
                poly_trend = np.polyval(poly_coeffs, x)
                poly_r2 = np.corrcoef(series, poly_trend)[0, 1] ** 2

                # Hodrick-Prescott filter trend
                try:
                    from statsmodels.tsa.filters.hp_filter import hpfilter

                    hp_cycle, hp_trend = hpfilter(series, lamb=1600)
                    hp_trend_strength = np.var(hp_trend) / np.var(series)
                except:
                    hp_trend_strength = None

                trend_classification = self._classify_trend(
                    slope, p_value, series.std()
                )

                patterns[f"{col}_trend"] = {
                    "linear_slope": slope,
                    "linear_r_squared": r_value**2,
                    "linear_p_value": p_value,
                    "linear_significant": p_value < 0.05,
                    "trend_direction": trend_classification,
                    "slope_normalized": slope / series.std() if series.std() > 0 else 0,
                    "polynomial_r2": poly_r2,
                    "hp_trend_strength": hp_trend_strength,
                    "trend_strength_category": self._categorize_trend_strength(
                        abs(slope), series.std()
                    ),
                }

                # Advanced seasonality detection with multiple methods
                seasonality_results = {}

                # STL decomposition
                try:
                    # Auto-detect period or use default
                    period = self._estimate_period(series)
                    stl = STL(
                        series, seasonal=min(period, len(series) // 3), period=period
                    )
                    stl_decomp = stl.fit()

                    seasonal_var = np.var(stl_decomp.seasonal)
                    total_var = np.var(series)
                    seasonal_strength = seasonal_var / total_var if total_var > 0 else 0

                    seasonality_results.update(
                        {
                            "stl_seasonal_strength": seasonal_strength,
                            "stl_period": period,
                            "stl_classification": self._classify_seasonality(
                                seasonal_strength
                            ),
                            "trend_strength": (
                                np.var(stl_decomp.trend.dropna()) / total_var
                                if total_var > 0
                                else 0
                            ),
                        }
                    )

                except Exception as e:
                    seasonality_results["stl_error"] = str(e)

                # X-13ARIMA-SEATS decomposition (if available)
                try:
                    if len(series) >= 36:  # Minimum for X-13
                        x13_arima_analysis(series)
                        seasonality_results["x13_seasonal_strength"] = "computed"
                except:
                    pass

                # Periodogram analysis for dominant frequencies
                try:
                    freqs, psd = periodogram(series, scaling="density")
                    dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
                    dominant_period = (
                        1 / freqs[dominant_freq_idx]
                        if freqs[dominant_freq_idx] > 0
                        else None
                    )
                    seasonality_results.update(
                        {
                            "dominant_period_periodogram": dominant_period,
                            "spectral_peak_power": psd[dominant_freq_idx],
                        }
                    )
                except:
                    pass

                patterns[f"{col}_seasonality"] = seasonality_results

                # Enhanced volatility clustering analysis
                volatility_results = {}
                returns = series.pct_change().dropna()
                if len(returns) > 20:
                    # Multiple volatility measures
                    rolling_vol = returns.rolling(
                        window=min(10, len(returns) // 3)
                    ).std()

                    # ARCH effects test
                    returns**2
                    arch_lm_stat = self._arch_lm_test(returns)

                    # Volatility persistence
                    vol_autocorr = (
                        rolling_vol.dropna().autocorr(lag=1)
                        if len(rolling_vol.dropna()) > 1
                        else 0
                    )

                    # GARCH-like volatility clustering
                    high_vol_periods = (
                        (rolling_vol > rolling_vol.quantile(0.8)).sum()
                        if len(rolling_vol.dropna()) > 0
                        else 0
                    )
                    vol_clustering_score = (
                        high_vol_periods / len(rolling_vol.dropna())
                        if len(rolling_vol.dropna()) > 0
                        else 0
                    )

                    volatility_results = {
                        "returns_volatility": returns.std(),
                        "rolling_vol_mean": rolling_vol.mean(),
                        "vol_autocorr": vol_autocorr,
                        "arch_lm_statistic": arch_lm_stat,
                        "vol_clustering_score": vol_clustering_score,
                        "volatility_persistent": (
                            abs(vol_autocorr) > 0.3
                            if not np.isnan(vol_autocorr)
                            else False
                        ),
                    }

                patterns[f"{col}_volatility"] = volatility_results

                # Structural break detection
                break_results = {}
                if len(series) > 50:
                    try:
                        # Simple CUSUM test for structural breaks
                        cumsum = np.cumsum(series - series.mean())
                        max_cusum = np.max(np.abs(cumsum))
                        cusum_stat = max_cusum / (series.std() * np.sqrt(len(series)))

                        # Find potential break points
                        break_point_idx = np.argmax(np.abs(cumsum))
                        break_point_pct = break_point_idx / len(series)

                        break_results = {
                            "cusum_statistic": cusum_stat,
                            "potential_break_point": break_point_pct,
                            "break_significant": cusum_stat > 1.5,  # Rough threshold
                            "pre_break_mean": (
                                series.iloc[:break_point_idx].mean()
                                if break_point_idx > 10
                                else None
                            ),
                            "post_break_mean": (
                                series.iloc[break_point_idx:].mean()
                                if len(series) - break_point_idx > 10
                                else None
                            ),
                        }
                    except:
                        pass

                patterns[f"{col}_structural_breaks"] = break_results

            except Exception as e:
                print(f"Enhanced temporal pattern analysis failed for {col}: {e}")

        return patterns

    def _classify_trend(self, slope: float, p_value: float, std: float) -> str:
        """Classify trend direction and strength"""
        if p_value > 0.05:
            return "no_trend"

        threshold = std * 0.01  # 1% of standard deviation
        if slope > threshold:
            return "increasing" if slope > 2 * threshold else "weakly_increasing"
        elif slope < -threshold:
            return "decreasing" if slope < -2 * threshold else "weakly_decreasing"
        else:
            return "stable"

    def _classify_seasonality(self, strength: float) -> str:
        """Classify seasonality strength"""
        if strength > 0.4:
            return "strong"
        elif strength > 0.2:
            return "moderate"
        elif strength > 0.05:
            return "weak"
        else:
            return "none"

    def _categorize_trend_strength(self, abs_slope: float, std: float) -> str:
        """Categorize trend strength"""
        normalized_slope = abs_slope / (std + 1e-8)
        if normalized_slope > 0.1:
            return "strong"
        elif normalized_slope > 0.05:
            return "moderate"
        elif normalized_slope > 0.01:
            return "weak"
        else:
            return "none"

    def _estimate_period(self, series: pd.Series) -> int:
        """Estimate dominant period using autocorrelation"""
        from statsmodels.tsa.stattools import acf

        try:
            max_lags = min(len(series) // 3, 100)
            acf_vals = acf(series, nlags=max_lags, fft=True)

            # Find peaks in ACF
            peaks, _ = find_peaks(acf_vals[1:], height=0.1)
            if len(peaks) > 0:
                dominant_lag = peaks[np.argmax(acf_vals[peaks + 1])] + 1
                return max(2, min(dominant_lag, 52))  # Reasonable bounds
            else:
                # Default periods based on series length
                if len(series) >= 365:
                    return 365  # Daily data -> yearly seasonality
                elif len(series) >= 52:
                    return 52  # Weekly data -> yearly seasonality
                elif len(series) >= 12:
                    return 12  # Monthly data -> yearly seasonality
                else:
                    return max(2, len(series) // 4)
        except:
            return max(2, min(12, len(series) // 4))

    def _arch_lm_test(self, returns: pd.Series, lags: int = 5) -> float:
        """ARCH LM test for volatility clustering"""
        try:
            from statsmodels.stats.diagnostic import het_arch

            if len(returns) > lags + 10:
                lm_stat, p_val, _, _ = het_arch(returns, nlags=lags)
                return p_val
        except:
            pass
        return np.nan

    def _suggest_lag_features(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Dict[str, Any]]:
        """Enhanced lag feature suggestions using multiple methods"""
        from statsmodels.tsa.ar_model import AutoReg
        from statsmodels.tsa.stattools import acf, ccf, pacf

        suggestions = {}
        max_lags = min(24, len(data) // 10)  # Adaptive max lags

        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) < max_lags + 20:
                continue

            try:
                # Enhanced autocorrelation analysis
                pacf_vals = pacf(series, nlags=max_lags, method="ols")[1:]
                acf_vals = acf(series, nlags=max_lags, fft=True)[1:]
                n = len(series)

                # Dynamic confidence intervals
                pacf_ci = 1.96 / np.sqrt(n)
                acf_ci = 1.96 / np.sqrt(n)

                # Score lags using multiple criteria
                lag_scores = []
                for lag in range(1, max_lags + 1):
                    p_val = pacf_vals[lag - 1]
                    a_val = acf_vals[lag - 1]

                    # Multi-criteria scoring
                    significance_score = 0
                    if abs(p_val) > pacf_ci:
                        significance_score += 0.6 * abs(p_val)
                    if abs(a_val) > acf_ci:
                        significance_score += 0.4 * abs(a_val)

                    # Seasonal lag bonus (12, 24, etc.)
                    if lag in [12, 24, 52] and lag < len(series) // 3:
                        significance_score *= 1.2

                    # Penalize very long lags
                    if lag > len(series) // 5:
                        significance_score *= 0.8

                    if significance_score > 0.1:  # Threshold
                        lag_scores.append((lag, significance_score))

                # Auto-regression model order selection
                ar_order = None
                try:
                    # Use AIC to select optimal AR order
                    aic_scores = []
                    max_ar_order = min(10, len(series) // 10)
                    for order in range(1, max_ar_order + 1):
                        try:
                            ar_model = AutoReg(series, lags=order).fit()
                            aic_scores.append((order, ar_model.aic))
                        except:
                            continue

                    if aic_scores:
                        ar_order = min(aic_scores, key=lambda x: x[1])[0]

                except:
                    pass

                # Information criteria-based lag selection
                information_criteria = {}
                try:
                    from statsmodels.tsa.vector_ar.var_model import VAR

                    # Single variable VAR for lag selection
                    model_data = series.values.reshape(-1, 1)
                    var_model = VAR(model_data)
                    lag_order_results = var_model.select_order(
                        maxlags=min(12, len(series) // 10)
                    )
                    information_criteria = {
                        "aic_optimal": lag_order_results.aic,
                        "bic_optimal": lag_order_results.bic,
                        "hqic_optimal": lag_order_results.hqic,
                    }
                except:
                    pass

                # Cross-correlation with other features (if multivariate)
                ccf_lags = {}
                if len(numeric_cols) > 1:
                    for other_col in numeric_cols[:5]:  # Limit to avoid explosion
                        if other_col != col and other_col in data.columns:
                            try:
                                other_series = data[other_col].dropna()
                                # Align series
                                min_len = min(len(series), len(other_series))
                                if min_len > max_lags + 10:
                                    s1 = series.iloc[-min_len:]
                                    s2 = other_series.iloc[-min_len:]
                                    cross_corr = ccf(s1, s2, adjusted=False)

                                    # Find significant cross-correlations
                                    significant_ccf_lags = []
                                    cc_ci = 1.96 / np.sqrt(min_len)
                                    for lag_idx, cc_val in enumerate(cross_corr):
                                        if abs(cc_val) > cc_ci:
                                            actual_lag = lag_idx - len(cross_corr) // 2
                                            if actual_lag > 0:  # Only positive lags
                                                significant_ccf_lags.append(
                                                    (actual_lag, cc_val)
                                                )

                                    if significant_ccf_lags:
                                        ccf_lags[other_col] = significant_ccf_lags[
                                            :3
                                        ]  # Top 3
                            except:
                                pass

                # Sort and select top lags
                lag_scores.sort(key=lambda x: x[1], reverse=True)
                top_lags = [lag for lag, _ in lag_scores[:5]]  # Top 5 lags

                if top_lags or ar_order or information_criteria or ccf_lags:
                    suggestions[col] = {
                        "autocorr_lags": top_lags,
                        "ar_optimal_order": ar_order,
                        "information_criteria": information_criteria,
                        "cross_correlation_lags": ccf_lags,
                        "recommended_lags": (
                            top_lags[:3]
                            if top_lags
                            else ([ar_order] if ar_order else [])
                        ),
                        "lag_selection_method": "multi_criteria",
                        "seasonal_lags": [
                            lag for lag in top_lags if lag in [12, 24, 52]
                        ],
                    }

            except Exception as e:
                print(f"Enhanced lag suggestion failed for {col}: {e}")

        return suggestions

    def _advanced_seasonality_tests(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Advanced seasonality tests using multiple statistical methods"""
        from scipy.stats import friedmanchisquare, kruskal
        from statsmodels.stats.diagnostic import acorr_ljungbox

        seasonality_results = {}
        time_col = getattr(config, "time_col", None)

        if not time_col or time_col not in data.columns:
            return seasonality_results

        for col in numeric_cols[:5]:  # Limit for performance
            series_data = data[[time_col, col]].dropna()
            if len(series_data) < 24:
                continue

            try:
                # Ensure datetime index
                series_data[time_col] = pd.to_datetime(series_data[time_col])
                series_data.set_index(time_col, inplace=True)
                series = series_data[col]

                tests_results = {}

                # Friedman test for seasonal patterns
                if len(series) >= 36:  # At least 3 years of monthly data
                    try:
                        # Group by month
                        monthly_groups = []
                        for month in range(1, 13):
                            month_data = series[series.index.month == month]
                            if (
                                len(month_data) >= 3
                            ):  # At least 3 observations per month
                                monthly_groups.append(month_data.values)

                        if len(monthly_groups) >= 6:  # At least 6 months with data
                            # Make groups equal length by truncating to minimum
                            min_length = min(len(group) for group in monthly_groups)
                            equal_groups = [
                                group[:min_length] for group in monthly_groups
                            ]

                            if min_length >= 3:
                                stat, p_val = friedmanchisquare(*equal_groups)
                                tests_results["friedman_test"] = {
                                    "statistic": stat,
                                    "p_value": p_val,
                                    "seasonal_significant": p_val < 0.05,
                                }
                    except Exception as e:
                        tests_results["friedman_test"] = {"error": str(e)}

                # Kruskal-Wallis test for day-of-week effects
                try:
                    dow_groups = [
                        series[series.index.dayofweek == dow] for dow in range(7)
                    ]
                    dow_groups = [group for group in dow_groups if len(group) >= 5]

                    if len(dow_groups) >= 5:  # At least 5 days with sufficient data
                        stat, p_val = kruskal(*dow_groups)
                        tests_results["kruskal_dow"] = {
                            "statistic": stat,
                            "p_value": p_val,
                            "dow_effect_significant": p_val < 0.05,
                        }
                except Exception as e:
                    tests_results["kruskal_dow"] = {"error": str(e)}

                # Ljung-Box test for serial correlation at seasonal lags
                seasonal_lags = (
                    [12, 24, 52]
                    if len(series) > 104
                    else [12] if len(series) > 24 else []
                )
                for lag in seasonal_lags:
                    if len(series) > lag + 10:
                        try:
                            lb_result = acorr_ljungbox(
                                series, lags=[lag], return_df=True
                            )
                            tests_results[f"ljungbox_lag_{lag}"] = {
                                "statistic": lb_result["lb_stat"].iloc[0],
                                "p_value": lb_result["lb_pvalue"].iloc[0],
                                "seasonal_correlation": lb_result["lb_pvalue"].iloc[0]
                                < 0.05,
                            }
                        except:
                            pass

                # Spectral analysis for periodic components
                try:
                    from scipy.signal import welch

                    frequencies, psd = welch(series, nperseg=min(len(series) // 4, 256))

                    # Find dominant frequencies
                    peak_indices = find_peaks(psd, height=np.percentile(psd, 90))[0]
                    dominant_periods = []

                    for peak_idx in peak_indices[:5]:  # Top 5 peaks
                        freq = frequencies[peak_idx]
                        if freq > 0:
                            period = 1 / freq
                            power = psd[peak_idx]
                            dominant_periods.append(
                                {"period": period, "frequency": freq, "power": power}
                            )

                    tests_results["spectral_analysis"] = {
                        "dominant_periods": sorted(
                            dominant_periods, key=lambda x: x["power"], reverse=True
                        )
                    }
                except:
                    pass

                # QS (Quarterly Seasonal) test simulation
                try:
                    if len(series) >= 48:  # At least 4 years of quarterly data
                        quarterly_means = []
                        for quarter in range(1, 5):
                            q_data = series[series.index.quarter == quarter]
                            if len(q_data) >= 4:
                                quarterly_means.append(q_data.mean())

                        if len(quarterly_means) == 4:
                            # Simple seasonal variance test
                            series.mean()
                            seasonal_variance = np.var(quarterly_means)
                            total_variance = series.var()
                            seasonal_ratio = seasonal_variance / (total_variance + 1e-8)

                            tests_results["quarterly_seasonality"] = {
                                "seasonal_variance_ratio": seasonal_ratio,
                                "significant": seasonal_ratio
                                > 0.1,  # Heuristic threshold
                            }
                except:
                    pass

                seasonality_results[col] = tests_results

            except Exception as e:
                print(f"Advanced seasonality test failed for {col}: {e}")

        return seasonality_results

    def _detect_change_points(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Detect structural change points using multiple algorithms"""
        time_col = getattr(config, "time_col", None)
        if not time_col or time_col not in data.columns:
            return {}

        change_points = {}

        for col in numeric_cols[:5]:
            series_data = data[[time_col, col]].dropna()
            if len(series_data) < 50:
                continue

            try:
                # Prepare time series
                series_data[time_col] = pd.to_datetime(series_data[time_col])
                series_data.set_index(time_col, inplace=True)
                series = series_data[col]

                detected_changes = {}

                # CUSUM-based change point detection
                try:
                    mean_val = series.mean()
                    cumsum = np.cumsum(series - mean_val)

                    # Find maximum deviation points
                    abs_cumsum = np.abs(cumsum)
                    change_idx = np.argmax(abs_cumsum)
                    max_cusum = abs_cumsum[change_idx]

                    # Standardize CUSUM statistic
                    cusum_stat = max_cusum / (series.std() * np.sqrt(len(series)))

                    if cusum_stat > 1.5:  # Significant change threshold
                        change_timestamp = series.index[change_idx]
                        pre_mean = series.iloc[:change_idx].mean()
                        post_mean = series.iloc[change_idx:].mean()

                        detected_changes["cusum"] = {
                            "change_point": change_timestamp,
                            "statistic": cusum_stat,
                            "pre_change_mean": pre_mean,
                            "post_change_mean": post_mean,
                            "magnitude": abs(post_mean - pre_mean),
                            "significant": True,
                        }
                except:
                    pass

                # Bayesian change point detection (simplified)
                try:
                    # Rolling window variance change detection
                    window_size = max(10, len(series) // 20)
                    rolling_mean = series.rolling(window=window_size).mean()
                    rolling_var = series.rolling(window=window_size).var()

                    # Detect mean shifts
                    mean_changes = []
                    for i in range(window_size, len(rolling_mean) - window_size):
                        before_mean = rolling_mean.iloc[i - window_size : i].mean()
                        after_mean = rolling_mean.iloc[i : i + window_size].mean()

                        if not (pd.isna(before_mean) or pd.isna(after_mean)):
                            change_magnitude = abs(after_mean - before_mean)
                            if change_magnitude > series.std():
                                mean_changes.append(
                                    {
                                        "timestamp": series.index[i],
                                        "magnitude": change_magnitude,
                                        "before": before_mean,
                                        "after": after_mean,
                                    }
                                )

                    # Detect variance changes
                    variance_changes = []
                    for i in range(window_size, len(rolling_var) - window_size):
                        before_var = rolling_var.iloc[i - window_size : i].mean()
                        after_var = rolling_var.iloc[i : i + window_size].mean()

                        if (
                            not (pd.isna(before_var) or pd.isna(after_var))
                            and before_var > 0
                        ):
                            var_ratio = max(after_var, before_var) / min(
                                after_var, before_var
                            )
                            if var_ratio > 2.0:  # 100% variance change
                                variance_changes.append(
                                    {
                                        "timestamp": series.index[i],
                                        "ratio": var_ratio,
                                        "before_var": before_var,
                                        "after_var": after_var,
                                    }
                                )

                    if mean_changes or variance_changes:
                        detected_changes["rolling_window"] = {
                            "mean_changes": sorted(
                                mean_changes, key=lambda x: x["magnitude"], reverse=True
                            )[:3],
                            "variance_changes": sorted(
                                variance_changes, key=lambda x: x["ratio"], reverse=True
                            )[:3],
                        }

                except:
                    pass

                # Page-Hinkley test for online change detection
                try:
                    # Simplified Page-Hinkley implementation
                    delta = series.std() * 0.5  # Detection threshold
                    lambda_param = series.std() * 0.1  # Drift parameter

                    cumulative_sum = 0
                    min_sum = 0
                    change_points_ph = []

                    for i, value in enumerate(series):
                        cumulative_sum += value - series.mean() - lambda_param
                        min_sum = min(min_sum, cumulative_sum)

                        if cumulative_sum - min_sum > delta:
                            change_points_ph.append(
                                {
                                    "index": i,
                                    "timestamp": series.index[i],
                                    "cumulative_sum": cumulative_sum - min_sum,
                                }
                            )
                            cumulative_sum = 0
                            min_sum = 0

                    if change_points_ph:
                        detected_changes["page_hinkley"] = {
                            "change_points": change_points_ph[:5],  # Top 5
                            "threshold": delta,
                        }

                except:
                    pass

                # Multiple change points using binary segmentation approach
                try:

                    def detect_single_change(subseries):
                        """Detect single change point in subseries"""
                        n = len(subseries)
                        if n < 20:  # Minimum segment size
                            return None

                        best_change = None
                        best_stat = 0

                        for k in range(10, n - 10):  # Leave margins
                            left_mean = subseries.iloc[:k].mean()
                            right_mean = subseries.iloc[k:].mean()

                            # Calculate test statistic
                            left_var = subseries.iloc[:k].var()
                            right_var = subseries.iloc[k:].var()

                            if left_var > 0 and right_var > 0:
                                pooled_var = (
                                    (k - 1) * left_var + (n - k - 1) * right_var
                                ) / (n - 2)
                                t_stat = abs(left_mean - right_mean) / np.sqrt(
                                    pooled_var * (1 / k + 1 / (n - k))
                                )

                                if t_stat > best_stat:
                                    best_stat = t_stat
                                    best_change = k

                        return (
                            (best_change, best_stat) if best_stat > 2.0 else None
                        )  # t-stat threshold

                    # Apply binary segmentation
                    segments_to_process = [(0, len(series), series)]
                    multiple_changes = []

                    while (
                        segments_to_process and len(multiple_changes) < 5
                    ):  # Max 5 change points
                        start, end, segment = segments_to_process.pop(0)

                        change_result = detect_single_change(segment)
                        if change_result:
                            rel_change_idx, stat = change_result
                            abs_change_idx = start + rel_change_idx

                            multiple_changes.append(
                                {
                                    "index": abs_change_idx,
                                    "timestamp": series.index[abs_change_idx],
                                    "statistic": stat,
                                    "segment_start": start,
                                    "segment_end": end,
                                }
                            )

                            # Add new segments to process
                            left_segment = series.iloc[start : start + rel_change_idx]
                            right_segment = series.iloc[start + rel_change_idx : end]

                            if len(left_segment) >= 20:
                                segments_to_process.append(
                                    (start, start + rel_change_idx, left_segment)
                                )
                            if len(right_segment) >= 20:
                                segments_to_process.append(
                                    (start + rel_change_idx, end, right_segment)
                                )

                    if multiple_changes:
                        detected_changes["binary_segmentation"] = {
                            "change_points": sorted(
                                multiple_changes,
                                key=lambda x: x["statistic"],
                                reverse=True,
                            )
                        }

                except:
                    pass

                if detected_changes:
                    change_points[col] = detected_changes

            except Exception as e:
                print(f"Change point detection failed for {col}: {e}")

        return change_points

    def _detect_regime_switching(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Detect regime switching patterns using Markov switching models and other methods"""
        time_col = getattr(config, "time_col", None)
        if not time_col or time_col not in data.columns:
            return {}

        regime_results = {}

        for col in numeric_cols[:3]:  # Limit for computational efficiency
            series_data = data[[time_col, col]].dropna()
            if len(series_data) < 100:  # Need sufficient data for regime detection
                continue

            try:
                series_data[time_col] = pd.to_datetime(series_data[time_col])
                series_data.set_index(time_col, inplace=True)
                series = series_data[col]

                regime_analysis = {}

                # Hidden Markov Model approach (simplified)
                try:
                    from sklearn.mixture import GaussianMixture

                    # Prepare data for regime detection
                    returns = series.pct_change().dropna()
                    features = np.column_stack(
                        [
                            returns.values,
                            returns.rolling(5).mean().dropna().values,
                            returns.rolling(5).std().dropna().values,
                        ]
                    )

                    # Remove any rows with NaN
                    features = features[~np.isnan(features).any(axis=1)]

                    if len(features) > 50:
                        # Fit GMM with different numbers of regimes
                        best_n_regimes = 2
                        best_score = -np.inf

                        for n_regimes in range(2, 5):  # Test 2-4 regimes
                            try:
                                gmm = GaussianMixture(
                                    n_components=n_regimes,
                                    random_state=config.random_state,
                                )
                                gmm.fit(features)
                                score = gmm.score(features)

                                if score > best_score:
                                    best_score = score
                                    best_n_regimes = n_regimes
                            except:
                                continue

                        # Fit best model
                        final_gmm = GaussianMixture(
                            n_components=best_n_regimes,
                            random_state=config.random_state,
                        )
                        final_gmm.fit(features)
                        regime_labels = final_gmm.predict(features)
                        regime_probs = final_gmm.predict_proba(features)

                        # Analyze regime characteristics
                        regime_stats = {}
                        for regime in range(best_n_regimes):
                            regime_mask = regime_labels == regime
                            regime_returns = returns.iloc[-len(regime_mask) :][
                                regime_mask
                            ]

                            regime_stats[f"regime_{regime}"] = {
                                "mean_return": regime_returns.mean(),
                                "volatility": regime_returns.std(),
                                "duration_pct": regime_mask.sum()
                                / len(regime_mask)
                                * 100,
                                "persistence": self._calculate_regime_persistence(
                                    regime_labels, regime
                                ),
                            }

                        # Regime transitions
                        transitions = []
                        for i in range(1, len(regime_labels)):
                            if regime_labels[i] != regime_labels[i - 1]:
                                transitions.append(
                                    {
                                        "from_regime": int(regime_labels[i - 1]),
                                        "to_regime": int(regime_labels[i]),
                                        "timestamp": (
                                            returns.index[i]
                                            if i < len(returns.index)
                                            else None
                                        ),
                                    }
                                )

                        regime_analysis["markov_switching"] = {
                            "n_regimes": best_n_regimes,
                            "regime_statistics": regime_stats,
                            "transitions": transitions[-10:],  # Last 10 transitions
                            "current_regime": int(regime_labels[-1]),
                            "current_regime_probability": float(
                                np.max(regime_probs[-1])
                            ),
                        }

                except:
                    pass

                # Threshold autoregressive model detection
                try:
                    # Simple TAR model: different behavior above/below threshold
                    median_val = series.median()
                    above_threshold = series > median_val

                    # Calculate statistics for each regime
                    high_regime_stats = {
                        "mean": series[above_threshold].mean(),
                        "std": series[above_threshold].std(),
                        "autocorr": (
                            series[above_threshold].autocorr()
                            if len(series[above_threshold]) > 1
                            else 0
                        ),
                        "duration_pct": above_threshold.sum() / len(series) * 100,
                    }

                    low_regime_stats = {
                        "mean": series[~above_threshold].mean(),
                        "std": series[~above_threshold].std(),
                        "autocorr": (
                            series[~above_threshold].autocorr()
                            if len(series[~above_threshold]) > 1
                            else 0
                        ),
                        "duration_pct": (~above_threshold).sum() / len(series) * 100,
                    }

                    # Test for regime differences
                    from scipy.stats import ttest_ind

                    t_stat, p_val = ttest_ind(
                        series[above_threshold], series[~above_threshold]
                    )

                    regime_analysis["threshold_autoregressive"] = {
                        "threshold": median_val,
                        "high_regime": high_regime_stats,
                        "low_regime": low_regime_stats,
                        "regime_difference_significant": p_val < 0.05,
                        "t_statistic": t_stat,
                        "p_value": p_val,
                    }

                except:
                    pass

                # Volatility regime switching
                try:
                    returns = series.pct_change().dropna()
                    if len(returns) > 50:
                        # Rolling volatility
                        vol_window = min(20, len(returns) // 5)
                        rolling_vol = returns.rolling(vol_window).std()

                        # High/low volatility regimes
                        vol_threshold = rolling_vol.quantile(0.7)  # 70th percentile
                        high_vol_periods = rolling_vol > vol_threshold

                        # Volatility regime persistence
                        vol_regime_changes = (
                            high_vol_periods != high_vol_periods.shift(1)
                        ).sum()
                        avg_regime_length = len(high_vol_periods) / (
                            vol_regime_changes + 1
                        )

                        regime_analysis["volatility_switching"] = {
                            "vol_threshold": vol_threshold,
                            "high_vol_pct": high_vol_periods.sum()
                            / len(high_vol_periods)
                            * 100,
                            "avg_regime_length": avg_regime_length,
                            "regime_switches": vol_regime_changes,
                            "current_regime": (
                                "high" if high_vol_periods.iloc[-1] else "low"
                            ),
                        }

                except:
                    pass

                if regime_analysis:
                    regime_results[col] = regime_analysis

            except Exception as e:
                print(f"Regime switching detection failed for {col}: {e}")

        return regime_results

    def _calculate_regime_persistence(
        self, regime_labels: np.ndarray, regime: int
    ) -> float:
        """Calculate average persistence (duration) of a regime"""
        regime_runs = []
        current_run = 0

        for label in regime_labels:
            if label == regime:
                current_run += 1
            else:
                if current_run > 0:
                    regime_runs.append(current_run)
                    current_run = 0

        if current_run > 0:
            regime_runs.append(current_run)

        return np.mean(regime_runs) if regime_runs else 0

    def _assess_forecasting_readiness(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Assess how ready each time series is for forecasting"""
        readiness_scores = {}

        for col in numeric_cols[:5]:
            series = data[col].dropna()
            if len(series) < 20:
                continue

            try:
                score_components = {}

                # Data sufficiency (more data = better)
                data_score = min(
                    len(series) / 100, 1.0
                )  # Normalize to [0,1], optimal at 100+ points
                score_components["data_sufficiency"] = data_score

                # Missing data penalty
                missing_pct = data[col].isnull().mean()
                missing_score = 1.0 - missing_pct
                score_components["completeness"] = missing_score

                # Stationarity assessment
                try:
                    from statsmodels.tsa.stattools import adfuller

                    adf_result = adfuller(series)
                    stationarity_score = 1.0 if adf_result[1] < 0.05 else 0.5
                except:
                    stationarity_score = 0.5
                score_components["stationarity"] = stationarity_score

                # Trend strength (moderate trend is good for forecasting)
                from scipy.stats import linregress

                x = np.arange(len(series))
                slope, _, r_value, p_value, _ = linregress(x, series)
                trend_strength = abs(r_value) if p_value < 0.05 else 0
                trend_score = min(trend_strength * 2, 1.0)  # Cap at 1.0
                score_components["trend_strength"] = trend_score

                # Seasonality (detectable seasonality helps forecasting)
                seasonality_score = 0
                try:
                    from statsmodels.tsa.seasonal import STL

                    if len(series) >= 24:
                        period = min(12, len(series) // 3)
                        stl = STL(series, seasonal=period).fit()
                        seasonal_var = np.var(stl.seasonal)
                        total_var = np.var(series)
                        seasonality_strength = (
                            seasonal_var / total_var if total_var > 0 else 0
                        )
                        seasonality_score = min(seasonality_strength * 2, 1.0)
                except:
                    pass
                score_components["seasonality"] = seasonality_score

                # Noise level (lower noise = better for forecasting)
                returns = series.pct_change().dropna()
                if len(returns) > 1:
                    noise_level = returns.std()
                    # Normalize noise score (lower noise = higher score)
                    noise_score = max(0, 1.0 - min(noise_level, 1.0))
                else:
                    noise_score = 0.5
                score_components["signal_to_noise"] = noise_score

                # Autocorrelation (predictable patterns)
                autocorr_score = 0
                try:
                    from statsmodels.tsa.stattools import acf

                    acf_vals = acf(series, nlags=min(20, len(series) // 2), fft=True)
                    significant_lags = (
                        np.abs(acf_vals[1:]) > 1.96 / np.sqrt(len(series))
                    ).sum()
                    autocorr_score = min(significant_lags / 10, 1.0)
                except:
                    pass
                score_components["autocorrelation"] = autocorr_score

                # Outlier penalty
                q75, q25 = np.percentile(series, [75, 25])
                iqr = q75 - q25
                outliers = (
                    (series < (q25 - 1.5 * iqr)) | (series > (q75 + 1.5 * iqr))
                ).sum()
                outlier_pct = outliers / len(series)
                outlier_score = max(
                    0, 1.0 - outlier_pct * 5
                )  # Heavy penalty for outliers
                score_components["outlier_robustness"] = outlier_score

                # Overall readiness score (weighted average)
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

                overall_score = sum(
                    score_components[component] * weights[component]
                    for component in score_components
                )

                # Readiness classification
                if overall_score >= 0.8:
                    readiness_level = "excellent"
                elif overall_score >= 0.6:
                    readiness_level = "good"
                elif overall_score >= 0.4:
                    readiness_level = "fair"
                else:
                    readiness_level = "poor"

                # Recommendations
                recommendations = []
                if score_components["data_sufficiency"] < 0.5:
                    recommendations.append("Collect more historical data")
                if score_components["completeness"] < 0.8:
                    recommendations.append("Address missing values")
                if score_components["stationarity"] < 0.7:
                    recommendations.append("Apply differencing or transformation")
                if score_components["outlier_robustness"] < 0.7:
                    recommendations.append("Clean outliers")
                if score_components["signal_to_noise"] < 0.5:
                    recommendations.append("Apply smoothing techniques")

                readiness_scores[col] = {
                    "overall_score": overall_score,
                    "readiness_level": readiness_level,
                    "component_scores": score_components,
                    "recommendations": recommendations,
                }

            except Exception as e:
                print(f"Forecasting readiness assessment failed for {col}: {e}")

        return readiness_scores

    def _analyze_volatility(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Advanced volatility analysis including GARCH effects and volatility clustering"""
        volatility_results = {}

        for col in numeric_cols[:5]:
            series = data[col].dropna()
            if len(series) < 50:
                continue

            try:
                volatility_analysis = {}

                # Calculate returns
                returns = series.pct_change().dropna()
                if len(returns) < 20:
                    continue

                # Basic volatility measures
                volatility_analysis["basic_measures"] = {
                    "returns_mean": returns.mean(),
                    "returns_std": returns.std(),
                    "annualized_volatility": returns.std()
                    * np.sqrt(252),  # Assuming daily data
                    "skewness": returns.skew(),
                    "kurtosis": returns.kurtosis(),
                    "sharpe_ratio": (
                        returns.mean() / returns.std() if returns.std() > 0 else 0
                    ),
                }

                # Volatility clustering tests
                try:
                    from statsmodels.stats.diagnostic import het_arch

                    # ARCH test
                    if len(returns) > 10:
                        arch_lm_stat, arch_p_val, _, _ = het_arch(returns, nlags=5)
                        volatility_analysis["arch_test"] = {
                            "statistic": arch_lm_stat,
                            "p_value": arch_p_val,
                            "volatility_clustering": arch_p_val < 0.05,
                        }
                except:
                    pass

                # Rolling volatility analysis
                try:
                    window_size = min(20, len(returns) // 5)
                    rolling_vol = returns.rolling(window_size).std()

                    volatility_analysis["rolling_volatility"] = {
                        "mean": rolling_vol.mean(),
                        "std": rolling_vol.std(),
                        "min": rolling_vol.min(),
                        "max": rolling_vol.max(),
                        "volatility_of_volatility": (
                            rolling_vol.std() / rolling_vol.mean()
                            if rolling_vol.mean() > 0
                            else 0
                        ),
                    }

                    # Volatility regime identification
                    vol_threshold_high = rolling_vol.quantile(0.75)
                    vol_threshold_low = rolling_vol.quantile(0.25)

                    high_vol_periods = (rolling_vol > vol_threshold_high).sum()
                    low_vol_periods = (rolling_vol < vol_threshold_low).sum()

                    volatility_analysis["volatility_regimes"] = {
                        "high_vol_periods": high_vol_periods,
                        "low_vol_periods": low_vol_periods,
                        "high_vol_pct": high_vol_periods
                        / len(rolling_vol.dropna())
                        * 100,
                        "low_vol_pct": low_vol_periods
                        / len(rolling_vol.dropna())
                        * 100,
                    }

                except:
                    pass

                # GARCH modeling readiness
                try:
                    # Test for GARCH effects
                    squared_returns = returns**2
                    from statsmodels.tsa.stattools import acf

                    # Autocorrelation in squared returns
                    acf_sq = acf(
                        squared_returns, nlags=min(10, len(squared_returns) // 4)
                    )
                    significant_acf = (
                        np.abs(acf_sq[1:]) > 1.96 / np.sqrt(len(squared_returns))
                    ).sum()

                    volatility_analysis["garch_effects"] = {
                        "squared_returns_acf": (
                            acf_sq[1:5].tolist()
                            if len(acf_sq) > 5
                            else acf_sq[1:].tolist()
                        ),
                        "significant_lags": int(significant_acf),
                        "garch_suitable": significant_acf > 0,
                    }

                except:
                    pass

                # Value at Risk (VaR) estimates
                try:
                    var_95 = np.percentile(returns, 5)
                    var_99 = np.percentile(returns, 1)

                    # Expected Shortfall (Conditional VaR)
                    es_95 = returns[returns <= var_95].mean()
                    es_99 = returns[returns <= var_99].mean()

                    volatility_analysis["risk_measures"] = {
                        "var_95": var_95,
                        "var_99": var_99,
                        "expected_shortfall_95": es_95,
                        "expected_shortfall_99": es_99,
                        "max_drawdown": (
                            returns.cumsum() - returns.cumsum().expanding().max()
                        ).min(),
                    }

                except:
                    pass

                volatility_results[col] = volatility_analysis

            except Exception as e:
                print(f"Volatility analysis failed for {col}: {e}")

        return volatility_results

    def _detect_cyclical_patterns(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Detect cyclical patterns using advanced spectral analysis"""
        cyclical_results = {}

        for col in numeric_cols[:5]:
            series = data[col].dropna()
            if len(series) < 50:
                continue

            try:
                cyclical_analysis = {}

                # Fourier Transform Analysis
                try:
                    from scipy.fft import fft, fftfreq

                    # Detrend the series
                    detrended = (
                        series - series.rolling(window=min(12, len(series) // 4)).mean()
                    )
                    detrended = detrended.dropna()

                    if len(detrended) > 20:
                        # Compute FFT
                        fft_vals = fft(detrended.values)
                        freqs = fftfreq(len(detrended))

                        # Get power spectrum
                        power_spectrum = np.abs(fft_vals) ** 2

                        # Find dominant frequencies (excluding DC component)
                        positive_freqs = freqs[: len(freqs) // 2][1:]  # Exclude DC
                        positive_power = power_spectrum[: len(power_spectrum) // 2][1:]

                        # Find peaks in power spectrum
                        peak_indices = find_peaks(
                            positive_power, height=np.percentile(positive_power, 85)
                        )[0]

                        dominant_cycles = []
                        for peak_idx in peak_indices[:5]:  # Top 5 peaks
                            freq = positive_freqs[peak_idx]
                            if freq > 0:
                                period = 1 / freq
                                power = positive_power[peak_idx]
                                dominant_cycles.append(
                                    {
                                        "period": period,
                                        "frequency": freq,
                                        "power": power,
                                        "power_normalized": power
                                        / np.sum(positive_power),
                                    }
                                )

                        cyclical_analysis["fourier_analysis"] = {
                            "dominant_cycles": sorted(
                                dominant_cycles, key=lambda x: x["power"], reverse=True
                            ),
                            "total_cycles_detected": len(dominant_cycles),
                        }

                except:
                    pass

                # Wavelet Analysis (simplified)
                try:
                    from scipy.signal import cwt, ricker

                    # Use Ricker wavelets for different scales
                    scales = np.arange(2, min(50, len(series) // 4))
                    coefficients = cwt(series.values, ricker, scales)

                    # Find scales with highest energy
                    energy_per_scale = np.sum(coefficients**2, axis=1)
                    dominant_scale_idx = np.argmax(energy_per_scale)
                    dominant_scale = scales[dominant_scale_idx]

                    cyclical_analysis["wavelet_analysis"] = {
                        "dominant_scale": dominant_scale,
                        "energy_at_dominant_scale": energy_per_scale[
                            dominant_scale_idx
                        ],
                        "scales_analyzed": len(scales),
                    }

                except:
                    pass

                # Business cycle detection (for economic data)
                try:
                    # Hodrick-Prescott filter for cycle extraction
                    from statsmodels.tsa.filters.hp_filter import hpfilter

                    hp_cycle, hp_trend = hpfilter(
                        series, lamb=1600
                    )  # Standard lambda for quarterly data

                    # Analyze cycle characteristics
                    cycle_peaks = find_peaks(hp_cycle)[0]
                    cycle_troughs = find_peaks(-hp_cycle)[0]

                    # Calculate cycle durations
                    cycle_durations = []
                    if len(cycle_peaks) > 1:
                        cycle_durations = np.diff(cycle_peaks)

                    cyclical_analysis["business_cycle"] = {
                        "cycle_component_std": hp_cycle.std(),
                        "trend_component_std": hp_trend.std(),
                        "n_peaks": len(cycle_peaks),
                        "n_troughs": len(cycle_troughs),
                        "avg_cycle_duration": (
                            np.mean(cycle_durations) if cycle_durations else None
                        ),
                        "cycle_amplitude": np.max(hp_cycle) - np.min(hp_cycle),
                    }

                except:
                    pass

                # Phase analysis
                try:
                    # Hilbert transform for instantaneous phase
                    from scipy.signal import hilbert

                    analytic_signal = hilbert(detrended.values)
                    instantaneous_phase = np.angle(analytic_signal)
                    instantaneous_amplitude = np.abs(analytic_signal)

                    # Phase consistency measure
                    phase_diff = np.diff(instantaneous_phase)
                    phase_consistency = (
                        1 - np.std(phase_diff) / np.pi
                    )  # Normalized to [0,1]

                    cyclical_analysis["phase_analysis"] = {
                        "phase_consistency": phase_consistency,
                        "mean_amplitude": np.mean(instantaneous_amplitude),
                        "amplitude_variability": np.std(instantaneous_amplitude)
                        / np.mean(instantaneous_amplitude),
                    }

                except:
                    pass

                cyclical_results[col] = cyclical_analysis

            except Exception as e:
                print(f"Cyclical pattern detection failed for {col}: {e}")

        return cyclical_results

    def _granger_causality_analysis(
        self, data: pd.DataFrame, numeric_cols: List[str], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Granger causality analysis between time series"""
        if len(numeric_cols) < 2:
            return {}

        causality_results = {}

        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # Test causality between all pairs of variables
            for i, var1 in enumerate(
                numeric_cols[:5]
            ):  # Limit to first 5 for computational efficiency
                for var2 in enumerate(numeric_cols[:5]):
                    if var1 != var2 and var1 in data.columns and var2 in data.columns:

                        # Prepare data
                        pair_data = data[[var1, var2]].dropna()
                        if len(pair_data) < 50:  # Need sufficient data
                            continue

                        try:
                            # Test both directions
                            max_lag = min(12, len(pair_data) // 10)

                            # var1 -> var2
                            try:
                                test_data_12 = pair_data[
                                    [var2, var1]
                                ].values  # Note: order matters in granger test
                                gc_result_12 = grangercausalitytests(
                                    test_data_12, maxlag=max_lag, verbose=False
                                )

                                # Extract p-values for different lags
                                p_values_12 = {}
                                for lag in range(1, max_lag + 1):
                                    if lag in gc_result_12:
                                        p_val = gc_result_12[lag][0]["ssr_ftest"][
                                            1
                                        ]  # F-test p-value
                                        p_values_12[lag] = p_val

                                # Find best lag (lowest p-value)
                                if p_values_12:
                                    best_lag_12 = min(
                                        p_values_12.keys(), key=lambda x: p_values_12[x]
                                    )
                                    best_p_val_12 = p_values_12[best_lag_12]
                                    causality_12 = best_p_val_12 < 0.05
                                else:
                                    best_lag_12 = None
                                    best_p_val_12 = 1.0
                                    causality_12 = False

                            except:
                                best_lag_12 = None
                                best_p_val_12 = 1.0
                                causality_12 = False

                            # var2 -> var1
                            try:
                                test_data_21 = pair_data[[var1, var2]].values
                                gc_result_21 = grangercausalitytests(
                                    test_data_21, maxlag=max_lag, verbose=False
                                )

                                p_values_21 = {}
                                for lag in range(1, max_lag + 1):
                                    if lag in gc_result_21:
                                        p_val = gc_result_21[lag][0]["ssr_ftest"][1]
                                        p_values_21[lag] = p_val

                                if p_values_21:
                                    best_lag_21 = min(
                                        p_values_21.keys(), key=lambda x: p_values_21[x]
                                    )
                                    best_p_val_21 = p_values_21[best_lag_21]
                                    causality_21 = best_p_val_21 < 0.05
                                else:
                                    best_lag_21 = None
                                    best_p_val_21 = 1.0
                                    causality_21 = False

                            except:
                                best_lag_21 = None
                                best_p_val_21 = 1.0
                                causality_21 = False

                            # Store results if any causality detected
                            if causality_12 or causality_21:
                                pair_key = f"{var1}_vs_{var2}"
                                causality_results[pair_key] = {
                                    "var1_causes_var2": {
                                        "significant": causality_12,
                                        "best_lag": best_lag_12,
                                        "p_value": best_p_val_12,
                                    },
                                    "var2_causes_var1": {
                                        "significant": causality_21,
                                        "best_lag": best_lag_21,
                                        "p_value": best_p_val_21,
                                    },
                                    "bidirectional": causality_12 and causality_21,
                                    "sample_size": len(pair_data),
                                }

                        except Exception as e:
                            print(
                                f"Granger causality test failed for {var1} vs {var2}: {e}"
                            )
                            continue

        except ImportError:
            print("statsmodels not available for Granger causality tests")
        except Exception as e:
            print(f"Granger causality analysis failed: {e}")

        return causality_results
