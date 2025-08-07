from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .analyze_correlation import CorrelationAnalyzer
from .analyze_distribution import DistributionAnalyzer
from .foreminer_aux import *


class PatternDetector(AnalysisStrategy):
    """SOTA Pattern Detection with Advanced ML Techniques"""

    @property
    def name(self) -> str:
        return "patterns"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        return {
            "feature_types": self._classify_features(data, config),
            "relationships": self._analyze_relationships(data, config),
            "distributions": self._fit_distributions(data, config),
        }

    def _classify_features(self, data, config) -> Dict[str, List[str]]:
        from scipy.stats import anderson, jarque_bera, shapiro
        from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
        from sklearn.preprocessing import PowerTransformer

        dist_summary = (
            DistributionAnalyzer().analyze(data, config).get("summary", pd.DataFrame())
        )
        classification = {
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

        for _, row in dist_summary.iterrows():
            feature = row["feature"]
            skew, kurt, std, mean = (
                row["skewness"],
                row["kurtosis"],
                row["std"],
                row["mean"],
            )
            col_data = data[feature].dropna()
            if len(col_data) < 10:
                continue

            # Enhanced normality testing
            is_normal = False
            if len(col_data) <= 5000:
                try:
                    _, jb_p = jarque_bera(col_data)
                    _, sw_p = shapiro(col_data.sample(min(5000, len(col_data))))
                    anderson_stat, critical_vals, _ = anderson(col_data, dist="norm")
                    is_normal = (
                        jb_p > 0.05 and sw_p > 0.05 and anderson_stat < critical_vals[2]
                    )
                except:
                    pass

            if is_normal or row.get("is_gaussian", False):
                classification["gaussian"].append(feature)

            # Advanced distribution classification
            if row.get("is_heavy_tailed", False) or kurt > 5:
                classification["heavy_tailed"].append(feature)

            if abs(skew) > 2:
                classification["highly_skewed"].append(feature)

            # Log-normal detection with better heuristics
            if (
                mean > 0
                and skew > 1
                and std > 0
                and col_data.min() > 0
                and np.log(col_data).std() < col_data.std()
            ):
                classification["log_normal_candidates"].append(feature)

            # Zero-inflated detection
            zero_pct = (col_data == 0).mean()
            if zero_pct > 0.1 and zero_pct < 0.9:
                classification["zero_inflated"].append(feature)

            # Bounded/discrete detection
            if col_data.min() >= 0 and col_data.max() <= 1:
                classification["bounded"].append(feature)
            elif (
                np.all(np.isclose(col_data % 1, 0))
                and col_data.var() / (col_data.mean() + 1e-8) < 3
            ):
                classification["count_like"].append(feature)

            # Power law detection
            if col_data.min() > 0:
                log_data = np.log(col_data)
                if log_data.std() > 1 and skew > 2:
                    # Simple power law test
                    x_log = np.log(np.sort(col_data)[::-1])
                    y_log = np.log(np.arange(1, len(x_log) + 1))
                    if len(x_log) > 10:
                        corr = np.corrcoef(
                            x_log[: len(x_log) // 2], y_log[: len(x_log) // 2]
                        )[0, 1]
                        if abs(corr) > 0.8:
                            classification["power_law"].append(feature)

            # Uniform-like detection
            if abs(skew) < 0.5 and abs(kurt) < 1.5:
                classification["uniform_like"].append(feature)

            # Mixture model detection with Bayesian approach
            if kurt > 1 or len(np.unique(col_data)) > 20:
                try:
                    # Try both regular and Bayesian GMM
                    gmm = GaussianMixture(
                        n_components=2, random_state=config.random_state
                    )
                    bgmm = BayesianGaussianMixture(
                        n_components=3, random_state=config.random_state
                    )

                    gmm.fit(col_data.values.reshape(-1, 1))
                    bgmm.fit(col_data.values.reshape(-1, 1))

                    # Use AIC/BIC for model selection
                    single_aic = 2 * 2 - 2 * np.sum(
                        np.log(col_data.std() * np.sqrt(2 * np.pi))
                        - 0.5 * ((col_data - col_data.mean()) / col_data.std()) ** 2
                    )

                    if gmm.converged_ and gmm.aic_ < single_aic:
                        classification["mixture_model"].append(feature)
                    elif (
                        bgmm.converged_
                        and hasattr(bgmm, "lower_bound_")
                        and bgmm.n_components_ > 1
                    ):
                        classification["bimodal"].append(feature)
                except:
                    pass

            # Test transformability to normal
            if not is_normal and col_data.min() >= 0:
                try:
                    pt = PowerTransformer(method="yeo-johnson")
                    transformed = pt.fit_transform(
                        col_data.values.reshape(-1, 1)
                    ).flatten()
                    _, trans_p = shapiro(
                        transformed.sample(min(5000, len(transformed)))
                    )
                    if trans_p > 0.05:
                        classification["transformable_to_normal"].append(feature)
                except:
                    pass

            # Default continuous classification
            if (
                feature
                not in classification["count_like"]
                + classification["gaussian"]
                + classification["bounded"]
                + classification["zero_inflated"]
            ):
                classification["continuous"].append(feature)

            # Seasonality detection
            if time_col and pd.api.types.is_datetime64_any_dtype(data[time_col]):
                try:
                    from statsmodels.tsa.seasonal import STL
                    from statsmodels.tsa.stattools import acf

                    series = data.set_index(time_col)[feature].dropna()
                    if len(series) >= 24:
                        # STL decomposition
                        stl = STL(series, seasonal=min(13, len(series) // 3)).fit()
                        seasonal_strength = np.var(stl.seasonal) / (
                            np.var(series) + 1e-8
                        )

                        # Autocorrelation analysis
                        autocorr = acf(
                            series, nlags=min(40, len(series) // 4), fft=True
                        )

                        if (
                            seasonal_strength > 0.15
                            or np.max(np.abs(autocorr[12:])) > 0.3
                        ):
                            classification["seasonal"].append(feature)
                except:
                    pass

        return classification

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

        patterns = {
            "nonlinear": [],
            "complex": [],
            "distance_corr": [],
            "monotonic": [],
            "tail_dependence": [],
            "conditional": [],
        }

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                f1, f2 = numeric_cols[i], numeric_cols[j]

                # Skip if too many missing values
                valid_data = data[[f1, f2]].dropna()
                if len(valid_data) < 10:
                    continue

                x, y = valid_data[f1], valid_data[f2]

                # Standard correlations
                pc = (
                    abs(pearson.loc[f1, f2])
                    if f1 in pearson.index and f2 in pearson.columns
                    else 0
                )

                try:
                    sc, _ = spearmanr(x, y)
                    sc = abs(sc)
                    tau, _ = kendalltau(x, y)
                    tau = abs(tau)
                except:
                    sc = tau = 0

                # Mutual information
                try:
                    if len(x) > 50:
                        mi_val = mutual_info_regression(
                            x.values.reshape(-1, 1), y, random_state=config.random_state
                        )[0]
                    else:
                        mi_val = 0
                except:
                    mi_val = 0

                # Distance correlation
                dc = 0
                if DCOR_AVAILABLE and len(x) <= 1000:  # Limit for performance
                    try:
                        dc = distance_correlation(x, y)
                    except:
                        pass

                # Nonlinear relationships
                if sc > 0.4 and sc - pc > 0.2:
                    patterns["nonlinear"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "pearson": pc,
                            "spearman": sc,
                            "kendall": tau,
                            "nonlinearity_score": sc - pc,
                        }
                    )

                # Complex relationships via mutual information
                if mi_val > 0.2 and pc < 0.4:
                    patterns["complex"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "mutual_info": mi_val,
                            "pearson": pc,
                            "complexity_score": mi_val - pc,
                        }
                    )

                # Distance correlation patterns
                if dc > 0.4 and pc < 0.4:
                    patterns["distance_corr"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "distance_corr": dc,
                            "pearson": pc,
                        }
                    )

                # Monotonic relationships
                if tau > 0.5:
                    patterns["monotonic"].append(
                        {
                            "feature1": f1,
                            "feature2": f2,
                            "kendall_tau": tau,
                            "relationship_type": "monotonic",
                        }
                    )

                # Tail dependence (simplified)
                if len(x) > 100:
                    try:
                        # Upper tail dependence
                        q95_x, q95_y = np.percentile(x, 95), np.percentile(y, 95)
                        upper_tail = ((x >= q95_x) & (y >= q95_y)).mean()

                        if upper_tail > 0.05:  # 5% threshold
                            patterns["tail_dependence"].append(
                                {
                                    "feature1": f1,
                                    "feature2": f2,
                                    "upper_tail_dep": upper_tail,
                                    "type": "upper_tail",
                                }
                            )
                    except:
                        pass

        # Sort and limit results
        for k in patterns:
            if patterns[k]:
                sort_key = lambda x: (
                    list(x.values())[-1]
                    if isinstance(list(x.values())[-1], (int, float))
                    else 0
                )
                patterns[k] = sorted(patterns[k], key=sort_key, reverse=True)[:10]

        return patterns

    def _fit_distributions(self, data, config) -> Dict[str, Dict[str, Any]]:
        import warnings

        from scipy import stats

        warnings.filterwarnings("ignore")

        # Extended list of distributions
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

        fits = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols[:15]:  # Limit for performance
            x = data[col].dropna()
            if len(x) < 30:
                continue

            # Sample for large datasets
            if len(x) > 2000:
                x = x.sample(2000, random_state=config.random_state)

            candidate_fits = []

            for name, dist in distributions:
                try:
                    # Fit distribution
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        params = dist.fit(x)

                    # Calculate goodness of fit metrics
                    log_likelihood = np.sum(dist.logpdf(x, *params))
                    k = len(params)  # number of parameters
                    n = len(x)  # sample size

                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood

                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.kstest(x, dist.cdf, args=params)

                    # Anderson-Darling test (where applicable)
                    ad_stat = ad_p = None
                    if name in ["normal", "exponential", "logistic", "gumbel_r"]:
                        try:
                            ad_result = stats.anderson(x, dist=name.replace("_r", ""))
                            ad_stat = ad_result.statistic
                            # Approximate p-value based on critical values
                            if ad_stat < ad_result.critical_values[2]:
                                ad_p = 0.05  # Rough approximation
                            else:
                                ad_p = 0.01
                        except:
                            pass

                    candidate_fits.append(
                        {
                            "distribution": name,
                            "params": params,
                            "aic": aic,
                            "bic": bic,
                            "log_likelihood": log_likelihood,
                            "ks_statistic": ks_stat,
                            "ks_pvalue": ks_p,
                            "ad_statistic": ad_stat,
                            "ad_pvalue": ad_p,
                            "fit_quality": ks_p if ks_p > 0 else 1e-10,  # For sorting
                        }
                    )

                except Exception:
                    continue

            if candidate_fits:
                # Sort by multiple criteria: AIC, then KS p-value
                candidate_fits.sort(key=lambda x: (x["aic"], -x["fit_quality"]))

                # Best fit for backward compatibility (flat structure)
                best_fit = candidate_fits[0]
                fits[col] = {
                    "distribution": best_fit["distribution"],
                    "params": best_fit["params"],
                    "aic": best_fit["aic"],
                    "bic": best_fit["bic"],
                    "ks_stat": best_fit["ks_statistic"],
                    "ks_pvalue": best_fit["ks_pvalue"],
                }

                # Store detailed results for advanced users
                fits[col]["_detailed"] = {
                    "best_fit": best_fit,
                    "alternatives": candidate_fits[1:3],  # Top 3 alternatives
                    "total_tested": len(candidate_fits),
                }

        return fits
