import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from .analyze_cluster import ClusterAnalyzer
from .analyze_correlation import CorrelationAnalyzer
from .analyze_dimension import DimensionalityAnalyzer
from .analyze_distribution import DistributionAnalyzer
from .analyze_feat import FeatureEngineeringAnalyzer
from .analyze_missing import MissingnessAnalyzer
from .analyze_outlier import OutlierAnalyzer
from .analyze_pattern import PatternDetector
from .analyze_ts import TimeSeriesAnalyzer
from .foreminer_aux import *
from .foreminer_aux import _run_analysis_worker

# Suppress known noise warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)


# ============================================================================
# COMPREHENSIVE ANALYSIS STRATEGIES
# ============================================================================

warnings.filterwarnings("ignore")


class SHAPAnalyzer(AnalysisStrategy):
    """SHAP-based feature explanation"""

    @property
    def name(self) -> str:
        return "shap_explanation"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        # This would require a fitted model to be passed in
        # For now, return placeholder
        return {"explanation": "SHAP analysis requires a fitted model"}


# ============================================================================
# MAIN ANALYZER CLASS (Clean and comprehensive)
# ============================================================================


class DatasetAnalyzer:
    """Clean, extensible dataset analyzer with all advanced functionality preserved"""

    def __init__(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        config: Optional[AnalysisConfig] = None,
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.time_col = time_col
        self.config = config or AnalysisConfig()
        self.verbose = verbose

        # Initialize components
        self.hooks = AnalysisHooks()
        self.plot_helper = PlotHelper()
        self._strategies: Dict[str, AnalysisStrategy] = {}
        self._results_cache: Dict[str, Any] = {}

        # Cache frequently used data
        self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Register default strategies and plotters
        self._register_default_strategies()
        self._register_default_plotters()

        # Setup
        self._setup()

    def _setup(self):
        """Initialize analyzer setup"""
        PlotHelper.setup_style(self.config)
        self._setup_time_index()

        if self.verbose:
            print(
                f"🔍 Initialized analyzer with {self.df.shape[0]:,} rows × {self.df.shape[1]} columns"
            )
            print(f"   • Numeric features: {len(self._numeric_cols)}")
            print(f"   • Categorical features: {len(self._categorical_cols)}")

    def _setup_time_index(self):
        """Setup time index if provided"""
        if self.time_col and self.time_col in self.df.columns:
            self.df.set_index(self.time_col, inplace=True)
            self.df.index = pd.to_datetime(self.df.index)

    def _register_default_strategies(self):
        """Register all available analysis strategies"""
        strategies = [
            DistributionAnalyzer(),
            CorrelationAnalyzer(),
            OutlierAnalyzer(),
            ClusterAnalyzer(),
            DimensionalityAnalyzer(),
            PatternDetector(),
            MissingnessAnalyzer(),
            FeatureEngineeringAnalyzer(),
            SHAPAnalyzer(),
        ]

        # Add time series analyzer if time column is provided
        if self.time_col:
            strategies.append(TimeSeriesAnalyzer())

        for strategy in strategies:
            self._strategies[strategy.name] = strategy

    def _register_default_plotters(self):
        """Register all available plotters"""
        plotters = [
            ("distributions", PlotHelper.plot_distributions),
            ("correlations", PlotHelper.plot_correlations),
            ("outliers", PlotHelper.plot_outliers_pca),
            ("clusters", PlotHelper.plot_clusters),
            ("dimensionality", PlotHelper.plot_dimensionality),
            ("missingness", PlotHelper.plot_missingness_analysis),
        ]

        if OPTIONAL_IMPORTS["networkx"]:
            plotters.append(
                ("correlation_network", PlotHelper.plot_correlation_network)
            )

        for analysis_type, plotter in plotters:
            self.hooks.register_plotter(analysis_type, plotter)

    def _log(self, message: str):
        if self.verbose:
            print(f"🔍 {message}")

    @lru_cache(maxsize=32)
    def _get_clean_numeric_data(self) -> pd.DataFrame:
        """Get cached clean numeric data"""
        return self.df[self._numeric_cols].dropna()

    # ========================================================================
    # PUBLIC API (Clean and comprehensive)
    # ========================================================================
    def analyze(self, analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run specified analyses in parallel with pre/post hooks"""
        analysis_types = analysis_types or list(self._strategies.keys())
        results = {}

        tasks = []
        with ThreadPoolExecutor() as executor:
            for analysis_type in analysis_types:
                if analysis_type not in self._strategies:
                    continue

                self._log(f"Running {analysis_type} analysis...")
                context = {
                    "data": self.df,
                    "config": self.config,
                    "type": analysis_type,
                }
                self.hooks.trigger(f"pre_{analysis_type}", context)

                strategy = self._strategies[analysis_type]
                tasks.append(
                    executor.submit(
                        _run_analysis_worker,
                        analysis_type,
                        strategy,
                        self.df,
                        self.config,
                    )
                )

            for future in as_completed(tasks):
                strategy_name, result, error = future.result()
                if error:
                    print(f"❌ {strategy_name} analysis failed:\n{error}")
                    continue

                results[strategy_name] = result
                self._results_cache[strategy_name] = result
                self.hooks.trigger(
                    f"post_{strategy_name}",
                    {
                        "data": self.df,
                        "config": self.config,
                        "type": strategy_name,
                        "result": result,
                    },
                )

        return results

    def plot(self, analysis_types: Optional[List[str]] = None):
        """Generate comprehensive plots for analysis results"""
        analysis_types = analysis_types or list(self._results_cache.keys())

        for analysis_type in analysis_types:
            if analysis_type in self._results_cache:
                self._log(f"Plotting {analysis_type}...")
                self.hooks.plot(
                    analysis_type,
                    self._results_cache[analysis_type],
                    self.df,
                    self.config,
                )

    def analyze_and_plot(self, analysis_types: Optional[List[str]] = None):
        """Run comprehensive analysis and generate all plots"""
        results = self.analyze(analysis_types)
        self.plot(analysis_types)
        return results

    def analyze_everything(self):
        """Run the complete comprehensive analysis workflow"""
        self._log("🚀 Starting comprehensive dataset analysis...")

        # Run all analyses
        results = self.analyze()

        # Generate comprehensive insights report
        self.print_detailed_insights()

        # Generate all plots
        self.plot()

        # Special network plot if available
        if "correlations" in results and OPTIONAL_IMPORTS["networkx"]:
            try:
                self.plot_correlation_network()
            except Exception as e:
                print(f"Network plot failed: {e}")

        self._log("🎉 Comprehensive analysis complete!")
        return results

    # ========================================================================
    # SPECIALIZED METHODS (Preserved from original)
    # ========================================================================

    def decompose_series(self, column: str, period: Optional[int] = None):
        """STL decomposition for time series"""
        if column not in self._numeric_cols:
            raise ValueError(f"Column {column} not found in numeric columns")

        series = self.df[column].dropna()
        if len(series) < 24:
            raise ValueError("Series too short for decomposition (minimum 24 points)")

        if period is None:
            period = min(max(2, len(series) // 10), 24)

        stl = STL(series, period=period, seasonal=7)
        decomposition = stl.fit()

        # Enhanced plotting
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        decomposition.observed.plot(ax=axes[0], title=f"Original Series: {column}")
        decomposition.trend.plot(ax=axes[1], title="Trend", color="orange")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonal", color="green")
        decomposition.resid.plot(ax=axes[3], title="Residual", color="red")

        plt.tight_layout()
        plt.show()

        return decomposition

    def plot_autocorrelations(self, column: str, lags: int = 40):
        """Enhanced ACF and PACF plots"""
        if column not in self._numeric_cols:
            raise ValueError(f"Column {column} not found in numeric columns")

        series = self.df[column].dropna()
        if len(series) < lags + 10:
            lags = max(10, len(series) // 3)

        # Compute ACF and PACF
        acf_vals = acf(series, nlags=lags, fft=True)
        pacf_vals = pacf(series, nlags=lags, method="ols")

        # Confidence intervals
        n = len(series)
        ci = 1.96 / np.sqrt(n)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # ACF plot
        ax1.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax1.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
        ax1.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
        ax1.fill_between(range(len(acf_vals)), -ci, ci, alpha=0.2, color="red")
        ax1.set_title(f"Autocorrelation Function - {column}")
        ax1.set_xlabel("Lags")
        ax1.set_ylabel("ACF")

        # PACF plot
        ax2.stem(range(len(pacf_vals)), pacf_vals, basefmt=" ")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
        ax2.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
        ax2.fill_between(range(len(pacf_vals)), -ci, ci, alpha=0.2, color="red")
        ax2.set_title(f"Partial Autocorrelation Function - {column}")
        ax2.set_xlabel("Lags")
        ax2.set_ylabel("PACF")

        plt.tight_layout()
        plt.show()

    def plot_correlation_network(
        self, method: str = "pearson", threshold: Optional[float] = None
    ):
        """Create correlation network graph"""
        if "correlations" not in self._results_cache:
            self._log("Running correlation analysis first...")
            self.analyze(["correlations"])

        correlations = self._results_cache["correlations"]
        PlotHelper.plot_correlation_network(correlations, self.df, self.config)

    @requires_library("shap")
    def explain_features(
        self, model, X: Optional[pd.DataFrame] = None, max_display: int = 10
    ):
        """SHAP feature explanation"""
        import shap

        X = X or self._get_clean_numeric_data()
        if X.empty:
            self._log("No data available for SHAP explanation.")
            return {}

        try:
            if hasattr(model, "predict_proba") and "Tree" in model.__class__.__name__:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                self._log("Using KernelExplainer (model-agnostic).")
                f = model.predict if hasattr(model, "predict") else model.predict_proba
                background = X.sample(
                    min(100, len(X)), random_state=self.config.random_state
                )
                explainer = shap.KernelExplainer(f, background)
                shap_values = explainer.shap_values(X.sample(min(500, len(X))))

        except Exception as e:
            self._log(f"SHAP explanation failed: {e}")
            return {}

        # Generate summary plot
        plt.figure(figsize=(10, 5))
        try:
            shap.summary_plot(shap_values, X, show=False, max_display=max_display)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
        except Exception as e:
            self._log(f"Failed to plot SHAP summary: {e}")

        return {
            "shap_values": shap_values,
            "expected_value": getattr(explainer, "expected_value", None),
        }

    # ========================================================================
    # COMPREHENSIVE INSIGHTS AND REPORTING
    # ========================================================================

    def print_detailed_insights(self):
        """Print comprehensive detailed insights with specific recommendations"""
        print("=" * 100)
        print("🔬 DATASET ANALYSIS")
        print("=" * 100)

        # Dataset Overview
        print(f"\n📊 DATASET OVERVIEW")
        print("-" * 40)
        print(f"   Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"   Numeric features: {len(self._numeric_cols)}")
        print(f"   Categorical features: {len(self._categorical_cols)}")
        print(
            f"   Memory usage: {self.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
        )

        missing_pct = (
            self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])
        ) * 100
        print(f"   Missing values: {missing_pct:.2f}% of total dataset")

        # Run analyses if not already cached
        required_analyses = ["distributions", "correlations", "outliers", "patterns"]
        for analysis in required_analyses:
            if analysis not in self._results_cache:
                try:
                    result = self._strategies[analysis].analyze(self.df, self.config)
                    self._results_cache[analysis] = result
                except Exception as e:
                    print(f"Failed to run {analysis}: {e}")

        # Distribution Analysis Insights
        if "distributions" in self._results_cache:
            dist_summary = self._results_cache["distributions"].get(
                "summary", pd.DataFrame()
            )
            if not dist_summary.empty:
                print(f"\n📈 DETAILED DISTRIBUTION ANALYSIS")
                print("-" * 40)

                gaussian_count = dist_summary["is_gaussian"].sum()
                skewed_count = dist_summary["is_skewed"].sum()
                heavy_tailed_count = dist_summary["is_heavy_tailed"].sum()

                print(
                    f"   ✓ Gaussian features: {gaussian_count}/{len(dist_summary)} ({gaussian_count / len(dist_summary) * 100:.1f}%)"
                )
                if gaussian_count > 0:
                    gaussian_features = dist_summary[dist_summary["is_gaussian"]][
                        "feature"
                    ].tolist()
                    print(f"     → {', '.join(gaussian_features[:5])}")

                print(
                    f"   ⚠️  Skewed features: {skewed_count}/{len(dist_summary)} ({skewed_count / len(dist_summary) * 100:.1f}%)"
                )
                if skewed_count > 0:
                    skewed_features = dist_summary[dist_summary["is_skewed"]][
                        "feature"
                    ].tolist()
                    print(f"     → {', '.join(skewed_features[:5])}")

                print(
                    f"   📏 Heavy-tailed features: {heavy_tailed_count}/{len(dist_summary)} ({heavy_tailed_count / len(dist_summary) * 100:.1f}%)"
                )

        # Correlation Insights
        if "correlations" in self._results_cache:
            correlations = self._results_cache["correlations"]
            if "pearson" in correlations:
                print(f"\n🔗 DETAILED CORRELATION ANALYSIS")
                print("-" * 40)

                corr_matrix = correlations["pearson"]
                strong_pos_pairs = []
                strong_neg_pairs = []

                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]

                        if corr_val > 0.7:
                            strong_pos_pairs.append((feat1, feat2, corr_val))
                        elif corr_val < -0.7:
                            strong_neg_pairs.append((feat1, feat2, corr_val))

                if strong_pos_pairs:
                    print(f"   💚 STRONG POSITIVE CORRELATIONS (r > 0.7):")
                    for feat1, feat2, corr in strong_pos_pairs[:5]:
                        print(f"     → {feat1} ↔ {feat2}: {corr:.3f}")

                if strong_neg_pairs:
                    print(f"   💔 STRONG NEGATIVE CORRELATIONS (r < -0.7):")
                    for feat1, feat2, corr in strong_neg_pairs[:5]:
                        print(f"     → {feat1} ↔ {feat2}: {corr:.3f}")
        # Pattern Detection Results
        if "patterns" in self._results_cache:
            patterns = self._results_cache["patterns"]
            print(f"\n🔍 ADVANCED PATTERN DETECTION")
            print("-" * 40)

            # Feature types
            if "feature_types" in patterns:
                feature_types = patterns["feature_types"]
                print("   📋 FEATURE TYPE CLASSIFICATION:")
                for ftype, features in feature_types.items():
                    if features:
                        type_name = ftype.replace("_", " ").title()
                        print(f"     → {type_name}: {len(features)} features")
                        print(f"       ∘ Top: {', '.join(features[:5])}")

                if feature_types.get("seasonal"):
                    print(f"\n   ⏳ SEASONAL FEATURES:")
                    print(f"     → {', '.join(feature_types['seasonal'][:5])}")

                if feature_types.get("transformable_to_normal"):
                    print(f"\n   🔄 TRANSFORMABLE TO GAUSSIAN:")
                    print(
                        f"     → {', '.join(feature_types['transformable_to_normal'][:5])}"
                    )

                if feature_types.get("bimodal") or feature_types.get("mixture_model"):
                    print(f"\n   🔀 MIXTURE DISTRIBUTIONS:")
                    if feature_types.get("bimodal"):
                        print(
                            f"     → Bimodal: {', '.join(feature_types['bimodal'][:5])}"
                        )
                    if feature_types.get("mixture_model"):
                        print(
                            f"     → Mixture Model: {', '.join(feature_types['mixture_model'][:5])}"
                        )

            # Relationships
            if "relationships" in patterns:
                rel_patterns = patterns["relationships"]

                if rel_patterns.get("nonlinear"):
                    print(f"\n   🌀 NON-LINEAR RELATIONSHIPS DETECTED:")
                    for rel in rel_patterns["nonlinear"][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(
                            f"       ∘ Nonlinearity score: {rel['nonlinearity_score']:.3f}"
                        )

                if rel_patterns.get("complex"):
                    print(f"\n   🧬 COMPLEX RELATIONSHIPS (High MI, Low Linear):")
                    for rel in rel_patterns["complex"][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(f"       ∘ Mutual Info: {rel['mutual_info']:.3f}")

                if rel_patterns.get("distance_corr"):
                    print(f"\n   📐 DISTANCE CORRELATION PATTERNS:")
                    for rel in rel_patterns["distance_corr"][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(
                            f"       ∘ DCor: {rel['distance_corr']:.3f}, Pearson: {rel['pearson']:.3f}"
                        )

                if rel_patterns.get("tail_dependence"):
                    print(f"\n   🧪 TAIL DEPENDENCE DETECTED:")
                    for rel in rel_patterns["tail_dependence"][:3]:
                        print(f"     → {rel['feature1']} ↔ {rel['feature2']}")
                        print(
                            f"       ∘ Upper Tail Dependency: {rel['upper_tail_dep']:.2%}"
                        )

            # Best-fit distributions
            if "distributions" in patterns and patterns["distributions"]:
                print(f"\n   📊 BEST-FIT DISTRIBUTIONS:")
                for feature, fit_info in list(patterns["distributions"].items())[:5]:
                    print(f"     → {feature}: {fit_info['distribution'].title()}")
                    print(
                        f"       ∘ AIC: {fit_info['aic']:.2f}, KS p-value: {fit_info['ks_pvalue']:.3f}"
                    )
                    if "_detailed" in fit_info and fit_info["_detailed"].get(
                        "alternatives"
                    ):
                        alt_names = [
                            alt["distribution"]
                            for alt in fit_info["_detailed"]["alternatives"]
                        ]
                        print(f"       ∘ Alt fits: {', '.join(alt_names)}")

        # Feature Engineering Suggestions
        if "feature_engineering" in self._results_cache:
            suggestions = self._results_cache["feature_engineering"]
            details = suggestions.get("_detailed", {})
            print(f"\n🛠️  FEATURE ENGINEERING RECOMMENDATIONS")
            print("-" * 45)

            # Transformations
            feature_suggestions = {
                k: v
                for k, v in suggestions.items()
                if k != "interactions" and not k.startswith("_")
            }
            if feature_suggestions:
                print("   🔄 RECOMMENDED TRANSFORMATIONS:")
                for feature, transforms in list(feature_suggestions.items())[:5]:
                    print(f"     → {feature}")
                    stats_info = (
                        details.get("transformations", {})
                        .get(feature, {})
                        .get("stats", {})
                    )
                    if stats_info:
                        print(
                            f"       ∘ Skew: {stats_info.get('skewness', 0):.2f} | "
                            f"Kurtosis: {stats_info.get('kurtosis', 0):.2f} | "
                            f"Outliers: {stats_info.get('outliers_pct', 0):.1f}%"
                        )
                    for i, transform in enumerate(transforms[:3], 1):
                        print(f"       {i}. {transform}")

            # Feature Importance
            if "feature_ranking" in details:
                ranking = details["feature_ranking"]
                if ranking:
                    print("\n   🏆 TOP RANKED FEATURES:")
                    for name, score in list(ranking.items())[:5]:
                        if name != "shap_importance":
                            print(f"     → {name}: importance = {score:.4f}")
                    if "shap_importance" in ranking:
                        print("     ∘ SHAP importance available.")

            # Interactions
            if "interactions" in suggestions:
                interactions = suggestions["interactions"]
                print(f"\n   🤝 INTERACTION FEATURES: {len(interactions)} suggested")
                for i, interaction in enumerate(interactions[:10], 1):
                    print(f"     {i}. {interaction}")
                if len(interactions) > 10:
                    print(f"     ...and {len(interactions)-10} more")

        # Time Series Insights (if applicable)
        if self.time_col and "timeseries" in self._results_cache:
            ts_results = self._results_cache["timeseries"]
            print(f"\n⏰ TIME SERIES ANALYSIS")
            print("-" * 50)

            # STATIONARITY
            if "stationarity" in ts_results and not ts_results["stationarity"].empty:
                stationarity = ts_results["stationarity"]
                if "is_stationary" in stationarity.columns:
                    stationary_mask = stationarity["is_stationary"]
                elif "consensus_stationary" in stationarity.columns:
                    stationary_mask = stationarity["consensus_stationary"]
                else:
                    stationary_mask = (
                        stationarity["is_stationary_adf"]
                        & stationarity["is_stationary_kpss"]
                    )

                stationary_count = stationary_mask.sum()
                total_features = len(stationarity)

                print(f"📈 Stationarity:")
                print(
                    f"   → {stationary_count}/{total_features} features are stationary"
                )

                # Breakdown by type
                if "stationarity_type" in stationarity.columns:
                    type_counts = stationarity["stationarity_type"].value_counts()
                    for typ, count in type_counts.items():
                        print(f"     ∘ {typ.replace('_', ' ').capitalize()}: {count}")

                # Highlight non-stationary features
                non_stationary = stationarity[~stationary_mask]
                if not non_stationary.empty:
                    print(
                        f"\n   ⚠️ Features that may require differencing or transformation:"
                    )
                    for _, row in non_stationary.iterrows():
                        print(
                            f"     - {row['feature']} (ADF p={row.get('adf_pvalue', np.nan):.4f}, "
                            f"KPSS p={row.get('kpss_pvalue', np.nan):.4f})"
                        )

            # LAG SUGGESTIONS
            if "lag_suggestions" in ts_results and ts_results["lag_suggestions"]:
                print(f"\n🔁 Lag Feature Suggestions:")
                for feature, suggestion in list(ts_results["lag_suggestions"].items())[
                    :5
                ]:
                    rec_lags = suggestion.get("recommended_lags", [])
                    seasonal = suggestion.get("seasonal_lags", [])
                    method = suggestion.get("lag_selection_method", "unknown")
                    print(
                        f"   → {feature}: Lags {rec_lags} "
                        f"{'(seasonal: ' + str(seasonal) + ')' if seasonal else ''} "
                        f"[{method}]"
                    )

            if (
                "change_point_detection" in ts_results
                and len(ts_results["change_point_detection"]) > 0
            ):
                print(f"\n🪓 Change Point Detection:")
                for col, methods in ts_results["change_point_detection"].items():
                    print(f"   → {col}:")
                    for method, result in methods.items():
                        if method == "cusum" and result.get("significant"):
                            ts = result["change_point"]
                            print(
                                f"     ∘ CUSUM break at {ts} (Δ={result['magnitude']:.3f})"
                            )
                        if method == "page_hinkley":
                            for cp in result["change_points"][:2]:
                                print(f"     ∘ Page-Hinkley shift at {cp['timestamp']}")

            if (
                "regime_switching" in ts_results
                and len(ts_results["regime_switching"]) > 0
            ):
                print(f"\n🔁 Regime Switching:")
                for col, result in ts_results["regime_switching"].items():
                    rs = result.get("markov_switching")
                    if rs:
                        print(
                            f"   → {col}: {rs['n_regimes']} regimes detected (current: {rs['current_regime']})"
                        )
                    vs = result.get("volatility_switching")
                    if vs:
                        print(
                            f"     ∘ Volatility regime: {vs['current_regime']} (switches: {vs['regime_switches']})"
                        )

            if (
                "forecasting_readiness" in ts_results
                and len(ts_results["forecasting_readiness"]) > 0
            ):
                print(f"\n📈 Forecasting Readiness:")
                fr_scores = ts_results["forecasting_readiness"]
                sorted_fr = sorted(
                    fr_scores.items(), key=lambda x: -x[1]["overall_score"]
                )
                for name, val in sorted_fr[:3]:
                    print(
                        f"   → {name}: {val['readiness_level']} (score={val['overall_score']:.2f})"
                    )
                    if val["recommendations"]:
                        print(
                            f"     ∘ Suggestions: {', '.join(val['recommendations'])}"
                        )

            if (
                "causality_analysis" in ts_results
                and len(ts_results["causality_analysis"]) > 0
            ):
                print(f"\n📣 Granger Causality:")
                for pair, result in ts_results["causality_analysis"].items():
                    if result["var1_causes_var2"]["significant"]:
                        print(
                            f"   → {pair}: {pair.split('_vs_')[0]} causes {pair.split('_vs_')[1]} (p={result['var1_causes_var2']['p_value']:.4f})"
                        )
                    if result["var2_causes_var1"]["significant"]:
                        print(
                            f"   → {pair}: {pair.split('_vs_')[1]} causes {pair.split('_vs_')[0]} (p={result['var2_causes_var1']['p_value']:.4f})"
                        )

        if "clusters" in self._results_cache:
            cluster_data = self._results_cache["clusters"]
            print(f"\n🧩 CLUSTERING INSIGHTS")
            print("-" * 40)

            if not cluster_data or "error" in cluster_data:
                error_msg = (
                    cluster_data.get("error", "Unknown error")
                    if isinstance(cluster_data, dict)
                    else "No clustering results available"
                )
                print(f"   ⚠️  {error_msg}")
            else:
                # Summary statistics
                if "summary" in cluster_data:
                    summary = cluster_data["summary"]
                    print(f"   📊 ANALYSIS SUMMARY:")
                    print(
                        f"     → Methods attempted: {summary.get('methods_attempted', 'N/A')}"
                    )
                    print(
                        f"     → Successful methods: {summary.get('successful_methods', 'N/A')}"
                    )
                    if summary.get("best_method"):
                        print(f"     ⭐ Best method: {summary['best_method'].upper()}")

                # Data characteristics
                if "data_characteristics" in cluster_data:
                    chars = cluster_data["data_characteristics"]
                    print(f"\n   📈 DATA CHARACTERISTICS:")
                    print(f"     → Samples: {chars.get('n_samples', 'N/A'):,}")
                    print(f"     → Features: {chars.get('n_features', 'N/A'):,}")

                    # Data quality indicators
                    variance = chars.get("data_variance", 0)
                    spread = chars.get("data_spread", 0)
                    if variance > 0:
                        print(f"     → Data variance: {variance:.3f}")
                    if spread > 0:
                        print(f"     → Data range: {spread:.3f}")

                # Optimal k analysis
                if "optimal_k_analysis" in cluster_data:
                    k_info = cluster_data["optimal_k_analysis"]
                    print(f"\n   🎯 OPTIMAL CLUSTERS ANALYSIS:")
                    print(
                        f"     → Estimated optimal k: {k_info.get('optimal_k', 'N/A')}"
                    )
                    print(
                        f"     → Confidence level: {k_info.get('confidence', 'N/A').title()}"
                    )

                    if "methods" in k_info and k_info["methods"]:
                        print(
                            f"     → Method agreement: {k_info.get('method_agreement', 'N/A')} different values"
                        )
                        method_results = k_info["methods"]
                        print(f"     → Individual estimates:")
                        for method, k_val in method_results.items():
                            print(f"        ∘ {method.title()}: {k_val}")

                # Preprocessing information
                if "preprocessing_info" in cluster_data:
                    prep = cluster_data["preprocessing_info"]
                    print(f"\n   🔧 PREPROCESSING APPLIED:")
                    print(
                        f"     → Scaling method: {prep.get('scaling_method', 'unknown').replace('_', ' ').title()}"
                    )

                    outlier_pct = prep.get("outlier_percentage", 0)
                    if outlier_pct > 0:
                        print(
                            f"     → Outliers detected: {prep.get('outliers_detected', 0)} ({outlier_pct:.1f}%)"
                        )

                    if prep.get("pca_applied", False):
                        var_exp = prep.get("pca_variance_explained", 0)
                        final_dims = prep.get("final_dimensions", "N/A")
                        print(
                            f"     → PCA dimensionality reduction: {final_dims} components ({var_exp:.1%} variance)"
                        )

                    if prep.get("curse_of_dimensionality_risk", False):
                        print(f"     ⚠️ High dimensionality risk detected")

                # Individual clustering results
                if "clustering_results" in cluster_data:
                    clustering_results = cluster_data["clustering_results"]
                    evaluations = cluster_data.get("evaluations", {})

                    print(f"\n   🔍 CLUSTERING METHODS RESULTS:")

                    # Group methods by type
                    method_groups = {
                        "Centroid-based": [],
                        "Hierarchical": [],
                        "Density-based": [],
                        "Probabilistic": [],
                        "Spectral & Others": [],
                        "Ensemble": [],
                    }

                    for method_name, result in clustering_results.items():
                        method_type = result.get("method_type", "unknown")
                        if "centroid" in method_type or method_name == "kmeans":
                            method_groups["Centroid-based"].append(
                                (method_name, result)
                            )
                        elif "hierarchical" in method_type:
                            method_groups["Hierarchical"].append((method_name, result))
                        elif "density" in method_type:
                            method_groups["Density-based"].append((method_name, result))
                        elif (
                            "probabilistic" in method_type or "bayesian" in method_type
                        ):
                            method_groups["Probabilistic"].append((method_name, result))
                        elif method_type == "ensemble":
                            method_groups["Ensemble"].append((method_name, result))
                        else:
                            method_groups["Spectral & Others"].append(
                                (method_name, result)
                            )

                    # Display results by group
                    for group_name, methods in method_groups.items():
                        if not methods:
                            continue

                        print(f"\n     📂 {group_name}:")
                        for method_name, result in methods:
                            print(f"       🔹 {method_name.upper()}:")

                            # Basic cluster information
                            n_clusters = result.get(
                                "n_clusters", result.get("best_k", 0)
                            )
                            if n_clusters > 0:
                                print(f"         → Clusters found: {n_clusters}")

                            # Noise points for density-based methods
                            if "noise_points" in result:
                                noise_count = result["noise_points"]
                                noise_pct = (
                                    evaluations.get(method_name, {}).get(
                                        "noise_ratio", 0
                                    )
                                    * 100
                                )
                                print(
                                    f"         → Noise points: {noise_count} ({noise_pct:.1f}%)"
                                )

                            # Quality scores
                            eval_data = evaluations.get(method_name, {})
                            if "silhouette_score" in eval_data:
                                sil_score = eval_data["silhouette_score"]
                                if sil_score > 0.7:
                                    quality_indicator = "Excellent 🟢"
                                elif sil_score > 0.5:
                                    quality_indicator = "Good 🟡"
                                elif sil_score > 0.25:
                                    quality_indicator = "Fair 🟠"
                                else:
                                    quality_indicator = "Poor 🔴"
                                print(
                                    f"         → Silhouette: {sil_score:.3f} ({quality_indicator})"
                                )

                            # Additional quality metrics
                            if "cluster_balance" in eval_data:
                                balance = eval_data["cluster_balance"]
                                print(f"         → Cluster balance: {balance:.3f}")

                            if "separation_ratio" in eval_data:
                                separation = eval_data["separation_ratio"]
                                print(f"         → Separation ratio: {separation:.2f}")

                            # Method-specific information
                            if (
                                method_name == "kmeans"
                                and "stability_score" in eval_data
                            ):
                                stability = eval_data["stability_score"]
                                print(f"         → Stability: {stability:.3f}")

                            if (
                                method_name == "hierarchical"
                                and "best_linkage_method" in result
                            ):
                                linkage_method = result["best_linkage_method"]
                                print(f"         → Best linkage: {linkage_method}")

                            if "gmm" in method_name and "model_comparison" in result:
                                best_bic = result.get("best_bic", 0)
                                cov_type = result.get("covariance_type", "unknown")
                                print(
                                    f"         → Best BIC: {best_bic:.2f} ({cov_type} covariance)"
                                )

                            if (
                                "bayesian" in method_name
                                and "effective_components" in result
                            ):
                                eff_comp = result["effective_components"]
                                print(f"         → Effective components: {eff_comp}")

                            if (
                                method_name == "ensemble"
                                and "participating_methods" in result
                            ):
                                n_methods = len(result["participating_methods"])
                                print(f"         → Consensus from {n_methods} methods")

                            # Cluster size distribution
                            if "cluster_sizes" in result and result["cluster_sizes"]:
                                sizes = result["cluster_sizes"]
                                if isinstance(sizes, dict):
                                    sum(sizes.values())
                                    largest_cluster = max(sizes.values())
                                    smallest_cluster = min(sizes.values())
                                    print(
                                        f"         → Size range: {smallest_cluster}-{largest_cluster} points"
                                    )

                                    # Show distribution for small number of clusters
                                    if len(sizes) <= 5:
                                        size_str = ", ".join(
                                            [
                                                f"C{k}:{v}"
                                                for k, v in sorted(sizes.items())
                                            ]
                                        )
                                        print(f"         → Distribution: {size_str}")

                            # Uncertainty for probabilistic methods
                            if "mean_assignment_entropy" in eval_data:
                                entropy = eval_data["mean_assignment_entropy"]
                                uncertainty = eval_data.get("assignment_uncertainty", 0)
                                print(
                                    f"         → Assignment entropy: {entropy:.3f} ±{uncertainty:.3f}"
                                )

                # Top recommendations
                if (
                    "recommendations" in cluster_data
                    and cluster_data["recommendations"]
                ):
                    print(f"\n   💡 KEY RECOMMENDATIONS:")
                    for i, rec in enumerate(
                        cluster_data["recommendations"][:4], 1
                    ):  # Show top 4
                        # Clean up recommendation formatting
                        clean_rec = (
                            rec.replace("🏆", "")
                            .replace("⚠️", "")
                            .replace("✅", "")
                            .replace("📊", "")
                            .replace("🌳", "")
                            .replace("🤝", "")
                            .strip()
                        )
                        if clean_rec.startswith("Best method:"):
                            print(f"     {i}. 🏆 {clean_rec}")
                        elif (
                            "noise" in clean_rec.lower()
                            or "uncertainty" in clean_rec.lower()
                        ):
                            print(f"     {i}. ⚠️ {clean_rec}")
                        elif (
                            "evidence" in clean_rec.lower()
                            or "confident" in clean_rec.lower()
                        ):
                            print(f"     {i}. ✅ {clean_rec}")
                        else:
                            print(f"     {i}. 💭 {clean_rec}")

                # Performance summary
                if "evaluations" in cluster_data:
                    successful_methods = len(
                        [
                            e
                            for e in cluster_data["evaluations"].values()
                            if "silhouette_score" in e
                        ]
                    )
                    total_methods = len(cluster_data.get("clustering_results", {}))

                    # Calculate average silhouette across all methods
                    silhouette_scores = [
                        e["silhouette_score"]
                        for e in cluster_data["evaluations"].values()
                        if "silhouette_score" in e and e["silhouette_score"] > 0
                    ]

                    print(f"\n   📊 PERFORMANCE SUMMARY:")
                    print(
                        f"     → Methods evaluated: {successful_methods}/{total_methods}"
                    )

                    if silhouette_scores:
                        avg_sil = np.mean(silhouette_scores)
                        best_sil = max(silhouette_scores)
                        print(f"     → Average silhouette: {avg_sil:.3f}")
                        print(f"     → Best silhouette: {best_sil:.3f}")

                        # Overall clustering quality assessment
                        if best_sil > 0.7:
                            overall_quality = (
                                "Excellent clustering structure detected 🎯"
                            )
                        elif best_sil > 0.5:
                            overall_quality = "Good clustering structure found 👍"
                        elif best_sil > 0.25:
                            overall_quality = "Moderate clustering structure present 🤔"
                        else:
                            overall_quality = "Weak clustering structure detected 😐"

                        print(f"     → Overall assessment: {overall_quality}")

        # Outlier Insights
        if "outliers" in self._results_cache:
            outlier_data = self._results_cache["outliers"]
            print(f"\n🚨 OUTLIER DETECTION SUMMARY")
            print("-" * 40)

            if not outlier_data or "error" in outlier_data:
                error_msg = (
                    outlier_data.get("error", "Unknown error")
                    if isinstance(outlier_data, dict)
                    else "No outlier results available"
                )
                print(f"   ⚠️  {error_msg}")
            else:
                # Analysis summary
                if "summary" in outlier_data:
                    summary = outlier_data["summary"]
                    print(f"   📊 ANALYSIS OVERVIEW:")
                    print(
                        f"     → Methods attempted: {summary.get('methods_attempted', 'N/A')}"
                    )
                    print(
                        f"     → Successful methods: {summary.get('successful_methods', 'N/A')}"
                    )
                    print(
                        f"     → Overall outlier rate: {summary.get('overall_outlier_rate', 0):.1%}"
                    )
                    if summary.get("best_method"):
                        print(f"     ⭐ Best method: {summary['best_method'].upper()}")

                # Data characteristics
                if "data_characteristics" in outlier_data:
                    chars = outlier_data["data_characteristics"]
                    print(f"\n   📈 DATA CHARACTERISTICS:")
                    print(
                        f"     → Total samples: {chars.get('total_samples', 'N/A'):,}"
                    )
                    print(
                        f"     → Analyzed samples: {chars.get('analyzed_samples', 'N/A'):,}"
                    )

                    missing_samples = chars.get("missing_samples", 0)
                    if missing_samples > 0:
                        missing_pct = (
                            missing_samples / chars.get("total_samples", 1) * 100
                        )
                        print(
                            f"     → Missing data: {missing_samples:,} samples ({missing_pct:.1f}%)"
                        )

                    print(f"     → Features analyzed: {chars.get('n_features', 'N/A')}")
                    print(
                        f"     → Target contamination: {chars.get('contamination_rate', 0):.1%}"
                    )

                # Preprocessing information
                if "preprocessing_info" in outlier_data:
                    prep = outlier_data["preprocessing_info"]
                    print(f"\n   🔧 PREPROCESSING APPLIED:")
                    print(
                        f"     → Scaling method: {prep.get('scaling_method', 'unknown').replace('_', ' ').title()}"
                    )
                    print(
                        f"     → Missing data strategy: {prep.get('handling_strategy', 'unknown').replace('_', ' ').title()}"
                    )

                    skewness = prep.get("data_skewness", 0)
                    if skewness > 0:
                        if skewness > 3:
                            skew_desc = "Highly skewed 🔴"
                        elif skewness > 1.5:
                            skew_desc = "Moderately skewed 🟡"
                        else:
                            skew_desc = "Low skewness 🟢"
                        print(f"     → Data skewness: {skewness:.2f} ({skew_desc})")

                    cond_num = prep.get("condition_number")
                    if cond_num:
                        if cond_num < 100:
                            cond_desc = "Well-conditioned 🟢"
                        elif cond_num < 1000:
                            cond_desc = "Acceptable 🟡"
                        else:
                            cond_desc = "Poorly conditioned 🔴"
                        print(f"     → Data condition: {cond_num:.1e} ({cond_desc})")

                # Individual method results
                if "outlier_results" in outlier_data:
                    outlier_results = outlier_data["outlier_results"]
                    evaluations = outlier_data.get("evaluations", {})

                    print(f"\n   🔍 DETECTION METHODS RESULTS:")

                    # Group methods by type
                    method_groups = {
                        "Statistical": [],
                        "Distance-based": [],
                        "Machine Learning": [],
                        "Advanced": [],
                        "Ensemble": [],
                    }

                    for method_name, result in outlier_results.items():
                        method_type = result.get("method_type", "unknown")
                        if method_type == "statistical":
                            method_groups["Statistical"].append((method_name, result))
                        elif "distance" in method_type or "density" in method_type:
                            method_groups["Distance-based"].append(
                                (method_name, result)
                            )
                        elif method_type in [
                            "ensemble",
                            "boundary_based",
                            "covariance_based",
                        ]:
                            method_groups["Machine Learning"].append(
                                (method_name, result)
                            )
                        elif method_type in ["histogram_based", "cluster_based"]:
                            method_groups["Advanced"].append((method_name, result))
                        elif method_type == "ensemble":
                            method_groups["Ensemble"].append((method_name, result))
                        else:
                            method_groups["Advanced"].append((method_name, result))

                    # Track for consensus calculation
                    consensus_votes = np.zeros(
                        outlier_data["data_characteristics"].get("total_samples", 0)
                    )
                    valid_methods = 0

                    # Display results by group
                    for group_name, methods in method_groups.items():
                        if not methods:
                            continue

                        print(f"\n     📂 {group_name}:")
                        for method_name, result in methods:
                            print(f"       🔹 {method_name.replace('_', ' ').title()}:")

                            # Basic detection statistics
                            count = result.get("count", 0)
                            pct = result.get("percentage", 0)
                            print(f"         → Outliers found: {count:,} ({pct:.2f}%)")

                            # Add to consensus
                            if "outliers" in result:
                                consensus_votes += result["outliers"].astype(int)
                                valid_methods += 1

                            # Quality metrics
                            eval_data = evaluations.get(method_name, {})
                            if "score_separation" in eval_data:
                                separation = eval_data["score_separation"]
                                if separation > 1.0:
                                    sep_desc = "Excellent 🟢"
                                elif separation > 0.5:
                                    sep_desc = "Good 🟡"
                                else:
                                    sep_desc = "Moderate 🟠"
                                print(
                                    f"         → Separation quality: {separation:.3f} ({sep_desc})"
                                )

                            if "isolation_score" in eval_data:
                                isolation = eval_data["isolation_score"]
                                print(f"         → Isolation score: {isolation:.3f}")

                            if "score_overlap" in eval_data:
                                overlap = eval_data["score_overlap"]
                                print(f"         → Score overlap: {overlap:.3f}")

                            # Method-specific details
                            if method_name == "knn_distance" and "k" in result:
                                print(f"         → K neighbors: {result['k']}")

                            if method_name == "dbscan" and "eps" in result:
                                print(f"         → Epsilon: {result['eps']:.3f}")

                            if "one_class_svm" in method_name and "kernel" in result:
                                print(f"         → Kernel: {result['kernel']}")

                            if (
                                method_name == "ensemble"
                                and "participating_methods" in result
                            ):
                                n_methods = len(result["participating_methods"])
                                print(f"         → Combined {n_methods} methods")

                                # Show consensus strength if available
                                if (
                                    "consensus_strength" in result
                                    and len(result["consensus_strength"]) > 0
                                ):
                                    avg_consensus = np.mean(
                                        result["consensus_strength"]
                                    )
                                    if avg_consensus < 0.3:
                                        consensus_desc = "High agreement 🎯"
                                    elif avg_consensus < 0.5:
                                        consensus_desc = "Moderate agreement 👍"
                                    else:
                                        consensus_desc = "Low agreement 🤔"
                                    print(
                                        f"         → Method consensus: {avg_consensus:.3f} ({consensus_desc})"
                                    )

                # Consensus analysis
                if valid_methods >= 2:
                    print(f"\n   🎯 CONSENSUS ANALYSIS:")

                    for threshold in [2, 3, max(2, valid_methods // 2)]:
                        if threshold <= valid_methods:
                            consensus_outliers = (consensus_votes >= threshold).sum()
                            consensus_pct = (
                                consensus_outliers / len(consensus_votes) * 100
                                if len(consensus_votes) > 0
                                else 0
                            )

                            if threshold == 2:
                                confidence_desc = "Moderate confidence"
                            elif threshold == 3:
                                confidence_desc = "High confidence"
                            else:
                                confidence_desc = "Very high confidence"

                            print(
                                f"     → ≥{threshold} methods agree: {consensus_outliers:,} outliers ({consensus_pct:.2f}%) - {confidence_desc}"
                            )

                    # Show most reliable outliers
                    max_votes = (
                        int(consensus_votes.max()) if len(consensus_votes) > 0 else 0
                    )
                    if max_votes > 0:
                        unanimous = (consensus_votes == max_votes).sum()
                        print(
                            f"     → Unanimous detection: {unanimous:,} outliers ({unanimous/len(consensus_votes)*100:.2f}%)"
                        )

                # Top recommendations
                if (
                    "recommendations" in outlier_data
                    and outlier_data["recommendations"]
                ):
                    print(f"\n   💡 KEY RECOMMENDATIONS:")
                    for i, rec in enumerate(
                        outlier_data["recommendations"][:4], 1
                    ):  # Show top 4
                        # Clean up recommendation formatting
                        clean_rec = (
                            rec.replace("🏆", "")
                            .replace("⚠️", "")
                            .replace("✅", "")
                            .replace("📊", "")
                            .replace("🔍", "")
                            .strip()
                        )
                        if clean_rec.startswith("Best method:"):
                            print(f"     {i}. 🏆 {clean_rec}")
                        elif "high" in clean_rec.lower() and (
                            "rate" in clean_rec.lower()
                            or "outlier" in clean_rec.lower()
                        ):
                            print(f"     {i}. ⚠️ {clean_rec}")
                        elif (
                            "excellent" in clean_rec.lower()
                            or "good" in clean_rec.lower()
                        ):
                            print(f"     {i}. ✅ {clean_rec}")
                        elif (
                            "skewed" in clean_rec.lower()
                            or "robust" in clean_rec.lower()
                        ):
                            print(f"     {i}. 🔧 {clean_rec}")
                        else:
                            print(f"     {i}. 💭 {clean_rec}")

                # Performance summary
                if "evaluations" in outlier_data:
                    evaluations = outlier_data["evaluations"]
                    successful_evals = len(
                        [e for e in evaluations.values() if "error" not in e]
                    )

                    # Calculate quality metrics across methods
                    separation_scores = [
                        e.get("score_separation", 0)
                        for e in evaluations.values()
                        if "score_separation" in e and e["score_separation"] > 0
                    ]

                    outlier_rates = [
                        r.get("percentage", 0) / 100 for r in outlier_results.values()
                    ]

                    print(f"\n   📊 DETECTION QUALITY SUMMARY:")
                    print(
                        f"     → Methods evaluated: {successful_evals}/{len(outlier_results)}"
                    )

                    if separation_scores:
                        avg_separation = np.mean(separation_scores)
                        best_separation = max(separation_scores)
                        print(f"     → Average separation: {avg_separation:.3f}")
                        print(f"     → Best separation: {best_separation:.3f}")

                    if outlier_rates:
                        rate_std = np.std(outlier_rates)
                        if rate_std < 0.02:
                            consistency_desc = "Very consistent 🎯"
                        elif rate_std < 0.05:
                            consistency_desc = "Consistent 👍"
                        else:
                            consistency_desc = "Variable 📊"
                        print(
                            f"     → Rate consistency: σ={rate_std:.3f} ({consistency_desc})"
                        )

                    # Overall assessment
                    if separation_scores and max(separation_scores) > 1.0:
                        overall_quality = "Excellent outlier detection quality 🌟"
                    elif separation_scores and max(separation_scores) > 0.5:
                        overall_quality = "Good outlier detection quality 👍"
                    elif successful_evals >= len(outlier_results) * 0.7:
                        overall_quality = "Satisfactory detection coverage 📊"
                    else:
                        overall_quality = "Limited detection reliability ⚠️"

                    print(f"     → Overall assessment: {overall_quality}")
        # Missingness Analysis
        if "missingness" in self._results_cache:
            missing = self._results_cache["missingness"]
            print(f"\n📉 MISSINGNESS ANALYSIS")
            print("-" * 45)

            # Missing rate summary
            if "missing_rate" in missing and not missing["missing_rate"].empty:
                print("   ❗ MISSING VALUE RATES:")
                for col, rate in missing["missing_rate"].items():
                    print(f"     → {col}: {rate:.1%}")

            # MNAR diagnostics
            if "missing_vs_target" in missing and missing["missing_vs_target"]:
                print("\n   🧪 MNAR DEPENDENCE ON TARGET:")
                for col, stats in missing["missing_vs_target"].items():
                    if "ttest_p" in stats:
                        print(
                            f"     → {col}: t-test p = {stats['ttest_p']:.4f}, KS p = {stats['ks_p']:.4f}, AUC = {stats.get('auc', 0):.2f}"
                        )
                    elif "chi2_p" in stats:
                        print(
                            f"     → {col}: Chi² p = {stats['chi2_p']:.4f}, "
                            f"Cramér's V = {stats.get('cramers_v', 0):.2f}, "
                            f"MI = {stats.get('mutual_info', 0):.3f}"
                        )
                    if stats.get("suggested_mnar"):
                        print("       ∘ ⚠️ Likely MNAR (target-dependent missingness)")

            # Correlated missingness (Pearson)
            if "missingness_correlation" in missing:
                corr = missing["missingness_correlation"]
                upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
                corr_pairs = corr.where(upper).stack().sort_values(ascending=False)
                corr_pairs = corr_pairs[corr_pairs > 0.3]
                if not corr_pairs.empty:
                    print("\n   🔗 CORRELATED MISSINGNESS (Pearson > 0.3):")
                    for (col1, col2), val in corr_pairs[:5].items():
                        print(f"     → {col1} ↔ {col2}: corr = {val:.2f}")

            # Jaccard similarity (optional)
            if "missingness_jaccard" in missing:
                jac = missing["missingness_jaccard"].copy()
                upper = np.triu(np.ones_like(jac, dtype=bool), k=1)
                jac_pairs = jac.where(upper).stack().sort_values(ascending=False)
                jac_pairs = jac_pairs[jac_pairs > 0.1]
                if not jac_pairs.empty:
                    print("\n   🧬 JACCARD SIMILARITY (Missing Co-occurrence > 0.1):")
                    for (col1, col2), val in jac_pairs[:5].items():
                        print(f"     → {col1} ∩ {col2}: Jaccard = {val:.2f}")

            # Clustering results
            if "missingness_clusters" in missing:
                cluster_map = missing["missingness_clusters"]
                print("\n   🧩 MISSINGNESS CLUSTERS:")
                from collections import defaultdict

                cluster_dict = defaultdict(list)
                for col, cluster_id in cluster_map.items():
                    cluster_dict[cluster_id].append(col)
                for cid, members in sorted(cluster_dict.items()):
                    print(
                        f"     • Cluster {cid}: {', '.join(members[:5])}"
                        + (f" (+{len(members) - 5} more)" if len(members) > 5 else "")
                    )

        # Dimensionality Reduction Summary
        if "dimensionality" in self._results_cache:
            dimred = self._results_cache["dimensionality"]
            print(f"\n🧮 DIMENSIONALITY REDUCTION")
            print("-" * 40)

            if not dimred or "error" in dimred:
                error_msg = (
                    dimred.get("error", "Unknown error")
                    if isinstance(dimred, dict)
                    else "No dimensionality results available"
                )
                print(f"   ⚠️  {error_msg}")
            else:
                # Data characteristics summary
                if "data_characteristics" in dimred:
                    chars = dimred["data_characteristics"]
                    print(f"   📊 DATA CHARACTERISTICS:")
                    print(f"     → Samples: {chars.get('n_samples', 'N/A'):,}")
                    print(f"     → Features: {chars.get('n_features', 'N/A'):,}")
                    print(
                        f"     → Effective rank: {chars.get('effective_rank', 'N/A')}"
                    )

                    # Data quality indicators
                    cond_num = chars.get("condition_number", 0)
                    if cond_num > 0:
                        if cond_num < 100:
                            quality = "Excellent 🟢"
                        elif cond_num < 1000:
                            quality = "Good 🟡"
                        else:
                            quality = "Poor (ill-conditioned) 🔴"
                        print(f"     → Data quality: {quality} (cond: {cond_num:.1e})")

                # Preprocessing info
                if "preprocessing_info" in dimred:
                    prep = dimred["preprocessing_info"]
                    print(f"\n   🔧 PREPROCESSING APPLIED:")
                    print(
                        f"     → Scaling method: {prep.get('scaling_method', 'unknown').replace('_', ' ').title()}"
                    )
                    if prep.get("features_removed", 0) > 0:
                        print(
                            f"     → Features removed (low variance): {prep['features_removed']}"
                        )
                    if "sampling_method" in prep:
                        print(f"     → Sampling: {prep['sampling_method'].title()}")

                # Method results
                if "embeddings" in dimred:
                    embeddings = dimred["embeddings"]
                    print(f"\n   🔍 REDUCTION METHODS SUCCESSFULLY APPLIED:")

                    # Separate linear and nonlinear methods
                    linear_methods = []
                    nonlinear_methods = []

                    for method, result in embeddings.items():
                        if result.get("method_type") == "linear":
                            linear_methods.append((method, result))
                        else:
                            nonlinear_methods.append((method, result))

                    # Display linear methods
                    if linear_methods:
                        print("     📐 Linear Methods:")
                        for method, result in linear_methods:
                            emb = result["embedding"]
                            print(f"       → {method.upper()}: {emb.shape}")

                            # Show explained variance for PCA-like methods
                            if "total_variance_explained" in result:
                                var_exp = result["total_variance_explained"]
                                print(f"         ∘ Variance explained: {var_exp:.1%}")

                            # Show sample values for 2D embeddings
                            if emb.shape[1] == 2 and emb.shape[0] > 0:
                                print(
                                    f"         ∘ Sample point: [{emb[0,0]:.2f}, {emb[0,1]:.2f}]"
                                )
                            else:
                                print(f"         ∘ Components: {emb.shape[1]}")

                    # Display nonlinear methods
                    if nonlinear_methods:
                        print("     🌀 Nonlinear Methods:")
                        for method, result in nonlinear_methods:
                            emb = result["embedding"]
                            print(f"       → {method.upper()}: {emb.shape}")

                            # Show method-specific parameters
                            if "perplexity" in result:
                                print(f"         ∘ Perplexity: {result['perplexity']}")
                            if "n_neighbors" in result and result["n_neighbors"]:
                                print(f"         ∘ Neighbors: {result['n_neighbors']}")

                            # Show sample values for 2D embeddings
                            if emb.shape[1] == 2 and emb.shape[0] > 0:
                                print(
                                    f"         ∘ Sample point: [{emb[0,0]:.2f}, {emb[0,1]:.2f}]"
                                )

                # Quality evaluation
                if "evaluation" in dimred and dimred["evaluation"]:
                    print(f"\n   📈 QUALITY EVALUATION:")
                    evaluations = dimred["evaluation"]

                    # Find best method based on combined score
                    best_method = None
                    best_score = -float("inf")

                    for method, metrics in evaluations.items():
                        if metrics:  # Only consider methods with evaluation metrics
                            score = 0
                            score_components = []

                            if "silhouette_score" in metrics:
                                sil_score = metrics["silhouette_score"]
                                score += sil_score * 0.4
                                score_components.append(f"silhouette: {sil_score:.3f}")

                            if "neighborhood_preservation" in metrics:
                                neigh_score = metrics["neighborhood_preservation"]
                                score += neigh_score * 0.4
                                score_components.append(
                                    f"preservation: {neigh_score:.3f}"
                                )

                            if "embedding_stability" in metrics:
                                stab_score = metrics["embedding_stability"]
                                score += stab_score * 0.2
                                score_components.append(f"stability: {stab_score:.3f}")

                            print(
                                f"     → {method.upper()}: {', '.join(score_components)}"
                            )

                            if score > best_score:
                                best_score = score
                                best_method = method

                    if best_method:
                        print(f"     ⭐ Best performing: {best_method.upper()}")

                # Recommendations
                if "recommendations" in dimred and dimred["recommendations"]:
                    print(f"\n   💡 RECOMMENDATIONS:")
                    for i, rec in enumerate(
                        dimred["recommendations"][:3], 1
                    ):  # Show top 3 recommendations
                        print(f"     {i}. {rec}")

                # Summary statistics
                total_methods = len(dimred.get("embeddings", {}))
                successful_evals = len(
                    [m for m in dimred.get("evaluation", {}).values() if m]
                )
                print(
                    f"\n   📝 SUMMARY: {total_methods} methods applied, {successful_evals} evaluated for quality"
                )

        # Data Quality Assessment
        print(f"\n✅ DATA QUALITY ASSESSMENT")
        print("-" * 40)

        completeness = (
            1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])
        ) * 100
        uniqueness = len(self.df.drop_duplicates()) / len(self.df) * 100

        print(f"   📊 Completeness: {completeness:.1f}%")
        print(f"   🔍 Uniqueness: {uniqueness:.1f}%")

        if self._categorical_cols:
            print(f"   📋 Categorical Feature Cardinality:")
            for col in self._categorical_cols[:5]:
                unique_count = self.df[col].nunique()
                unique_pct = unique_count / len(self.df) * 100
                print(f"     → {col}: {unique_count} unique values ({unique_pct:.1f}%)")

        overall_quality = (completeness + uniqueness) / 2
        quality_status = (
            "Excellent"
            if overall_quality > 95
            else (
                "Good"
                if overall_quality > 85
                else "Fair" if overall_quality > 70 else "Poor"
            )
        )
        quality_emoji = (
            "🟢"
            if overall_quality > 95
            else (
                "🟡" if overall_quality > 85 else "🟠" if overall_quality > 70 else "🔴"
            )
        )

        print(
            f"   {quality_emoji} Overall Quality Score: {overall_quality:.1f}% ({quality_status})"
        )

    # ========================================================================
    # EXTENSIBILITY METHODS
    # ========================================================================

    def register_strategy(self, strategy: AnalysisStrategy):
        """Register a custom analysis strategy"""
        self._strategies[strategy.name] = strategy

    def register_hook(self, event: str, callback: Callable):
        """Register a custom hook"""
        self.hooks.register_hook(event, callback)

    def register_plotter(self, analysis_type: str, plotter: Callable):
        """Register a custom plotter"""
        self.hooks.register_plotter(analysis_type, plotter)

    def get_results(self, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached results for an analysis type"""
        return self._results_cache.get(analysis_type)

    def get_available_analyses(self) -> List[str]:
        """Get list of available analysis types"""
        return list(self._strategies.keys())
