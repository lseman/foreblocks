"""ForeMiner — comprehensive dataset analysis."""

import logging
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from .analyzers.cluster import ClusterAnalyzer
from .analyzers.correlation import CorrelationAnalyzer
from .analyzers.dimension import DimensionalityAnalyzer
from .analyzers.distribution import DistributionAnalyzer
from .analyzers.feat import FeatureEngineeringAnalyzer
from .analyzers.graph import GraphAnalyzer
from .analyzers.group import CategoricalGroupAnalyzer
from .analyzers.missing import MissingnessAnalyzer
from .analyzers.outlier import OutlierAnalyzer
from .analyzers.pattern import PatternDetector
from .analyzers.ts import TimeSeriesAnalyzer
from .core import (
    OPTIONAL_IMPORTS,
    AnalysisConfig,
    AnalysisHooks,
    AnalysisStrategy,
    _run_analysis_worker,
)
from .plotting import PlotHelper
from .report import DatasetReportPrinter

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================


class DatasetAnalyzer:
    """Clean, extensible dataset analyzer with all advanced functionality."""

    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str | None = None,
        config: AnalysisConfig | None = None,
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.time_col = time_col
        self.config = config or AnalysisConfig()
        self.verbose = verbose

        self.logger = logging.getLogger(self.__class__.__name__)
        level = logging.INFO if self.verbose else logging.WARNING
        level = getattr(logging, str(self.config.log_level).upper(), level)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        self.hooks = AnalysisHooks()
        self.plot_helper = PlotHelper()
        self._strategies: dict[str, AnalysisStrategy] = {}
        self._results_cache: dict[str, Any] = {}

        self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        self._register_default_strategies()
        self._register_default_plotters()
        self._setup()

    def _setup(self):
        PlotHelper.setup_style(self.config)
        self._setup_time_index()
        if self.verbose:
            self.logger.info(
                f"🔍 Initialized analyzer with {self.df.shape[0]:,} rows × {self.df.shape[1]} columns"
            )
            self.logger.info(f"   • Numeric features: {len(self._numeric_cols)}")
            self.logger.info(
                f"   • Categorical features: {len(self._categorical_cols)}"
            )

    def _setup_time_index(self):
        if self.time_col and self.time_col in self.df.columns:
            self.df.set_index(self.time_col, inplace=True)
            self.df.index = pd.to_datetime(self.df.index, errors="coerce")
            if self.df.index.hasnans:
                self._log(
                    f"Warning: time column '{self.time_col}' contains invalid dates"
                )

    def _register_default_strategies(self):
        strategies = [
            DistributionAnalyzer(),
            CorrelationAnalyzer(),
            OutlierAnalyzer(),
            ClusterAnalyzer(),
            DimensionalityAnalyzer(),
            PatternDetector(),
            MissingnessAnalyzer(),
            FeatureEngineeringAnalyzer(),
            CategoricalGroupAnalyzer(),
            GraphAnalyzer(),
        ]
        if self.time_col:
            from .analyzers.ts import TimeSeriesAnalyzer

            strategies.append(TimeSeriesAnalyzer())

        for strategy in strategies:
            self._strategies[strategy.name] = strategy

    def _register_default_plotters(self):
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

    def _log(self, message: str, level: str = "info"):
        if not self.verbose:
            return
        log_fn = getattr(self.logger, level, self.logger.info)
        log_fn(f"🔍 {message}")

    def _resolve_analysis_types(self, analysis_types: list[str] | None) -> list[str]:
        requested = analysis_types or list(self._strategies.keys())
        resolved = []
        for analysis_type in requested:
            if analysis_type not in self._strategies:
                self._log(f"Skipping unknown analysis type: {analysis_type}")
                continue
            resolved.append(analysis_type)
        return resolved

    def _build_hook_context(
        self, analysis_type: str, result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        context = {"data": self.df, "config": self.config, "type": analysis_type}
        if result is not None:
            context["result"] = result
        return context

    def _get_numeric_series(
        self,
        column: str,
        *,
        min_length: int = 1,
        error_template: str = "Series too short (minimum {minimum} points)",
    ) -> pd.Series:
        if column not in self._numeric_cols:
            raise ValueError(f"Column {column} not found in numeric columns")
        series = self.df[column].dropna()
        if len(series) < min_length:
            raise ValueError(error_template.format(minimum=min_length))
        return series

    def _get_analysis_result(
        self, name: str, key: str | None = None, default: Any = None
    ) -> Any:
        if name not in self._results_cache:
            try:
                self._results_cache[name] = self._strategies[name].analyze(
                    self.df, self.config
                )
            except Exception as e:
                self._results_cache[name] = {"error": str(e)}
        result = self._results_cache[name]
        if key is None:
            return result
        if isinstance(result, dict):
            return result.get(key, default)
        return default

    @lru_cache(maxsize=32)
    def _get_clean_numeric_data(self) -> pd.DataFrame:
        return self.df[self._numeric_cols].dropna()

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def analyze(self, analysis_types: list[str] | None = None) -> dict[str, Any]:
        """Run specified analyses in parallel with pre/post hooks."""
        resolved_types = self._resolve_analysis_types(analysis_types)
        results = {}
        started = time.perf_counter()
        max_workers = getattr(self.config, "max_workers", None)

        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for analysis_type in resolved_types:
                self._log(f"Starting {analysis_type} analysis...")
                context = self._build_hook_context(analysis_type)
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
                    self.logger.error(f"❌ {strategy_name} analysis failed:\n{error}")
                    continue
                self.logger.info(f"✅ {strategy_name} analysis complete")
                results[strategy_name] = result
                self._results_cache[strategy_name] = result
                self.hooks.trigger(
                    f"post_{strategy_name}",
                    self._build_hook_context(strategy_name, result),
                )

        elapsed = time.perf_counter() - started
        self._log(
            f"Completed {len(results)}/{len(resolved_types)} analyses in {elapsed:.2f}s"
        )
        return results

    def plot(self, analysis_types: list[str] | None = None):
        """Generate plots for analysis results."""
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

    def analyze_and_plot(self, analysis_types: list[str] | None = None):
        """Run comprehensive analysis and generate all plots."""
        results = self.analyze(analysis_types)
        self.plot(analysis_types)
        return results

    def analyze_everything(self, plot: bool = False):
        """Run the complete comprehensive analysis workflow."""
        self._log("🚀 Starting comprehensive dataset analysis...")
        results = self.analyze()
        if hasattr(self, "print_insights"):
            self.print_insights()
        if plot:
            self.plot()
            if "correlations" in results and OPTIONAL_IMPORTS["networkx"]:
                try:
                    self.plot_correlation_network()
                except Exception as e:
                    self.logger.warning(f"Network plot failed: {e}")
        self._log("🎉 Comprehensive analysis complete!")
        return results

    # ========================================================================
    # SPECIALIZED METHODS
    # ========================================================================

    def decompose_series(self, column: str, period: int | None = None):
        """STL decomposition for time series."""
        series = self._get_numeric_series(
            column,
            min_length=24,
            error_template="Series too short for decomposition (minimum 24 points)",
        )
        if period is None:
            period = min(max(2, len(series) // 10), 24)
        stl = STL(series, period=period, seasonal=7)
        decomposition = stl.fit()
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=axes[0], title=f"Original Series: {column}")
        decomposition.trend.plot(ax=axes[1], title="Trend", color="orange")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonal", color="green")
        decomposition.resid.plot(ax=axes[3], title="Residual", color="red")
        plt.tight_layout()
        plt.show()
        return decomposition

    def plot_autocorrelations(self, column: str, lags: int = 40):
        """Enhanced ACF and PACF plots."""
        series = self._get_numeric_series(column, min_length=10)
        if len(series) < lags + 10:
            lags = max(10, len(series) // 3)
        acf_vals = acf(series, nlags=lags, fft=True)
        pacf_vals = pacf(series, nlags=lags, method="ols")
        n = len(series)
        ci = 1.96 / np.sqrt(n)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        x = range(len(acf_vals))
        ax1.stem(x, acf_vals, basefmt=" ")
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax1.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
        ax1.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
        ax1.fill_between(x, -ci, ci, alpha=0.2, color="red")
        ax1.set_title(f"Autocorrelation Function - {column}")
        ax1.set_xlabel("Lags")
        ax1.set_ylabel("ACF")
        ax2.stem(x, pacf_vals, basefmt=" ")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
        ax2.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
        ax2.fill_between(x, -ci, ci, alpha=0.2, color="red")
        ax2.set_title(f"Partial Autocorrelation Function - {column}")
        ax2.set_xlabel("Lags")
        ax2.set_ylabel("PACF")
        plt.tight_layout()
        plt.show()

    def plot_correlation_network(
        self, method: str = "pearson", threshold: float | None = None
    ):
        """Create correlation network graph."""
        if "correlations" not in self._results_cache:
            self._log("Running correlation analysis first...")
            self.analyze(["correlations"])
        correlations = self._results_cache["correlations"]
        PlotHelper.plot_correlation_network(correlations, self.df, self.config)

    def explain_features(
        self, model, X: pd.DataFrame | None = None, max_display: int = 10
    ):
        """SHAP feature explanation."""
        try:
            import shap
        except ImportError:
            self._log("SHAP not available.")
            return {}

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
    # EXTENSIBILITY METHODS
    # ========================================================================

    def register_strategy(self, strategy: AnalysisStrategy):
        """Register a custom analysis strategy."""
        self._strategies[strategy.name] = strategy

    def register_hook(self, event: str, callback: Callable):
        """Register a custom hook."""
        self.hooks.register_hook(event, callback)

    def register_plotter(self, analysis_type: str, plotter: Callable):
        """Register a custom plotter."""
        self.hooks.register_plotter(analysis_type, plotter)

    def get_results(self, analysis_type: str) -> dict[str, Any] | None:
        """Get cached results for an analysis type."""
        return self._results_cache.get(analysis_type)

    def get_available_analyses(self) -> list[str]:
        """Get list of available analysis types."""
        return list(self._strategies.keys())

    def analyze_intelligent_summary(self) -> str:
        """Highest level SOTA mining: Combine all analyses into a summary."""
        self._log("🧠 Generating intelligent SOTA summary...")
        results = self.analyze()
        lines = [
            f"ForeMiner SOTA Intelligence Report ({self.df.shape[0]} rows)",
            "=" * 60,
        ]
        ts = results.get("timeseries", {})
        patt = results.get("patterns", {})
        motifs = ts.get("motif_discovery", {})
        if motifs:
            cols = list(motifs.keys())
            lines.append(
                f"• Motifs: Found recurring patterns in {len(cols)} features: {', '.join(cols[:3])}"
            )
        causality = patt.get("causality", {})
        infl = causality.get("directed_influence", [])
        if infl:
            top = sorted(infl, key=lambda x: x["strength"], reverse=True)[0]
            lines.append(
                f"• Causality: {top['source']} -> {top['target']} (strength={top['strength']:.3f})"
            )
        anomalies = patt.get("anomalies", {})
        global_count = len(anomalies.get("global_anomalies", []))
        if global_count > 0:
            lines.append(
                f"• Anomalies: {global_count} global anomaly points identified."
            )
        if self.verbose:
            print("\n".join(lines))
        return "\n".join(lines)
