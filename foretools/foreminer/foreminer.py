import warnings
from collections import Counter
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
from .analyze_graph import GraphAnalyzer
from .analyze_group import CategoricalGroupAnalyzer
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
            CategoricalGroupAnalyzer(),
            GraphAnalyzer(),
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

        # # Generate all plots
        # self.plot()

        # # Special network plot if available
        # if "correlations" in results and OPTIONAL_IMPORTS["networkx"]:
        #     try:
        #         self.plot_correlation_network()
        #     except Exception as e:
        #         print(f"Network plot failed: {e}")

        # self._log("🎉 Comprehensive analysis complete!")
        # return results

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
        """Generate a comprehensive, well-formatted dataset analysis report (cleaner/DRY)."""

        # ----------------------- Formatting helpers -----------------------
        def section(title, level=1):
            bars = {1: "="*80, 2: "-"*50, 3: "-"*30, 4: "-"*30}
            prefixes = {1: "📊 ", 2: "🔍 ", 3: "🔎 ", 4: "   🔍 "}
            indent = "" if level < 4 else "   "
            print()
            if level == 1:
                print(bars[1]); print(f"{prefixes[1]}{title.upper()}"); print(bars[1])
            elif level == 2:
                print(f"{prefixes[2]}{title.upper()}"); print(bars[2])
            elif level == 3:
                print(f"{prefixes[3]}{title}"); print("   " + bars[3])
            else:
                print(f"{prefixes[4]}{title}"); print("      " + bars[4])

        def metric(label, value, unit="", status=None, indent=0):
            status_emoji = {"excellent":"🟢","good":"🟡","fair":"🟠","poor":"🔴","warning":"⚠️","info":"ℹ️"}
            emoji = status_emoji.get(status, "")
            print(f"{'   '*indent}• {label}: {value}{unit} {emoji}")

        def rec(text, priority="normal", indent=0):
            icons = {"high":"🚨","medium":"⚠️","low":"💡","normal":"💡"}
            print(f"{'   '*indent}{icons.get(priority,'💡')} {text}")

        def pct(num, den, zeros_as="0.0%"):
            if not den: return zeros_as
            return f"{(num/den)*100:.1f}%"

        def safe_get(name, key=None, default=None):
            """Get a cached analysis result, running it if needed."""
            if name not in self._results_cache:
                try:
                    self._results_cache[name] = self._strategies[name].analyze(self.df, self.config)
                except Exception as e:
                    self._results_cache[name] = {"error": str(e)}
            res = self._results_cache[name]
            return res if key is None else res.get(key, default)

        def top_list(items, n=5):
            items = list(items)
            return ", ".join(items[:n]) + (f" ... and {len(items)-n} more" if len(items) > n else "")

        # Tunable thresholds in one place
        TH = {
            "missing": (1, 5, 15),            # excellent/good/fair/p poor %
            "corr":    (0.7, 0.5),            # strong, moderate
            "skew_hi": 2.0,
            "outlier_hi_pct": 5.0,
            "bimodal_bc": 0.55,
            "sil_good": 0.5,
            "sil_excellent": 0.7,
        }

        # ----------------------- Executive summary -----------------------
        section("EXECUTIVE SUMMARY", 1)
        print("Dataset Overview:")
        metric("Shape", f"{self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        metric("Memory Usage", f"{self.df.memory_usage(deep=True).sum()/(1024**2):.2f} MB")
        metric("Numeric Features", len(self._numeric_cols))
        metric("Categorical Features", len(self._categorical_cols))

        miss_pct = (self.df.isna().sum().sum() / (self.df.shape[0]*self.df.shape[1] or 1)) * 100
        miss_status = "excellent" if miss_pct < TH["missing"][0] else \
                    "good" if miss_pct < TH["missing"][1] else \
                    "fair" if miss_pct < TH["missing"][2] else "poor"
        metric("Missing Values", f"{miss_pct:.2f}%", status=miss_status)

        # Warm up the analyses you rely on later (lazy-run + cache)
        for a in ["distributions","correlations","outliers","patterns","clusters","timeseries","missingness","feature_engineering","dimensionality"]:
            safe_get(a)  # ignore error here; handled when reading

        # ----------------------- Distribution analysis -----------------------
        dist_summary = safe_get("distributions","summary", pd.DataFrame())
        if not dist_summary.empty:
            section("DISTRIBUTION ANALYSIS", 1)

            total_features = len(dist_summary)
            gaussian = int(dist_summary.get("is_gaussian", False).sum())
            skewed = int(dist_summary.get("is_skewed", False).sum())
            heavy = int(dist_summary.get("is_heavy_tailed", False).sum())
            avg_out_pct = float(dist_summary.get("outlier_pct_z>3", pd.Series([0])).mean() or 0)

            print("Statistical Properties:")
            metric("Normal Distributions", f"{gaussian}/{total_features} ({pct(gaussian,total_features)})")
            metric("Skewed Distributions", f"{skewed}/{total_features} ({pct(skewed,total_features)})")
            metric("Heavy-Tailed", f"{heavy}/{total_features} ({pct(heavy,total_features)})")
            metric("Average Outlier Rate", f"{avg_out_pct:.2f}%")

            section("Feature Quality Assessment", 3)
            if gaussian:
                feats = dist_summary.loc[dist_summary["is_gaussian"], "feature"].tolist()
                print(f"✅ Normal Distributions ({gaussian}):")
                print(f"   {top_list(feats)}")
                rec("Safe for parametric methods and linear models")

            hi_skew = dist_summary.loc[dist_summary["skewness"].abs() > TH["skew_hi"]]
            if not hi_skew.empty:
                print(f"\n⚠️ Highly Skewed Features ({len(hi_skew)}):")
                for _, r in hi_skew.head(3).iterrows():
                    direction = "right" if r["skewness"] > 0 else "left"
                    print(f"   • {r['feature']}: {r['skewness']:.2f} ({direction}-skewed)")
                rec("Apply log/sqrt transform or use robust methods", "medium")

            hi_out = dist_summary.loc[dist_summary.get("outlier_pct_z>3", 0) > TH["outlier_hi_pct"]]
            if not hi_out.empty:
                print(f"\n🚨 High Outlier Rate Features ({len(hi_out)} > {TH['outlier_hi_pct']}%):")
                for _, r in hi_out.head(3).iterrows():
                    print(f"   • {r['feature']}: {r['outlier_pct_z>3']:.1f}% outliers")
                rec("Apply robust scaling or outlier treatment", "high")

            bi = dist_summary.loc[dist_summary.get("bimodality_coeff", 0) > TH["bimodal_bc"], "feature"]
            if not bi.empty:
                print(f"\n🔀 Potential Bimodal Distributions ({len(bi)}):")
                print(f"   {top_list(bi.tolist(), 3)}")
                rec("Investigate mixtures or stratify data")

            section("Preprocessing Recommendations", 3)
            normal_pct = gaussian / total_features * 100 if total_features else 0
            if normal_pct > 70: rec("Dataset largely normal — parametric methods recommended")
            elif normal_pct > 30: rec("Mixed distributions — use a hybrid approach")
            else: rec("Non-normal dominant — consider robust/non-parametric methods")
            if skewed > total_features * 0.5: rec("Many skewed features — batch transform", "medium")

        # ----------------------- Correlation analysis -----------------------
        corrs = safe_get("correlations") or {}
        if isinstance(corrs, dict) and "pearson" in corrs and corrs["pearson"] is not None:
            section("CORRELATION ANALYSIS", 1)
            cm = corrs["pearson"]
            strong_pos, strong_neg, mod = [], [], []
            strong, moderate = TH["corr"]

            cols = cm.columns
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    v = cm.iloc[i, j]; f1, f2 = cols[i], cols[j]
                    if v > strong: strong_pos.append((f1,f2,v))
                    elif v < -strong: strong_neg.append((f1,f2,v))
                    elif abs(v) > moderate: mod.append((f1,f2,v))

            print("Correlation Summary:")
            metric("Strong Positive (>0.7)", len(strong_pos))
            metric("Strong Negative (<-0.7)", len(strong_neg))
            metric("Moderate (0.5–0.7)", len(mod))

            if strong_pos:
                section("Strong Positive Correlations", 3)
                for f1,f2,v in strong_pos[:5]:
                    metric(f"{f1} ↔ {f2}", f"{v:.3f}")
            if strong_neg:
                section("Strong Negative Correlations", 3)
                for f1,f2,v in strong_neg[:5]:
                    metric(f"{f1} ↔ {f2}", f"{v:.3f}")
            if strong_pos or strong_neg:
                rec("Consider dimensionality reduction or feature selection", "medium")
    
        # ----------------------- Enhanced Pattern detection -----------------------
        patt = safe_get("patterns") or {}
        if patt:
            section("ADVANCED PATTERN DETECTION", 1)

            # Feature types
            ft = patt.get("feature_types", {})
            if ft:
                section("Feature Type Classification", 3)
                for ftype, features in ft.items():
                    if not features:
                        continue
                    label = ftype.replace("_", " ").title()
                    metric(label, f"{len(features)} features")
                    print(f"   {top_list(features)}")
                if ft.get("seasonal"):
                    section("Temporal Patterns", 3)
                    print(f"🕐 Seasonal: {top_list(ft['seasonal'])}")
                    rec("Engineer time features & use seasonal models")
                if ft.get("transformable_to_normal"):
                    section("Transformation Opportunities", 3)
                    print(f"🔄 Transformable: {top_list(ft['transformable_to_normal'])}")
                    rec("Apply Box-Cox or Yeo-Johnson")

            # Relationship patterns
            rel = patt.get("relationships", {})
            if rel:
                section("Relationship Patterns", 3)

                # Spec for each pattern type
                pattern_specs = {
                    "nonlinear": {
                        "label": "🌀 Non-Linear Relationships",
                        "fields": [("nonlinearity_score", "score", 3)],
                        "extras": lambda r: (
                            f"Best fit: {r['functional_form'].title()} (R²={r['functional_r2']:.3f})"
                            if r.get("functional_form") != "none" and r.get("functional_r2", 0) > 0.3 else None
                        ),
                        "rec": "Try polynomial/kernels or specific functional forms"
                    },
                    "complex": {
                        "label": "🧬 Complex Dependencies (High MI, Low Linear)",
                        "fields": [("mutual_info", "MI", 3)],
                        "extras": lambda r: [
                            f"Ensemble strength: {r['ensemble_score']:.3f}" if r.get("ensemble_score") else None,
                            f"Copula dependence: {r['copula_tau']:.3f}" if r.get("copula_tau", 0) > 0.2 else None
                        ],
                        "rec": "Explore interactions, non-linear models, or copula-based approaches"
                    },
                    "regime_switching": {
                        "label": "🔄 Regime-Switching Relationships",
                        "fields": [("regime_diff", "regime diff", 3)],
                        "extras": lambda r: [
                            f"Low regime corr: {r['regime1_corr']:.3f}",
                            f"High regime corr: {r['regime2_corr']:.3f}",
                            f"Split at: {r['split_point']:.2f}",
                            f"Strength: {r.get('regime_strength', 'unknown')}"
                        ],
                        "rec": "Consider regime-aware models, mixture models, or threshold effects"
                    },
                    "functional_forms": {
                        "label": "📐 Detected Functional Relationships",
                        "fields": [("r2_score", "R²", 3)],
                        "extras": lambda r: [
                            f"Complexity: {r.get('complexity', 'moderate')}",
                            (
                                f"Alternatives: {', '.join([f'{k}({v:.2f})' 
                                                            for k, v in r.get('all_forms', {}).items() 
                                                            if k != r['functional_form'] and v > r['r2_score'] - 0.1][:2])}"

                            )
                        ],
                        "rec": "Use detected functional forms for feature engineering or model selection"
                    },
                    "ensemble_strong": {
                        "label": "🎯 Strong Multi-Method Dependencies",
                        "fields": [("ensemble_score", "ensemble", 3)],
                        "extras": lambda r: [
                            f"Pearson: {r.get('pearson', 0):.3f}",
                            f"Spearman: {r.get('spearman', 0):.3f}",
                            f"Distance: {r['distance_corr']:.3f}" if r.get("distance_corr", 0) > 0 else None,
                            f"MI: {r['mutual_info']:.3f}" if r.get("mutual_info", 0) > 0 else None,
                            f"Strength: {r.get('strength', 'strong')}"
                        ],
                        "rec": "High-confidence relationships - prioritize for modeling"
                    },
                    "tail_dependence": {
                        "label": "🎭 Tail Dependence Patterns",
                        "fields": [("tail_type", "type", None)],
                        "extras": lambda r: [
                            f"Upper tail: {r['upper_tail_dep']:.3f}",
                            f"Lower tail: {r.get('lower_tail_dep', 0):.3f}" if r.get("lower_tail_dep", 0) > 0 else None,
                            (
                                f"Asymmetric: +{r['asymmetric_pos']:.3f}, -{r['asymmetric_neg']:.3f}"
                                if max(r.get("asymmetric_pos", 0), r.get("asymmetric_neg", 0)) > 0.02 else None
                            )
                        ],
                        "rec": "Consider copulas, extreme value models, or tail-aware approaches"
                    },
                    "copula_dependence": {
                        "label": "🔗 Copula-Based Dependencies",
                        "fields": [("kendall_tau", "τ", 3)],
                        "extras": lambda r: [
                            f"Type: {r.get('dependence_type', 'body')} dependence",
                            f"Tail coefficient: {r.get('tail_dep', 0):.3f}" if r.get("tail_dep", 0) > 0 else None
                        ],
                        "rec": "Consider copula models for capturing rank-based dependencies"
                    },
                    "monotonic": {
                        "label": "📈 Monotonic Relationships",
                        "fields": [("kendall_tau", "τ", 3)],
                        "extras": lambda r: [
                            f"Type: {r.get('relationship_type', 'monotonic')}",
                            f"Spearman: {r.get('spearman', 0):.3f}" if r.get("spearman", 0) > 0 else None
                        ],
                        "rec": "Monotonic transforms, rank-based methods, or ordinal approaches"
                    },
                    "distance_corr": {
                        "label": "🌐 Distance Correlation Patterns",
                        "fields": [("distance_corr", "dcor", 3)],
                        "extras": lambda r: [
                            f"vs Pearson: {r['pearson']:.3f}",
                            f"Strength: {r.get('strength', 'moderate')}"
                        ],
                        "rec": "Non-linear dependence detected - try kernel methods or GAMs"
                    }
                }

                # Generic printer
                for ptype, spec in pattern_specs.items():
                    if not rel.get(ptype):
                        continue
                    print(f"\n{spec['label']}:")
                    for r in rel[ptype][:3]:
                        # Main metric
                        f1, f2 = r['feature1'], r['feature2']
                        for field, label, prec in spec["fields"]:
                            if field in r:
                                val = r[field]
                                fmt_val = f"{val:.{prec}f}" if isinstance(val, (int, float)) and prec else str(val)
                                metric(f"{f1} ↔ {f2}", f"{label}: {fmt_val}")
                                break
                        # Extras
                        extras = spec["extras"](r)
                        if extras:
                            if isinstance(extras, str):
                                extras = [extras]
                            for ex in extras:
                                if ex:
                                    print(f"     └─ {ex}")
                    rec(spec["rec"])


            # Best-fit distributions (enhanced)
            fits = patt.get("distributions", {})
            if fits:
                section("Statistical Distribution Fitting", 3)
                for feat, info in list(fits.items())[:5]:
                    ks = info.get("ks_pvalue", 0)
                    aic = info.get("aic", float('nan'))
                    distribution = info.get("distribution", "unknown")
                    
                    # Enhanced quality assessment
                    if ks > 0.1:
                        quality, status = "Excellent", "excellent"
                        quality_emoji = "✅"
                    elif ks > 0.05:
                        quality, status = "Good", "good"
                        quality_emoji = "✅"
                    elif ks > 0.01:
                        quality, status = "Fair", "fair"
                        quality_emoji = "⚠️"
                    else:
                        quality, status = "Poor", "poor"
                        quality_emoji = "❌"
                    
                    print(f"   • {feat}: {distribution.title()} {quality_emoji}")
                    metric("AIC", f"{aic:.2f}" if not np.isnan(aic) else "N/A", indent=1)
                    metric("KS p-value", f"{ks:.4f}", status=status, indent=1)
                    print(f"     └─ Fit Quality: {quality}")
                    
                    # Add parameter info if available
                    params = info.get("parameters", {})
                    if params:
                        param_str = ", ".join([f"{k}={v:.2f}" for k, v in list(params.items())[:3]])
                        print(f"     └─ Parameters: {param_str}")
                
                rec("Use fitted distributions for simulation, anomaly detection, or Bayesian priors")

            # NEW: Summary statistics and recommendations
            total_patterns = sum(len(patterns) for patterns in rel.values() if patterns)
            if total_patterns > 0:
                section("Pattern Summary", 3)
                metric("Total Relationship Patterns", str(total_patterns))
                
                # Count pattern types
                pattern_counts = {k: len(v) for k, v in rel.items() if v}
                top_pattern_types = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                
                print("   Most Common Pattern Types:")
                for pattern_type, count in top_pattern_types:
                    pattern_label = pattern_type.replace("_", " ").title()
                    print(f"   • {pattern_label}: {count} relationships")
                
                # Advanced recommendations
                print("\n📋 Advanced Modeling Recommendations:")
                
                if rel.get("regime_switching"):
                    print("   • Consider regime-switching models (Markov switching, threshold VAR)")
                if rel.get("functional_forms"):
                    print("   • Leverage detected functional forms for feature engineering")
                if rel.get("tail_dependence") or rel.get("copula_dependence"):
                    print("   • Explore copula-based models for dependency modeling")
                if rel.get("ensemble_strong"):
                    print("   • High-confidence relationships - prioritize for interaction terms")
                if total_patterns > 10:
                    print("   • Rich relationship structure - consider ensemble methods")
                if any(r.get("ensemble_score", 0) > 0.7 for patterns in rel.values() for r in patterns):
                    print("   • Very strong dependencies detected - check for potential data leakage")

        # ----------------------- Outlier detection -----------------------
        out = safe_get("outliers") or {}
        section("OUTLIER DETECTION ANALYSIS", 1)
        if not out or "error" in out:
            print(f"⚠️ {out.get('error','No outlier results') if isinstance(out, dict) else 'No outlier results'}")
        else:
            # Summary
            summary = out.get("summary", {})
            if summary:
                print("Analysis Overview:")
                metric("Methods Attempted", summary.get("methods_attempted","N/A"))
                metric("Successful Methods", summary.get("successful_methods","N/A"))
                rate = summary.get("overall_outlier_rate", 0.0)
                status = "warning" if rate > 0.10 else "good" if rate > 0.05 else "excellent"
                metric("Overall Outlier Rate", f"{rate:.1%}", status=status)
                if summary.get("best_method"): metric("Best Method", summary["best_method"].upper())

            # Preprocessing info
            chars = out.get("data_characteristics", {})
            if chars:
                section("Analysis Scope", 3)
                metric("Total Samples", f"{chars.get('total_samples',0):,}")
                metric("Analyzed Samples", f"{chars.get('analyzed_samples',0):,}")
                metric("Features Analyzed", chars.get("n_features","N/A"))
                ms = chars.get("missing_samples", 0)
                if ms:
                    metric("Missing Data", f"{ms:,} ({(ms/(chars.get('total_samples',1)))*100:.1f}%)")

            prep = out.get("preprocessing_info", {})
            if prep:
                section("Data Preprocessing", 3)
                metric("Scaling Method", prep.get("scaling_method","unknown").replace("_"," ").title())
                metric("Missing Data Strategy", prep.get("handling_strategy","unknown").replace("_"," ").title())
                skew = prep.get("data_skewness", 0)
                if skew:
                    status = "poor" if skew > 3 else "fair" if skew > 1.5 else "good"
                    desc = "Highly skewed" if skew > 3 else "Moderately skewed" if skew > 1.5 else "Low skewness"
                    metric("Data Skewness", f"{skew:.2f} ({desc})", status=status)

            # Per-method results (compact)
            methods = out.get("outlier_results", {})
            evals = out.get("evaluations", {})
            if methods:
                section("Detailed Method Results", 2)
                for mname, res in methods.items():
                    print(f"\n   🔹 {mname.replace('_',' ').title()}:")
                    cnt, pct_val = res.get("count", 0), res.get("percentage", 0.0)
                    rate_status = "warning" if pct_val > 10 else "fair" if pct_val > 5 else "good" if pct_val > 1 else "excellent"
                    metric("Outliers Detected", f"{cnt:,} ({pct_val:.2f}%)", status=rate_status, indent=1)
                    ev = evals.get(mname, {})
                    if "score_separation" in ev:
                        sep = ev["score_separation"]
                        st = "excellent" if sep > 1.0 else "good" if sep > 0.5 else "fair"
                        metric("Separation Quality", f"{sep:.3f}", status=st, indent=1)
                    if "isolation_score" in ev:
                        metric("Isolation Score", f"{ev['isolation_score']:.3f}", indent=1)
                    if "score_overlap" in ev:
                        ov = ev["score_overlap"]
                        st = "excellent" if ov < 0.1 else "good" if ov < 0.3 else "fair"
                        metric("Score Overlap", f"{ov:.3f}", status=st, indent=1)

            # Recommendations
            recs = out.get("recommendations", [])
            if recs:
                section("Outlier Treatment Recommendations", 2)
                for i, r in enumerate(recs[:5], 1):
                    clean = r.replace("🏆","").replace("⚠️","").replace("✅","").strip()
                    pr = "high" if "best method" in clean.lower() else "medium" if "high" in clean.lower() and ("rate" in clean.lower() or "outlier" in clean.lower()) else "normal"
                    rec(f"{i}. {clean}", pr)

        # ----------------------- Clustering -----------------------
        cl = safe_get("clusters") or {}
        section("CLUSTERING ANALYSIS", 1)

        if not cl or "error" in cl:
            print(f"⚠️ {cl.get('error','No clustering results') if isinstance(cl, dict) else 'No clustering results'}")
        else:
            # --- Summary ---
            if cl.get("summary"):
                s = cl["summary"]
                print("Analysis Summary:")
                metric("Methods Attempted", s.get("methods_attempted", "N/A"))
                metric("Successful Methods", s.get("successful_methods", "N/A"))
                if s.get("best_method"):
                    bm = s["best_method"]
                    metric("Best Method", bm.upper())

            # --- Optimal k ---
            ok = cl.get("optimal_k_analysis", {})
            if ok:
                section("Optimal Cluster Analysis", 3)
                metric("Estimated Optimal K", ok.get("optimal_k", "N/A"))
                conf = ok.get("confidence", "N/A")
                metric("Confidence Level", conf.title() if isinstance(conf, str) else conf)

            # --- Method performance ---
            res = cl.get("clustering_results", {}) or {}
            evals = cl.get("evaluations", {}) or {}
            if res:
                section("Method Performance", 3)
                for mname, r in list(res.items())[:5]:
                    section(mname.replace("_", " ").title(), 4)

                    # Clusters found (do NOT use truthiness)
                    ncl = r.get("n_clusters", r.get("best_k"))
                    if ncl is not None:
                        metric("Clusters Found", ncl, indent=1)
                        if isinstance(ncl, int) and ncl <= 1:
                            print("   ⓘ Single or no cluster detected (silhouette may be undefined).")

                    # Cluster sizes
                    sizes = r.get("cluster_sizes")
                    if isinstance(sizes, dict) and len(sizes) > 0:
                        largest, smallest = max(sizes.values()), min(sizes.values())
                        # Handle edge cases safely
                        balance = (smallest / largest) if largest else 0.0
                        metric("Size Range", f"{smallest}-{largest} points", indent=1)
                        st = "excellent" if balance > 0.7 else "good" if balance > 0.5 else "fair"
                        metric("Balance Ratio", f"{balance:.2f}", status=st, indent=1)
                    else:
                        print("   ⓘ No per-cluster size breakdown available.")

                    # Silhouette (prefer evaluated; fall back to result)
                    sil = None
                    if mname in evals:
                        sil = evals[mname].get("silhouette_score")
                    if sil is None:
                        sil = r.get("silhouette")

                    if sil is not None:
                        if np.isfinite(sil):
                            st = (
                                "excellent" if sil > TH["sil_excellent"]
                                else "good" if sil > TH["sil_good"]
                                else "fair"
                            )
                            metric("Silhouette Score", f"{sil:.3f}", status=st, indent=1)
                        else:
                            print("   ⓘ Silhouette not defined for this labeling.")

                    # Method-specific extras
                    if "eps" in r:
                        metric("Epsilon (DBSCAN)", f"{r['eps']:.3f}", indent=1)
                    if "min_samples" in r:
                        metric("Min Samples (DBSCAN)", r["min_samples"], indent=1)
                    if "best_k" in r and r.get("best_k") is not None:
                        metric("Best K (internal)", r["best_k"], indent=1)

            # --- Recommendations ---
            recs = cl.get("recommendations") or []
            if recs:
                section("Recommendations", 2)
                for _rec in recs:
                    # Keep it tidy: strip emojis if you want, or leave as-is
                    print(f"💡 {_rec}")

        # ----------------------- Time series -----------------------
        ts: Dict[str, Any] = safe_get("timeseries") or {}
        if ts:

            section("TIME SERIES ANALYSIS", 1)

            stationarity_df: pd.DataFrame = ts.get("stationarity", pd.DataFrame())
            readiness: Dict[str, Any] = ts.get("forecasting_readiness", {}) or {}
            temporal: Dict[str, Any] = ts.get("temporal_patterns", {}) or {}

            # ---------- Overview ----------
            if not stationarity_df.empty:
                total = len(stationarity_df)
                stationary = int(stationarity_df["is_stationary"].sum())
                ready = sum(1 for v in readiness.values()
                            if v.get("readiness_level") in {"excellent", "good"})

                print("Analysis Overview:")
                metric("Total Series", total)
                metric("Stationary Series", f"{stationary}/{total} ({pct(stationary, total)})")
                metric("Forecast-Ready", f"{ready}/{total} ({pct(ready, total)})")

                # ---------- Stationarity ----------
                section("Stationarity Assessment", 3)
                st_feats = stationarity_df.loc[stationarity_df["is_stationary"], "feature"].tolist()
                non_st = stationarity_df.loc[~stationarity_df["is_stationary"]]

                if st_feats:
                    print(f"✅ Stationary ({len(st_feats)}):")
                    for f in st_feats[:3]:
                        row = stationarity_df[stationarity_df["feature"] == f].iloc[0]
                        metric(f, f"ADF p={row.get('adf_pvalue', np.nan):.4f}", indent=1)
                    if len(st_feats) > 3:
                        print(f"   ... and {len(st_feats)-3} more")

                if not non_st.empty:
                    print(f"\n⚠️ Non-Stationary ({len(non_st)}):")
                    for _, r in non_st.head(3).iterrows():
                        metric(r["feature"], str(r.get("stationarity_type", "unknown")), indent=1)

                # ---------- Trends & Seasonality (compact) ----------
                trends = {k: v for k, v in temporal.items() if k.endswith("_trend")}
                seas   = {k: v for k, v in temporal.items() if k.endswith("_seasonality")}

                strong_trends = []
                for k, v in trends.items():
                    if v.get("linear_significant") and v.get("trend_direction") not in {"no_trend", "stable"}:
                        r2 = v.get("linear_r_squared", 0) or 0
                        strong_trends.append((k.replace("_trend", ""), r2))

                seas_list = []
                for k, v in seas.items():
                    cls = v.get("stl_classification", "none")
                    if cls in {"moderate", "strong"}:
                        seas_list.append((k.replace("_seasonality", ""), v.get("stl_seasonal_strength", 0) or 0))

                if strong_trends or seas_list:
                    section("Detected Temporal Patterns", 3)

                if strong_trends:
                    print("📈 Significant Trends:")
                    for f, r2 in sorted(strong_trends, key=lambda x: -x[1])[:3]:
                        metric(f, f"R²={r2:.3f}", indent=1)

                if seas_list:
                    print("\n🔄 Seasonal Patterns:")
                    for f, s in sorted(seas_list, key=lambda x: -x[1])[:3]:
                        metric(f, f"strength={s:.3f}", indent=1)

                # ---------- Lag & Seasonality Suggestions (per-series + summary) ----------
                lag_sugg: Dict[str, Any] = ts.get("lag_suggestions", {}) or {}

                # Build compact per-series rows:
                #   feature -> rec_lags, seasonal_lags, stl_period, stl_strength
                lag_rows = []
                # Ensure we cover features that have either lag_suggestions or seasonality info
                candidate_feats = set(lag_sugg.keys()) | {k.replace("_seasonality", "") for k in seas.keys()}
                for f in sorted(candidate_feats):
                    ls = lag_sugg.get(f, {}) or {}
                    seas_info = temporal.get(f"{f}_seasonality", {}) if temporal else {}

                    rec_lags = ls.get("recommended_lags", []) or []
                    seas_lags = ls.get("seasonal_lags", []) or []
                    stl_period = seas_info.get("stl_period", None)
                    seas_strength = seas_info.get("stl_seasonal_strength", None)

                    # Only show if something interesting exists
                    if rec_lags or seas_lags or stl_period is not None or seas_strength is not None:
                        lag_rows.append({
                            "feature": f,
                            "rec_lags": rec_lags,
                            "seas_lags": seas_lags,
                            "stl_period": int(stl_period) if isinstance(stl_period, (int, float, np.integer)) and not pd.isna(stl_period) else None,
                            "seas_strength": float(seas_strength) if isinstance(seas_strength, (int, float)) and not pd.isna(seas_strength) else None,
                        })

                if lag_rows:
                    section("Lag & Seasonality Suggestions", 3)

                    # Rank rows: prefer those with a period, then stronger seasonality, then more lags
                    def _rank_key(r):
                        return (
                            0 if r["stl_period"] is None else -1,
                            0 if r["seas_strength"] is None else -r["seas_strength"],
                            -len(r["rec_lags"]),
                        )

                    lag_rows_sorted = sorted(lag_rows, key=_rank_key)

                    N = 6  # cap per-series prints
                    print("Per-series:")
                    for r in lag_rows_sorted[:N]:
                        parts = []
                        if r["rec_lags"]:
                            parts.append(f"lags={r['rec_lags']}")
                        if r["seas_lags"]:
                            parts.append(f"seasonal_lags={r['seas_lags']}")
                        if r["stl_period"] is not None:
                            parts.append(f"P={r['stl_period']}")
                        if r["seas_strength"] is not None:
                            parts.append(f"strength={r['seas_strength']:.3f}")
                        metric(r["feature"], " | ".join(parts) if parts else "—", indent=1)

                    if len(lag_rows_sorted) > N:
                        print(f"   ... and {len(lag_rows_sorted) - N} more")

                    # Summary of most common lags/periods
                    all_rec_lags   = [lag for r in lag_rows for lag in (r["rec_lags"] or [])]
                    all_seas_lags  = [lag for r in lag_rows for lag in (r["seas_lags"] or [])]
                    period_counts  = Counter([r["stl_period"] for r in lag_rows if r["stl_period"] is not None])
                    rec_counts     = Counter(all_rec_lags)
                    seaslag_counts = Counter(all_seas_lags)

                    summary_bits = []
                    if rec_counts:
                        most_rec = ", ".join(f"{k}({v})" for k, v in rec_counts.most_common(3))
                        summary_bits.append(f"Top lags: {most_rec}")
                    if seaslag_counts:
                        most_seas = ", ".join(f"{k}({v})" for k, v in seaslag_counts.most_common(3))
                        summary_bits.append(f"Top seasonal lags: {most_seas}")
                    if period_counts:
                        most_p = ", ".join(f"P={k}({v})" for k, v in period_counts.most_common(3))
                        summary_bits.append(f"Top periods: {most_p}")

                    if summary_bits:
                        print("\nSummary:")
                        for sline in summary_bits:
                            metric("", sline, indent=1)

                # ---------- Recommendations ----------
                if strong_trends or seas_list or not stationarity_df.empty:
                    section("Time Series Recommendations", 3)

                if not stationarity_df.empty:
                    frac_st = (stationary / max(1, len(stationarity_df))) * 100
                    if frac_st < 50:
                        rec("Apply differencing to achieve stationarity", "medium")

                fr = (ready / max(1, len(stationarity_df))) * 100 if not stationarity_df.empty else 0
                if fr > 70:
                    rec("Well-suited for forecasting")
                elif fr < 30:
                    rec("Significant preprocessing needed before forecasting", "medium")

                if seas_list:
                    rec("Use seasonal models (e.g., SARIMA, STL)")
                if strong_trends:
                    rec("Include trend terms or detrend")
                    
        # ----------------------- Feature engineering -----------------------
        fe = safe_get("feature_engineering") or {}
        if fe:
            section("FEATURE ENGINEERING RECOMMENDATIONS", 1)
            details = fe.get("_detailed", {})
            trans = details.get("transformations", {})
            inter = details.get("interactions", [])
            enc = details.get("encodings", {})
            rank = details.get("feature_ranking", {})

            print("Engineering Scope:")
            metric("Numeric Features", len(trans))
            metric("Categorical Features", len(enc))
            metric("Interaction Candidates", len(inter))

            if trans:
                section("Priority Transformations", 3)
                def score_item(st):
                    s = 0
                    skew = abs(st.get("skewness",0)); kurt = st.get("kurtosis",0); out = st.get("outliers_pct",0); cv = st.get("cv",0)
                    if skew > 2: s += 3
                    elif skew > 1: s += 2
                    if kurt > 3: s += 2
                    elif kurt < -1: s += 1
                    if out > 10: s += 3
                    elif out > 5: s += 1
                    if cv > 3: s += 1
                    return s

                scored = []
                for f, d in trans.items():
                    st = d.get("stats", {})
                    scored.append((f, score_item(st), d))
                scored.sort(key=lambda x: -x[1])
                hi = [x for x in scored if x[1] >= 5][:3]
                md = [x for x in scored if 2 <= x[1] < 5][:3]

                if hi:
                    print("🚨 High Priority:")
                    for f, _, d in hi:
                        issues = []
                        st = d.get("stats", {})
                        if abs(st.get("skewness",0)) > 2: issues.append(f"skew={st.get('skewness',0):.2f}")
                        if st.get("kurtosis",0) > 3: issues.append(f"kurtosis={st.get('kurtosis',0):.2f}")
                        if st.get("outliers_pct",0) > 5: issues.append(f"outliers={st.get('outliers_pct',0):.1f}%")
                        print(f"   📌 {f}")
                        if issues: print(f"      Issues: {', '.join(issues)}")
                        if d.get("recommended"):
                            print("      Top transforms:")
                            for i, t in enumerate(d["recommended"][:2], 1):
                                print(f"        {i}. {t}")
                if md:
                    print("\n⚠️ Medium Priority:")
                    for f, _, d in md:
                        st = d.get("stats", {})
                        issues = []
                        if abs(st.get("skewness",0)) > 1: issues.append(f"skew={st.get('skewness',0):.2f}")
                        if st.get("outliers_pct",0) > 5: issues.append(f"outliers={st.get('outliers_pct',0):.1f}%")
                        first = (d.get("recommended") or ["Standard scaling"])[0]
                        metric(f, f"{', '.join(issues)} → {first}", indent=1)

            if rank:
                section("Feature Importance Analysis", 3)
                main = {k:v for k,v in rank.items() if k != "shap_importance"}
                if main:
                    top = list(main.items())[:8]
                    hi = [(f,s) for f,s in top if s > 0.1]
                    md = [(f,s) for f,s in top if 0.05 <= s <= 0.1]
                    lo = [(f,s) for f,s in top if s < 0.05]
                    if hi:
                        print("🥇 High Value (>0.1):")
                        for f,s in hi[:3]: metric(f, f"{s:.4f}", indent=1)
                    if md:
                        print("\n🥈 Medium (0.05–0.1):")
                        for f,s in md[:3]: metric(f, f"{s:.4f}", indent=1)
                    if lo:
                        print(f"\n📉 Low Value: {len(lo)} features < 0.05 (consider removal)")

            if enc:
                section("Categorical Encoding Strategy", 3)
                simple, complex_, high = [], [], []
                for f, info in enc.items():
                    card = info.get("cardinality", 0); nullp = info.get("null_percentage", 0.0)
                    strat = (info.get("strategies") or ["OneHot"])[0]
                    if card <= 5: simple.append((f,card,strat,nullp))
                    elif card <= 20: complex_.append((f,card,strat,nullp))
                    else: high.append((f,card,strat,nullp))
                if simple:
                    print("✅ Simple (≤5):")
                    for f,c,s,n in simple[:3]:
                        metric(f, f"{c} cats{', ' + f'{n:.1f}% null' if n>0 else ''} → {s}", indent=1)
                if complex_:
                    print("\n🔄 Advanced (6–20):")
                    for f,c,s,n in complex_[:3]:
                        metric(f, f"{c} cats{', ' + f'{n:.1f}% null' if n>0 else ''} → {s}", indent=1)
                if high:
                    print("\n🔀 High Cardinality (>20):")
                    for f,c,s,_ in high[:3]:
                        metric(f, f"{c} categories → {s}", indent=1)
                        if c > 100: rec("Consider hashing or rare-category grouping", "medium", 1)

            if inter:
                section("Interaction Features", 3)
                metric("Total Candidates", len(inter))
                rec(f"Start with top {min(10,len(inter))} interactions by importance")


        # ----------------------- Missingness -----------------------
        miss = safe_get("missingness") or {}
        if miss:
            section("MISSINGNESS ANALYSIS", 1)
            rates = miss.get("missing_rate", pd.Series(dtype=float))
            if hasattr(rates, "empty") and not rates.empty:
                section("Missing Value Rates", 3)
                rates = rates.sort_values(ascending=False)
                hi = ((rates > 0.5) | (rates > 0.2)).sum()
                mid = ((rates <= 0.2) & (rates > 0.05)).sum()
                for col, r in rates.items():
                    if r <= 0: continue
                    st = "poor" if r > 0.5 else "fair" if r > 0.2 else "good" if r > 0.05 else "excellent"
                    metric(col, f"{r:.1%}", status=st)
                if hi: rec(f"{hi} features with >20% missing", "high")
                if mid: rec(f"{mid} features with 5–20% missing", "medium")

            if miss.get("missing_vs_target"):
                section("Target-Dependent Missingness (MNAR Detection)", 3)
                for col, st in miss["missing_vs_target"].items():
                    if not st.get("suggested_mnar"): continue
                    if "ttest_p" in st:
                        metric(f"{col} (numeric)", f"t-test p={st['ttest_p']:.4f}, AUC={st.get('auc',0):.2f}", indent=1)
                    elif "chi2_p" in st:
                        metric(f"{col} (categorical)", f"Chi² p={st['chi2_p']:.4f}, Cramér's V={st.get('cramers_v',0):.2f}", indent=1)
                    print("     ⚠️ Likely MNAR")
                rec("Use MNAR-aware imputation where flagged", "high")

            corr_miss = miss.get("missingness_correlation")
            if corr_miss is not None and hasattr(corr_miss, "values"):
                upper = np.triu(np.ones_like(corr_miss, dtype=bool), 1)
                pairs = corr_miss.where(upper).stack().sort_values(ascending=False)
                strong = pairs[pairs > 0.5]
                if not strong.empty:
                    section("Correlated Missingness Patterns", 3)
                    print("Strong correlations (>0.5) in missingness:")
                    for (c1,c2), v in strong[:5].items():
                        metric(f"{c1} ↔ {c2}", f"{v:.2f}", indent=1)
                    rec("Consider joint imputation for correlated missing patterns")

        # ----------------------- Categorical Group Analysis -----------------------
        group_results = safe_get("categorical_groups") or {}
        if group_results and "error" not in group_results:
            section("CATEGORICAL GROUP ANALYSIS", 1)
            
            # Group Overview
            group_info = group_results.get("group_info", {})
            if group_info:
                categorical_col = group_info.get("categorical_column", "Unknown")
                section(f"Group Overview: {categorical_col}", 3)
                
                metric("Total Groups", str(group_info.get("n_groups", 0)))
                metric("Total Samples", str(group_info.get("total_samples", 0)))
                
                # Group sizes
                group_sizes = group_info.get("group_sizes", {})
                if group_sizes:
                    print("   Group Sizes:")
                    for group, size in group_sizes.items():
                        print(f"     • {group}: {size} samples")
                
                # Balance assessment
                balanced = group_info.get("balanced_groups", True)
                balance_status = "excellent" if balanced else "warning"
                metric("Groups Balanced", "Yes" if balanced else "No", status=balance_status, indent=1)
                
                if group_info.get("missing_percentage", 0) > 10:
                    metric("Missing Data", f"{group_info['missing_percentage']:.1f}%", status="warning", indent=1)
            
            # Summary of significant findings
            summary = group_results.get("summary", {})
            if summary:
                section("Analysis Summary", 3)
                
                total_vars = summary.get("total_variables_tested", 0)
                significant_vars = summary.get("significant_variables", [])
                n_significant = len(significant_vars)
                
                metric("Variables Tested", str(total_vars))
                
                if n_significant > 0:
                    significance_status = "excellent" if n_significant > total_vars * 0.3 else "good"
                    metric("Significant Variables", f"{n_significant}/{total_vars}", status=significance_status)
                    
                    if significant_vars:
                        print("   Variables with significant differences:")
                        for var in significant_vars[:5]:  # Show top 5
                            print(f"     • {var}")
                        if len(significant_vars) > 5:
                            print(f"     • ... and {len(significant_vars) - 5} more")
                else:
                    metric("Significant Variables", "0", status="warning")
            
            # Variable-specific analyses
            variable_analyses = group_results.get("variable_analyses", {})
            if variable_analyses:
                section("Variable-by-Variable Analysis", 3)
                
                for var_name, var_results in list(variable_analyses.items())[:3]:  # Show top 3 variables
                    print(f"\n🔍 Analysis for: {var_name}")
                    
                    # Descriptive statistics summary
                    descriptives = var_results.get("descriptive_statistics", {})
                    if descriptives:
                        print("   Group Statistics:")
                        for group_name, stats in descriptives.items():
                            mean_val = stats.get("mean", 0)
                            std_val = stats.get("std", 0)
                            n_val = stats.get("n", 0)
                            print(f"     • {group_name}: μ={mean_val:.3f} (±{std_val:.3f}), n={n_val}")
                    
                    # Best test result
                    all_tests = {}
                    all_tests.update(var_results.get("parametric_tests", {}))
                    all_tests.update(var_results.get("nonparametric_tests", {}))
                    all_tests.update(var_results.get("modern_methods", {}))
                    
                    # Find most reliable significant test
                    best_test = None
                    best_reliability = 0
                    
                    for test_name, test_result in all_tests.items():
                        if isinstance(test_result, dict) and "error" not in test_result:
                            significant = test_result.get("significant", False)
                            p_value = test_result.get("p_value", 1.0)
                            assumption_met = test_result.get("assumption_met", True)
                            
                            # Calculate reliability score
                            reliability = 1.0
                            if "bootstrap" in test_name or "permutation" in test_name:
                                reliability *= 1.2
                            if "welch" in test_name:
                                reliability *= 1.1
                            if not assumption_met:
                                reliability *= 0.7
                            if significant:
                                reliability *= 1.5
                            
                            if reliability > best_reliability:
                                best_reliability = reliability
                                best_test = (test_name, test_result)
                    
                    if best_test:
                        test_name, test_result = best_test
                        method_name = test_result.get("method", test_name.replace("_", " ").title())
                        p_value = test_result.get("p_value", 1.0)
                        significant = test_result.get("significant", False)
                        
                        status = "excellent" if significant and p_value < 0.01 else "good" if significant else "fair"
                        metric("Best Test", method_name, status=status, indent=1)
                        metric("P-value", f"{p_value:.6f}", status=status, indent=2)
                        
                        # Effect size information
                        effect_info = ""
                        if "cohens_d" in test_result:
                            effect_size = test_result.get("effect_size", "unknown")
                            cohens_d = test_result["cohens_d"]
                            effect_info = f"Cohen's d = {cohens_d:.3f} ({effect_size})"
                        elif "eta_squared" in test_result:
                            effect_size = test_result.get("effect_size", "unknown")
                            eta_sq = test_result["eta_squared"]
                            effect_info = f"η² = {eta_sq:.3f} ({effect_size})"
                        elif "rank_biserial_correlation" in test_result:
                            effect_size = test_result.get("effect_size", "unknown")
                            rbc = test_result["rank_biserial_correlation"]
                            effect_info = f"r = {rbc:.3f} ({effect_size})"
                        
                        if effect_info:
                            metric("Effect Size", effect_info, indent=2)
                        
                        # Result interpretation
                        if significant:
                            print(f"     └─ ✅ Significant difference detected")
                            if "groups_compared" in test_result:
                                groups = test_result["groups_compared"]
                                print(f"     └─ Between: {' vs '.join(groups)}")
                        else:
                            print(f"     └─ ❌ No significant difference")
                    
                    # Show specific recommendations for this variable
                    var_recommendations = var_results.get("recommendations", [])
                    if var_recommendations:
                        print("   Key Insights:")
                        for _rec in var_recommendations[:2]:  # Show top 2 recommendations
                            print(f"     • {_rec}")
            
            # Categorical associations
            categorical_associations = group_results.get("categorical_associations", {})
            if categorical_associations:
                section("Categorical Variable Associations", 3)
                
                for cat_var, assoc_results in categorical_associations.items():
                    chi_square = assoc_results.get("chi_square", {})
                    if chi_square and "error" not in chi_square:
                        p_value = chi_square.get("p_value", 1.0)
                        significant = chi_square.get("significant", False)
                        cramers_v = chi_square.get("cramers_v", 0)
                        effect_size = chi_square.get("effect_size", "unknown")
                        
                        print(f"\n📊 {categorical_col} ↔ {cat_var}")
                        
                        status = "excellent" if significant and p_value < 0.01 else "good" if significant else "fair"
                        metric("Chi-square p-value", f"{p_value:.6f}", status=status, indent=1)
                        metric("Cramér's V", f"{cramers_v:.3f} ({effect_size})", indent=1)
                        
                        if significant:
                            print("     └─ ✅ Significant association detected")
                        else:
                            print("     └─ ❌ No significant association")
                        
                        # Fisher's exact test for 2x2 tables
                        fisher_exact = assoc_results.get("fisher_exact", {})
                        if fisher_exact and "error" not in fisher_exact:
                            odds_ratio = fisher_exact.get("odds_ratio", 1.0)
                            fisher_p = fisher_exact.get("p_value", 1.0)
                            metric("Fisher's Exact p-value", f"{fisher_p:.6f}", indent=1)
                            metric("Odds Ratio", f"{odds_ratio:.3f}", indent=1)
            
            # Method performance overview
            if variable_analyses:
                section("Statistical Methods Performance", 3)
                
                # Count method usage and success rates
                method_counts = {}
                method_significant = {}
                
                for var_results in variable_analyses.values():
                    all_tests = {}
                    all_tests.update(var_results.get("parametric_tests", {}))
                    all_tests.update(var_results.get("nonparametric_tests", {}))
                    all_tests.update(var_results.get("modern_methods", {}))
                    
                    for test_name, test_result in all_tests.items():
                        if isinstance(test_result, dict) and "error" not in test_result:
                            method_name = test_result.get("method", test_name)
                            method_counts[method_name] = method_counts.get(method_name, 0) + 1
                            
                            if test_result.get("significant", False):
                                method_significant[method_name] = method_significant.get(method_name, 0) + 1
                
                print("   Method Usage and Success Rates:")
                for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    significant_count = method_significant.get(method, 0)
                    success_rate = (significant_count / count * 100) if count > 0 else 0
                    print(f"     • {method}: {count} uses, {significant_count} significant ({success_rate:.0f}%)")
            
            # Overall recommendations
            recommendations = group_results.get("recommendations", [])
            if recommendations:
                section("Key Recommendations", 3)
                for i, _rec in enumerate(recommendations[:5], 1):
                    print(f"   {i}. {_rec}")
            
            # Statistical assumptions summary
            if variable_analyses:
                section("Statistical Assumptions Summary", 3)
                
                normality_violations = 0
                variance_violations = 0
                total_variables = len(variable_analyses)
                
                for var_results in variable_analyses.values():
                    assumptions = var_results.get("assumption_tests", {})
                    
                    if not assumptions.get("all_groups_normal", True):
                        normality_violations += 1
                    
                    levene_test = assumptions.get("homogeneity_tests", {}).get("levene", {})
                    if not levene_test.get("equal_variances", True):
                        variance_violations += 1
                
                # Normality assessment
                norm_pct = (total_variables - normality_violations) / total_variables * 100 if total_variables > 0 else 0
                norm_status = "excellent" if norm_pct > 80 else "good" if norm_pct > 60 else "warning"
                metric("Variables with Normal Distributions", f"{total_variables - normality_violations}/{total_variables} ({norm_pct:.0f}%)", status=norm_status)
                
                # Variance homogeneity assessment
                var_pct = (total_variables - variance_violations) / total_variables * 100 if total_variables > 0 else 0
                var_status = "excellent" if var_pct > 80 else "good" if var_pct > 60 else "warning"
                metric("Variables with Equal Variances", f"{total_variables - variance_violations}/{total_variables} ({var_pct:.0f}%)", status=var_status)
                
                # Recommendations based on assumption violations
                if normality_violations > total_variables * 0.5:
                    print("   💡 Many variables violate normality - non-parametric methods recommended")
                if variance_violations > total_variables * 0.5:
                    print("   🔧 Many variables have unequal variances - use Welch's tests")
            
            # Final summary box
            if summary.get("n_significant", 0) > 0:
                section("🎯 CONCLUSION", 3)
                n_sig = summary.get("n_significant", 0)
                total_vars = summary.get("total_variables_tested", 0)
                categorical_col = group_info.get("categorical_column", "groups")
                
                if n_sig == total_vars:
                    print(f"   🌟 ALL variables show significant differences between {categorical_col}")
                elif n_sig > total_vars * 0.5:
                    print(f"   ✅ MAJORITY of variables ({n_sig}/{total_vars}) show significant differences between {categorical_col}")
                else:
                    print(f"   📊 SOME variables ({n_sig}/{total_vars}) show significant differences between {categorical_col}")
                
                print(f"   📈 Consider using {categorical_col} as a strong predictor in your models")
            
            else:
                section("📋 CONCLUSION", 3)
                categorical_col = group_info.get("categorical_column", "groups")
                print(f"   ❌ NO significant differences found between {categorical_col}")
                print(f"   🤔 {categorical_col} may not be a useful predictor for these variables")
                
                # Suggest potential reasons
                min_group_size = group_info.get("min_group_size", 0)
                if min_group_size < 30:
                    print("   💡 Small sample sizes may limit statistical power")
                if not group_info.get("balanced_groups", True):
                    print("   ⚖️ Unbalanced groups may affect statistical power")

        elif group_results and "error" in group_results:
            section("CATEGORICAL GROUP ANALYSIS - ERROR", 1)
            print(f"❌ Analysis failed: {group_results.get('error', 'Unknown error')}")
            print("💡 Check your data quality and try again")

        else:
            # No group analysis was performed
            pass

        # ----------------------- Graph Network Analysis -----------------------
        graph_results = safe_get("graph_analysis") or {}
        if graph_results and "error" not in graph_results:
            section("GRAPH NETWORK ANALYSIS", 1)
            
            # Data Overview
            data_info = graph_results.get("data_info", {})
            if data_info:
                section("Network Construction Overview", 3)
                
                original_shape = data_info.get("original_shape", (0, 0))
                analyzed_shape = data_info.get("analyzed_shape", (0, 0))
                
                metric("Original Data", f"{original_shape[0]} samples × {original_shape[1]} features")
                metric("Analyzed Data", f"{analyzed_shape[0]} samples × {analyzed_shape[1]} features")
                
                if data_info.get("was_subsampled", False):
                    metric("Subsampling", "Applied for performance", status="warning", indent=1)
                
                constant_removed = len(data_info.get("constant_columns_removed", []))
                if constant_removed > 0:
                    metric("Constant Columns Removed", str(constant_removed), status="warning", indent=1)
            
            # Summary and Best Graph
            summary = graph_results.get("summary", {})
            if summary:
                section("Network Summary", 3)
                
                successful = summary.get("successful_constructions", 0)
                total = summary.get("total_attempted", 0)
                success_rate = (successful / total * 100) if total > 0 else 0
                
                success_status = "excellent" if success_rate == 100 else "good" if success_rate >= 80 else "warning"
                metric("Graph Construction Success", f"{successful}/{total} ({success_rate:.0f}%)", status=success_status)
                
                best_graph = summary.get("best_graph_type")
                if best_graph:
                    print(f"   🏆 Best Network Type: {best_graph.replace('_', ' ').title()}")
                    
                    # Best graph metrics
                    best_nodes = summary.get("best_graph_nodes", 0)
                    best_edges = summary.get("best_graph_edges", 0)
                    best_density = summary.get("best_graph_density", 0)
                    best_connected = summary.get("best_graph_connected", False)
                    best_clustering = summary.get("best_graph_clustering", 0)
                    
                    metric("Network Size", f"{best_nodes} nodes, {best_edges} edges", indent=1)
                    
                    density_status = "excellent" if 0.1 <= best_density <= 0.7 else "good" if best_density > 0 else "warning"
                    metric("Network Density", f"{best_density:.3f}", status=density_status, indent=1)
                    
                    connection_status = "excellent" if best_connected else "warning"
                    metric("Connectivity", "Fully Connected" if best_connected else "Disconnected", status=connection_status, indent=1)
                    
                    clustering_status = "excellent" if best_clustering > 0.5 else "good" if best_clustering > 0.3 else "fair"
                    metric("Clustering Coefficient", f"{best_clustering:.3f}", status=clustering_status, indent=1)
            
            # Network Topology Analysis
            topology_results = graph_results.get("network_topology", {})
            if topology_results and best_graph and best_graph in topology_results:
                section("Network Topology Analysis", 3)
                
                topology = topology_results[best_graph]
                if "error" not in topology:
                    
                    # Connectivity Details
                    print("🔗 Connectivity Analysis:")
                    if topology.get("is_connected", False):
                        diameter = topology.get("diameter", 0)
                        avg_path = topology.get("average_path_length", 0)
                        radius = topology.get("radius", 0)
                        
                        metric("Network Diameter", str(diameter), indent=1)
                        metric("Average Path Length", f"{avg_path:.2f}", indent=1)
                        metric("Network Radius", str(radius), indent=1)
                        
                        if avg_path <= 3:
                            print("     └─ ⚡ Efficient information flow")
                        elif avg_path > 5:
                            print("     └─ 🌉 Long communication paths")
                    else:
                        n_components = topology.get("n_components", 0)
                        largest_size = topology.get("largest_component_size", 0)
                        metric("Connected Components", str(n_components), indent=1)
                        metric("Largest Component", f"{largest_size} nodes", indent=1)
                    
                    # Small World Properties
                    if topology.get("is_small_world", False):
                        sigma = topology.get("small_world_sigma", 0)
                        print(f"\n🌍 Small World Network:")
                        metric("Small-world σ", f"{sigma:.2f}", status="excellent", indent=1)
                        print("     └─ ✅ Efficient global connectivity with local clustering")
                    
                    # Degree Statistics
                    degree_stats = topology.get("degree_stats", {})
                    if degree_stats:
                        print(f"\n📊 Degree Distribution:")
                        metric("Mean Degree", f"{degree_stats.get('mean', 0):.1f}", indent=1)
                        metric("Degree Range", f"{degree_stats.get('min', 0)} - {degree_stats.get('max', 0)}", indent=1)
                        metric("Degree Std Dev", f"{degree_stats.get('std', 0):.1f}", indent=1)
                    
                    # Assortativity
                    assortativity = topology.get("degree_assortativity")
                    if assortativity is not None:
                        assort_status = "good" if abs(assortativity) < 0.3 else "warning"
                        metric("Degree Assortativity", f"{assortativity:.3f}", status=assort_status, indent=1)
                        if assortativity > 0.1:
                            print("     └─ 📈 Similar-degree nodes tend to connect")
                        elif assortativity < -0.1:
                            print("     └─ 📉 Dissimilar-degree nodes tend to connect")
            
            # Community Detection Results
            community_results = graph_results.get("communities", {})
            if community_results and best_graph and best_graph in community_results:
                section("Community Structure Analysis", 3)
                
                communities = community_results[best_graph]
                successful_methods = [method for method in communities.keys() if "error" not in communities[method]]
                
                if successful_methods:
                    # Find best community detection method
                    best_method = None
                    best_modularity = -1
                    
                    for method in successful_methods:
                        comm_info = communities[method]
                        modularity = comm_info.get("modularity", -1)
                        if modularity > best_modularity:
                            best_modularity = modularity
                            best_method = method
                    
                    if best_method:
                        comm_info = communities[best_method]
                        n_communities = comm_info.get("n_communities", 0)
                        
                        print(f"🏘️ Community Detection Results:")
                        metric("Best Method", best_method.replace("_", " ").title(), indent=1)
                        metric("Communities Found", str(n_communities), indent=1)
                        
                        modularity_status = "excellent" if best_modularity > 0.5 else "good" if best_modularity > 0.3 else "fair"
                        metric("Modularity Score", f"{best_modularity:.3f}", status=modularity_status, indent=1)
                        
                        if best_modularity > 0.5:
                            print("     └─ 🎯 Strong community structure detected")
                        elif best_modularity > 0.3:
                            print("     └─ 📊 Moderate community structure")
                        elif best_modularity > 0:
                            print("     └─ 🔍 Weak community structure")
                        else:
                            print("     └─ 🌐 No clear community structure")
                    
                    # Show all successful methods
                    if len(successful_methods) > 1:
                        print("   📋 All Community Detection Results:")
                        for method in successful_methods:
                            comm_info = communities[method]
                            n_comm = comm_info.get("n_communities", 0)
                            modularity = comm_info.get("modularity", 0)
                            print(f"     • {method.replace('_', ' ').title()}: {n_comm} communities (Q={modularity:.3f})")
                else:
                    print("🏘️ Community Detection:")
                    print("   ❌ No successful community detection methods")
            
            # Centrality Analysis
            centrality_results = graph_results.get("centralities", {})
            if centrality_results and best_graph and best_graph in centrality_results:
                section("Node Importance Analysis", 3)
                
                centralities = centrality_results[best_graph]
                
                # Most important nodes by different measures
                print("⭐ Most Important Variables:")
                
                for centrality_type in ["degree", "pagerank", "betweenness", "eigenvector", "closeness"]:
                    if centrality_type in centralities and "error" not in centralities[centrality_type]:
                        cent_info = centralities[centrality_type]
                        top_nodes = cent_info.get("top_nodes", [])
                        description = cent_info.get("description", "")
                        
                        if top_nodes:
                            top_node = top_nodes[0]
                            node_name = top_node[0]
                            score = top_node[1]
                            
                            centrality_name = centrality_type.replace("_", " ").title()
                            print(f"   • {centrality_name}: {node_name} ({score:.3f})")
                            if len(description) > 0 and centrality_type == "degree":
                                print(f"     └─ {description}")
                
                # Show top 3 most central nodes overall
                if "pagerank" in centralities and "error" not in centralities["pagerank"]:
                    print("\n👑 Top 3 Most Influential Variables:")
                    top_pagerank = centralities["pagerank"].get("top_nodes", [])[:3]
                    for i, (node, score) in enumerate(top_pagerank, 1):
                        print(f"     {i}. {node} (PageRank: {score:.4f})")
            
            # Network Embeddings
            embedding_results = graph_results.get("embeddings", {})
            if embedding_results and best_graph and best_graph in embedding_results:
                section("Network Embeddings Analysis", 3)
                
                embeddings = embedding_results[best_graph]
                successful_embeddings = [method for method in embeddings.keys() if "error" not in embeddings[method]]
                
                if successful_embeddings:
                    print("🧠 Available Embeddings:")
                    
                    for method in successful_embeddings:
                        emb_info = embeddings[method]
                        dimensions = emb_info.get("dimensions", 0)
                        method_name = emb_info.get("method", method)
                        description = emb_info.get("description", "")
                        
                        metric(method_name, f"{dimensions}D embedding", indent=1)
                        if description:
                            print(f"     └─ {description}")
                    
                    # Show most similar pairs for Node2Vec
                    if "node2vec" in embeddings and "error" not in embeddings["node2vec"]:
                        similar_pairs = embeddings["node2vec"].get("most_similar_pairs", [])
                        if similar_pairs:
                            print("\n🔗 Most Similar Variable Pairs (Node2Vec):")
                            for i, (var1, var2, similarity) in enumerate(similar_pairs[:3], 1):
                                print(f"     {i}. {var1} ↔ {var2} (similarity: {similarity:.3f})")
                else:
                    print("🧠 Network Embeddings:")
                    print("   ❌ No successful embedding methods")
            
            # Graph Construction Methods Comparison
            graphs = graph_results.get("graphs", {})
            if len(graphs) > 1:
                section("Graph Construction Methods Comparison", 3)
                
                print("📊 Network Construction Results:")
                for graph_type, graph_info in graphs.items():
                    if "error" not in graph_info:
                        nodes = graph_info.get("n_nodes", 0)
                        edges = graph_info.get("n_edges", 0)
                        edge_types = graph_info.get("edge_types", [])
                        
                        method_name = graph_type.replace("_", " ").title()
                        print(f"   • {method_name}: {nodes} nodes, {edges} edges")
                        
                        if edge_types:
                            edge_type_str = ", ".join(edge_types)
                            print(f"     └─ Edge types: {edge_type_str}")
                        
                        # Add density info from topology if available
                        if graph_type in topology_results:
                            density = topology_results[graph_type].get("density", 0)
                            print(f"     └─ Density: {density:.3f}")
                    else:
                        error_msg = graph_info.get("error", "Unknown error")
                        print(f"   • {graph_type.replace('_', ' ').title()}: ❌ Failed ({error_msg})")
            
            # Key Recommendations
            recommendations = graph_results.get("recommendations", [])
            if recommendations:
                section("Key Insights & Recommendations", 3)
                for i, _rec in enumerate(recommendations[:6], 1):
                    print(f"   {i}. {_rec}")
            
            # Final Summary
            if summary and best_graph:
                section("🎯 NETWORK ANALYSIS CONCLUSION", 3)
                
                best_nodes = summary.get("best_graph_nodes", 0)
                best_edges = summary.get("best_graph_edges", 0)
                best_connected = summary.get("best_graph_connected", False)
                
                if best_edges > 0:
                    if best_connected and best_nodes >= 5:
                        print(f"   🌟 STRONG network structure detected with {best_nodes} interconnected variables")
                        print(f"   🔗 Rich connectivity ({best_edges} relationships) enables advanced graph-based analysis")
                    elif best_edges >= best_nodes:
                        print(f"   ✅ MODERATE network structure with {best_edges} relationships among {best_nodes} variables")
                        print(f"   📊 Suitable for community analysis and centrality-based feature ranking")
                    else:
                        print(f"   📈 SPARSE network with {best_edges} key relationships identified")
                        print(f"   🔍 Focus on high-centrality variables for feature selection")
                    
                    # Advanced analysis recommendations
                    if best_nodes >= 10 and best_edges >= 20:
                        print(f"   🚀 Network is complex enough for graph machine learning algorithms")
                    
                    if embedding_results.get(best_graph, {}).get("node2vec") and "error" not in embedding_results[best_graph]["node2vec"]:
                        print(f"   🧠 Node embeddings available - excellent for feature engineering")
                else:
                    print(f"   ❌ NO significant network structure detected")
                    print(f"   💡 Variables appear to be largely independent - traditional analysis may be more suitable")

        elif graph_results and "error" in graph_results:
            section("GRAPH NETWORK ANALYSIS - ERROR", 1)
            print(f"❌ Analysis failed: {graph_results.get('error', 'Unknown error')}")
            print("💡 Ensure you have at least 2 numeric columns with sufficient variance")

        else:
            # No graph analysis was performed
            pass
        # ----------------------- Data quality assessment -----------------------
        section("COMPREHENSIVE DATA QUALITY ASSESSMENT", 1)
        completeness = (1 - self.df.isna().sum().sum() / (self.df.size or 1)) * 100
        uniqueness = (len(self.df.drop_duplicates()) / (len(self.df) or 1)) * 100
        consistency = 80.0
        ds = safe_get("distributions","summary", pd.DataFrame())
        if not ds.empty:
            gaussian_pct = (ds.get("is_gaussian", False).sum() / len(ds)) * 100
            consistency = min(100, 60 + gaussian_pct * 0.4)
        validity = 95.0
        out_sum = safe_get("outliers","summary", {})
        if out_sum:
            validity = max(50, 100 - (out_sum.get("overall_outlier_rate",0)*500))

        def band(x): return "excellent" if x>95 else "good" if x>85 else "fair" if x>70 else "poor"
        print("Quality Dimensions:")
        metric("Completeness", f"{completeness:.1f}%", status=band(completeness))
        metric("Uniqueness", f"{uniqueness:.1f}%", status=band(uniqueness))
        metric("Consistency", f"{consistency:.1f}%", status=band(consistency))
        metric("Validity", f"{validity:.1f}%", status=band(validity))

        if self._categorical_cols:
            section("Categorical Feature Health", 3)
            issues = []
            n = len(self.df)
            for col in self._categorical_cols[:8]:
                u = self.df[col].nunique(); up = (u/(n or 1))*100; nullp = self.df[col].isna().mean()*100
                if up > 95: st, desc = "warning", "Very high cardinality"; issues.append(f"{col}: {desc}")
                elif up > 50: st, desc = "fair", "High cardinality"
                elif u < 2: st, desc = "poor", "Constant/near-constant"; issues.append(f"{col}: {desc}")
                else: st, desc = "good", "Appropriate cardinality"
                print(f"   • {col}:")
                metric("Unique Values", f"{u} ({up:.1f}%)", status=st, indent=1)
                if nullp>0: metric("Missing", f"{nullp:.1f}%", status=band(100-nullp), indent=1)
            if issues:
                section("Categorical Issues", 3)
                for it in issues: rec(it, "medium")

        overall = (completeness + uniqueness + consistency + validity)/4
        section("Overall Quality Summary", 3)
        metric("Overall Quality Score", f"{overall:.1f}%")
        label = "Excellent - analysis-ready" if overall>90 else \
                "Good - minor preprocessing" if overall>80 else \
                "Fair - moderate preprocessing" if overall>70 else \
                "Poor - extensive preprocessing"
        print(f"   {'🟢' if overall>90 else '🟡' if overall>80 else '🟠' if overall>70 else '🔴'} {label}")

        # ----------------------- Executive recommendations -----------------------
        section("EXECUTIVE RECOMMENDATIONS", 1)
        recs = []
        if completeness < 95: recs.append(("Handle missing data (impute/repair patterns)" if completeness>=80 else "Address critical missing data patterns", "high" if completeness<80 else "medium"))
        if uniqueness < 95: recs.append(("Investigate and handle duplicate records","medium"))
        if not ds.empty:
            skewed = int(ds.get("is_skewed", False).sum()); total = len(ds)
            if total and skewed > total*0.5: recs.append(("Apply transformations to address widespread skewness","high"))
            elif total and skewed > total*0.3: recs.append(("Consider transformations for skewed features","medium"))
        if out_sum:
            rate = out_sum.get("overall_outlier_rate",0)
            if rate>0.1: recs.append(("Implement robust outlier detection and treatment","high"))
            elif rate>0.05: recs.append(("Review and validate detected outliers","medium"))
        cl_evals = (safe_get("clusters") or {}).get("evaluations", {})
        if cl_evals:
            best_sil = max([e.get("silhouette_score",0) for e in cl_evals.values()] + [0])
            if best_sil > 0.7: recs.append(("Strong clustering structure — consider segmentation","low"))
            elif best_sil > 0.5: recs.append(("Moderate clustering potential — explore segmentation","low"))

        prio = {"high":1,"medium":2,"low":3}
        recs.sort(key=lambda x: prio[x[1]])
        section("Action Items by Priority", 3)
        current = None
        icons = {"high":"🚨","medium":"⚠️","low":"💡"}
        for text, p in recs:
            if p != current:
                current = p; print(f"\n   {icons[p]} {p.upper()} PRIORITY:")
            rec(text, p, 1)
        if not recs: rec("Dataset in excellent condition — proceed with modeling", "low")

        # ----------------------- Completion summary -----------------------
        section("ANALYSIS COMPLETION SUMMARY", 1)
        mapping = {
            "distributions":"Distribution Analysis","correlations":"Correlation Analysis",
            "patterns":"Pattern Detection","feature_engineering":"Feature Engineering",
            "timeseries":"Time Series Analysis","clusters":"Clustering Analysis",
            "outliers":"Outlier Detection","missingness":"Missingness Analysis",
            "dimensionality":"Dimensionality Reduction"
        }
        completed = [v for k,v in mapping.items() if k in self._results_cache]
        print(f"✅ Completed Analyses: {len(completed)}")
        for name in completed: print(f"   • {name}")
        print(f"\n📊 Dataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"💾 Memory Usage: {self.df.memory_usage(deep=True).sum()/(1024**2):.2f} MB")
        print(f"🎯 Overall Quality: {overall:.1f}% ({label.split(' - ')[0]})")

        print("\n" + "="*80)
        print("📋 COMPREHENSIVE DATASET ANALYSIS COMPLETE")
        print("="*80)
        print("🔍 Use these insights to guide preprocessing and modeling.")
        print("📈 Consider the priority recommendations for optimal results.")
        print("="*80)


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
