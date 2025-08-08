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

        # ----------------------- Pattern detection -----------------------
        patt = safe_get("patterns") or {}
        if patt:
            section("ADVANCED PATTERN DETECTION", 1)

            # Feature types
            ft = patt.get("feature_types", {})
            if ft:
                section("Feature Type Classification", 3)
                for ftype, features in ft.items():
                    if not features: continue
                    label = ftype.replace("_"," ").title()
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

            # Relationships
            rel = patt.get("relationships", {})
            if rel:
                section("Relationship Patterns", 3)
                if rel.get("nonlinear"):
                    print("🌀 Non-Linear:")
                    for r in rel["nonlinear"][:3]:
                        metric(f"{r['feature1']} ↔ {r['feature2']}", f"score: {r['nonlinearity_score']:.3f}")
                    rec("Try polynomial/kernels")
                if rel.get("complex"):
                    print("\n🧬 High MI, low linear:")
                    for r in rel["complex"][:3]:
                        metric(f"{r['feature1']} ↔ {r['feature2']}", f"MI: {r['mutual_info']:.3f}")
                    rec("Explore interactions / non-linear models")

            # Best-fit distributions
            fits = patt.get("distributions", {})
            if fits:
                section("Statistical Distribution Fitting", 3)
                for feat, info in list(fits.items())[:5]:
                    ks = info.get("ks_pvalue", 0)
                    quality = ("Excellent","excellent") if ks > 0.1 else \
                            ("Good","good") if ks > 0.05 else \
                            ("Fair","fair") if ks > 0.01 else ("Poor","poor")
                    print(f"   • {feat}: {info['distribution'].title()}")
                    metric("AIC", f"{info.get('aic', float('nan')):.2f}", indent=1)
                    metric("KS p-value", f"{ks:.4f}", status=quality[1], indent=1)
                    print(f"     Fit Quality: {quality[0]}")

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
        ts = safe_get("timeseries") or {}
        if ts:
            section("TIME SERIES ANALYSIS", 1)
            stationarity_df = ts.get("stationarity", pd.DataFrame())
            readiness = ts.get("forecasting_readiness", {})
            temporal = ts.get("temporal_patterns", {})

            if not stationarity_df.empty:
                total = len(stationarity_df)
                stationary = int(stationarity_df["is_stationary"].sum())
                ready = sum(1 for v in readiness.values() if v.get("readiness_level") in {"excellent","good"})
                print("Analysis Overview:")
                metric("Total Series", total)
                metric("Stationary Series", f"{stationary}/{total} ({pct(stationary,total)})")
                metric("Forecast-Ready", f"{ready}/{total} ({pct(ready,total)})")

                section("Stationarity Assessment", 3)
                st_feats = stationarity_df.loc[stationarity_df["is_stationary"], "feature"].tolist()
                non_st = stationarity_df.loc[~stationarity_df["is_stationary"]]
                if st_feats:
                    print(f"✅ Stationary ({len(st_feats)}):")
                    for f in st_feats[:3]:
                        row = stationarity_df[stationarity_df["feature"] == f].iloc[0]
                        metric(f, f"ADF p={row['adf_pvalue']:.4f}", indent=1)
                    if len(st_feats) > 3: print(f"   ... and {len(st_feats)-3} more")
                if not non_st.empty:
                    print(f"\n⚠️ Non-Stationary ({len(non_st)}):")
                    for _, r in non_st.head(3).iterrows():
                        metric(r["feature"], r["stationarity_type"], indent=1)

                # Trends & seasonality (compact)
                section("Temporal Patterns", 3)
                trends = {k:v for k,v in temporal.items() if k.endswith("_trend")}
                seas = {k:v for k,v in temporal.items() if k.endswith("_seasonality")}
                strong_trends = []
                for k,v in trends.items():
                    if v.get("linear_significant") and v.get("trend_direction") not in {"no_trend","stable"}:
                        strong_trends.append((k.replace("_trend",""), v.get("linear_r_squared",0)))
                if strong_trends:
                    print("📈 Significant Trends:")
                    for f, r2 in sorted(strong_trends, key=lambda x: -x[1])[:3]:
                        metric(f, f"R²={r2:.3f}", indent=1)

                seas_list = []
                for k,v in seas.items():
                    cls = v.get("stl_classification","none")
                    if cls in {"moderate","strong"}:
                        seas_list.append((k.replace("_seasonality",""), v.get("stl_seasonal_strength",0)))
                if seas_list:
                    print("\n🔄 Seasonal Patterns:")
                    for f, s in sorted(seas_list, key=lambda x: -x[1])[:3]:
                        metric(f, f"strength={s:.3f}", indent=1)

                section("Time Series Recommendations", 3)
                if total:
                    if (stationary/total*100) < 50: rec("Apply differencing to achieve stationarity", "medium")
                    fr = (ready/total*100)
                    if fr > 70: rec("Well-suited for forecasting")
                    elif fr < 30: rec("Significant preprocessing needed before forecasting", "medium")
                if seas_list: rec("Use seasonal models (SARIMA, STL).")
                if strong_trends: rec("Include trend terms or detrend.")

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
