from typing import Any, Dict

import pandas as pd

from .core import (
    report_metric,
    report_pct,
    report_recommendation,
    report_section,
    top_list,
)


class DatasetReportPrinter:
    """Report-focused rendering utilities for DatasetAnalyzer outputs."""

    def __init__(self, analyzer: Any):
        self.analyzer = analyzer

    def safe_get(self, name: str, key: str = None, default: Any = None) -> Any:
        return self.analyzer._get_analysis_result(name, key, default)

    def warmup_core_analyses(self) -> None:
        """Populate cache for analyses used throughout the report."""
        for analysis_name in [
            "distributions",
            "correlations",
            "outliers",
            "patterns",
            "clusters",
            "timeseries",
            "missingness",
            "feature_engineering",
            "dimensionality",
        ]:
            self.safe_get(analysis_name)

    def print_executive_summary(self, thresholds: Dict[str, Any]) -> None:
        """Print the report's executive summary section."""
        report_section("EXECUTIVE SUMMARY", 1)
        print("Dataset Overview:")
        report_metric(
            "Shape",
            f"{self.analyzer.df.shape[0]:,} rows × {self.analyzer.df.shape[1]} columns",
        )
        report_metric(
            "Memory Usage",
            f"{self.analyzer.df.memory_usage(deep=True).sum()/(1024**2):.2f} MB",
        )
        report_metric("Numeric Features", len(self.analyzer._numeric_cols))
        report_metric("Categorical Features", len(self.analyzer._categorical_cols))

        miss_pct = (
            self.analyzer.df.isna().sum().sum()
            / (self.analyzer.df.shape[0] * self.analyzer.df.shape[1] or 1)
        ) * 100
        missing_thresholds = thresholds["missing"]
        miss_status = (
            "excellent"
            if miss_pct < missing_thresholds[0]
            else (
                "good"
                if miss_pct < missing_thresholds[1]
                else "fair" if miss_pct < missing_thresholds[2] else "poor"
            )
        )
        report_metric("Missing Values", f"{miss_pct:.2f}%", status=miss_status)

    def print_distribution_analysis(self, thresholds: Dict[str, Any]) -> None:
        """Print distribution analysis findings."""
        dist_summary = self.safe_get("distributions", "summary", pd.DataFrame())
        if dist_summary.empty:
            return

        report_section("DISTRIBUTION ANALYSIS", 1)

        total_features = len(dist_summary)
        gaussian = int(dist_summary.get("is_gaussian", False).sum())
        skewed = int(dist_summary.get("is_skewed", False).sum())
        heavy = int(dist_summary.get("is_heavy_tailed", False).sum())
        avg_out_pct = float(
            dist_summary.get("outlier_pct_z>3", pd.Series([0])).mean() or 0
        )

        print("Statistical Properties:")
        report_metric(
            "Normal Distributions",
            f"{gaussian}/{total_features} ({report_pct(gaussian, total_features)})",
        )
        report_metric(
            "Skewed Distributions",
            f"{skewed}/{total_features} ({report_pct(skewed, total_features)})",
        )
        report_metric(
            "Heavy-Tailed",
            f"{heavy}/{total_features} ({report_pct(heavy, total_features)})",
        )
        report_metric("Average Outlier Rate", f"{avg_out_pct:.2f}%")

        report_section("Feature Quality Assessment", 3)
        if gaussian:
            feats = dist_summary.loc[dist_summary["is_gaussian"], "feature"].tolist()
            print(f"✅ Normal Distributions ({gaussian}):")
            print(f"   {top_list(feats)}")
            report_recommendation("Safe for parametric methods and linear models")

        hi_skew = dist_summary.loc[dist_summary["skewness"].abs() > thresholds["skew_hi"]]
        if not hi_skew.empty:
            print(f"\n⚠️ Highly Skewed Features ({len(hi_skew)}):")
            for _, row in hi_skew.head(3).iterrows():
                direction = "right" if row["skewness"] > 0 else "left"
                print(
                    f"   • {row['feature']}: {row['skewness']:.2f} ({direction}-skewed)"
                )
            report_recommendation(
                "Apply log/sqrt transform or use robust methods", "medium"
            )

        hi_out = dist_summary.loc[
            dist_summary.get("outlier_pct_z>3", 0) > thresholds["outlier_hi_pct"]
        ]
        if not hi_out.empty:
            print(
                f"\n🚨 High Outlier Rate Features ({len(hi_out)} > {thresholds['outlier_hi_pct']}%):"
            )
            for _, row in hi_out.head(3).iterrows():
                print(f"   • {row['feature']}: {row['outlier_pct_z>3']:.1f}% outliers")
            report_recommendation("Apply robust scaling or outlier treatment", "high")

        bi = dist_summary.loc[
            dist_summary.get("bimodality_coeff", 0) > thresholds["bimodal_bc"], "feature"
        ]
        if not bi.empty:
            print(f"\n🔀 Potential Bimodal Distributions ({len(bi)}):")
            print(f"   {top_list(bi.tolist(), 3)}")
            report_recommendation("Investigate mixtures or stratify data")

        report_section("Preprocessing Recommendations", 3)
        normal_pct = gaussian / total_features * 100 if total_features else 0
        if normal_pct > 70:
            report_recommendation(
                "Dataset largely normal — parametric methods recommended"
            )
        elif normal_pct > 30:
            report_recommendation("Mixed distributions — use a hybrid approach")
        else:
            report_recommendation(
                "Non-normal dominant — consider robust/non-parametric methods"
            )
        if skewed > total_features * 0.5:
            report_recommendation("Many skewed features — batch transform", "medium")

    def print_correlation_analysis(self, thresholds: Dict[str, Any]) -> None:
        """Print correlation analysis findings."""
        corrs = self.safe_get("correlations") or {}
        if not (
            isinstance(corrs, dict) and "pearson" in corrs and corrs["pearson"] is not None
        ):
            return

        report_section("CORRELATION ANALYSIS", 1)
        cm = corrs["pearson"]
        strong_pos, strong_neg, moderate_pairs = [], [], []
        strong_cutoff, moderate_cutoff = thresholds["corr"]

        cols = cm.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = cm.iloc[i, j]
                f1, f2 = cols[i], cols[j]
                if value > strong_cutoff:
                    strong_pos.append((f1, f2, value))
                elif value < -strong_cutoff:
                    strong_neg.append((f1, f2, value))
                elif abs(value) > moderate_cutoff:
                    moderate_pairs.append((f1, f2, value))

        print("Correlation Summary:")
        report_metric("Strong Positive (>0.7)", len(strong_pos))
        report_metric("Strong Negative (<-0.7)", len(strong_neg))
        report_metric("Moderate (0.5–0.7)", len(moderate_pairs))

        if strong_pos:
            report_section("Strong Positive Correlations", 3)
            for f1, f2, value in strong_pos[:5]:
                report_metric(f"{f1} ↔ {f2}", f"{value:.3f}")
        if strong_neg:
            report_section("Strong Negative Correlations", 3)
            for f1, f2, value in strong_neg[:5]:
                report_metric(f"{f1} ↔ {f2}", f"{value:.3f}")
        if strong_pos or strong_neg:
            report_recommendation(
                "Consider dimensionality reduction or feature selection", "medium"
            )

    def print_sota_insights(self) -> None:
        """Print SOTA-specific insights (Motifs, Causality, Anomalies)."""
        report_section("INTELLIGENT FACT DISCOVERY (SOTA)", 1)
        
        # 1. Motifs
        ts_results = self.safe_get("timeseries") or {}
        motifs = ts_results.get("motif_discovery", {})
        if motifs:
            report_section("Recurring Time-Series Motifs", 3)
            for col, info in motifs.items():
                m_list = info.get("top_motifs", [])
                if m_list:
                    print(f"   • {col}: Found {len(m_list)} recurring patterns (Window={info['window_size']})")
                    report_recommendation(f"Pattern periodicity in {col} suggests predictable cyclicality.")

        # 2. Directed Influence (Causality)
        patt_results = self.safe_get("patterns") or {}
        causality = patt_results.get("causality", {})
        infl = causality.get("directed_influence", [])
        if infl:
            report_section("Directed Influence & Causality", 3)
            # Sort by strength
            infl = sorted(infl, key=lambda x: x['strength'], reverse=True)
            for item in infl[:5]:
                conf = item.get("confidence", "low").upper()
                report_metric(f"{item['source']} → {item['target']}", f"Strength: {item['strength']:.3f} ({conf})")
                if item['strength'] > 0.1:
                    report_recommendation(f"{item['source']} is a potential leading indicator for {item['target']}.", "medium")

        # 3. Anomaly Mining
        anomalies = patt_results.get("anomalies", {})
        global_anom = anomalies.get("global_anomalies", [])
        if global_anom:
            report_section("Multi-variate Anomaly Clusters", 3)
            print(f"   • Detected {len(global_anom)} global anomaly points using Isolation Forest.")
            report_recommendation(f"Inspect indices: {top_list([str(i) for i in global_anom[:5]])}", "high")

        # 4. Trend Coordination
        coordination = ts_results.get("trend_coordination", {})
        alignments = coordination.get("phase_alignments", [])
        if alignments:
            report_section("Trend & Phase Coordination", 3)
            for al in alignments[:3]:
                report_metric(f"{al['pair'][0]} ↔ {al['pair'][1]}", f"Lag: {al['lag']} ({al['interpretation']})")
