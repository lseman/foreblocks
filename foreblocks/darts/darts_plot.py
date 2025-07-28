import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from scipy import stats

warnings.filterwarnings("ignore")


class StreamlinedDARTSAnalyzer:
    """
    Streamlined statistical analyzer for DARTS multi-fidelity search results.
    Provides comprehensive analysis with minimal code repetition.
    """

    def __init__(self, search_results: Dict[str, Any]):
        """Initialize analyzer with search results from DARTSTrainer.multi_fidelity_search()"""
        self.search_results = search_results
        self.candidates = search_results.get("candidates", [])
        self.top_candidates = search_results.get("top_candidates", [])
        self.trained_candidates = search_results.get("trained_candidates", [])
        self.best_candidate = search_results.get("best_candidate", {})

        # Core data structures
        self.analysis_df = self._create_analysis_dataframe()
        self.patterns = self._analyze_all_patterns()

    def _create_analysis_dataframe(self) -> pd.DataFrame:
        """Create comprehensive dataframe with all architectural information"""
        data = []

        for candidate in self.candidates:
            if not candidate.get("success", False):
                continue

            row = self._extract_candidate_features(candidate)
            print(
                f"Extracted features for candidate {candidate['candidate_id']}: {row}"
            )
            data.append(row)

        return pd.DataFrame(data)

    def _extract_candidate_features(self, candidate: Dict) -> Dict:
        """Extract all features from a single candidate"""
        row = {
            # Basic info
            "candidate_id": candidate["candidate_id"],
            "zero_cost_score": candidate["score"],
            "num_operations": len(candidate.get("selected_ops", [])),
            "hidden_dim": candidate.get("hidden_dim", 0),
            "num_cells": candidate.get("num_cells", 0),
            "num_nodes": candidate.get("num_nodes", 0),
        }

        # Add zero-cost metrics
        self._add_zcm_features(candidate, row)

        # Add operation features
        self._add_operation_features(candidate, row)

        # Add encoder/decoder features
        self._add_encoder_decoder_features(candidate, row)

        # Add training status
        self._add_training_status(candidate, row)

        return row

    def _add_zcm_features(self, candidate: Dict, row: Dict):
        """Add zero-cost metrics to row"""
        metrics = candidate.get("metrics", {})
        print(
            f"Processing candidate {candidate['candidate_id']} with metrics: {metrics}"
        )
        for metric_name, value in metrics.items():
            print(f"Adding metric {metric_name} with value {value} to row")

            if isinstance(value, (int, float)):
                row[f"zcm_{metric_name}"] = value

            if metric_name == "metrics":
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        row[f"zcm_{sub_metric}"] = sub_value

        print(
            f"Added zero-cost metrics for candidate {candidate['candidate_id']}: {row}"
        )

    def _add_operation_features(self, candidate: Dict, row: Dict):
        """Add operation presence features to row"""
        selected_ops = candidate.get("selected_ops", [])
        all_ops = [
            "Identity",
            "TimeConv",
            "GRN",
            "Wavelet",
            "Fourier",
            "TCN",
            "ResidualMLP",
            "ConvMixer",
            "MultiScaleConv",
            "PyramidConv",
        ]

        for op in all_ops:
            row[f"op_{op}"] = 1 if op in selected_ops else 0

    def _add_encoder_decoder_features(self, candidate: Dict, row: Dict):
        """Add encoder/decoder features to row"""
        model = candidate.get("model")
        if model is None:
            return

        # Extract encoder info

        self._extract_component_info(model, "forecast_encoder", "encoder", row)

        # Extract decoder info
        self._extract_component_info(model, "forecast_decoder", "decoder", row)

        # Extract attention info
        self._extract_attention_info(model, row)

    def _extract_component_info(
        self, model, component_attr: str, component_type: str, row: Dict
    ):
        """Generic method to extract encoder or decoder information"""
        # try:
        component = getattr(model, component_attr, None)
        if component is None or not hasattr(component, "alphas"):
            return

        weights = F.softmax(component.alphas, dim=-1)

        # Get component types
        components_attr = (
            f"{component_type}s"
            if not component_type.endswith("r")
            else f"{component_type}s"
        )
        components = getattr(component, components_attr, [])

        # print(pato)
        if components:
            print(
                f"Extracting {component_type} information for candidate {row['candidate_id']}"
            )
            component_types = [type(comp).__name__ for comp in components]
            row[f"available_{component_type}s"] = ",".join(component_types)
            row[f"num_{component_type}_types"] = len(component_types)

            # Track weights and most likely choice
            max_idx = weights.argmax().item()
            for i, comp_type in enumerate(component_types):
                clean_name = (
                    comp_type.replace("Encoder", "")
                    .replace("Decoder", "")
                    .replace("RNN", "")
                    .lower()
                )
                row[f"has_{clean_name}_{component_type}"] = 1
                row[f"{clean_name}_{component_type}_weight"] = weights[i].item()

            # Most likely choice
            row[f"likely_{component_type}"] = component_types[max_idx]
            row[f"likely_{component_type}_weight"] = weights[max_idx].item()

            # Handle RNN names for decoders
            if component_type == "decoder" and hasattr(component, "rnn_names"):
                rnn_names = component.rnn_names
                row["rnn_names"] = ",".join(rnn_names)

                for i, rnn_name in enumerate(rnn_names):
                    if i < len(weights):
                        row[f"{rnn_name.lower()}_rnn_weight"] = weights[i].item()
                        if i == max_idx:
                            row["likely_rnn_type"] = rnn_name
                            row["likely_rnn_weight"] = weights[i].item()

    def _extract_attention_info(self, model, row: Dict):
        """Extract attention mechanism information"""
        try:
            decoder = getattr(model, "forecast_decoder", None)
            if decoder is None or not hasattr(decoder, "attention_alphas"):
                row["uses_attention"] = 0
                return

            attention_weights = F.softmax(decoder.attention_alphas, dim=-1)
            row["has_attention_alphas"] = 1
            row["num_attention_options"] = len(attention_weights)

            max_attention_idx = attention_weights.argmax().item()
            if max_attention_idx == len(attention_weights) - 1:
                row["likely_attention"] = "no_attention"
                row["uses_attention"] = 0
            else:
                row["likely_attention"] = f"attention_layer_{max_attention_idx}"
                row["uses_attention"] = 1

            row["attention_weight"] = attention_weights[max_attention_idx].item()

            if hasattr(decoder, "attention_bridges"):
                row["num_attention_bridges"] = len(decoder.attention_bridges)

        except Exception as e:
            row["attention_extraction_error"] = str(e)

    def _add_training_status(self, candidate: Dict, row: Dict):
        """Add training status and results"""
        # Check if in top candidates
        row["is_top_candidate"] = any(
            tc["candidate_id"] == candidate["candidate_id"]
            for tc in self.top_candidates
        )

        # Find trained info
        trained_info = self._find_trained_info(candidate["candidate_id"])

        if trained_info:
            row["was_trained"] = True
            row["val_loss"] = trained_info["val_loss"]
            row["is_best"] = trained_info == self.best_candidate
            self._add_final_architecture_info(trained_info, row)
        else:
            row["was_trained"] = False
            row["val_loss"] = np.nan
            row["is_best"] = False

    def _find_trained_info(self, candidate_id: str) -> Optional[Dict]:
        """Find training information for a candidate"""
        for tc in self.trained_candidates:
            if tc["candidate"]["candidate_id"] == candidate_id:
                return tc
        return None

    def _add_final_architecture_info(self, trained_info: Dict, row: Dict):
        """Add final architecture choices from trained models"""
        try:
            search_results = trained_info.get("search_results", {})
            model = (
                search_results.get("model")
                if search_results
                else trained_info.get("model")
            )

            if model is None:
                return

            # Final encoder choice
            self._extract_final_component_choice(
                model, "forecast_encoder", "encoder", row
            )

            # Final decoder choice
            self._extract_final_component_choice(
                model, "forecast_decoder", "decoder", row
            )

            # Final attention choice
            self._extract_final_attention_choice(model, row)

        except Exception as e:
            row["final_arch_extraction_error"] = str(e)

    def _extract_final_component_choice(
        self, model, component_attr: str, component_type: str, row: Dict
    ):
        """Extract final component choice from trained model"""
        component = getattr(model, component_attr, None)
        if component is None or not hasattr(component, "alphas"):
            return

        weights = F.softmax(component.alphas, dim=-1)
        selected_idx = weights.argmax().item()

        components_attr = f"{component_type}s"
        components = getattr(component, components_attr, [])

        if selected_idx < len(components):
            selected_component = type(components[selected_idx]).__name__
            row[f"final_{component_type}"] = selected_component
            row[f"final_{component_type}_weight"] = weights[selected_idx].item()

            clean_name = (
                selected_component.replace("Encoder", "")
                .replace("Decoder", "")
                .replace("RNN", "")
            )
            row[f"final_{component_type}_clean"] = clean_name

            # Handle RNN types for decoders
            if component_type == "decoder" and hasattr(component, "rnn_names"):
                rnn_names = component.rnn_names
                if selected_idx < len(rnn_names):
                    row["final_rnn_type"] = rnn_names[selected_idx]

    def _extract_final_attention_choice(self, model, row: Dict):
        """Extract final attention choice from trained model"""
        decoder = getattr(model, "forecast_decoder", None)
        if decoder is None or not hasattr(decoder, "attention_alphas"):
            return

        attention_weights = F.softmax(decoder.attention_alphas, dim=-1)
        selected_idx = attention_weights.argmax().item()

        row["final_attention_idx"] = selected_idx
        row["final_attention_weight"] = attention_weights[selected_idx].item()

        if selected_idx == len(attention_weights) - 1:
            row["final_attention_choice"] = "no_attention"
            row["final_uses_attention"] = 0
        else:
            row["final_attention_choice"] = f"attention_layer_{selected_idx}"
            row["final_uses_attention"] = 1

    def _analyze_all_patterns(self) -> Dict[str, Any]:
        """Analyze all architectural patterns in one pass"""
        patterns = {
            "component_analysis": {},
            "attention_analysis": {},
            "final_analysis": {},
            "operation_analysis": {},
        }

        # Analyze components (encoders, decoders, RNNs)
        component_types = ["encoder", "decoder", "rnn_type"]
        for comp_type in component_types:
            patterns["component_analysis"][comp_type] = (
                self._analyze_component_patterns(comp_type)
            )

        # Analyze attention
        patterns["attention_analysis"] = self._analyze_attention_patterns()

        # Analyze final choices for trained models
        patterns["final_analysis"] = self._analyze_final_patterns()

        # Analyze operations
        patterns["operation_analysis"] = self._analyze_operation_patterns()

        return patterns

    def _analyze_component_patterns(self, component_type: str) -> Dict[str, Any]:
        """Generic method to analyze any component type patterns"""
        likely_col = f"likely_{component_type}"
        weight_col = f"likely_{component_type}_weight"

        if likely_col not in self.analysis_df.columns:
            return {}

        # Get unique component types
        component_values = set(self.analysis_df[likely_col].dropna())

        freq_data = {}
        for comp_value in component_values:
            mask = self.analysis_df[likely_col] == comp_value
            freq_data[comp_value] = {
                "overall": mask.sum(),
                "top_candidates": (mask & self.analysis_df["is_top_candidate"]).sum(),
                "trained": (mask & self.analysis_df["was_trained"]).sum(),
                "avg_weight": (
                    self.analysis_df[mask][weight_col].mean()
                    if weight_col in self.analysis_df.columns and mask.sum() > 0
                    else 0
                ),
            }

        return freq_data

    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze attention usage patterns"""
        if "uses_attention" not in self.analysis_df.columns:
            return {}

        attention_stats = {
            "overall_usage": self.analysis_df["uses_attention"].sum(),
            "usage_rate": self.analysis_df["uses_attention"].mean(),
            "top_candidates_usage": self.analysis_df[
                self.analysis_df["is_top_candidate"]
            ]["uses_attention"].sum(),
            "trained_usage": self.analysis_df[self.analysis_df["was_trained"]][
                "uses_attention"
            ].sum(),
        }

        if "likely_attention" in self.analysis_df.columns:
            attention_choices = (
                self.analysis_df["likely_attention"].value_counts().to_dict()
            )
            attention_stats["attention_choices"] = attention_choices

        return attention_stats

    def _analyze_final_patterns(self) -> Dict[str, Any]:
        """Analyze final architecture patterns for trained models"""
        trained_df = self.analysis_df[self.analysis_df["was_trained"]]
        if trained_df.empty:
            return {}

        final_analysis = {}

        # Analyze final components
        final_components = ["final_encoder", "final_decoder", "final_rnn_type"]
        for comp in final_components:
            if comp in trained_df.columns:
                final_analysis[f"{comp}s"] = trained_df[comp].value_counts().to_dict()

                # Add weight information
                weight_col = f"{comp}_weight"
                if weight_col in trained_df.columns:
                    weights = {}
                    for comp_value in final_analysis[f"{comp}s"].keys():
                        if pd.notna(comp_value):
                            mask = trained_df[comp] == comp_value
                            weights[comp_value] = trained_df[mask][weight_col].mean()
                    final_analysis[f"{comp}_weights"] = weights

        # Analyze final attention
        if "final_uses_attention" in trained_df.columns:
            final_analysis["final_attention_usage"] = trained_df[
                "final_uses_attention"
            ].sum()
            final_analysis["final_attention_rate"] = trained_df[
                "final_uses_attention"
            ].mean()

            if "final_attention_choice" in trained_df.columns:
                final_analysis["final_attention_choices"] = (
                    trained_df["final_attention_choice"].value_counts().to_dict()
                )

        return final_analysis

    def _analyze_operation_patterns(self) -> Dict[str, Any]:
        """Analyze operation frequency patterns"""
        op_columns = [col for col in self.analysis_df.columns if col.startswith("op_")]
        operation_names = [col.replace("op_", "") for col in op_columns]

        patterns = {
            "overall": {},
            "top_candidates": {},
            "trained_candidates": {},
            "total_candidates": len(self.analysis_df),
            "total_top": len(self.analysis_df[self.analysis_df["is_top_candidate"]]),
            "total_trained": len(self.analysis_df[self.analysis_df["was_trained"]]),
        }

        # Calculate frequencies for each category
        for category in ["overall", "top_candidates", "trained_candidates"]:
            if category == "overall":
                df_subset = self.analysis_df
            elif category == "top_candidates":
                df_subset = self.analysis_df[self.analysis_df["is_top_candidate"]]
            else:  # trained_candidates
                df_subset = self.analysis_df[self.analysis_df["was_trained"]]

            for op_col, op_name in zip(op_columns, operation_names):
                patterns[category][op_name] = df_subset[op_col].sum()

        return patterns

    def analyze_correlations(self) -> pd.DataFrame:
        """Analyze correlations between architectural features and performance"""
        numeric_cols = self.analysis_df.select_dtypes(include=[np.number]).columns

        # Define feature groups
        arch_features = ["num_operations", "hidden_dim", "num_cells", "num_nodes"] + [
            col
            for col in numeric_cols
            if any(
                prefix in col for prefix in ["op_", "encoder", "decoder", "attention"]
            )
        ]

        performance_metrics = ["zero_cost_score"] + [
            col for col in numeric_cols if col.startswith("zcm_")
        ]

        print("Performance metrics:", performance_metrics)

        if "val_loss" in numeric_cols and not self.analysis_df["val_loss"].isna().all():
            performance_metrics.append("val_loss")

        # Compute correlations
        corr_data = []
        for arch_feat in arch_features:
            for perf_metric in performance_metrics:
                if (
                    arch_feat in self.analysis_df.columns
                    and perf_metric in self.analysis_df.columns
                ):
                    clean_data = self.analysis_df[[arch_feat, perf_metric]].dropna()
                    if len(clean_data) > 1 and clean_data[arch_feat].var() > 0:
                        corr, p_value = stats.pearsonr(
                            clean_data[arch_feat], clean_data[perf_metric]
                        )
                        corr_data.append(
                            {
                                "architectural_feature": arch_feat,
                                "performance_metric": perf_metric,
                                "correlation": corr,
                                "p_value": p_value,
                                "significant": p_value < 0.05,
                                "sample_size": len(clean_data),
                            }
                        )

        return pd.DataFrame(corr_data)

    def plot_encoder_decoder_analysis(
        self, figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """Create focused encoder/decoder/attention analysis plot"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(
            "Encoder/Decoder/Attention Architecture Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # Row 1: Component selection patterns
        self._plot_encoder_analysis(axes[0])
        self._plot_decoder_analysis(axes[1])
        self._plot_attention_analysis(axes[2])

        # Row 2: Final selections for trained models
        # self._plot_final_encoder_choices(axes[1, 0])
        # self._plot_final_decoder_choices(axes[1, 1])
        # self._plot_final_attention_choices(axes[1, 2])

        plt.tight_layout()
        return fig

    def plot_operation_analysis(
        self, figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """Create focused operation/block analysis plot with a centered plot on second row"""
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Operation and Block Analysis", fontsize=16, fontweight="bold")

        # Use 2 rows, 4 columns for flexible layout
        gs = GridSpec(2, 4, figure=fig)

        # Top row: two full-width plots
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])

        # Bottom row: centered half-width (middle 2 columns out of 4)
        ax3 = fig.add_subplot(gs[1, 1:3])

        # Fill the axes
        self._plot_operation_frequencies(ax1)
        self._plot_operation_performance_impact(ax2)
        self._plot_architecture_complexity(ax3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        return fig

    def plot_correlation_analysis(
        self, figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """Create focused correlation analysis plot"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(
            "Architecture-Performance Correlation Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # Correlation heatmap
        self._plot_correlation_heatmap(axes[0])

        # Top correlations bar chart
        self._plot_top_correlations(axes[1])

        plt.tight_layout()
        return fig

    def plot_performance_analysis(
        self, figsize: Tuple[int, int] = (16, 8)
    ) -> plt.Figure:
        """Create focused performance analysis plot"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("Performance Analysis", fontsize=16, fontweight="bold")

        # Performance distributions
        self._plot_performance_distributions(axes[0])

        # Training success analysis
        self._plot_training_success_analysis(axes[1])

        plt.tight_layout()
        return fig

    def _plot_encoder_analysis(self, ax):
        """Plot encoder selection patterns"""
        encoder_data = self.patterns["component_analysis"].get("encoder", {})

        if not encoder_data:
            ax.text(
                0.5,
                0.5,
                "No Encoder Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Encoder Selection Patterns")
            return

        encoders = list(encoder_data.keys())
        overall_counts = [encoder_data[enc]["overall"] for enc in encoders]
        top_counts = [encoder_data[enc]["top_candidates"] for enc in encoders]
        trained_counts = [encoder_data[enc]["trained"] for enc in encoders]
        avg_weights = [encoder_data[enc]["avg_weight"] for enc in encoders]

        x = np.arange(len(encoders))
        width = 0.25

        bars1 = ax.bar(
            x - width,
            overall_counts,
            width,
            label="All Candidates",
            alpha=0.8,
            color="skyblue",
        )
        bars2 = ax.bar(
            x, top_counts, width, label="Top Candidates", alpha=0.8, color="orange"
        )
        bars3 = ax.bar(
            x + width,
            trained_counts,
            width,
            label="Trained",
            alpha=0.8,
            color="lightgreen",
        )

        # Add weight information
        # for i, (bar, weight) in enumerate(zip(bars1, avg_weights)):
        #     if weight > 0:
        #         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
        #                f'α:{weight:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_title("Encoder Selection Patterns")
        ax.set_xlabel("Encoder Types")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [enc.replace("Encoder", "") for enc in encoders], rotation=45, ha="right"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_decoder_analysis(self, ax):
        """Plot decoder selection patterns"""
        decoder_data = self.patterns["component_analysis"].get("decoder", {})

        if not decoder_data:
            ax.text(
                0.5,
                0.5,
                "No Decoder Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Decoder Selection Patterns")
            return

        decoders = list(decoder_data.keys())
        overall_counts = [decoder_data[dec]["overall"] for dec in decoders]
        top_counts = [decoder_data[dec]["top_candidates"] for dec in decoders]
        trained_counts = [decoder_data[dec]["trained"] for dec in decoders]
        avg_weights = [decoder_data[dec]["avg_weight"] for dec in decoders]

        x = np.arange(len(decoders))
        width = 0.25

        bars1 = ax.bar(
            x - width,
            overall_counts,
            width,
            label="All Candidates",
            alpha=0.8,
            color="lightcoral",
        )
        bars2 = ax.bar(
            x, top_counts, width, label="Top Candidates", alpha=0.8, color="gold"
        )
        bars3 = ax.bar(
            x + width,
            trained_counts,
            width,
            label="Trained",
            alpha=0.8,
            color="lightseagreen",
        )

        # Add weight information
        # for i, (bar, weight) in enumerate(zip(bars1, avg_weights)):
        #     if weight > 0:
        #         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
        #                f'α:{weight:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_title("Decoder Selection Patterns")
        ax.set_xlabel("Decoder Types")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [dec.replace("Decoder", "") for dec in decoders], rotation=45, ha="right"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_attention_analysis(self, ax):
        """Plot attention usage analysis"""
        attention_data = self.patterns["attention_analysis"]

        if not attention_data or "attention_choices" not in attention_data:
            # Fallback to simple usage statistics
            if attention_data:
                categories = ["Uses Attention", "No Attention"]
                usage_counts = [
                    attention_data.get("overall_usage", 0),
                    len(self.analysis_df) - attention_data.get("overall_usage", 0),
                ]
                colors = ["#4ECDC4", "#FF6B6B"]
                wedges, texts, autotexts = ax.pie(
                    usage_counts,
                    labels=categories,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                )
                for autotext in autotexts:
                    autotext.set_color("white")
                    autotext.set_fontweight("bold")
                ax.set_title("Attention Usage")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Attention Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Attention Usage")
            return

        choices = attention_data["attention_choices"]
        labels = list(choices.keys())
        sizes = list(choices.values())

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors[: len(labels)],
            startangle=90,
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax.set_title("Attention Bridge Selection")

    def _plot_final_encoder_choices(self, ax):
        """Plot final encoder choices for trained models"""
        final_data = self.patterns["final_analysis"]

        if "final_encoders" not in final_data:
            ax.text(
                0.5,
                0.5,
                "No Final Encoder Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Final Encoder Choices")
            return

        encoders = final_data["final_encoders"]
        encoder_weights = final_data.get("final_encoder_weights", {})

        encoder_names = list(encoders.keys())
        encoder_counts = list(encoders.values())
        weights = [encoder_weights.get(name, 0) for name in encoder_names]

        bars = ax.bar(
            range(len(encoder_names)), encoder_counts, alpha=0.8, color="lightblue"
        )

        # Add weight information
        # for i, (bar, weight, name) in enumerate(zip(bars, weights, encoder_names)):
        #     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
        #            f'{encoders[name]}\nα:{weight:.3f}',
        #            ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax.set_title("Final Encoder Choices (Trained Models)")
        ax.set_xlabel("Encoder Types")
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(encoder_names)))
        ax.set_xticklabels(
            [name.replace("Encoder", "") for name in encoder_names],
            rotation=45,
            ha="right",
        )
        ax.grid(True, alpha=0.3)

    def _plot_final_decoder_choices(self, ax):
        """Plot final decoder/RNN choices for trained models"""
        final_data = self.patterns["final_analysis"]

        # Prefer RNN types if available, otherwise use decoder types
        if "final_rnn_types" in final_data:
            choices = final_data["final_rnn_types"]
            title = "Final RNN Type Choices (Trained Models)"
            xlabel = "RNN Types"
        elif "final_decoders" in final_data:
            choices = final_data["final_decoders"]
            title = "Final Decoder Choices (Trained Models)"
            xlabel = "Decoder Types"
        else:
            ax.text(
                0.5,
                0.5,
                "No Final Decoder Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Final Decoder/RNN Choices")
            return

        choice_names = list(choices.keys())
        choice_counts = list(choices.values())

        bars = ax.bar(
            range(len(choice_names)), choice_counts, alpha=0.8, color="lightcoral"
        )

        # Add count labels
        for bar, count in zip(bars, choice_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(choice_names)))
        ax.set_xticklabels(
            [name.replace("Decoder", "") for name in choice_names],
            rotation=45,
            ha="right",
        )
        ax.grid(True, alpha=0.3)

    def _plot_final_attention_choices(self, ax):
        """Plot final attention choices for trained models"""
        final_data = self.patterns["final_analysis"]

        if "final_attention_choices" not in final_data:
            # Show simple usage rate if available
            if "final_attention_rate" in final_data:
                rate = final_data["final_attention_rate"] * 100
                categories = ["Uses Attention", "No Attention"]
                sizes = [rate, 100 - rate]
                colors = ["#4ECDC4", "#FF6B6B"]

                wedges, texts, autotexts = ax.pie(
                    sizes,
                    labels=categories,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                )
                for autotext in autotexts:
                    autotext.set_color("white")
                    autotext.set_fontweight("bold")
                ax.set_title("Final Attention Usage (Trained)")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Final Attention Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Final Attention Choices")
            return

        choices = final_data["final_attention_choices"]
        choice_names = list(choices.keys())
        choice_counts = list(choices.values())

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        wedges, texts, autotexts = ax.pie(
            choice_counts,
            labels=choice_names,
            autopct="%1.1f%%",
            colors=colors[: len(choice_names)],
            startangle=90,
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax.set_title("Final Attention Choices (Trained)")

    def _plot_operation_frequencies(self, ax):
        """Plot operation frequency analysis"""
        op_data = self.patterns["operation_analysis"]

        operations = list(op_data["overall"].keys())
        overall_freq = [op_data["overall"][op] for op in operations]
        trained_freq = [op_data["trained_candidates"][op] for op in operations]

        x = np.arange(len(operations))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            overall_freq,
            width,
            label="All Candidates",
            alpha=0.7,
            color="lightblue",
        )
        bars2 = ax.bar(
            x + width / 2,
            trained_freq,
            width,
            label="Trained Models",
            alpha=0.7,
            color="orange",
        )

        ax.set_title("Operation Usage Frequency")
        ax.set_xlabel("Operations")
        ax.set_ylabel("Frequency")
        ax.set_xticks(x)
        ax.set_xticklabels(operations, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_operation_performance_impact(self, ax):
        """Plot performance impact of each operation"""
        op_columns = [col for col in self.analysis_df.columns if col.startswith("op_")]
        operations = [col.replace("op_", "") for col in op_columns]

        impacts = []
        p_values = []

        for op_col in op_columns:
            has_op = self.analysis_df[op_col] == 1
            no_op = self.analysis_df[op_col] == 0

            if len(self.analysis_df[has_op]) > 0 and len(self.analysis_df[no_op]) > 0:
                with_op_score = self.analysis_df[has_op]["zero_cost_score"].mean()
                without_op_score = self.analysis_df[no_op]["zero_cost_score"].mean()
                impact = with_op_score - without_op_score

                _, p_value = stats.ttest_ind(
                    self.analysis_df[has_op]["zero_cost_score"].dropna(),
                    self.analysis_df[no_op]["zero_cost_score"].dropna(),
                )
                impacts.append(impact)
                p_values.append(p_value)
            else:
                impacts.append(0)
                p_values.append(1.0)

        # Color bars by significance
        colors = ["red" if p < 0.05 else "lightblue" for p in p_values]

        bars = ax.bar(range(len(operations)), impacts, color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        ax.set_title("Operation Performance Impact\n(Red = Significant, p < 0.05)")
        ax.set_xlabel("Operations")
        ax.set_ylabel("Performance Impact")
        ax.set_xticks(range(len(operations)))
        ax.set_xticklabels(operations, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    def _plot_architecture_complexity(self, ax):
        """Plot architecture complexity analysis"""
        if "num_operations" not in self.analysis_df.columns:
            ax.text(
                0.5,
                0.5,
                "No Complexity Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Architecture Complexity")
            return

        # Bin by number of operations
        complexity_bins = self.analysis_df["num_operations"].value_counts().sort_index()

        bars = ax.bar(
            complexity_bins.index, complexity_bins.values, alpha=0.7, color="lightgreen"
        )

        # Add count labels
        for bar, count in zip(bars, complexity_bins.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Architecture Complexity Distribution")
        ax.set_xlabel("Number of Operations")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    def _plot_operation_success_rates(self, ax):
        """Plot success rates (training rate) for each operation"""
        op_data = self.patterns["operation_analysis"]

        operations = list(op_data["overall"].keys())
        success_rates = []

        for op in operations:
            overall = op_data["overall"][op]
            trained = op_data["trained_candidates"][op]
            rate = trained / overall if overall > 0 else 0
            success_rates.append(rate * 100)

        bars = ax.bar(
            range(len(operations)), success_rates, alpha=0.7, color="mediumpurple"
        )

        # Add percentage labels
        for bar, rate in zip(bars, success_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Operation Training Success Rates")
        ax.set_xlabel("Operations")
        ax.set_ylabel("Success Rate (%)")
        ax.set_xticks(range(len(operations)))
        ax.set_xticklabels(operations, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(success_rates) * 1.1 if success_rates else 100)

    def _plot_top_correlations(self, ax):
        """Plot top correlations as bar chart"""
        corr_df = self.analyze_correlations()

        if corr_df.empty:
            ax.text(
                0.5,
                0.5,
                "No Correlation Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Top Correlations")
            return

        # Get top correlations by absolute value
        top_corr = corr_df.loc[corr_df["correlation"].abs().nlargest(10).index]

        if top_corr.empty:
            ax.text(
                0.5,
                0.5,
                "No Strong Correlations",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Top Correlations")
            return

        # Create labels and colors
        labels = [
            (
                f"{row['architectural_feature'][:15]}..."
                if len(row["architectural_feature"]) > 15
                else row["architectural_feature"]
            )
            for _, row in top_corr.iterrows()
        ]
        correlations = top_corr["correlation"].values
        colors = ["red" if corr < 0 else "blue" for corr in correlations]

        bars = ax.barh(range(len(labels)), correlations, color=colors, alpha=0.7)

        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            ax.text(
                bar.get_width() + 0.01 if corr > 0 else bar.get_width() - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{corr:.3f}",
                ha="left" if corr > 0 else "right",
                va="center",
                fontweight="bold",
            )

        ax.set_title("Top Architecture-Performance Correlations")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)

    def _plot_performance_distributions(self, ax):
        """Plot performance metric distributions"""
        if "zero_cost_score" not in self.analysis_df.columns:
            ax.text(
                0.5,
                0.5,
                "No Performance Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Distributions")
            return

        # Plot distributions for different groups
        all_scores = self.analysis_df["zero_cost_score"].dropna()
        top_scores = self.analysis_df[self.analysis_df["is_top_candidate"]][
            "zero_cost_score"
        ].dropna()
        trained_scores = self.analysis_df[self.analysis_df["was_trained"]][
            "zero_cost_score"
        ].dropna()

        ax.hist(
            all_scores, bins=20, alpha=0.5, label="All Candidates", color="lightblue"
        )
        if not top_scores.empty:
            ax.hist(
                top_scores, bins=15, alpha=0.7, label="Top Candidates", color="orange"
            )
        if not trained_scores.empty:
            ax.hist(
                trained_scores,
                bins=10,
                alpha=0.8,
                label="Trained Models",
                color="green",
            )

        ax.set_title("Zero-Cost Score Distributions")
        ax.set_xlabel("Zero-Cost Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_training_success_analysis(self, ax):
        """Plot training success analysis"""
        # Calculate success rates by score quartiles
        if "zero_cost_score" not in self.analysis_df.columns:
            ax.text(
                0.5,
                0.5,
                "No Score Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Training Success Analysis")
            return

        # Create quartiles
        quartiles = pd.qcut(
            self.analysis_df["zero_cost_score"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )

        success_rates = []
        quartile_labels = []

        for quartile in ["Q1", "Q2", "Q3", "Q4"]:
            mask = quartiles == quartile
            total = mask.sum()
            trained = self.analysis_df[mask]["was_trained"].sum()
            success_rate = trained / total * 100 if total > 0 else 0

            success_rates.append(success_rate)
            quartile_labels.append(f"{quartile}\n(n={total})")

        bars = ax.bar(
            range(len(quartile_labels)), success_rates, alpha=0.7, color="lightcoral"
        )

        # Add percentage labels
        for bar, rate in zip(bars, success_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Training Success Rate by Score Quartile")
        ax.set_xlabel("Score Quartiles")
        ax.set_ylabel("Training Success Rate (%)")
        ax.set_xticks(range(len(quartile_labels)))
        ax.set_xticklabels(quartile_labels)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(success_rates) * 1.2 if success_rates else 100)

    def _plot_correlation_heatmap(self, ax):
        """Plot correlation heatmap"""
        corr_df = self.analyze_correlations()

        if corr_df.empty:
            ax.text(
                0.5,
                0.5,
                "No Correlation Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Create pivot table for heatmap
        heatmap_data = corr_df.pivot(
            index="architectural_feature",
            columns="performance_metric",
            values="correlation",
        )

        # Create heatmap
        im = ax.imshow(
            heatmap_data.values,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )

        # Set labels
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
        ax.set_yticklabels(heatmap_data.index)

        # Add correlation values
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                if not pd.isna(heatmap_data.iloc[i, j]):
                    corr_val = heatmap_data.iloc[i, j]
                    text_color = "white" if abs(corr_val) > 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{corr_val:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                    )

        ax.set_title("Architecture-Performance Correlations")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label("Correlation Coefficient")

    def generate_insights_report(self) -> str:
        """Generate comprehensive insights report"""
        report = []
        report.append("=" * 80)
        report.append("STREAMLINED DARTS ANALYSIS REPORT")
        report.append("=" * 80)

        # Overview
        report.append(f"\n📊 SEARCH OVERVIEW")
        report.append(f"Total candidates: {len(self.candidates)}")
        report.append(f"Top candidates: {len(self.top_candidates)}")
        report.append(f"Trained models: {len(self.trained_candidates)}")

        # Component analysis
        component_data = self.patterns["component_analysis"]
        if component_data:
            report.append(f"\n🏗️ COMPONENT ANALYSIS")
            for comp_type, data in component_data.items():
                if data:
                    report.append(f"  {comp_type.title()} Usage:")
                    for comp_name, freq_data in data.items():
                        clean_name = comp_name.replace("Encoder", "").replace(
                            "Decoder", ""
                        )
                        avg_weight = freq_data.get("avg_weight", 0)
                        report.append(
                            f"    {clean_name}: {freq_data['trained']} trained "
                            f"(α={avg_weight:.3f})"
                        )

        # Final selections
        final_data = self.patterns["final_analysis"]
        if final_data:
            report.append(f"\n🎯 FINAL SELECTIONS")
            for key, value in final_data.items():
                if isinstance(value, dict) and not key.endswith("_weights"):
                    component_type = key.replace("final_", "").replace("s", "")
                    report.append(f"  {component_type.title()}:")
                    for comp, count in value.items():
                        report.append(f"    {comp}: {count} models")

        # Strong correlations
        corr_df = self.analyze_correlations()
        strong_corr = corr_df[abs(corr_df["correlation"]) > 0.3]

        if not strong_corr.empty:
            report.append(f"\n🔗 STRONG CORRELATIONS (|r| > 0.3)")
            for _, row in strong_corr.head(5).iterrows():
                sig = "*" if row["p_value"] < 0.05 else ""
                report.append(
                    f"  {row['architectural_feature']} ↔ {row['performance_metric']}: "
                    f"r={row['correlation']:.3f}{sig}"
                )

        # Key insights
        report.append(f"\n💡 KEY INSIGHTS")

        # Best performing components
        if component_data:
            for comp_type, data in component_data.items():
                if data:
                    best_comp = max(
                        data.items(),
                        key=lambda x: x[1]["trained"] / max(x[1]["overall"], 1),
                    )
                    success_rate = best_comp[1]["trained"] / max(
                        best_comp[1]["overall"], 1
                    )
                    clean_name = (
                        best_comp[0].replace("Encoder", "").replace("Decoder", "")
                    )
                    report.append(
                        f"  • Best {comp_type}: {clean_name} "
                        f"({success_rate:.1%} success rate)"
                    )

        # Attention usage insight
        attention_data = self.patterns["attention_analysis"]
        if attention_data:
            usage_rate = attention_data.get("usage_rate", 0)
            if usage_rate > 0.5:
                report.append(
                    f"  • High attention usage ({usage_rate:.1%}) suggests complex dependencies"
                )
            else:
                report.append(
                    f"  • Low attention usage ({usage_rate:.1%}) indicates simpler patterns work well"
                )

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def _plot_top_correlations(self, ax):
        """Plot top correlations as bar chart"""
        corr_df = self.analyze_correlations()

        if corr_df.empty:
            ax.text(
                0.5,
                0.5,
                "No Correlation Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Top Correlations")
            return

        # Get top correlations by absolute value
        top_corr = corr_df.loc[corr_df["correlation"].abs().nlargest(10).index]

        if top_corr.empty:
            ax.text(
                0.5,
                0.5,
                "No Strong Correlations",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Top Correlations")
            return

        # Create labels and colors
        labels = [
            (
                f"{row['architectural_feature'][:15]}..."
                if len(row["architectural_feature"]) > 15
                else row["architectural_feature"]
            )
            for _, row in top_corr.iterrows()
        ]
        correlations = top_corr["correlation"].values
        colors = ["red" if corr < 0 else "blue" for corr in correlations]

        bars = ax.barh(range(len(labels)), correlations, color=colors, alpha=0.7)

        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            ax.text(
                bar.get_width() + 0.01 if corr > 0 else bar.get_width() - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{corr:.3f}",
                ha="left" if corr > 0 else "right",
                va="center",
                fontweight="bold",
            )

        ax.set_title("Top Architecture-Performance Correlations")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)

    def _plot_performance_distributions(self, ax):
        """Plot performance metric distributions"""
        if "zero_cost_score" not in self.analysis_df.columns:
            ax.text(
                0.5,
                0.5,
                "No Performance Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Distributions")
            return

        # Plot distributions for different groups
        all_scores = self.analysis_df["zero_cost_score"].dropna()
        top_scores = self.analysis_df[self.analysis_df["is_top_candidate"]][
            "zero_cost_score"
        ].dropna()
        trained_scores = self.analysis_df[self.analysis_df["was_trained"]][
            "zero_cost_score"
        ].dropna()

        ax.hist(
            all_scores, bins=20, alpha=0.5, label="All Candidates", color="lightblue"
        )
        if not top_scores.empty:
            ax.hist(
                top_scores, bins=15, alpha=0.7, label="Top Candidates", color="orange"
            )
        if not trained_scores.empty:
            ax.hist(
                trained_scores,
                bins=10,
                alpha=0.8,
                label="Trained Models",
                color="green",
            )

        ax.set_title("Zero-Cost Score Distributions")
        ax.set_xlabel("Zero-Cost Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_training_success_analysis(self, ax):
        """Plot training success analysis"""
        # Calculate success rates by score quartiles
        if "zero_cost_score" not in self.analysis_df.columns:
            ax.text(
                0.5,
                0.5,
                "No Score Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Training Success Analysis")
            return

        # Create quartiles
        quartiles = pd.qcut(
            self.analysis_df["zero_cost_score"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )

        success_rates = []
        quartile_labels = []

        for quartile in ["Q1", "Q2", "Q3", "Q4"]:
            mask = quartiles == quartile
            total = mask.sum()
            trained = self.analysis_df[mask]["was_trained"].sum()
            success_rate = trained / total * 100 if total > 0 else 0

            success_rates.append(success_rate)
            quartile_labels.append(f"{quartile}\n(n={total})")

        bars = ax.bar(
            range(len(quartile_labels)), success_rates, alpha=0.7, color="lightcoral"
        )

        # Add percentage labels
        for bar, rate in zip(bars, success_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Training Success Rate by Score Quartile")
        ax.set_xlabel("Score Quartiles")
        ax.set_ylabel("Training Success Rate (%)")
        ax.set_xticks(range(len(quartile_labels)))
        ax.set_xticklabels(quartile_labels)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(success_rates) * 1.2 if success_rates else 100)

    def plot_architecture_space_exploration(
        self, figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """Visualize the exploration of architectural space"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 1. Hidden dimension vs performance
        scatter_data = self.analysis_df.dropna(subset=["zero_cost_score"])

        scatter = axes[0].scatter(
            scatter_data["hidden_dim"],
            scatter_data["zero_cost_score"],
            c=scatter_data["num_operations"],
            s=scatter_data["num_cells"] * 50,
            alpha=0.6,
            cmap="viridis",
        )

        axes[0].set_xlabel("Hidden Dimension")
        axes[0].set_ylabel("Zero-Cost Score")
        axes[0].set_title(
            "Architecture Space Exploration\n(Size=Cells, Color=Operations)"
        )
        axes[0].grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label("Number of Operations")

        # Highlight top candidates
        if len(self.analysis_df[self.analysis_df["is_top_candidate"]]) > 0:
            top_data = self.analysis_df[self.analysis_df["is_top_candidate"]]
            axes[0].scatter(
                top_data["hidden_dim"],
                top_data["zero_cost_score"],
                s=100,
                facecolors="none",
                edgecolors="red",
                linewidth=2,
                label="Top Candidates",
            )
            axes[0].legend()

        # 2. Performance distribution by architecture configuration
        config_performance = {}
        for _, row in self.analysis_df.iterrows():
            config = f"{row['num_cells']}C-{row['num_nodes']}N"
            if config not in config_performance:
                config_performance[config] = []
            config_performance[config].append(row["zero_cost_score"])

        configs = list(config_performance.keys())
        scores = [config_performance[config] for config in configs]

        bp = axes[1].boxplot(scores, labels=configs, patch_artist=True)

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_xlabel("Architecture Configuration (Cells-Nodes)")
        axes[1].set_ylabel("Zero-Cost Score")
        axes[1].set_title("Performance Distribution by Architecture")
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig

    def plot_metric_architecture_correlations(
        self, method: str = "pearson", figsize=(16, 10)
    ) -> plt.Figure:
        """
        Plot heatmap of correlations between zero-cost metrics and architectural features.

        Args:
            method: 'pearson' or 'spearman'
        """
        from scipy.stats import spearmanr

        df = self.analysis_df.copy()

        # Identify features and metrics
        metrics_cols = [col for col in df.columns if col.startswith("zcm_")]
        arch_cols = [
            col
            for col in df.columns
            if col in ["num_operations", "hidden_dim", "num_cells", "num_nodes"]
            or col.startswith("op_")
        ]

        df = df[metrics_cols + arch_cols].dropna()
        corr_matrix = pd.DataFrame(index=metrics_cols, columns=arch_cols)

        # Compute correlations
        for metric in metrics_cols:
            for feat in arch_cols:
                if method == "pearson":
                    corr, _ = stats.pearsonr(df[metric], df[feat])
                else:
                    corr, _ = spearmanr(df[metric], df[feat])
                corr_matrix.at[metric, feat] = corr

        corr_matrix = corr_matrix.astype(float)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": f"{method.title()} Correlation"},
            ax=ax,
        )
        ax.set_title(
            f"{method.title()} Correlation: Zero-Cost Metrics vs Architectural Features",
            fontsize=14,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig

    def plot_architecture_configurations(
        self, top_k: int = 20, figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot the frequency of combined architecture configurations like '2C-3N-128H'.

        Args:
            top_k: Number of most frequent configurations to show
            figsize: Size of the output figure
        """
        if not all(
            col in self.analysis_df.columns
            for col in ["num_cells", "num_nodes", "hidden_dim"]
        ):
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "Missing columns in data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Architecture Configuration Frequencies")
            return fig

        # Generate combined configuration label
        self.analysis_df["arch_config"] = self.analysis_df.apply(
            lambda row: f"{row['num_cells']}C-{row['num_nodes']}N-{row['hidden_dim']}H",
            axis=1,
        )

        # Count frequencies
        config_counts = (
            self.analysis_df["arch_config"]
            .value_counts()
            .sort_values(ascending=True)
            .tail(top_k)
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            x=config_counts.values, y=config_counts.index, palette="Purples_d", ax=ax
        )

        ax.set_title("Top Architecture Configurations", fontsize=14, fontweight="bold")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Architecture Config")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        return fig


def analyze_darts_search(
    search_results: Dict[str, Any], save_plots: bool = True, plot_dir: str = "./plots/"
) -> StreamlinedDARTSAnalyzer:
    """
    Streamlined analysis of DARTS search results with focused plots.

    Args:
        search_results: Results from DARTSTrainer.multi_fidelity_search()
        save_plots: Whether to save plots to files
        plot_dir: Directory to save plots

    Returns:
        StreamlinedDARTSAnalyzer instance with all analysis results
    """
    import os

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = StreamlinedDARTSAnalyzer(search_results)

    print("🔍 Analyzing DARTS search results...")

    # Create focused analysis plots
    print("🏗️ Creating encoder/decoder/attention analysis...")
    fig1 = analyzer.plot_encoder_decoder_analysis()

    print("📊 Creating operation/block analysis...")
    fig2 = analyzer.plot_operation_analysis()

    # print("🔗 Creating correlation analysis...")
    # fig3 = analyzer.plot_correlation_analysis()

    print("📈 Creating performance analysis...")
    fig4 = analyzer.plot_performance_analysis()

    fig5 = analyzer.plot_architecture_space_exploration()

    fig6 = analyzer.plot_metric_architecture_correlations(method="pearson")

    fig7 = analyzer.plot_architecture_configurations(top_k=20)

    if save_plots:
        # Save all plots
        fig1.savefig(
            f"{plot_dir}/encoder_decoder_analysis.png", dpi=300, bbox_inches="tight"
        )
        fig2.savefig(f"{plot_dir}/operation_analysis.png", dpi=300, bbox_inches="tight")
        # fig3.savefig(f"{plot_dir}/correlation_analysis.png", dpi=300, bbox_inches='tight')
        fig4.savefig(
            f"{plot_dir}/performance_analysis.png", dpi=300, bbox_inches="tight"
        )
        fig5.savefig(
            f"{plot_dir}/architecture_space_exploration.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig6.savefig(
            f"{plot_dir}/metric_architecture_correlations.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig7.savefig(
            f"{plot_dir}/architecture_configurations.png", dpi=300, bbox_inches="tight"
        )
        # Save analysis data
        analyzer.analysis_df.to_csv(f"{plot_dir}/analysis_data.csv", index=False)

        # Save correlation analysis
        corr_df = analyzer.analyze_correlations()
        if not corr_df.empty:
            corr_df.to_csv(f"{plot_dir}/correlations.csv", index=False)

        # Save patterns data
        import json

        with open(f"{plot_dir}/analysis_patterns.json", "w") as f:
            # Convert numpy values to regular Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            json.dump(convert_numpy(analyzer.patterns), f, indent=2)

    # Generate and display insights report
    print("📝 Generating insights report...")
    report = analyzer.generate_insights_report()
    print(report)

    if save_plots:
        with open(f"{plot_dir}/insights_report.txt", "w") as f:
            f.write(report)
        print(f"\n💾 All analysis saved to {plot_dir}/")
        print("📁 Files created:")
        print("  • encoder_decoder_analysis.png - Encoder/decoder/attention patterns")
        print("  • operation_analysis.png - Operation usage and performance impact")
        print("  • correlation_analysis.png - Architecture-performance correlations")
        print(
            "  • performance_analysis.png - Performance distributions and success rates"
        )
        print("  • analysis_data.csv - Raw analysis dataframe")
        print("  • correlations.csv - Detailed correlation results")
        print("  • analysis_patterns.json - Structured pattern analysis results")
        print("  • insights_report.txt - Generated insights and recommendations")
        plt.close("all")
    else:
        plt.show()

    return analyzer


analyze_enhanced_darts_search = analyze_darts_search

