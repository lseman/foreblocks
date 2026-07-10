"""Plotting utilities for foreminer analysis results."""

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .core import AnalysisConfig, OPTIONAL_IMPORTS, requires_library


class PlotHelper:
    """Comprehensive plotting utilities for all analysis types."""

    @staticmethod
    def setup_style(config: AnalysisConfig):
        plt.style.use(config.plot_style)

    @staticmethod
    def plot_distributions(
        data: dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot enhanced distribution analysis results."""
        if "summary" not in data:
            return
        summary_df = data["summary"]
        if summary_df.empty:
            return

        numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
        n_features = len(numeric_cols)
        ncols = 3
        nrows = (n_features + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for i, col in enumerate(numeric_cols):
            col_data = original_df[col].dropna()
            sns.histplot(col_data, kde=True, stat="density", alpha=0.7, ax=axes[i])
            axes[i].axvline(
                col_data.mean(), color="red", linestyle="--", alpha=0.8, label="Mean"
            )
            axes[i].axvline(
                col_data.median(),
                color="orange",
                linestyle="--",
                alpha=0.8,
                label="Median",
            )
            axes[i].set_title(
                f"{col}\n(Skew: {skew(col_data):.2f}, Kurt: {kurtosis(col_data):.2f})",
                fontsize=10,
            )
            axes[i].legend(fontsize=8)

        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlations(
        data: dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot comprehensive correlation matrices."""
        if not data:
            return
        n_methods = len(data)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        axes = np.atleast_1d(axes)

        for i, (method, corr_matrix) in enumerate(data.items()):
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                ax=axes[i],
                cbar_kws={"shrink": 0.8},
            )
            axes[i].set_title(f"{method.replace('_', ' ').title()} Correlation")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_outliers_pca(
        data: dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot outlier detection results using PCA projection."""
        if not data:
            return
        if "outlier_results" in data and isinstance(data["outlier_results"], dict):
            data = data["outlier_results"]

        method = None
        outliers = None
        for k, v in data.items():
            if isinstance(v, dict) and "outliers" in v:
                outliers = np.asarray(v["outliers"], dtype=bool)
                method = k
                break
        if method is None or outliers is None:
            print("No outlier mask available to plot.")
            return

        numeric = original_df.select_dtypes(include=[np.number])
        mask = numeric.notna().all(axis=1)
        clean = numeric[mask]

        if len(outliers) != len(numeric):
            m = min(len(outliers), len(numeric))
            outliers = outliers[:m]
            mask = mask.iloc[:m]
            clean = numeric.iloc[:m][mask.iloc[:m]]
        aligned_outliers = outliers[mask.values]
        if len(aligned_outliers) != len(clean):
            print("Outlier mask length does not align with cleaned data.")
            return

        pca = PCA(n_components=2, random_state=config.random_state)
        pca_data = pca.fit_transform(StandardScaler().fit_transform(clean))

        plt.figure(figsize=config.figure_size)
        colors = np.where(aligned_outliers, "red", "blue")
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.6, s=50)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title(f"Outlier Detection: {method.replace('_', ' ').title()}")

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                label="Normal",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Outlier",
            ),
        ]
        plt.legend(handles=handles)
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_clusters(
        data: dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot clustering results (robust to missing labels/mismatch)."""
        if not data:
            return
        clean = original_df.select_dtypes(include=[np.number]).dropna()
        X2 = PCA(n_components=2, random_state=config.random_state).fit_transform(
            StandardScaler().fit_transform(clean)
        )

        usable = []
        for method, res in data.items():
            if not isinstance(res, dict):
                continue
            labels = res.get("labels")
            if labels is None:
                continue
            labels = np.asarray(labels)
            if labels.ndim != 1:
                continue
            if len(labels) != len(X2):
                m = min(len(labels), len(X2))
                if m < 2:
                    continue
                labels = labels[:m]
                X2m = X2[:m]
            else:
                X2m = X2
            usable.append((method, res, labels, X2m))

        if not usable:
            print("No cluster labels available to plot.")
            return

        n_methods = len(usable)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        axes = np.atleast_1d(axes)

        for i, (method, res, labels, X2m) in enumerate(usable):
            scatter = axes[i].scatter(
                X2m[:, 0], X2m[:, 1], c=labels, cmap="viridis", alpha=0.7, s=50
            )
            if "n_clusters" in res:
                ncl = res["n_clusters"]
            elif "best_k" in res:
                ncl = res["best_k"]
            else:
                u = np.unique(labels)
                ncl = int(len(u) - (1 if -1 in u else 0))
            title = f"{method.replace('_', ' ').title()} • clusters={ncl}"
            if "silhouette" in res and isinstance(res["silhouette"], (int, float)):
                title += f"\nSilhouette: {res['silhouette']:.3f}"
            axes[i].set_title(title)
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            plt.colorbar(scatter, ax=axes[i])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dimensionality(
        data: dict[str, Any],
        original_df: pd.DataFrame,
        config: AnalysisConfig,
        label_key: str = "labels",
    ):
        """Plot 2D embeddings from the dimensionality analyzer."""
        candidates = []
        for name in ["pca", "umap", "tsne"]:
            if (
                name in data
                and isinstance(data[name], dict)
                and "embedding" in data[name]
            ):
                X = np.asarray(data[name]["embedding"])
                if X.size == 0:
                    continue
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                candidates.append((name.upper(), X, data[name]))
        if not candidates:
            print("No dimensionality embeddings found to plot.")
            return

        labels = None
        if label_key in data and data[label_key] is not None:
            labels = np.asarray(data[label_key])
        elif label_key in original_df.columns:
            labels = original_df[label_key].to_numpy()

        try:
            plt.style.use(getattr(config, "plot_style", "default"))
        except Exception:
            pass

        n = len(candidates)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()

        def _scatter2d(ax, X2d, y=None, title=""):
            if X2d.shape[1] == 1:
                X2d = np.c_[X2d[:, 0], np.zeros(len(X2d))]
            elif X2d.shape[1] > 2:
                X2d = X2d[:, :2]

            if y is None or (hasattr(y, "__len__") and len(y) != len(X2d)):
                ax.scatter(X2d[:, 0], X2d[:, 1], s=10, alpha=0.8)
            else:
                y_arr = np.asarray(y)
                if len(y_arr) != len(X2d):
                    ax.scatter(X2d[:, 0], X2d[:, 1], s=10, alpha=0.8)
                else:
                    _, y_codes = np.unique(y_arr, return_inverse=True)
                    ax.scatter(X2d[:, 0], X2d[:, 1], c=y_codes, s=10, alpha=0.85)
                    uniq = np.unique(y_arr)
                    if len(uniq) <= 10:
                        handles = [
                            plt.Line2D([0], [0], marker="o", linestyle="", label=str(u))
                            for u in uniq[:10]
                        ]
                        ax.legend(
                            handles=handles, title="labels", loc="best", fontsize=8
                        )
            ax.set_title(title)
            ax.set_xlabel("dim-1")
            ax.set_ylabel("dim-2")
            ax.grid(True, alpha=0.2)

        last_i = 0
        for i, (name, X, meta) in enumerate(candidates):
            ax = axes[i]
            title = name
            if name == "PCA" and "explained_variance_ratio" in meta:
                r = np.asarray(meta["explained_variance_ratio"])
                if r.size >= 2:
                    title += f" ({r[:2].sum():.2%} var)"
                elif r.size == 1:
                    title += f" ({r[0]:.2%} var)"
            _scatter2d(ax, X, labels, title)
            last_i = i

        for j in range(last_i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    @requires_library("networkx")
    def plot_correlation_network(
        data: dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Create correlation network graph."""
        if not data:
            return
        method = "pearson"
        if method not in data:
            method = list(data.keys())[0]
        corr_matrix = data[method]
        threshold = config.correlation_threshold

        G = nx.Graph()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    G.add_edge(
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        weight=corr_val,
                        correlation=corr_matrix.iloc[i, j],
                    )
        if not G.edges():
            print(f"No correlations above threshold {threshold}")
            return

        plt.figure(figsize=config.figure_size)
        pos = nx.spring_layout(G, k=1, iterations=50, seed=config.random_state)
        edges = G.edges()
        weights = [G[u][v]["weight"] * 4 for u, v in edges]
        colors = ["red" if G[u][v]["correlation"] < 0 else "blue" for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color=colors)
        nx.draw_networkx_nodes(
            G, pos, node_color="lightblue", node_size=1500, alpha=0.9
        )
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        plt.title(f"Correlation Network ({method.title()}, threshold={threshold})")
        plt.axis("off")
        plt.show()

    @staticmethod
    def plot_missingness_analysis(
        data: dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot missingness analysis results."""
        if not data:
            return
        if "missing_rate" in data and not data["missing_rate"].empty:
            plt.figure(figsize=(10, 6))
            data["missing_rate"].plot(kind="bar")
            plt.title("Missing Data Rate by Feature")
            plt.ylabel("Missing Rate")
            plt.xlabel("Features")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        if OPTIONAL_IMPORTS.get("missingno"):
            try:
                import missingno as msno

                msno.matrix(original_df, figsize=(12, 6))
                plt.title("Missing Data Matrix")
                plt.tight_layout()
                plt.show()
                msno.heatmap(original_df, figsize=(10, 8))
                plt.title("Missingness Correlation Heatmap")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Missingness visualization failed: {e}")
