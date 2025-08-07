import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress known noise warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

def _run_analysis_worker(strategy_name, strategy, df, config):
    try:
        result = strategy.analyze(df, config)
        return strategy_name, result, None
    except Exception as e:
        tb = traceback.format_exc()
        return strategy_name, None, f"{e}\n{tb}"
    
# Optional imports with graceful fallbacks
OPTIONAL_IMPORTS = {}
for lib_name, import_stmt in [
    ("phik", "import phik"),
    ("networkx", "import networkx as nx"),
    ("umap", "from umap import UMAP"),
    ("hdbscan", "from hdbscan import HDBSCAN"),
    ("hierarchy", "from scipy.cluster.hierarchy import dendrogram, linkage, fcluster"),
    ("shap", "import shap"),
    ("missingno", "import missingno as msno"),
    ("mice", "from statsmodels.imputation.mice import MICEData"),
]:
    try:
        exec(import_stmt)
        OPTIONAL_IMPORTS[lib_name] = True
    except ImportError:
        OPTIONAL_IMPORTS[lib_name] = False


def requires_library(lib_name: str):
    """Decorator to check if optional library is available"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OPTIONAL_IMPORTS.get(lib_name, False):
                raise ImportError(
                    f"{lib_name} not available. Install with: pip install {lib_name}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator

warnings.filterwarnings("ignore")

# Optional dependencies
try:
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False

try:
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

try:
    HAS_PYCLUSTERING = True
except ImportError:
    HAS_PYCLUSTERING = False

try:
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    HAS_OPENTSNE = True
except ImportError:
    HAS_OPENTSNE = False

try:
    HAS_TRIMAP = True
except ImportError:
    HAS_TRIMAP = False

try:
    HAS_MULTICORE_TSNE = True
except ImportError:
    HAS_MULTICORE_TSNE = False



# ============================================================================
# CORE CONFIGURATION & UTILITIES
# ============================================================================


@dataclass
class AnalysisConfig:
    """Centralized configuration for analysis parameters"""

    confidence_level: float = 0.05
    max_clusters: int = 10
    sample_size_threshold: int = 5000
    correlation_threshold: float = 0.7
    outlier_contamination: float = 0.1
    time_series_min_periods: int = 24
    plot_style: str = "seaborn-v0_8"
    figure_size: Tuple[int, int] = (12, 8)
    random_state: int = 42


class AnalysisHooks:
    """Hook system for extensible analysis pipeline"""

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {}
        self._plotters: Dict[str, List[Callable]] = {}

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for an analysis event"""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def register_plotter(self, analysis_type: str, plotter: Callable) -> None:
        """Register a plotter for an analysis type"""
        if analysis_type not in self._plotters:
            self._plotters[analysis_type] = []
        self._plotters[analysis_type].append(plotter)

    def trigger(self, event: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger all hooks for an event"""
        results = {}
        for callback in self._hooks.get(event, []):
            try:
                result = callback(context)
                if result:
                    results[callback.__name__] = result
            except Exception as e:
                print(f"Hook {callback.__name__} failed: {e}")
        return results

    def plot(
        self,
        analysis_type: str,
        data: Any,
        original_df: pd.DataFrame,
        config: AnalysisConfig,
    ) -> None:
        """Trigger all plotters for an analysis type"""
        for plotter in self._plotters.get(analysis_type, []):
            try:
                plotter(data, original_df, config)
            except Exception as e:
                print(f"Plotter {plotter.__name__} failed: {e}")


# ============================================================================
# ANALYSIS STRATEGIES (Strategy Pattern)
# ============================================================================


class AnalysisStrategy(ABC):
    """Base class for all analysis strategies"""

    @abstractmethod
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ============================================================================
# CORRELATION STRATEGIES
# ============================================================================


class CorrelationStrategy(ABC):
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class PearsonCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.corr(method="pearson")


class SpearmanCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.corr(method="spearman")


class MutualInfoCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        n_features = len(data.columns)
        mi_matrix = np.zeros((n_features, n_features))

        for i, col_i in enumerate(data.columns):
            for j, col_j in enumerate(data.columns):
                if i == j:
                    mi_matrix[i, j] = 1.0
                else:
                    X = data[[col_i]].fillna(data[col_i].median())
                    y = data[col_j].fillna(data[col_j].median())
                    mi_matrix[i, j] = mutual_info_regression(X, y, random_state=42)[0]

        return pd.DataFrame(mi_matrix, index=data.columns, columns=data.columns)


class DistanceCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        def dcorr(X: np.ndarray, Y: np.ndarray) -> float:
            n = len(X)
            a = squareform(pdist(X.reshape(-1, 1), "euclidean"))
            b = squareform(pdist(Y.reshape(-1, 1), "euclidean"))

            A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()

            dcov2_xy = (A * B).sum() / (n * n)
            dcov2_xx = (A * A).sum() / (n * n)
            dcov2_yy = (B * B).sum() / (n * n)

            return (
                np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx * dcov2_yy))
                if dcov2_xx * dcov2_yy > 0
                else 0
            )

        n_features = len(data.columns)
        matrix = np.zeros((n_features, n_features))

        for i, col_i in enumerate(data.columns):
            for j, col_j in enumerate(data.columns):
                matrix[i, j] = dcorr(
                    data[col_i].dropna().values, data[col_j].dropna().values
                )

        return pd.DataFrame(matrix, index=data.columns, columns=data.columns)


@requires_library("phik")
class PhiKCorrelation(CorrelationStrategy):
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.phik_matrix()


# ============================================================================
# COMPREHENSIVE PLOTTING HELPERS
# ============================================================================

class PlotHelper:
    """Comprehensive plotting utilities for all analysis types"""

    @staticmethod
    def setup_style(config: AnalysisConfig):
        plt.style.use(config.plot_style)

    @staticmethod
    def plot_distributions(
        data: Dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot enhanced distribution analysis results"""
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

            # Enhanced histogram with KDE
            sns.histplot(col_data, kde=True, stat="density", alpha=0.7, ax=axes[i])

            # Add mean and median lines
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

        # Hide unused subplots
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlations(
        data: Dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot comprehensive correlation matrices"""
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
        data: Dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot outlier detection results using PCA projection"""
        if not data:
            return

        # Get the first available method
        method = list(data.keys())[0]
        outlier_info = data[method]
        outliers = np.asarray(outlier_info["outliers"])

        # Ensure alignment: drop rows with NaNs and filter outliers accordingly
        numeric_data = original_df.select_dtypes(include=[np.number])
        mask = numeric_data.notna().all(axis=1)
        clean_data = numeric_data[mask]
        aligned_outliers = outliers[mask.values]  # Filter outliers to match clean_data

        # PCA projection
        pca = PCA(n_components=2, random_state=config.random_state)
        pca_data = pca.fit_transform(StandardScaler().fit_transform(clean_data))

        plt.figure(figsize=config.figure_size)
        colors = ["red" if x else "blue" for x in aligned_outliers]

        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.6, s=50)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title(f"Outlier Detection: {method.replace('_', ' ').title()}")

        # Legend
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="Normal"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Outlier"),
        ]
        plt.legend(handles=handles)
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_clusters(
        data: Dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot comprehensive clustering results"""
        if not data:
            return

        clean_data = original_df.select_dtypes(include=[np.number]).dropna()

        # PCA for visualization
        pca = PCA(n_components=2, random_state=config.random_state)
        pca_data = pca.fit_transform(StandardScaler().fit_transform(clean_data))

        n_methods = len(data)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        axes = np.atleast_1d(axes)

        for i, (method, result) in enumerate(data.items()):
            scatter = axes[i].scatter(
                pca_data[:, 0],
                pca_data[:, 1],
                c=result["labels"],
                cmap="viridis",
                alpha=0.7,
                s=50,
            )

            title = f"{method.title()}\n"
            if "best_k" in result:
                title += f"k={result['best_k']}"
            elif "n_clusters" in result:
                title += f"clusters={result['n_clusters']}"

            if "silhouette" in result:
                title += f"\nSilhouette: {result['silhouette']:.3f}"

            axes[i].set_title(title)
            axes[i].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            axes[i].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            plt.colorbar(scatter, ax=axes[i])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dimensionality(
        data: Dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot dimensionality reduction results"""
        if not data:
            return

        n_methods = len(data)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.atleast_1d(axes).flatten()

        for i, (method, reduced_data) in enumerate(data.items()):
            scatter = axes[i].scatter(
                reduced_data[:, 0],
                reduced_data[:, 1],
                alpha=0.6,
                s=30,
                c=range(len(reduced_data)),
                cmap="viridis",
            )
            axes[i].set_title(f"{method.upper()}")
            axes[i].set_xlabel("Component 1")
            axes[i].set_ylabel("Component 2")
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    @requires_library("networkx")
    def plot_correlation_network(
        data: Dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Create correlation network graph"""
        if not data:
            return

        method = "pearson"  # Default method
        if method not in data:
            method = list(data.keys())[0]

        corr_matrix = data[method]
        threshold = config.correlation_threshold

        G = nx.Graph()

        # Add edges above threshold
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

        # Draw edges with thickness proportional to correlation
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
        data: Dict[str, Any], original_df: pd.DataFrame, config: AnalysisConfig
    ):
        """Plot missingness analysis results"""
        if not data:
            return

        # Missing rate visualization
        if "missing_rate" in data and not data["missing_rate"].empty:
            missing_rate = data["missing_rate"]

            plt.figure(figsize=(10, 6))
            missing_rate.plot(kind="bar")
            plt.title("Missing Data Rate by Feature")
            plt.ylabel("Missing Rate")
            plt.xlabel("Features")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # Missingness heatmap if missingno available
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
