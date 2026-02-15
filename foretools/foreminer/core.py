import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _import_available(import_stmt: str) -> bool:
    """Execute an import statement and return availability."""
    try:
        exec(import_stmt, globals())
        return True
    except ImportError:
        return False


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
    OPTIONAL_IMPORTS[lib_name] = _import_available(import_stmt)


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


def report_section(title: str, level: int = 1) -> None:
    width = 80
    bars = {1: "=" * width, 2: "-" * width, 3: "." * width, 4: "." * width}
    prefixes = {1: "", 2: "", 3: "  ", 4: "    "}
    print()
    if level == 1:
        print(bars[1])
        print(f"{prefixes[1]}{title.upper()}")
        print(bars[1])
    elif level == 2:
        print(f"{prefixes[2]}{title.upper()}")
        print(bars[2])
    elif level == 3:
        print(f"{prefixes[3]}{title}")
        print(prefixes[3] + bars[3])
    else:
        print(f"{prefixes[4]}{title}")
        print(prefixes[4] + bars[4])


def report_metric(
    label: str,
    value: Any,
    unit: str = "",
    status: Optional[str] = None,
    indent: int = 0,
) -> None:
    status_tag = {
        "excellent": "[EXCELLENT]",
        "good": "[GOOD]",
        "fair": "[FAIR]",
        "poor": "[POOR]",
        "warning": "[WARNING]",
        "info": "[INFO]",
    }
    tag = status_tag.get(status, "")
    spacer = " " * (12 - len(tag)) if tag else " " * 12
    value_text = f"{value}{unit}"
    print(f"{'   '*indent}{tag}{spacer}{label:<34} : {value_text}")


def report_recommendation(text: str, priority: str = "normal", indent: int = 0) -> None:
    priority_tag = {
        "high": "[HIGH]",
        "medium": "[MEDIUM]",
        "low": "[LOW]",
        "normal": "[NOTE]",
    }
    tag = priority_tag.get(priority, "[NOTE]")
    print(f"{'   '*indent}{tag:<8} {text}")


def report_pct(num: float, den: float, zeros_as: str = "0.0%") -> str:
    if not den:
        return zeros_as
    return f"{(num/den)*100:.1f}%"


def top_list(items: List[str], n: int = 5) -> str:
    return ", ".join(items[:n]) + (
        f" ... and {len(items)-n} more" if len(items) > n else ""
    )


def quality_band(value: float) -> str:
    return (
        "excellent"
        if value > 95
        else "good" if value > 85 else "fair" if value > 70 else "poor"
    )


def _detect_kmedoids_backend() -> Tuple[bool, Optional[str]]:
    candidates = [
        ("sklearn_extra", "from sklearn_extra.cluster import KMedoids"),
        ("pyclustering", "from pyclustering.cluster.kmedoids import kmedoids"),
        ("kmedoids", "import kmedoids"),
    ]
    for source, stmt in candidates:
        if _import_available(stmt):
            return True, source
    return False, None


# Optional dependencies with corrected import names
HAS_PYOD = _import_available("import pyod")
HAS_HDBSCAN = _import_available("import hdbscan")
HAS_KMEDOIDS, KMEDOIDS_SOURCE = _detect_kmedoids_backend()
HAS_PYCLUSTERING = _import_available(
    "from pyclustering.cluster.dbscan import dbscan\nfrom pyclustering.cluster.kmeans import kmeans"
)
HAS_UMAP = _import_available("import umap")
HAS_OPENTSNE = _import_available("import openTSNE")
HAS_TRIMAP = _import_available("import trimap")
HAS_MULTICORE_TSNE = _import_available("from MulticoreTSNE import MulticoreTSNE")

# Optional: Print available libraries for debugging
def print_available_libraries():
    """Print which optional libraries are available."""
    libraries = {
        'PyOD (Outlier Detection)': HAS_PYOD,
        'HDBSCAN': HAS_HDBSCAN,
        'K-medoids': f"{HAS_KMEDOIDS} ({KMEDOIDS_SOURCE})" if HAS_KMEDOIDS else False,
        'PyClustering': HAS_PYCLUSTERING,
        'UMAP': HAS_UMAP,
        'OpenTSNE': HAS_OPENTSNE,
        'TriMap': HAS_TRIMAP,
        'MulticoreTSNE': HAS_MULTICORE_TSNE,
    }
    
    print("Available optional libraries:")
    for lib, status in libraries.items():
        status_str = "✓" if status else "✗"
        print(f"  {status_str} {lib}: {status}")

if __name__ == "__main__":
    print_available_libraries()



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
        """Plot outlier detection results using PCA projection (robust)."""
        if not data:
            return

        # If caller passed the whole outlier block, dive into the methods dict.
        if "outlier_results" in data and isinstance(data["outlier_results"], dict):
            data = data["outlier_results"]

        # Pick the first method that contains an 'outliers' mask
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

        # Numeric data + NaN alignment
        numeric = original_df.select_dtypes(include=[np.number])
        mask = numeric.notna().all(axis=1)
        clean = numeric[mask]

        # If lengths mismatch, try to trim to min length
        if len(outliers) != len(numeric):
            m = min(len(outliers), len(numeric))
            outliers = outliers[:m]
            mask = mask.iloc[:m]
            clean = numeric.iloc[:m][mask.iloc[:m]]

        aligned_outliers = outliers[mask.values]

        if len(aligned_outliers) != len(clean):
            # As a last resort: bail gracefully
            print("Outlier mask length does not align with cleaned data.")
            return

        # PCA projection
        pca = PCA(n_components=2, random_state=config.random_state)
        pca_data = pca.fit_transform(StandardScaler().fit_transform(clean))

        plt.figure(figsize=config.figure_size)
        colors = np.where(aligned_outliers, "red", "blue")
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.6, s=50)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title(f"Outlier Detection: {method.replace('_', ' ').title()}")

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
        """Plot clustering results (robust to missing labels/mismatch)."""
        if not data:
            return

        clean = original_df.select_dtypes(include=[np.number]).dropna()
        X2 = PCA(n_components=2, random_state=config.random_state).fit_transform(
            StandardScaler().fit_transform(clean)
        )

        # Only keep methods that have a valid labels array of matching length
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
            # Match length to X2 if needed (defensive)
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
                X2m[:, 0], X2m[:, 1],
                c=labels, cmap="viridis", alpha=0.7, s=50
            )

            # Compute cluster count if not present
            if "n_clusters" in res:
                ncl = res["n_clusters"]
            elif "best_k" in res:
                ncl = res["best_k"]
            else:
                u = np.unique(labels)
                ncl = int(len(u) - (1 if -1 in u else 0))

            title = f"{method.replace('_',' ').title()} • clusters={ncl}"
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
        data: Dict[str, Any],
        original_df: pd.DataFrame,
        config: "AnalysisConfig",
        label_key: str = "labels",  # or "cluster_labels" if that's what you use
    ):
        """
        Plot available 2D embeddings from the dimensionality analyzer.
        Supports keys: 'pca', 'umap', 'tsne', each with data[k]['embedding'] = (n, d<=2+).
        Gracefully handles 1D embeddings and missing labels.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Collect available embeddings
        candidates = []
        for name in ["pca", "umap", "tsne"]:
            if name in data and isinstance(data[name], dict) and "embedding" in data[name]:
                X = np.asarray(data[name]["embedding"])
                if X.size == 0:
                    continue
                # Ensure 2D array shape (n, d)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                candidates.append((name.upper(), X, data[name]))

        if not candidates:
            print("No dimensionality embeddings found to plot.")
            return

        # Optional labels (same length as rows in embeddings)
        labels = None
        if label_key in data and data[label_key] is not None:
            labels = np.asarray(data[label_key])
        elif label_key in original_df.columns:
            labels = original_df[label_key].to_numpy()

        # Style
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
            # If embedding is 1D, pad a zero column to avoid "tuple index out of range"
            if X2d.shape[1] == 1:
                X2d = np.c_[X2d[:, 0], np.zeros(len(X2d))]
            elif X2d.shape[1] > 2:
                X2d = X2d[:, :2]

            if y is None or (hasattr(y, "__len__") and len(y) != len(X2d)):
                ax.scatter(X2d[:, 0], X2d[:, 1], s=10, alpha=0.8)
            else:
                # Categorical coloring without seaborn
                y_arr = np.asarray(y)
                # Make sure it's same length
                if len(y_arr) != len(X2d):
                    y_arr = None
                    ax.scatter(X2d[:, 0], X2d[:, 1], s=10, alpha=0.8)
                else:
                    # Encode categories to ints for coloring
                    _, y_codes = np.unique(y_arr, return_inverse=True)
                    sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=y_codes, s=10, alpha=0.85)
                    # Optional legend with up to 10 classes
                    uniq = np.unique(y_arr)
                    if len(uniq) <= 10:
                        handles = []
                        for i, u in enumerate(uniq[:10]):
                            handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                                      label=str(u)))
                        ax.legend(handles=handles, title="labels", loc="best", fontsize=8)

            ax.set_title(title)
            ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2"); ax.grid(True, alpha=0.2)

        for i, (name, X, meta) in enumerate(candidates):
            ax = axes[i]
            title = name
            # Add explained variance info if PCA
            if name == "PCA" and "explained_variance_ratio" in meta:
                r = np.asarray(meta["explained_variance_ratio"])
                if r.size >= 2:
                    title += f" ({r[:2].sum():.2%} var)"
                elif r.size == 1:
                    title += f" ({r[0]:.2%} var)"
            _scatter2d(ax, X, labels, title)

        # Hide any extra axes
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
