"""Core types, utilities, and optional imports for foreminer."""

import traceback
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# ── Warnings ──────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Optional imports ──────────────────────────────────────────────────────


def _import_available(import_stmt: str) -> bool:
    try:
        exec(import_stmt, globals())
        return True
    except ImportError:
        return False


OPTIONAL_IMPORTS: dict[str, bool] = {}
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


# ── Decorators ────────────────────────────────────────────────────────────


def requires_library(lib_name: str):
    """Decorator to check if optional library is available."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OPTIONAL_IMPORTS.get(lib_name, False):
                raise ImportError(f"{lib_name} not available. pip install {lib_name}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ── Reporting helpers ─────────────────────────────────────────────────────


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
    label: str, value: Any, unit: str = "", status: str | None = None, indent: int = 0
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
    print(f"{'   ' * indent}{tag}{spacer}{label:<34} : {value}{unit}")


def report_recommendation(text: str, priority: str = "normal", indent: int = 0) -> None:
    priority_tag = {
        "high": "[HIGH]",
        "medium": "[MEDIUM]",
        "low": "[LOW]",
        "normal": "[NOTE]",
    }
    tag = priority_tag.get(priority, "[NOTE]")
    print(f"{'   ' * indent}{tag:<8} {text}")


def report_pct(num: float, den: float, zeros_as: str = "0.0%") -> str:
    if not den:
        return zeros_as
    return f"{(num / den) * 100:.1f}%"


def top_list(items: list[str], n: int = 5) -> str:
    return ", ".join(items[:n]) + (
        f" ... and {len(items) - n} more" if len(items) > n else ""
    )


def quality_band(value: float) -> str:
    return (
        "excellent"
        if value > 95
        else "good" if value > 85 else "fair" if value > 70 else "poor"
    )


# ── KMedoids backend detection ───────────────────────────────────────────


def _detect_kmedoids_backend() -> tuple[bool, str | None]:
    candidates = [
        ("sklearn_extra", "from sklearn_extra.cluster import KMedoids"),
        ("pyclustering", "from pyclustering.cluster.kmedoids import kmedoids"),
        ("kmedoids", "import kmedoids"),
    ]
    for source, stmt in candidates:
        if _import_available(stmt):
            return True, source
    return False, None


# ── Optional dependency flags ────────────────────────────────────────────

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


def print_available_libraries():
    """Print which optional libraries are available."""
    for lib, status in {
        "PyOD": HAS_PYOD,
        "HDBSCAN": HAS_HDBSCAN,
        "K-medoids": f"{HAS_KMEDOIDS} ({KMEDOIDS_SOURCE})" if HAS_KMEDOIDS else False,
        "PyClustering": HAS_PYCLUSTERING,
        "UMAP": HAS_UMAP,
        "OpenTSNE": HAS_OPENTSNE,
        "TriMap": HAS_TRIMAP,
        "MulticoreTSNE": HAS_MULTICORE_TSNE,
    }.items():
        print(f"  {'✓' if status else '✗'} {lib}: {status}")


if __name__ == "__main__":
    print_available_libraries()


# ── Configuration ─────────────────────────────────────────────────────────


@dataclass
class AnalysisConfig:
    """Centralized configuration for analysis parameters."""

    confidence_level: float = 0.05
    max_clusters: int = 10
    sample_size_threshold: int = 5000
    correlation_threshold: float = 0.7
    outlier_contamination: float = 0.1
    time_series_min_periods: int = 24
    plot_style: str = "seaborn-v0_8"
    figure_size: tuple[int, int] = (12, 8)
    random_state: int = 42
    log_level: str = "INFO"
    max_workers: int | None = None


# ── Hooks ─────────────────────────────────────────────────────────────────


class AnalysisHooks:
    """Hook system for extensible analysis pipeline."""

    def __init__(self):
        self._hooks: dict[str, list[Callable]] = {}
        self._plotters: dict[str, list[Callable]] = {}

    def register_hook(self, event: str, callback: Callable) -> None:
        self._hooks.setdefault(event, []).append(callback)

    def register_plotter(self, analysis_type: str, plotter: Callable) -> None:
        self._plotters.setdefault(analysis_type, []).append(plotter)

    def trigger(self, event: str, context: dict[str, Any]) -> dict[str, Any]:
        results = {}
        for callback in self._hooks.get(event, []):
            try:
                r = callback(context)
                if r:
                    results[callback.__name__] = r
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
        for plotter in self._plotters.get(analysis_type, []):
            try:
                plotter(data, original_df, config)
            except Exception as e:
                print(f"Plotter {plotter.__name__} failed: {e}")


# ── Strategy ABC ──────────────────────────────────────────────────────────


class AnalysisStrategy(ABC):
    """Base class for all analysis strategies."""

    @abstractmethod
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> dict[str, Any]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ── Worker ────────────────────────────────────────────────────────────────


def _run_analysis_worker(strategy_name, strategy, df, config):
    try:
        result = strategy.analyze(df, config)
        return strategy_name, result, None
    except Exception as e:
        return strategy_name, None, f"{e}\n{traceback.format_exc()}"
