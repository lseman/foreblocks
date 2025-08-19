import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import (
    anderson,
    entropy,
    jarque_bera,
    kurtosis,
    normaltest,
    shapiro,
    skew,
)
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress warnings
for cat in (RuntimeWarning, FutureWarning, UserWarning, ConvergenceWarning, ValueWarning):
    warnings.filterwarnings("ignore", category=cat)


class DistributionAnalyzer:
    """Optimized Distribution Analyzer with batched + parallel computations."""

    @property
    def name(self) -> str:
        return "distributions"

    def analyze(self, data: pd.DataFrame, config) -> Dict[str, Any]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Pre-filter columns with sufficient data
        valid_cols = [c for c in numeric_cols if data[c].count() >= 8]
        if not valid_cols:
            return {"summary": pd.DataFrame()}
        
        # Parallel compute stats
        rows = self._batch_compute_stats(data[valid_cols], config)
        return {"summary": pd.DataFrame(rows)}

    def _batch_compute_stats(self, data: pd.DataFrame, cfg) -> List[Dict[str, Any]]:
        """Parallel batch compute statistics for all columns."""
        results = []
        with ThreadPoolExecutor() as ex:
            futures = {ex.submit(self._compute_one_col, data[col], col, cfg): col for col in data.columns}
            for fut in as_completed(futures):
                results.append(fut.result())
        return results

    def _compute_one_col(self, series: pd.Series, col: str, cfg) -> Dict[str, Any]:
        """Compute stats for a single column."""
        x = series.dropna()
        n = len(x)
        x_values = x.to_numpy(dtype=np.float64, copy=False)

        # Quantiles (NumPy version is faster than pandas)
        q05, q25, q50, q75, q95 = np.quantile(x_values, [0.05, 0.25, 0.5, 0.75, 0.95])

        # Basic stats
        mean_val = float(np.mean(x_values))
        std_val = float(np.std(x_values, ddof=1))
        min_val = float(np.min(x_values))
        max_val = float(np.max(x_values))

        # Pre-compute common values
        value_range = max_val - min_val
        cv = (std_val / abs(mean_val)) if mean_val != 0 else np.nan
        iqr = float(q75 - q25)

        # Shape statistics
        skew_val = float(skew(x_values))
        kurt_val = float(kurtosis(x_values, fisher=False))
        excess_kurt = kurt_val - 3.0
        bimodality_coeff = (skew_val**2 + 1.0) / kurt_val if kurt_val != 0 else np.nan

        # Entropy
        ent = self._fast_hist_entropy(x_values)

        # Tail ratio
        denom = q05 - min_val
        tail_ratio = ((max_val - q95) / (denom + 1e-12)) if denom > 0 else np.nan

        # Normality tests (sample if needed)
        sample_size = min(n, 5000)
        if sample_size < n:
            sample_idx = np.random.RandomState(cfg.random_state).choice(n, sample_size, replace=False)
            sample = x_values[sample_idx]
        else:
            sample = x_values
        norm_tests = self._fast_normality_tests(x_values, sample)

        # Flags
        is_gaussian = (
            norm_tests["normaltest_p"] > getattr(cfg, 'confidence_level', 0.05) and 
            abs(skew_val) < 1
        )
        is_skewed = abs(skew_val) > 1
        is_heavy_tailed = kurt_val > 3

        # Outliers
        outlier_pct = self._fast_z_outlier_pct(x_values, mean_val, std_val)

        return {
            "feature": col,
            "count": n,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "range": value_range,
            "cv": cv,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "excess_kurtosis": excess_kurt,
            "bimodality_coeff": bimodality_coeff,
            "entropy": ent,
            "q1": float(q25),
            "median": float(q50),
            "q3": float(q75),
            "iqr": iqr,
            "tail_ratio": float(tail_ratio),
            **norm_tests,
            "is_gaussian": bool(is_gaussian),
            "is_skewed": bool(is_skewed),
            "is_heavy_tailed": bool(is_heavy_tailed),
            "outlier_pct_z>3": outlier_pct,
        }

    @staticmethod
    def _fast_hist_entropy(x_values: np.ndarray) -> float:
        """Fast histogram entropy using numpy operations."""
        try:
            n = len(x_values)
            bins = max(10, min(int(np.sqrt(n)), 80))
            hist, _ = np.histogram(x_values, bins=bins, density=True)
            hist = hist + 1e-12
            return float(entropy(hist, base=2))
        except Exception:
            return np.nan

    @staticmethod
    def _fast_normality_tests(x: np.ndarray, sample: np.ndarray) -> Dict[str, float]:
        """Optimized normality tests with better error handling."""
        results = {}
        tests = {
            "normaltest_p": lambda: normaltest(x)[1],
            "shapiro_p": lambda: shapiro(sample)[1] if len(sample) <= 5000 else np.nan,
            "jarque_bera_p": lambda: jarque_bera(x)[1],
            "anderson_stat": lambda: anderson(x).statistic,
        }
        for key, test_func in tests.items():
            try:
                results[key] = float(test_func())
            except Exception:
                results[key] = np.nan
        return results

    @staticmethod
    def _fast_z_outlier_pct(x: np.ndarray, mean: float, std: float, thr: float = 3.0) -> float:
        """Vectorized outlier percentage calculation."""
        if not np.isfinite(std) or std == 0:
            return 0.0
        z = np.abs((x - mean) / std)
        return float(np.mean(z > thr) * 100)
