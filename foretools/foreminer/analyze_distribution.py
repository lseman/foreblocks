import warnings
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

from .foreminer_aux import *

# —————————— Warnings ——————————
for cat in (RuntimeWarning, FutureWarning, UserWarning, ConvergenceWarning, ValueWarning):
    warnings.filterwarnings("ignore", category=cat)


class DistributionAnalyzer(AnalysisStrategy):
    """SOTA Distribution Analyzer with advanced statistical diagnostics."""

    @property
    def name(self) -> str:
        return "distributions"

    # —————————— Public API ——————————
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_cols: List[str] = data.select_dtypes(include=[np.number]).columns.tolist()
        rows: List[Dict[str, Any]] = []

        for col in numeric_cols:
            x = data[col].dropna()
            if len(x) < 8:
                continue
            try:
                rows.append(self._compute_stats(x, col, config))
            except Exception as e:
                # Keep the same behavior: log and continue
                print(f"[⚠️] Failed to analyze {col}: {e}")

            # Optional: if you want to be quieter, just pass
            # except Exception:
            #     pass

        return {"summary": pd.DataFrame(rows)}

    # —————————— Helpers ——————————
    @staticmethod
    def _freedman_bins(x: pd.Series, max_bins: int = 80) -> int:
        """Freedman–Diaconis rule, capped for stability."""
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        n = len(x)
        if iqr <= 0 or n <= 1:
            return 30  # fallback
        bin_width = 2 * iqr / (n ** (1 / 3))
        if bin_width <= 0:
            return 30
        bins = int(np.ceil((x.max() - x.min()) / bin_width))
        return int(np.clip(bins, 10, max_bins))

    @staticmethod
    def _hist_entropy(x: pd.Series) -> float:
        """Histogram-based entropy with small epsilon for numerical stability."""
        bins = DistributionAnalyzer._freedman_bins(x)
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist.astype(float) + 1e-12
        return float(entropy(hist, base=2))

    @staticmethod
    def _normality_tests(x: pd.Series, sample: pd.Series) -> Dict[str, float]:
        """Run multiple normality tests safely; return NaN on failure."""
        def _safe(callable_):
            try:
                return float(callable_())
            except Exception:
                return float("nan")

        p_norm = _safe(lambda: normaltest(x)[1])
        p_shap = _safe(lambda: shapiro(sample)[1])
        p_jb = _safe(lambda: jarque_bera(x)[1])
        anderson_stat = _safe(lambda: anderson(x).statistic)

        return {
            "normaltest_p": p_norm,
            "shapiro_p": p_shap,
            "jarque_bera_p": p_jb,
            "anderson_stat": anderson_stat,
        }

    @staticmethod
    def _z_outlier_pct(x: pd.Series, mean: float, std: float, thr: float = 3.0) -> float:
        if not np.isfinite(std) or std == 0:
            return 0.0
        z = (x - mean) / std
        return float((z.abs() > thr).mean() * 100)

    # —————————— Core computation ——————————
    def _compute_stats(self, x: pd.Series, col: str, cfg: AnalysisConfig) -> Dict[str, Any]:
        x = x.dropna()
        n = len(x)
        sample = x.sample(min(n, 5000), random_state=cfg.random_state)

        # Central tendency & dispersion
        mean = float(x.mean())
        std = float(x.std())
        min_val = float(x.min())
        max_val = float(x.max())
        value_range = max_val - min_val
        cv = (std / abs(mean)) if mean != 0 else np.nan

        # Shape
        skew_val = float(skew(x))
        kurt_val = float(kurtosis(x, fisher=False))
        excess_kurt = kurt_val - 3.0
        bimodality_coeff = (skew_val**2 + 1.0) / kurt_val if kurt_val != 0 else np.nan

        # Entropy (histogram)
        ent = self._hist_entropy(x)

        # Quantiles and tails
        q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
        iqr = float(q3 - q1)
        lower_5 = float(x.quantile(0.05))
        upper_95 = float(x.quantile(0.95))
        denom = (lower_5 - min_val)
        tail_ratio = ((max_val - upper_95) / (denom + 1e-12)) if denom > 0 else np.nan

        # Normality tests
        norm_tests = self._normality_tests(x, sample)

        # Flags (keep same logic)
        is_gaussian = (
            norm_tests["normaltest_p"] > cfg.confidence_level and abs(skew_val) < 1
        )
        is_skewed = abs(skew_val) > 1
        is_heavy_tailed = kurt_val > 3

        # Outliers (z > 3)
        outlier_pct = self._z_outlier_pct(x, mean, std, thr=3.0)

        return {
            "feature": col,
            "count": n,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "range": value_range,
            "cv": cv,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "excess_kurtosis": excess_kurt,
            "bimodality_coeff": bimodality_coeff,
            "entropy": ent,
            "q1": float(q1),
            "median": float(q2),
            "q3": float(q3),
            "iqr": iqr,
            "tail_ratio": float(tail_ratio),
            **norm_tests,
            "is_gaussian": bool(is_gaussian),
            "is_skewed": bool(is_skewed),
            "is_heavy_tailed": bool(is_heavy_tailed),
            "outlier_pct_z>3": outlier_pct,
        }
