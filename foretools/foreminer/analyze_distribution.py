import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, normaltest, shapiro, skew
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning

from .foreminer_aux import *

# Suppress known noise warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)


# ============================================================================
# COMPREHENSIVE ANALYSIS STRATEGIES
# ============================================================================


class DistributionAnalyzer(AnalysisStrategy):
    """Comprehensive distribution analysis with advanced statistical features"""

    @property
    def name(self) -> str:
        return "distributions"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        summary_data = []

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 8:
                continue

            try:
                stats_dict = self._compute_comprehensive_stats(col_data, col, config)
                summary_data.append(stats_dict)
            except Exception as e:
                print(f"Distribution analysis failed for {col}: {e}")

        return {"summary": pd.DataFrame(summary_data)}

    def _compute_comprehensive_stats(
        self, col_data: pd.Series, col_name: str, config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Compute comprehensive distribution statistics"""
        mean, std = col_data.mean(), col_data.std()
        min_val, max_val = col_data.min(), col_data.max()
        range_val = max_val - min_val
        cv = std / abs(mean) if mean != 0 else np.inf
        skew_val = skew(col_data)
        kurt_val = kurtosis(col_data, fisher=False)

        # Histogram for entropy
        hist, _ = np.histogram(col_data, bins=30, density=True)
        hist += 1e-10
        entr = entropy(hist, base=2)

        # Quartiles and IQR
        q1, q2, q3 = col_data.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        # Normality tests
        _, p_norm = normaltest(col_data)
        sample_size = min(5000, len(col_data))
        _, p_shapiro = shapiro(
            col_data.sample(sample_size, random_state=config.random_state)
        )

        # Advanced distribution features
        bimodality_coeff = (skew_val**2 + 1) / kurt_val if kurt_val != 0 else 0
        lower_5 = col_data.quantile(0.05)
        upper_95 = col_data.quantile(0.95)
        tail_ratio = (
            (max_val - upper_95) / (lower_5 - min_val + 1e-6)
            if (lower_5 > min_val)
            else np.nan
        )

        return {
            "feature": col_name,
            "count": len(col_data),
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "cv": cv,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "excess_kurtosis": kurt_val - 3,
            "entropy": entr,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "normaltest_p": p_norm,
            "shapiro_p": p_shapiro,
            "is_gaussian": p_norm > config.confidence_level and abs(skew_val) < 1,
            "is_skewed": abs(skew_val) > 1,
            "is_heavy_tailed": abs(kurt_val) > 3,
            "bimodality_coeff": bimodality_coeff,
            "tail_ratio": tail_ratio,
        }
