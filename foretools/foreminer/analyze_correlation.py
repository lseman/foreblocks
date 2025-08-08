from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

from .foreminer_aux import *

try:
    import phik
    HAS_PHIK = True
except ImportError:
    HAS_PHIK = False


class CorrelationAnalyzer(AnalysisStrategy):
    """Advanced correlation analysis with multiple methods"""

    @property
    def name(self) -> str:
        return "correlations"

    def __init__(self):
        self.strategies = {
            "pearson": self._pearson_correlation,
            "spearman": self._spearman_correlation,
            "mutual_info": self._mutual_info_correlation,
            "distance": self._distance_correlation,
        }
        if HAS_PHIK:
            self.strategies["phik"] = self._phik_correlation

    # ==============================
    # Correlation computation methods
    # ==============================

    def _pearson_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Pearson correlation."""
        return df.corr(method="pearson")

    def _spearman_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Spearman correlation."""
        return df.corr(method="spearman")

    def _mutual_info_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mutual information-based correlation matrix."""
        cols = df.columns
        mi_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))),
                                 index=cols, columns=cols)
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i >= j:
                    continue
                mi = mutual_info_regression(df[[col1]], df[col2])[0]
                mi_matrix.loc[col1, col2] = mi
                mi_matrix.loc[col2, col1] = mi
        return mi_matrix

    def _distance_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute distance correlation matrix."""
        def distance_corr(x, y):
            x = np.atleast_2d(x).T
            y = np.atleast_2d(y).T
            a = squareform(pdist(x, 'euclidean'))
            b = squareform(pdist(y, 'euclidean'))
            A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
            dcov2_xy = (A * B).mean()
            dcov2_xx = (A * A).mean()
            dcov2_yy = (B * B).mean()
            return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx * dcov2_yy))

        cols = df.columns
        dist_corr_matrix = pd.DataFrame(np.eye(len(cols)),
                                        index=cols, columns=cols)
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i >= j:
                    continue
                dc = distance_corr(df[col1].values, df[col2].values)
                dist_corr_matrix.loc[col1, col2] = dc
                dist_corr_matrix.loc[col2, col1] = dc
        return dist_corr_matrix

    def _phik_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute PhiK correlation (if available)."""
        return df.phik_matrix()

    # ==============================
    # Main analysis entry point
    # ==============================

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        if numeric_data.empty:
            return {}

        results = {}
        for method, func in self.strategies.items():
            try:
                results[method] = func(numeric_data)
            except Exception as e:
                print(f"Failed to compute {method} correlation: {e}")

        return results
