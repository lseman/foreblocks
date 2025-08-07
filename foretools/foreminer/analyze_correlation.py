from typing import Any, Dict

import numpy as np
import pandas as pd

from .foreminer_aux import *


class CorrelationAnalyzer(AnalysisStrategy):
    """Advanced correlation analysis with multiple methods"""

    @property
    def name(self) -> str:
        return "correlations"

    def __init__(self):
        self.strategies = {
            "pearson": PearsonCorrelation(),
            "spearman": SpearmanCorrelation(),
            "mutual_info": MutualInfoCorrelation(),
            "distance": DistanceCorrelation(),
        }
        if OPTIONAL_IMPORTS["phik"]:
            self.strategies["phik"] = PhiKCorrelation()

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        if numeric_data.empty:
            return {}

        results = {}
        for method, strategy in self.strategies.items():
            try:
                results[method] = strategy.compute(numeric_data)
            except Exception as e:
                print(f"Failed to compute {method} correlation: {e}")

        return results
