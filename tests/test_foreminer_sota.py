import os
import sys

import numpy as np
import pandas as pd

from foretools.foreminer.foreminer import DatasetAnalyzer

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_foreminer_sota():
    # 1. Create synthetic dataset with patterns
    n = 300
    t = np.arange(n)
    
    # Feature A: Sine wave (Motif)
    a = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.1, n)
    
    # Feature B: Lagged A (Causality)
    b = np.roll(a, 5) + np.random.normal(0, 0.1, n)
    b[:5] = np.random.normal(0, 0.1, 5)
    
    # Feature C: Random with anomalies
    c = np.random.normal(0, 1, n)
    c[50] = 10
    c[150] = -10
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "feature_a": a,
        "feature_b": b,
        "feature_c": c
    })
    
    analyzer = DatasetAnalyzer(df, time_col="timestamp", verbose=False)

    results = analyzer.analyze(["distributions", "correlations", "missingness"])

    assert {"distributions", "correlations", "missingness"} <= set(results)
    assert analyzer.get_results("distributions") is results["distributions"]
    assert analyzer.get_results("correlations") is results["correlations"]
    assert analyzer.get_results("missingness") is results["missingness"]

if __name__ == "__main__":
    test_foreminer_sota()
