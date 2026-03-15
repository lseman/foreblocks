import pandas as pd
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foretools.foreminer.foreminer import DatasetAnalyzer

def test_foreminer_sota():
    print("🚀 Starting ForeMiner SOTA Verification...")
    
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
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="H"),
        "feature_a": a,
        "feature_b": b,
        "feature_c": c
    })
    
    print("🔧 Running DatasetAnalyzer...")
    analyzer = DatasetAnalyzer(df, time_col="timestamp", verbose=False)
    
    # Run intelligence summary
    summary = analyzer.analyze_intelligent_summary()
    print("\n--- Summary Output ---")
    print(summary)
    
    # Run full insights
    print("\n--- Detailed Insights ---")
    analyzer.print_detailed_insights()
    
    print("\n✅ Verification Complete!")

if __name__ == "__main__":
    test_foreminer_sota()
