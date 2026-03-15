import pandas as pd
import numpy as np
from foretools.fengineer import FeatureEngineer, FeatureConfig

def test_sota_fengineer():
    # Create sample time series data
    np.random.seed(42)
    n_samples = 200
    time = np.arange(n_samples)
    
    # Trend + Seasonal + Noise
    trend = 0.5 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 12)
    noise = np.random.normal(0, 1, n_samples)
    y = trend + seasonal + noise
    
    X = pd.DataFrame({
        "feat1": y + np.random.normal(0, 0.5, n_samples),
        "feat2": np.random.randn(n_samples)
    })
    
    # Configure with SOTA flags
    config = FeatureConfig(
        create_rocket=True,
        create_decomposition=True,
        create_adaptive_lags=True,
        rocket_n_kernels=50 # Small for test
    )
    
    # Initialize and Fit
    fe = FeatureEngineer(config)
    print("Fitting SOTA FeatureEngineer...")
    fe.fit(X, y)
    
    # Transform
    X_trans = fe.transform(X)
    print("Transformed shape:", X_trans.shape)
    print("Sample columns:", X_trans.columns[:20].tolist())
    
    # Basic Checks
    assert any("trend" in col for col in X_trans.columns)
    assert any("seasonal" in col for col in X_trans.columns)
    assert any("rk" in col for col in X_trans.columns)
    assert any("lag" in col for col in X_trans.columns)
    
    print("SOTA Feature Engineering Verification Passed!")

if __name__ == "__main__":
    test_sota_fengineer()
