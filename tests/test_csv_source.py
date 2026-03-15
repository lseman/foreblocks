import pandas as pd
import numpy as np
import os
from foreblocks.data.csv import CSVSource

def test_csv_source():
    # Create a dummy CSV
    csv_path = "test_data.csv"
    df = pd.DataFrame({
        "time": range(100),
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "target": np.random.randn(100)
    })
    df.to_csv(csv_path, index=False)
    
    try:
        # Test Source
        source = CSVSource(file_path=csv_path, target_column="target", time_column="time")
        
        # Analyze
        info = source.load_and_analyze()
        print("CSV Info:", info)
        
        assert info["input_size"] == 2
        assert info["output_size"] == 1
        assert "feat1" in info["features"]
        assert "feat2" in info["features"]
        assert info["target"] == "target"
        
        # Forward
        X, y, t = source.forward()
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("t shape:", t.shape if t is not None else "None")
        
        assert X.shape == (100, 2)
        assert y.shape == (100, 1)
        assert t.shape == (100,)
        
        print("CSVSource Verification Passed!")
        
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

if __name__ == "__main__":
    test_csv_source()
