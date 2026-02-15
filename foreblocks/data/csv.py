import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from foreblocks.ui.node_spec import node
from foreblocks.ui.auto_spec import PortOutBundle
from typing import Annotated

@node(
    type_id="csv_source",
    name="CSV Source",
    category="Data",
    outputs=["X", "y", "time_features"],
    color="bg-gradient-to-br from-indigo-700 to-indigo-800",
)
class CSVSource:
    """
    Loads data from a CSV file and automatically detects dimensions.
    """
    
    def __init__(
        self,
        file_path: str = "",
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        sep: str = ",",
        header: int = 0,
        index_col: Optional[int] = None,
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.time_column = time_column
        self.feature_columns = feature_columns
        self.sep = sep
        self.header = header
        self.index_col = index_col
        
        self._df = None
        self._input_size = 0
        self._output_size = 0

    def load_and_analyze(self) -> Dict[str, Any]:
        """
        Loads the CSV and returns metadata for the UI (auto-detect dimensions).
        """
        if not self.file_path:
            return {"error": "No file path provided"}
        
        try:
            self._df = pd.read_csv(
                self.file_path, 
                sep=self.sep, 
                header=self.header, 
                index_col=self.index_col
            )
            
            all_cols = self._df.columns.tolist()
            
            # Auto-detect target if not set (assume last column)
            target = self.target_column or all_cols[-1]
            
            # Features = all columns except target and time
            features = self.feature_columns or [
                c for c in all_cols if c != target and c != self.time_column
            ]
            
            self._input_size = len(features)
            self._output_size = 1 # Multivariate targets support can be added later
            
            return {
                "columns": all_cols,
                "input_size": self._input_size,
                "output_size": self._output_size,
                "num_samples": len(self._df),
                "features": features,
                "target": target,
                "time": self.time_column,
                "dtypes": self._df.dtypes.apply(lambda x: str(x)).to_dict()
            }
        except Exception as e:
            return {"error": str(e)}

    def forward(self) -> Annotated[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], PortOutBundle("X", "y", "time_features")]:
        """
        Returns the processed arrays.
        """
        if self._df is None:
            res = self.load_and_analyze()
            if "error" in res:
                raise RuntimeError(f"CSV loading failed: {res['error']}")
        
        target = self.target_column or self._df.columns[-1]
        features = self.feature_columns or [
            c for c in self._df.columns if c != target and c != self.time_column
        ]
        
        X = self._df[features].values.astype(np.float32)
        y = self._df[[target]].values.astype(np.float32)
        
        time_features = None
        if self.time_column and self.time_column in self._df.columns:
            # Simple conversion to timestamps or ordinal if it's datetime
            if pd.api.types.is_datetime64_any_dtype(self._df[self.time_column]):
                time_features = self._df[self.time_column].astype(np.int64).values.astype(np.float32)
            else:
                time_features = self._df[self.time_column].values
                
        return X, y, time_features

    def py_spec(self):
        """Custom codegen for the CSV source"""
        return {
            "imports": ["from foreblocks.data.csv import CSVSource"],
            "ctor": "CSVSource",
            "var_prefix": "loader",
            "bind": {
                "kwargs": {
                    "file_path": "@config:file_path",
                    "target_column": "@config:target_column",
                    "time_column": "@config:time_column",
                    "sep": "@config:sep",
                }
            }
        }
