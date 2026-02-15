import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class CorrelationFilter:
    """Removes highly correlated features with multiple selection strategies."""

    def __init__(
        self,
        threshold: float = 0.95,
        method: str = "variance",
        min_features: int = 2,
        handle_missing: bool = True,
    ):
        """
        Args:
            threshold: Correlation threshold above which to remove features
            method: Strategy for choosing which feature to drop ('variance', 'target_corr', 'random')
            min_features: Minimum number of features to keep
            handle_missing: Whether to handle missing values before correlation
        """
        self.threshold = threshold
        self.method = method
        self.min_features = min_features
        self.handle_missing = handle_missing
        self.features_to_drop_: List[str] = []
        self.correlation_pairs_: List[Tuple[str, str, float]] = []
        self.feature_rankings_: Dict[str, float] = {}

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CorrelationFilter":
        """Fit correlation filter."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) < 2:
            self.features_to_drop_ = []
            return self

        try:
            # Handle missing values if requested
            X_corr = X[numerical_cols].copy()
            if self.handle_missing:
                X_corr = X_corr.fillna(X_corr.median())

            # Calculate correlation matrix
            corr_matrix = X_corr.corr()

            # Handle NaN correlations (constant features)
            corr_matrix = corr_matrix.fillna(0)

            # Find highly correlated pairs
            upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            high_corr_mask = (corr_matrix.abs() > self.threshold) & upper_tri
            high_corr_pairs = np.where(high_corr_mask)

            # Store correlation pairs for inspection
            self.correlation_pairs_ = []
            feature_drop_scores = {}

            for i, j in zip(*high_corr_pairs):
                col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                self.correlation_pairs_.append((col1, col2, abs(corr_value)))

                # Calculate drop scores based on method
                if self.method == "variance":
                    score1 = X_corr[col1].var()
                    score2 = X_corr[col2].var()
                    keep_col = col1 if score1 > score2 else col2
                    drop_col = col2 if score1 > score2 else col1

                elif self.method == "target_corr" and y is not None:
                    # Keep feature more correlated with target
                    try:
                        score1 = abs(X_corr[col1].corr(y))
                        score2 = abs(X_corr[col2].corr(y))
                        keep_col = col1 if score1 > score2 else col2
                        drop_col = col2 if score1 > score2 else col1
                    except Exception:
                        # Fallback to variance if target correlation fails
                        score1 = X_corr[col1].var()
                        score2 = X_corr[col2].var()
                        keep_col = col1 if score1 > score2 else col2
                        drop_col = col2 if score1 > score2 else col1

                else:  # random or fallback
                    drop_col = np.random.choice([col1, col2])
                    keep_col = col1 if drop_col == col2 else col2

                # Track drop scores (higher score = more likely to drop)
                feature_drop_scores[drop_col] = (
                    feature_drop_scores.get(drop_col, 0) + 1
                )

            # Sort features by drop frequency and select final drops
            candidate_drops = sorted(
                feature_drop_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Ensure we don't drop too many features
            max_drops = len(numerical_cols) - self.min_features
            self.features_to_drop_ = [col for col, _ in candidate_drops[:max_drops]]

            # Store feature rankings for inspection
            remaining_features = set(numerical_cols) - set(self.features_to_drop_)
            self.feature_rankings_ = {
                col: feature_drop_scores.get(col, 0) for col in remaining_features
            }

        except Exception as e:
            warnings.warn(f"Correlation analysis failed: {e}")
            self.features_to_drop_ = []
            self.correlation_pairs_ = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove correlated features."""
        return X.drop(columns=self.features_to_drop_, errors="ignore")
