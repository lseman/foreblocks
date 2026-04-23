import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from foretools.aux.adaptive_mi import AdaptiveMI


class CorrelationFilter:
    """Removes highly correlated features with multiple selection strategies."""

    def __init__(
        self,
        threshold: float = 0.95,
        method: str = "variance",
        dependence_metric: str = "pearson",
        min_features: int = 2,
        handle_missing: bool = True,
        random_state: int = 42,
        mi_subsample: int = 2000,
        mi_min_overlap: int = 50,
        mi_bins: int = 16,
    ):
        """
        Args:
            threshold: Correlation threshold above which to remove features
            method: Strategy for choosing which feature to drop ('variance', 'target_corr', 'random')
            dependence_metric: Dependence measure used to find redundant pairs ('pearson', 'adaptive_mi')
            min_features: Minimum number of features to keep
            handle_missing: Whether to handle missing values before correlation
        """
        self.threshold = threshold
        self.method = method
        self.dependence_metric = str(dependence_metric).lower()
        self.min_features = min_features
        self.handle_missing = handle_missing
        self.features_to_drop_: list[str] = []
        self.correlation_pairs_: list[tuple[str, str, float]] = []
        self.feature_rankings_: dict[str, float] = {}
        self.random_state = int(random_state)
        self.mi_subsample = int(mi_subsample)
        self.mi_min_overlap = int(mi_min_overlap)
        self.mi_bins = int(mi_bins)
        self.ami_scorer_ = AdaptiveMI(
            subsample=self.mi_subsample,
            spearman_gate=0.0,
            min_overlap=self.mi_min_overlap,
            ks=(3, 5, 10),
            n_bins=self.mi_bins,
            random_state=self.random_state,
        )

    def _prepare_target(self, y: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(y):
            return pd.to_numeric(y, errors="coerce")

        y_str = y.astype("string").fillna("MISSING")
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y_str)
        return pd.Series(encoded, index=y.index, dtype="int32")

    def _compute_dependence_matrix(self, X_corr: pd.DataFrame) -> pd.DataFrame:
        if self.dependence_metric == "adaptive_mi":
            return self.ami_scorer_.matrix(X_corr).fillna(0.0)
        return X_corr.corr().fillna(0.0)

    def _score_against_target(self, feature: pd.Series, y: pd.Series) -> float:
        if self.dependence_metric == "adaptive_mi":
            return float(
                self.ami_scorer_.score(
                    feature.to_numpy(copy=False),
                    y.to_numpy(copy=False),
                )
            )

        return abs(float(feature.corr(y)))

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CorrelationFilter":
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

            dep_matrix = self._compute_dependence_matrix(X_corr)
            dep_matrix = dep_matrix.fillna(0.0)

            y_aligned = None
            if y is not None:
                common_idx = X_corr.index.intersection(y.index)
                if len(common_idx) > 0:
                    X_corr = X_corr.loc[common_idx]
                    y_aligned = self._prepare_target(y.loc[common_idx])

            # Find highly correlated pairs
            upper_tri = np.triu(np.ones_like(dep_matrix, dtype=bool), k=1)
            high_corr_mask = (dep_matrix.abs() > self.threshold) & upper_tri
            high_corr_pairs = np.where(high_corr_mask)

            # Store correlation pairs for inspection
            self.correlation_pairs_ = []
            feature_drop_scores = {}
            target_scores: dict[str, float] = {}

            for i, j in zip(*high_corr_pairs):
                col1, col2 = dep_matrix.index[i], dep_matrix.columns[j]
                corr_value = dep_matrix.iloc[i, j]
                self.correlation_pairs_.append((col1, col2, abs(corr_value)))

                # Calculate drop scores based on method
                if self.method == "variance":
                    score1 = X_corr[col1].var()
                    score2 = X_corr[col2].var()
                    keep_col = col1 if score1 > score2 else col2
                    drop_col = col2 if score1 > score2 else col1

                elif self.method == "target_corr" and y_aligned is not None:
                    # Keep feature more correlated with target
                    try:
                        if col1 not in target_scores:
                            target_scores[col1] = self._score_against_target(
                                X_corr[col1], y_aligned
                            )
                        if col2 not in target_scores:
                            target_scores[col2] = self._score_against_target(
                                X_corr[col2], y_aligned
                            )
                        score1 = target_scores[col1]
                        score2 = target_scores[col2]
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
                feature_drop_scores[drop_col] = feature_drop_scores.get(drop_col, 0) + 1

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
