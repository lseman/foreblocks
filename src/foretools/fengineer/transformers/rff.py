"""Random Fourier Features (RFF) transformer.

Efficiently approximates kernel methods by mapping input features to a
higher-dimensional space using random projections and trigonometric functions.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import LabelEncoder, StandardScaler

from foretools.aux.adaptive_mi import AdaptiveMI

from .aux import BaseFeatureTransformer, require_fitted

if TYPE_CHECKING:
    pass


class RandomFourierFeaturesTransformer(BaseFeatureTransformer):
    """
    Random Fourier Features transformer for approximating RBF kernels.

    Improvements over original:
    - Fixed scaling factor for proper Monte Carlo approximation
    - Better gamma estimation using median pairwise distances
    - Target-aware feature selection with multiple criteria
    - Local random state management
    - Robust handling of missing features
    - Better error handling and validation
    """

    def __init__(
        self,
        config: Any,
        n_components: int = 50,
        gamma: float | str = "auto",
        kernel: str = "rbf",
        max_features: int = 50,
        feature_selection_method: str = "variance",
        handle_missing_features: str = "error",
    ):
        super().__init__(config)
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = kernel
        self.max_features = max_features
        self.feature_selection_method = feature_selection_method
        self.handle_missing_features = handle_missing_features

        # Validation
        if kernel not in ["rbf", "laplacian"]:
            raise ValueError(f"Unsupported kernel: {kernel}. Use 'rbf' or 'laplacian'.")
        if feature_selection_method not in ["variance", "f_score", "mutual_info"]:
            raise ValueError(
                f"Unsupported feature selection: {feature_selection_method}."
            )
        if handle_missing_features not in ["error", "ignore", "impute"]:
            raise ValueError(
                f"Unsupported missing handling: {handle_missing_features}."
            )

        self.gamma_ = None
        self.random_weights_ = None
        self.random_offset_ = None
        self.scaler_ = None
        self.selected_features_ = []
        self.feature_medians_ = pd.Series(dtype="float64")
        self._random_state = None
        self.ami_scorer = AdaptiveMI(
            subsample=min(getattr(config, "max_rows_score", 2000), 2000),
            spearman_gate=getattr(config, "mi_spearman_gate", 0.05),
            min_overlap=getattr(config, "mi_min_overlap", 50),
            ks=(3, 5, 10),
            n_bins=getattr(config, "mi_bins", 16),
            random_state=getattr(config, "random_state", 42),
        )

    def _get_random_state(self) -> np.random.RandomState:
        if self._random_state is None:
            seed = getattr(self.config, "random_state", None)
            self._random_state = np.random.RandomState(seed)
        return self._random_state

    def _estimate_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, str) and self.gamma == "auto":
            n_samples = min(1000, X.shape[0])
            if X.shape[0] > n_samples:
                rng = self._get_random_state()
                indices = rng.choice(X.shape[0], n_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            try:
                pairwise_dists = pdist(X_sample, metric="euclidean")
                if len(pairwise_dists) == 0:
                    return 1.0
                median_dist = np.median(pairwise_dists)
                if median_dist == 0:
                    return 1.0
                return 1.0 / (2 * median_dist**2)
            except Exception:
                return 1.0 / X.shape[1]
        return float(self.gamma)

    def _select_features_variance(self, X: pd.DataFrame) -> list[str]:
        numerical_cols = self.get_numerical_cols(X)
        if len(numerical_cols) <= self.max_features:
            return numerical_cols
        variances = X[numerical_cols].var()
        return variances.nlargest(self.max_features).index.tolist()

    def _select_features_target_aware(
        self, X: pd.DataFrame, y: pd.Series, method: str
    ) -> list[str]:
        numerical_cols = self.get_numerical_cols(X)
        if len(numerical_cols) <= self.max_features:
            return numerical_cols

        X_clean = X[numerical_cols].fillna(X[numerical_cols].median())
        if pd.api.types.is_numeric_dtype(y):
            y_clean = pd.to_numeric(y, errors="coerce")
            if y_clean.isna().any():
                fill_value = y_clean.median()
                y_clean = y_clean.fillna(fill_value)
        else:
            y_str = y.astype("string").fillna("MISSING")
            encoder = LabelEncoder()
            y_clean = pd.Series(
                encoder.fit_transform(y_str),
                index=y.index,
                dtype="int32",
            )

        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]

        if len(X_clean) == 0:
            return numerical_cols[: self.max_features]

        try:
            if method == "f_score":
                from sklearn.feature_selection import SelectKBest, f_regression

                selector = SelectKBest(score_func=f_regression, k=self.max_features)
                selector.fit(X_clean, y_clean)
                selected_mask = selector.get_support()
                return [
                    col
                    for col, selected in zip(numerical_cols, selected_mask)
                    if selected
                ]
            scores = self.ami_scorer.score_pairwise(X_clean.values, y_clean.values)
            score_series = (
                pd.Series(scores, index=numerical_cols)
                .fillna(0.0)
                .clip(lower=0.0)
                .sort_values(ascending=False)
            )
            return score_series.head(self.max_features).index.tolist()
        except Exception as e:
            warnings.warn(
                f"Target-aware feature selection failed: {e}. Falling back to variance."
            )
            return self._select_features_variance(X)

    def _select_features(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> list[str]:
        if self.feature_selection_method == "variance" or y is None:
            return self._select_features_variance(X)
        return self._select_features_target_aware(X, y, self.feature_selection_method)

    def _generate_random_weights(
        self, n_features: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = self._get_random_state()
        if self.kernel == "rbf":
            weights = rng.normal(
                0, np.sqrt(2 * self.gamma_), (n_features, self.n_components)
            )
        elif self.kernel == "laplacian":
            weights = rng.laplace(
                0, 1 / np.sqrt(self.gamma_), (n_features, self.n_components)
            )
        else:
            weights = rng.normal(
                0, np.sqrt(2 * self.gamma_), (n_features, self.n_components)
            )
        offset = rng.uniform(0, 2 * np.pi, self.n_components)
        return weights, offset

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "RandomFourierFeaturesTransformer":
        if not getattr(self.config, "create_rff", False):
            self.selected_features_ = []
            self.is_fitted = True
            return self

        self.selected_features_ = self._select_features(X, y)

        if len(self.selected_features_) < 2:
            warnings.warn(
                "Less than 2 numerical features available. RFF will be minimal."
            )
            self.is_fitted = True
            return self

        X_selected = X[self.selected_features_].copy()
        self.feature_medians_ = X_selected.median()
        X_selected = X_selected.fillna(self.feature_medians_)

        self.scaler_ = StandardScaler()
        X_normalized = self.scaler_.fit_transform(X_selected)

        self.gamma_ = self._estimate_gamma(X_normalized)
        self.random_weights_, self.random_offset_ = self._generate_random_weights(
            len(self.selected_features_)
        )

        self.is_fitted = True
        return self

    @require_fitted
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if not getattr(self.config, "create_rff", False) or not self.selected_features_:
            return self._empty_df(X.index)

        available_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = [f for f in self.selected_features_ if f not in X.columns]

        if missing_features:
            if self.handle_missing_features == "error":
                raise ValueError(f"Missing features in transform: {missing_features}")
            elif self.handle_missing_features == "ignore":
                warnings.warn(
                    f"Missing features will be imputed with median: {missing_features}"
                )

        X_selected = pd.DataFrame(index=X.index)
        for feature in self.selected_features_:
            if feature in X.columns:
                X_selected[feature] = X[feature]
            else:
                median_val = self.feature_medians_.get(feature, 0.0)
                X_selected[feature] = median_val

        for feature in self.selected_features_:
            if feature in self.feature_medians_.index:
                X_selected[feature] = X_selected[feature].fillna(
                    self.feature_medians_[feature]
                )

        try:
            X_normalized = self.scaler_.transform(X_selected[self.selected_features_])
        except Exception as e:
            raise ValueError(f"Error during scaling: {e}")

        projection = np.dot(X_normalized, self.random_weights_) + self.random_offset_
        features = {}
        scaling_factor = np.sqrt(1.0 / self.n_components)

        if self.kernel in ["rbf", "laplacian"]:
            cos_features = np.cos(projection) * scaling_factor
            sin_features = np.sin(projection) * scaling_factor
            for i in range(self.n_components):
                features[f"rff_cos_{i}"] = cos_features[:, i]
                features[f"rff_sin_{i}"] = sin_features[:, i]
        else:
            for i in range(self.n_components):
                features[f"rff_{i}"] = projection[:, i]

        return pd.DataFrame(features, index=X.index)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance based on random weight magnitudes."""
        if not self.is_fitted or self.random_weights_ is None:
            return pd.Series()
        importance = np.mean(np.abs(self.random_weights_), axis=1)
        return pd.Series(importance, index=self.selected_features_)

    def get_feature_names_out(self) -> list[str]:
        """Get output feature names."""
        if not self.is_fitted:
            return []
        feature_names = []
        if self.kernel in ["rbf", "laplacian"]:
            for i in range(self.n_components):
                feature_names.extend([f"rff_cos_{i}", f"rff_sin_{i}"])
        else:
            for i in range(self.n_components):
                feature_names.append(f"rff_{i}")
        return feature_names
