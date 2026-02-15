import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler

from .base import BaseFeatureTransformer


class RandomFourierFeaturesTransformer(BaseFeatureTransformer):
    """
    Random Fourier Features (RFF) transformer for approximating RBF kernels.

    Efficiently approximates kernel methods by mapping input features to a
    higher-dimensional space using random projections and trigonometric functions.

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
        gamma: Union[float, str] = "auto",
        kernel: str = "rbf",
        max_features: int = 50,
        feature_selection_method: str = "variance",
        handle_missing_features: str = "error",  # 'error', 'ignore', 'impute'
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
                f"Unsupported feature selection method: {feature_selection_method}"
            )

        if handle_missing_features not in ["error", "ignore", "impute"]:
            raise ValueError(
                f"Unsupported missing feature handling: {handle_missing_features}"
            )

        # Fitted attributes
        self.gamma_ = None
        self.random_weights_ = None
        self.random_offset_ = None
        self.scaler_ = None
        self.selected_features_ = None
        self.feature_medians_ = None
        self._random_state = None

    def _get_random_state(self) -> np.random.RandomState:
        """Get local random state to avoid polluting global numpy random state."""
        if self._random_state is None:
            seed = getattr(self.config, "random_state", None)
            self._random_state = np.random.RandomState(seed)
        return self._random_state

    def _estimate_gamma(self, X: np.ndarray) -> float:
        """
        Estimate gamma parameter using median pairwise distances.
        More robust than simple 1/n_features approach.
        """
        if isinstance(self.gamma, str) and self.gamma == "auto":
            # Sample subset for efficiency if data is large
            n_samples = min(1000, X.shape[0])
            if X.shape[0] > n_samples:
                rng = self._get_random_state()
                indices = rng.choice(X.shape[0], n_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            # Calculate pairwise distances and use median
            try:
                pairwise_dists = pdist(X_sample, metric="euclidean")
                if len(pairwise_dists) == 0:
                    return 1.0  # Fallback
                median_dist = np.median(pairwise_dists)
                if median_dist == 0:
                    return 1.0  # Fallback for identical points
                return 1.0 / (2 * median_dist**2)  # Standard RBF gamma
            except Exception:
                # Fallback to simple heuristic
                return 1.0 / X.shape[1]

        return float(self.gamma)

    def _select_features_variance(self, X: pd.DataFrame) -> List[str]:
        """Select features with highest variance."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) <= self.max_features:
            return numerical_cols

        # Select features with highest variance
        variances = X[numerical_cols].var()
        return variances.nlargest(self.max_features).index.tolist()

    def _select_features_target_aware(
        self, X: pd.DataFrame, y: pd.Series, method: str
    ) -> List[str]:
        """Select features based on relationship with target."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) <= self.max_features:
            return numerical_cols

        # Prepare data
        X_clean = X[numerical_cols].fillna(X[numerical_cols].median())
        y_clean = y.fillna(y.median()) if y.dtype in ["float64", "int64"] else y

        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]

        if len(X_clean) == 0:
            return numerical_cols[: self.max_features]

        try:
            if method == "f_score":
                selector = SelectKBest(score_func=f_regression, k=self.max_features)
            else:  # mutual_info
                selector = SelectKBest(
                    score_func=mutual_info_regression, k=self.max_features
                )

            selector.fit(X_clean, y_clean)
            selected_mask = selector.get_support()
            return [
                col for col, selected in zip(numerical_cols, selected_mask) if selected
            ]

        except Exception as e:
            warnings.warn(
                f"Target-aware feature selection failed: {e}. Falling back to variance-based selection."
            )
            return self._select_features_variance(X)

    def _select_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> List[str]:
        """Select features using specified method."""
        if self.feature_selection_method == "variance" or y is None:
            return self._select_features_variance(X)
        else:
            return self._select_features_target_aware(
                X, y, self.feature_selection_method
            )

    def _generate_random_weights(
        self, n_features: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random weights for Fourier features."""
        rng = self._get_random_state()

        if self.kernel == "rbf":
            # For RBF kernel: w ~ N(0, 2*gamma*I)
            weights = rng.normal(
                0, np.sqrt(2 * self.gamma_), (n_features, self.n_components)
            )
        elif self.kernel == "laplacian":
            # For Laplacian kernel: w ~ Laplace(0, sqrt(gamma))
            weights = rng.laplace(
                0, 1 / np.sqrt(self.gamma_), (n_features, self.n_components)
            )
        else:
            # Fallback to RBF
            weights = rng.normal(
                0, np.sqrt(2 * self.gamma_), (n_features, self.n_components)
            )

        # Random phase offset
        offset = rng.uniform(0, 2 * np.pi, self.n_components)
        return weights, offset

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "RandomFourierFeaturesTransformer":
        """Fit the Random Fourier Features transformer."""
        if not getattr(self.config, "create_rff", False):
            self.selected_features_ = []
            self.is_fitted = True
            return self

        # Select features
        self.selected_features_ = self._select_features(X, y)

        if len(self.selected_features_) < 2:
            warnings.warn(
                "Less than 2 numerical features available. RFF transformation will be minimal."
            )
            self.is_fitted = True
            return self

        # Prepare data
        X_selected = X[self.selected_features_].copy()

        # Store medians for consistent imputation
        self.feature_medians_ = X_selected.median()
        X_selected = X_selected.fillna(self.feature_medians_)

        # Normalize features
        self.scaler_ = StandardScaler()
        X_normalized = self.scaler_.fit_transform(X_selected)

        # Estimate gamma and generate weights
        self.gamma_ = self._estimate_gamma(X_normalized)
        self.random_weights_, self.random_offset_ = self._generate_random_weights(
            len(self.selected_features_)
        )

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using Random Fourier Features."""
        if not getattr(self.config, "create_rff", False) or not self.selected_features_:
            return pd.DataFrame(index=X.index)

        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform.")

        # Handle missing features
        available_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = [f for f in self.selected_features_ if f not in X.columns]

        if missing_features:
            if self.handle_missing_features == "error":
                raise ValueError(
                    f"Missing features in transform data: {missing_features}"
                )
            elif self.handle_missing_features == "ignore":
                warnings.warn(
                    f"Missing features will be imputed with median: {missing_features}"
                )

        # Prepare feature matrix
        X_selected = pd.DataFrame(index=X.index)

        # Add available features
        for feature in self.selected_features_:
            if feature in X.columns:
                X_selected[feature] = X[feature]
            else:
                # Impute missing features with stored median
                median_val = self.feature_medians_.get(feature, 0.0)
                X_selected[feature] = median_val

        # Fill NaN values with stored medians
        for feature in self.selected_features_:
            if feature in self.feature_medians_:
                X_selected[feature] = X_selected[feature].fillna(
                    self.feature_medians_[feature]
                )

        # Normalize and transform
        try:
            X_normalized = self.scaler_.transform(X_selected[self.selected_features_])
        except Exception as e:
            raise ValueError(f"Error during scaling: {e}")

        # Generate projections
        projection = np.dot(X_normalized, self.random_weights_) + self.random_offset_

        # Generate RFF features with correct scaling
        features = {}

        if self.kernel in ["rbf", "laplacian"]:
            # Correct scaling factor for Monte Carlo approximation
            scaling_factor = np.sqrt(1.0 / self.n_components)

            cos_features = np.cos(projection) * scaling_factor
            sin_features = np.sin(projection) * scaling_factor

            for i in range(self.n_components):
                features[f"rff_cos_{i}"] = cos_features[:, i]
                features[f"rff_sin_{i}"] = sin_features[:, i]
        else:
            # For other kernels, just use the projection
            for i in range(self.n_components):
                features[f"rff_{i}"] = projection[:, i]

        return pd.DataFrame(features, index=X.index)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance based on random weight magnitudes."""
        if not self.is_fitted or self.random_weights_ is None:
            return pd.Series()

        # Average absolute weight per input feature
        importance = np.mean(np.abs(self.random_weights_), axis=1)
        return pd.Series(importance, index=self.selected_features_)

    def get_feature_names_out(self) -> List[str]:
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


class FourierTransformer(BaseFeatureTransformer):
    """Creates Fourier features for periodic patterns."""

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FourierTransformer":
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.fourier_configs_ = {}

        if not self.config.create_fourier:
            self.is_fitted = True
            return self

        for col in self.numerical_cols_:
            if col not in X.columns:
                continue

            data = X[col].fillna(X[col].median())

            # Skip if insufficient variation
            if data.var() < 1e-6:
                continue

            try:
                # Normalize and compute FFT
                data_norm = (data - data.mean()) / (data.std() + 1e-8)
                fft_vals = fft.fft(data_norm.values)
                freqs = fft.fftfreq(len(data_norm))

                # Extract dominant frequencies
                magnitude = np.abs(fft_vals)
                top_freq_idx = np.argsort(magnitude)[
                    -(self.config.n_fourier_terms + 1) : -1
                ]

                valid_frequencies = [
                    freqs[idx] for idx in top_freq_idx if freqs[idx] != 0
                ]

                if valid_frequencies:
                    self.fourier_configs_[col] = {
                        "frequencies": valid_frequencies,
                        "mean": data.mean(),
                        "std": data.std(),
                    }
            except Exception as e:
                warnings.warn(f"Fourier analysis failed for {col}: {e}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.create_fourier or not self.fourier_configs_:
            return pd.DataFrame(index=X.index)

        features = {}
        for col, config in self.fourier_configs_.items():
            if col not in X.columns:
                continue

            data = X[col].fillna(config["mean"])

            for i, freq in enumerate(config["frequencies"]):
                features[f"{col}_fourier_cos_{i}"] = np.cos(
                    2 * np.pi * freq * np.arange(len(data))
                )
                features[f"{col}_fourier_sin_{i}"] = np.sin(
                    2 * np.pi * freq * np.arange(len(data))
                )

        return pd.DataFrame(features, index=X.index)


class ClusteringTransformer(BaseFeatureTransformer):
    """Modern clustering/embedding-based feature generator."""

    def __init__(self, config, strategies=("kmeans", "gmm", "hdbscan", "umap")):
        super().__init__(config)
        self.strategies = strategies
        self.scaler_ = None
        self.models_ = {}
        self.cluster_features_ = []

    def fit(self, X, y=None):
        if not self.config.create_clustering:
            self.is_fitted = True
            return self

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            self.is_fitted = True
            return self

        Xnum = X[num_cols].fillna(X[num_cols].median())
        self.cluster_features_ = (
            Xnum.var().nlargest(min(20, len(num_cols))).index.tolist()
        )

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(Xnum[self.cluster_features_])

        n_clusters = max(2, min(self.config.n_clusters, len(X_scaled) // 10))

        # --- Fit models ---
        if "kmeans" in self.strategies:
            km = KMeans(
                n_clusters=n_clusters, n_init=10, random_state=self.config.random_state
            )
            km.fit(X_scaled)
            self.models_["kmeans"] = km

        if "gmm" in self.strategies:
            gmm = GaussianMixture(
                n_components=n_clusters, random_state=self.config.random_state
            )
            gmm.fit(X_scaled)
            self.models_["gmm"] = gmm

        if "hdbscan" in self.strategies:
            hdb = hdbscan.HDBSCAN(min_cluster_size=15)
            hdb.fit(X_scaled)
            self.models_["hdbscan"] = hdb

        if "umap" in self.strategies:
            import umap

            reducer = umap.UMAP(n_components=3, random_state=self.config.random_state)
            reducer.fit(X_scaled)
            self.models_["umap"] = reducer

        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.models_:
            return pd.DataFrame(index=X.index)

        available = [c for c in self.cluster_features_ if c in X.columns]
        if not available:
            return pd.DataFrame(index=X.index)

        Xnum = X[available].fillna(X[available].median())
        X_scaled = self.scaler_.transform(Xnum)

        feats = {}

        for strat, model in self.models_.items():
            if strat == "kmeans":
                feats["kmeans_id"] = model.predict(X_scaled)
                dist = model.transform(X_scaled)
                for i in range(dist.shape[1]):
                    feats[f"kmeans_dist_{i}"] = dist[:, i]

            elif strat == "gmm":
                feats["gmm_id"] = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)
                for i in range(probs.shape[1]):
                    feats[f"gmm_prob_{i}"] = probs[:, i]

            elif strat == "hdbscan":
                feats["hdbscan_id"] = model.labels_
                feats["hdbscan_outlier"] = model.outlier_scores_

            elif strat == "umap":
                embedding = model.transform(X_scaled)
                for i in range(embedding.shape[1]):
                    feats[f"umap_{i}"] = embedding[:, i]

        return pd.DataFrame(feats, index=X.index, dtype=np.float32)
