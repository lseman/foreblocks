import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import fft
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from foretools.aux.adaptive_mi import AdaptiveMI

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
        gamma: float | str = "auto",
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
        self.ami_scorer = AdaptiveMI(
            subsample=min(getattr(config, "max_rows_score", 2000), 2000),
            spearman_gate=getattr(config, "mi_spearman_gate", 0.05),
            min_overlap=getattr(config, "mi_min_overlap", 50),
            ks=(3, 5, 10),
            n_bins=getattr(config, "mi_bins", 16),
            random_state=getattr(config, "random_state", 42),
        )

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

    def _select_features_variance(self, X: pd.DataFrame) -> list[str]:
        """Select features with highest variance."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) <= self.max_features:
            return numerical_cols

        # Select features with highest variance
        variances = X[numerical_cols].var()
        return variances.nlargest(self.max_features).index.tolist()

    def _select_features_target_aware(
        self, X: pd.DataFrame, y: pd.Series, method: str
    ) -> list[str]:
        """Select features based on relationship with target."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) <= self.max_features:
            return numerical_cols

        # Prepare data
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

        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]

        if len(X_clean) == 0:
            return numerical_cols[: self.max_features]

        try:
            if method == "f_score":
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
                f"Target-aware feature selection failed: {e}. Falling back to variance-based selection."
            )
            return self._select_features_variance(X)

    def _select_features(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> list[str]:
        """Select features using specified method."""
        if self.feature_selection_method == "variance" or y is None:
            return self._select_features_variance(X)
        else:
            return self._select_features_target_aware(
                X, y, self.feature_selection_method
            )

    def _generate_random_weights(
        self, n_features: int
    ) -> tuple[np.ndarray, np.ndarray]:
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
        self, X: pd.DataFrame, y: pd.Series | None = None
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


class FourierTransformer(BaseFeatureTransformer):
    """Creates Fourier features for periodic patterns."""

    _TEMPORAL_TOKENS = (
        "time",
        "date",
        "day",
        "week",
        "month",
        "quarter",
        "year",
        "hour",
        "minute",
        "second",
        "season",
        "elapsed",
    )

    @classmethod
    def _looks_generated(cls, col: str) -> bool:
        generated_prefixes = (
            "row_",
            "rff_",
            "kmeans_",
            "gmm_",
            "umap_",
            "hdbscan_",
        )
        generated_suffixes = ("_bin", "_te")
        return (
            "__" in col
            or col.startswith(generated_prefixes)
            or col.endswith(generated_suffixes)
        )

    @classmethod
    def _temporal_priority(cls, col: str) -> int:
        lowered = col.lower()
        return int(any(token in lowered for token in cls._TEMPORAL_TOKENS))

    def _select_source_columns(self, X: pd.DataFrame) -> list[str]:
        cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            return []

        if getattr(self.config, "fourier_exclude_generated_sources", True):
            cols = [c for c in cols if not self._looks_generated(c)]
            if not cols:
                return []

        max_cols = int(getattr(self.config, "fourier_max_source_features", 12))
        ranking = []
        for col in cols:
            vals = pd.to_numeric(X[col], errors="coerce")
            ranking.append(
                (
                    self._temporal_priority(col),
                    float(vals.var(skipna=True)),
                    col,
                )
            )
        ranking.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [col for _, _, col in ranking[: max(1, min(max_cols, len(ranking)))]]

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "FourierTransformer":
        self.numerical_cols_ = self._select_source_columns(X)
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
                        "frequencies": valid_frequencies[
                            : max(1, int(getattr(self.config, "n_fourier_terms", 3)))
                        ],
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

    def __init__(self, config, strategies: tuple[str, ...] | None = None):
        super().__init__(config)
        config_strategies = getattr(config, "clustering_strategies", None)
        if strategies is not None:
            self.strategies = tuple(strategies)
        elif config_strategies:
            self.strategies = tuple(config_strategies)
        else:
            # Keep the default robust and dependency-light.
            self.strategies = ("kmeans", "gmm")
        self.scaler_ = None
        self.models_ = {}
        self.cluster_features_ = []
        self.fill_values_ = pd.Series(dtype="float64")

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "ClusteringTransformer":
        if not self.config.create_clustering:
            self.is_fitted = True
            return self

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            self.is_fitted = True
            return self

        Xnum = X[num_cols].copy()
        self.fill_values_ = Xnum.median(numeric_only=True)
        Xnum = Xnum.fillna(self.fill_values_)
        self.cluster_features_ = (
            Xnum.var()
            .nlargest(min(getattr(self.config, "clustering_max_features", 20), len(num_cols)))
            .index.tolist()
        )
        if len(self.cluster_features_) < 2:
            self.is_fitted = True
            return self

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(Xnum[self.cluster_features_])

        n_clusters = max(2, min(self.config.n_clusters, len(X_scaled) // 10))
        n_clusters = min(n_clusters, len(X_scaled))
        if n_clusters < 2:
            self.is_fitted = True
            return self

        # --- Fit models ---
        if "kmeans" in self.strategies:
            try:
                km = KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=self.config.random_state,
                )
                km.fit(X_scaled)
                self.models_["kmeans"] = km
            except Exception as e:
                warnings.warn(f"KMeans clustering failed: {e}")

        if "gmm" in self.strategies:
            try:
                gmm = GaussianMixture(
                    n_components=n_clusters, random_state=self.config.random_state
                )
                gmm.fit(X_scaled)
                self.models_["gmm"] = gmm
            except Exception as e:
                warnings.warn(f"GMM clustering failed: {e}")

        if "hdbscan" in self.strategies:
            try:
                import hdbscan

                hdb = hdbscan.HDBSCAN(min_cluster_size=15)
                hdb.fit(X_scaled)
                self.models_["hdbscan"] = hdb
            except Exception as e:
                warnings.warn(f"HDBSCAN clustering failed: {e}")

        if "umap" in self.strategies:
            try:
                import umap

                reducer = umap.UMAP(
                    n_components=3, random_state=self.config.random_state
                )
                reducer.fit(X_scaled)
                self.models_["umap"] = reducer
            except Exception as e:
                warnings.warn(f"UMAP embedding failed: {e}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models_ or not self.cluster_features_:
            return pd.DataFrame(index=X.index)

        Xnum = pd.DataFrame(index=X.index)
        for col in self.cluster_features_:
            if col in X.columns:
                Xnum[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                Xnum[col] = self.fill_values_.get(col, 0.0)
        Xnum = Xnum.fillna(self.fill_values_.reindex(self.cluster_features_).fillna(0.0))
        X_scaled = self.scaler_.transform(Xnum[self.cluster_features_])

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
                try:
                    import hdbscan

                    labels, strengths = hdbscan.approximate_predict(model, X_scaled)
                    feats["hdbscan_id"] = labels
                    feats["hdbscan_strength"] = strengths
                except Exception:
                    continue

            elif strat == "umap":
                embedding = model.transform(X_scaled)
                for i in range(embedding.shape[1]):
                    feats[f"umap_{i}"] = embedding[:, i]

        return pd.DataFrame(feats, index=X.index, dtype=np.float32)
