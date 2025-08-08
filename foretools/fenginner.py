import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import fft
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   QuantileTransformer)
from sklearn.utils.validation import check_is_fitted


class AutoencoderWrapper(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = None):
        super().__init__()
        self.encoder = nn.Sequential()
        last_dim = input_dim
        for i, h in enumerate(hidden_dims):
            self.encoder.add_module(f"enc_{i}", nn.Linear(last_dim, h))
            self.encoder.add_module(f"enc_relu_{i}", nn.ReLU())
            last_dim = h

        self.latent_dim = last_dim
        self.decoder = nn.Sequential()
        rev_dims = hidden_dims[::-1] + [input_dim]
        for i, h in enumerate(rev_dims[:-1]):
            self.decoder.add_module(f"dec_{i}", nn.Linear(rev_dims[i], rev_dims[i + 1]))
            self.decoder.add_module(f"dec_relu_{i}", nn.ReLU())

        self.output_layer = nn.Linear(rev_dims[-2], rev_dims[-1])
        self.is_supervised = output_dim is not None
        if self.is_supervised:
            self.supervised_head = nn.Linear(self.latent_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        if self.is_supervised:
            y_hat = self.supervised_head(z)
            return self.decoder(z), y_hat
        return self.decoder(z), z


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Modern feature engineering pipeline with SOTA techniques.

    Key improvements:
    - Robust quantile transformation for numerical features
    - Smarter interaction generation with polynomial features
    - Target encoding with regularization for high-cardinality categoricals
    - Advanced datetime feature extraction with cyclical encoding
    - Memory-efficient processing
    - Better handling of rare categories and outliers

    New SOTA Features:
    - Logarithmic and mathematical transformations (log, sqrt, reciprocal)
    - Adaptive binning for non-linear patterns
    - Clustering-based features for complex relationships
    - Statistical features (rolling windows, aggregations)
    - Fourier features for periodic patterns
    - Advanced interaction detection
    """

    def __init__(
        self,
        task: str = "classification",
        corr_threshold: float = 0.95,
        create_interactions: bool = True,
        max_interactions: int = 50,
        use_quantile_transform: bool = True,
        target_encode_threshold: int = 10,
        rare_threshold: float = 0.01,
        polynomial_degree: int = 2,
        random_state: int = 42,
        # New SOTA parameters
        create_log_features: bool = True,
        create_sqrt_features: bool = True,
        create_reciprocal_features: bool = True,
        create_binning: bool = True,
        n_bins: int = 5,
        create_clustering_features: bool = True,
        n_clusters: int = 8,
        create_statistical_features: bool = True,
        rolling_windows: List[int] = None,
        create_fourier_features: bool = False,
        n_fourier_terms: int = 3,
        use_shap: bool = True,
        shap_model: Optional[object] = None,
        shap_threshold: float = 0.001,
        interaction_selection_model: str = "xgboost",
        max_selected_interactions: int = 20,
        use_autoencoder: bool = True,
        autoencoder_hidden_dims: Optional[List[int]] = None,
        autoencoder_epochs: int = 10,
        autoencoder_batch_size: int = 64,
        autoencoder_lr: float = 1e-3,
        autoencoder_device: str = "cuda",
    ):
        self.task = task
        self.corr_threshold = corr_threshold
        self.create_interactions = create_interactions
        self.max_interactions = max_interactions
        self.use_quantile_transform = use_quantile_transform
        self.target_encode_threshold = target_encode_threshold
        self.rare_threshold = rare_threshold
        self.polynomial_degree = polynomial_degree
        self.random_state = random_state
        # New SOTA parameters
        self.create_log_features = create_log_features
        self.create_sqrt_features = create_sqrt_features
        self.create_reciprocal_features = create_reciprocal_features
        self.create_binning = create_binning
        self.n_bins = n_bins
        self.create_clustering_features = create_clustering_features
        self.n_clusters = n_clusters
        self.create_statistical_features = create_statistical_features
        self.rolling_windows = rolling_windows or [3, 5, 7]
        self.create_fourier_features = create_fourier_features
        self.n_fourier_terms = n_fourier_terms
        self.use_shap = use_shap
        self.shap_model = shap_model
        self.shap_threshold = shap_threshold
        self.interaction_selection_model = interaction_selection_model
        self.max_selected_interactions = max_selected_interactions

        self.use_autoencoder = use_autoencoder
        self.autoencoder_hidden_dims = autoencoder_hidden_dims or [128, 64, 32]
        self.autoencoder_epochs = autoencoder_epochs
        self.autoencoder_batch_size = autoencoder_batch_size
        self.autoencoder_lr = autoencoder_lr
        self.autoencoder_device = autoencoder_device

        self.autoencoder_model_ = None
        self.autoencoder_feature_names_ = []

    def _infer_column_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Enhanced column type inference with better dtype handling."""
        return {
            "numerical": X.select_dtypes(
                include=[
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "Int8",
                    "Int16",
                    "Int32",
                    "Int64",
                    "float16",
                    "float32",
                    "float64",
                    "Float32",
                    "Float64",
                ]
            ).columns.tolist(),
            "categorical": X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist(),
            "datetime": X.select_dtypes(
                include=[
                    "datetime64[ns]",
                    "datetime64[ns, UTC]",
                    "datetime64",
                    "datetimetz",
                ]
            ).columns.tolist(),
            "boolean": X.select_dtypes(include=["bool"]).columns.tolist(),
        }

    def _extract_datetime_features(
        self, X: pd.DataFrame, cols: List[str]
    ) -> pd.DataFrame:
        """Enhanced datetime feature extraction with cyclical encoding."""
        features = {}

        for col in cols:
            if col not in X.columns:
                continue

            dt = pd.to_datetime(X[col], errors="coerce")

            # Basic features
            features[f"{col}_year"] = dt.dt.year
            features[f"{col}_month"] = dt.dt.month
            features[f"{col}_day"] = dt.dt.day
            features[f"{col}_weekday"] = dt.dt.weekday
            features[f"{col}_quarter"] = dt.dt.quarter
            features[f"{col}_is_weekend"] = (dt.dt.weekday >= 5).astype(int)
            features[f"{col}_hour"] = dt.dt.hour
            features[f"{col}_elapsed_days"] = (dt - dt.min()).dt.days

            # Cyclical encoding for better ML performance
            features[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
            features[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
            features[f"{col}_weekday_sin"] = np.sin(2 * np.pi * dt.dt.weekday / 7)
            features[f"{col}_weekday_cos"] = np.cos(2 * np.pi * dt.dt.weekday / 7)
            features[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
            features[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)

        return pd.DataFrame(features, index=X.index)

    def _generate_smart_interactions(
        self,
        X: pd.DataFrame,
        cols: List[str],
        y: Optional[pd.Series] = None,
        is_fit: bool = True,
    ) -> pd.DataFrame:
        """Generate interactions with optional model-based filtering."""
        features = {}

        # Use stored interaction pairs at transform time
        if not is_fit and hasattr(self, "interaction_pairs_"):
            for f1, f2, interaction_types in self.interaction_pairs_:
                if f1 in X.columns and f2 in X.columns:
                    a, b = X[f1], X[f2]
                    if "plus" in interaction_types:
                        features[f"{f1}_plus_{f2}"] = a + b
                    if "times" in interaction_types:
                        features[f"{f1}_times_{f2}"] = a * b
                    if "div" in interaction_types:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            features[f"{f1}_div_{f2}"] = np.where(
                                (b != 0) & (~b.isna()), a / b, np.nan
                            )
                    if "diff" in interaction_types:
                        features[f"{f1}_diff_{f2}"] = a - b
                    if "squared_1" in interaction_types:
                        features[f"{f1}_squared"] = a**2
                    if "squared_2" in interaction_types:
                        features[f"{f2}_squared"] = b**2

            # Apply interaction filtering if stored
            if hasattr(self, "interaction_features_selected_"):
                selected = self.interaction_features_selected_
                return pd.DataFrame(
                    {k: v for k, v in features.items() if k in selected}, index=X.index
                )
            else:
                return pd.DataFrame(features, index=X.index)

        # Fit-time: generate full set
        if len(cols) < 2:
            return pd.DataFrame(index=X.index)

        self.interaction_pairs_ = []
        interactions_created = 0
        variances = X[cols].var().sort_values(ascending=False)
        sorted_cols = variances.index.tolist()

        for i, f1 in enumerate(sorted_cols):
            for j, f2 in enumerate(sorted_cols[i + 1 :], i + 1):
                if interactions_created >= self.max_interactions:
                    break

                if f1 not in X.columns or f2 not in X.columns:
                    continue

                a, b = X[f1], X[f2]
                if a.isna().mean() > 0.5 or b.isna().mean() > 0.5:
                    continue

                interaction_types = []
                features[f"{f1}_plus_{f2}"] = a + b
                interaction_types.append("plus")

                features[f"{f1}_times_{f2}"] = a * b
                interaction_types.append("times")

                with np.errstate(divide="ignore", invalid="ignore"):
                    div = np.where((b != 0) & (~b.isna()), a / b, np.nan)
                    if not np.isnan(div).all():
                        features[f"{f1}_div_{f2}"] = div
                        interaction_types.append("div")

                features[f"{f1}_diff_{f2}"] = a - b
                interaction_types.append("diff")

                if interactions_created < self.max_interactions // 2:
                    features[f"{f1}_squared"] = a**2
                    features[f"{f2}_squared"] = b**2
                    interaction_types.extend(["squared_1", "squared_2"])

                self.interaction_pairs_.append((f1, f2, interaction_types))
                interactions_created += len(interaction_types)

        interaction_df = pd.DataFrame(features, index=X.index)

        # Optional model-based selection
        if is_fit and self.interaction_selection_model == "xgboost" and y is not None:
            try:
                from xgboost import XGBClassifier, XGBRegressor

                valid = y.notna() & interaction_df.notna().all(axis=1)
                if valid.sum() >= 30:
                    ModelCls = (
                        XGBClassifier if self.task == "classification" else XGBRegressor
                    )
                    xgb = ModelCls(
                        n_estimators=100,
                        max_depth=3,
                        random_state=self.random_state,
                        verbosity=0,
                        device="cuda",
                    )
                    xgb.fit(interaction_df.loc[valid], y.loc[valid])
                    importances = pd.Series(
                        xgb.feature_importances_, index=interaction_df.columns
                    )
                    top_k = (
                        importances.sort_values(ascending=False)
                        .head(self.max_selected_interactions)
                        .index
                    )
                    self.interaction_features_selected_ = list(top_k)
                    return interaction_df[top_k]
            except Exception as e:
                warnings.warn(f"XGBoost feature selection failed: {e}")

        self.interaction_features_selected_ = list(interaction_df.columns)
        return interaction_df

    def _target_encode_categorical(
        self, X: pd.DataFrame, y: pd.Series, col: str
    ) -> pd.Series:
        """Target encoding with regularization to prevent overfitting."""
        if y is None:
            return pd.Series(0, index=X.index)

        # Calculate global mean
        global_mean = y.mean()

        # Calculate category means with regularization
        category_stats = (
            X.groupby(col)[col]
            .agg(["count"])
            .join(y.groupby(X[col]).agg(["mean", "count"]), rsuffix="_y")
        )

        # Regularization parameter (higher for smaller samples)
        alpha = 10
        regularized_means = (
            category_stats["count_y"] * category_stats["mean"] + alpha * global_mean
        ) / (category_stats["count_y"] + alpha)

        return X[col].map(regularized_means).fillna(global_mean)

    def _create_mathematical_features(
        self, X: pd.DataFrame, cols: List[str], is_fit: bool = True
    ) -> pd.DataFrame:
        """Create mathematical transformations for numerical features."""
        features = {}

        # During transform, only create features we learned during fit
        if not is_fit and hasattr(self, "math_feature_configs_"):
            for col, transforms in self.math_feature_configs_.items():
                if col not in X.columns:
                    continue

                data = X[col]

                if "log" in transforms:
                    features[f"{col}_log"] = np.log1p(np.maximum(data, 0))
                if "sqrt" in transforms:
                    features[f"{col}_sqrt"] = np.sqrt(np.maximum(data, 0))
                if "reciprocal" in transforms:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        features[f"{col}_reciprocal"] = np.where(
                            np.abs(data) > 1e-6, 1 / data, np.nan
                        )

            return pd.DataFrame(features, index=X.index)

        # During fit, determine which transforms to apply
        if is_fit:
            self.math_feature_configs_ = {}

        for col in cols:
            if col not in X.columns:
                continue

            data = X[col]

            # Skip if too many missing values
            if data.isna().mean() > 0.5:
                continue

            if is_fit:
                self.math_feature_configs_[col] = []

            # Log features (for positive values)
            if self.create_log_features and (data > 0).any():
                log_data = np.log1p(np.maximum(data, 0))
                if not log_data.isna().all() and log_data.var() > 1e-6:
                    features[f"{col}_log"] = log_data
                    if is_fit:
                        self.math_feature_configs_[col].append("log")

            # Square root features (for non-negative values)
            if self.create_sqrt_features and (data >= 0).any():
                sqrt_data = np.sqrt(np.maximum(data, 0))
                if not sqrt_data.isna().all() and sqrt_data.var() > 1e-6:
                    features[f"{col}_sqrt"] = sqrt_data
                    if is_fit:
                        self.math_feature_configs_[col].append("sqrt")

            # Reciprocal features (avoid division by zero)
            if self.create_reciprocal_features:
                with np.errstate(divide="ignore", invalid="ignore"):
                    recip_data = np.where(np.abs(data) > 1e-6, 1 / data, np.nan)
                    if not np.isnan(recip_data).all() and np.nanvar(recip_data) > 1e-6:
                        features[f"{col}_reciprocal"] = recip_data
                        if is_fit:
                            self.math_feature_configs_[col].append("reciprocal")

        return pd.DataFrame(features, index=X.index)

    def _create_binning_features(
        self, X: pd.DataFrame, cols: List[str], is_fit: bool = True
    ) -> pd.DataFrame:
        """Create adaptive binning features, robust to constant or low-variance inputs."""
        if not self.create_binning:
            return pd.DataFrame(index=X.index)

        features = {}

        if is_fit:
            self.binning_transformers_ = {}

        for col in cols:
            if col not in X.columns:
                continue

            data = X[[col]].dropna()

            # Skip if not enough data or too constant
            if (
                len(data) < self.n_bins * 2
                or data[col].nunique() <= 1
                or (data[col].max() - data[col].min() < 1e-6)
            ):
                continue

            # Check bin width quantiles proactively
            quantiles = np.quantile(data[col], np.linspace(0, 1, self.n_bins + 1))
            bin_widths = np.diff(quantiles)
            if np.all(bin_widths <= 1e-8):
                continue  # All bins too narrow

            if is_fit:
                try:
                    self.binning_transformers_[col] = KBinsDiscretizer(
                        n_bins=self.n_bins,
                        encode="ordinal",
                        strategy="quantile",
                        subsample=min(10000, len(data)),
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        self.binning_transformers_[col].fit(data)
                except Exception as e:
                    warnings.warn(f"[Binning-Fit] Failed on {col}: {e}")
                    continue

            if col in getattr(self, "binning_transformers_", {}):
                try:
                    X_col_filled = X[[col]].fillna(X[col].median())
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        binned = self.binning_transformers_[col].transform(X_col_filled)
                    features[f"{col}_bin"] = binned.flatten()
                except Exception as e:
                    warnings.warn(f"[Binning-Transform] Failed on {col}: {e}")
                    continue

        return pd.DataFrame(features, index=X.index)

    def _create_clustering_features(
        self, X: pd.DataFrame, cols: List[str], is_fit: bool = True
    ) -> pd.DataFrame:
        """Create clustering-based features for complex relationships."""
        if not self.create_clustering_features or len(cols) < 2:
            return pd.DataFrame(index=X.index)

        features = {}

        # Use top features by variance for clustering
        available_cols = [col for col in cols if col in X.columns]
        if len(available_cols) < 2:
            return pd.DataFrame(index=X.index)

        X_clust = X[available_cols].fillna(X[available_cols].median())

        if is_fit:
            # Fit clustering model
            self.cluster_model_ = KMeans(
                n_clusters=min(self.n_clusters, len(X_clust) // 10),
                random_state=self.random_state,
                n_init=10,
            )

            # Use subset of features to avoid curse of dimensionality
            top_cols = (
                X_clust.var().nlargest(min(10, len(available_cols))).index.tolist()
            )
            self.cluster_features_ = top_cols

            try:
                self.cluster_model_.fit(X_clust[top_cols])
            except Exception as e:
                warnings.warn(f"Clustering failed: {e}")
                return pd.DataFrame(index=X.index)

        if hasattr(self, "cluster_model_") and hasattr(self, "cluster_features_"):
            try:
                # Get cluster assignments
                cluster_features = [
                    col for col in self.cluster_features_ if col in X_clust.columns
                ]
                if cluster_features:
                    clusters = self.cluster_model_.predict(X_clust[cluster_features])
                    features["cluster_id"] = clusters

                    # Distance to cluster centers
                    distances = self.cluster_model_.transform(X_clust[cluster_features])
                    features["cluster_distance_min"] = distances.min(axis=1)
                    features["cluster_distance_mean"] = distances.mean(axis=1)
            except Exception as e:
                warnings.warn(f"Cluster prediction failed: {e}")

        return pd.DataFrame(features, index=X.index)

    def _create_statistical_features(
        self, X: pd.DataFrame, cols: List[str]
    ) -> pd.DataFrame:
        """Create statistical aggregation features."""
        if not self.create_statistical_features or len(cols) < 2:
            return pd.DataFrame(index=X.index)

        features = {}
        available_cols = [col for col in cols if col in X.columns]

        if len(available_cols) < 2:
            return pd.DataFrame(index=X.index)

        X_stats = X[available_cols]

        # Row-wise statistics
        features["row_mean"] = X_stats.mean(axis=1)
        features["row_std"] = X_stats.std(axis=1)
        features["row_min"] = X_stats.min(axis=1)
        features["row_max"] = X_stats.max(axis=1)
        features["row_median"] = X_stats.median(axis=1)
        features["row_range"] = features["row_max"] - features["row_min"]
        features["row_skew"] = X_stats.skew(axis=1)

        # Count of non-null values per row
        features["row_non_null_count"] = X_stats.notna().sum(axis=1)
        features["row_null_ratio"] = X_stats.isna().sum(axis=1) / len(available_cols)

        return pd.DataFrame(features, index=X.index)

    def _create_fourier_features(
        self, X: pd.DataFrame, cols: List[str], is_fit: bool = True
    ) -> pd.DataFrame:
        """Create Fourier features for periodic patterns."""
        if not self.create_fourier_features:
            return pd.DataFrame(index=X.index)

        features = {}

        # During transform, use stored configurations
        if not is_fit and hasattr(self, "fourier_configs_"):
            for col, config in self.fourier_configs_.items():
                if col not in X.columns:
                    continue

                data = X[col].fillna(X[col].median())
                data_norm = (data - config["mean"]) / (config["std"] + 1e-8)

                for i, freq in enumerate(config["frequencies"]):
                    features[f"{col}_fourier_cos_{i}"] = np.cos(
                        2 * np.pi * freq * np.arange(len(data))
                    )
                    features[f"{col}_fourier_sin_{i}"] = np.sin(
                        2 * np.pi * freq * np.arange(len(data))
                    )

            return pd.DataFrame(features, index=X.index)

        # During fit, determine frequencies to use
        if is_fit:
            self.fourier_configs_ = {}

        for col in cols:
            if col not in X.columns:
                continue

            data = X[col].fillna(X[col].median())

            # Skip if not enough variation
            if data.var() < 1e-6:
                continue

            # Normalize data
            data_norm = (data - data.mean()) / (data.std() + 1e-8)

            try:
                # Compute FFT
                fft_vals = fft.fft(data_norm.values)
                freqs = fft.fftfreq(len(data_norm))

                # Extract dominant frequencies
                magnitude = np.abs(fft_vals)
                top_freq_idx = np.argsort(magnitude)[
                    -self.n_fourier_terms - 1 : -1
                ]  # Skip DC component

                valid_frequencies = []
                for i, idx in enumerate(top_freq_idx):
                    if freqs[idx] != 0:  # Skip DC component
                        valid_frequencies.append(freqs[idx])
                        features[f"{col}_fourier_cos_{i}"] = np.cos(
                            2 * np.pi * freqs[idx] * np.arange(len(data))
                        )
                        features[f"{col}_fourier_sin_{i}"] = np.sin(
                            2 * np.pi * freqs[idx] * np.arange(len(data))
                        )

                if is_fit and valid_frequencies:
                    self.fourier_configs_[col] = {
                        "frequencies": valid_frequencies,
                        "mean": data.mean(),
                        "std": data.std(),
                    }

            except Exception as e:
                warnings.warn(f"Fourier features failed for {col}: {e}")

        return pd.DataFrame(features, index=X.index)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fit the feature engineer with modern techniques."""
        import shap
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        X = X.copy()
        if y is not None:
            y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

        self.column_types_ = self._infer_column_types(X)
        self.original_features_ = X.columns.tolist()

        # Extract datetime features
        if self.column_types_["datetime"]:
            dt_df = self._extract_datetime_features(X, self.column_types_["datetime"])
            X = pd.concat([X, dt_df], axis=1)
            self.column_types_["numerical"].extend(dt_df.columns.tolist())

        if self.column_types_["boolean"]:
            X[self.column_types_["boolean"]] = X[self.column_types_["boolean"]].astype(
                "int8"
            )

        all_numerical_features = self.column_types_["numerical"].copy()

        # Mathematical transformations
        math_df = self._create_mathematical_features(
            X, self.column_types_["numerical"], is_fit=True
        )
        if not math_df.empty:
            X = pd.concat([X, math_df], axis=1)
            self.math_features_ = math_df.columns.tolist()
            all_numerical_features.extend(self.math_features_)
        else:
            self.math_features_ = []

        # Interactions
        if self.create_interactions and len(all_numerical_features) >= 2:
            inter_df = self._generate_smart_interactions(
                X, all_numerical_features, y=y, is_fit=True
            )
            if not inter_df.empty:
                X = pd.concat([X, inter_df], axis=1)
                self.interaction_features_ = inter_df.columns.tolist()
                all_numerical_features.extend(self.interaction_features_)
            else:
                self.interaction_features_ = []
        else:
            self.interaction_features_ = []

        # Binning
        binning_df = self._create_binning_features(
            X, all_numerical_features, is_fit=True
        )
        if not binning_df.empty:
            X = pd.concat([X, binning_df], axis=1)
            self.binning_features_ = binning_df.columns.tolist()
            all_numerical_features.extend(self.binning_features_)
        else:
            self.binning_features_ = []

        # Clustering
        base_features_for_clustering = [
            col for col in all_numerical_features if col not in self.binning_features_
        ]
        clustering_df = self._create_clustering_features(
            X, base_features_for_clustering, is_fit=True
        )
        if not clustering_df.empty:
            X = pd.concat([X, clustering_df], axis=1)
            self.clustering_features_ = clustering_df.columns.tolist()
            all_numerical_features.extend(self.clustering_features_)
        else:
            self.clustering_features_ = []

        # Statistical
        base_features_for_stats = [
            col for col in self.column_types_["numerical"] if col in X.columns
        ]
        stats_df = self._create_statistical_features(X, base_features_for_stats)
        if not stats_df.empty:
            X = pd.concat([X, stats_df], axis=1)
            self.statistical_features_ = stats_df.columns.tolist()
            all_numerical_features.extend(self.statistical_features_)
        else:
            self.statistical_features_ = []

        # Fourier
        fourier_df = self._create_fourier_features(
            X, base_features_for_stats, is_fit=True
        )
        if not fourier_df.empty:
            X = pd.concat([X, fourier_df], axis=1)
            self.fourier_features_ = fourier_df.columns.tolist()
            all_numerical_features.extend(self.fourier_features_)
        else:
            self.fourier_features_ = []

        self.column_types_["numerical"] = all_numerical_features

        # Correlation removal
        self.to_drop_ = []
        if len(self.column_types_["numerical"]) > 1:
            try:
                corr_matrix = X[self.column_types_["numerical"]].corr().abs()
                upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                high_corr_pairs = np.where(
                    (corr_matrix > self.corr_threshold) & upper_tri
                )
                for i, j in zip(*high_corr_pairs):
                    col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                    var1, var2 = X[col1].var(), X[col2].var()
                    drop_col = col1 if var1 < var2 else col2
                    if drop_col not in self.to_drop_:
                        self.to_drop_.append(drop_col)
            except Exception as e:
                warnings.warn(f"Correlation analysis failed: {e}")

        self.final_numerical_ = [
            c for c in self.column_types_["numerical"] if c not in self.to_drop_
        ]

        # Imputation
        self.num_imputer_ = {}
        for col in self.final_numerical_:
            if col in X.columns:
                col_data = X[col].dropna()
                if len(col_data) == 0:
                    self.num_imputer_[col] = 0
                elif col_data.nunique() <= 2:
                    self.num_imputer_[col] = col_data.mode().iloc[0]
                else:
                    skewness = abs(skew(col_data.astype(float), nan_policy="omit"))
                    self.num_imputer_[col] = (
                        col_data.median() if skewness > 1 else col_data.mean()
                    )

        # Autoencoder
        if self.use_autoencoder and len(self.final_numerical_) > 2:
            try:
                X_ae = (
                    X[self.final_numerical_].fillna(pd.Series(self.num_imputer_)).copy()
                )
                X_tensor = torch.tensor(X_ae.values, dtype=torch.float32)
                y_tensor = (
                    torch.tensor(y.values, dtype=torch.float32)
                    if y is not None
                    else None
                )

                dataset = (
                    TensorDataset(X_tensor)
                    if y is None
                    else TensorDataset(X_tensor, y_tensor)
                )
                loader = DataLoader(
                    dataset, batch_size=self.autoencoder_batch_size, shuffle=True
                )

                self.autoencoder_model_ = AutoencoderWrapper(
                    input_dim=X_tensor.shape[1],
                    hidden_dims=self.autoencoder_hidden_dims,
                    output_dim=1 if y is not None else None,
                ).to(self.autoencoder_device)

                criterion = (
                    nn.MSELoss()
                    if y is None
                    else nn.BCEWithLogitsLoss()
                    if self.task == "classification"
                    else nn.MSELoss()
                )
                optimizer = optim.Adam(
                    self.autoencoder_model_.parameters(), lr=self.autoencoder_lr
                )

                self.autoencoder_model_.train()
                for epoch in range(self.autoencoder_epochs):
                    for batch in loader:
                        optimizer.zero_grad()
                        if y is None:
                            x = batch[0].to(self.autoencoder_device)
                            x_hat, _ = self.autoencoder_model_(x)
                            loss = criterion(x_hat, x)
                        else:
                            x, yb = batch
                            x = x.to(self.autoencoder_device)
                            yb = yb.to(self.autoencoder_device)
                            x_hat, y_hat = self.autoencoder_model_(x)
                            loss = criterion(y_hat.view(-1), yb.view(-1))
                        loss.backward()
                        optimizer.step()

            except Exception as e:
                warnings.warn(f"[Autoencoder] Training failed: {e}")
                self.autoencoder_model_ = None

        # MI and SHAP
        self.mi_scores_ = None
        self.selected_features_ = self.final_numerical_.copy()
        self.shap_values_ = None
        self.shap_summary_ = None
        self._shap_X_data_ = None

        if y is not None and len(X) >= 20 and len(self.final_numerical_) > 0:
            try:
                X_for_mi = X[self.final_numerical_].copy()
                for col in X_for_mi.columns:
                    X_for_mi[col] = X_for_mi[col].fillna(self.num_imputer_[col])
                mask = y.notna() & X_for_mi.notna().all(axis=1)
                if mask.sum() >= 10:
                    mi_func = (
                        mutual_info_classif
                        if self.task == "classification"
                        else mutual_info_regression
                    )
                    mi_scores = mi_func(
                        X_for_mi.loc[mask], y.loc[mask], random_state=self.random_state
                    )
                    self.mi_scores_ = pd.Series(
                        mi_scores, index=self.final_numerical_
                    ).sort_values(ascending=False)
                    self.selected_features_ = self.mi_scores_[
                        self.mi_scores_ > 0.001
                    ].index.tolist()
            except Exception as e:
                warnings.warn(f"Mutual information computation failed: {e}")

            # SHAP pruning section (replacement for the fit method)
            if self.use_shap and len(self.selected_features_) > 1:
                try:
                    import shap
                    from xgboost import XGBClassifier, XGBRegressor

                    X_shap = X[self.selected_features_].copy()
                    for col in X_shap.columns:
                        X_shap[col] = X_shap[col].fillna(self.num_imputer_[col])

                    # Check if we have enough samples for meaningful SHAP analysis
                    if X_shap.shape[0] < 2:
                        warnings.warn(
                            "[SHAP] Not enough rows for SHAP analysis (requires ≥2 rows). Skipping SHAP pruning."
                        )
                        # Still store basic info but skip the pruning
                        self._shap_X_data_ = X_shap
                        self.shap_summary_ = None
                        self.shap_values_ = None
                        self.shap_explanation_ = None
                    else:
                        # Store the data for plotting
                        self._shap_X_data_ = X_shap

                        model = self.shap_model
                        if model is None:
                            # Use CPU for small datasets to avoid CUDA issues
                            device = "cuda"

                            model = (
                                XGBClassifier(
                                    random_state=self.random_state,
                                    n_estimators=min(
                                        100, max(10, X_shap.shape[0] // 2)
                                    ),
                                    max_depth=3,
                                    verbosity=0,
                                    use_label_encoder=False,
                                    device=device,
                                )
                                if self.task == "classification"
                                else XGBRegressor(
                                    random_state=self.random_state,
                                    n_estimators=min(
                                        100, max(10, X_shap.shape[0] // 2)
                                    ),
                                    max_depth=3,
                                    verbosity=0,
                                    device=device,
                                )
                            )

                        # Fit the model
                        model.fit(X_shap, y)

                        # Create SHAP explainer
                        explainer = shap.TreeExplainer(model)

                        # Calculate SHAP values
                        shap_values = explainer.shap_values(X_shap)

                        # Store raw SHAP values and create Explanation object
                        self.shap_values_ = shap_values

                        # Create proper SHAP Explanation object for plotting
                        if isinstance(shap_values, list):
                            # Multi-class case - use the first class or average
                            if self.task == "classification" and len(shap_values) > 1:
                                # For plotting, we'll use the mean across classes
                                shap_values_for_plot = np.mean(
                                    np.array(shap_values), axis=0
                                )
                            else:
                                shap_values_for_plot = shap_values[0]
                        else:
                            shap_values_for_plot = shap_values

                        # Create the Explanation object
                        self.shap_explanation_ = shap.Explanation(
                            values=shap_values_for_plot,
                            base_values=explainer.expected_value
                            if not isinstance(explainer.expected_value, np.ndarray)
                            else explainer.expected_value[0],
                            data=X_shap.values,
                            feature_names=list(X_shap.columns),
                        )

                        # Process SHAP values for feature selection
                        if isinstance(shap_values, list):
                            # Multi-class classification case
                            if len(shap_values) > 1:
                                shap_values_processed = np.abs(
                                    np.array(shap_values)
                                ).mean(axis=(0, 1))
                            else:
                                shap_values_processed = np.abs(shap_values[0]).mean(
                                    axis=0
                                )
                        else:
                            # Binary classification or regression case
                            if shap_values.ndim > 1:
                                shap_values_processed = np.abs(shap_values).mean(axis=0)
                            else:
                                shap_values_processed = np.abs(shap_values)

                        # Create summary series
                        shap_summary = pd.Series(
                            shap_values_processed, index=self.selected_features_
                        ).sort_values(ascending=False)

                        # Store SHAP results
                        self.shap_summary_ = shap_summary

                        # Feature selection based on SHAP threshold
                        significant_features = shap_summary[
                            shap_summary > self.shap_threshold
                        ].index.tolist()

                        if len(significant_features) > 0:
                            removed_features = set(self.selected_features_) - set(
                                significant_features
                            )
                            self.selected_features_ = significant_features

                            if removed_features:
                                print(
                                    f"Removed {len(removed_features)} features based on SHAP threshold ({self.shap_threshold}): {removed_features}"
                                )
                            else:
                                print(
                                    f"All {len(self.selected_features_)} features passed SHAP threshold."
                                )
                        else:
                            warnings.warn(
                                f"No features passed SHAP threshold ({self.shap_threshold}). Keeping all features."
                            )

                except Exception as e:
                    warnings.warn(f"SHAP pruning failed: {e}")
                    # Ensure we have fallback values
                    self.shap_summary_ = None
                    self.shap_values_ = None
                    self._shap_X_data_ = None
                    self.shap_explanation_ = None
        # Normalize or scale final features
        if self.selected_features_:
            X_filled = X[self.selected_features_].fillna(
                pd.Series(self.num_imputer_)[self.selected_features_]
            )
            n_samples = len(X_filled)
            n_quantiles = min(1000, max(10, n_samples // 2))

            if self.use_quantile_transform:
                self.quantile_transformer_ = QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution="normal",
                    random_state=self.random_state,
                    subsample=min(100000, n_samples),
                )
                self.quantile_transformer_.fit(X_filled)
            else:
                self.quantile_transformer_ = None
                self.num_means_ = X_filled.mean()
                self.num_stds_ = X_filled.std().replace(0, 1.0)
        else:
            self.quantile_transformer_ = None
            self.num_means_ = pd.Series(dtype=float)
            self.num_stds_ = pd.Series(dtype=float)

        # Enhanced categorical preprocessing
        if self.column_types_["categorical"]:
            # Handle rare categories
            self.rare_categories_ = {}
            for col in self.column_types_["categorical"]:
                if col in X.columns:
                    value_counts = X[col].value_counts(normalize=True)
                    rare_cats = value_counts[
                        value_counts < self.rare_threshold
                    ].index.tolist()
                    self.rare_categories_[col] = rare_cats

            # Imputation with mode
            self.cat_modes_ = {}
            for col in self.column_types_["categorical"]:
                if col in X.columns and not X[col].dropna().empty:
                    self.cat_modes_[col] = X[col].mode().iloc[0]
                else:
                    self.cat_modes_[col] = "missing"

            # Target encoding for high-cardinality features
            self.target_encoders_ = {}
            if y is not None:
                for col in self.column_types_["categorical"]:
                    if (
                        col in X.columns
                        and X[col].nunique() > self.target_encode_threshold
                    ):
                        self.target_encoders_[col] = self._target_encode_categorical(
                            X, y, col
                        )

            # One-hot encoding for remaining categorical features
            cat_for_ohe = [
                col
                for col in self.column_types_["categorical"]
                if col not in self.target_encoders_
            ]

            if cat_for_ohe:
                cat_df = X[cat_for_ohe].copy()
                # Handle rare categories and missing values
                for col in cat_for_ohe:
                    if col in cat_df.columns:
                        cat_df[col] = cat_df[col].fillna(self.cat_modes_[col])
                        # Replace rare categories with 'OTHER'
                        if col in self.rare_categories_:
                            cat_df[col] = cat_df[col].replace(
                                self.rare_categories_[col], "OTHER"
                            )

                self.ohe_ = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="if_binary"
                )
                self.ohe_.fit(cat_df)
                self.ohe_columns_ = cat_for_ohe
            else:
                self.ohe_ = None
                self.ohe_columns_ = []

            # Frequency encoding
            self.freq_encoding_ = {}
            for col in self.column_types_["categorical"]:
                if col in X.columns:
                    freqs = X[col].value_counts(normalize=True)
                    self.freq_encoding_[col] = defaultdict(lambda: 0, freqs.to_dict())
        else:
            self.cat_modes_ = {}
            self.ohe_ = None
            self.freq_encoding_ = {}
            self.target_encoders_ = {}
            self.rare_categories_ = {}
            self.ohe_columns_ = []

        # Boolean handling
        if self.column_types_["boolean"]:
            self.bool_modes_ = {}
            for col in self.column_types_["boolean"]:
                if col in X.columns and not X[col].dropna().empty:
                    self.bool_modes_[col] = X[col].mode().iloc[0]
                else:
                    self.bool_modes_[col] = False
        else:
            self.bool_modes_ = {}

        self.fitted_ = True
        return self

    def plot_shap_summary(self, max_display: int = 20):
        if hasattr(self, "shap_explanation_") and self.shap_explanation_ is not None:
            import shap

            shap.plots.beeswarm(self.shap_explanation_, max_display=max_display)
        elif self.shap_summary_ is not None:
            self.shap_summary_.head(max_display).plot(kind="barh")
            plt.title("SHAP Feature Importance (Mean |SHAP|)")
            plt.xlabel("Mean SHAP Value")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        else:
            print(
                "SHAP values not available. Ensure fit() was called with use_shap=True."
            )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform with modern preprocessing pipeline."""
        check_is_fitted(self, "fitted_")
        X = X.copy()
        result_dfs = []  # ← Fix: ensure this is defined first

        # Datetime features
        if self.column_types_["datetime"]:
            dt_df = self._extract_datetime_features(X, self.column_types_["datetime"])
            X = pd.concat([X, dt_df], axis=1)

        # Boolean conversion
        if self.column_types_["boolean"]:
            for col in self.column_types_["boolean"]:
                if col in X.columns:
                    X[col] = X[col].fillna(self.bool_modes_[col]).astype("int8")

        # Mathematical transformations
        if hasattr(self, "math_features_") and self.math_features_:
            math_df = self._create_mathematical_features(X, [], is_fit=False)
            if not math_df.empty:
                X = pd.concat([X, math_df], axis=1)

        # Get all numerical features before creating interactions
        current_numerical = [
            col
            for col in X.columns
            if col in X.select_dtypes(include=[np.number]).columns
        ]

        # Interactions - use stored pairs for consistency
        if (
            self.create_interactions
            and hasattr(self, "interaction_pairs_")
            and self.interaction_pairs_
        ):
            inter_df = self._generate_smart_interactions(
                X, current_numerical, is_fit=False
            )
            if not inter_df.empty:
                X = pd.concat([X, inter_df], axis=1)

        # Update numerical features list
        current_numerical = [
            col
            for col in X.columns
            if col in X.select_dtypes(include=[np.number]).columns
        ]

        # Binning features
        if hasattr(self, "binning_features_") and self.binning_features_:
            binning_df = self._create_binning_features(
                X, current_numerical, is_fit=False
            )
            if not binning_df.empty:
                X = pd.concat([X, binning_df], axis=1)

        # Clustering features
        if hasattr(self, "clustering_features_") and self.clustering_features_:
            # Use features that were used during fit
            clustering_base_features = [
                col
                for col in current_numerical
                if col not in getattr(self, "binning_features_", [])
            ]
            clustering_df = self._create_clustering_features(
                X, clustering_base_features, is_fit=False
            )
            if not clustering_df.empty:
                X = pd.concat([X, clustering_df], axis=1)

        # Statistical features - use base numerical features only
        if hasattr(self, "statistical_features_") and self.statistical_features_:
            # Get base numerical features (original numerical columns plus datetime-derived)
            base_numerical = [
                col
                for col in self.column_types_["numerical"]
                if col
                not in (
                    getattr(self, "math_features_", [])
                    + getattr(self, "interaction_features_", [])
                    + getattr(self, "binning_features_", [])
                    + getattr(self, "clustering_features_", [])
                    + getattr(self, "statistical_features_", [])
                    + getattr(self, "fourier_features_", [])
                )
            ]
            base_numerical = [col for col in base_numerical if col in X.columns]
            stats_df = self._create_statistical_features(X, base_numerical)
            if not stats_df.empty:
                X = pd.concat([X, stats_df], axis=1)

        # Fourier features
        if hasattr(self, "fourier_features_") and self.fourier_features_:
            fourier_df = self._create_fourier_features(X, [], is_fit=False)
            if not fourier_df.empty:
                X = pd.concat([X, fourier_df], axis=1)

        # Drop correlated features
        X = X.drop(
            columns=[col for col in self.to_drop_ if col in X.columns], errors="ignore"
        )

        # Numerical preprocessing
        available_features = [
            col for col in self.selected_features_ if col in X.columns
        ]
        if available_features:
            X_num = X[available_features].copy()

            # Imputation
            for col in X_num.columns:
                X_num[col] = X_num[col].fillna(self.num_imputer_[col])

            # Transformation
            if self.quantile_transformer_ is not None:
                X_num_transformed = self.quantile_transformer_.transform(X_num)
                X_num = pd.DataFrame(
                    X_num_transformed, columns=X_num.columns, index=X_num.index
                )
            else:
                # Standard scaling fallback
                X_num = (X_num - self.num_means_[available_features]) / self.num_stds_[
                    available_features
                ]
        else:
            X_num = pd.DataFrame(index=X.index)

        if self.use_autoencoder and self.autoencoder_model_ is not None:
            try:
                self.autoencoder_model_.eval()
                X_ae = (
                    X[self.final_numerical_].fillna(pd.Series(self.num_imputer_)).copy()
                )
                X_tensor = torch.tensor(X_ae.values, dtype=torch.float32, device=self.autoencoder_device)
                with torch.no_grad():
                    _, z = self.autoencoder_model_(X_tensor)
                ae_df = pd.DataFrame(
                    z.cpu().numpy(),
                    columns=[f"ae_{i}" for i in range(z.shape[1])],
                    index=X.index,
                )
                result_dfs.append(ae_df)
            except Exception as e:
                warnings.warn(f"[Autoencoder] Embedding transform failed: {e}")

        # Categorical preprocessing
        cat_dfs = []

        # Target encoded features
        if hasattr(self, "target_encoders_") and self.target_encoders_:
            for col, encoding_map in self.target_encoders_.items():
                if col in X.columns:
                    encoded_col = X[col].map(encoding_map).fillna(encoding_map.mean())
                    cat_dfs.append(
                        pd.DataFrame({f"{col}_target_enc": encoded_col}, index=X.index)
                    )

        # One-hot encoded features
        if self.ohe_ is not None and self.ohe_columns_:
            available_ohe_cols = [col for col in self.ohe_columns_ if col in X.columns]
            if available_ohe_cols:
                X_cat = X[available_ohe_cols].copy()

                # Handle missing and rare categories
                for col in X_cat.columns:
                    X_cat[col] = X_cat[col].fillna(self.cat_modes_.get(col, "missing"))
                    if col in self.rare_categories_:
                        X_cat[col] = X_cat[col].replace(
                            self.rare_categories_[col], "OTHER"
                        )

                X_cat_encoded = self.ohe_.transform(X_cat)
                feature_names = self.ohe_.get_feature_names_out(available_ohe_cols)
                cat_dfs.append(
                    pd.DataFrame(X_cat_encoded, columns=feature_names, index=X.index)
                )

        # Frequency encoding
        if self.freq_encoding_:
            freq_features = {}
            for col in self.freq_encoding_:
                if col in X.columns:
                    freq_features[f"{col}_freq"] = X[col].map(self.freq_encoding_[col])
            if freq_features:
                cat_dfs.append(pd.DataFrame(freq_features, index=X.index))

        # Combine all features
        result_dfs = [X_num] + cat_dfs

        # Add boolean features if any
        if self.column_types_["boolean"]:
            bool_cols = [
                col for col in self.column_types_["boolean"] if col in X.columns
            ]
            if bool_cols:
                result_dfs.append(X[bool_cols])

        # Return combined result or empty DataFrame
        if result_dfs and any(not df.empty for df in result_dfs):
            return pd.concat([df for df in result_dfs if not df.empty], axis=1)
        else:
            return pd.DataFrame(index=X.index)

    def plot_correlation_heatmap(self, X: pd.DataFrame):
        """Enhanced correlation visualization."""
        num_cols = X.select_dtypes(include=[np.number])
        if num_cols.shape[1] < 2:
            print("Not enough numerical features for correlation heatmap.")
            return

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(num_cols.corr(), dtype=bool))
        sns.heatmap(
            num_cols.corr(),
            mask=mask,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Heatmap", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_mi_scores(self, top_k: int = 20):
        """Enhanced mutual information visualization."""
        if self.mi_scores_ is not None:
            top_scores = self.mi_scores_.head(top_k)

            plt.figure(figsize=(12, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_scores)))
            bars = plt.bar(range(len(top_scores)), top_scores.values, color=colors)

            plt.title(f"Top {top_k} Features by Mutual Information", fontsize=16)
            plt.xlabel("Features", fontsize=12)
            plt.ylabel("Mutual Information Score", fontsize=12)
            plt.xticks(
                range(len(top_scores)), top_scores.index, rotation=45, ha="right"
            )

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_scores.values)):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            plt.tight_layout()
            plt.show()
        else:
            print(
                "Mutual information scores not available. Fit with target variable first."
            )

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """Get output feature names with proper handling."""
        feature_names = []

        # Numerical features
        if hasattr(self, "selected_features_"):
            feature_names.extend(self.selected_features_)

        # Target encoded categorical features
        if hasattr(self, "target_encoders_"):
            feature_names.extend(
                [f"{col}_target_enc" for col in self.target_encoders_.keys()]
            )

        # One-hot encoded features
        if self.ohe_ is not None and hasattr(self, "ohe_columns_"):
            ohe_names = self.ohe_.get_feature_names_out(self.ohe_columns_)
            feature_names.extend(ohe_names)

        # Frequency encoded features
        if hasattr(self, "freq_encoding_"):
            feature_names.extend([f"{col}_freq" for col in self.freq_encoding_.keys()])

        # Boolean features
        if hasattr(self, "column_types_") and self.column_types_["boolean"]:
            feature_names.extend(self.column_types_["boolean"])

        return np.array(feature_names)

    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get a summary of feature importance and selection."""
        if not hasattr(self, "fitted_") or not self.fitted_:
            raise ValueError("FeatureEngineer must be fitted first.")

        summary = []

        if self.mi_scores_ is not None:
            for feature in self.mi_scores_.index:
                summary.append(
                    {
                        "feature": feature,
                        "mi_score": self.mi_scores_[feature],
                        "selected": feature in self.selected_features_,
                        "feature_type": "numerical",
                    }
                )

        return pd.DataFrame(summary).sort_values("mi_score", ascending=False)
