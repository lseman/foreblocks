# Standard library
import copy
import importlib
import math
import time
import warnings

# Scientific computing and data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Machine learning and preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Typing
from typing import Dict, List, Optional, Tuple, Union

# EWT
import ewtpy
from ewtpy import EWT1D

class TimeSeriesPreprocessor:
    """State-of-the-art preprocessing for time series data."""

    def __init__(
        self,
        normalize=True,
        differencing=False,
        detrend=True,
        apply_ewt=True,
        window_size=24,
        horizon=10,
        remove_outliers=True,
        outlier_threshold=0.05,
        outlier_method="iqr",
        impute_method="auto",
        ewt_bands=5,
        trend_imf_idx=0,
        log_transform=False,
        filter_window=5,
        filter_polyorder=2,
        apply_filter=True,
    ):

        self.normalize = normalize
        self.differencing = differencing
        self.detrend = detrend
        self.apply_ewt = apply_ewt
        self.window_size = window_size
        self.horizon = horizon
        self.ewt_bands = ewt_bands
        self.trend_imf_idx = trend_imf_idx
        self.log_transform = log_transform
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.apply_filter = apply_filter
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.outlier_method = outlier_method
        self.impute_method = impute_method

        self.scaler = None
        self.log_offset = None
        self.diff_values = None
        self.trend_component = None
        self.ewt_components = None
        self.ewt_boundaries = None

    def _plot_comparison(self, original, cleaned, title="Preprocessing Comparison", time_stamps=None):
        # Ensure 2D shape
        original = np.atleast_2d(original)
        cleaned = np.atleast_2d(cleaned)

        #print(original.shape, cleaned.shape)

        # Transpose if needed (1, n) → (n, 1)
        if original.shape[0] == 1:
            original = original.T
        if cleaned.shape[0] == 1:
            cleaned = cleaned.T

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x = time_stamps if time_stamps is not None else np.arange(original.shape[0])

        axs[0].set_title("Original", fontsize=12)
        axs[1].set_title("Cleaned", fontsize=12)

        n_features = original.shape[1]
        for i in range(n_features):
            axs[0].plot(x, original[:, i], label=f"Feature {i}", linewidth=1.5, alpha=0.8)
            axs[1].plot(x, cleaned[:, i], label=f"Feature {i}", linewidth=1.5, alpha=0.8)

        axs[0].legend(loc="upper right", fontsize=9)
        axs[1].legend(loc="upper right", fontsize=9)

        axs[0].grid(True)
        axs[1].grid(True)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    def generate_time_features(self, timestamps, freq='h'):
        df = pd.DataFrame({'ts': pd.to_datetime(timestamps)})
        df['month'] = df.ts.dt.month / 12.0
        df['day'] = df.ts.dt.day / 31.0
        df['weekday'] = df.ts.dt.weekday / 6.0
        df['hour'] = df.ts.dt.hour / 23.0 if freq in ['h', 'H'] else 0.0
        return df[['month', 'day', 'weekday', 'hour']].values.astype(np.float32)


    def adaptive_filter(self, data):
        return savgol_filter(data, self.filter_window, self.filter_polyorder, axis=0)

    def _remove_outliers(self, data):
        method = self.outlier_method
        data = np.asarray(data)
        is_1d = data.ndim == 1
        data = data.reshape(-1, 1) if is_1d else data
        print(
            f"[Outlier Removal] Method: {method}, Threshold: {self.outlier_threshold}"
        )

        if method == "iqr":
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower = Q1 - self.outlier_threshold * IQR
            upper = Q3 + self.outlier_threshold * IQR
            return np.clip(data, lower, upper)

        elif method == "zscore":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            z = (data - mean) / std
            return np.where(
                np.abs(z) <= self.outlier_threshold,
                data,
                np.clip(
                    data,
                    mean - self.outlier_threshold * std,
                    mean + self.outlier_threshold * std,
                ),
            )

        elif method == "mad":
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            z = np.abs((data - med) / (mad + 1e-6))
            return np.where(z <= self.outlier_threshold * 1.4826, data, med)

        elif method == "quantile":
            lower = np.percentile(data, self.outlier_threshold * 100, axis=0)
            upper = np.percentile(data, (1 - self.outlier_threshold) * 100, axis=0)
            return np.clip(data, lower, upper)

        elif method == "isolation_forest":
            df = pd.DataFrame(data)
            for col in df.columns:
                df[f"{col}_mean"] = df[col].rolling(5, min_periods=1).mean()
                df[f"{col}_std"] = df[col].rolling(5, min_periods=1).std().fillna(0)
            X = StandardScaler().fit_transform(
                df.fillna(method="ffill").fillna(method="bfill")
            )
            preds = IsolationForest(
                contamination=self.outlier_threshold, random_state=42
            ).fit_predict(X)
            result = np.where(preds[:, None] == 1, data, np.nan)
            return result

        elif method == "lof":
            preds = LocalOutlierFactor(
                n_neighbors=20, contamination=self.outlier_threshold
            ).fit_predict(data)
            return np.where(preds[:, None] == 1, data, np.nan)

        elif method == "ecod":
            from pyod.models.ecod import ECOD

            preds = ECOD().fit(data).predict(data)
            return np.where(preds[:, None] == 0, data, np.nan)

        else:
            raise ValueError(f"Unsupported outlier removal method: {method}")

    def _impute_missing(self, data):
        df = pd.DataFrame(data)
        method = self.impute_method
        print(f"[Imputation] Method: {method}")

        if method == "auto":
            missing = df.isna().mean()
            result = pd.DataFrame(index=df.index)
            for col in df.columns:
                col_missing = missing[col]
                if col_missing < 0.05:
                    print(f" - Feature {col}: interpolate")
                    result[col] = df[col].interpolate().ffill().bfill()
                elif col_missing < 0.2 and df.corr()[col].drop(col).abs().max() > 0.6:
                    print(f" - Feature {col}: knn")
                    result[col] = (
                        KNNImputer(n_neighbors=3).fit_transform(df[[col]]).ravel()
                    )
                else:
                    try:
                        from fancyimpute import IterativeImputer

                        print(f" - Feature {col}: iterative")
                        result[col] = (
                            IterativeImputer().fit_transform(df[[col]]).ravel()
                        )
                    except ImportError:
                        print(f" - Feature {col}: mean (fallback)")
                        result[col] = df[col].fillna(df[col].mean())
            return result.values

        elif method == "interpolate":
            return df.interpolate().ffill().bfill().values
        elif method == "mean":
            return df.fillna(df.mean()).values
        elif method == "ffill":
            return df.ffill().bfill().values
        elif method == "bfill":
            return df.bfill().ffill().values
        elif method == "knn":
            return KNNImputer(n_neighbors=5).fit_transform(df)
        elif method == "iterative":
            try:
                from fancyimpute import IterativeImputer

                return IterativeImputer().fit_transform(df)
            except ImportError:
                warnings.warn("fancyimpute not installed. Using mean instead.")
                return df.fillna(df.mean()).values
        else:
            raise ValueError(f"Unsupported impute method: {self.impute_method}")

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size - self.horizon + 1):
            X.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size : i + self.window_size + self.horizon])
        return np.array(X), np.array(y)
    
    def fit_transform(self, data, time_stamps=None, feats=None):
        processed = data.copy()
        original = data.copy()

        if self.log_transform:
            min_val = processed.min(axis=0)
            self.log_offset = np.where(min_val <= 0, np.abs(min_val) + 1.0, 0.0)
            processed = np.log(processed + self.log_offset)

        imputed = self._impute_missing(processed)
        self._plot_comparison(processed, imputed, title="After Imputation")
        processed = imputed
        
        if self.remove_outliers:
            cleaned = np.zeros_like(processed)
            for i in range(processed.shape[1]):
                _clean = self._remove_outliers(processed[:, i])
                cleaned[:, i] = _clean.ravel()
                
            self._plot_comparison(processed, cleaned, title="After Outlier Removal")
            processed = cleaned

        if self.apply_ewt:
            self.ewt_components = []
            self.ewt_boundaries = []
            if self.detrend:
                self.trend_component = np.zeros_like(processed)

            for i in range(processed.shape[1]):
                signal = processed[:, i]
                ewt, _, boundaries = EWT1D(signal, N=self.ewt_bands)
                self.ewt_components.append(ewt)
                self.ewt_boundaries.append(boundaries)

                if self.detrend:
                    trend = ewt[:, self.trend_imf_idx]
                    self.trend_component[:, i] = trend
                    detrended = signal - trend

                    # Plotting
                    x = time_stamps if time_stamps is not None else np.arange(len(signal))
                    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

                    axs[0].plot(x, signal, label="Original", alpha=0.8)
                    axs[0].plot(x, trend, label="Trend", linestyle="-", linewidth=2.5)
                    axs[0].set_title(f"Feature {i} - Original and Trend", fontsize=12)
                    axs[0].legend()
                    axs[0].grid(True)

                    axs[1].plot(x, detrended, label="Detrended", color="tab:orange", linewidth=2)
                    axs[1].set_title(f"Feature {i} - Detrended", fontsize=12)
                    axs[1].legend()
                    axs[1].grid(True)

                    plt.tight_layout()
                    plt.show()

                    processed[:, i] = detrended

        if self.apply_filter:
            filtered = self.adaptive_filter(processed)
            self._plot_comparison(processed, filtered, title="After Adaptive Filtering")
            processed = filtered

        if self.differencing:
            self.diff_values = processed[0:1].copy()
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0

        if self.normalize:
            self.scaler = StandardScaler()
            processed = self.scaler.fit_transform(processed)

        if time_stamps is not None:
            time_features = self.generate_time_features(time_stamps)
            processed = np.concatenate((processed, time_features), axis=1)

        return self._create_sequences(processed, feats) + (processed,)

    def transform(self, data):
        """
        Transform data using fitted parameters

        Args:
            data: Time series data of shape [samples, features]

        Returns:
            X: Input sequences
            y: Target sequences (if possible to create)
        """
        # Check if the preprocessor has been fitted
        if self.normalize and self.scaler is None:
            raise ValueError(
                "Preprocessor has not been fitted yet. Call fit_transform first."
            )

        # Make a copy to avoid modifying the original data
        processed_data = data.copy()

        # Apply log transform if specified
        if self.log_transform:
            if self.log_offset is None:
                raise ValueError("Log offset not set. Call fit_transform first.")
            processed_data = np.log(processed_data + self.log_offset)

        # If multivariate, process each feature independently
        n_samples, n_features = processed_data.shape

        # Store original data for EWT processing
        original_data = processed_data.copy()

        # Apply EWT if specified
        if self.apply_ewt:
            if self.ewt_boundaries is None:
                raise ValueError("EWT boundaries not set. Call fit_transform first.")

            # Process each feature independently for EWT
            for i in range(n_features):
                # Apply EWT to the current feature using stored boundaries
                feature_data = original_data[:, i]

                # For transform, use the boundaries detected during fit
                ewt, _, _ = ewtpy.EWT1D(
                    feature_data,
                    N=len(self.ewt_boundaries[i]),
                    detect="given_bounds",
                    boundaries=self.ewt_boundaries[i],
                )

                # If detrending is enabled, remove the trend component
                if self.detrend:
                    # Extract the trend component
                    trend = ewt[:, self.trend_imf_idx]
                    plt.plot(trend)
                    plt.title(f"Trend Component for Feature {i}")
                    plt.show()

                    # Remove trend from the data
                    processed_data[:, i] = processed_data[:, i] - trend
                    plt.plot(processed_data[:, i])
                    plt.title(f"Processed Feature {i} after Detrending")
                    plt.show()

        # Apply differencing if specified
        if self.differencing:
            if self.diff_values is None:
                raise ValueError(
                    "Differencing values not set. Call fit_transform first."
                )

            # Store the first value for inverse transform
            prev_value = processed_data[0:1].copy()

            # Apply differencing
            processed_data[1:] = np.diff(processed_data, axis=0)
            processed_data[0] = 0

        # Apply normalization if specified
        if self.normalize:
            processed_data = self.scaler.transform(processed_data)

        # Apply adaptive filtering if specified
        if self.apply_filter:
            processed_data = self.adaptive_filter(processed_data)

        # Create input/output sequences
        X, y = self._create_sequences(processed_data)

        return X, y

    def inverse_transform(self, predictions):
        """
        Inverse transform forecasted values back to original scale

        Args:
            predictions: Forecasted values

        Returns:
            Predictions in the original scale
        """
        # Inverse normalization
        if self.normalize:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit_transform first.")

            predictions = self.scaler.inverse_transform(predictions)

        # Inverse differencing
        if self.differencing:
            if self.diff_values is None:
                raise ValueError(
                    "Differencing values not set. Call fit_transform first."
                )

            # Initialize with the last known value
            last_value = self.diff_values[-1]
            result = np.zeros_like(predictions)

            # Integrate the differences
            for i in range(len(predictions)):
                last_value = last_value + predictions[i]
                result[i] = last_value

            predictions = result

        # Add back trend component if detrending was applied
        if self.detrend and self.trend_component is not None:
            # For simplicity, we'll extrapolate the trend from the training data
            # In a production system, you might want to use a more sophisticated approach

            # Check if we have enough trend data
            if len(self.trend_component) >= len(predictions):
                # Use the corresponding portion of the trend
                trend_to_add = self.trend_component[: len(predictions)]
            else:
                # Extrapolate trend for longer predictions
                # This is a simple linear extrapolation
                n_samples, n_features = predictions.shape
                trend_to_add = np.zeros_like(predictions)

                for i in range(n_features):
                    # Extract trend for the current feature
                    feature_trend = self.trend_component[:, i]

                    # Compute slope for extrapolation (using last few points)
                    window = min(10, len(feature_trend) // 2)
                    slope = (feature_trend[-1] - feature_trend[-window - 1]) / window

                    # Extrapolate trend
                    for j in range(n_samples):
                        if j < len(feature_trend):
                            trend_to_add[j, i] = feature_trend[j]
                        else:
                            # Linear extrapolation
                            trend_to_add[j, i] = feature_trend[-1] + slope * (
                                j - len(feature_trend) + 1
                            )

            # Add trend back to predictions
            predictions = predictions + trend_to_add

        # Inverse log transform if applied
        if self.log_transform:
            if self.log_offset is None:
                raise ValueError("Log offset not set. Call fit_transform first.")
            predictions = np.exp(predictions) - self.log_offset

        return predictions
    
    def _create_sequences(self, data, feats=None):
        """
        Create input/output sequences for training or inference

        Args:
            data: Preprocessed time series data
            feats: List of column indices to extract for y

        Returns:
            X: Input sequences of shape [n_sequences, window_size, n_features]
            y: Target sequences of shape [n_sequences, horizon, len(feats)]
        """
        n_samples, n_features = data.shape
        feats = list(range(n_features)) if feats is None else feats
        X, y = [], []

        for i in range(n_samples - self.window_size - self.horizon + 1):
            X.append(data[i : i + self.window_size])  # full features
            y_seq = data[i + self.window_size : i + self.window_size + self.horizon]
            y.append(y_seq[:, feats])  # select only specified features
        print(f"X shape: {np.array(X).shape}, y shape: {np.array(y).shape}")
        return np.array(X), np.array(y)


    def get_ewt_components(self):
        """
        Get the decomposed EWT components (IMFs)

        Returns:
            List of IMF components if EWT was applied, None otherwise
        """
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self):
        """
        Get the trend component extracted during detrending

        Returns:
            Trend component if detrending was applied, None otherwise
        """
        return self.trend_component if self.detrend else None


# Example DataLoader for time series
class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset for time series data"""

    def __init__(self, X, y=None):
        """
        Initialize dataset

        Args:
            X: Input sequences of shape [n_sequences, seq_len, n_features]
            y: Target sequences of shape [n_sequences, horizon, n_features]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=32):
    """
    Create PyTorch DataLoaders for training and validation

    Args:
        X_train: Training input sequences
        y_train: Training target sequences
        X_val: Validation input sequences
        y_val: Validation target sequences
        batch_size: Batch size

    Returns:
        train_dataloader: DataLoader for training
        val_dataloader: DataLoader for validation (if validation data provided)
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Create validation dataloader if validation data provided
    if X_val is not None and y_val is not None:
        val_dataset = TimeSeriesDataset(X_val, y_val)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        return train_dataloader, val_dataloader

    return train_dataloader, None
