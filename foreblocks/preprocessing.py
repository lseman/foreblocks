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
        self.outlier_threshold = outlier_threshold
        self.outlier_method = outlier_method
        self.impute_method = impute_method
        self.ewt_bands = ewt_bands
        self.trend_imf_idx = trend_imf_idx
        self.log_transform = log_transform
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.apply_filter = apply_filter
        self.remove_outliers = remove_outliers

        # Fitted parameters
        self.scaler = None
        self.log_offset = None
        self.diff_values = None
        self.trend_component = None
        self.ewt_components = None
        self.ewt_boundaries = None

    def fit_transform(self, data, time_stamps=None, feats=None):
        """Preprocess input data and return (X, y, full_processed_data)."""
        processed = data.copy()

        if self.log_transform:
            min_val = processed.min(axis=0)
            self.log_offset = np.where(min_val <= 0, np.abs(min_val) + 1.0, 0.0)
            processed = np.log(processed + self.log_offset)

        processed = self._impute_missing(processed)
        self._plot_comparison(data, processed, "After Imputation", time_stamps)

        if self.remove_outliers:
            cleaned = np.stack([self._remove_outliers(processed[:, i]).ravel() for i in range(processed.shape[1])], axis=1)
            self._plot_comparison(processed, cleaned, "After Outlier Removal", time_stamps)
            processed = cleaned

        if self.apply_ewt:
            processed = self._apply_ewt_and_detrend(processed, time_stamps)

        if self.apply_filter:
            filtered = self.adaptive_filter(processed)
            self._plot_comparison(processed, filtered, "After Adaptive Filtering", time_stamps)
            processed = filtered

        if self.differencing:
            self.diff_values = processed[0:1].copy()
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0

        if self.normalize:
            self.scaler = StandardScaler()
            processed = self.scaler.fit_transform(processed)

        if time_stamps is not None:
            time_feats = self.generate_time_features(time_stamps)
            processed = np.concatenate((processed, time_feats), axis=1)

        return self._create_sequences(processed, feats) + (processed,)

    def transform(self, data):
        """Apply the same transformation as in fit_transform."""
        if self.normalize and self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        processed = data.copy()

        if self.log_transform:
            processed = np.log(processed + self.log_offset)

        if self.apply_ewt:
            if self.ewt_boundaries is None:
                raise ValueError("EWT not fitted. Call fit_transform first.")
            for i in range(processed.shape[1]):
                ewt, _, _ = EWT1D(processed[:, i], N=len(self.ewt_boundaries[i]), detect="given_bounds", boundaries=self.ewt_boundaries[i])
                if self.detrend:
                    processed[:, i] -= ewt[:, self.trend_imf_idx]

        if self.differencing:
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0

        if self.normalize:
            processed = self.scaler.transform(processed)

        if self.apply_filter:
            processed = self.adaptive_filter(processed)

        return self._create_sequences(processed)

    def inverse_transform(self, predictions):
        """Inverse transform predicted values to the original scale."""
        if self.normalize:
            predictions = self.scaler.inverse_transform(predictions)

        if self.differencing:
            last_value = self.diff_values[-1]
            result = np.zeros_like(predictions)
            for i in range(len(predictions)):
                last_value += predictions[i]
                result[i] = last_value
            predictions = result

        if self.detrend and self.trend_component is not None:
            n = len(predictions)
            trend_to_add = np.zeros_like(predictions)
            for i in range(predictions.shape[1]):
                trend = self.trend_component[:, i]
                if n <= len(trend):
                    trend_to_add[:, i] = trend[:n]
                else:
                    slope = (trend[-1] - trend[-min(10, len(trend))]) / min(10, len(trend))
                    for j in range(n):
                        trend_to_add[j, i] = trend[j] if j < len(trend) else trend[-1] + slope * (j - len(trend))
            predictions += trend_to_add

        if self.log_transform:
            predictions = np.exp(predictions) - self.log_offset

        return predictions

    def _remove_outliers(self, data_col):
        method = self.outlier_method
        threshold = self.outlier_threshold
        x = data_col.reshape(-1, 1)

        if method == "iqr":
            Q1, Q3 = np.percentile(x, [25, 75], axis=0)
            IQR = Q3 - Q1
            return np.clip(x, Q1 - threshold * IQR, Q3 + threshold * IQR)

        elif method == "zscore":
            z = (x - x.mean()) / x.std()
            return np.clip(x, x.mean() - threshold * x.std(), x.mean() + threshold * x.std())

        elif method == "mad":
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            return np.where(np.abs((x - med) / (mad + 1e-6)) <= threshold * 1.4826, x, med)

        elif method == "quantile":
            low, high = np.percentile(x, [threshold * 100, 100 - threshold * 100])
            return np.clip(x, low, high)

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            pred = IsolationForest(contamination=threshold).fit_predict(x)
            return np.where(pred == 1, x, np.nan)

        elif method == "lof":
            pred = LocalOutlierFactor(n_neighbors=20, contamination=threshold).fit_predict(x)
            return np.where(pred == 1, x, np.nan)

        elif method == "ecod":
            from pyod.models.ecod import ECOD
            pred = ECOD().fit(x).predict(x)
            return np.where(pred == 0, x, np.nan)

        raise ValueError(f"Unsupported outlier method: {method}")

    def _impute_missing(self, data):
        df = pd.DataFrame(data)
        method = self.impute_method

        if method == "auto":
            filled = df.copy()
            for col in df.columns:
                if df[col].isna().mean() < 0.05:
                    filled[col] = df[col].interpolate().ffill().bfill()
                elif df[col].isna().mean() < 0.2:
                    filled[col] = KNNImputer(n_neighbors=3).fit_transform(df[[col]]).ravel()
                else:
                    try:
                        from fancyimpute import IterativeImputer
                        filled[col] = IterativeImputer().fit_transform(df[[col]]).ravel()
                    except ImportError:
                        filled[col] = df[col].fillna(df[col].mean())
            return filled.values

        if method == "mean":
            return df.fillna(df.mean()).values
        elif method == "interpolate":
            return df.interpolate().ffill().bfill().values
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
                warnings.warn("fancyimpute not installed, using mean fallback")
                return df.fillna(df.mean()).values

        raise ValueError(f"Unsupported imputation method: {method}")

    def _apply_ewt_and_detrend(self, data, time_stamps):
        self.ewt_components = []
        self.ewt_boundaries = []
        if self.detrend:
            self.trend_component = np.zeros_like(data)

        for i in range(data.shape[1]):
            signal = data[:, i]
            ewt, _, bounds = EWT1D(signal, N=self.ewt_bands)
            self.ewt_components.append(ewt)
            self.ewt_boundaries.append(bounds)

            if self.detrend:
                trend = ewt[:, self.trend_imf_idx]
                self.trend_component[:, i] = trend
                data[:, i] = signal - trend
        return data

    def adaptive_filter(self, data):
        return savgol_filter(data, self.filter_window, self.filter_polyorder, axis=0)

    def generate_time_features(self, timestamps, freq='h'):
        df = pd.DataFrame({'ts': pd.to_datetime(timestamps)})
        df['month'] = df.ts.dt.month / 12.0
        df['day'] = df.ts.dt.day / 31.0
        df['weekday'] = df.ts.dt.weekday / 6.0
        df['hour'] = df.ts.dt.hour / 23.0 if freq.lower() == 'h' else 0.0
        return df[['month', 'day', 'weekday', 'hour']].values.astype(np.float32)

    def _create_sequences(self, data, feats=None):
        feats = list(range(data.shape[1])) if feats is None else feats
        X, y = [], []
        for i in range(len(data) - self.window_size - self.horizon + 1):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size:i+self.window_size+self.horizon][:, feats])
        return np.array(X), np.array(y)

    def _plot_comparison(self, original, cleaned, title="Preprocessing Comparison", time_stamps=None):
        original = np.atleast_2d(original)
        cleaned = np.atleast_2d(cleaned)
        if original.shape[0] == 1: original = original.T
        if cleaned.shape[0] == 1: cleaned = cleaned.T

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x = time_stamps if time_stamps is not None else np.arange(original.shape[0])

        for i in range(original.shape[1]):
            axs[0].plot(x, original[:, i], label=f"Feature {i}")
            axs[1].plot(x, cleaned[:, i], label=f"Feature {i}")
        axs[0].set_title("Original")
        axs[1].set_title("Cleaned")
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        axs[1].grid(True)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def get_ewt_components(self):
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self):
        return self.trend_component if self.detrend else None
