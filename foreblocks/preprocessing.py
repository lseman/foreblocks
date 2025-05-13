# Standard Library
import copy
import importlib
import math
import time
import warnings

# Scientific Computing and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Machine Learning and Preprocessing
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

# Empirical Wavelet Transform
import ewtpy
from ewtpy import EWT1D

# Typing
from typing import Any, Dict, List, Optional, Tuple, Union


class TimeSeriesPreprocessor:
    """
    State-of-the-art preprocessing for time series data with advanced features:
    - Automatic configuration based on data statistics
    - Log transformation for skewed data
    - Outlier removal with multiple methods
    - Empirical Wavelet Transform (EWT) for decomposition
    - Detrending and differencing for stationarity
    - Adaptive filtering with Savitzky-Golay
    - Time feature generation
    - Missing value imputation with multiple strategies
    """
    def __init__(
        self,
        normalize=True,
        differencing=False,
        detrend=False,
        apply_ewt=False,
        window_size=24,
        horizon=10,
        remove_outliers=False,
        outlier_threshold=0.05,
        outlier_method="iqr",
        impute_method="auto",
        ewt_bands=5,
        trend_imf_idx=0,
        log_transform=False,
        filter_window=5,
        filter_polyorder=2,
        apply_filter=False,
        self_tune=False,
        generate_time_features=False,
        apply_imputation=False,
    ):
        """
        Initialize the TimeSeriesPreprocessor with the specified parameters.
        
        Args:
            normalize: Whether to normalize data using StandardScaler
            differencing: Whether to apply differencing for stationarity
            detrend: Whether to remove trend component using EWT
            apply_ewt: Whether to apply Empirical Wavelet Transform
            window_size: Size of sliding window for sequence creation
            horizon: Prediction horizon length
            remove_outliers: Whether to remove outliers
            outlier_threshold: Threshold for outlier detection
            outlier_method: Method for outlier detection (iqr, zscore, mad, quantile, isolation_forest, lof, ecod)
            impute_method: Method for missing value imputation (auto, mean, interpolate, ffill, bfill, knn, iterative)
            ewt_bands: Number of frequency bands for EWT
            trend_imf_idx: Index of IMF component considered as trend
            log_transform: Whether to apply log transformation for skewed data
            filter_window: Window size for Savitzky-Golay filter
            filter_polyorder: Polynomial order for Savitzky-Golay filter
            apply_filter: Whether to apply Savitzky-Golay filter
            self_tune: Whether to automatically configure preprocessing based on data statistics
            generate_time_features: Whether to generate calendar features from timestamps
            apply_imputation: Whether to apply imputation for missing values
        """
        # Configuration parameters
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
        self.self_tune = self_tune
        self.generate_time_features = generate_time_features
        self.apply_imputation = apply_imputation
        
        # Fitted parameters (initialized as None)
        self.scaler = None
        self.log_offset = None
        self.diff_values = None
        self.trend_component = None
        self.ewt_components = None
        self.ewt_boundaries = None
        self.log_transform_flags = None

    def auto_configure(self, data: np.ndarray) -> None:
        """
        Automatically configure preprocessing parameters based on data statistics.
        
        Args:
            data: Input time series data of shape [samples, features]
        """
        if not self.self_tune:
            return
            
        print("\n📊 [Self-Tuning Preprocessing Configuration]")
        
        # Basic stats
        mean_vals = np.nanmean(data, axis=0)
        std_vals = np.nanstd(data, axis=0)
        missing_rate = np.mean(np.isnan(data))
        
        # Skewness analysis for log transform
        from scipy.stats import skew
        skews = skew(data, nan_policy='omit')
        self.log_transform_flags = (np.abs(skews) > 1).tolist()
        print(f"→ Skewness per feature: {np.round(skews, 3)}")
        print(f"→ Log transform (per feature): {self.log_transform_flags}")
        
        # Stationarity detection with ADF test
        try:
            from statsmodels.tsa.stattools import adfuller
            pvals = [
                adfuller(data[:, i][~np.isnan(data[:, i])])[1]
                if np.sum(~np.isnan(data[:, i])) > 10 else 1.0
                for i in range(data.shape[1])
            ]
            self.detrend = any(p > 0.05 for p in pvals)
            print(f"→ ADF p-values: {np.round(pvals, 4)} → Detrend? {'✅' if self.detrend else '❌'}")
        except ImportError:
            print("→ ADF test skipped (statsmodels not installed)")
        
        # Seasonality detection
        try:
            from statsmodels.tsa.seasonal import STL
            seasonal_flags = []
            for i in range(data.shape[1]):
                try:
                    res = STL(data[:, i], period=24, robust=True).fit()
                    seasonal_flags.append(res.seasonal.std() > 0.1 * res.trend.std())
                except Exception:
                    seasonal_flags.append(False)
            print(f"→ Seasonality flags (STL): {seasonal_flags}")
        except ImportError:
            print("→ STL skipped (statsmodels not installed)")
        
        # SNR for filter
        stds = np.nanstd(data, axis=0)
        valid_stds = stds[~np.isnan(stds) & (stds > 1e-8)]
        if valid_stds.size > 0:
            snr = np.nanmean(np.abs(data)) / np.mean(valid_stds)
        else:
            snr = 0.0  # or float('inf') depending on your logic
        self.apply_filter = snr < 2
        print(f"→ SNR = {snr:.2f} → Apply filter? {'✅' if self.apply_filter else '❌'}")
        
        # Imputation suggestion
        if missing_rate == 0:
            self.impute_method = None
        elif missing_rate < 0.05:
            self.impute_method = "interpolate"
        elif missing_rate < 0.15:
            self.impute_method = "knn"
        else:
            self.impute_method = "iterative"
        print(f"→ Missing rate: {missing_rate:.3f} → Imputation: {self.impute_method or 'None'}")
        
        # EWT band estimation via entropy
        from scipy.stats import entropy
        band_suggestions = []
        for i in range(data.shape[1]):
            valid_data = data[:, i][~np.isnan(data[:, i])]
            if len(valid_data) > 0:
                hist, _ = np.histogram(valid_data, bins=20, density=True)
                band_suggestions.append(int(np.clip(entropy(hist) * 1.5, 2, 10)))
        
        if band_suggestions:
            self.ewt_bands = int(np.round(np.mean(band_suggestions)))
            print(f"→ EWT bands (entropy-based): {self.ewt_bands}")
        
        # Outlier method suggestion
        if data.shape[0] > 1000:
            self.outlier_method = "ecod"
        elif np.any(np.abs(skews) > 2.5):
            self.outlier_method = "zscore"
        else:
            self.outlier_method = "iqr"
        print(f"→ Outlier method: {self.outlier_method}")
        
        # Summary (optional pretty table)
        try:
            from tabulate import tabulate
            print(tabulate([
                ["Shape", data.shape],
                ["Detrend", self.detrend],
                ["Log Transform (any)", any(self.log_transform_flags)],
                ["Imputation", self.impute_method],
                ["EWT Bands", self.ewt_bands],
                ["Filter?", self.apply_filter],
                ["Outlier Method", self.outlier_method]
            ], headers=["Config", "Value"], tablefmt="pretty"))
        except ImportError:
            pass
            
        print("✅ Self-tuning configuration complete.\n")

    def fit_transform(self, data: np.ndarray, time_stamps=None, feats=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess input data and return (X, y, full_processed_data).
        
        Args:
            data: Input time series data of shape [samples, features]
            time_stamps: Optional timestamps for the data
            feats: Optional subset of features to use for target
            
        Returns:
            X: Input sequences of shape [num_sequences, window_size, features]
            y: Target sequences of shape [num_sequences, horizon, target_features]
            processed: Full processed data of shape [samples, features]
        """
        # If processed is tensor, convert to numpy
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        processed = data.copy()
        
        # Auto-configure preprocessing parameters if requested
        self.auto_configure(processed)
        
        # Apply log transform if needed
        if self.log_transform:
            min_val = processed.min(axis=0)
            self.log_offset = np.where(min_val <= 0, np.abs(min_val) + 1.0, 0.0)
            processed = np.log(processed + self.log_offset)
        
        # Impute missing values if needed
        if self.apply_imputation:
            processed = self._impute_missing(processed)
            self._plot_comparison(data, processed, "After Imputation", time_stamps)
        
        # Remove outliers if needed
        if self.remove_outliers:
            cleaned = np.stack([
                self._remove_outliers(processed[:, i]).ravel() 
                for i in range(processed.shape[1])
            ], axis=1)
            self._plot_comparison(processed, cleaned, "After Outlier Removal", time_stamps)
            processed = cleaned
        
        # Apply EWT and detrending if needed
        if self.apply_ewt:
            processed = self._apply_ewt_and_detrend(processed, time_stamps)
        
        # Apply filtering if needed
        if self.apply_filter:
            filtered = self.adaptive_filter(processed)
            self._plot_comparison(processed, filtered, "After Adaptive Filtering", time_stamps)
            processed = filtered
        
        # Apply differencing if needed
        if self.differencing:
            self.diff_values = processed[0:1].copy()
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0
        
        # Normalize if needed
        if self.normalize:
            self.scaler = StandardScaler()
            processed = self.scaler.fit_transform(processed)
        
        # Generate time features if requested
        if time_stamps is not None and self.generate_time_features:
            time_feats = self.generate_time_features(time_stamps)
            processed = np.concatenate((processed, time_feats), axis=1)
        
        # Create sequences and return
        X, y = self._create_sequences(processed, feats)
        return X, y, processed

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the same transformation as in fit_transform.
        
        Args:
            data: Input time series data
            
        Returns:
            X: Input sequences
        """
        if self.normalize and self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        processed = data.copy()
        
        # Apply log transform if needed
        if self.log_transform:
            processed = np.log(processed + self.log_offset)
        
        # Apply EWT if needed
        if self.apply_ewt:
            if self.ewt_boundaries is None:
                raise ValueError("EWT not fitted. Call fit_transform first.")
                
            for i in range(processed.shape[1]):
                try:
                    from PyEWT import EWT1D
                    ewt, _, _ = EWT1D(processed[:, i], N=len(self.ewt_boundaries[i]), 
                                     detect="given_bounds", boundaries=self.ewt_boundaries[i])
                    
                    # Apply detrending if needed
                    if self.detrend:
                        processed[:, i] -= ewt[:, self.trend_imf_idx]
                except ImportError:
                    warnings.warn("PyEWT not installed. Skipping EWT.")
                    break
        
        # Apply differencing if needed
        if self.differencing:
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0
        
        # Apply normalization if needed
        if self.normalize:
            processed = self.scaler.transform(processed)
        
        # Apply filtering if needed
        if self.apply_filter:
            processed = self.adaptive_filter(processed)
        
        # Create sequences and return
        return self._create_sequences(processed)[0]

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predicted values to the original scale.
        
        Args:
            predictions: Predicted values
            
        Returns:
            Original-scale predictions
        """
        # Inverse normalization
        if self.normalize:
            predictions = self.scaler.inverse_transform(predictions)
        
        # Inverse differencing
        if self.differencing:
            last_value = self.diff_values[-1]
            result = np.zeros_like(predictions)
            for i in range(len(predictions)):
                last_value += predictions[i]
                result[i] = last_value
            predictions = result
        
        # Add trend back if detrended
        if self.detrend and self.trend_component is not None:
            n = len(predictions)
            trend_to_add = np.zeros_like(predictions)
            
            for i in range(predictions.shape[1]):
                trend = self.trend_component[:, i]
                
                if n <= len(trend):
                    trend_to_add[:, i] = trend[:n]
                else:
                    # Extrapolate trend for future points
                    look_back = min(10, len(trend))
                    slope = (trend[-1] - trend[-look_back]) / look_back
                    
                    for j in range(n):
                        if j < len(trend):
                            trend_to_add[j, i] = trend[j]
                        else:
                            trend_to_add[j, i] = trend[-1] + slope * (j - len(trend) + 1)
            
            predictions += trend_to_add
        
        # Inverse log transform
        if self.log_transform:
            predictions = np.exp(predictions) - self.log_offset
        
        return predictions

    def _remove_outliers(self, data_col: np.ndarray) -> np.ndarray:
        """
        Remove outliers from a data column using the specified method.
        
        Args:
            data_col: Data column
            
        Returns:
            Cleaned data column
        """
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
            try:
                from pyod.models.ecod import ECOD
                pred = ECOD().fit(x).predict(x)
                return np.where(pred == 0, x, np.nan)
            except ImportError:
                warnings.warn("pyod not installed. Using IQR instead.")
                Q1, Q3 = np.percentile(x, [25, 75], axis=0)
                IQR = Q3 - Q1
                return np.clip(x, Q1 - threshold * IQR, Q3 + threshold * IQR)
        
        raise ValueError(f"Unsupported outlier method: {method}")

    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        """
        Impute missing values in the data.
        
        Args:
            data: Input data with potential missing values
            
        Returns:
            Data with imputed values
        """
        df = pd.DataFrame(data)
        method = self.impute_method
        
        if method == "auto":
            filled = df.copy()
            for col in df.columns:
                missing_rate = df[col].isna().mean()
                if missing_rate < 0.05:
                    filled[col] = df[col].interpolate().ffill().bfill()
                elif missing_rate < 0.2:
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

    def _apply_ewt_and_detrend(self, data: np.ndarray, time_stamps=None) -> np.ndarray:
        """
        Apply Empirical Wavelet Transform and detrend data.
        
        Args:
            data: Input data
            time_stamps: Optional timestamps
            
        Returns:
            Transformed data
        """
        try:
            from PyEWT import EWT1D
        except ImportError:
            warnings.warn("PyEWT not installed. Skipping EWT.")
            return data
            
        self.ewt_components = []
        self.ewt_boundaries = []
        
        if self.detrend:
            self.trend_component = np.zeros_like(data)
            
        for i in range(data.shape[1]):
            signal = data[:, i]
            ewt, _, bounds = EWT1D(signal, N=self.ewt_bands)
            self.ewt_components.append(ewt)
            print(f"→ EWT bands for feature {i}: {bounds}")
            self.ewt_boundaries.append(bounds)
            
            if self.detrend:
                trend = ewt[:, self.trend_imf_idx]
                self.trend_component[:, i] = trend
                data[:, i] = signal - trend
                
        return data

    def adaptive_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to the data.
        
        Args:
            data: Input data
            
        Returns:
            Filtered data
        """
        # Ensure window size is odd
        window = self.filter_window
        if window % 2 == 0:
            window += 1
            
        # Ensure window size is at least polyorder + 1
        window = max(window, self.filter_polyorder + 2)
        
        # Apply filter
        return savgol_filter(data, window, self.filter_polyorder, axis=0)

    def generate_time_features(self, timestamps, freq='h') -> np.ndarray:
        """
        Generate time features from timestamps.
        
        Args:
            timestamps: Array of timestamps
            freq: Frequency of the data ('h' for hourly, etc.)
            
        Returns:
            Time features array
        """
        df = pd.DataFrame({'ts': pd.to_datetime(timestamps)})
        df['month'] = df.ts.dt.month / 12.0
        df['day'] = df.ts.dt.day / 31.0
        df['weekday'] = df.ts.dt.weekday / 6.0
        df['hour'] = df.ts.dt.hour / 23.0 if freq.lower() == 'h' else 0.0
        
        return df[['month', 'day', 'weekday', 'hour']].values.astype(np.float32)

    def _create_sequences(self, data: np.ndarray, feats=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences from the data.
        
        Args:
            data: Input data
            feats: Optional subset of features for the target
            
        Returns:
            X: Input sequences
            y: Target sequences
        """
        feats = list(range(data.shape[1])) if feats is None else feats
        X, y = [], []
        
        max_idx = len(data) - self.window_size - self.horizon + 1
        for i in range(max_idx):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size:i+self.window_size+self.horizon][:, feats])
            
        return np.array(X), np.array(y)

    def _plot_comparison(self, original: np.ndarray, cleaned: np.ndarray, 
                        title: str = "Preprocessing Comparison", time_stamps=None) -> None:
        """
        Plot a comparison between original and processed data.
        
        Args:
            original: Original data
            cleaned: Processed data
            title: Plot title
            time_stamps: Optional timestamps for x-axis
        """
        original = np.atleast_2d(original)
        cleaned = np.atleast_2d(cleaned)
        
        if original.shape[0] == 1: 
            original = original.T
        if cleaned.shape[0] == 1: 
            cleaned = cleaned.T
            
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

    def get_ewt_components(self) -> Optional[List]:
        """Get the EWT components if EWT was applied."""
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self) -> Optional[np.ndarray]:
        """Get the trend component if detrending was applied."""
        return self.trend_component if self.detrend else None