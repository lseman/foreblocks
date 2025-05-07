# TimeSeriesPreprocessor Tutorial

This tutorial explains how to use the `TimeSeriesPreprocessor` class for preparing time series data for forecasting models. The TimeSeriesPreprocessor implements preprocessing techniques to enhance the quality of your time series data.

## Table of Contents

- [TimeSeriesPreprocessor Tutorial](#timeseriespreprocessor-tutorial)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Key Features](#key-features)
  - [Basic Usage](#basic-usage)
  - [Advanced Preprocessing Techniques](#advanced-preprocessing-techniques)
    - [Outlier Removal](#outlier-removal)
    - [Missing Value Imputation](#missing-value-imputation)
    - [Empirical Wavelet Transform (EWT)](#empirical-wavelet-transform-ewt)
    - [Detrending](#detrending)
    - [Normalization](#normalization)
    - [Differencing](#differencing)
    - [Adaptive Filtering](#adaptive-filtering)
    - [Log Transformation](#log-transformation)
    - [Time Features Generation](#time-features-generation)
  - [Complete Example](#complete-example)
  - [Inverse Transformation](#inverse-transformation)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues and Solutions](#common-issues-and-solutions)
    - [Tips for Better Preprocessing](#tips-for-better-preprocessing)

## Overview

The `TimeSeriesPreprocessor` class provides a comprehensive set of preprocessing techniques for time series data. Proper preprocessing is crucial for time series forecasting as it can significantly improve model performance by removing noise, handling outliers, addressing seasonality, and transforming the data into a format that models can learn from effectively.

## Installation

Before using the `TimeSeriesPreprocessor` class, ensure you have the necessary dependencies installed:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
pip install pyod  # For outlier detection methods
pip install ewtpy  # For Empirical Wavelet Transform
```

For some advanced imputation methods, you might need additional packages:

```bash
pip install fancyimpute  # For iterative imputation
```

## Key Features

- **Outlier Removal**: Multiple methods for detecting and handling outliers in time series data
- **Missing Value Imputation**: Various strategies for filling in missing values
- **Empirical Wavelet Transform (EWT)**: Signal decomposition to separate trend, seasonality, and noise
- **Detrending**: Remove trend components for better forecasting of stationary components
- **Normalization**: Standardize data to improve model training
- **Differencing**: Make time series stationary by computing differences between consecutive observations
- **Adaptive Filtering**: Remove noise while preserving important signal characteristics
- **Log Transformation**: Handle exponential growth and stabilize variance
- **Time Features Generation**: Extract calendar features from timestamps

## Basic Usage

Here's a simple example of how to use the `TimeSeriesPreprocessor` class:

```python
import numpy as np
import pandas as pd
from your_module import TimeSeriesPreprocessor

# Sample data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
data = data.reshape(-1, 1)  # Reshape for single variable

# Create the preprocessor
preprocessor = TimeSeriesPreprocessor(
    normalize=True,
    differencing=False,
    detrend=True,
    apply_ewt=True,
    window_size=24,
    horizon=10,
    remove_outliers=True
)

# Fit and transform the data
X, y, processed_data = preprocessor.fit_transform(data, time_stamps=dates)

print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

## Advanced Preprocessing Techniques

### Outlier Removal

The preprocessor offers several methods for outlier detection and removal:

```python
preprocessor = TimeSeriesPreprocessor(
    remove_outliers=True,
    outlier_threshold=0.05,  # Threshold for outlier detection
    outlier_method="iqr"     # Method: "iqr", "zscore", "mad", "quantile", "isolation_forest", "lof", "ecod"
)
```

Available outlier detection methods:

- **IQR (Interquartile Range)**: Uses the difference between 75th and 25th percentiles to identify outliers
- **Z-Score**: Identifies points that are a certain number of standard deviations away from the mean
- **MAD (Median Absolute Deviation)**: Robust method using median instead of mean
- **Quantile**: Clips values below and above specified quantiles
- **Isolation Forest**: Uses isolation mechanism in a tree-based model
- **LOF (Local Outlier Factor)**: Identifies outliers based on local density deviation
- **ECOD**: Empirical Cumulative Outlier Detection, a parameter-free method

### Missing Value Imputation

For handling missing values, several imputation strategies are available:

```python
preprocessor = TimeSeriesPreprocessor(
    impute_method="auto"  # Method: "auto", "interpolate", "mean", "ffill", "bfill", "knn", "iterative"
)
```

Available imputation methods:

- **Auto**: Automatically selects the best method based on missing percentage and feature correlations
- **Interpolate**: Linear interpolation between valid observations
- **Mean**: Replace missing values with feature means
- **FFill**: Forward fill (propagate last valid observation forward)
- **BFill**: Backward fill (use next valid observation)
- **KNN**: K-Nearest Neighbors imputation
- **Iterative**: Multiple imputation by chained equations (requires fancyimpute)

### Empirical Wavelet Transform (EWT)

EWT decomposes a signal into different frequency components (Intrinsic Mode Functions or IMFs):

```python
preprocessor = TimeSeriesPreprocessor(
    apply_ewt=True,
    ewt_bands=5,    # Number of frequency bands
    trend_imf_idx=0  # Index of the IMF containing trend (usually 0)
)
```

The EWT decomposition is useful for:
- Separating trend from seasonal and residual components
- Isolating different seasonal patterns with different frequencies
- Removing noise by reconstructing the signal without certain components

### Detrending

Remove trend components from time series:

```python
preprocessor = TimeSeriesPreprocessor(
    detrend=True,
    apply_ewt=True,   # Required for EWT-based detrending
    trend_imf_idx=0   # IMF index containing the trend component
)
```

Detrending helps create more stationary time series, which are typically easier for forecasting models to learn.

### Normalization

Standardize the data to have zero mean and unit variance:

```python
preprocessor = TimeSeriesPreprocessor(
    normalize=True  # Apply StandardScaler to the data
)
```

Normalization is particularly important for neural network-based forecasting models.

### Differencing

Apply differencing to make time series stationary:

```python
preprocessor = TimeSeriesPreprocessor(
    differencing=True  # Compute first-order differences
)
```

Differencing is a classic technique for handling non-stationary time series.

### Adaptive Filtering

Apply Savitzky-Golay filter to smooth the data while preserving important features:

```python
preprocessor = TimeSeriesPreprocessor(
    apply_filter=True,
    filter_window=5,       # Window size for filtering
    filter_polyorder=2     # Polynomial order for Savitzky-Golay filter
)
```

Adaptive filtering helps remove noise while maintaining the overall structure of the signal.

### Log Transformation

Apply logarithmic transformation for data with exponential growth or skewed distribution:

```python
preprocessor = TimeSeriesPreprocessor(
    log_transform=True  # Apply natural logarithm transformation
)
```

The preprocessor automatically handles negative values by adding an offset.

### Time Features Generation

Extract calendar features from timestamps:

```python
import pandas as pd
from your_module import TimeSeriesPreprocessor

# Create timestamps
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.random.randn(100, 1)

preprocessor = TimeSeriesPreprocessor(window_size=24, horizon=10)
X, y, processed = preprocessor.fit_transform(data, time_stamps=dates)
```

Time features include:
- Month (normalized to 0-1)
- Day of month (normalized to 0-1)
- Day of week (normalized to 0-1)
- Hour of day for hourly data (normalized to 0-1)

## Complete Example

Here's a comprehensive example that uses most of the features:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from your_module import TimeSeriesPreprocessor

# Generate synthetic time series data
np.random.seed(42)
n_samples = 200
timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

# Create a time series with trend, seasonality, and noise
t = np.linspace(0, 4*np.pi, n_samples)
trend = 0.1 * t
seasonality1 = 2 * np.sin(t)  # Daily pattern
seasonality2 = 1 * np.sin(t/24)  # Weekly pattern
noise = np.random.normal(0, 0.5, n_samples)

# Combine components
data = (trend + seasonality1 + seasonality2 + noise).reshape(-1, 1)

# Add some outliers
outlier_indices = np.random.choice(n_samples, 10, replace=False)
data[outlier_indices] = data[outlier_indices] + 5 * np.random.randn(10, 1)

# Add some missing values
missing_indices = np.random.choice(n_samples, 15, replace=False)
data[missing_indices] = np.nan

# Create preprocessor with various techniques enabled
preprocessor = TimeSeriesPreprocessor(
    normalize=True,
    differencing=False,
    detrend=True,
    apply_ewt=True,
    window_size=24,
    horizon=12,
    remove_outliers=True,
    outlier_threshold=0.05,
    outlier_method="iqr",
    impute_method="auto",
    ewt_bands=5,
    trend_imf_idx=0,
    log_transform=False,
    filter_window=5,
    filter_polyorder=2,
    apply_filter=True
)

# Fit and transform the data
X, y, processed_data = preprocessor.fit_transform(data, time_stamps=timestamps)

# Visualize the results
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.title('Original Data with Outliers and Missing Values')
plt.plot(data)

plt.subplot(3, 1, 2)
plt.title('Processed Data')
plt.plot(processed_data)

plt.subplot(3, 1, 3)
plt.title('EWT Components')
ewt_components = preprocessor.get_ewt_components()
if ewt_components:
    for i, imf in enumerate(ewt_components[0].T):
        plt.plot(imf, label=f'IMF {i}')
    plt.legend()

plt.tight_layout()
plt.show()

print(f"Input sequence shape: {X.shape}")
print(f"Target sequence shape: {y.shape}")
```

## Inverse Transformation

To convert model predictions back to the original scale:

```python
# Assuming you have fitted the preprocessor and trained a model
model_predictions = model.predict(X_test)  # Shape [samples, horizon, features]

# Convert back to original scale
original_scale_predictions = preprocessor.inverse_transform(model_predictions)
```

The `inverse_transform` method applies the inverse of all transformations in the correct order:
1. Inverse normalization (if `normalize=True`)
2. Inverse differencing (if `differencing=True`)
3. Add back trend (if `detrend=True`)
4. Inverse log transform (if `log_transform=True`)

## Troubleshooting

### Common Issues and Solutions

1. **Missing Dependencies**:
   If you encounter errors related to missing modules, ensure you have installed all required packages. Some methods, like `"iterative"` imputation, require additional libraries.

2. **Failed Inverse Transform**:
   Ensure that you call `fit_transform` before using `inverse_transform`. The preprocessor needs to store certain statistics from the training data to correctly revert the transformations.

3. **Memory Issues with Large Datasets**:
   For large time series datasets, you might encounter memory issues, especially with methods like EWT. Consider processing data in chunks or using less memory-intensive methods.

4. **Unexpected NaN Values**:
   If you see NaN values in the processed data, check your imputation method and ensure it's suitable for your data. Some methods might not handle certain patterns of missing values well.

5. **Poor Forecasting Results**:
   If your forecasting model performs poorly after preprocessing, try different combinations of preprocessing techniques. Not all techniques are appropriate for all types of time series.

### Tips for Better Preprocessing

1. **Visualize Before and After**: Always visualize your data before and after preprocessing to understand the impact of each transformation.

2. **Start Simple**: Begin with basic preprocessing (e.g., normalization, outlier removal) and progressively add more complex techniques if needed.

3. **Domain Knowledge**: Use domain knowledge to select appropriate preprocessing techniques. For example, financial time series often benefit from log transformations.

4. **Cross-Validation**: Use time series cross-validation to evaluate the impact of different preprocessing configurations on your forecasting model.

5. **Feature Importance**: After preprocessing, analyze feature importance to understand which aspects of your preprocessed data are most useful for forecasting.