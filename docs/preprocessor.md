# TimeSeriesPreprocessor Tutorial

This tutorial explains how to use the `TimeSeriesPreprocessor` class for preparing time series data for forecasting models. The TimeSeriesPreprocessor implements advanced preprocessing techniques to enhance the quality of your time series data, with intelligent auto-configuration capabilities.

## Table of Contents

- [TimeSeriesPreprocessor Tutorial](#timeseriespreprocessor-tutorial)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Key Features](#key-features)
  - [Basic Usage](#basic-usage)
  - [Automatic Configuration](#automatic-configuration)
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
  - [Integration with TimeSeriesSeq2Seq](#integration-with-timeseriesseq2seq)
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
pip install PyEWT  # For Empirical Wavelet Transform
```

For some advanced imputation methods, you might need additional packages:

```bash
pip install fancyimpute  # For iterative imputation
pip install statsmodels  # For statistical tests and seasonal decomposition
```

## Key Features

- **Automatic Configuration**: Intelligent detection of appropriate preprocessing methods based on data properties
- **Outlier Removal**: Multiple methods for detecting and handling outliers in time series data
- **Missing Value Imputation**: Various strategies for filling in missing values
- **Empirical Wavelet Transform (EWT)**: Signal decomposition to separate trend, seasonality, and noise
- **Detrending**: Remove trend components for better forecasting of stationary components
- **Normalization**: Standardize data to improve model training
- **Differencing**: Make time series stationary by computing differences between consecutive observations
- **Adaptive Filtering**: Remove noise while preserving important signal characteristics
- **Log Transformation**: Handle exponential growth and stabilize variance
- **Time Features Generation**: Extract calendar features from timestamps
- **Robust Implementation**: Handle edge cases gracefully with appropriate fallbacks

## Basic Usage

Here's a simple example of how to use the `TimeSeriesPreprocessor` class:

```python
import numpy as np
import pandas as pd
from foreblocks.preprocessing import TimeSeriesPreprocessor

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

## Automatic Configuration

One of the most powerful features of the `TimeSeriesPreprocessor` is its ability to automatically configure preprocessing techniques based on data characteristics:

```python
import numpy as np
import pandas as pd
from foreblocks.preprocessing import TimeSeriesPreprocessor

# Generate data with trend, seasonality, outliers, and missing values
n_samples = 200
data = generate_complex_time_series(n_samples)  # Your function to generate data

# Create preprocessor with self-tuning enabled
preprocessor = TimeSeriesPreprocessor(self_tune=True)

# Fit and transform data
X, y, processed_data = preprocessor.fit_transform(data)

# The preprocessor automatically determined the best parameters:
print(f"Detected outlier method: {preprocessor.outlier_method}")
print(f"Detected need for detrending: {preprocessor.detrend}")
print(f"Detected need for log transform: {preprocessor.log_transform}")
print(f"Selected imputation method: {preprocessor.impute_method}")
print(f"Configured EWT bands: {preprocessor.ewt_bands}")
```

During the auto-configuration process, the preprocessor runs several statistical tests and analyses:

1. **Stationarity Testing**: Using Augmented Dickey-Fuller (ADF) test to determine if detrending is needed
2. **Skewness Analysis**: Calculating skewness to decide if log transformation would be beneficial
3. **Missing Value Analysis**: Evaluating the pattern and percentage of missing values
4. **Signal-to-Noise Ratio**: Estimating SNR to decide if filtering should be applied
5. **Entropy-Based EWT Configuration**: Using entropy to determine the optimal number of EWT bands
6. **Outlier Method Selection**: Choosing the most appropriate outlier detection method based on data size and distribution

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

The refactored implementation handles edge cases gracefully, such as all-NaN columns or insufficient data for certain methods.

### Missing Value Imputation

For handling missing values, several imputation strategies are available:

```python
preprocessor = TimeSeriesPreprocessor(
    apply_imputation=True,
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

The implementation now includes more robust handling of edge cases, such as columns with all missing values or very sparse data.

### Empirical Wavelet Transform (EWT)

EWT decomposes a signal into different frequency components (Intrinsic Mode Functions or IMFs):

```python
preprocessor = TimeSeriesPreprocessor(
    apply_ewt=True,
    ewt_bands=5,     # Number of frequency bands
    trend_imf_idx=0  # Index of the IMF containing trend (usually 0)
)
```

The EWT decomposition is useful for:
- Separating trend from seasonal and residual components
- Isolating different seasonal patterns with different frequencies
- Removing noise by reconstructing the signal without certain components

The refactored implementation includes better error handling for when the PyEWT library is not available, falling back to simpler methods.

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
    normalize=True,           # Apply normalization
    use_robust_scaler=False   # Set to True to use RobustScaler instead of StandardScaler
)
```

The refactored implementation now supports both StandardScaler and RobustScaler, with the latter being more resistant to outliers. The auto-configuration will choose robust scaling when significant outliers are detected.

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

The improved implementation ensures that filter parameters are always valid, automatically adjusting the window size to be compatible with the polynomial order.

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
from foreblocks.preprocessing import TimeSeriesPreprocessor

# Create timestamps
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.random.randn(100, 1)

preprocessor = TimeSeriesPreprocessor(
    window_size=24, 
    horizon=10,
    generate_time_features=True
)
X, y, processed = preprocessor.fit_transform(data, time_stamps=dates)

# The time features are added as additional columns to the processed data
print(f"Processed data shape with time features: {processed.shape}")
```

Time features include:
- Month (normalized to 0-1)
- Day of month (normalized to 0-1)
- Day of week (normalized to 0-1)
- Hour of day for hourly data (normalized to 0-1)

## Integration with TimeSeriesSeq2Seq

The `TimeSeriesPreprocessor` is designed to work seamlessly with the `TimeSeriesSeq2Seq` class:

```python
from foreblocks import TimeSeriesSeq2Seq, ModelConfig, TrainingConfig

# Create model configuration
model_config = ModelConfig(
    model_type="lstm",
    input_size=data.shape[1],
    output_size=1,
    hidden_size=64,
    target_len=24
)

# Create training configuration
training_config = TrainingConfig(
    num_epochs=100,
    learning_rate=0.001,
    patience=10
)

# Initialize model
model = TimeSeriesSeq2Seq(
    model_config=model_config,
    training_config=training_config,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Use the built-in preprocess method that uses TimeSeriesPreprocessor internally
X_train, y_train, processed_data = model.preprocess(
    data, 
    self_tune=True,
    window_size=48,
    horizon=24
)

# Train the model with the preprocessed data
train_loader = create_dataloader(X_train, y_train)
history = model.train_model(train_loader)
```

This integration provides a seamless workflow from raw data to trained model.

## Complete Example

Here's a comprehensive example that uses most of the features:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from foreblocks import TimeSeriesSeq2Seq, ModelConfig, TrainingConfig
from foreblocks.preprocessing import TimeSeriesPreprocessor

# Generate synthetic time series data
np.random.seed(42)
n_samples = 500
timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

# Create a time series with trend, seasonality, and noise
t = np.linspace(0, 8*np.pi, n_samples)
trend = 0.1 * t
seasonality1 = 2 * np.sin(t)  # Daily pattern
seasonality2 = 1 * np.sin(t/24)  # Weekly pattern
noise = np.random.normal(0, 0.5, n_samples)

# Combine components
data = (trend + seasonality1 + seasonality2 + noise).reshape(-1, 1)

# Add some outliers
outlier_indices = np.random.choice(n_samples, 15, replace=False)
data[outlier_indices] = data[outlier_indices] + 5 * np.random.randn(15, 1)

# Add some missing values
missing_indices = np.random.choice(n_samples, 25, replace=False)
data[missing_indices] = np.nan

# Create preprocessor with auto-tuning
preprocessor = TimeSeriesPreprocessor(
    self_tune=True,
    window_size=48,
    horizon=24
)

# Fit and transform the data
X, y, processed_data = preprocessor.fit_transform(data, time_stamps=timestamps)

# Visualize the results
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.title('Original Data with Outliers and Missing Values')
plt.plot(timestamps, data)
plt.grid(True)

plt.subplot(4, 1, 2)
plt.title('Processed Data')
plt.plot(timestamps, processed_data)
plt.grid(True)

plt.subplot(4, 1, 3)
plt.title('EWT Components')
ewt_components = preprocessor.get_ewt_components()
if ewt_components and len(ewt_components) > 0:
    for i, imf in enumerate(ewt_components[0].T):
        plt.plot(timestamps, imf, label=f'IMF {i}')
    plt.legend()
    plt.grid(True)

plt.subplot(4, 1, 4)
plt.title('Trend Component')
trend_component = preprocessor.get_trend_component()
if trend_component is not None:
    plt.plot(timestamps, trend_component)
    plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Input sequence shape: {X.shape}")
print(f"Target sequence shape: {y.shape}")

# Split data into train/test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create PyTorch DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                             torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                            torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create and train model
model_config = ModelConfig(
    model_type="lstm",
    input_size=X.shape[2],
    output_size=1,
    hidden_size=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    target_len=24
)

training_config = TrainingConfig(
    num_epochs=100,
    learning_rate=0.001,
    patience=10
)

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    training_config=training_config
)

# Train the model
history = model.train_model(train_loader, val_loader=test_loader)

# Make predictions
predictions = model.predict(X_test)

# Convert predictions back to original scale
original_scale_predictions = preprocessor.inverse_transform(predictions)

# Plot original vs predicted values
plt.figure(figsize=(15, 6))
plt.title('Original vs Predicted Values')
plt.plot(y_test[:, 0, 0], label='Original')
plt.plot(original_scale_predictions[:, 0, 0], label='Predicted')
plt.legend()
plt.grid(True)
plt.show()
```

## Inverse Transformation

To convert model predictions back to the original scale:

```python
# Assuming you have fitted the preprocessor and trained a model
model_predictions = model.predict(X_test)  # Shape [samples, horizon, features]

# Convert back to original scale
original_scale_predictions = preprocessor.inverse_transform(model_predictions)
```

The improved `inverse_transform` method applies the inverse of all transformations in the correct order:
1. Inverse normalization (if `normalize=True`)
2. Inverse differencing (if `differencing=True`)
3. Add back trend (if `detrend=True`)
4. Inverse log transform (if `log_transform=True`)

The method now also handles edge cases better, such as predictions beyond the original data length, by extrapolating trend components if necessary.

## Troubleshooting

### Common Issues and Solutions

1. **Missing Dependencies**:
   If you encounter errors related to missing modules, ensure you have installed all required packages. The refactored implementation now provides better error messages and fallbacks when optional dependencies are missing.

   ```python
   # Example error message
   # → EWT skipped (PyEWT not installed)
   # → ADF test skipped (statsmodels not installed)
   ```

2. **NaN or Infinity Values in Results**:
   The new implementation has improved handling of invalid values:

   ```python
   # Better handling of NaN values in input data
   preprocessor = TimeSeriesPreprocessor(
       apply_imputation=True,
       impute_method="knn",  # Choose a robust imputation method
       use_robust_scaler=True  # Use RobustScaler for better handling of outliers
   )
   ```

3. **Memory Issues with Large Datasets**:
   For large time series datasets, you might encounter memory issues, especially with methods like EWT. The refactored implementation is more memory-efficient.

   ```python
   # For very large datasets, disable memory-intensive operations
   preprocessor = TimeSeriesPreprocessor(
       apply_ewt=False,  # Disable EWT for very large datasets
       apply_filter=True,  # Use filtering instead for smoothing
       outlier_method="iqr"  # Use simpler outlier detection
   )
   ```

4. **Poor Forecasting Results**:
   The auto-tuning feature can help identify the best preprocessing techniques for your data:

   ```python
   # Let the preprocessor determine the best configuration
   preprocessor = TimeSeriesPreprocessor(self_tune=True)
   X, y, processed = preprocessor.fit_transform(data)
   ```

### Tips for Better Preprocessing

1. **Visualize Before and After**: Always visualize your data before and after preprocessing to understand the impact of each transformation.

2. **Use Auto-Configuration**: Start with `self_tune=True` to let the preprocessor analyze your data's characteristics.

3. **Domain Knowledge**: Incorporate domain knowledge when overriding auto-configuration:

   ```python
   # Example for financial time series
   preprocessor = TimeSeriesPreprocessor(
       self_tune=True,
       log_transform=True,  # Financial data often benefits from log transform
       differencing=True    # For non-stationary financial time series
   )
   ```

4. **Cross-Validation**: Use time series cross-validation to evaluate the impact of different preprocessing configurations:

   ```python
   # Example time series cross-validation
   from sklearn.model_selection import TimeSeriesSplit
   
   tscv = TimeSeriesSplit(n_splits=5)
   for train_idx, test_idx in tscv.split(data):
       train_data, test_data = data[train_idx], data[test_idx]
       # Process and evaluate with different preprocessing configs
   ```

5. **Ensemble Preprocessing**: For critical applications, consider using multiple preprocessing configurations and ensemble the resulting models.

6. **Monitoring and Adaptation**: In production systems, monitor data distribution and periodically retune preprocessing parameters as data characteristics evolve.