# TimeSeriesPreprocessor Guide

`TimeSeriesPreprocessor` provides a configurable preprocessing pipeline for multivariate time series, including optional auto-configuration (`self_tune=True`).

Related docs:
- [Custom Blocks](custom_blocks.md)
- [Transformer](transformer.md)

---

## Import

```python
from foreblocks import TimeSeriesPreprocessor
```

---

## Quick start

```python
import numpy as np
import pandas as pd
from foreblocks import TimeSeriesPreprocessor

data = np.random.randn(1200, 4)  # [T, F]
timestamps = pd.date_range("2025-01-01", periods=len(data), freq="h")

pre = TimeSeriesPreprocessor(
    self_tune=True,
    window_size=96,
    horizon=24,
    generate_time_features=True,
)

X, y, processed, time_feats = pre.fit_transform(data, time_stamps=timestamps)
```

Return values from `fit_transform(...)`:
- `X`: input windows
- `y`: forecast targets
- `processed`: transformed full series
- `time_feats`: optional time features (or `None`)

---

## Core constructor options

### Windowing
- `window_size` (default `24`)
- `horizon` (default `10`)

### Main transforms
- `normalize` (`True`)
- `differencing` (`False`)
- `detrend` (`False`)
- `log_transform` (`False`)
- `apply_filter` (`False`)
- `apply_ewt` (`False`)

### Outliers and imputation
- `remove_outliers` (`False`)
- `outlier_method` (default `"iqr"`)
- `outlier_threshold` (default `0.05`)
- `apply_imputation` (`False`)
- `impute_method` (default `"auto"`)

### Auto-configuration / UX
- `self_tune` (`False`)
- `generate_time_features` (`False`)
- `plot` (`False`)
- `verbose` (`False`)

---

## Typical workflow

```python
# Fit on train data only
X_train, y_train, train_processed, tf_train = pre.fit_transform(train_data, time_stamps=train_ts)

# Transform validation/test using fitted state
X_val = pre.transform(val_data, time_stamps=val_ts)
X_test = pre.transform(test_data, time_stamps=test_ts)

# Invert model outputs back to data scale
pred_real = pre.inverse_transform(pred_scaled)
```

---

## Manual configuration example

```python
pre = TimeSeriesPreprocessor(
    normalize=True,
    scaling_method="robust",   # when heavy tails are expected
    remove_outliers=True,
    outlier_method="ecod",
    apply_imputation=True,
    impute_method="knn",
    apply_filter=True,
    filter_method="savgol",
    detrend=True,
    generate_time_features=True,
)
```

---

## Operational notes

- Input must be 2D: `[T, F]`.
- Call `fit_transform` before `transform`; otherwise `transform` raises.
- `transform` returns windows `X` (not `(X, y, processed)`).
- `inverse_transform` accepts shape `[N, D]` or `[N, H, D]`.
- Auto-configuration prints detected settings when enabled.

---

## When to enable `self_tune`

- Use `self_tune=True` for heterogeneous datasets or quick baselines.
- Use manual configuration for controlled ablations/reproducibility studies.
