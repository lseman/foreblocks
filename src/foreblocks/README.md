# Foreblocks - Comprehensive Time Series Forecasting and Graph Modeling Library

Foreblocks is a comprehensive, production-ready library for time-series forecasting, anomaly detection, graph-based modeling, and neural network operations. It features custom Triton/CUDA kernels, Kolmogorov-Arnold Networks (KANs), advanced time-series preprocessing, and state-space models.

## Overview

Foreblocks provides:

- **Custom Operations & Kernels**: Triton and CUDA implementations for attention, normalization, Mamba/state-space models, and kernel operations
- **Time-Series Handler**: Comprehensive preprocessing, filtering, imputation, and feature engineering pipeline
- **Anomaly Detection**: Multiple anomaly detection models including TranAD, OmniAnomaly, DAGMM, AnomalyTransformer, PatchTST, and diffusion-based models
- **Neural Network Layers**: Embeddings (including Rotary Positional Encoding), graph layers, and normalization layers
- **Model Architectures**: Kolmogorov-Arnold Networks (KANs), graph forecasting models, and sequence models
- **Core Training & Evaluation**: Training loops, loss functions, conformal prediction, quantization, and NAS utilities
- **Studio UI**: Web-based studio server for model discovery and visualization

## Package organization

Start in `models/` for assembled forecasters, `nn/` for reusable PyTorch layers,
and `training/` for fitting and checkpointing. The contributor map describes
the data, processing, feature, evaluation, and acceleration packages and their
relationships. [Repository map](../../docs/reference/repository-map.md)

## Filter organization

`processing/filters/` owns numerical filtering algorithms. Private modules
`_ssa.py`, `_lowess.py`, and `_kalman.py` hold reconstruction and smoothing;
the public filter modules handle NumPy arrays, per-column filtering, and NaNs.
`processing/auto_filter/filters/` adapts these algorithms to pandas Series and
the automatic filter registry. Defaults and failure policies belong to the
adapters so their existing interfaces remain stable.

Kalman retains two distinct models: the array filter uses a scalar local-level
model with iterative noise estimation; the automatic filter uses a two-state
local-linear-trend model with RTS smoothing. They share a numerical home, not
interchangeable equations or defaults.

Basic outlier processing does not load TranAD. Its detector is imported when
`method="tranad"` is selected; legacy `outlier.TranAD` and
`outlier.TranADDetector` attributes remain available through lazy imports.

## Core API

### Time-Series Handler

```python
from foreblocks.processing import TimeSeriesHandler
from foreblocks.processing.auto_filter import (
    auto_filter,
    suggest_weights,
    tune_weights,
    tune_filter,
    ScoringWeights,
    TuneFilterResult,
)
```

### Anomaly Detection

```python
from foreblocks.models.anomaly import (
    ForeblocksAnomalyDetector,  # generic wrapper: reconstruction/forecasting/representation/hybrid modes
    TranAD,
    OmniAnomaly,
    DAGMM,
    AnomalyTransformer,
)
```

### Kolmogorov-Arnold Networks

```python
from foreblocks.models.kan import (
    Backbone,
    KANModel,
    TokenRouter,
)
from foreblocks.models.kan.poly import (
    ChebyshevPolynomials,
    JacobiPolynomials,
    # ... other polynomial bases
)
```

### Custom Operations & Kernels

```python
from foreblocks.ops.attention import triton_apply_rope, triton_paged_decode
from foreblocks.kernels.normalization.rms_norm import rms_norm
from foreblocks.kernels.activations.swiglu import swiglu_gate
from foreblocks.ops.mamba import (
    causal_depthwise_conv1d,
    mamba2_split_conv1d_scan_combined,
    chunked_ssd_forward,
)
```

## Key Features

1. **Triton/CUDA Kernels**: Custom implementations for softmax, GELU, layer norm, RMS norm, SwiGLU, and attention operations
2. **Mamba/State-Space Models**: Full Mamba2 implementation with causal convolutions and state space duality
3. **Kolmogorov-Arnold Networks**: Complete KAN implementation with multiple polynomial basis functions
4. **Automatic Filter Selection**: Optuna-based auto-tuning of filter selection weights and filter parameters
5. **Pure NumPy Kalman Filter**: Independent Kalman filter and RTS smoother implementation without pykalman dependency
6. **Graph Forecasting**: Spatiotemporal graph neural network models for forecasting
7. **Comprehensive Anomaly Detection**: Multiple state-of-the-art anomaly detection models

## Dependencies

- PyTorch
- Triton (for custom kernels)
- NumPy, Pandas, SciPy
- Optuna (for auto-filter tuning)
- Matplotlib (for visualization)
- Numba (optional LOWESS acceleration; NumPy fallback available)
- Statsmodels (for ADF, STL, and EWT ARIMA selection)

LOWESS smoothing, ACF, Yule–Walker PACF, and Ljung–Box scoring use native
NumPy/SciPy implementations. These preprocessing functions do not require
Statsmodels. LOWESS preserves missing positions and supports robust iterations
and `delta` interpolation in both array and pandas filter APIs.
