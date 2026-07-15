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

## Directory Structure

```
foreblocks/
├── anomaly/                      # Anomaly detection models and utilities
│   ├── models/                   # Anomaly detection model implementations
│   │   ├── base.py              # Base anomaly detector class
│   │   ├── reconstruction.py    # Reconstruction-based anomaly detectors
│   │   ├── forecasting.py       # Forecasting-based anomaly detectors
│   │   ├── representation.py    # Representation-based anomaly detectors
│   │   ├── tranad.py            # TranAD (Transformer-based) model
│   │   ├── omni_anomaly.py      # OmniAnomaly model
│   │   ├── dagmm.py             # DAGMM (Deep Autoencoding Gaussian Mixture) model
│   │   ├── anomaly_transformer.py # AnomalyTransformer model
│   │   ├── diffusion.py         # Diffusion-based anomaly detection
│   │   ├── frequency.py         # Frequency-domain anomaly detection
│   │   ├── patch_tst.py         # PatchTST-based anomaly detection
│   │   └── state_space.py       # State-space model anomaly detection
│   ├── detector.py              # Anomaly detector interface
│   ├── online.py                # Online anomaly detection utilities
│   ├── calibration.py           # Anomaly score calibration
│   ├── windows.py               # Windowing utilities for anomaly detection
│   └── modes.py                 # Anomaly detection modes and configurations
│
├── core/                         # Core functionality
│   ├── model.py                 # Core model base classes
│   ├── att.py                   # Attention utilities
│   ├── sampling.py              # Sampling utilities
│   ├── extend.py                # Extension utilities
│   ├── __init__.py
│   ├── evaluation/              # Evaluation and benchmarking
│   │   ├── benchmark.py         # Benchmarking utilities
│   │   ├── model_evaluator.py   # Model evaluation interface
│   │   └── __init__.py
│   └── training/                # Training utilities
│       ├── trainer.py           # Main trainer implementation
│       ├── training_loop.py     # Training loop implementation
│       ├── losses.py            # Loss functions
│       ├── history.py           # Training history tracking
│       ├── logging.py           # Training logging utilities
│       ├── visualization.py     # Training visualization
│       ├── quantization.py      # Model quantization
│       ├── conformal.py         # Conformal prediction utilities
│       ├── conformal_trainer.py # Conformal training implementation
│       ├── batch_io.py          # Batch I/O utilities
│       ├── nas.py               # Neural Architecture Search utilities
│       └── llrd.py              # Layer-wise learning rate decay
│
├── data/                         # Data loading and preprocessing utilities
│
├── experimental/                 # Experimental features and research code
│
├── layers/                       # Neural network layer implementations
│   ├── embeddings/              # Embedding layers
│   │   └── rotary.py            # Rotary Positional Encoding (RoPE)
│   ├── graph/                   # Graph neural network layers
│   │   ├── layers/              # Graph layer implementations
│   │   │   └── message_passing.py # Message passing layers
│   │   └── spatiotemporal/      # Spatiotemporal graph layers
│   └── norms/                   # Normalization layers
│
├── models/                       # Model implementations
│   ├── kan/                     # Kolmogorov-Arnold Network models
│   │   ├── __init__.py
│   │   ├── backbone.py          # KAN backbone implementation
│   │   ├── model.py             # KAN model implementation
│   │   ├── router.py            # KAN router implementation
│   │   └── poly/                # Polynomial basis functions
│   │       ├── types.py         # Polynomial types
│   │       ├── utils.py         # Polynomial utilities
│   │       ├── hahn.py          # Hahn polynomials
│   │       ├── chebyshev.py     # Chebyshev polynomials
│   │       ├── jacobi.py        # Jacobi polynomials
│   │       ├── gegenbauer.py    # Gegenbauer polynomials
│   │       ├── laguerre.py      # Laguerre polynomials
│   │       ├── hermite.py       # Hermite polynomials
│   │       ├── wavelet.py       # Wavelet polynomials
│   │       └── fourier.py       # Fourier polynomials
│   ├── graph_forecasting.py     # Graph forecasting models
│   └── ...                      # Other model implementations
│
├── modules/                      # Reusable neural network modules and components
│
├── ops/                          # Custom operations and kernels
│   ├── attention/               # Attention operations
│   │   ├── fused_rope.py        # Fused Rotary Positional Encoding
│   │   ├── paged_decode.py      # Paged attention decode
│   │   ├── chunked_causal_linear_attention.py # Chunked causal linear attention
│   │   ├── fused_norm_gate.py   # Fused norm-gate operations
│   │   ├── fla_backend.py       # FLA (Flash Linear Attention) backend
│   │   ├── fla_delta_rule.py    # FLA delta rule
│   │   ├── fla_gated_delta_rule.py # FLA gated delta rule
│   │   ├── fla_gdn2.py          # FLA GDN2
│   │   ├── gla.py               # FLA GLA
│   │   ├── kda.py               # FLA KDA
│   │   └── linear_attention.py  # FLA linear attention
│   ├── kernels/                 # Low-level kernels
│   │   ├── layer_norm.py        # Layer normalization kernel
│   │   ├── rms_norm.py          # RMS normalization kernel
│   │   ├── grouped_gemm.py      # Grouped GEMM kernel
│   │   ├── swiglu.py            # SwiGLU activation kernel
│   │   ├── softmax.py           # Softmax kernel (Triton)
│   │   └── gelu.py              # GELU kernel (Triton)
│   ├── mamba/                   # Mamba/state-space model operations
│   │   ├── causal_conv1d.py     # Causal convolution 1D
│   │   ├── fused_dt.py          # Fused delta-time operations
│   │   ├── mamba2_combined.py   # Mamba2 combined operations
│   │   ├── ssd.py               # State Space Duality
│   │   └── triton_ops.py        # Triton operations for Mamba
│   ├── graph/                   # Graph operations
│   │   └── message_passing.py   # Graph message passing operations
│   └── raven/                   # Raven operations
│
├── sequence/                     # Sequence modeling utilities and components
│
├── ts_handler/                   # Time-series preprocessing and filtering pipeline
│   ├── preprocessing.py         # TimeSeriesHandler main class
│   ├── utils.py                 # Time-series utilities
│   ├── diagnostics.py           # Time-series diagnostics
│   ├── plotting.py              # Time-series plotting utilities
│   ├── transforms.py            # Time-series transformations
│   ├── time_features.py         # Time feature extraction
│   ├── windowing.py             # Windowing utilities
│   ├── pipeline.py              # Processing pipeline
│   ├── auto_configure.py        # Auto-configuration utilities
│   ├── filters/                 # Signal processing filters
│   │   ├── utils.py             # Filter utilities
│   │   ├── savgol.py            # Savitzky-Golay filter
│   │   ├── kalman.py            # Kalman filter (pure NumPy implementation)
│   │   ├── lowess.py            # LOESS/LOWESS filter
│   │   ├── wiener.py            # Wiener filter
│   │   ├── emd.py               # Empirical Mode Decomposition
│   │   ├── ssa.py               # Singular Spectrum Analysis
│   │   └── stl.py               # STL decomposition
│   └── auto_filter/             # Automatic filter selection and tuning
│       ├── __init__.py
│       ├── metrics.py           # Filter scoring metrics
│       ├── runner.py            # Auto-filter execution
│       ├── tuning.py            # Optuna-based tuning (tune_weights, tune_filter)
│       ├── heuristics.py        # Heuristic weight suggestion
│       ├── visualization.py     # Filter result visualization
│       ├── registry.py          # Filter registry
│       └── filters/             # Filter implementations
│           ├── classical.py     # Classical filters
│           ├── wavelet.py       # Wavelet filters
│           ├── tv.py            # Total variation denoising
│           ├── lowess.py        # LOESS filters
│           ├── smoothers.py     # Smoothing filters
│           ├── penalized.py     # Penalized filters
│           ├── ssa.py           # SSA filters
│           ├── kalman_rts.py    # Kalman RTS smoother
│           ├── bilateral.py     # Bilateral filter
│           ├── decomposition.py # Decomposition filters
│           └── deep.py          # Deep learning filters
│
├── ui/                           # User interface components and studio server
│   ├── auto_spec.py             # Auto specification utilities
│   ├── discovery.py             # Model discovery utilities
│   ├── node_spec.py             # Node specification utilities
│   └── __init__.py
│
├── third_party/                  # Third-party integrations and utilities
│   ├── flash_softpick_attn.py   # Flash softpick attention
│   └── vsgd.py                  # Variational SGD utilities
│
├── config.py                     # Library configuration
├── studio_server.py              # Studio server implementation
├── __init__.py                   # Package initialization
└── README.md                     # This file

## Core API

### Time-Series Handler

```python
from foreblocks.ts_handler.preprocessing import TimeSeriesHandler
from foreblocks.ts_handler.auto_filter import (
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
from foreblocks.anomaly.models import (
    ReconstructionAnomalyDetector,
    ForecastingAnomalyDetector,
    TranAD,
    OmniAnomaly,
    DAGMM,
    AnomalyTransformer,
    PatchTSTAnomalyDetector,
)
```

### Kolmogorov-Arnold Networks

```python
from foreblocks.models.kan import (
    KANBackbone,
    KANModel,
    KANRouter,
)
from foreblocks.models.kan.poly import (
    ChebyshevPoly,
    JacobiPoly,
    FourierPoly,
    # ... other polynomial bases
)
```

### Custom Operations & Kernels

```python
from foreblocks.ops.attention import fused_rope, paged_decode
from foreblocks.ops.kernels import (
    SoftmaxTritonFunction,
    GeluTritonFunction,
    layer_norm,
    rms_norm,
    swiglu,
)
from foreblocks.ops.mamba import (
    causal_conv1d,
    mamba2_combined,
    ssd,
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
- Statsmodels (for statistical tests)
