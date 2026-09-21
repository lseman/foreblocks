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
├── kernels/                      # Triton/CUDA accelerated kernels, no nn.Module surface
│   ├── attention/                # Fused RoPE, paged/chunked attention kernels
│   ├── mamba/                    # Causal conv1d, mamba2_combined, SSD, Triton ops
│   ├── linear/                   # Grouped GEMM and related linear-algebra kernels
│   ├── normalization/            # Layer norm, RMS norm kernels
│   ├── activations/              # Softmax, GELU, SwiGLU kernels
│   ├── elementwise/               # Elementwise ops
│   ├── graph/                    # Graph message-passing kernels
│   ├── experimental/              # Vendored attention_kernels sub-project (own setup.py)
│   └── patching.py
│
├── ops/                           # Tensor ops and execution dispatch on top of kernels/
│   ├── attention/                # fused_rope, paged_decode, chunked linear attention
│   └── mamba/                    # ssd, causal_conv1d, mamba2_combined
│
├── integrations/                  # Optional external backends
│   ├── fla/                      # flash-linear-attention adapters (delta rule, GLA, RAVEN, …)
│   └── softpick.py
│
├── nn/                            # Reusable nn.Module primitives
│   ├── attention/                 # Attention config, variants, KV cache, layer.py (AttentionLayer)
│   ├── transformer/                # Encoder/decoder stack, patching, fusions, tuner
│   ├── blocks/                    # Research blocks: TCN, ODE, Fourier, wavelets, xLSTM, recurrent.py (LSTM/GRU enc-dec)
│   ├── heads/                     # Head composition: engine/composer.py (HeadComposer), core/types.py (HeadSpec), blocks/
│   ├── moe/                       # Experts, routers, feed-forward MoE
│   ├── routing/                   # Gate/skip routing (GateSkip, MoD)
│   ├── sequence/                  # mamba/, raven/ alternative sequence backbones
│   ├── normalization/             # Group/layer/RMS/temporal/RevIN norm modules
│   ├── embeddings/                # Rotary, ALiBi, positional, time embeddings
│   ├── residual/                  # Residual connection utilities
│   └── graph/                     # Graph neural network layers
│
├── models/                        # Assembled models + composition APIs
│   ├── forecasting.py             # ForecastingModel, BaseHead
│   ├── graph.py                   # GraphForecastingModel
│   ├── config.py                  # ModelConfig
│   ├── distillation.py            # DistilledForecastingModel, QuantizedForecastingModel
│   ├── sequence.py                 # Sequence-model composition
│   ├── baselines/                 # Named end-to-end models (NBEATS, Informer, Autoformer, TimesNet, …)
│   ├── kan/                       # Kolmogorov-Arnold Networks: backbone.py, model.py, router.py, poly/ (Chebyshev, Jacobi, …)
│   └── anomaly/                   # Anomaly-detection applications layer
│       ├── models/                # TranAD, OmniAnomaly, DAGMM, AnomalyTransformer, PatchTST, diffusion, …
│       ├── detector.py
│       ├── online.py
│       ├── calibration.py
│       └── windows.py
│
├── training/                      # Training orchestration
│   ├── trainer.py                 # Trainer
│   ├── config.py                  # TrainingConfig
│   ├── sampling.py                # ScheduledSampling
│   ├── losses.py
│   ├── conformal/                 # Conformal prediction
│   ├── optimization/              # nas.py, llrd.py (layer-wise LR decay)
│   ├── execution/
│   ├── state/
│   └── telemetry/
│
├── evaluation/                    # Evaluation and benchmarking
│   ├── model_evaluator.py         # ModelEvaluator
│   ├── benchmark.py
│   └── visualization.py
│
├── quantization/                  # FakeQuantize, DynamicQuantizedLinear
│
├── data/                          # Dataset and dataloader helpers
│   ├── dataset.py                 # TimeSeriesDataset, create_dataloaders
│   └── csv.py
│
├── processing/                    # Time-series preprocessing and filtering pipeline (TimeSeriesHandler)
│   ├── core/
│   ├── transforms/                # Time-series transformations, time features, windowing
│   ├── filters/                   # Savitzky-Golay, Kalman (pure NumPy), LOESS, Wiener, EMD, SSA, STL
│   ├── auto_filter/                # Optuna-based auto-tuning (auto_filter, tune_weights, tune_filter)
│   └── tools/
│
├── studio/                        # Node/spec auto-discovery backend for apps/webui
│   ├── auto_spec.py
│   ├── discovery.py
│   └── node_spec.py
│
├── studio_server.py               # Local HTTP server for the built Studio frontend
├── __init__.py                    # Package initialization (lazy public exports)
└── README.md                      # This file

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
