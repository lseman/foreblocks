# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**foreBlocks** is a modular PyTorch library for time-series forecasting (v0.1.15). It ships two packages:
- `foreblocks`: forecasting models, training, evaluation, preprocessing, DARTS neural architecture search, MLTracker experiment tracking
- `foretools`: companion utilities — synthetic data generation, decomposition (VMD, EWT), feature engineering, benchmarking

## Development Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install with specific extras
pip install -e ".[preprocessing,darts,mltracker]"
pip install -e ".[all]"   # everything

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_foo.py

# Run a single test
pytest tests/test_foo.py::test_bar

# Linting / formatting
ruff check .
ruff format .

# Launch MLTracker TUI
mltracker-tui

# Build docs (MkDocs Material)
mkdocs serve
mkdocs build
```

## Architecture

### Core Data Flow

```
Raw Time Series
  → TimeSeriesHandler      (scaling, imputation, time features)
  → create_dataloaders     (PyTorch DataLoader + TimeSeriesDataset)
  → ForecastingModel       (head + encoder/decoder backbone)
  → Trainer                (training loop, validation, NAS hooks, MLTracker)
  → ModelEvaluator         (metrics, plots, cross-validation)
```

### Key Modules

**`foreblocks/core/`** — Central abstractions
- `ForecastingModel`: unified wrapper for direct/autoregressive/seq2seq modes
- `BaseHead`: interface all prediction heads conform to
- `ConformalPredictionEngine`: post-hoc uncertainty quantification
- `heads/`: 11+ preprocessing heads — RevIN, decomposition, wavelet, FFT, patch embedding, multi-scale convolution

**`foreblocks/tf/`** — Transformer stack
- `attention/`: standard, linear, Kimi, hybrid attention backends with kernel variants
- `experts/`: MoE routing, dispatchers, expert logging
- `embeddings/`: rotary PE, Informer time embedding
- `norms/`: RMSNorm, LayerNorm, RevIN, temporal norm, group norm
- `popular/`: reference implementations — DLinear, FED, PatchTST, iTransformer, TFT, AutoTransformer
- `skip/`: GateSkip, LayerSkip (dynamic depth), residual gating

**`foreblocks/blocks/`** — Encoder/decoder primitives
- Pairs available: LSTM, GRU, Attention
- Specialty: TCN, Fourier, ODE, xLSTM, Wavelets, Graph convolution

**`foreblocks/training/`** — Training & experiment management
- `Trainer`: training loop, early stopping, NAS hooks, MLTracker integration, quantization
- `MLTracker`: logs metrics, hyperparameters, artifacts, model snapshots

**`foreblocks/ts_handler/`** — Preprocessing pipeline
- `TimeSeriesHandler`: windowing, scaling (StandardScaler/RobustScaler), outlier handling, imputation, seasonal decomposition, time features

**`foreblocks/darts/`** — Neural Architecture Search
- `DARTSSearcher`: differentiable architecture search
- Evaluation pipeline for candidate architectures; hooks into `Trainer`

**`foreblocks/mltracker/`** — Experiment tracking
- Local FastAPI server + Textual TUI (`mltracker-tui`)
- `dashboard_v2`: monitoring dashboard for runs

**`foreblocks/mamba/`** and **`foreblocks/kan/`** — Alternative backbone implementations (Mamba SSM, Kolmogorov-Arnold Networks)

**`foretools/`** — Companion ecosystem
- `tsgen/`: synthetic time-series generation (AR, seasonal, trend, noise)
- `bohb/`: Bayesian + random hyperparameter optimization
- `fengineer/`: feature engineering pipeline
- `foreminer/`: dataset mining, changepoint detection
- `vmd/`, `ewt/`, `arima/`: decomposition and baseline methods
- `benchmarking/`: comparison against neuralforecast and external baselines

**`foreblocks/mamba/`** and **`foreblocks/kan/`** — Alternative backbone implementations (Mamba SSM, Kolmogorov-Arnold Networks)

**`flash-attention/`** — Vendored FA4 implementation using CuTeDSL (Hopper/Blackwell GPUs)

**`tree/`** — C++ gradient boosting tree solver (CMake build, scalar-only)

**`webui/`** — Visual node editor for building forecasting pipelines
- Frontend: React 19 + ReactFlow + TailwindCSS + Zustand (`src/TimeSeriesNodeEditor.tsx`)
- Backend: FastAPI async server (`server.py`) with WebSocket streaming and thread-pool execution
- Components decorated with `@node()` are discoverable by the webui
- Start: `cd webui && npm install && npm run build && python server.py`

### Optional Extras System

Extras are defined in `pyproject.toml`. Heavy dependencies (pandas, scipy, scikit-learn, optuna, neuralforecast, etc.) are optional — import guards using `try/except ImportError` are the norm in this codebase. Follow the same pattern when adding code that requires optional dependencies.

### CLI Entry Point

`mltracker-tui` → `foreblocks.mltracker.mltracker_tui:main`

## Code Conventions

- **Imports**: isort black-profile, `force_single_line = True`, `float_to_top = True`
- **Type hints**: used throughout; static checking via `basedpyright`
- **Models**: `nn.Module` subclasses; avoid functional-only style for complex components
- **Lazy loading**: `foreblocks/__init__.py` uses `__getattr__` for deferred imports — follow this pattern for any new top-level exports
- **Optional deps**: guard with `try/except ImportError`; heavy extras (pandas, scipy, optuna, neuralforecast) are never in core
- **Reproducibility**: explicit seeds, clear train/val/test splits in experiments
- **Docs URL**: https://foreblocks.laioseman.com/docs/
