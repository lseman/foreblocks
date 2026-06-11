---
title: Getting Started
description: Install foreblocks and run your first training loop.
editLink: true
---


[[toc]]
# Getting Started

This guide gives you the shortest reliable path to a working `foreblocks` training loop, then shows where to branch when you need preprocessing, search, or richer tooling.

If you want the broader mental model first, start from [Overview](overview).

<div class="callout-grid">
  <div class="glass-card">
    <strong>Goal</strong>
    <span>Run one small training job end to end with the stable public API.</span>
  </div>
  <div class="glass-card">
    <strong>Good first success</strong>
    <span>You can import the package, train a baseline model, and compute metrics on held-out data.</span>
  </div>
  <div class="glass-card">
    <strong>Only then</strong>
    <span>Add preprocessing extras, custom transformer internals, DARTS, or tracking-heavy workflows.</span>
  </div>
</div>

## 1. Install

This project targets Python 3.10 and newer.

### PyPI (stable)

```bash
pip install foreblocks
```bash

### With extras

| Extra | Adds |
| --- | --- |
| `preprocessing` | `TimeSeriesHandler`, windowing, scaling, filtering, imputation |
| `darts` | Architecture search, NAS, and evaluation helpers |
| `mltracker` | Experiment tracking API, local dashboard, and CLI TUI |
| `studio` | Studio frontend launcher and bundled server command |
| `vmd` | VMD decomposition and analysis helpers |
| `wavelets` | Wavelet preprocessing and multiwavelet feature extraction |
| `benchmark` | External forecasting baselines and spreadsheet readers |
| `foreminer` | Changepoint detection and dataset mining |
| `all` | All runtime extras |

### Install examples

```bash
# DARTS only
pip install "foreblocks[darts]"

# Multiple extras
pip install "foreblocks[vmd,wavelets]"

# Everything (large)
pip install "foreblocks[all]"
```toml

::: info What this first run should validate
- Your `foreblocks` import path is correct.
- Dataloader shapes line up with the trainer.
- The model trains without needing optional subsystems.
- Evaluation works on held-out data.
:::

## 3. Validate the import surface first

Run a quick import check before the full example:

```bash
python -c "from foreblocks import ForecastingModel, Trainer; print('foreblocks import OK')"
```python

### What this validates

- The import path works correctly
- Dataloader shapes match the trainer expectations
- The model trains without optional subsystems
- Evaluation works on held-out data

## 5. Trainer and MLTracker notes

`Trainer` initializes MLTracker automatically if installed. Pass `auto_track=False` during local smoke tests.

## 6. Shape expectations

### Direct forecasting

| Tensor | Shape | Description |
| --- | --- | --- |
| `X` | `[N, T, F]` | Samples × input timesteps × features |
| `y` | `[N, H]` | Samples × horizon |

### Encoder / decoder (seq2seq)

| Tensor | Shape | Description |
| --- | --- | --- |
| `X` | `[N, T, F]` | Samples × input timesteps × features |
| `y` | `[N, H, D]` | Samples × horizon × output channels |

Decoder-based models have stricter dimension contracts. Read the [Custom Blocks](custom_blocks) guide before wiring custom modules.

## 7. Starting from raw time series

When your starting point is a single `[T, D]` array, use `TimeSeriesHandler` instead of building windows manually:

```python
import numpy as np
import pandas as pd
from foreblocks import TimeSeriesHandler

# Load raw data
raw = np.random.randn(240, 3)  # [T, F]
timestamps = pd.date_range("2025-01-01", periods=len(raw), freq="h")

# Configure preprocessing
pre = TimeSeriesHandler(
    window_size=24,   # sequence length
    horizon=6,        # forecast horizon
    normalize=True,   # standardize features
    generate_time_features=False,
    verbose=False,
)

# Fit and transform
X, y, processed, time_feats = pre.fit_transform(raw, time_stamps=timestamps)

# Transform validation data using fitted state
X_val = pre.transform(val_raw, time_stamps=val_timestamps)
```bash
:::

Continue with [Preprocessor Guide](preprocessor) for advanced options like filtering, outlier handling, and feature engineering.

## 8. When to add DARTS

If the basic training loop works and you want architecture search instead of hand-selecting blocks:

```bash
pip install "foreblocks[darts]"
```toml
Baseline works and metrics are acceptable?
├── No → improve your data
│   ├── Raw series (single array)   → Preprocessor Guide
│   └── Feature engineering        → Feature Engineering
│
├── Yes, but I want better accuracy
│   ├── Try a stronger backbone     → Transformer Guide
│   ├── Add MoE feedforward         → MoE Guide
│   ├── Add preprocessing heads     → Custom Blocks Guide
│   └── Search architectures        → DARTS Guide
│
├── Yes, but I need uncertainty
│   └── Post-hoc conformal intervals → Uncertainty Guide
│
├── Yes, but I need richer evaluation
│   └── Metrics, plots, CV           → Evaluation Guide
│
└── Yes, but training is slow or OOM
    └── AMP, gradient checkpointing  → Configuration Reference
```text

## Related pages

- [Overview](overview) - Mental model of the stack
- [Public API](reference/public-api) - Complete API reference
- [Configuration](reference/configuration) - All configuration options
- [Troubleshooting](troubleshooting) - Common issues and fixes
