---
title: Home
description: Modular time-series forecasting, architecture search, and companion utilities for PyTorch.
lastUpdated: false
editLink: true
---

# ForeBlocks Docs

<div class="hero-panel">
  <span class="eyebrow">foreBlocks v{{ $frontmatter.foreBlocksVersion }} | Documentation</span>
  <h1>Modular time-series forecasting for PyTorch</h1>
  <p class="hero-lead">
    <code>foreblocks</code> provides forecasting models, training, evaluation, and preprocessing.
    <code>darts</code> is a standalone differentiable neural-architecture-search package.
    <code>foretools</code> offers companion utilities for synthetic data, feature engineering, decomposition, and hyperparameter search.
    All three ship in the same <code>foreblocks</code> wheel. The docs guide you from a simple baseline to advanced architectures and architecture search.
  </p>
  <div class="hero-actions">
    <a class="md-button md-button--primary" href="getting-started/">Run your first model</a>
    <a class="md-button" href="overview/">Read the overview</a>
    <a class="md-button" href="reference/public-api/">Browse the public API</a>
  </div>
</div>

<div class="metric-strip">
  <div class="metric-chip"><strong>Start small</strong><span>Use <code>ForecastingModel</code> + <code>Trainer</code> + <code>ModelEvaluator</code> for a quick baseline.</span></div>
  <div class="metric-chip"><strong>Raw series ready</strong><span><code>TimeSeriesHandler</code> converts raw <code>[T, D]</code> arrays into training windows automatically.</span></div>
  <div class="metric-chip"><strong>Search included</strong><span>DARTS guides cover screening, bilevel search, retraining, and analysis.</span></div>
  <div class="metric-chip"><strong>Toolbox utilities</strong><span><code>foretools</code> adds synthetic data generation, BOHB search, VMD decomposition, and feature engineering.</span></div>
</div>

## Choose your route

<div class="path-grid">
  <div class="path-card">
    <p class="route-kicker">Path 01</p>
    <h3><a href="getting-started/">Train a baseline model</a></h3>
    <p>Start with the smallest reliable public API path. Validates imports, dataloader shapes, and evaluation before adding extras.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 02</p>
    <h3><a href="preprocessor/">Handle raw time series</a></h3>
    <p>Use <code>TimeSeriesHandler</code> to automatically scale, filter, window, and prepare raw multivariate arrays.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 03</p>
    <h3><a href="transformer/">Explore advanced backbones</a></h3>
    <p>Learn transformer attention, MoE routing, custom blocks, and Custom Mamba for stronger models.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 04</p>
    <h3><a href="darts/">Automate architecture search</a></h3>
    <p>Run DARTS to automatically discover optimal architecture combinations from a defined operation pool.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 05</p>
    <h3><a href="uncertainty/">Add uncertainty intervals</a></h3>
    <p>Use conformal prediction to generate prediction intervals when you need coverage guarantees.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 06</p>
    <h3><a href="foretools/">Use companion utilities</a></h3>
    <p>Access synthetic data generation, BOHB search, VMD decomposition, and feature engineering tools.</p>
  </div>
</div>

## Recommended reading order

<div class="step-grid">
  <div class="step-card">
    <strong>1. Run a baseline</strong>
    <span>Follow <a href="getting-started/">Getting Started</a> to validate the core training loop with minimal dependencies.</span>
  </div>
  <div class="step-card">
    <strong>2. Understand the stack</strong>
    <span>Read <a href="overview/">Overview</a> to see how <code>foreblocks</code> and <code>foretools</code> connect.</span>
  </div>
  <div class="step-card">
    <strong>3. Explore the public API</strong>
    <span>Browse <a href="reference/public-api/">Public API</a> to learn stable imports before diving into subsystems.</span>
  </div>
  <div class="step-card">
    <strong>4. Choose your workflow</strong>
    <span>Pick a guide based on your task: preprocessing, transformers, DARTS search, uncertainty, or foretools.</span>
  </div>
</div>

## Install by intent

| Need | Command | Next page |
| --- | --- | --- |
| Core forecasting | `pip install foreblocks` | [Getting Started](getting-started) |
| Raw-series preprocessing | `pip install "foreblocks[preprocessing]"` | [Preprocessor Guide](preprocessor) |
| Architecture search (DARTS) | `pip install "foreblocks[darts]"` | [DARTS Guide](darts) |
| Experiment tracking (MLTracker) | `pip install "foreblocks[mltracker]"` | [Web UI](webui) |
| VMD decomposition | `pip install "foreblocks[vmd]"` | [VMD Guide](foretools/vmd) |
| Wavelet preprocessing | `pip install "foreblocks[wavelets]"` | [Preprocessor Guide](preprocessor) |
| All runtime extras | `pip install "foreblocks[all]"` | [Overview](overview) |

## Documentation map

### Tutorials

Step-by-step runnable paths:

- [Getting Started](getting-started)
- [Train a Direct Model](tutorials/train-direct-model)
- [Run a DARTS Search](tutorials/darts-multifidelity-search)
- [Generate Synthetic Series](tutorials/generate-synthetic-series)
- [Optimize With BOHB](tutorials/optimize-with-bohb)

### Guides

Topic-based deep dives:

- [Preprocessor](preprocessor)
- [Custom Blocks](custom_blocks)
- [Transformer](transformer)
- [Mixture of Experts](moe)
- [Custom Mamba](hybrid-mamba)
- [KAN Backbone](kan)
- [DARTS](darts)
- [Evaluation & Metrics](evaluation)
- [Uncertainty Quantification](uncertainty)
- [Web UI](webui)
- [Troubleshooting](troubleshooting)

### Foretools

Companion utilities and helpers:

- [Foretools Overview](foretools/index)
- [Time Series Generator](foretools/tsgen)
- [BOHB Search](foretools/bohb)
- [VMD Decomposition](foretools/vmd)
- [AutoDA Augmentation](foretools/tsaug)
- [Feature Engineering](foretools/feature-engineering)

### Architecture

Internals and system notes:

- [System Overview](architecture/system-overview)
- [Forecasting Pipeline](architecture/forecasting-pipeline)
- [DARTS Search Pipeline](architecture/darts-pipeline)

## Stable public entry points

Start here before using deep imports:

### Core

```python
from foreblocks import (
    ForecastingModel,
    Trainer,
    ModelEvaluator,
    TimeSeriesHandler,
    TimeSeriesDataset,
    create_dataloaders,
    ModelConfig,
    TrainingConfig,
)
```python

### Uncertainty

```python
from foreblocks.core import ConformalPredictionEngine
```python

### Wavelet & frequency attention

Spectral attention is available today through the transformer stack rather than a
standalone import — select it with the `att_type` argument:

```python
from foreblocks import TransformerEncoder

# Wavelet-domain (Haar DWT) attention
enc = TransformerEncoder(input_size=8, d_model=64, num_layers=2, att_type="dwt")

# Frequency-domain (FEDformer-style) attention
enc = TransformerEncoder(input_size=8, d_model=64, num_layers=2, att_type="frequency")
