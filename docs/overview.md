# Overview

This repository has two layers that work well together but serve different purposes:

- `foreblocks`: the main forecasting library
- `foretools`: companion utilities for generation, search, decomposition, and analysis

The docs are organized to make that split explicit while still showing how the pieces connect in a single workflow.

<div class="callout-grid">
  <div class="glass-card">
    <strong>Start here if you are new</strong>
    <span><a href="getting-started/">Getting Started</a> is still the safest first read.</span>
  </div>
  <div class="glass-card">
    <strong>Keep the first run small</strong>
    <span>Validate the public API path before opening the more specialist subsystems.</span>
  </div>
  <div class="glass-card">
    <strong>Branch by workflow</strong>
    <span>Use the guide that matches your actual task instead of reading every subsystem in order.</span>
  </div>
</div>

## Best starting page by goal

| Goal | Best starting page |
| --- | --- |
| Run a first end-to-end training loop | [Getting Started](getting-started.md) |
| Start from raw multivariate series | [Preprocessor Guide](preprocessor.md) |
| Understand stable top-level imports | [Public API](reference/public-api.md) |
| Customize model blocks or training internals | [Custom Blocks Guide](custom_blocks.md) |
| Work with transformer backbones | [Transformer Guide](transformer.md) |
| Enable expert routing | [MoE Guide](moe.md) |
| Run neural architecture search | [DARTS Guide](darts.md) |
| Generate synthetic time series | [Time Series Generator](foretools/tsgen.md) |
| Run budgeted hyperparameter search | [BOHB Search](foretools/bohb.md) |
| Diagnose install or shape issues | [Troubleshooting](troubleshooting.md) |

## Mental model of the repo

<div class="path-grid">
  <div class="path-card">
    <p class="route-kicker">Layer 01</p>
    <h3>Stable public surface</h3>
    <p>The safest imports live at the top level of <code>foreblocks</code>: model assembly, trainer loop, dataloaders, configs, evaluator, and preprocessing bridge.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Layer 02</p>
    <h3>Optional workflow extras</h3>
    <p>Preprocessing, DARTS, tracking, VMD, and other heavier dependencies are packaged as extras so the base install stays lean.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Layer 03</p>
    <h3>Specialist subsystems</h3>
    <p>Transformer internals, MoE, Hybrid Mamba, uncertainty, and architecture notes are best treated as focused branches once the baseline path is healthy.</p>
  </div>
</div>

## What is stable today

The most reliable public surface is still the top-level `foreblocks` import path:

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
```

The DARTS stack has its own public namespace:

```python
from foreblocks.darts import DARTSTrainer
```

Treat deeper imports as subsystem-level APIs unless a topic guide explicitly tells you to use them directly.

## Install map

The packaging reflects real feature boundaries:

| Need | Suggested install |
| --- | --- |
| Minimal forecasting core | `pip install foreblocks` |
| Preprocessing, filtering, statistics | `pip install "foreblocks[preprocessing]"` |
| DARTS training, search, and analysis | `pip install "foreblocks[darts]"` |
| MLTracker UI and API clients | `pip install "foreblocks[mltracker]"` |
| VMD utilities | `pip install "foreblocks[vmd]"` |
| All runtime extras | `pip install "foreblocks[all]"` |

## How the docs are layered

### Tutorials

Runnable paths first. Use these when you want a clear success condition and a smaller number of moving parts.

### Guides

Subsystem pages that explain capabilities, important configuration knobs, and how modules are meant to be composed.

### Architecture notes

Pages that explain internal structure and code layout. These are more useful when you are extending, debugging, or reviewing implementation choices.

### Reference

Stable surfaces, configuration maps, and repository orientation.

## Repository landmarks

| Area | Purpose |
| --- | --- |
| `foreblocks/core` | model assembly, heads, conformal utilities |
| `foreblocks/training` | trainer loop, optimizer/scheduler integration |
| `foreblocks/evaluation` | evaluator, metrics, benchmark helpers |
| `foreblocks/ts_handler` | preprocessing, filtering, imputation, window creation |
| `foreblocks/tf` | transformer stack, attention variants, MoE, norms, embeddings |
| `foreblocks/darts` | architecture search configs, search loops, analysis |
| `foreblocks/mltracker` | experiment tracking and local dashboards |
| `foretools` | synthetic data, BOHB, VMD, exploratory tooling |

## Recommended reading tracks

### Track A: I just want a model training

1. [Getting Started](getting-started.md)
2. [Public API](reference/public-api.md)
3. [Evaluation & Metrics](evaluation.md)

### Track B: I have raw data and need preprocessing

1. [Getting Started](getting-started.md)
2. [Preprocessor Guide](preprocessor.md)
3. [Feature Engineering](foretools/feature-engineering.md)

### Track C: I want automated search or more advanced architectures

1. [Getting Started](getting-started.md)
2. [Transformer Guide](transformer.md) or [MoE Guide](moe.md)
3. [DARTS Guide](darts.md)
4. [DARTS Search Pipeline](architecture/darts-pipeline.md)

## Practical notes

- The project is broad. Not every internal module should be treated as stable public API.
- `ForecastingModel` plus `Trainer` is still the best first path for a new user.
- `TimeSeriesHandler` is the main bridge from raw arrays into the trainer loop.
- DARTS is a staged workflow, not just a single training function.
- `foretools` is worth browsing even if you primarily use `foreblocks`, especially for data generation and search tooling.
