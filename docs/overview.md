# Documentation Overview

This repository has two layers that work well together but serve different purposes:

- `foreblocks`: the main forecasting library
- `foretools`: companion utilities for generation, search, decomposition, and analysis

The docs are organized to make that split explicit, while still showing how the pieces connect in a single workflow.

## Best entry point by goal

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
| Diagnose import/setup issues | [Troubleshooting](troubleshooting.md) |

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

Treat deeper imports as subsystem-level APIs, not general entry points, unless a topic guide tells you to use them directly.

## How the docs are layered

### Tutorials

Runnable paths first. These are the pages to use when you want to verify the environment, shape expectations, and basic success criteria.

### Guides

Subsystem pages that explain capabilities, important configuration knobs, and how the modules are intended to be composed.

### Architecture notes

Pages that explain how code is divided internally. These are most useful when you are extending or debugging the implementation.

### Reference

Stable surfaces, configuration maps, and repository orientation.

## Install map

The packaging now reflects the actual feature boundaries more closely:

| Need | Suggested install |
| --- | --- |
| minimal forecasting core | `pip install foreblocks` |
| preprocessing, filtering, statistics | `pip install "foreblocks[preprocessing]"` |
| DARTS training/search helpers | `pip install "foreblocks[darts]"` |
| DARTS analyzer and richer search visuals | `pip install "foreblocks[darts-analysis]"` |
| MLTracker UI and API clients | `pip install "foreblocks[mltracker]"` |
| VMD utilities | `pip install "foreblocks[vmd]"` |
| all runtime extras | `pip install "foreblocks[all]"` |

## Repository landmarks

| Area | Purpose |
| --- | --- |
| `foreblocks/core` | model assembly, heads, conformal utilities |
| `foreblocks/training` | trainer loop, optimizer/scheduler integration |
| `foreblocks/evaluation` | evaluator, metrics, benchmark helpers |
| `foreblocks/ts_handler` | preprocessing, filtering, imputation, window creation |
| `foreblocks/tf` | transformer stack, attention, MoE, norms, embeddings |
| `foreblocks/darts` | architecture search configs, search loops, analysis |
| `foreblocks/mltracker` | experiment tracking and local dashboards |
| `foretools` | synthetic data, BOHB, VMD, exploratory tooling |

## Recommended reading order

1. [Docs Home](index.md)
2. [Getting Started](getting-started.md)
3. [Public API](reference/public-api.md)
4. [Preprocessor Guide](preprocessor.md)
5. [Transformer Guide](transformer.md)
6. [MoE Guide](moe.md) or [DARTS Guide](darts.md), depending on your workflow
7. [Foretools Overview](foretools/index.md) if you also need tooling outside the core training loop

## Practical notes

- The project is broad. Not every internal module is meant to be treated as stable public API.
- `ForecastingModel` plus `Trainer` remains the shortest and safest path for a new user.
- `TimeSeriesHandler` is the best bridge from raw arrays into the trainer loop.
- DARTS is not just a single training function. It is a staged NAS workflow with zero-cost screening, differentiable search, discrete derivation, and final retraining.
- `foretools` is worth browsing even if you only use `foreblocks`, especially for data generation and search tooling.
