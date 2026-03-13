# System Overview

This page explains how the repository is organized at a subsystem level.

## Two layers

The repository contains two related but distinct layers:

- `foreblocks`: the main forecasting library
- `foretools`: auxiliary tools for generation, decomposition, and analysis

## Core runtime path

The main `foreblocks` training flow is:

1. prepare data as windows
2. build a `ForecastingModel`
3. train with `Trainer`
4. evaluate with `ModelEvaluator`

## Main subsystems

| Path | Role |
| --- | --- |
| `foreblocks/core` | core model assembly, heads, conformal prediction, sampling |
| `foreblocks/training` | training loop, optimizer/scheduler handling, NAS-aware training support |
| `foreblocks/evaluation` | evaluation, prediction helpers, metrics, plotting |
| `foreblocks/ts_handler` | preprocessing, normalization, filtering, imputation, window creation |
| `foreblocks/tf` | transformer stack, attention variants, MoE, norms, embeddings |
| `foreblocks/darts` | neural architecture search and finalization workflow |
| `foreblocks/mltracker` | experiment tracking support |
| `foretools/tsgen` | synthetic data generation |

## Public API boundary

The stable starting point is the top-level import surface:

```python
from foreblocks import ForecastingModel, Trainer, ModelEvaluator
```

That boundary is safer than importing deep internal modules unless you are extending the library itself.

## Design intent

The codebase is organized around modular composition:

- model composition in `ForecastingModel`
- backbone specialization in recurrent and transformer blocks
- optional preprocessing before training
- optional architecture search through DARTS
- optional synthetic data generation in `foretools`

## Related pages

- [Forecasting Pipeline](forecasting-pipeline.md)
- [Public API](../reference/public-api.md)
- [Repository Map](../reference/repository-map.md)
