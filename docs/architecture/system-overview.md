---
title: System Overview
description: High-level architecture, subsystem map, and design intent.
editLink: true
---

# System Overview

This page explains how the repository is organized at a subsystem level.

## Python packages

The distribution includes four import packages. Use `foreblocks` to build and
train forecasting models, `darts` for architecture search, `foretools` for
companion analysis workflows, and `mltracker` for experiment tracking.

## Core runtime path

The main `foreblocks` training flow is:

1. prepare data as windows
2. build a `ForecastingModel`
3. train with `Trainer`
4. evaluate with `ModelEvaluator`

## Main subsystems

| Path | Role |
| --- | --- |
| `foreblocks/kernels` | compute tier — Triton/CUDA kernels, no `nn.Module` API surface |
| `foreblocks/ops` | tensor ops and execution dispatch on top of `kernels/` |
| `foreblocks/nn` | reusable `nn.Module` primitives: attention config/variants/cache, transformer stack, blocks, heads, MoE, routing, sequence backbones, norms, embeddings, graph |
| `foreblocks/models` | assembled models: `forecasting.py` (`ForecastingModel`), `graph.py`, `baselines/` (NBEATS, Informer, …), `kan/`, `anomaly/` (applications layer) |
| `foreblocks/training` | Trainer, config, sampling, conformal prediction, optimization (LLRD, NAS) |
| `foreblocks/evaluation` | ModelEvaluator, benchmarking, metrics |
| `foreblocks/studio` | Studio node/spec auto-discovery backend for `apps/webui` |
| `foreblocks/processing` | preprocessing, normalization, filtering, imputation, window creation |
| `darts` | neural architecture search and finalization workflow |
| `foretools` | synthetic data generation, feature engineering, decomposition, BOHB search |
| `mltracker` | experiment tracking support |

## Public API boundary

The stable starting point is the top-level import surface:

```python
from foreblocks import ForecastingModel, Trainer, ModelEvaluator
```

That boundary is safer than importing deep internal modules unless you are extending the library itself.

## How the packages fit together

Forecasting models compose layers from `foreblocks.nn`. Tensor operations in
`foreblocks.ops` use kernels or optional external backends where available.
Training and evaluation consume the assembled model. DARTS builds search
models using some of the same layers and kernels. The contributor map lists
the source locations for each subsystem. [Repository map](../reference/repository-map.md)

## Choosing where to start

| Your goal | Start here |
| --- | --- |
| Train a forecasting model from scratch | [Getting Started](../getting-started) |
| Use raw `[T, D]` time-series data | [Preprocessor Guide](../preprocessor) |
| Try different model architectures | [Transformer Guide](../transformer) or [DARTS Guide](../darts) |
| Add uncertainty estimates | [Uncertainty Quantification](../uncertainty) |
| Search for a better architecture | [Run A DARTS Search](../tutorials/darts-multifidelity-search) |
| Generate synthetic test data | [Generate Synthetic Series](../tutorials/generate-synthetic-series) |

## Design intent

The codebase is organized around modular composition:

- model composition in `ForecastingModel`
- backbone specialization in recurrent and transformer blocks
- optional preprocessing before training
- optional architecture search through DARTS
- optional synthetic data generation in `foretools`

## Related pages

- [Forecasting Pipeline](forecasting-pipeline.md)
- [DARTS Search Pipeline](darts-pipeline.md)
- [Public API](../reference/public-api)
- [Repository Map](../reference/repository-map)
