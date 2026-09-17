---
title: System Overview
description: High-level architecture, subsystem map, and design intent.
editLink: true
---

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
| `foreblocks/ops` | compute tier — Triton/CUDA kernels, no `nn.Module` API surface |
| `foreblocks/layers` | reusable `nn.Module` primitives (norms, embeddings, graph) |
| `foreblocks/attention` | attention config, variants, cache, and execution |
| `foreblocks/modules` | composable model modules (attention wrappers, MoE, blocks, heads, skip) |
| `foreblocks/core` | core model assembly (`model`, `att`, `sampling`, `extend`), plus `training/` (Trainer) and `evaluation/` (ModelEvaluator) |
| `foreblocks/models` | assembled models: `popular/` (NBEATS, Informer, …), `transformer/`, `kan/`, `sequence/` (mamba, raven backbones), `anomaly/` (applications layer) |
| `foreblocks/studio` | Studio node/spec auto-discovery backend for `apps/webui` |
| `foreblocks/ts_handler` | preprocessing, normalization, filtering, imputation, window creation |
| `darts` | neural architecture search and finalization workflow |
| `foretools` | synthetic data generation, feature engineering, decomposition, BOHB search |
| `mltracker` | experiment tracking support |

## Public API boundary

The stable starting point is the top-level import surface:

```python
from foreblocks import ForecastingModel, Trainer, ModelEvaluator
```

That boundary is safer than importing deep internal modules unless you are extending the library itself.

## Dependency graph

```text
foreblocks (main)
├── foreblocks/ops              — compute kernels: kernels/, attention/, mamba/, raven/, graph/
├── foreblocks/layers           — norms/, embeddings/, graph/ (nn.Module primitives)
├── foreblocks/attention        — attention config, variants, cache, execution
├── foreblocks/modules          — moe/, blocks/, heads/, skip/ (composable modules)
├── foreblocks/core             — ForecastingModel, heads, conformal, sampling
│   ├── training/                 — Trainer, optimizer/scheduler
│   └── evaluation/                — ModelEvaluator, metrics
├── foreblocks/models
│   ├── popular/                 — NBEATS, Informer, Autoformer, TimesNet, …
│   ├── transformer/              — Transformer stack, attention, MoE
│   ├── kan/                      — Kolmogorov-Arnold Network
│   ├── sequence/                 — mamba/ (Hybrid Mamba SSM), raven/ (recurrent)
│   └── anomaly/                  — anomaly-detection applications layer
├── foreblocks/studio            — Studio node/spec auto-discovery backend
├── foreblocks/ts_handler        — TimeSeriesHandler, preprocessing
├── mltracker                    — Experiment tracking
└── darts (standalone)           — Neural architecture search

foretools (companion)
├── foretools/tsgen          — Synthetic time-series generation
├── foretools/bohb           — Bayesian hyperparameter search
├── foretools/emd_like       — VMD / EMD decomposition
├── foretools/tsaug          — AutoDA augmentation
├── foretools/fengineer      — Feature engineering pipeline
└── foretools/foreminer      — Changepoint detection & mining
```

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
