---
title: Repository Map
description: Quick path through the repo for contributors and power users.
editLink: true
---

# Repository Map

This page gives a quick path through the repository for contributors and power users.

## Top-level areas

| Path | Purpose |
| --- | --- |
| [`README.md`](https://github.com/lseman/foreblocks/blob/main/README) | GitHub landing page |
| [`docs/.vitepress/config.js`](https://github.com/lseman/foreblocks/blob/main/docs/.vitepress/config.js) | Navigation and site structure for the `/docs/` site |
| `site/landing/` | Hand-authored source for the published site root (`site/` itself is otherwise CI-generated build output, gitignored) |
| `docs/` | VitePress source for the versioned documentation site |
| `examples/` | Notebooks and runnable examples |
| `apps/` | Frontends: `apps/webui/` (Studio node editor), `apps/mltracker-dashboard/` (MLTracker dashboard v2) |
| `src/foreblocks/` | Main forecasting library |
| `src/foretools/` | Companion tooling |
| `src/darts/` | Neural architecture search (DARTS) |
| `src/mltracker/` | Experiment tracking backend |
| `projects/` | Standalone sub-projects, not part of the `foreblocks` distribution — see [`projects/README.md`](https://github.com/lseman/foreblocks/blob/main/projects/README.md) |

## `src/foreblocks/`

| Path | Purpose |
| --- | --- |
| [`foreblocks/__init__.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/__init__.py) | Top-level public exports (lazy-loaded) |
| `foreblocks/kernels/` | Triton/CUDA accelerated kernels, grouped by op family: `attention/`, `mamba/`, `linear/`, `normalization/`, `activations/`, `elementwise/`, `graph/`, `experimental/` (vendored `attention_kernels` sub-project) |
| `foreblocks/ops/` | Tensor operations and execution dispatch on top of `kernels/`: `attention/` (fused RoPE, paged/chunked decode), `mamba/` (SSD, conv1d) |
| `foreblocks/integrations/` | Optional external backends: `fla/` (flash-linear-attention adapters — delta rule, gated deltanet, GLA, RAVEN), `softpick.py` |
| `foreblocks/nn/` | Reusable `nn.Module` building blocks: `attention/`, `transformer/`, `blocks/`, `heads/`, `moe/`, `routing/`, `sequence/` (`mamba/`, `raven/`), `normalization/`, `embeddings/`, `residual/`, `graph/` |
| `foreblocks/models/` | Assembled models + composition APIs: `forecasting.py` (`ForecastingModel`), `graph.py` (`GraphForecastingModel`), `config.py` (`ModelConfig`), `sequence.py`, `distillation.py`, `baselines/` (NBEATS, Informer, Autoformer, TimesNet, …), `kan/` (Kolmogorov-Arnold backbone), `anomaly/` (anomaly-detection applications layer) |
| `foreblocks/training/` | Training orchestration: `trainer.py` (`Trainer`), `config.py` (`TrainingConfig`), `losses.py`, `sampling.py`, `conformal/`, `optimization/`, `execution/`, `state/`, `telemetry/` |
| `foreblocks/evaluation/` | `model_evaluator.py` (`ModelEvaluator`), `benchmark.py`, `visualization.py` |
| `foreblocks/quantization/` | Quantization configs and modules (`FakeQuantize`, `DynamicQuantizedLinear`) |
| `foreblocks/data/` | Dataset and dataloader helpers (`dataset.py`, `csv.py`) |
| `foreblocks/processing/` | Preprocessing, filtering, imputation, and sequence construction (`TimeSeriesHandler`): `core/`, `filters/`, `auto_filter/`, `transforms/`, `tools/` |
| `foreblocks/studio/` | Studio node/spec auto-discovery backend consumed by `apps/webui` |
| `foreblocks/studio_server.py` | Local HTTP server for the built Studio frontend (`apps/webui/dist`) |

## Package organization (tiered layout)

| Tier | Package | What lives here |
| --- | --- | --- |
| compute | `foreblocks/kernels/` | Triton/CUDA kernels, no `nn.Module` API surface |
| ops | `foreblocks/ops/` | Tensor ops and execution dispatch on top of `kernels/` |
| integrations | `foreblocks/integrations/` | Optional external backends (flash-linear-attention, RAVEN) |
| primitives | `foreblocks/nn/` | Reusable `nn.Module` primitives — attention, transformer, blocks, heads, moe, routing, sequence, normalization, embeddings, graph |
| models | `foreblocks/models/` | Fully assembled models + composition, incl. `baselines/`, `kan/`, `anomaly/` |
| training/evaluation | `foreblocks/training/`, `foreblocks/evaluation/` | Training loop, config, and evaluation/benchmarking |
| applications | `foreblocks/studio/` | Studio backend (node/spec discovery) — no heavy deps |

Conventions:

- `kernels/` is pure compute. If it imports `torch.nn` as an API surface, it belongs in `nn/`.
- `nn/blocks/` holds research blocks; `models/baselines/` holds the named end-to-end models.
- `models/anomaly/` is nested under `models/` because it's an applications layer built on `models/`+`nn/`, not a foundational tier.
- `studio/` avoids the naming collision with the `apps/webui` frontend it serves specs to.
- Frontend assets under `apps/webui/dist/` and `apps/mltracker-dashboard/dist/`: package only built `dist` assets; keep `node_modules`, runtime databases, and local tracker artifacts out of git and release archives.
- `src/mltracker/mltracker_data/`: prefer `.foreblocks/mltracker_data`, `~/.foreblocks/mltracker_data`, or an explicit user-configured run directory rather than the package tree.

## `foretools/`

| Path | Purpose |
| --- | --- |
| `foretools/tsgen/` | Synthetic time-series generation |
| `foretools/bohb/` | BOHB, TPE configuration, pruning, and optimization plots |
| `foretools/foreminer/` | Exploratory analysis and diagnostics |
| `foretools/fengineer/` | Feature engineering: `transformers/`, `selectors/`, `filters/` |
| `foretools/decomposition/` | Signal decomposition: `emd/` (EMD/EEMD/CEEMDAN/VMD), `ewt/` (Empirical Wavelet Transform) |
| `foretools/stats/` | Standalone statistical utilities: mutual information, distance correlation, HSIC, Bayesian Blocks binning |
| `foretools/arima/` | ARIMA model utilities |
| `foretools/benchmarking/` | Benchmarking frameworks |
| `foretools/tsaug/` | AutoDA-Timeseries: automated data augmentation with adaptive policy |

## `projects/`

Standalone sub-projects that live in this repository but are not part of the `foreblocks`
distribution (no entry in `pyproject.toml`, not importable as `foreblocks.*`):

| Path | Purpose |
| --- | --- |
| `projects/tree/` | ForeTree — standalone C++23/CUDA tree-model library (histogram splitting), pybind bindings, own CMake build |
| `projects/scheduling/` | Neural schedulers (RL/GNN) for the Offline Nanosatellite Task Scheduling (ONTS) problem |

## Recommended entry points by task

| Task | Entry point |
| --- | --- |
| Training a baseline model | [`README.md`](https://github.com/lseman/foreblocks/blob/main/README), [Getting Started](../getting-started) |
| Understanding architecture composition | [`src/foreblocks/models/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/models) |
| Working with graph forecasting | [`src/foreblocks/models/graph.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/models/graph.py), [`src/foreblocks/nn/graph/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/nn/graph) |
| Writing Triton kernels | [`src/foreblocks/kernels/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/kernels) |
| Configuring runs | [`src/foreblocks/models/config.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/models/config.py) (`ModelConfig`), [`src/foreblocks/training/config.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/training/config.py) (`TrainingConfig`) |
| Building dataloaders | [`src/foreblocks/data/dataset.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/data/dataset.py) |
| Adding preprocessing logic | [`src/foreblocks/processing/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/processing) |
| Exploring transformer internals | [`src/foreblocks/nn/transformer/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/nn/transformer) |
| Working on architecture search | [`src/darts/`](https://github.com/lseman/foreblocks/tree/main/src/darts) |
| Using SSM / Mamba-style blocks | [`src/foreblocks/nn/sequence/mamba/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/nn/sequence/mamba) |
| Anomaly detection | [`src/foreblocks/models/anomaly/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/models/anomaly) |
| Generating synthetic data | [`src/foretools/tsgen/`](https://github.com/lseman/foreblocks/tree/main/src/foretools/tsgen) |
| Running hyperparameter search | [`src/foretools/bohb/`](https://github.com/lseman/foreblocks/tree/main/src/foretools/bohb) |
| Augmenting training data adaptively | [`src/foretools/tsaug/`](https://github.com/lseman/foreblocks/tree/main/src/foretools/tsaug) |

## Related pages

- [System Overview](../architecture/system-overview)
- [Public API](public-api)
- [Foretools Overview](../foretools/index)
- [Documentation Workflow](../contributing/docs-workflow)
