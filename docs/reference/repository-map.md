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
| [`foreblocks/__init__.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/__init__.py) | Top-level public exports |
| [`foreblocks/config.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/config.py) | Public configuration dataclasses (`ModelConfig`, `TrainingConfig`) |
| `foreblocks/ops/` | Low-level compute kernels (Triton/CUDA): `kernels/` (grouped_gemm, swiglu, norms), `attention/` (fused RoPE, paged/chunked), `mamba/` (SSD, conv1d), `raven/`, `graph/`, `experimental/` |
| `foreblocks/layers/` | Reusable `nn.Module` primitives: `norms/`, `embeddings/`, `graph/` |
| `foreblocks/attention/` | Attention config, variants (`implementations/`), KV cache (`cache/`), execution and preparation helpers |
| `foreblocks/modules/` | Composable model modules: `moe/`, `blocks/`, `heads/`, `skip/` |
| `foreblocks/models/` | Assembled models + composition APIs (`ForecastingModel`, `GraphForecastingModel`): `popular/` named models (NBEATS, Informer, Autoformer, TimesNet, …), `transformer/` stack, `kan/` (Kolmogorov-Arnold backbone), `sequence/` (`mamba/`, `raven/` alternative backbones), `anomaly/` (anomaly-detection applications layer) |
| `foreblocks/core/` | Core forecasting internals (`model`, `att`, `sampling`, `extend`), plus `training/` (Trainer) and `evaluation/` (ModelEvaluator) |
| `foreblocks/data/` | Dataset and dataloader helpers |
| `foreblocks/ts_handler/` | Preprocessing and sequence construction |
| `foreblocks/studio/` | Studio node/spec auto-discovery backend consumed by `apps/webui` |
| `foreblocks/studio_server.py` | Local HTTP server for the built Studio frontend (`apps/webui/dist`) |

## Package organization (tiered layout)

| Tier | Package | What lives here |
| --- | --- | --- |
| compute | `foreblocks/ops/` | Triton/CUDA kernels, no `nn.Module` API surface |
| primitives | `foreblocks/layers/` | Reusable `nn.Module` primitives (norms, embeddings, graph) |
| attention | `foreblocks/attention/` | Attention config, variants, cache, execution |
| modules | `foreblocks/modules/` | Composable model modules (moe, blocks, heads, skip) |
| core | `foreblocks/core/` | Model assembly internals + training + evaluation |
| models | `foreblocks/models/` | Fully assembled models + composition, incl. `popular/`, `transformer/`, `kan/`, `sequence/`, `anomaly/` |
| applications | `foreblocks/studio/` | Studio backend (node/spec discovery) — no heavy deps |

Conventions:

- `ops/` is pure compute. If it imports `torch.nn` as an API surface, it belongs in `layers/` or `modules/`.
- `modules/blocks/` holds research blocks; `models/popular/` holds the named end-to-end models.
- `models/anomaly/` is nested under `models/` because it's an applications layer built on `models/`+`core/`, not a foundational tier.
- `studio/` (formerly `ui/`) avoids the naming collision with the `apps/webui` frontend it serves specs to.
- Frontend assets under `apps/webui/dist/` and `apps/mltracker-dashboard/dist/`: package only built `dist` assets; keep `node_modules`, runtime databases, and local tracker artifacts out of git and release archives.
- `src/mltracker/mltracker_data/`: prefer `.foreblocks/mltracker_data`, `~/.foreblocks/mltracker_data`, or an explicit user-configured run directory rather than the package tree.

See [Reorg Migration Map](reorg-migration) for old → new import path history.

## `foretools/`

| Path | Purpose |
| --- | --- |
| `foretools/tsgen/` | Synthetic time-series generation |
| `foretools/bohb/` | BOHB, TPE configuration, pruning, and optimization plots |
| `foretools/foreminer/` | Exploratory analysis and diagnostics |
| `foretools/fengineer/` | Feature engineering utilities |
| `foretools/emd_like/` | Decomposition tools |
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
| Working with graph forecasting | [`src/foreblocks/models/graph_forecasting.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/models/graph_forecasting.py), [`src/foreblocks/layers/graph/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/layers/graph) |
| Writing Triton kernels | [`src/foreblocks/ops/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/ops) |
| Configuring runs | [`src/foreblocks/config.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/config.py) |
| Building dataloaders | [`src/foreblocks/data/dataset.py`](https://github.com/lseman/foreblocks/blob/main/src/foreblocks/data/dataset.py) |
| Adding preprocessing logic | [`src/foreblocks/ts_handler/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/ts_handler) |
| Exploring transformer internals | [`src/foreblocks/models/transformer/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/models/transformer) |
| Working on architecture search | [`src/darts/`](https://github.com/lseman/foreblocks/tree/main/src/darts) |
| Using SSM / Mamba-style blocks | [`src/foreblocks/models/sequence/mamba/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/models/sequence/mamba) |
| Anomaly detection | [`src/foreblocks/models/anomaly/`](https://github.com/lseman/foreblocks/tree/main/src/foreblocks/models/anomaly) |
| Generating synthetic data | [`src/foretools/tsgen/`](https://github.com/lseman/foreblocks/tree/main/src/foretools/tsgen) |
| Running hyperparameter search | [`src/foretools/bohb/`](https://github.com/lseman/foreblocks/tree/main/src/foretools/bohb) |
| Augmenting training data adaptively | [`src/foretools/tsaug/`](https://github.com/lseman/foreblocks/tree/main/src/foretools/tsaug) |

## Related pages

- [System Overview](../architecture/system-overview)
- [Public API](public-api)
- [Foretools Overview](../foretools/index)
- [Documentation Workflow](../contributing/docs-workflow)
