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
| `web/` | Static landing page assets for the published site root |
| `docs/` | VitePress source for the versioned documentation site |
| `examples/` | Notebooks and runnable examples |
| `foreblocks/` | Main forecasting library |
| `foretools/` | Companion tooling |
| `darts/` | Neural architecture search (DARTS) |

## `foreblocks/`

| Path | Purpose |
| --- | --- |
| [`foreblocks/__init__.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/__init__.py) | Top-level public exports |
| [`foreblocks/config.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/config.py) | Public configuration dataclasses (`ModelConfig`, `TrainingConfig`) |
| `foreblocks/ops/` | Low-level compute kernels (Triton/CUDA): `kernels/` (grouped_gemm, swiglu, norms), `attention/` (FLA, fused RoPE, paged/chunked), `mamba/` (SSD, conv1d), `raven/` |
| `foreblocks/layers/` | Reusable `nn.Module` primitives: `norms/`, `embeddings/`, `graph/` |
| `foreblocks/modules/` | Composable model modules: `attention/`, `moe/`, `blocks/`, `heads/`, `skip/` |
| `foreblocks/models/` | Assembled models + composition APIs (`ForecastingModel`, `GraphForecastingModel`), `popular/` named models, `transformer/` stack, `kan/` (Kolmogorov-Arnold backbone) |
| `foreblocks/sequence/` | Alternative sequence backbones: `mamba/` (original), `mamba_hybrid/` (formerly `custom_mamba`), `raven/` (formerly `custom_raven`) |
| `foreblocks/core/` | Core forecasting internals (`model`, `att`, `sampling`, `extend`), plus `training/` (trainer) and `evaluation/` (metrics) |
| `foreblocks/data/` | Dataset and dataloader helpers |
| `foreblocks/ts_handler/` | Preprocessing and sequence construction |
| `foreblocks/mltracker/` | Experiment tracking |
| `foreblocks/experimental/` | Not-yet-stable components: `attention_kernels/` (formerly `custom_att`, has own `setup.py`) |
| `foreblocks/third_party/` | Small vendored compatibility helpers; larger external projects should stay outside the wheel |

## Package organization (post-reorg)

The package was reorganized into a tiered layout. There are **no compatibility
shims** — old import paths (`foreblocks.transformer.*`, `foreblocks.blocks.*`,
`foreblocks.custom_mamba.*`, `foreblocks.custom_raven.*`, `foreblocks.custom_att.*`)
were hard-renamed. See [Reorg Migration Map](reorg-migration) for the full
old → new table.

| Tier | Package | What lives here |
| --- | --- | --- |
| compute | `foreblocks/ops/` | Triton/CUDA kernels, no `nn.Module` API surface |
| primitives | `foreblocks/layers/` | Reusable `nn.Module` primitives (norms, embeddings, graph) |
| modules | `foreblocks/modules/` | Composable model modules (attention, moe, blocks, heads, skip) |
| models | `foreblocks/models/` | Fully assembled models + composition, incl. `popular/` and `transformer/` |
| backbones | `foreblocks/sequence/` | Alternative sequence backbones (mamba, mamba_hybrid, raven) |
| experimental | `foreblocks/experimental/` | Not-yet-stable sub-projects (e.g. attention_kernels) |

Conventions:

- `ops/` is pure compute. If it imports `torch.nn` as an API surface, it belongs in `layers/` or `modules/`.
- `modules/blocks/` holds research blocks; `models/popular/` holds the named end-to-end models (NBEATS, Informer, …). The `transformer/popular/` and `blocks/popular/` split was merged into `models/popular/`.
- `sequence/` drops the old `custom_` prefix; `custom_mamba` → `mamba_hybrid` to distinguish it from the original `mamba`.
- Frontend assets under `foreblocks/studio/` and `foreblocks/mltracker/dashboard_v2/`: package only built `dist` assets; keep `node_modules`, runtime databases, and local tracker artifacts out of git and release archives.
- `mltracker/mltracker_data/`: prefer `.foreblocks/mltracker_data`, `~/.foreblocks/mltracker_data`, or an explicit user-configured run directory rather than the package tree.

## `foretools/`

| Path | Purpose |
| --- | --- |
| `foretools/tsgen/` | Synthetic time-series generation |
| `foretools/bohb/` | BOHB, TPE configuration, pruning, and optimization plots |
| `foretools/foreminer/` | Exploratory analysis and diagnostics |
| `foretools/fengineer/` | Feature engineering utilities |
| `foretools/emd_like/` | Decomposition tools |
| `foretools/tsaug/` | AutoDA-Timeseries: automated data augmentation with adaptive policy |

## Recommended entry points by task

| Task | Entry point |
| --- | --- |
| Training a baseline model | [`README.md`](https://github.com/lseman/foreblocks/blob/main/README), [Getting Started](../getting-started) |
| Understanding architecture composition | [`foreblocks/models/`](https://github.com/lseman/foreblocks/tree/main/foreblocks/models) |
| Working with graph forecasting | [`foreblocks/models/graph_forecasting.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/models/graph_forecasting.py), [`foreblocks/layers/graph/`](https://github.com/lseman/foreblocks/tree/main/foreblocks/layers/graph) |
| Writing Triton kernels | [`foreblocks/ops/`](https://github.com/lseman/foreblocks/tree/main/foreblocks/ops) |
| Configuring runs | [`foreblocks/config.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/config.py) |
| Building dataloaders | [`foreblocks/data/dataset.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/data/dataset.py) |
| Adding preprocessing logic | [`foreblocks/ts_handler/preprocessing.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/ts_handler/preprocessing.py) |
| Exploring transformer internals | [`foreblocks/models/transformer/transformer.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/models/transformer/transformer.py) |
| Working on architecture search | [`darts/`](https://github.com/lseman/foreblocks/tree/main/darts) |
| Using SSM / Mamba-style blocks | [`foreblocks/sequence/mamba_hybrid/`](https://github.com/lseman/foreblocks/tree/main/foreblocks/sequence/mamba_hybrid) |
| Generating synthetic data | [`foretools/tsgen/`](https://github.com/lseman/foreblocks/tree/main/foretools/tsgen) |
| Running hyperparameter search | [`foretools/bohb/`](https://github.com/lseman/foreblocks/tree/main/foretools/bohb) |
| Augmenting training data adaptively | [`foretools/tsaug/`](https://github.com/lseman/foreblocks/tree/main/foretools/tsaug) |

## Related pages

- [System Overview](../architecture/system-overview)
- [Public API](public-api)
- [Foretools Overview](../foretools/index)
- [Documentation Workflow](../contributing/docs-workflow)
