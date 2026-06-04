# Repository Map

This page gives a quick path through the repository for contributors and power users.

## Top-level areas

| Path | Purpose |
| --- | --- |
| [`README.md`](https://github.com/lseman/foreblocks/blob/main/README.md) | GitHub landing page |
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
| `foreblocks/models/` | Model-level composition APIs (`ForecastingModel`, `GraphForecastingModel`) |
| `foreblocks/blocks/` | Research-inspired building blocks and forecasting heads that can be composed into models |
| `foreblocks/layers/` | Reusable lower-level layer families, currently focused on graph convolutions and graph construction |
| `foreblocks/core/` | Core forecasting internals, model heads, and composition utilities |
| `foreblocks/training/` | Trainer and training support |
| `foreblocks/evaluation/` | Evaluation and metrics |
| `foreblocks/data/` | Dataset and dataloader helpers |
| `foreblocks/ts_handler/` | Preprocessing and sequence construction |
| `foreblocks/transformer/` | Transformer stack and advanced attention |
| `foreblocks/mltracker/` | Experiment tracking |
| `foreblocks/custom_mamba/` | Hybrid Mamba SSM blocks (HybridMambaBlock, HybridMamba2Block, SSD) |
| `foreblocks/mamba/` | Original Mamba backbone with MoE, positional encoding, and eval tools |
| `foreblocks/custom_raven/` | Experimental Raven/FLA-inspired sequence blocks |
| `foreblocks/custom_att/` | Experimental custom attention kernels and benchmarks |
| `foreblocks/kan/` | Kolmogorov-Arnold Network backbone (see [KAN Backbone](../kan.md)) |
| `foreblocks/third_party/` | Small vendored compatibility helpers; larger external projects should stay outside the wheel |

## Package organization notes

The package currently has a few historical names that are worth preserving for
compatibility but not copying for new code:

| Area | Current state | Recommended direction |
| --- | --- | --- |
| `custom_mamba/`, `mamba/`, `custom_raven/` | Multiple sequence-model families live as sibling packages. `custom_mamba` is already public in docs and tests. | Keep existing imports as compatibility shims, but place future SSM-style families under a neutral namespace such as `foreblocks/sequence/` or `foreblocks/models/sequence/`. |
| `custom_att/` | Kernel experiment package with its own `setup.py`, tests, and benchmarks under `foreblocks/`. | Move future kernel experiments under `foreblocks/experimental/` or `foreblocks/transformer/attention/kernels/` when they are production-ready. |
| `blocks/` vs `layers/` | `blocks/` contains user-facing research blocks; `layers/graph/` contains lower-level graph layers. | Treat `blocks/` as composable model blocks and `layers/` as primitive reusable layers. Avoid adding one-off model heads directly to `layers/`. |
| `transformer/popular/` and `blocks/popular/` | Some heads are re-exported from both places for registry convenience. | Keep the implementation in one package and make the other package a thin compatibility/registry wrapper. |
| Frontend assets under `foreblocks/studio/` and `foreblocks/mltracker/dashboard_v2/` | Source, built assets, and local `node_modules` can coexist in working trees. | Package only built `dist` assets. Keep `node_modules`, runtime databases, and local tracker artifacts out of git and out of release archives. |
| `mltracker/mltracker_data/` | Runtime experiment data can become very large if it lives under the Python package tree. | Prefer `.foreblocks/mltracker_data`, `~/.foreblocks/mltracker_data`, or an explicit user-configured run directory. |

## Suggested staged cleanup

1. **No-break cleanup**
   - Keep public imports unchanged.
   - Ensure generated frontend folders, `node_modules`, `__pycache__`, build logs, and tracker runtime data are ignored and absent from source distributions.
   - Fix stale docs that reference removed compatibility paths.

2. **Compatibility namespace**
   - Add a neutral namespace for new sequence models, for example `foreblocks/sequence/`.
   - Re-export existing `custom_mamba` and `custom_raven` objects from that namespace.
   - Mark old `custom_*` names as compatibility paths in docs rather than primary homes.

3. **Implementation moves**
   - Move stable custom attention kernels into the existing transformer kernel layout.
   - Move experimental kernels into `foreblocks/experimental/` until they have tests and a stable API.
   - Keep import shims in the old locations for at least one minor release.

4. **Breaking rename window**
   - Only after shims and docs have existed for a release, consider renaming packages or removing old paths.
   - Do this in a versioned migration with import warnings and a changelog entry.

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
| Training a baseline model | [`README.md`](https://github.com/lseman/foreblocks/blob/main/README.md), [Getting Started](../getting-started.md) |
| Understanding architecture composition | [`foreblocks/models/`](https://github.com/lseman/foreblocks/tree/main/foreblocks/models) |
| Working with graph forecasting | [`foreblocks/models/graph_forecasting.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/models/graph_forecasting.py), [`foreblocks/layers/graph/`](https://github.com/lseman/foreblocks/tree/main/foreblocks/layers/graph) |
| Configuring runs | [`foreblocks/config.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/config.py) |
| Building dataloaders | [`foreblocks/data/dataset.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/data/dataset.py) |
| Adding preprocessing logic | [`foreblocks/ts_handler/preprocessing.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/ts_handler/preprocessing.py) |
| Exploring transformer internals | [`foreblocks/transformer/transformer.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/transformer/transformer.py) |
| Working on architecture search | [`darts/`](https://github.com/lseman/foreblocks/tree/main/darts) |
| Using SSM / Mamba-style blocks | [`foreblocks/custom_mamba/blocks/`](https://github.com/lseman/foreblocks/tree/main/foreblocks/custom_mamba/blocks) |
| Generating synthetic data | [`foretools/tsgen/`](https://github.com/lseman/foreblocks/tree/main/foretools/tsgen) |
| Running hyperparameter search | [`foretools/bohb/`](https://github.com/lseman/foreblocks/tree/main/foretools/bohb) |
| Augmenting training data adaptively | [`foretools/tsaug/`](https://github.com/lseman/foreblocks/tree/main/foretools/tsaug) |

## Related pages

- [System Overview](../architecture/system-overview.md)
- [Public API](public-api.md)
- [Foretools Overview](../foretools/index.md)
- [Documentation Workflow](../contributing/docs-workflow.md)
