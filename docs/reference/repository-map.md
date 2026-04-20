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

## `foreblocks/`

| Path | Purpose |
| --- | --- |
| [`foreblocks/__init__.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/__init__.py) | Top-level public exports |
| `foreblocks/core/` | `ForecastingModel`, heads, conformal tools |
| `foreblocks/training/` | Trainer and training support |
| `foreblocks/evaluation/` | Evaluation and metrics |
| `foreblocks/ts_handler/` | Preprocessing and sequence construction |
| `foreblocks/tf/` | Transformer stack and advanced attention |
| `foreblocks/darts/` | Neural architecture search |
| `foreblocks/mltracker/` | Experiment tracking |
| `foreblocks/hybrid_mamba/` | Hybrid Mamba SSM blocks (HybridMambaBlock, HybridMamba2Block, SSD) |
| `foreblocks/mamba/` | Original Mamba backbone with MoE, positional encoding, and eval tools |
| `foreblocks/kan/` | Kolmogorov-Arnold Network backbone |

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
| Understanding architecture composition | [`foreblocks/core/model.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/core/model.py) |
| Adding preprocessing logic | [`foreblocks/ts_handler/preprocessing.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/ts_handler/preprocessing.py) |
| Exploring transformer internals | [`foreblocks/tf/transformer.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/tf/transformer.py) |
| Working on architecture search | [`foreblocks/darts/`](https://github.com/lseman/foreblocks/tree/main/foreblocks/darts) |
| Using SSM / Mamba-style blocks | [`foreblocks/hybrid_mamba/layers.py`](https://github.com/lseman/foreblocks/blob/main/foreblocks/hybrid_mamba/layers.py) |
| Generating synthetic data | [`foretools/tsgen/`](https://github.com/lseman/foreblocks/tree/main/foretools/tsgen) |
| Running hyperparameter search | [`foretools/bohb/`](https://github.com/lseman/foreblocks/tree/main/foretools/bohb) |
| Augmenting training data adaptively | [`foretools/tsaug/`](https://github.com/lseman/foreblocks/tree/main/foretools/tsaug) |

## Related pages

- [System Overview](../architecture/system-overview.md)
- [Public API](public-api.md)
- [Foretools Overview](../foretools/index.md)
- [Documentation Workflow](../contributing/docs-workflow.md)
