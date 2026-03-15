# Repository Map

This page gives a quick path through the repository for contributors and power users.

## Top-level areas

| Path | Purpose |
| --- | --- |
| `README.md` | GitHub landing page |
| `web/` | static landing page assets for the published site root |
| `docs/` | MkDocs source for the versioned documentation site |
| `examples/` | notebooks and runnable examples |
| `foreblocks/` | main forecasting library |
| `foretools/` | companion tooling |

## `foreblocks/`

| Path | Purpose |
| --- | --- |
| `foreblocks/__init__.py` | top-level public exports |
| `foreblocks/core/` | `ForecastingModel`, heads, conformal tools |
| `foreblocks/training/` | trainer and training support |
| `foreblocks/evaluation/` | evaluation and metrics |
| `foreblocks/ts_handler/` | preprocessing and sequence construction |
| `foreblocks/tf/` | transformer stack and advanced attention |
| `foreblocks/darts/` | neural architecture search |
| `foreblocks/mltracker/` | experiment tracking |
| `foreblocks/hybrid_mamba/` | Hybrid Mamba SSM blocks (HybridMambaBlock, HybridMamba2Block, SSD) |
| `foreblocks/mamba/` | Original Mamba backbone with MoE, positional encoding, and eval tools |
| `foreblocks/kan/` | Kolmogorov-Arnold Network backbone |

## `foretools/`

| Path | Purpose |
| --- | --- |
| `foretools/tsgen/` | synthetic time-series generation |
| `foretools/bohb/` | BOHB, TPE configuration, pruning, and optimization plots |
| `foretools/foreminer/` | exploratory analysis and diagnostics |
| `foretools/fengineer/` | feature engineering utilities |
| `foretools/vmd/` | decomposition tools |
| `foretools/tsaug/` | AutoDA-Timeseries: automated data augmentation with adaptive policy |

## Recommended entry points by task

- training a baseline model: `README.md`, `docs/getting-started.md`
- understanding architecture composition: `foreblocks/core/model.py`
- adding preprocessing logic: `foreblocks/ts_handler/preprocessing.py`
- exploring transformer internals: `foreblocks/tf/transformer.py`
- working on search: `foreblocks/darts/`
- generating synthetic data: `foretools/tsgen/`
- running black-box hyperparameter search: `foretools/bohb/`
- using SSM / Mamba-style blocks: `foreblocks/hybrid_mamba/layers.py`
- augmenting training data adaptively: `foretools/tsaug/`

## Related pages

- [System Overview](../architecture/system-overview.md)
- [Public API](public-api.md)
- [Foretools Overview](../foretools/index.md)
- [Documentation Workflow](../contributing/docs-workflow.md)
