# Repository Map

This page gives a quick path through the repository for contributors and power users.

## Top-level areas

| Path | Purpose |
| --- | --- |
| `README.md` | GitHub landing page |
| `docs/` | static landing page assets for the published site root |
| `wiki/` | MkDocs source for the versioned documentation site |
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
| `foreblocks/pre/` | preprocessing and sequence construction |
| `foreblocks/tf/` | transformer stack and advanced attention |
| `foreblocks/darts/` | neural architecture search |
| `foreblocks/mltracker/` | experiment tracking |

## `foretools/`

| Path | Purpose |
| --- | --- |
| `foretools/tsgen/` | synthetic time-series generation |
| `foretools/foreminer/` | exploratory analysis and diagnostics |
| `foretools/fengineer/` | feature engineering utilities |
| `foretools/vmd/` | decomposition tools |

## Recommended entry points by task

- training a baseline model: `README.md`, `wiki/getting-started.md`
- understanding architecture composition: `foreblocks/core/model.py`
- adding preprocessing logic: `foreblocks/pre/preprocessing.py`
- exploring transformer internals: `foreblocks/tf/transformer.py`
- working on search: `foreblocks/darts/`
- generating synthetic data: `foretools/tsgen/`

## Related pages

- [System Overview](../architecture/system-overview.md)
- [Public API](public-api.md)
- [Documentation Workflow](../contributing/docs-workflow.md)
