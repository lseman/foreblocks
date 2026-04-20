# Foretools Overview

`foretools` is the companion toolbox that sits next to `foreblocks`.

Use `foreblocks` when you are building and training forecasting models. Use `foretools` when you need support utilities around that workflow: synthetic data, black-box search, exploratory diagnostics, decomposition, or feature engineering.

## Best-documented tools

| Tool | When to use it | Docs |
| --- | --- | --- |
| `foretools/tsgen` | create synthetic series with known structure and ground-truth components | [Time Series Generator](tsgen.md) |
| `foretools/bohb` | run budgeted hyperparameter optimization with Hyperband + TPE | [BOHB Search](bohb.md) |
| `foretools/emd_like` | decompose signals into oscillatory modes with VMD, EMD-family methods, hierarchical VMD, and multivariate support | [VMD Decomposition](vmd.md) |
| `foretools/fengineer` | automated feature engineering with transforms, interactions, MI selection, and RFECV | [Feature Engineering](feature-engineering.md) |
| `foretools/tsaug` | data augmentation — jitter, scaling, time-warp, window-slice, and AutoDA search | [AutoDA Augmentation](tsaug.md) |

## Other foretools areas

### `foretools/foreminer`

`foreminer` is an exploratory-analysis toolkit for understanding your time series before modelling.

Key capabilities:

- **Changepoint detection** — locate structural breaks in long series
- **Cluster analysis** — group series or windows by similarity
- **Dimensionality diagnostics** — PCA and UMAP projections of window embeddings
- **Group-level summaries** — aggregate statistics and seasonal decomposition across cohorts
- **Stationarity checks** — ADF and KPSS tests with automated reporting

Quick import path:

```python
from foretools.foreminer import ForeMiner

miner = ForeMiner(series)          # series: [T, D] numpy array
report = miner.run()               # returns a dict of diagnostic frames
miner.plot_changepoints()
miner.plot_clusters(n_clusters=4)
```

`foreminer` is primarily notebook-oriented. It does not expose a stable training-time API and is best used in exploratory phases before committing to a preprocessing and model pipeline.

### `foretools/foraug` (tsaug)

Data augmentation utilities. See [AutoDA Augmentation](tsaug.md) for the full guide.

## How `foretools` fits the repo

- `foreblocks` is the main model and training API.
- `foretools` is a set of practical companion modules. Some are notebook-oriented and some are reusable library code.
- `foretools` imports are deeper and less consolidated than `foreblocks`, so the safest entry points are the specific modules documented here.

## Recommended reading

1. [Time Series Generator](tsgen.md) if you need synthetic datasets or decomposition examples.
2. [BOHB Search](bohb.md) if you need hyperparameter optimization outside the `foreblocks.darts` neural architecture search stack.
3. [VMD Decomposition](vmd.md) if you need decomposition, denoising, or mode extraction workflows.
4. [Feature Engineering](feature-engineering.md) if you need automated feature construction, mutual information selection, or RFECV-based pruning.
5. [Repository Map](../reference/repository-map.md) if you want the broader code layout.

## Related pages

- [Generate Synthetic Series](../tutorials/generate-synthetic-series.md)
- [Optimize With BOHB](../tutorials/optimize-with-bohb.md)
- [VMD Decomposition](vmd.md)
- [Feature Engineering](feature-engineering.md)
- [Overview](../overview.md)
