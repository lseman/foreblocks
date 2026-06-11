---
title: Foretools Overview
description: Companion toolbox — synthetic data, search, decomposition, feature engineering.
editLink: true
---

# Foretools Overview

`foretools` is the companion toolbox that sits next to `foreblocks`.

Use `foreblocks` when you are building and training forecasting models. Use `foretools` when you need support utilities around that workflow: synthetic data, black-box search, exploratory diagnostics, decomposition, or feature engineering.

## Best-documented tools

| Tool | When to use it | Docs |
| --- | --- | --- |
| `foretools/tsgen` | create synthetic series with known structure and ground-truth components | [Time Series Generator](tsgen) |
| `foretools/bohb` | run budgeted hyperparameter optimization with Hyperband + TPE | [BOHB Search](bohb) |
| `foretools/emd_like` | decompose signals into oscillatory modes with VMD, EMD-family methods, hierarchical VMD, and multivariate support | [VMD Decomposition](vmd) |
| `foretools/fengineer` | automated feature engineering with transforms, interactions, MI selection, and RFECV | [Feature Engineering](feature-engineering) |
| `foretools/tsaug` | data augmentation — jitter, scaling, time-warp, window-slice, and AutoDA search | [AutoDA Augmentation](tsaug) |

## Other foretools areas

### `foretools/foreminer`

`foreminer` is an exploratory-analysis toolkit for understanding your time series before modelling.

Key capabilities (each is a registered analysis key — see `get_available_analyses()`):

- **Distributions & outliers** — per-feature distribution and anomaly diagnostics
- **Correlations & graph analysis** — pairwise correlations and correlation-network structure
- **Clustering & dimensionality** — group series/windows by similarity; PCA/UMAP projections
- **Patterns & timeseries** — seasonality, trend, and temporal-structure diagnostics
- **Feature engineering & SHAP** — candidate features and SHAP-based importance explanations
- **Missingness & categorical groups** — gap analysis and cohort-level summaries

Quick import path:

```python
import pandas as pd
from foretools.foreminer.foreminer import DatasetAnalyzer

# df: a pandas DataFrame; pass time_col if you have an explicit timestamp column
analyzer = DatasetAnalyzer(df, time_col="timestamp")

# Discover the analysis keys registered in your build, then run a subset
print(analyzer.get_available_analyses())
# e.g. ['distributions', 'correlations', 'outliers', 'clusters', ...]
results = analyzer.analyze(["correlations", "outliers"])

# Run-and-plot in one step, or fetch a single analysis result
analyzer.analyze_and_plot(["clusters"])
corr = analyzer.get_results("correlations")
