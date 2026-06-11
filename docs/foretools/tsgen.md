---
title: Time Series Generator
description: Synthetic time-series generator with explicit structural components for testing and prototyping.
editLink: true
---


[[toc]]
# Time Series Generator

`foretools/tsgen/ts_gen.py` provides a pedagogical synthetic time-series generator with explicit structural components. It is useful when you need data that is fully controlled, explainable, and easy to visualize.

## Import path

```python
from foretools.tsgen import TimeSeriesGenerator
```python

## What `make()` returns

### `df`

A tidy dataframe with one row per timestamp per series.

Expected columns:

- `series`
- `time`
- `y`
- optional exogenous columns such as `x1`, `x2`, ...
- optional calendar columns such as `month`, `dayofweek`, `fourier_wk_sin`, `fourier_yr_cos`

### `meta`

A metadata dictionary that contains the ground truth behind the generated series.

Most useful keys:

- `meta["time_index"]`: shared pandas time index
- `meta["params"]`: the configuration used to generate the data
- `meta["components"]`: arrays for trend, seasonality, cycle, noise, regime bias, and exogenous effect
- `meta["states"]`: latent regime state path when regime switching is enabled
- `meta["missing_mask"]`: missing-value mask if missingness was injected
- `meta["splits"]`: chronological split boundaries if `splits=` was requested

## Supported component families

| Component | Main options | Notes |
| --- | --- | --- |
| Trend | `linear`, `poly`, `piecewise` | `piecewise` uses `knots` and per-segment `slopes` |
| Seasonality | list of seasonal specs | each seasonal spec supports `period`, `amplitude`, `phase`, `harmonics` |
| Cycle | `period`, `amplitude`, `freq_drift`, `phase` | low-frequency oscillation separate from the main seasonal terms |
| Noise | `ar`, `ma`, `sigma` | ARMA-style base noise |
| Regime | `n_states`, transition matrix `p`, `state_bias`, `state_sigma_scale` | adds discrete state-dependent bias and volatility |
| Heteroskedasticity | `{"type": "arch1", "alpha0": ..., "alpha1": ...}` | replaces the base noise with ARCH-like noise |
| Outliers | `prob`, `scale` | injects heavy-tailed spikes |
| Missingness | `prob`, `block_prob`, `block_max` | applies missing values to `y` only |
| Exogenous inputs | `n_features`, `types`, `beta` | supported types are `random_walk`, `seasonal`, `binary_event`, `noise` |
| Multivariate mixing | `n_factors`, `mix_strength` | induces cross-series correlation after component synthesis |
| Calendar features | `add_calendar=True` | adds date parts and Fourier encodings |

## Common examples

### 1. Fully observed structural series

Use this when you want clean series for plotting, teaching, or debugging a model's ability to recover trend and seasonality.

```python
df, meta = gen.make(
    n_series=2,
    n_steps=400,
    trend={"type": "linear", "slope": 0.03, "intercept": 5.0},
    seasonality=[{"period": 7.0, "amplitude": 3.0, "harmonics": 2}],
    cycle={"period": 160.0, "amplitude": 1.2},
    noise={"ar": [0.5], "ma": [0.1], "sigma": 0.4},
    return_components=True,
)
```json

### 3. Create a train/val/test split quickly

`make_train_ready()` pivots the generated data into wide format and reserves the last two horizons for validation and test.

```python
dataset = gen.make_train_ready(
    n_series=3,
    n_steps=240,
    horizon=24,
    trend={"type": "linear", "slope": 0.02},
    seasonality=[{"period": 24.0, "amplitude": 1.5}],
    noise={"ar": [0.3], "sigma": 0.5},
)

train_df = dataset["train"]
val_df = dataset["val"]
test_df = dataset["test"]
meta = dataset["meta"]
```text

- `plot_series()` shows the observed `y` path for a single series.
- `plot_decompose()` renders separate trend, seasonality, cycle, and noise plots.

If you need a richer figure that overlays all components at once, use the companion notebook `foretools/tsgen/ts_gen_complete_series.ipynb`.

## Practical behavior to know

- Missingness is applied after the full signal is constructed, and only to the `y` column.
- When `heterosked={"type": "arch1", ...}` is enabled, the generator replaces the previously built base noise with ARCH-like noise for clarity.
- Calendar features are shared across all series because they come from the common timestamp index.
- Multivariate factor mixing happens after the univariate structural components are combined, so it affects the final series rather than individual components.
- `return_components=False` keeps the dataframe output but omits the large component arrays from `meta`.

## Recommended workflow

1. Start with `return_components=True` so you can inspect the exact decomposition.
2. Keep missingness and outliers off for first-pass model debugging.
3. Add regimes, exogenous drivers, and missingness only after the baseline workflow is working.
4. Use the notebook examples when you want publication-style plots or teaching material.

## Related pages

- [Generate Synthetic Series](../tutorials/generate-synthetic-series)
- [Foretools Overview](index)
- [Repository Map](../reference/repository-map)
