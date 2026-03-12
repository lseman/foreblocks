# Time Series Generator

`foretools/tsgen/ts_gen.py` provides a pedagogical synthetic time-series generator with explicit structural components. It is useful when you need data that is fully controlled, explainable, and easy to visualize.

## Import path

```python
from foretools.tsgen.ts_gen import TimeSeriesGenerator
```

## Core workflow

```python
from foretools.tsgen.ts_gen import TimeSeriesGenerator

gen = TimeSeriesGenerator(random_state=42)

df, meta = gen.make(
    n_series=3,
    n_steps=365,
    freq="D",
    trend={
        "type": "piecewise",
        "knots": [120, 250],
        "slopes": [0.08, -0.03, 0.05],
        "intercept": 18.0,
    },
    seasonality=[
        {"period": 7.0, "amplitude": 4.2, "harmonics": 2},
        {"period": 30.5, "amplitude": 1.8, "harmonics": 1},
    ],
    cycle={"period": 180.0, "amplitude": 2.4, "freq_drift": 0.08, "phase": 0.2},
    noise={"ar": [0.55], "ma": [0.2], "sigma": 0.9},
    regime={
        "n_states": 2,
        "p": [[0.96, 0.04], [0.08, 0.92]],
        "state_bias": [0.0, 3.2],
        "state_sigma_scale": [1.0, 1.35],
    },
    exog={
        "n_features": 2,
        "types": ["random_walk", "seasonal"],
        "beta": [0.22, -1.1],
    },
    add_calendar=True,
    return_components=True,
)
```

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
```

### 2. Stress-test with missingness, regimes, and exogenous drivers

Use this when you want a harder forecasting or preprocessing benchmark.

```python
df, meta = gen.make(
    n_series=4,
    n_steps=730,
    trend={"type": "piecewise", "knots": [180, 500], "slopes": [0.05, -0.02, 0.01]},
    seasonality=[
        {"period": 7.0, "amplitude": 2.0, "harmonics": 2},
        {"period": 30.5, "amplitude": 1.0, "harmonics": 1},
    ],
    regime={
        "n_states": 3,
        "p": [[0.95, 0.04, 0.01], [0.05, 0.90, 0.05], [0.02, 0.08, 0.90]],
        "state_bias": [0.0, 1.5, -1.0],
        "state_sigma_scale": [1.0, 1.4, 1.8],
    },
    exog={
        "n_features": 3,
        "types": ["random_walk", "binary_event", "seasonal"],
        "beta": [0.1, 2.0, -0.7],
    },
    missing={"prob": 0.02, "block_prob": 0.01, "block_max": 7},
    outliers={"prob": 0.01, "scale": 5.0},
    return_components=True,
)
```

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
```

## Plotting helpers

`TimeSeriesGenerator` includes simple plotting helpers for fast inspection:

```python
gen.plot_series(df, series_id=0)
gen.plot_decompose(meta, series_id=0)
```

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

- [Generate Synthetic Series](../tutorials/generate-synthetic-series.md)
- [Foretools Overview](index.md)
- [Repository Map](../reference/repository-map.md)
