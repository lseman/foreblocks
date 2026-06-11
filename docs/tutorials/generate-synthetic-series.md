---
title: Generate Synthetic Series
description: Use foretools/tsgen for testing models without external datasets.
editLink: true
---

# Generate Synthetic Series

`foretools/tsgen` provides a pedagogical synthetic time-series generator that is useful for:

- testing models without external datasets
- teaching decomposition concepts
- producing controlled trend/seasonality/noise examples
- validating visualization and preprocessing flows

## Main entry point

The generator lives in:

- `foretools/tsgen/ts_gen.py`

There is also a companion notebook:

- `foretools/tsgen/ts_gen_complete_series.ipynb`
- [Time Series Generator](../foretools/tsgen) documents the full API and component behavior

## Minimal example

```python
from foretools.tsgen import TimeSeriesGenerator

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
