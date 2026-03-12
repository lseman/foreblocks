# Foretools Overview

`foretools` is the companion toolbox that sits next to `foreblocks`.

Use `foreblocks` when you are building and training forecasting models. Use `foretools` when you need support utilities around that workflow: synthetic data, black-box search, exploratory diagnostics, decomposition, or feature engineering.

## Best-documented tools

| Tool | When to use it | Docs |
| --- | --- | --- |
| `foretools/tsgen` | create synthetic series with known structure and ground-truth components | [Time Series Generator](tsgen.md) |
| `foretools/bohb` | run budgeted hyperparameter optimization with Hyperband + TPE | [BOHB Search](bohb.md) |

## Other foretools areas

| Path | Purpose |
| --- | --- |
| `foretools/foreminer` | exploratory analysis and diagnostics |
| `foretools/fengineer` | feature engineering utilities |
| `foretools/vmd` | decomposition tools |
| `foretools/foraug` | augmentation-oriented utilities |

## How `foretools` fits the repo

- `foreblocks` is the main model and training API.
- `foretools` is a set of practical companion modules. Some are notebook-oriented and some are reusable library code.
- `foretools` imports are deeper and less consolidated than `foreblocks`, so the safest entry points are the specific modules documented here.

## Recommended reading

1. [Time Series Generator](tsgen.md) if you need synthetic datasets or decomposition examples.
2. [BOHB Search](bohb.md) if you need hyperparameter optimization outside the `foreblocks.darts` neural architecture search stack.
3. [Repository Map](../reference/repository-map.md) if you want the broader code layout.

## Related pages

- [Generate Synthetic Series](../tutorials/generate-synthetic-series.md)
- [Optimize With BOHB](../tutorials/optimize-with-bohb.md)
- [Overview](../overview.md)
