# Documentation Overview

This repository has two related layers:

- `foreblocks`: the main forecasting library
- `foretools`: companion tooling for generation, analysis, and experimentation

If you are entering the codebase for the first time, use this page to decide where to start.

The documentation is now organized as a versioned wiki inside the repository:

- `wiki/index.md`: wiki home
- `wiki/tutorials/`: runnable workflows
- `wiki/architecture/`: subsystem and pipeline explanations
- `wiki/reference/`: stable API and repository reference
- root-level guide pages in `wiki/`: major subsystem guides

Published site layout:

- `/`: custom landing page
- `/docs/`: versioned documentation site

## Start Here

| Goal | Best entry point |
| --- | --- |
| Browse the docs like a wiki | [Docs Home](index.md) |
| Train a small model end to end | [Getting Started](getting-started.md) |
| Understand preprocessing and window creation | [Preprocessor Guide](preprocessor.md) |
| Inject custom modules into the forecasting stack | [Custom Blocks Guide](custom_blocks.md) |
| Work with transformer backbones | [Transformer Guide](transformer.md) |
| Enable mixture-of-experts routing | [MoE Guide](moe.md) |
| Run neural architecture search | [DARTS Guide](darts.md) |
| Unblock a failed setup or shape mismatch | [Troubleshooting](troubleshooting.md) |
| Generate synthetic time series | [Time Series Generator](foretools/tsgen.md) |
| Run budgeted hyperparameter search | [BOHB Search](foretools/bohb.md) |
| Decompose a signal into modes before modeling | [VMD Decomposition](foretools/vmd.md) |

## Stable Public Surface

The most reliable top-level imports today are:

```python
from foreblocks import (
    ForecastingModel,
    Trainer,
    ModelEvaluator,
    TimeSeriesHandler,
    TimeSeriesDataset,
    create_dataloaders,
    ModelConfig,
    TrainingConfig,
    LSTMEncoder,
    LSTMDecoder,
    GRUEncoder,
    GRUDecoder,
    TransformerEncoder,
    TransformerDecoder,
    AttentionLayer,
)
```

When documentation and internals diverge, prefer this exported surface first.

## How The Repo Is Organized

| Area | Purpose |
| --- | --- |
| `foreblocks/core` | forecasting model assembly, heads, conformal prediction |
| `foreblocks/training` | trainer, training loop, scheduler/optimizer integration |
| `foreblocks/evaluation` | evaluator, plotting, metrics |
| `foreblocks/ts_handler` | preprocessing, imputation, filtering, feature generation |
| `foreblocks/tf` | transformer stack, attention variants, norms, MoE |
| `foreblocks/darts` | architecture search, search configs, search evaluation |
| `foreblocks/mltracker` | experiment tracking UI and storage |
| `foretools/tsgen` | synthetic series generation and pedagogical notebooks |
| `foretools/bohb` | budgeted hyperparameter optimization with TPE-backed BOHB |
| `examples/` | notebooks demonstrating specific subsystems |

## Recommended Reading Order

1. [Docs Home](index.md)
2. [Getting Started](getting-started.md)
3. [Public API](reference/public-api.md)
4. [Preprocessor Guide](preprocessor.md)
5. [Custom Blocks Guide](custom_blocks.md)
6. [Transformer Guide](transformer.md)
7. [MoE Guide](moe.md) or [DARTS Guide](darts.md), depending on your use case
8. [Foretools Overview](foretools/index.md) if you also need synthetic data or search tooling

## Practical Notes

- The project is broad, so not every internal module should be treated as stable API.
- `ForecastingModel` plus `Trainer` is the main training path.
- `TimeSeriesHandler` is useful when you want the library to create windows and optional time features from a raw `[T, D]` array.
- `TimeSeriesDataset` and `create_dataloaders` are the simplest bridge from NumPy arrays into the trainer loop.
- `foretools` is not just internal support code. It contains useful standalone utilities, especially the synthetic time-series generator.
- The most mature `foretools` docs currently cover `tsgen` and `bohb`, which are the two main workflow-oriented tools in this repository.

## Suggested Next Steps

- If you want a runnable baseline, follow [Getting Started](getting-started.md).
- If you want synthetic data for demos or notebooks, read [Time Series Generator](foretools/tsgen.md).
- If you want to tune models or benchmark search behavior, read [BOHB Search](foretools/bohb.md).
- If you want signal decomposition or mode extraction, read [VMD Decomposition](foretools/vmd.md).
- If you are extending the library, read the topic guide closest to the subsystem you plan to modify.
