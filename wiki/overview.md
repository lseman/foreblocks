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
| Generate synthetic time series | `foretools/tsgen/ts_gen.py` and `foretools/tsgen/ts_gen_complete_series.ipynb` |

## Stable Public Surface

The most reliable top-level imports today are:

```python
from foreblocks import (
    ForecastingModel,
    Trainer,
    ModelEvaluator,
    TimeSeriesPreprocessor,
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
| `foreblocks/pre` | preprocessing, imputation, filtering, feature generation |
| `foreblocks/tf` | transformer stack, attention variants, norms, MoE |
| `foreblocks/darts` | architecture search, search configs, search evaluation |
| `foreblocks/mltracker` | experiment tracking UI and storage |
| `foretools/tsgen` | synthetic series generation and pedagogical notebooks |
| `examples/` | notebooks demonstrating specific subsystems |

## Recommended Reading Order

1. [Docs Home](index.md)
2. [Getting Started](getting-started.md)
3. [Public API](reference/public-api.md)
4. [Preprocessor Guide](preprocessor.md)
5. [Custom Blocks Guide](custom_blocks.md)
6. [Transformer Guide](transformer.md)
7. [MoE Guide](moe.md) or [DARTS Guide](darts.md), depending on your use case

## Practical Notes

- The project is broad, so not every internal module should be treated as stable API.
- `ForecastingModel` plus `Trainer` is the main training path.
- `TimeSeriesPreprocessor` is useful when you want the library to create windows and optional time features from a raw `[T, D]` array.
- `foretools` is not just internal support code. It contains useful standalone utilities, especially the synthetic time-series generator.

## Suggested Next Steps

- If you want a runnable baseline, follow [Getting Started](getting-started.md).
- If you want synthetic data for demos or notebooks, open `foretools/tsgen/ts_gen_complete_series.ipynb`.
- If you are extending the library, read the topic guide closest to the subsystem you plan to modify.
