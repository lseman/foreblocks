# Public API

This page documents the main top-level imports exposed by `foreblocks`.

## Recommended import surface

```python
from foreblocks import (
    ForecastingModel,
    Trainer,
    ModelEvaluator,
    TimeSeriesPreprocessor,
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

## Core classes

### `ForecastingModel`

The core forecasting wrapper. It supports:

- `direct`
- `autoregressive`
- `seq2seq`
- `transformer_seq2seq`

Typical constructor roles:

- `encoder` and `decoder` for recurrent or transformer workflows
- `head` for direct forecasting
- preprocessing and normalization injection points
- optional attention module

### `Trainer`

The main training orchestrator. It handles:

- training and validation loops
- early stopping
- scheduler stepping
- optional MLTracker integration
- optional NAS-aware training
- optional conformal support

### `ModelEvaluator`

Post-training utility for:

- batch prediction
- metrics computation
- rolling cross-validation
- training-curve visualization

### `TimeSeriesPreprocessor`

Preprocessing and windowing pipeline for raw multivariate series.

Use it when your raw data starts as `[T, D]` and you want the library to build training windows.

## Model blocks

### Recurrent blocks

- `LSTMEncoder`
- `LSTMDecoder`
- `GRUEncoder`
- `GRUDecoder`

### Transformer blocks

- `TransformerEncoder`
- `TransformerDecoder`
- `AttentionLayer`

## Utility exports

### `create_dataloaders`

Builds train and validation PyTorch dataloaders from NumPy arrays.

### `TimeSeriesDataset`

Dataset wrapper used by the dataloader helper.

### `ModelConfig` and `TrainingConfig`

Dataclasses for model-level and training-level configuration.

## Guidance

- Prefer top-level imports unless you are modifying internals.
- Treat deep imports as implementation details unless a subsystem guide explicitly recommends them.
- `foretools` does not currently expose an equally stable top-level import surface. For those utilities, prefer the documented deep imports in the dedicated `Foretools` pages.

## Related pages

- [Configuration](configuration.md)
- [Repository Map](repository-map.md)
- [Overview](../overview.md)
- [Foretools Overview](../foretools/index.md)
