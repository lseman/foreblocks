---
title: Public API
description: Stable import surface — ForecastingModel, Trainer, ModelEvaluator, and more.
editLink: true
---


[[toc]]
# Public API

This page documents the main top-level imports exposed by `foreblocks`.

## Recommended import surface

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
    GraphForecastingModel,
    TransformerTuner,
)
```

`foreblocks/__init__.py` lazy-loads (`__getattr__` + `TYPE_CHECKING`) this stable,
deliberately small `__all__`. Anything not in it is an internal import path and
can move without a deprecation cycle; anything in it moving is a breaking change.

## Resolution targets

Where each export actually lives — useful when the name alone doesn't tell you
the subpackage:

| Export | Resolves to |
| --- | --- |
| `AttentionLayer` | `foreblocks.nn.attention.layer` |
| `ForecastingModel`, `GraphForecastingModel` | `foreblocks.models` |
| `Trainer` | `foreblocks.training` |
| `ModelEvaluator` | `foreblocks.evaluation` |
| `TimeSeriesHandler` | `foreblocks.processing` |
| `TimeSeriesDataset`, `create_dataloaders` | `foreblocks.data` |
| `ModelConfig`, `TrainingConfig` | `foreblocks.config` |
| `LSTMEncoder`, `LSTMDecoder`, `GRUEncoder`, `GRUDecoder` | `foreblocks.nn.blocks.recurrent` |
| `TransformerEncoder` | `foreblocks.nn.transformer.encoder` |
| `TransformerDecoder` | `foreblocks.nn.transformer.decoder` |
| `TransformerTuner` | `foreblocks.tuning.transformer` |
