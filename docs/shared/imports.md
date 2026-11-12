# Shared Imports Reference

Common import patterns used throughout the foreBlocks documentation.
Examples are grouped by task.

## Core foreblocks

### Baseline

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
)
```

### Transformers

```python
from foreblocks import TransformerEncoder, TransformerDecoder
```

### Config only

```python
from foreblocks.training.config import TrainingConfig
from foreblocks.models.config import ModelConfig
```


See [Public API](../reference/public-api) for the full import surface.

## Transformer internals

### Schedules

```python
from foreblocks.nn.routing.mod import (
    LayerDropoutSchedule,
    MoDBudgetScheduler,
)
```

### MoE feedforward

```python
from foreblocks.nn.moe.feedforward import FeedForwardBlock
```

### Attention config

```python
from foreblocks.nn.transformer.config import TransformerConfig
from foreblocks.nn.attention.config import (
    AttentionConfig,
    AttentionShapeConfig,
    AttentionPositionConfig,
    AttentionVariantConfig,
)
```


## Uncertainty / Conformal

```python
from foreblocks.training.conformal import ConformalPredictionEngine
```

## Wavelet & frequency attention

Select via `attention=` on the transformer — no separate import needed:

```python
from foreblocks import TransformerEncoder

# Wavelet-domain (Haar DWT) attention
enc = TransformerEncoder(input_size=8, d_model=64, num_layers=2, attention="dwt")

# Frequency-domain (FEDformer-style) attention
enc = TransformerEncoder(input_size=8, d_model=64, num_layers=2, attention="frequency")
```

## DARTS (separate package)

DARTS is imported as a standalone package, not under `foreblocks`:

### Search

```python
from darts import DARTSTrainer
```

### Config

```python
from darts import DARTSTrainConfig
```


## foretools

### BOHB search

```python
from foretools.bohb import BOHB, PruningConfig, TPEConf
from foretools.bohb.plotter import OptimizationPlotter
from foretools.bohb.trial import TrialPruned
```

### Time series generator

```python
from foretools.tsgen import TimeSeriesGenerator
```

### VMD decomposition

```python
import numpy as np
from foretools.decomposition.emd import FastVMD
```

### AutoDA augmentation

```python
import torch
from foretools.tsaug import AutoDATimeseries, AutoDATrainer
```

### Feature engineering

```python
from foretools.fengineer import FeatureEngineer
from foretools.fengineer.transformers import FeatureConfig
```


## Install extras quick ref

### Core

```bash
pip install foreblocks
```

### DARTS search

```bash
pip install "foreblocks[darts]"
```

### VMD decomposition

```bash
pip install "foreblocks[vmd]"
```

### All extras

```bash
pip install "foreblocks[all]"
```


See [Getting Started](../getting-started) for the full install map and extras table.
