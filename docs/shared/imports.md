# Shared Imports Reference

Common import patterns used throughout the foreBlocks documentation.
Each block uses VitePress `::: code-group` tabbed syntax.

## Core foreblocks

::: code-group

```python [Baseline]
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

```python [Transformers]
from foreblocks import TransformerEncoder, TransformerDecoder
```

```python [Config only]
from foreblocks.training.config import TrainingConfig
from foreblocks.models.config import ModelConfig
```

:::

See [Public API](../reference/public-api) for the full import surface.

## Transformer internals

::: code-group

```python [Schedules]
from foreblocks.nn.routing.mod import (
    LayerDropoutSchedule,
    MoDBudgetScheduler,
)
```

```python [MoE feedforward]
from foreblocks.nn.moe.feedforward import FeedForwardBlock
```

```python [Attention config]
from foreblocks.nn.transformer.config import TransformerConfig
from foreblocks.nn.attention.config import (
    AttentionConfig,
    AttentionShapeConfig,
    AttentionPositionConfig,
    AttentionVariantConfig,
)
```

:::

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

::: code-group

```python [Search]
from darts import DARTSTrainer
```

```python [Config]
from darts import DARTSTrainConfig
```

:::

## foretools

::: code-group

```python [BOHB search]
from foretools.bohb import BOHB, PruningConfig, TPEConf
from foretools.bohb.plotter import OptimizationPlotter
from foretools.bohb.trial import TrialPruned
```

```python [Time series generator]
from foretools.tsgen import TimeSeriesGenerator
```

```python [VMD decomposition]
import numpy as np
from foretools.decomposition.emd import FastVMD
```

```python [AutoDA augmentation]
import torch
from foretools.tsaug import AutoDATimeseries, AutoDATrainer
```

```python [Feature engineering]
from foretools.fengineer import FeatureEngineer
from foretools.fengineer.transformers import FeatureConfig
```

:::

## Install extras quick ref

::: code-group

```bash [Core]
pip install foreblocks
```

```bash [DARTS search]
pip install "foreblocks[darts]"
```

```bash [VMD decomposition]
pip install "foreblocks[vmd]"
```

```bash [All extras]
pip install "foreblocks[all]"
```

:::

See [Getting Started](../getting-started) for the full install map and extras table.
