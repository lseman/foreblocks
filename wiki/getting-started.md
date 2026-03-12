# Getting Started

This guide gives you the shortest reliable path to a working `foreblocks` training loop, then points you to the subsystem-specific guides.

If you want the broader documentation map first, start from [Docs Home](index.md).

## Install

```bash
pip install foreblocks
```

For development:

```bash
git clone https://github.com/lseman/foreblocks.git
cd foreblocks
pip install -e ".[dev]"
```

## Minimal Training Example

The direct forecasting strategy is the simplest way to verify your environment and understand the library flow.

```python
import numpy as np
import torch
import torch.nn as nn

from foreblocks import (
    ForecastingModel,
    ModelEvaluator,
    Trainer,
    TrainingConfig,
    create_dataloaders,
)

seq_len = 24
horizon = 6
n_features = 4

rng = np.random.default_rng(0)
X_train = rng.normal(size=(64, seq_len, n_features)).astype("float32")
y_train = rng.normal(size=(64, horizon)).astype("float32")
X_val = rng.normal(size=(16, seq_len, n_features)).astype("float32")
y_val = rng.normal(size=(16, horizon)).astype("float32")

train_loader, val_loader = create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=16,
)

head = nn.Sequential(
    nn.Flatten(),
    nn.Linear(seq_len * n_features, 64),
    nn.GELU(),
    nn.Linear(64, horizon),
)

model = ForecastingModel(
    head=head,
    forecasting_strategy="direct",
    model_type="head_only",
    target_len=horizon,
)

trainer = Trainer(
    model,
    config=TrainingConfig(
        num_epochs=5,
        batch_size=16,
        patience=3,
        use_amp=False,
    ),
    auto_track=False,
)

history = trainer.train(train_loader, val_loader)

evaluator = ModelEvaluator(trainer)
metrics = evaluator.compute_metrics(torch.tensor(X_val), torch.tensor(y_val))

print("final_train_loss:", history.train_losses[-1])
print("metrics:", metrics)
```

What this establishes:

- your `foreblocks` import path is correct
- dataloaders are shaped correctly
- the trainer loop runs
- evaluation works on held-out data

## Shape Expectations

For the minimal direct path above:

- `X`: `[N, T, F]`
- `y`: any shape that matches the output of your direct head

For encoder/decoder forecasting:

- inputs are usually `[N, T, F]`
- targets are typically `[N, H, D]`
- decoder-based models have stricter dimension contracts, so use the topic guides before customizing them heavily

## Using The Preprocessor

When your starting point is a raw multivariate series shaped `[T, D]`, `TimeSeriesPreprocessor` can handle transforms and window creation for you.

```python
import numpy as np
from foreblocks import TimeSeriesPreprocessor

raw = np.random.randn(240, 3)

pre = TimeSeriesPreprocessor(
    window_size=24,
    horizon=6,
    normalize=True,
    generate_time_features=False,
    verbose=False,
)

X, y, processed, time_feat = pre.fit_transform(raw)
X_next = pre.transform(raw)

print(X.shape, y.shape, processed.shape, time_feat)
print(X_next.shape)
```

Use the preprocessor guide for the full transform/filter/imputation options:

- [Preprocessor Guide](preprocessor.md)

## Where To Go Next

- For the full documentation map: [Docs Home](index.md)
- For model composition and injection points: [Custom Blocks Guide](custom_blocks.md)
- For transformer backbones and patching: [Transformer Guide](transformer.md)
- For mixture-of-experts routing: [MoE Guide](moe.md)
- For architecture search: [DARTS Guide](darts.md)
- For repository orientation: [Documentation Overview](overview.md)

## Notes On Current Behavior

- `Trainer` can initialize MLTracker automatically; pass `auto_track=False` during local smoke tests if you only want the training loop.
- `TrainingConfig` now includes conformal options, but you do not need them for the basic path above.
- The direct strategy is the best first step. Move to seq2seq or transformer workflows once the basic loop is already running.
