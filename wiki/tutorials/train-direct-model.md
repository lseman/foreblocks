# Train A Direct Model

This tutorial is the shortest end-to-end path through `foreblocks`.

It uses:

- NumPy arrays as the data source
- `create_dataloaders(...)` for batching
- `ForecastingModel(..., forecasting_strategy="direct")`
- `Trainer` for optimization
- `ModelEvaluator` for basic metrics

## Why start here

The direct strategy is the least opinionated path in the current codebase. It avoids decoder-shape contracts and is therefore the best first integration target.

## Example

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

## Expected shapes

- `X_train`: `[N, T, F]`
- `y_train`: any shape compatible with the head output
- head output: should match the target tensor passed to the trainer

In this example the head returns `[N, horizon]`, so `y_train` is also `[N, horizon]`.

## What to change next

- replace the MLP head with a larger direct projection
- switch to `TimeSeriesPreprocessor` if your source data starts as raw `[T, D]`
- move to encoder/decoder or transformer-based strategies after this baseline works

## Related pages

- [Getting Started](../getting-started.md)
- [Public API](../reference/public-api.md)
- [Forecasting Pipeline](../architecture/forecasting-pipeline.md)
