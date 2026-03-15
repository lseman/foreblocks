# Getting Started

This guide gives you the shortest reliable path to a working `foreblocks` training loop, then points you to the right extra packages and guides when you want preprocessing, search, or richer tooling.

If you want the broader map first, start from [Docs Home](index.md) or [Overview](overview.md).

## Install

Base install:

```bash
pip install foreblocks
```

Optional extras:

| Workflow | Install |
| --- | --- |
| plotting helpers only | `pip install "foreblocks[plotting]"` |
| preprocessing / scientific stack | `pip install "foreblocks[preprocessing]"` |
| DARTS training + search flow | `pip install "foreblocks[darts]"` |
| DARTS analyzer with pandas/seaborn | `pip install "foreblocks[darts-analysis]"` |
| MLTracker API + TUI | `pip install "foreblocks[mltracker]"` |
| all runtime extras | `pip install "foreblocks[all]"` |

For development:

```bash
git clone https://github.com/lseman/foreblocks.git
cd foreblocks
pip install -e ".[dev]"
```

## Minimal training example

The direct forecasting strategy is still the best way to verify your environment and learn the library flow.

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

## Shape expectations

For the direct path above:

- `X`: `[N, T, F]`
- `y`: any shape that matches the output of your direct head

For encoder/decoder forecasting:

- inputs are usually `[N, T, F]`
- targets are typically `[N, H, D]`
- decoder-based models have stricter dimension contracts, so use the topic guides before customizing them heavily

## Starting from raw multivariate data

When your starting point is a raw series shaped `[T, D]`, use `TimeSeriesHandler` instead of building windows manually.

```python
import numpy as np
from foreblocks import TimeSeriesHandler

raw = np.random.randn(240, 3)

pre = TimeSeriesHandler(
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

Install the needed scientific stack first if you plan to use preprocessing features:

```bash
pip install "foreblocks[preprocessing]"
```

## Moving into DARTS search

If the core training path works and you want to search over architectures instead of hand-picking them:

```bash
pip install "foreblocks[darts]"
```

Then start from:

- [DARTS Guide](darts.md)
- [Run A DARTS Search](tutorials/darts-multifidelity-search.md)
- [DARTS Search Pipeline](architecture/darts-pipeline.md)

If you also want the result analyzer and richer plots:

```bash
pip install "foreblocks[darts-analysis]"
```

## Where to go next

- For preprocessing and window creation: [Preprocessor Guide](preprocessor.md)
- For model composition and injection points: [Custom Blocks Guide](custom_blocks.md)
- For transformer backbones and patching: [Transformer Guide](transformer.md)
- For expert routing: [MoE Guide](moe.md)
- For neural architecture search: [DARTS Guide](darts.md)
- For setup problems and import mismatches: [Troubleshooting](troubleshooting.md)

## Notes on current behavior

- `Trainer` can initialize MLTracker automatically. Pass `auto_track=False` during local smoke tests if you only want the training loop.
- `TrainingConfig` includes conformal options, but you do not need them for the basic path above.
- `TimeSeriesDataset` is also available if you want to build PyTorch dataloaders manually instead of using `create_dataloaders(...)`.
- The direct strategy is still the best first step. Move to seq2seq, transformer, or DARTS workflows once the baseline loop is already running.
