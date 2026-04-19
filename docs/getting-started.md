# Getting Started

This guide gives you the shortest reliable path to a working `foreblocks` training loop, then shows where to branch when you need preprocessing, search, or richer tooling.

If you want the broader mental model first, start from [Overview](overview.md).

<div class="callout-grid">
  <div class="glass-card">
    <strong>Goal</strong>
    <span>Run one small training job end to end with the stable public API.</span>
  </div>
  <div class="glass-card">
    <strong>Good first success</strong>
    <span>You can import the package, train a baseline model, and compute metrics on held-out data.</span>
  </div>
  <div class="glass-card">
    <strong>Only then</strong>
    <span>Add preprocessing extras, custom transformer internals, DARTS, or tracking-heavy workflows.</span>
  </div>
</div>

## 1. Install

### PyPI (stable)

```bash
pip install foreblocks
```

### Editable / dev

```bash
git clone https://github.com/lseman/foreblocks.git
cd foreblocks
pip install -e ".[dev]"
```

### With extras

| Workflow | Install |
| --- | --- |
| Plotting helpers | `pip install "foreblocks[plotting]"` |
| Raw-series preprocessing and scientific utilities | `pip install "foreblocks[preprocessing]"` |
| DARTS training, search, and analysis | `pip install "foreblocks[darts]"` |
| MLTracker API and TUI | `pip install "foreblocks[mltracker]"` |
| Everything | `pip install "foreblocks[all]"` |

## 2. Keep the first run intentionally small

```mermaid
flowchart LR
    A[NumPy arrays] --> B[create_dataloaders]
    B --> C[ForecastingModel]
    C --> D[Trainer.train]
    D --> E[ModelEvaluator]
    E --> F[metrics]
```

::: info What this first run should validate
- Your `foreblocks` import path is correct.
- Dataloader shapes line up with the trainer.
- The model trains without needing optional subsystems.
- Evaluation works on held-out data.
:::

## 3. Minimal training example

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
X_val   = rng.normal(size=(16, seq_len, n_features)).astype("float32")
y_val   = rng.normal(size=(16, horizon)).astype("float32")

train_loader, val_loader = create_dataloaders(
    X_train, y_train, X_val, y_val, batch_size=16
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
    config=TrainingConfig(num_epochs=5, batch_size=16, patience=3, use_amp=False),
    auto_track=False,  # disable MLTracker during smoke tests
)

history = trainer.train(train_loader, val_loader)

evaluator = ModelEvaluator(trainer)
metrics = evaluator.compute_metrics(torch.tensor(X_val), torch.tensor(y_val))

print("final_train_loss:", history.train_losses[-1])
print("metrics:", metrics)
```

## 4. Shape expectations

### Direct forecasting

| Tensor | Shape |
| --- | --- |
| `X` | `[N, T, F]` - samples x input timesteps x features |
| `y` | any shape matching your head's output |

### Encoder / decoder

| Tensor | Shape |
| --- | --- |
| `X` | `[N, T, F]` |
| `y` | `[N, H, D]` - samples x horizon x output channels |

Decoder-based models have stricter dimension contracts. Read the [Custom Blocks](custom_blocks.md) guide before wiring custom modules.

## 5. Starting from raw series instead of windows

When your starting point is a single `[T, D]` array, use `TimeSeriesHandler` instead of building windows manually:

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
```

::: tip Extra required
```bash
pip install "foreblocks[preprocessing]"
```
:::

Continue with [Preprocessor Guide](preprocessor.md) once the baseline path itself is working.

## 6. When to add DARTS

If the basic training loop works and you want architecture search instead of hand-selecting blocks:

```bash
pip install "foreblocks[darts]"
```

Then continue with:

- [DARTS Guide](darts.md)
- [Run A DARTS Search](tutorials/darts-multifidelity-search.md)
- [DARTS Search Pipeline](architecture/darts-pipeline.md)

## 7. Where to go next

<div class="path-grid">
  <div class="path-card">
    <p class="route-kicker">Next</p>
    <h3><a href="preprocessor/">Raw series preprocessing</a></h3>
    <p>Scaling, filtering, window generation, and feature engineering from multivariate arrays.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Next</p>
    <h3><a href="custom_blocks/">Model composition</a></h3>
    <p>Injection points, block composition, and dimension contracts for custom architectures.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Next</p>
    <h3><a href="transformer/">Transformer backbones</a></h3>
    <p>Attention, patching, norms, and the configurable transformer stack.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Next</p>
    <h3><a href="evaluation/">Evaluation and metrics</a></h3>
    <p>Metrics, plots, cross-validation, and evaluation helpers after training.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Next</p>
    <h3><a href="uncertainty/">Uncertainty quantification</a></h3>
    <p>Conformal intervals and post-hoc uncertainty methods once point forecasting is in place.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Safety net</p>
    <h3><a href="troubleshooting/">Troubleshooting</a></h3>
    <p>Import issues, install problems, and the shape mismatches most likely to show up in a first run.</p>
  </div>
</div>

## Notes

- `Trainer` initializes MLTracker automatically. Pass `auto_track=False` during local smoke tests.
- `TrainingConfig` includes conformal options, but you do not need them for the baseline path above.
- `TimeSeriesDataset` is available if you want to build PyTorch dataloaders manually.
- The direct strategy is still the best first step. Move to seq2seq, transformers, or DARTS after the small baseline loop already succeeds.
