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

This project targets Python 3.10 and newer.

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

| Extra | Adds |
| --- | --- |
| `preprocessing` | `TimeSeriesHandler`, windowing, scaling, filtering, imputation |
| `darts` | Architecture search, NAS, and evaluation helpers |
| `mltracker` | Experiment tracking API, local dashboard, and CLI TUI |
| `studio` | Studio frontend launcher and bundled server command |
| `vmd` | VMD decomposition and analysis helpers |
| `wavelets` | Wavelet preprocessing and attention utilities |
| `benchmark` | External forecasting baselines and spreadsheet readers |
| `foreminer` | Changepoint detection and dataset mining |
| `all` | All runtime extras |

### Install examples

```bash
# DARTS only
pip install "foreblocks[darts]"

# Multiple extras
pip install "foreblocks[vmd,wavelets]"

# Everything (large)
pip install "foreblocks[all]"
```

::: note Documentation site
Full guides, API reference, and examples: [https://foreblocks.laioseman.com/](https://foreblocks.laioseman.com/)
:::

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

## 3. Validate the import surface first

Run a quick import check before the full example:

```bash
python -c "from foreblocks import ForecastingModel, Trainer; print('foreblocks import OK')"
```

If this fails, verify your Python environment and installed extras.

## 4. Minimal training example

This example trains a direct forecasting model with a simple custom head.

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

# === Configuration ===
# Shapes: X = [N, T, F], y = [N, H]
seq_len = 24    # input sequence length
horizon = 6     # forecast horizon
n_features = 4  # number of input features
batch_size = 16

# === Generate synthetic data ===
rng = np.random.default_rng(0)
X_train = rng.normal(size=(64, seq_len, n_features)).astype("float32")
y_train = rng.normal(size=(64, horizon)).astype("float32")
X_val = rng.normal(size=(16, seq_len, n_features)).astype("float32")
y_val = rng.normal(size=(16, horizon)).astype("float32")

# === Build dataloaders ===
train_loader, val_loader = create_dataloaders(
    X_train, y_train, X_val, y_val, batch_size=batch_size,
)

# === Define a simple head ===
head = nn.Sequential(
    nn.Flatten(),
    nn.Linear(seq_len * n_features, 64),
    nn.GELU(),
    nn.Linear(64, horizon),
)

# === Assemble model ===
model = ForecastingModel(
    head=head,
    forecasting_strategy="direct",
    model_type="head_only",
    target_len=horizon,
)

# === Train ===
trainer = Trainer(
    model,
    config=TrainingConfig(
        num_epochs=5,
        batch_size=batch_size,
        patience=3,
        use_amp=False,
    ),
    auto_track=False,
)

history = trainer.train(train_loader, val_loader)

# === Evaluate ===
evaluator = ModelEvaluator(trainer)
metrics = evaluator.compute_metrics(torch.tensor(X_val), torch.tensor(y_val))

print(f"Final training loss: {history.train_losses[-1]:.4f}")
print(f"Metrics: {metrics}")
```

### What this validates

- The import path works correctly
- Dataloader shapes match the trainer expectations
- The model trains without optional subsystems
- Evaluation works on held-out data

## 5. Trainer and MLTracker notes

`Trainer` initializes MLTracker automatically if installed. Pass `auto_track=False` during local smoke tests.

## 6. Shape expectations

### Direct forecasting

| Tensor | Shape | Description |
| --- | --- | --- |
| `X` | `[N, T, F]` | Samples × input timesteps × features |
| `y` | `[N, H]` | Samples × horizon |

### Encoder / decoder (seq2seq)

| Tensor | Shape | Description |
| --- | --- | --- |
| `X` | `[N, T, F]` | Samples × input timesteps × features |
| `y` | `[N, H, D]` | Samples × horizon × output channels |

Decoder-based models have stricter dimension contracts. Read the [Custom Blocks](custom_blocks.md) guide before wiring custom modules.

## 7. Starting from raw time series

When your starting point is a single `[T, D]` array, use `TimeSeriesHandler` instead of building windows manually:

```python
import numpy as np
import pandas as pd
from foreblocks import TimeSeriesHandler

# Load raw data
raw = np.random.randn(240, 3)  # [T, F]
timestamps = pd.date_range("2025-01-01", periods=len(raw), freq="h")

# Configure preprocessing
pre = TimeSeriesHandler(
    window_size=24,   # sequence length
    horizon=6,        # forecast horizon
    normalize=True,   # standardize features
    generate_time_features=False,
    verbose=False,
)

# Fit and transform
X, y, processed, time_feats = pre.fit_transform(raw, time_stamps=timestamps)

# Transform validation data using fitted state
X_val = pre.transform(val_raw, time_stamps=val_timestamps)
```

::: tip Extra required
Install `foreblocks[preprocessing]` to get `TimeSeriesHandler`:

```bash
pip install "foreblocks[preprocessing]"
```
:::

Continue with [Preprocessor Guide](preprocessor.md) for advanced options like filtering, outlier handling, and feature engineering.

## 8. When to add DARTS

If the basic training loop works and you want architecture search instead of hand-selecting blocks:

```bash
pip install "foreblocks[darts]"
```

Then continue with:

- [DARTS Guide](darts.md)
- [Run A DARTS Search](tutorials/darts-multifidelity-search.md)
- [DARTS Search Pipeline](architecture/darts-pipeline.md)

## 9. Where to go next

Use this decision tree to pick the next page based on your goals:

```
Baseline works and metrics are acceptable?
├── No → improve your data
│   ├── Raw series (single array)   → Preprocessor Guide
│   └── Feature engineering        → Feature Engineering
│
├── Yes, but I want better accuracy
│   ├── Try a stronger backbone     → Transformer Guide
│   ├── Add MoE feedforward         → MoE Guide
│   ├── Add preprocessing heads     → Custom Blocks Guide
│   └── Search architectures        → DARTS Guide
│
├── Yes, but I need uncertainty
│   └── Post-hoc conformal intervals → Uncertainty Guide
│
├── Yes, but I need richer evaluation
│   └── Metrics, plots, CV           → Evaluation Guide
│
└── Yes, but training is slow or OOM
    └── AMP, gradient checkpointing  → Configuration Reference
```

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
    <p>Import issues, install problems, and shape mismatches in a first run.</p>
  </div>
</div>

## Notes

- `Trainer` initializes MLTracker automatically if installed. Pass `auto_track=False` for local smoke tests.
- `TrainingConfig` includes conformal and NAS options, but you do not need them for the baseline path.
- `TimeSeriesDataset` is available if you want to build PyTorch dataloaders manually.
- The direct strategy is still the best first step. Move to seq2seq, transformers, or DARTS after the baseline succeeds.

::: note Documentation site
Full guides, API reference, and examples: [https://foreblocks.laioseman.com/](https://foreblocks.laioseman.com/)
:::

## Public API quick reference

```python
from foreblocks import (
    ForecastingModel,   # Core model wrapper
    Trainer,            # Training loop
    ModelEvaluator,     # Evaluation and metrics
    TimeSeriesHandler,  # Preprocessing (requires [preprocessing] extra)
    TimeSeriesDataset,  # Dataset wrapper
    create_dataloaders, # Dataloader helper
    ModelConfig,        # Model configuration
    TrainingConfig,     # Training configuration
)
```

## Related pages

- [Overview](overview.md) - Mental model of the stack
- [Public API](reference/public-api.md) - Complete API reference
- [Configuration](reference/configuration.md) - All configuration options
- [Troubleshooting](troubleshooting.md) - Common issues and fixes
