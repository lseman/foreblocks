# foreBlocks

[![PyPI Version](https://img.shields.io/pypi/v/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![Python Versions](https://img.shields.io/pypi/pyversions/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![License](https://img.shields.io/github/license/lseman/foreblocks)](LICENSE)

![ForeBlocks Logo](web/logo.svg#gh-light-mode-only)
![ForeBlocks Logo](web/logo_dark.svg#gh-dark-mode-only)

**foreBlocks** is a modular PyTorch toolkit for time-series forecasting, experiment management, and companion utilities.

This repository is structured as two cooperating packages:

- `foreblocks`: forecasting models, training, evaluation, preprocessing, DARTS search, and conformal uncertainty.
- `foretools`: companion utilities for synthetic data, feature engineering, decomposition, and hyperparameter search.

The recommended workflow is:

1. start with the stable top-level public API in `foreblocks`
2. validate one small training loop end to end
3. add preprocessing, search, or specialist tooling only when the baseline path works

## Install

This package requires Python 3.10 or newer.

### Core install

```bash
pip install foreblocks
```

### Optional extras

| Extra | Adds |
| --- | --- |
| `preprocessing` | `TimeSeriesHandler`, windowing, scaling, filtering, imputation, and time-feature generation |
| `darts` | DARTS architecture search, evaluation, and NAS dependencies |
| `mltracker` | experiment tracking API, local dashboard, and CLI TUI |
| `studio` | Studio frontend launcher and bundled server command |
| `vmd` | VMD decomposition, search support, and analysis helpers |
| `wavelets` | wavelet preprocessing and attention-head utilities |
| `benchmark` | external forecasting baselines and spreadsheet readers |
| `foreminer` | changepoint detection, dataset mining, and analysis utilities |
| `all` | all runtime extras above |

Examples:

```bash
pip install "foreblocks[darts]"
pip install "foreblocks[mltracker]"
pip install "foreblocks[studio]"
pip install "foreblocks[vmd,wavelets]"
pip install "foreblocks[all]"
```

### Local development install

```bash
git clone https://github.com/lseman/foreblocks.git
cd foreblocks
pip install -e ".[dev]"
```

### Launch the Studio frontend

```bash
foreblocks-studio
```

By default, this opens a browser on `127.0.0.1` or `localhost`.

Optional flags:

```bash
foreblocks-studio --open
foreblocks-studio --no-open
foreblocks-studio --host 0.0.0.0 --port 8080
```

## Quickstart

The smallest reliable path is a direct forecasting model with a custom head. This path avoids extra dependencies and verifies that the public API is wired correctly.

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

print(history.train_losses[-1], metrics)
```

### Why this path

- validates that the import surface works
- checks dataloader shapes and model output sizes
- avoids optional subsystems during the first run
- keeps the first success criterion small and confirmable

If you start from a raw `[T, D]` time series instead of windows, use `TimeSeriesHandler` from `foreblocks.ts_handler` after installing `foreblocks[preprocessing]`.

## Public API

The most stable first imports are exposed from the top-level `foreblocks` package:

| Import | Purpose |
| --- | --- |
| `ForecastingModel` | Core forecasting wrapper for direct, autoregressive, and seq2seq-style models |
| `Trainer` | Training loop with NAS hooks, MLTracker integration, and optional conformal support |
| `ModelEvaluator` | Prediction helpers, metrics, cross-validation, and training-curve plots |
| `TimeSeriesHandler` | Raw-series preprocessing, windowing, scaling, and imputation bridge |
| `TimeSeriesDataset` | Dataset wrapper used by the dataloader helper |
| `create_dataloaders` | Build train/validation PyTorch dataloaders from NumPy arrays |
| `ModelConfig`, `TrainingConfig` | Lightweight configuration dataclasses |
| `LSTMEncoder`, `LSTMDecoder`, `GRUEncoder`, `GRUDecoder` | Recurrent encoder/decoder blocks |
| `TransformerEncoder`, `TransformerDecoder` | Transformer backbones and related advanced features |
| `AttentionLayer` | Attention module entry point |

## Repository map

| Path | What it contains |
| --- | --- |
| `foreblocks/core` | `ForecastingModel`, heads, conformal utilities, sampling |
| `foreblocks/training` | `Trainer`, training loop, quantization utilities |
| `foreblocks/evaluation` | `ModelEvaluator`, benchmarking helpers |
| `foreblocks/ts_handler` | `TimeSeriesHandler`, imputation, filtering, outlier handling |
| `foreblocks/tf` | transformer stack, attention variants, MoE, norms, embeddings |
| `foreblocks/darts` | neural architecture search pipeline and evaluation |
| `foreblocks/mltracker` | experiment tracking server, logging, and TUI integration |
| `foretools` | synthetic time series, BOHB search, feature engineering, decomposition |
| `examples/` | runnable demos and notebooks |
| `web/` | static landing page assets for the published site root |
| `docs/` | VitePress source for the documentation site |

## Documentation map

Start here if you are new to the repository:

- [Documentation Overview](docs/overview.md)
- [Getting Started](docs/getting-started.md)
- [Docs home](docs/index.md)

Topic guides:

- [Preprocessor Guide](docs/preprocessor.md)
- [Custom Blocks Guide](docs/custom_blocks.md)
- [Transformer Guide](docs/transformer.md)
- [MoE Guide](docs/moe.md)
- [DARTS Guide](docs/darts.md)
- [Troubleshooting](docs/troubleshooting.md)

Companion tooling:

- [Foretools Overview](docs/foretools/index.md)
- [Time Series Generator](docs/foretools/tsgen.md)
- [BOHB Search](docs/foretools/bohb.md)
- [VMD Decomposition](docs/foretools/vmd.md)

Examples and notebooks:

- `examples/adaptive_mrmr_demo.py`
- `foretools/tsgen/ts_gen_complete_series.ipynb`
- `foretools/tsgen/ts_gen_doc.ipynb`
- `examples/`

There is a repository-local docs navigation file at [`docs/.vitepress/config.js`](docs/.vitepress/config.js).

## Current project status

- The repository is broad and still evolving. Some subsystems are more mature than others.
- The top-level imports listed above are the safest place to start.
- `Trainer` supports MLTracker and conformal prediction; use `auto_track=False` during local smoke tests.
- Decoder-based seq2seq and transformer workflows have stricter dimension contracts than the direct forecasting path.
- `TrainingConfig` now centralizes trainer, NAS, MLTracker, and conformal settings.

## Contributing

Documentation improvements are especially valuable here because `foreblocks` spans forecasting models, search, preprocessing, and auxiliary tooling. If you add or change a public API, update:

1. this `README.md`
2. the relevant guide under `docs/`
3. at least one runnable example or notebook
