# foreBlocks

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lseman/foreblocks/main/web/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/lseman/foreblocks/main/web/logo.svg">
  <img alt="ForeBlocks Logo" src="https://raw.githubusercontent.com/lseman/foreblocks/main/web/logo.svg">
</picture>

[![PyPI Version](https://img.shields.io/pypi/v/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![Python Versions](https://img.shields.io/pypi/pyversions/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![License](https://img.shields.io/github/license/lseman/foreblocks)](LICENSE)

**foreBlocks** is a modular PyTorch toolkit for time-series forecasting, experiment management, and companion utilities, including configurable transformer backbones with modern attention variants such as MoBA.

This repository is structured as three cooperating packages, all distributed in the same `foreblocks` wheel:

- `foreblocks`: forecasting models, training, evaluation, preprocessing, and conformal uncertainty.
- `darts`: a standalone neural architecture search package for time-series forecasting (DARTS-style differentiable NAS), imported as `import darts`.
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
| `darts` | dependencies for the standalone `darts` NAS package: search, evaluation, and analysis |
| `mltracker` | experiment tracking API, local dashboard, and CLI TUI |
| `studio` | Studio frontend launcher and bundled server command |
| `vmd` | VMD decomposition, search support, and analysis helpers |
| `wavelets` | wavelet preprocessing and multiwavelet feature extraction |
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

## Documentation

For detailed guides, examples, and API reference:

- [Getting Started](docs/getting-started.md) - Quickstart with a minimal training loop
- [Overview](docs/overview.md) - Architecture and mental model
- [Public API](docs/reference/public-api.md) - Stable import surface
- [DARTS Guide](docs/darts.md) - Architecture search
- [Preprocessor Guide](docs/preprocessor.md) - Raw series handling

Full documentation: [https://foreblocks.laioseman.com/docs/](https://foreblocks.laioseman.com/docs/)

### Documentation site structure

```
docs/           - VitePress source for the documentation site
web/            - Static landing page assets for the published site
examples/       - Runnable demos and notebooks
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

### Why this path

- validates that the import surface works
- checks dataloader shapes and model output sizes
- avoids optional subsystems during the first run
- keeps the first success criterion small and confirmable

### From raw time series

If you start from a raw `[T, D]` array instead of pre-built windows, use `TimeSeriesHandler` after installing `foreblocks[preprocessing]`:

```python
from foreblocks import TimeSeriesHandler

pre = TimeSeriesHandler(
    window_size=seq_len,
    horizon=horizon,
    normalize=True,
)
X, y, processed, time_feats = pre.fit_transform(raw_data, time_stamps=timestamps)
```

See [Preprocessor Guide](docs/preprocessor.md) for more details.

## Architecture search with `darts`

DARTS-style differentiable architecture search lives in its own top-level package, `darts`. It searches a cell-based space of time-series operations, then trains the discovered architecture. Install the extra and import from `darts`:

```bash
pip install "foreblocks[darts]"
```

```python
from darts import DARTSTrainer

trainer = DARTSTrainer(
    input_dim=5,
    hidden_dims=[32, 64, 128],
    forecast_horizon=24,
    seq_length=48,
    device="auto",
)

# One call runs candidate generation, ranking, short DARTS training,
# discrete derivation, and final retraining.
results = trainer.multi_fidelity_search(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_candidates=20,
    search_epochs=20,
    final_epochs=80,
    top_k=5,
)

best_model = results["final_model"]
trainer.save_best_model("best_darts_model.pth")
```

`DARTSTrainer` is the high-level entry point; it builds the search model internally and exposes finer-grained steps (`train_darts_model`, `derive_final_architecture`, `train_final_model`, `evaluate_zero_cost_metrics`, …) for custom pipelines. `darts` ships in the same `foreblocks` wheel, so no separate install is required beyond the `darts` extra.

The package also exposes (lazily, see `darts.__all__`) model components (`TimeSeriesDARTS`, `DARTSCell`), configuration dataclasses (`DARTSConfig`, `DARTSSearchSpaceConfig`, `DARTSTrainConfig`, `FinalTrainConfig`, `MultiFidelitySearchConfig`, `AblationSearchConfig`, `RobustPoolSearchConfig`), evaluation helpers (`compute_metrics`, `evaluate_on_loader`, `plot_alpha_evolution`, …), and `ArchitectureInspector`.

See the [DARTS Guide](docs/darts.md) for the full search-space, multi-fidelity, and evaluation workflow.

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
| `TransformerEncoder`, `TransformerDecoder` | Transformer backbones with advanced attention variants, MoE, residual routing, and sparse options such as MoBA |
| `AttentionLayer` | Attention module for custom architectures |

## Repository map

| Path | What it contains |
| --- | --- |
| `foreblocks/core` | `ForecastingModel`, heads, conformal utilities, sampling |
| `foreblocks/training` | `Trainer`, training loop, quantization utilities |
| `foreblocks/evaluation` | `ModelEvaluator`, benchmarking helpers |
| `foreblocks/ts_handler` | `TimeSeriesHandler`, imputation, filtering, outlier handling |
| `foreblocks/transformer` | Transformer stack, attention variants (including MoBA), MoE, fused norms, and embeddings |
| `mltracker` | experiment tracking server, logging, and TUI integration |
| `foreblocks/kan` | Kolmogorov-Arnold Network backbone |
| `foreblocks/mamba` | Original Mamba SSM backbone with MoE and positional encoding |
| `foreblocks/custom_mamba` | Hybrid Mamba / Mamba-2 style SSM blocks for forecasting |
| `foreblocks/custom_raven` | Raven-inspired recurrent sequence blocks |
| `foreblocks/custom_att` | Experimental attention kernels and benchmarks |
| `foreblocks/blocks` | Higher-level reusable blocks: dropout, N-BEATS, popular model blocks, graph blocks, and forecasting heads |
| `foreblocks/layers` | Lower-level neural network layers, currently graph-focused primitives |
| `foreblocks/blocks/wavelets.py` | Multiwavelet feature extraction blocks |
| `foreblocks/evaluation/benchmark.py` | External forecasting baselines and spreadsheet readers |
| `darts` | standalone DARTS NAS package: search space, search/training pipeline, evaluation, and architecture inspection |
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
- [Mixture of Experts Guide](docs/moe.md)
- [Hybrid Mamba Guide](docs/hybrid-mamba.md)
- [DARTS Guide](docs/darts.md)
- [Evaluation & Metrics](docs/evaluation.md)
- [Uncertainty Quantification](docs/uncertainty.md)
- [Web UI](docs/webui.md)
- [Troubleshooting](docs/troubleshooting.md)

Companion tooling:

- [Foretools Overview](docs/foretools/index.md)
- [Time Series Generator](docs/foretools/tsgen.md)
- [BOHB Search](docs/foretools/bohb.md)
- [VMD Decomposition](docs/foretools/vmd.md)
- [AutoDA Augmentation](docs/foretools/tsaug.md)

Examples and notebooks:

- `examples/adaptive_mrmr_demo.py`
- `foretools/tsgen/ts_gen_complete_series.ipynb`
- `foretools/tsgen/ts_gen_doc.ipynb`
- `foretools/`

There is a repository-local docs navigation file at [`docs/.vitepress/config.js`](docs/.vitepress/config.js).

## Current project status

- The repository is broad and still evolving. Some subsystems are more mature than others.
- The top-level imports listed above are the safest place to start.
- `Trainer` supports MLTracker and conformal prediction; use `auto_track=False` during local smoke tests.
- Decoder-based seq2seq and transformer workflows have stricter dimension contracts than the direct forecasting path.
- `TrainingConfig` now centralizes trainer, NAS, MLTracker, and conformal settings.
- The transformer stack includes multiple attention backends, including dense, sparse, local-window, and MoBA-style block-routed attention.

## Contributing

Documentation improvements are especially valuable here because `foreblocks` spans forecasting models, search, preprocessing, and auxiliary tooling. If you add or change a public API, update:

1. this `README.md`
2. the relevant guide under `docs/`
3. at least one runnable example or notebook
