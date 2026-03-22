# foreBlocks

[![PyPI Version](https://img.shields.io/pypi/v/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![Python Versions](https://img.shields.io/pypi/pyversions/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![License](https://img.shields.io/github/license/lseman/foreblocks)](LICENSE)

![ForeBlocks Logo](web/logo.svg#gh-light-mode-only)
![ForeBlocks Logo](web/logo_dark.svg#gh-dark-mode-only)

**foreBlocks** is a modular PyTorch library for time-series forecasting. The repository combines:

- `foreblocks`: forecasting models, training, evaluation, preprocessing, and DARTS search
- `foretools`: companion utilities, synthetic data generation, decomposition, and analysis notebooks

The project is best approached as a research toolkit rather than a single monolithic framework. The most stable public entry points are the top-level imports exported from `foreblocks`.

## Install

```bash
pip install foreblocks
```

Install optional extras when you need specific subsystems:

| Extra | Adds |
| --- | --- |
| `darts` | DARTS search plus analysis dependencies |
| `mltracker` | experiment tracking API and UI dependencies |
| `vmd` | VMD decomposition and Optuna-based search support |
| `wavelets` | optional wavelet backends |
| `benchmark` | external forecasting baselines and spreadsheet readers |
| `foreminer` | changepoint-detection support |
| `all` | all runtime extras above |

Examples:

```bash
pip install "foreblocks[darts]"
pip install "foreblocks[mltracker]"
pip install "foreblocks[vmd,wavelets]"
pip install "foreblocks[all]"
```

Local development install:

```bash
git clone https://github.com/lseman/foreblocks.git
cd foreblocks
pip install -e ".[dev]"
```

## Validated Quickstart

The example below is intentionally small and uses the most reliable path through the current API: a direct forecaster with a custom head, trained through `Trainer`.

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

This path was smoke-tested in the repository. Once that is working, move on to encoder/decoder models, preprocessing, and DARTS.

## Public API

These are the top-level imports currently exposed by `foreblocks`:

| Import | Purpose |
| --- | --- |
| `ForecastingModel` | Core forecasting wrapper for direct, autoregressive, and seq2seq-style models |
| `Trainer` | Training loop with NAS hooks, MLTracker integration, and optional conformal support |
| `ModelEvaluator` | Prediction helpers, metrics, cross-validation, and training-curve plots |
| `TimeSeriesHandler` | Time-series handling pipeline for windowing, scaling, filtering, imputation, and time features |
| `TimeSeriesDataset` | Dataset wrapper used by the dataloader helper |
| `create_dataloaders` | Build train/validation PyTorch dataloaders from NumPy arrays |
| `ModelConfig`, `TrainingConfig` | Lightweight configuration dataclasses |
| `LSTMEncoder`, `LSTMDecoder`, `GRUEncoder`, `GRUDecoder` | Recurrent encoder/decoder blocks |
| `TransformerEncoder`, `TransformerDecoder` | Transformer backbones and related advanced features |
| `AttentionLayer` | Attention module entry point |

## Repository Map

| Path | What it contains |
| --- | --- |
| `foreblocks/core` | `ForecastingModel`, heads, conformal utilities, sampling |
| `foreblocks/training` | `Trainer`, training loop, quantization utilities |
| `foreblocks/evaluation` | `ModelEvaluator`, benchmarking helpers |
| `foreblocks/ts_handler` | `TimeSeriesHandler`, imputation, filtering, outlier handling |
| `foreblocks/tf` | transformer stack, attention variants, MoE, norms, embeddings |
| `foreblocks/darts` | neural architecture search pipeline and evaluation |
| `foretools/tsgen` | synthetic time-series generator and notebooks |
| `examples/` | notebooks and runnable usage examples |
| `web/` | static landing page assets for the published site root |
| `docs/` | MkDocs source for the versioned documentation site published under `/docs/` |

## Documentation Map

Start here if you are new to the repository:

- [Documentation Overview](docs/overview.md)
- [Getting Started](docs/getting-started.md)
- [Docs Home](docs/index.md)

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

Useful notebooks and examples:

- [Synthetic Series Notebook](foretools/tsgen/ts_gen_complete_series.ipynb)
- [TS Generator Documentation Notebook](foretools/tsgen/ts_gen_doc.ipynb)
- [AdaptiveMRMR Demo](examples/adaptive_mrmr_demo.py)
- [Example notebooks](examples/)

There is also a repository-local docs navigation file at [`mkdocs.yml`](mkdocs.yml). The current publishing model is:

- site root `/`: custom landing page from `web/index.html`
- site docs `/docs/`: MkDocs site built from `docs/`

## Current Project Status

- The repository is broad and still evolving. Some subsystems are more mature than others.
- The top-level imports listed above are the safest place to start.
- `Trainer` supports MLTracker and conformal prediction, but you can disable tracking during local smoke tests with `auto_track=False`.
- `MultiAttention` now includes an experimental attention-matching KV compaction mode for dense paged causal decode. Enable it with `use_attention_matching_compaction=True` and `use_mla=False`.
- For decoder-based seq2seq and transformer workflows, use the topic guides before wiring custom modules, because dimension contracts are stricter than the direct head path.
- `TrainingConfig` now lives in a single canonical location and includes trainer, NAS, MLTracker, and conformal settings.

## Contributing

Documentation improvements are especially valuable here because the repository spans forecasting models, search, preprocessing, and auxiliary tooling. If you add or change a public API, update:

1. this `README.md`
2. the relevant guide under `docs/`
3. at least one runnable example or notebook
