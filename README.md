# foreBlocks: Modular Deep Learning Library for Time Series Forecasting

[![PyPI Version](https://img.shields.io/pypi/v/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![Python Versions](https://img.shields.io/pypi/pyversions/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![License](https://img.shields.io/github/license/lseman/foreblocks)](LICENSE)

![ForeBlocks Logo](logo.svg#gh-light-mode-only)
![ForeBlocks Logo](logo_dark.svg#gh-dark-mode-only)

**foreBlocks** is a flexible and modular deep learning library for time series forecasting, built on PyTorch. It provides a wide range of neural network architectures and forecasting strategies through a clean, research-friendly API — enabling fast experimentation and scalable deployment.

🔗 **[GitHub Repository](https://github.com/lseman/foreblocks)** · **[PyPI](https://pypi.org/project/foreblocks/)**

---

## 🚀 Quick Start

```bash
pip install foreblocks
```

Or install from source:

```bash
git clone https://github.com/lseman/foreblocks.git
cd foreblocks
pip install -e .
```

```python
import torch
from foreblocks import (
    ForecastingModel, Trainer, TrainingConfig,
    create_dataloaders, LSTMEncoder, LSTMDecoder,
)

# Build a seq2seq LSTM forecaster
encoder = LSTMEncoder(input_size=8, hidden_size=128, num_layers=2)
decoder = LSTMDecoder(hidden_size=128, output_size=1, num_layers=2)

model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    forecasting_strategy="seq2seq",
    target_len=24,
)

# Create data loaders
train_loader, val_loader = create_dataloaders(
    X_train, y_train, X_val, y_val, batch_size=32
)

# Train
config = TrainingConfig(num_epochs=100, learning_rate=1e-3, use_amp=True)
trainer = Trainer(model, config=config)
trainer.train(train_loader, val_loader)
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **`ForecastingModel`** | Unified model wrapping any encoder/decoder pair with pluggable heads |
| **Modular Blocks** | LSTM, GRU, Transformer, TCN, xLSTM, ODE, Graph, Fourier, Wavelet |
| **Head System** | 13+ composable heads: RevIN, DAIN, Decomposition, PatchEmbed, FFT, Wavelet, Time2Vec, … |
| **Forecasting Strategies** | `seq2seq`, `autoregressive`, `direct`, `transformer_seq2seq` |
| **Conformal Prediction** | 7 methods: `split`, `local`, `rolling`, `agaci`, `enbpi`, `cptc`, `afocp` |
| **NAS via HeadComposer** | Differentiable head search with configurable alpha LR and warmup |
| **`Trainer`** | AMP, gradient clipping, early stopping, LR scheduling, MLTracker SQLite logging |
| **`ModelEvaluator`** | MAE, RMSE, MAPE, SMAPE, MASE, coverage, sharpness, Winkler score |

---

## 🏗️ Architecture Overview

```
Input Tensor  ──┬──[input head]──► Encoder ──[encoder head]──►
                │                                               │
                │                                               ▼
                │                                     Attention Module
                │                                               │
                └──────────────────────────────────────────────►
                                                                │
                                                                ▼
                                                    Decoder ──[decoder head]──►
                                                                │
                                                      [output head / head_only]
                                                                │
                                                         Output Tensor
```

`ForecastingModel` glues any encoder, decoder, and attention module together and routes each forward pass through the chosen `forecasting_strategy`. Heads can be inserted at eight positions (`input`, `encoder`, `attention`, `decoder`, `output`, `input_norm`, `output_norm`, `head`) via `add_head()`.

---

## 🔮 Forecasting Strategies

### LSTM Seq2Seq

```python
from foreblocks import ForecastingModel, LSTMEncoder, LSTMDecoder

encoder = LSTMEncoder(input_size=8, hidden_size=128, num_layers=2)
decoder = LSTMDecoder(hidden_size=128, output_size=1, num_layers=2)

model = ForecastingModel(
    encoder=encoder, decoder=decoder,
    forecasting_strategy="seq2seq",
    target_len=24,
)
```

### Transformer Seq2Seq

```python
from foreblocks import ForecastingModel, TransformerEncoder, TransformerDecoder

encoder = TransformerEncoder(input_size=8, d_model=128, nhead=4, num_layers=3)
decoder = TransformerDecoder(d_model=128, output_size=1, nhead=4, num_layers=3)

model = ForecastingModel(
    encoder=encoder, decoder=decoder,
    forecasting_strategy="transformer_seq2seq",
    target_len=24,
)
```

### Autoregressive & Direct

```python
from foreblocks import ForecastingModel, LSTMEncoder

encoder = LSTMEncoder(input_size=8, hidden_size=128, num_layers=2)

# Autoregressive: step-by-step, feeding last prediction as next input
ar_model = ForecastingModel(
    encoder=encoder,
    forecasting_strategy="autoregressive",
    target_len=24, output_size=1,
)

# Direct: single-shot multi-step projection
direct_model = ForecastingModel(
    encoder=encoder,
    forecasting_strategy="direct",
    target_len=24, output_size=1,
)
```

---

## ⚙️ Advanced Features

### Head System

```python
from foreblocks.core.heads import RevinHead, PatchEmbedHead, FFTTopKHead

# Add a reversible instance normalisation head at the input
model.add_head(RevinHead(num_features=8), position="input")

# Stack patch embedding before the encoder
model.add_head(PatchEmbedHead(input_size=8, patch_size=16, d_model=128), position="encoder")

# FFT spectral filtering at the output
model.add_head(FFTTopKHead(top_k=16), position="output")

# Inspect all heads
print(model.list_heads())
```

Available heads: `RevinHead`, `DAINHead`, `DecompositionHead`, `DifferencingHead`,
`DropoutTSHead`, `FFTTopKHead`, `HaarWaveletTopKHead`, `LearnableFourierSeasonalHead`,
`MultiKernelConvHead`, `MultiscaleConvHead`, `PatchEmbedHead`, `Time2VecHead`, `TimeAttentionHead`

### Neural Architecture Search (HeadComposer)

```python
from foreblocks import TrainingConfig
from foreblocks.core.heads import HeadComposer, HeadSpec, RevinHead, DAINHead, PatchEmbedHead

# Define a search space
candidates = [RevinHead(8), DAINHead(8), PatchEmbedHead(8, 16, 128)]
composer = HeadComposer([HeadSpec(h) for h in candidates])
model.set_head_composer(composer)

# Enable NAS in training
config = TrainingConfig(
    num_epochs=100,
    train_nas=True,
    nas_alpha_lr=3e-4,
    nas_warmup_epochs=10,
)
trainer = Trainer(model, config=config)
trainer.train(train_loader, val_loader)
```

### Conformal Prediction Intervals

```python
from foreblocks import TrainingConfig, Trainer

config = TrainingConfig(
    conformal_enabled=True,
    conformal_method="enbpi",   # split | local | rolling | agaci | enbpi | cptc | afocp
    conformal_quantile=0.9,
)
trainer = Trainer(model, config=config)
trainer.train(train_loader, val_loader)

# Calibrate on held-out data, then predict with intervals
trainer.calibrate_conformal(cal_loader)
preds, lower, upper = trainer.predict_with_intervals(X_test)
coverage = trainer.compute_coverage(y_test, lower, upper)
trainer.plot_intervals(y_test, lower, upper)
```

### Scheduled Sampling

```python
from foreblocks.aux.utils import linear_schedule, exponential_schedule, inverse_sigmoid_schedule

# Linear decay: teacher forcing ratio goes from 1.0 → 0.0 over training
model = ForecastingModel(
    encoder=encoder, decoder=decoder,
    forecasting_strategy="seq2seq",
    target_len=24,
    scheduled_sampling_fn=linear_schedule(start=1.0, end=0.0, total_steps=100),
)
```

---

## 🛠️ Configuration Reference

### `TrainingConfig`

| Parameter | Default | Description |
|---|---|---|
| `num_epochs` | `100` | Maximum training epochs |
| `learning_rate` | `0.001` | Optimiser learning rate |
| `batch_size` | `32` | Mini-batch size |
| `patience` | `10` | Early-stopping patience (epochs) |
| `use_amp` | `True` | Mixed-precision (AMP) training |
| `gradient_clip_val` | `None` | Gradient norm clipping value |
| `scheduler_type` | `None` | LR scheduler: `cosine`, `step`, `plateau`, … |
| `weight_decay` | `0.0` | L2 regularisation coefficient |
| `train_nas` | `False` | Enable differentiable NAS |
| `nas_alpha_lr` | `3e-4` | Architecture parameter learning rate |
| `nas_warmup_epochs` | `5` | Epochs before NAS search begins |
| `conformal_enabled` | `False` | Enable conformal prediction |
| `conformal_method` | `"split"` | Conformal method (`split`, `local`, `rolling`, `agaci`, `enbpi`, `cptc`, `afocp`) |
| `conformal_quantile` | `0.9` | Coverage target quantile |

### `ModelConfig`

| Parameter | Description |
|---|---|
| `input_size` | Number of input features |
| `hidden_size` | Encoder/decoder hidden units |
| `num_layers` | Number of recurrent/transformer layers |
| `output_size` | Number of output targets |
| `target_len` | Forecast horizon |
| `dropout` | Dropout rate |
| `forecasting_strategy` | One of `seq2seq`, `autoregressive`, `direct`, `transformer_seq2seq` |
| `model_type` | One of `lstm`, `transformer`, `informer-like`, `head_only` |
| `use_attention` | Whether to add an attention module |
| `nhead` | Number of attention heads (Transformer) |

---

## 📖 Documentation

| Resource | Link |
|---|---|
| Full Docs | [lseman.github.io/foreblocks](https://lseman.github.io/foreblocks) |
| GitHub Repo | [github.com/lseman/foreblocks](https://github.com/lseman/foreblocks) |
| PyPI Package | [pypi.org/project/foreblocks](https://pypi.org/project/foreblocks/) |
| Issue Tracker | [GitHub Issues](https://github.com/lseman/foreblocks/issues) |

---

## 🩺 Troubleshooting

<details>
<summary><strong>CUDA out of memory</strong></summary>

Reduce `batch_size` in `TrainingConfig`, or enable gradient checkpointing:

```python
config = TrainingConfig(batch_size=8, gradient_clip_val=1.0)
```

Alternatively, use `use_amp=True` (default) to reduce memory via mixed precision.
</details>

<details>
<summary><strong>Loss is NaN during training</strong></summary>

Common causes:
- Learning rate too high → lower `learning_rate`
- Inputs not normalised → add a `RevinHead` or `DAINHead` at `position="input"`
- Gradient explosion → set `gradient_clip_val=1.0`

```python
model.add_head(RevinHead(num_features=input_size), position="input")
config = TrainingConfig(learning_rate=1e-4, gradient_clip_val=1.0)
```
</details>

<details>
<summary><strong>Conformal intervals are too wide</strong></summary>

Try a different conformal method or adjust the quantile:

```python
config = TrainingConfig(
    conformal_enabled=True,
    conformal_method="agaci",   # adaptive method
    conformal_quantile=0.80,    # 80 % coverage
)
```
</details>

---

## 💡 Best Practices

- **Always normalise inputs** — use `RevinHead` or `DAINHead` at `position="input"` unless your data is already z-scored.
- **Start simple** — begin with `seq2seq` + LSTM before exploring Transformer or `head_only` strategies.
- **Use AMP** — `use_amp=True` is the default; disable only when debugging NaN losses.
- **Monitor with MLTracker** — `Trainer` auto-creates a SQLite experiment log; pass `auto_track=True` (default).
- **Validate conformal coverage** — after calibration, call `compute_coverage()` and aim for ≥ your target quantile.
- **NAS warmup** — set `nas_warmup_epochs` to at least 5–10 % of `num_epochs` to stabilise base weights before architecture search.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository and create a feature branch.
2. Run `npm test` / `pytest` and ensure all tests pass.
3. Follow existing code style (`ruff` / `black`).
4. Open a pull request with a clear description.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
