# foreBlocks: Modular Deep Learning Library for Time Series Forecasting

[![PyPI Version](https://img.shields.io/pypi/v/tracernaut.svg)](https://pypi.org/project/foreblocks/)
[![Python Versions](https://img.shields.io/pypi/pyversions/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![License](https://img.shields.io/github/license/lseman/foreblocks)](LICENSE)

![ForeBlocks Logo](logo.svg#gh-light-mode-only)
![ForeBlocks Logo](logo_dark.svg#gh-dark-mode-only)

**foreBlocks** is a flexible and modular deep learning library for time series forecasting, built on PyTorch. It provides a wide range of neural network architectures and forecasting strategies through a clean, research-friendly API — enabling fast experimentation and scalable deployment.

🔗 **[GitHub Repository](https://github.com/lseman/foreblocks)**

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/lseman/foreblocks
cd foreblocks
pip install -e .
````

Or install directly via PyPI:

```bash
pip install foreblocks
```

```python
from foreblocks import TimeSeriesSeq2Seq, ModelConfig, TrainingConfig
import pandas as pd
import torch

# Load your time series dataset
data = pd.read_csv('your_data.csv')
X = data.values

# Configure the model
model_config = ModelConfig(
    model_type="lstm",
    input_size=X.shape[1],
    output_size=1,
    hidden_size=64,
    target_len=24,
    teacher_forcing_ratio=0.5
)

# Initialize and train
model = TimeSeriesSeq2Seq(model_config=model_config)
X_train, y_train, _ = model.preprocess(X, self_tune=True)

# Create DataLoader and start training
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

history = model.train_model(train_loader)
predictions = model.predict(X_test)
```

---

## ✨ Key Features

| Feature                     | Description                                                        |
| --------------------------- | ------------------------------------------------------------------ |
| 🔧 **Multiple Strategies**  | Seq2Seq, Autoregressive, and Direct forecasting modes              |
| 🧩 **Modular Design**       | Easily swap and extend model components                            |
| 🤖 **Advanced Models**      | LSTM, GRU, Transformer, VAE, and more                              |
| ⚡ **Smart Preprocessing**   | Automatic normalization, differencing, EWT, and outlier handling   |
| 🎯 **Attention Modules**    | Pluggable attention layers for enhanced temporal modeling          |
| 📊 **Multivariate Support** | Designed for multi-feature time series with dynamic input handling |
| 📈 **Training Utilities**   | Built-in trainer with callbacks, metrics, and visualizations       |
| 🔍 **Transparent API**      | Clean and extensible codebase with complete documentation          |

---

## 📖 Documentation

| Section       | Description                                      | Link                           |
| ------------- | ------------------------------------------------ | ------------------------------ |
| Preprocessing | EWT, normalization, differencing, outliers       | [Guide](docs/preprocessor.md)  |
| Custom Blocks | Registering new encoder/decoder/attention blocks | [Guide](docs/custom_blocks.md) |
| Transformers  | Transformer-based modules                        | [Docs](docs/transformer.md)    |
| Fourier       | Frequency-based forecasting layers               | [Docs](docs/fourier.md)        |
| Wavelet       | Wavelet transform modules                        | [Docs](docs/wavelet.md)        |
| DARTS         | Architecture search for forecasting              | [Docs](docs/darts.md)          |

---

## 🏗️ Architecture Overview

ForeBlocks is built around clean and extensible abstractions:

* `TimeSeriesSeq2Seq`: High-level interface for forecasting workflows
* `ForecastingModel`: Core model engine combining encoders, decoders, and heads
* `TimeSeriesPreprocessor`: Adaptive preprocessing with feature engineering
* `Trainer`: Handles training loop, validation, and visual feedback

---

## 🔮 Forecasting Models

### 1. **Sequence-to-Sequence** (default)

```python
ModelConfig(
    model_type="lstm",
    strategy="seq2seq",
    input_size=3,
    output_size=1,
    hidden_size=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    target_len=24
)
```

### 2. **Autoregressive**

```python
ModelConfig(
    model_type="lstm",
    strategy="autoregressive",
    input_size=1,
    output_size=1,
    hidden_size=64,
    target_len=12
)
```

### 3. **Direct Multi-Step**

```python
ModelConfig(
    model_type="lstm",
    strategy="direct",
    input_size=5,
    output_size=1,
    hidden_size=128,
    target_len=48
)
```

### 4. **Transformer-based**

```python
ModelConfig(
    model_type="transformer",
    strategy="transformer_seq2seq",
    input_size=4,
    output_size=4,
    hidden_size=128,
    dim_feedforward=512,
    nheads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    target_len=96
)
```

---

## ⚙️ Advanced Features

### Multi-Encoder/Decoder

```python
ModelConfig(
    multi_encoder_decoder=True,
    input_size=5,
    output_size=1,
    hidden_size=64,
    model_type="lstm",
    target_len=24
)
```

### Attention Integration

```python
from foreblocks.attention import AttentionLayer

attention = AttentionLayer(
    method="dot",
    attention_backend="self",
    encoder_hidden_size=64,
    decoder_hidden_size=64
)

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    attention_module=attention
)
```

### Custom Preprocessing

```python
X_train, y_train, _ = model.preprocess(
    X,
    normalize=True,
    differencing=True,
    detrend=True,
    apply_ewt=True,
    window_size=48,
    horizon=24,
    remove_outliers=True,
    outlier_method="iqr",
    self_tune=True
)
```

### Scheduled Sampling

```python
def schedule(epoch): return max(0.0, 1.0 - 0.1 * epoch)

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    scheduled_sampling_fn=schedule
)
```

---

## 🧪 Examples

### LSTM + Attention

```python
model_config = ModelConfig(
    model_type="lstm",
    input_size=3,
    output_size=1,
    hidden_size=64,
    target_len=24
)

attention = AttentionLayer(
    method="dot",
    encoder_hidden_size=64,
    decoder_hidden_size=64
)

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    attention_module=attention
)
```

### Transformer Model

```python
model_config = ModelConfig(
    model_type="transformer",
    input_size=4,
    output_size=4,
    hidden_size=128,
    dim_feedforward=512,
    nheads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    target_len=96
)

training_config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=15
)

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    training_config=training_config
)
```

---

## 🛠️ Configuration Reference

### `ModelConfig`

| Parameter               | Type  | Description                        | Default |
| ----------------------- | ----- | ---------------------------------- | ------- |
| `model_type`            | str   | "lstm", "gru", "transformer", etc. | "lstm"  |
| `input_size`            | int   | Number of input features           | —       |
| `output_size`           | int   | Number of output features          | —       |
| `hidden_size`           | int   | Hidden layer dimension             | 64      |
| `target_len`            | int   | Forecast steps                     | —       |
| `num_encoder_layers`    | int   | Encoder depth                      | 1       |
| `num_decoder_layers`    | int   | Decoder depth                      | 1       |
| `teacher_forcing_ratio` | float | Ratio of teacher forcing           | 0.5     |

### `TrainingConfig`

| Parameter       | Type  | Description             | Default |
| --------------- | ----- | ----------------------- | ------- |
| `num_epochs`    | int   | Training epochs         | 100     |
| `learning_rate` | float | Learning rate           | 1e-3    |
| `batch_size`    | int   | Mini-batch size         | 32      |
| `patience`      | int   | Early stopping patience | 10      |
| `weight_decay`  | float | L2 regularization       | 0.0     |

---

## 🩺 Troubleshooting

<details>
<summary><strong>🔴 Dimension Mismatch</strong></summary>

* Confirm `input_size` and `output_size` match your data
* Ensure encoder/decoder hidden sizes are compatible

</details>

<details>
<summary><strong>🟡 Memory Issues</strong></summary>

* Reduce `batch_size`, `hidden_size`, or sequence length
* Use gradient accumulation or mixed precision

</details>

<details>
<summary><strong>🟠 Poor Predictions</strong></summary>

* Try different `strategy`
* Use attention mechanisms
* Fine-tune hyperparameters (e.g. `hidden_size`, dropout)

</details>

<details>
<summary><strong>🔵 Training Instability</strong></summary>

* Clip gradients (`clip_grad_norm_`)
* Use learning rate schedulers (`ReduceLROnPlateau`)

</details>

---

## 💡 Best Practices

* ✅ Always normalize input data
* ✅ Evaluate with appropriate multi-step metrics (e.g. MAPE, MAE)
* ✅ Try simple models (LSTM) before complex ones (Transformer)
* ✅ Use `self_tune=True` in preprocessing for best defaults
* ✅ Split validation data chronologically

---

## 🤝 Contributing

We welcome contributions! Visit the [GitHub repo](https://github.com/lseman/foreblocks) to:

* Report bugs 🐛
* Request features 💡
* Improve documentation 📚
* Submit PRs 🔧

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).