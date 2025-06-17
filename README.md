# foreblocks: Modular Time Series Forecasting Library

[![PyPI Version](https://img.shields.io/pypi/v/tracernaut.svg)](https://pypi.org/project/foreblocks/)
[![Python Versions](https://img.shields.io/pypi/pyversions/foreblocks.svg)](https://pypi.org/project/foreblocks/)
[![License](https://img.shields.io/github/license/lseman/foreblocks)](LICENSE)

![ForeBlocks Logo](logo.svg#gh-light-mode-only)
![ForeBlocks Logo](logo_dark.svg#gh-dark-mode-only)

**ForeBlocks** is a flexible, modular deep learning framework for time series forecasting built on PyTorch. It provides various neural network architectures and forecasting strategies to tackle complex time series prediction problems with an intuitive, research-friendly API.

🔗 **[GitHub Repository](https://github.com/lseman/foreblocks)**

---

## 🚀 Quick Start

```bash
# Installation
git clone https://github.com/lseman/foreblocks
cd foreblocks
pip install -e .
```

or

```bash
# Instalattion via pip
pip install foreblocks
```

```python
from foreblocks import TimeSeriesSeq2Seq, ModelConfig, TrainingConfig
import torch
import pandas as pd

# Load and prepare data
data = pd.read_csv('your_data.csv')
X = data.values

# Configure model
model_config = ModelConfig(
    model_type="lstm",
    input_size=X.shape[1],
    output_size=1,
    hidden_size=64,
    target_len=24,  # Forecast 24 steps ahead
    teacher_forcing_ratio=0.5
)

# Initialize and train
model = TimeSeriesSeq2Seq(model_config=model_config)
X_train, y_train, _ = model.preprocess(X, self_tune=True)

# Create DataLoader and train
from torch.utils.data import TensorDataset, DataLoader
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

| Feature | Description |
|---------|-------------|
| **🔧 Multiple Strategies** | Seq2Seq, Autoregressive, and Direct forecasting approaches |
| **🧩 Modular Design** | Easily customize and extend components |
| **🤖 Advanced Models** | LSTM, GRU, Transformer, and VAE-based architectures |
| **⚡ Smart Preprocessing** | Adaptive data preprocessing with automatic configuration |
| **🎯 Attention Mechanisms** | Various attention modules for improved performance |
| **📊 Multi-Feature Support** | Specialized architectures for multivariate time series |
| **📈 Training Utilities** | Built-in trainer with callbacks, metrics, and visualizations |
| **🔍 Transparent API** | Intuitive interface with extensive documentation |

---

## 📖 Documentation

Explore the documentation:


| Section         | Description                                     | Link                                |
|-----------------|-------------------------------------------------|-------------------------------------|
| Preprocessing   | Data cleaning, EWT, normalization, differencing | [Guide](docs/preprocessor.md)       |
| Custom Blocks   | Register your own modules                       | [Guide](docs/custom_blocks.md)      |
| Transformers    | Core transformer blocks                         | [Docs](docs/transformer.md)         |
| Fourier         | Fourier-based forecasting layers                | [Docs](docs/fourier.md)             |
| Wavelet         | Wavelet transform layers                        | [Docs](docs/wavelet.md)             |
| DARTS           | Differentiable Architecture Search              | [Docs](docs/darts.md)               |

---

## 🏗️ Architecture Overview

ForeBlocks follows a clean, modular design:

- **`TimeSeriesSeq2Seq`**: High-level interface for building and training models
- **`ForecastingModel`**: Main model class integrating encoders, decoders, and strategies
- **`TimeSeriesPreprocessor`**: Advanced data preparation with automatic feature detection
- **`Trainer`**: Manages training, evaluation, and visualization

---

## ForecastingModel

![ForecastingModel](docs/imgs/forecasting_model_diagram.svg)

---

## 🎯 Forecasting Models

### 1. Sequence-to-Sequence (Default)
*Best for: Most time series problems*

```python
model_config = ModelConfig(
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

### 2. Autoregressive
*Best for: When each prediction depends on previous predictions*

```python
model_config = ModelConfig(
    model_type="lstm",
    strategy="autoregressive",
    input_size=1,
    output_size=1,
    hidden_size=64,
    target_len=12
)
```

### 3. Direct Multi-Step
*Best for: Independent multi-step predictions*

```python
model_config = ModelConfig(
    model_type="lstm",
    strategy="direct",
    input_size=5,
    output_size=1,
    hidden_size=128,
    target_len=48
)
```

### 4. Transformer-based
*Best for: Long sequences with complex dependencies*

```python
model_config = ModelConfig(
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

## 🔧 Advanced Features

### Multi-Encoder-Decoder Architecture
Process different features with separate encoders:

```python
model_config = ModelConfig(
    multi_encoder_decoder=True,
    input_size=5,  # 5 different features
    output_size=1,
    hidden_size=64,
    model_type="lstm",
    target_len=24
)
```

### Attention Mechanisms
Improve performance with attention:

```python
from foreblocks.attention import AttentionLayer

attention_module = AttentionLayer(
    method="dot",
    attention_backend="self",
    encoder_hidden_size=64,
    decoder_hidden_size=64
)

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    attention_module=attention_module
)
```

### Custom Preprocessing Pipeline
Fine-tune data preparation:

```python
X_train, y_train, processed_data = model.preprocess(
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
Control teacher forcing dynamically:

```python
def scheduled_sampling_fn(epoch):
    return max(0.0, 1.0 - 0.1 * epoch)  # Linear decay

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    scheduled_sampling_fn=scheduled_sampling_fn
)
```

---

## 📚 Examples

### LSTM with Attention
```python
from foreblocks import TimeSeriesSeq2Seq, ModelConfig, AttentionLayer
import torch.nn as nn

# Configure model with attention
model_config = ModelConfig(
    model_type="lstm",
    input_size=3,
    output_size=1,
    hidden_size=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    target_len=24
)

attention = AttentionLayer(
    method="dot",
    encoder_hidden_size=64,
    decoder_hidden_size=64
)

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    attention_module=attention,
    output_block=nn.Sequential(nn.Dropout(0.1), nn.ReLU())
)
```

### Transformer Model
```python
from foreblocks import TimeSeriesSeq2Seq, ModelConfig, TrainingConfig

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
    learning_rate=0.0001,
    weight_decay=1e-5,
    patience=15
)

model = TimeSeriesSeq2Seq(
    model_config=model_config,
    training_config=training_config
)
```

---

## 🔧 Configuration Reference

### ModelConfig Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_type` | str | Model architecture ("lstm", "gru", "transformer") | "lstm" |
| `input_size` | int | Number of input features | Required |
| `output_size` | int | Number of output features | Required |
| `hidden_size` | int | Hidden layer dimensions | 64 |
| `target_len` | int | Forecast horizon length | Required |
| `num_encoder_layers` | int | Number of encoder layers | 1 |
| `num_decoder_layers` | int | Number of decoder layers | 1 |
| `teacher_forcing_ratio` | float | Teacher forcing probability | 0.5 |

### TrainingConfig Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_epochs` | int | Training epochs | 100 |
| `learning_rate` | float | Learning rate | 0.001 |
| `batch_size` | int | Batch size | 32 |
| `patience` | int | Early stopping patience | 10 |
| `weight_decay` | float | L2 regularization | 0.0 |

---

## 🚨 Troubleshooting

### Common Issues & Solutions

<details>
<summary><strong>🔴 Dimensionality Mismatch</strong></summary>

**Problem**: Tensor dimension errors during training/inference

**Solution**:
- Check encoder/decoder `hidden_size` compatibility
- Verify `output_size` matches target dimensions
- Ensure input data shape matches `input_size`

```python
# Debug dimensions
print(f"Input shape: {X.shape}")
print(f"Model expects: {model_config.input_size} features")
```
</details>

<details>
<summary><strong>🟡 Memory Issues</strong></summary>

**Problem**: CUDA out of memory or system RAM exhaustion

**Solutions**:
- Reduce `batch_size` or sequence length
- Use gradient accumulation
- Consider model size reduction

```python
# Gradient accumulation example
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```
</details>

<details>
<summary><strong>🟠 Poor Performance</strong></summary>

**Problem**: Model not learning or poor predictions

**Solutions**:
- Try different forecasting strategies
- Adjust `teacher_forcing_ratio`
- Add attention mechanisms
- Experiment with architectures (LSTM vs Transformer)
- Tune hyperparameters

```python
# Performance tuning checklist
model_config = ModelConfig(
    hidden_size=128,  # Try larger hidden size
    num_encoder_layers=3,  # Add more layers
    teacher_forcing_ratio=0.3,  # Reduce teacher forcing
    # Add dropout, attention, etc.
)
```
</details>

<details>
<summary><strong>🔵 Training Issues</strong></summary>

**Problem**: Slow convergence or gradient problems

**Solutions**:
- Use gradient clipping
- Learning rate scheduling
- Proper weight initialization

```python
# Gradient clipping
import torch.nn.utils as utils
utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)
```
</details>

---

## 💡 Best Practices

### 🎯 Performance Tips
- **Always normalize** input data for better convergence
- **Use appropriate metrics** (MAE, RMSE, MAPE) for time series
- **Validate on multi-step** predictions, not just one-step
- **Consider model ensembling** for critical applications

### 📊 Data Preparation
- Handle missing values before feeding to model
- Consider seasonal decomposition for seasonal data
- Use the built-in preprocessing with `self_tune=True`

### 🔄 Training Strategy
- Start with simple models (LSTM) before trying complex ones (Transformer)
- Use validation sets for hyperparameter tuning
- Monitor both training and validation metrics

---

## 🤝 Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/lseman/foreblocks) for:
- 🐛 Bug reports
- 💡 Feature requests
- 📝 Documentation improvements
- 🔧 Code contributions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

