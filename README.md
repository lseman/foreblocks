# ForeBlocks Tutorial

This tutorial explains how to use the `ForecastingModel` class for time series forecasting with PyTorch. The ForecastingModel is a versatile framework that supports different forecasting strategies including sequence-to-sequence (seq2seq), autoregressive, and direct approaches.

## Table of Contents

- [ForeBlocks Tutorial](#foreblocks-tutorial)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Key Features](#key-features)
  - [Basic Usage](#basic-usage)
  - [Forecasting Strategies](#forecasting-strategies)
    - [1. Sequence-to-Sequence (seq2seq)](#1-sequence-to-sequence-seq2seq)
    - [2. Autoregressive](#2-autoregressive)
    - [3. Direct](#3-direct)
  - [Advanced Features](#advanced-features)
    - [Attention Mechanism](#attention-mechanism)
    - [Multi-Encoder-Decoder Architecture](#multi-encoder-decoder-architecture)
    - [Teacher Forcing and Scheduled Sampling](#teacher-forcing-and-scheduled-sampling)
    - [Transformer-based Models](#transformer-based-models)
  - [Complete Examples](#complete-examples)
    - [LSTM-based Seq2Seq Model with Attention](#lstm-based-seq2seq-model-with-attention)
    - [Transformer-based Model](#transformer-based-model)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues and Solutions](#common-issues-and-solutions)
    - [Tips for Better Performance](#tips-for-better-performance)

## Overview

The `ForecastingModel` class is designed to be a flexible framework for time series forecasting. It supports various neural network architectures and forecasting strategies, allowing you to experiment with different approaches for your specific forecasting problem.

➡️ [Preprocessing Guide](docs/preprocessor.md)  
➡️ [Custom Blocks Guide](docs/custom_blocks.md)


## Installation

Before using the `ForecastingModel` class, ensure you have PyTorch installed:

```bash
pip install torch
```

You'll also need to have your own encoder and decoder modules defined, which will be used by the `ForecastingModel`.

## Key Features

- **Multiple Forecasting Strategies**: Supports sequence-to-sequence (seq2seq), autoregressive, and direct forecasting approaches.
- **Attention Mechanism**: Optional attention module to improve forecasting performance.
- **Multi-Encoder-Decoder Architecture**: Ability to use separate encoder-decoder pairs for each input feature.
- **Teacher Forcing**: Control the rate of teacher forcing during training.
- **Scheduled Sampling**: Gradually decrease teacher forcing ratio during training.
- **Transformer Support**: Support for transformer-based models.
- **Pre/Post Processing**: Customizable input preprocessing and output postprocessing.

## Basic Usage

Here's a simple example of how to use the `ForecastingModel` class:

```python
import torch
import torch.nn as nn
from foreblocks import ForecastingModel, LSTMEncoder, LSTMDecoder

# Define input parameters
input_size = 1  # Number of features in input
hidden_size = 64
num_layers = 2
output_size = 1  # Number of features to predict
target_len = 10  # Prediction horizon

# Create encoder and decoder
encoder = LSTMEncoder(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers
)

decoder = LSTMDecoder(
    input_size=output_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers
)

# Create the forecasting model
model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    target_len=target_len,
    forecasting_strategy="seq2seq",
    teacher_forcing_ratio=0.5,
    output_size=output_size
)

# Use the model for training
batch_size = 32
seq_len = 50  # Input sequence length

# Create dummy data
x = torch.randn(batch_size, seq_len, input_size)
y = torch.randn(batch_size, target_len, output_size)

# Forward pass
outputs = model(x, y)

# Calculate loss and backpropagate
criterion = nn.MSELoss()
loss = criterion(outputs, y)
loss.backward()
```

## Forecasting Strategies

The `ForecastingModel` supports three main forecasting strategies:

### 1. Sequence-to-Sequence (seq2seq)

The sequence-to-sequence approach uses an encoder to process the input sequence and a decoder to generate predictions one step at a time. This is the default strategy and is suitable for most forecasting problems.

```python
model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    forecasting_strategy="seq2seq",
    target_len=12  # Predict 12 steps ahead
)
```

### 2. Autoregressive

The autoregressive approach generates predictions one step at a time, with each prediction used as input for the next step.

```python
model = ForecastingModel(
    encoder=None,  # No encoder needed
    decoder=ar_decoder,  # Custom autoregressive decoder
    forecasting_strategy="autoregressive",
    target_len=12
)
```

### 3. Direct

The direct approach predicts all future time steps at once using a single decoder. This is useful for models like Informer.

```python
model = ForecastingModel(
    encoder=None,  # Not used in direct strategy
    decoder=direct_decoder,  # Custom decoder for direct forecasting
    forecasting_strategy="direct",
    target_len=12
)
```

## Advanced Features

### Attention Mechanism

To enable attention mechanism in your forecasting model:

```python
attention_module = YourAttentionModule()  # Custom attention module

model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    attention_module=attention_module,
    forecasting_strategy="seq2seq"
)
```

The attention mechanism helps the model focus on relevant parts of the input sequence when making predictions.

### Multi-Encoder-Decoder Architecture

For multivariate time series with heterogeneous features, you can use the multi-encoder-decoder architecture:

```python
model = ForecastingModel(
    encoder=base_encoder,  # Base encoder to be cloned for each feature
    decoder=base_decoder,  # Base decoder to be cloned for each feature
    multi_encoder_decoder=True,
    input_processor_output_size=input_size,  # Number of features
    forecasting_strategy="seq2seq"
)
```

This creates a separate encoder-decoder pair for each input feature, which are then aggregated using a learnable linear layer.

### Teacher Forcing and Scheduled Sampling

Teacher forcing is a technique used during training where the ground truth is used as input to the decoder at each time step. You can control the teacher forcing ratio:

```python
model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    teacher_forcing_ratio=0.5  # Use teacher forcing 50% of the time
)
```

You can also implement scheduled sampling to gradually decrease the teacher forcing ratio during training:

```python
def scheduled_sampling_fn(epoch):
    # Gradually decrease teacher forcing ratio
    return max(0.0, 1.0 - 0.1 * epoch)

model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    teacher_forcing_ratio=1.0,
    scheduled_sampling_fn=scheduled_sampling_fn
)
```

### Transformer-based Models

The `ForecastingModel` also supports transformer-based architectures:

```python
from your_module import TransformerEncoder, TransformerDecoder, TimeSeriesEncoder

# Time series encoders
enc_embedding = TimeSeriesEncoder(input_size, d_model)
dec_embedding = TimeSeriesEncoder(input_size, d_model)

# Transformer modules
encoder = TransformerEncoder(...)
decoder = TransformerDecoder(...)

model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    model_type="transformer",
    forecasting_strategy="seq2seq",
    enc_embbedding=enc_embedding,
    dec_embedding=dec_embedding
)
```

## Complete Examples

### LSTM-based Seq2Seq Model with Attention

```python
import torch
import torch.nn as nn
from foreblocks import ForecastingModel, LSTMEncoder, LSTMDecoder, AttentionModule

# Parameters
input_size = 3  # Multivariate time series with 3 features
hidden_size = 64
num_layers = 2
output_size = 1  # Predict one feature
target_len = 24  # Forecast horizon of 24 time steps

# Create encoder and decoder
encoder = LSTMEncoder(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=True  # Use bidirectional LSTM for encoder
)

decoder = LSTMDecoder(
    input_size=output_size,
    hidden_size=hidden_size,
    output_size=hidden_size,  # Output size before final projection
    num_layers=num_layers
)

# Create attention module
attention = AttentionModule(
    hidden_size=hidden_size * 2,  # Doubled because of bidirectional encoder
    attention_type="general"
)

# Create model
model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    attention_module=attention,
    target_len=target_len,
    forecasting_strategy="seq2seq",
    teacher_forcing_ratio=0.5,
    output_size=output_size,
    output_block=nn.Sequential(
        nn.Dropout(0.1),
        nn.ReLU()
    )
)

# Example training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    # Get your batch data
    x_batch, y_batch = get_batch()  # Your data loading function
    
    # Forward pass
    outputs = model(x_batch, y_batch, epoch=epoch)
    
    # Calculate loss
    loss = criterion(outputs, y_batch)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Inference
with torch.no_grad():
    test_input = get_test_data()  # Your test data
    predictions = model(test_input)
```

### Transformer-based Model

```python
import torch
import torch.nn as nn
from your_module import (
    ForecastingModel, 
    TransformerEncoder, 
    TransformerDecoder, 
    TimeSeriesEncoder
)

# Parameters
input_size = 4  # Multivariate time series with 4 features
d_model = 128
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
output_size = 4  # Predict all features
target_len = 96  # Long forecast horizon

# Create time series encoders
enc_embedding = TimeSeriesEncoder(input_size, d_model)
dec_embedding = TimeSeriesEncoder(input_size, d_model)

# Create transformer encoder and decoder
encoder = TransformerEncoder(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=0.1,
    input_size=input_size
)

decoder = TransformerDecoder(
    d_model=d_model,
    nhead=nhead,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=0.1,
    output_size=d_model
)

# Create model
model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    model_type="transformer",
    forecasting_strategy="seq2seq",
    target_len=target_len,
    output_size=output_size,
    enc_embbedding=enc_embedding,
    dec_embedding=dec_embedding,
    teacher_forcing_ratio=0.7
)

# Training and inference similar to the LSTM example
```

## Troubleshooting

### Common Issues and Solutions

1. **Dimensionality Mismatch**:
   If you encounter dimension errors, check that your encoder and decoder have compatible hidden sizes, and that the output_size parameter matches your target dimensions.

2. **Memory Issues**:
   For long sequences or large batch sizes, you might encounter memory issues. Try reducing batch size or sequence length, or use gradient accumulation.

3. **Poor Performance**:
   - Try different forecasting strategies
   - Adjust teacher forcing ratio
   - Add attention mechanism
   - Experiment with different architectures (LSTM vs Transformer)
   - Tune hyperparameters like hidden size and number of layers

4. **Slow Training**:
   If training is slow, consider using a simpler model or reducing the target sequence length. For transformer models, you might need to optimize the d_model and number of layers.

5. **Vanishing/Exploding Gradients**:
   Use techniques like gradient clipping and proper initialization to mitigate these issues.

```python
# Example of gradient clipping
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Tips for Better Performance

1. **Data Normalization**: Always normalize your input data to improve model convergence.

2. **Learning Rate Scheduling**: Use learning rate scheduling to improve convergence.

3. **Evaluation Metrics**: Use appropriate metrics for time series forecasting (MAE, RMSE, MAPE).

4. **Multi-Step Validation**: Validate on multi-step predictions, not just one-step-ahead forecasts.

5. **Model Ensembling**: Combine predictions from multiple models for better performance.

Remember that time series forecasting is challenging, and different problems might require different approaches. Experiment with the various features of the `ForecastingModel` class to find what works best for your specific problem.