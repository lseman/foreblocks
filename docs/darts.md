# ForeBlocks DARTS: Neural Architecture Search for Time Series

**ForeBlocks DARTS** is an advanced Neural Architecture Search (NAS) module that automatically discovers optimal neural network architectures for time series forecasting. It combines DARTS (Differentiable Architecture Search) with zero-cost metrics for efficient architecture evaluation.

---

## 🎯 Overview

This module implements a multi-fidelity architecture search strategy that:
- **Generates** diverse neural architectures automatically
- **Evaluates** architectures using zero-cost metrics (no training required)
- **Optimizes** promising candidates with differentiable search
- **Derives** final high-performance architectures

### Key Features

| Feature | Description |
|---------|-------------|
| **🚀 Zero-Cost Evaluation** | Screen architectures without training using 10+ metrics |
| **🔍 Multi-Fidelity Search** | Progressive refinement from zero-cost to full training |
| **🧩 Modular Operations** | Attention, Spectral, Convolution, and MLP components |
| **⚡ Efficient Screening** | Evaluate 100+ architectures in minutes |
| **🎯 Automated Discovery** | No manual architecture engineering required |
| **📊 Comprehensive Metrics** | 10+ zero-cost indicators for architecture quality |

---

## 🚀 Quick Start

```python
from foreblocks.darts import multi_fidelity_darts_search
import torch

# Your data loaders
train_loader = ...  # Your training DataLoader
val_loader = ...    # Your validation DataLoader
test_loader = ...   # Your test DataLoader

# Run architecture search
results = multi_fidelity_darts_search(
    input_dim=5,                    # Number of input features
    hidden_dims=[32, 64, 128],      # Hidden layer sizes to explore
    forecast_horizon=24,            # Steps to predict ahead
    seq_length=48,                  # Input sequence length
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_candidates=20,              # Architectures to evaluate
    full_train_epochs=50           # Final training epochs
)

# Get the best discovered architecture
best_model = results['final_model']
```

---

## 📖 Architecture Search Process

### Phase 1: Candidate Generation
The search begins by generating diverse neural architectures from our comprehensive operation library:

```python
# Available operations for time series
operations = [
    "Identity",       # Skip connection
    "TimeConv",       # Temporal convolution
    "GRN",           # Gated Residual Network
    "Wavelet",       # Wavelet transform
    "Fourier",       # Fourier transform
    "Attention",     # Self-attention
    "TCN",           # Temporal Convolutional Network
    "ResidualMLP",   # Residual MLP block
    "ConvMixer",     # ConvMixer block
    "Transformer"    # Full transformer encoder
]
```

### Phase 2: Zero-Cost Evaluation
Each candidate is evaluated using multiple metrics **without training**:

```python
metrics = evaluate_zero_cost_metrics(model, val_loader, device)
# Returns: synflow, naswot, grasp, fisher, jacob_cov, etc.
```

### Phase 3: DARTS Optimization
Top candidates undergo differentiable architecture search:

```python
results = train_darts_model(
    model=candidate_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    arch_learning_rate=3e-4,
    model_learning_rate=1e-3
)
```

### Phase 4: Final Training
The best architecture is derived and trained extensively:

```python
final_results = train_final_model(
    model=derived_model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=100
)
```

---

## 🔧 Zero-Cost Metrics

The module uses 10+ zero-cost metrics to evaluate architecture quality:

### Core Metrics

| Metric | Description | What it Measures |
|--------|-------------|------------------|
| **SynFlow** | Synaptic flow preservation | Network expressivity |
| **NASWOT** | Neural Architecture Search Without Training | Activation diversity |
| **GraSP** | Gradient Signal Preservation | Gradient flow quality |
| **Fisher** | Fisher Information | Parameter sensitivity |
| **SNIP** | Connection sensitivity | Pruning importance |
| **Jacob_Cov** | Jacobian covariance | Feature diversity |
| **Sensitivity** | Input perturbation response | Model robustness |
| **Weight_Cond** | Weight matrix conditioning | Optimization landscape |
| **Param_Count** | Parameter efficiency | Model complexity |
| **FLOPs** | Computational cost | Inference efficiency |

### Metric Aggregation

Metrics are combined using learned weights:

```python
# Dataset-specific weights (automatically applied)
weights = {
    "synflow": 0.3,
    "naswot": 0.25,
    "grasp": 0.2,
    "fisher": 0.25,
    # ... other metrics
}

aggregate_score = sum(metric_value * weight for metric, weight in weights.items())
```

---

## ⚙️ Configuration Options

### Basic Configuration

```python
# Simple search configuration
results = multi_fidelity_darts_search(
    input_dim=3,
    hidden_dims=[64],
    forecast_horizon=12,
    seq_length=24,
    num_candidates=10,
    full_train_epochs=30
)
```

### Advanced Configuration

```python
# Comprehensive search with custom parameters
results = multi_fidelity_darts_search(
    input_dim=10,                   # Input features
    hidden_dims=[32, 64, 128, 256], # Multiple model sizes
    forecast_horizon=48,            # Long-term forecasting
    seq_length=96,                  # Long input sequences
    num_candidates=50,              # Extensive search
    full_train_epochs=100,          # Thorough training
    device="cuda"                   # GPU acceleration
)
```

### DARTS Training Configuration

```python
# Fine-tune DARTS training process
train_results = train_darts_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    arch_learning_rate=3e-4,        # Architecture parameter LR
    model_learning_rate=1e-3,       # Model weight LR
    arch_weight_decay=1e-3,         # Architecture regularization
    model_weight_decay=1e-4,        # Model regularization
    patience=10,                    # Early stopping patience
    tau_max=1.0,                    # Gumbel-Softmax temperature
    tau_min=0.1,
    loss_type="huber",              # Loss function
    use_swa=True                    # Stochastic Weight Averaging
)
```

---

## 📊 Understanding Results

### Search Results Structure

```python
results = {
    'final_model': model,              # Best discovered architecture
    'candidates': [...],               # All evaluated architectures
    'top_candidates': [...],           # Top performers from zero-cost
    'trained_candidates': [...],       # DARTS-optimized models
    'best_candidate': {...}            # Selected best architecture
}
```

### Candidate Information

```python
candidate = {
    'model': model,                    # PyTorch model
    'metrics': {...},                  # Zero-cost metric values
    'score': 0.85,                     # Aggregate score
    'selected_ops': [...],             # Operations in architecture
    'hidden_dim': 64,                  # Hidden layer size
    'num_cells': 2,                    # Number of DARTS cells
    'num_nodes': 3                     # Nodes per cell
}
```

---

## 🎯 Advanced Features

### Custom Operation Sets

Define your own operation pool:

```python
# Custom operations for domain-specific problems
custom_ops = [
    "Identity",
    "Attention",
    "Wavelet",
    "CustomTimeOp"  # Your custom operation
]

# Use in architecture generation
model = TimeSeriesDARTS(
    input_dim=input_dim,
    selected_ops=custom_ops,
    # ... other parameters
)
```

### Operation Expansion

Intelligently expand operation sets based on performance:

```python
# Analyze what operations work well
op_importance = [
    ("Attention", 0.95),
    ("Wavelet", 0.87),
    ("GRN", 0.82)
]

# Expand based on categories and complementary ops
expanded_ops = expand_operations(
    initial_ops=["Identity", "Attention"],
    op_importance=op_importance,
    top_k=3,
    max_ops=8
)
```

### Metric Weight Adaptation

Learn optimal metric weights from your data:

```python
# Track performance across architectures
metrics_history = track_nas_performance(
    metrics_history={},
    arch_name="arch_1",
    metrics=computed_metrics
)

# Adapt weights based on actual performance correlation
optimal_weights = adapt_metric_weights(
    metrics_history=metrics_history,
    val_accuracies={"arch_1": 0.95, "arch_2": 0.87},
    epochs=10,
    lr=0.01
)
```

---

## 🔍 Zero-Cost Metric Details

### SynFlow
Measures synaptic flow preservation through the network:

```python
def compute_synflow(model, inputs):
    # Set parameters to absolute values
    # Compute gradient flow
    # Return flow preservation score
```

### NASWOT (Neural Architecture Search Without Training)
Evaluates activation diversity:

```python
def compute_naswot(model, inputs):
    # Capture layer activations
    # Compute binary activation matrices
    # Calculate matrix ranks
```

### GraSP (Gradient Signal Preservation)
Measures gradient flow quality:

```python
def compute_grasp(model, inputs, targets):
    # Compute first and second-order gradients
    # Analyze gradient preservation
    # Handle transformer architectures
```

### Transformer-Aware Processing
Special handling for attention-based models:

```python
# Detects transformer architectures automatically
if is_transformer_model(model):
    score = compute_grasp_transformer_aware(model, inputs, targets)
else:
    score = compute_grasp(model, inputs, targets)
```

---


## 🧩 Neural Building Blocks

ForeBlocks DARTS includes a comprehensive library of neural operations specifically designed for time series forecasting. Each operation captures different temporal patterns and can be automatically combined during the search process.

### Temporal Operations

| Operation | Description | Best For |
|-----------|-------------|----------|
| **TimeConv** | Standard 1D temporal convolution | Local temporal patterns |
| **TCN** | Temporal Convolutional Network with dilation | Multi-scale temporal dependencies |
| **ConvMixer** | Depthwise + pointwise convolution | Efficient feature mixing |

### Attention Mechanisms

| Operation | Description | Best For |
|-----------|-------------|----------|
| **Attention** | Self-attention with multi-head support | Long-range dependencies |
| **Transformer** | Full transformer encoder block | Complex sequential patterns |

### Spectral Analysis

| Operation | Description | Best For |
|-----------|-------------|----------|
| **Wavelet** | Multi-scale analysis via dilated convolutions | Multi-resolution temporal features |
| **Fourier** | Frequency domain analysis with FFT | Periodic and cyclical patterns |

### Advanced Neural Blocks

| Operation | Description | Best For |
|-----------|-------------|----------|
| **GRN** | Gated Residual Network with sophisticated gating | Complex feature transformations |
| **ResidualMLP** | MLP with skip connections | Non-linear feature mapping |
| **Identity** | Simple projection or skip connection | Preserving input information |

### Operation Characteristics

```python
# Example: Creating a custom operation mix
custom_ops = [
    "Identity",      # Always include for stability
    "Attention",     # For long-range dependencies
    "Wavelet",       # For multi-scale analysis
    "GRN",          # For complex transformations
    "Fourier"       # For frequency patterns
]

model = TimeSeriesDARTS(
    input_dim=5,
    selected_ops=custom_ops,
    # ... other parameters
)
```

### Smart Design Features

- **Dimension Consistency**: All operations output to the same latent dimension
- **GPU Optimization**: Efficient implementations with memory-safe operations
- **Fallback Mechanisms**: Robust alternatives for complex dependencies
- **Progressive Complexity**: Can start with simple ops and expand based on performance


---

## 🚨 Troubleshooting

### Common Issues

<details>
<summary><strong>🔴 CUDA Out of Memory</strong></summary>

**Problem**: GPU memory exhaustion during search

**Solutions**:
- Reduce `num_candidates`
- Use smaller `hidden_dims`
- Decrease batch size in data loaders
- Enable gradient checkpointing

```python
# Memory-efficient configuration
results = multi_fidelity_darts_search(
    num_candidates=10,              # Reduced candidates
    hidden_dims=[32, 64],           # Smaller models
    # Use smaller batch sizes in your loaders
)
```
</details>

<details>
<summary><strong>🟡 Zero-Cost Metric Errors</strong></summary>

**Problem**: Metrics return -1.0 (error sentinel)

**Solutions**:
- Check model architecture compatibility
- Ensure proper input/target shapes
- Verify device consistency

```python
# Debug zero-cost metrics
metrics = evaluate_zero_cost_metrics(
    model=model,
    dataloader=val_loader,
    device=device,
    num_batches=1,              # Start with single batch
    batch_size=4                # Small batch for debugging
)
```
</details>

<details>
<summary><strong>🟠 Architecture Search Convergence</strong></summary>

**Problem**: Search doesn't find good architectures

**Solutions**:
- Increase `num_candidates`
- Expand operation pool
- Adjust zero-cost metric weights
- Use more training epochs

```python
# Enhanced search configuration
results = multi_fidelity_darts_search(
    num_candidates=30,              # More exploration
    full_train_epochs=100,          # Longer training
    # Custom metric weights
)
```
</details>

---

## 📈 Performance Tips

### Efficient Search Strategy
- Start with 10-20 candidates for initial exploration
- Use multiple `hidden_dims` to explore model sizes
- Enable mixed precision training (`autocast`)
- Use Stochastic Weight Averaging for final models

### Zero-Cost Metric Optimization
- Evaluate on representative validation batches
- Use multiple batches for stable metric estimates
- Consider dataset-specific metric weights
- Monitor correlation between metrics and actual performance

### Memory Management
- Use gradient checkpointing for large models
- Clear GPU cache between evaluations
- Process candidates in smaller batches
- Use CPU for metric computation when needed

---

## 🔬 Research Extensions

### Custom Metrics
Add domain-specific zero-cost metrics:

```python
def compute_custom_metric(model, inputs, targets):
    """Your custom architecture evaluation metric"""
    # Implement your metric logic
    return score

# Add to evaluation pipeline
metrics["custom_metric"] = compute_custom_metric(model, inputs, targets)
```

### Multi-Objective Search
Optimize for multiple objectives:

```python
# Weight accuracy vs efficiency
weights = {
    "performance_metrics": 0.7,
    "efficiency_metrics": 0.3
}
```

### Progressive Search
Implement progressive architecture refinement:

```python
# Stage 1: Coarse search
initial_results = multi_fidelity_darts_search(num_candidates=50, epochs=20)

# Stage 2: Refine best candidates
refined_results = multi_fidelity_darts_search(
    initial_architectures=initial_results['top_candidates'],
    epochs=100
)
```

---

## 📚 References

This implementation builds upon several key research papers:

- **DARTS**: Liu et al. "DARTS: Differentiable Architecture Search" (ICLR 2019)
- **Zero-Cost NAS**: Mellor et al. "Neural Architecture Search without Training" (ICML 2021)
- **SynFlow**: Tanaka et al. "Pruning neural networks without any data" (NeurIPS 2020)
- **GraSP**: Wang et al. "Picking Winning Tickets Before Training" (ICLR 2020)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 🔧 Additional zero-cost metrics
- 🏗️ New neural operations for time series
- 📊 Enhanced visualization tools
- ⚡ Performance optimizations
- 📝 Documentation improvements

---

## 📄 License

This module is part of ForeBlocks and follows the same MIT License.

---
