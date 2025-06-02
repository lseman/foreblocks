# Advanced Fourier Neural Networks for Time Series

## Overview

Fourier Neural Networks (FNNs) leverage the mathematical properties of the Fourier transform to process time series data in the frequency domain. Unlike traditional approaches that operate solely in the time domain, FNNs transform data to frequency space, apply learned transformations, and convert back to the time domain. This approach is particularly powerful for capturing periodic patterns, long-range dependencies, and multi-scale temporal dynamics.

## Mathematical Foundation

### Discrete Fourier Transform in Neural Networks

For a discrete signal **x[n]** of length **N**, the Discrete Fourier Transform (DFT) is:

```
X[k] = Σ(n=0 to N-1) x[n] * e^(-2πikn/N)
```

Where:
- **X[k]** is the frequency domain representation
- **k** represents the frequency bin
- **i** is the imaginary unit

The key insight is that convolution in the time domain becomes element-wise multiplication in the frequency domain, making certain operations more efficient.

---

## Core Architecture Components

### 1. SpectralConv1D - Optimized Frequency Domain Convolution

```
Input Signal [B, C, L]
         |
         v
+------------------+
|   FFT Transform  |  ← torch.fft.rfft()
+------------------+
         |
         v
   Complex Spectrum [B, C, L//2+1]
         |
         v
+------------------+
| Complex Weights  |  ← Learnable complex parameters
| Multiplication   |
+------------------+
         |
         v
+------------------+
|  IFFT Transform  |  ← torch.fft.irfft()
+------------------+
         |
         v
Output Signal [B, L, C_out]
```

**Key Optimizations:**
- **Complex-valued weights**: Direct complex parameter storage for efficiency
- **Vectorized operations**: Einstein summation for batch processing
- **Truncated spectrum**: Process only relevant frequency modes

```python
# Optimized complex multiplication using einsum
out_ft_trunc = torch.einsum("bcm,com->bom", x_ft_trunc, self.weights[:, :, :modes])
```

### 2. FNO1DLayer - Complete Processing Unit

```
Input [B, L, C_in]
        |
        +-- Residual Path ----+
        |                    |
        v                    |
+----------------+           |
| Permute Dims   |           |
| [B,L,C]→[B,C,L]|           |
+----------------+           |
        |                    |
        v                    |
+----------------+           |
| SpectralConv1D |           |
+----------------+           |
        |                    |
        v                    |
+----------------+           |
| Back to [B,L,C]|           |
+----------------+           |
        |                    |
        v                    v
+----------------+  +----------------+
|     Add        |  | Residual Proj  |
|   Residual     |<-| (if needed)    |
+----------------+  +----------------+
        |
        v
+----------------+
|  Layer Norm    |
+----------------+
        |
        v
+----------------+
|   GELU Act     |
+----------------+
        |
        v
Output [B, L, C_out]
```

---

## Advanced Frequency Processing

### 3. FourierBlock - Selective Frequency Processing

**Frequency Mode Selection Strategies:**

| Method | Description | Use Case |
|--------|-------------|----------|
| **Lowest Modes** | Select frequencies [0, 1, 2, ..., k] | Global patterns, trends |
| **Random Sampling** | Randomly sample k frequencies | Diverse pattern capture |
| **Adaptive** | Learn which frequencies to use | Task-specific optimization |

```
Input [B, L, C]
        |
        v
+----------------------+
|    FFT Transform     |
+----------------------+
        |
        v
+----------------------+
| Frequency Selection  |  ← Only process selected modes
| modes = [0,2,5,7...] |
+----------------------+
        |
        v
+----------------------+
| Complex Arithmetic   |  ← Manual complex multiplication
| Real: xr*wr - xi*wi  |
| Imag: xr*wi + xi*wr  |
+----------------------+
        |
        v
+----------------------+
|   IFFT Transform     |
+----------------------+
        |
        v
Output [B, L, C_out]
```

**Vectorized Processing:**
```python
# Process all selected frequencies simultaneously
real_part = torch.einsum("bcf,cof->bof", xr, wr) - torch.einsum("bcf,cof->bof", xi, wi)
imag_part = torch.einsum("bcf,cof->bof", xr, wi) + torch.einsum("bcf,cof->bof", xi, wr)
```

---

## Fourier Feature Engineering

### 4. FourierFeatures - Enhanced Positional Encoding

**Mathematical Foundation:**

For input **x** and time **t**, the Fourier encoding is:

```
signal = 2π * t * f + φ
encoding = [sin(signal), cos(signal)]
```

Where:
- **f** are learnable frequencies
- **φ** are optional phase shifts

**Frequency Initialization Strategies:**

```
Linear:     f = linspace(1, scale, num_freq)
Log:        f = logspace(0, log10(scale), num_freq)
Geometric:  f = scale^linspace(0, 1, num_freq)
Random:     f = rand(num_freq) * scale
Gaussian:   f = randn(num_freq) * scale/sqrt(input_size)
```

**Architecture Flow:**

```
Input Features [B, L, D]
        |
        +-- Original Path --------+
        |                        |
        v                        |
+------------------+             |
| Time Generation  |             |
| t = [0,1/L,2/L...] |           |
+------------------+             |
        |                        |
        v                        |
+------------------+             |
| Frequency Matrix |             |
| [D, num_freq]    |             |
+------------------+             |
        |                        |
        v                        |
+------------------+             |
| Sin/Cos Encoding |             |
| [B,L,2*D*F]      |             |
+------------------+             |
        |                        |
        v                        |
+------------------+             |
| Layer Norm       |             |
| (optional)       |             |
+------------------+             |
        |                        |
        v                        v
+------------------+    +------------------+
|    Concatenate   |<---| Original Features|
| [x, fourier_enc] |    |                  |
+------------------+    +------------------+
        |
        v
+------------------+
| Projection MLP   |  ← 1-N layer network
+------------------+
        |
        v
Output [B, L, output_size]
```

### 5. AdaptiveFourierFeatures - Attention-Weighted Encoding

**Key Innovation: Frequency-Content Attention**

```
Input [B, L, D]
        |
        v
+------------------+
| Query Generation |  ← Linear(input_features)
| Q = W_q * x      |
+------------------+
        |
        v
+------------------+
| Frequency Embeds |  ← For each frequency f_i
| K = W_k * f      |
| V = W_v * f      |
+------------------+
        |
        v
+------------------+
| Multi-Head Attn  |  ← Attention(Q, K, V)
| Weights: [B,L,F] |
+------------------+
        |
        v
+------------------+
| Weighted Fourier |  ← Apply attention to sin/cos
| Features         |
+------------------+
        |
        v
+------------------+
| Gated Output     |  ← gate * projection + residual
+------------------+
        |
        v
Enhanced Features [B, L, output_size]
```

**Attention Mechanism:**
- **Query**: Generated from input features (what pattern to look for)
- **Key/Value**: Generated from frequency embeddings (available frequencies)
- **Output**: Dynamically weighted sinusoidal features

---

## Implementation Optimizations

### Memory Efficiency Improvements

| Optimization | Traditional | Optimized | Benefit |
|--------------|-------------|-----------|---------|
| **Complex Weights** | Separate real/imag | Native complex | 50% memory reduction |
| **Vectorized Ops** | Loop over modes | Einstein summation | 3-5x speedup |
| **In-place FFT** | Copy operations | Direct transforms | 30% memory savings |
| **Frequency Caching** | Recompute each step | Cache computations | 2x faster inference |

### Numerical Stability Features

**1. Gradient-Safe Complex Operations**
```python
# Avoid gradient issues with complex arithmetic
weights = nn.Parameter(torch.randn(..., dtype=torch.cfloat))
```

**2. Proper Frequency Scaling**
```python
# Prevent frequency explosion
scale = 1 / math.sqrt(in_channels * out_channels)
frequencies = scale * frequencies
```

**3. Normalized Time Sequences**
```python
# Consistent time normalization
time = torch.linspace(0, 1, seq_len)  # Always [0,1] range
```

---

## Advanced Configuration Examples

### High-Performance Time Series Model
```python
# Optimized for speed and accuracy
fourier_layer = FNO1DLayer(
    in_channels=64,
    out_channels=64,
    modes=32  # Process 32 frequency modes
)

fourier_features = FourierFeatures(
    input_size=10,
    output_size=64,
    num_frequencies=16,
    freq_init="log",        # Better for natural signals
    use_layernorm=True,     # Numerical stability
    projector_layers=2,     # Deep feature extraction
    activation="silu"       # Smooth activation
)
```

### Research Configuration
```python
# Maximum flexibility for experimentation
fourier_block = FourierBlock(
    in_channels=32,
    out_channels=64,
    seq_len=1024,
    modes=64,
    mode_select_method="random"  # Explore diverse patterns
)

adaptive_features = AdaptiveFourierFeatures(
    input_size=10,
    output_size=64,
    num_frequencies=32,
    freq_attention_heads=8,  # Rich attention patterns
    attention_dim=64,
    use_gaussian=True,       # Random frequency exploration
    dropout=0.1
)
```

### Efficiency-Focused Setup
```python
# Minimal computation for real-time applications
efficient_fno = FNO1DLayer(
    in_channels=16,
    out_channels=16,
    modes=8  # Only essential frequencies
)

simple_features = FourierFeatures(
    input_size=5,
    output_size=16,
    num_frequencies=4,
    freq_init="linear",
    use_layernorm=False,   # Skip normalization
    projector_layers=1,    # Single projection
    dropout=0.0
)
```

---

## Frequency Domain Analysis

### Understanding Frequency Modes

**Low Frequencies (0-10% of spectrum):**
- Capture global trends and seasonality
- Essential for long-term dependencies
- Most energy concentrated here

**Mid Frequencies (10-50% of spectrum):**
- Capture periodic patterns
- Business cycles, weekly patterns
- Balance between global and local

**High Frequencies (50%+ of spectrum):**
- Capture noise and rapid changes
- Often less informative
- Can be safely truncated


---

## Applications and Use Cases

### 1. Periodic Time Series
- **Weather data**: Daily/seasonal cycles
- **Financial markets**: Trading patterns
- **IoT sensors**: Regular monitoring cycles

### 2. Long-Range Dependencies
- **Climate modeling**: Multi-year patterns
- **Economic forecasting**: Long-term trends
- **Astronomical data**: Orbital periods

### 3. Multi-Scale Analysis
- **Medical signals**: ECG, EEG patterns
- **Network traffic**: Multiple time scales
- **Audio processing**: Harmonic content

---

## Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| **FFT Transform** | O(N log N) | O(N) |
| **Spectral Conv** | O(N × modes) | O(modes) |
| **Traditional Conv** | O(N × kernel) | O(kernel) |

**Advantage**: For long sequences (N > 1000), FFT-based operations become more efficient than traditional convolutions.

### Memory Usage Patterns

```
Memory Usage by Component:

Fourier Features: O(input_size × num_frequencies)
Spectral Conv:    O(channels × modes)
FFT Buffers:      O(batch × channels × seq_len)
Gradients:        O(parameters) [same as forward]

Total: Typically 2-3x traditional CNN memory
```

---

## Best Practices

### 1. Frequency Mode Selection
- **Start with lowest modes** for global patterns
- **Add random modes** for pattern diversity
- **Monitor frequency usage** during training

### 2. Initialization Strategies
- **Log-spaced frequencies** for natural signals
- **Gaussian frequencies** for exploration
- **Proper scaling** to prevent gradient issues

### 3. Training Tips
- **Warm-up learning rates** for frequency parameters
- **Separate learning rates** for different components
- **Gradient clipping** for numerical stability

### 4. Debugging and Analysis
- **Visualize frequency spectra** of learned weights
- **Monitor attention patterns** in adaptive features
- **Track frequency utilization** across training

---

## Integration with Transformers

### Fourier-Enhanced Attention
```python
class FourierTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, modes=32):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, nhead)
        self.fourier_conv = SpectralConv1D(d_model, d_model, modes)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Standard attention
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Fourier processing
        fourier_out = self.fourier_conv(x)
        x = self.norm2(x + fourier_out)

        return x
```
