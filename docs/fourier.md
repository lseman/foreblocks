# Understanding Fourier Neural Networks for Time Series

## Introduction

Fourier Neural Networks (FNNs) are neural network architectures that utilize the Fourier transform to process signals in the frequency domain. Unlike convolutional neural networks that operate in the spatial domain, FNNs transform data to the frequency domain, apply learned filters, and then transform back to the spatial domain.

This document explains the components from the provided code implementation.

## Basic Building Blocks

### 1. SpectralConv1D

The `SpectralConv1D` class implements convolution operations in the frequency domain.

```python
class SpectralConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1 / math.sqrt(in_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
```

#### Function

1. **Fourier Transform**: Transforms the input signal from the time domain to the frequency domain using Fast Fourier Transform (FFT).
2. **Complex-valued Convolution**: Applies a learned filter in the frequency domain.
3. **Inverse Transform**: Transforms the filtered signal back to the time domain.

#### Mathematical Formulation

For an input signal $x \in \mathbb{R}^{B \times C \times L}$:

1. Compute the Fourier transform: $\hat{x} = \mathcal{F}(x)$
2. Apply the complex-valued filter:
   $\hat{y}_r = \hat{x}_r W_r - \hat{x}_i W_i$
   $\hat{y}_i = \hat{x}_r W_i + \hat{x}_i W_r$
   where $\hat{x}_r$ and $\hat{x}_i$ are the real and imaginary parts of $\hat{x}$, and $W_r$ and $W_i$ are learnable weights.
3. Compute the inverse Fourier transform: $y = \mathcal{F}^{-1}(\hat{y})$

### 2. Frequency Mode Selection

The `get_frequency_modes` function selects which frequency components to use in the Fourier transform.

```python
def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index
```

#### Selection Methods

1. **Random Selection**: Randomly samples frequency components, which can capture diverse patterns.
2. **Lowest Modes**: Selects the lowest frequency components, which typically capture more global patterns in the data.

#### Parameters

- `seq_len`: Length of the input sequence
- `modes`: Number of frequency components to select
- `mode_select_method`: Method for selecting frequency components

## Layer Implementations

### 3. FNO1DLayer

The `FNO1DLayer` class integrates spectral convolution with residual connections, normalization, and non-linear activation.

```python
class FNO1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.spectral = SpectralConv1D(in_channels, out_channels, modes)

        # Align residual channels if needed
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
```

#### Components

1. **Spectral Convolution**: Processes the input in the frequency domain.
2. **Residual Connection**: Adds the input to the output of the spectral convolution.
   - Uses a 1×1 convolution if dimensions don't match.
3. **Normalization**: Applies layer normalization to stabilize training.
4. **Activation**: Applies GELU (Gaussian Error Linear Unit) non-linearity.

#### Forward Pass

The forward pass includes:
1. Permuting dimensions for spectral convolution
2. Applying spectral convolution
3. Adding the residual connection
4. Applying normalization and activation

### 4. FourierBlock

The `FourierBlock` class implements an alternative approach to spectral convolution with explicit frequency mode selection.

```python
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=16, mode_select_method='random'):
        super().__init__()
        print('FourierBlock (real-valued weights, AMP-compatible) initialized.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.index = get_frequency_modes(seq_len, modes, mode_select_method)
        self.modes = len(self.index)

        scale = 1 / math.sqrt(in_channels * out_channels)

        # Use real-valued weights for both real and imaginary parts
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))
```

#### Features

1. **Selective Frequency Processing**: Only processes specific frequency modes selected via `get_frequency_modes`.
2. **AMP Compatibility**: Uses real-valued weights for better compatibility with automatic mixed precision training.
3. **Explicit Frequency Domain Operations**: Manually handles complex arithmetic for clearer implementation.

#### Forward Process

In the forward pass, this module:
1. Transforms the input to the frequency domain
2. Processes only the selected frequency modes
3. Reconstructs the output in the frequency domain
4. Transforms back to the time domain

## Fourier Feature Encoding

### 5. FourierFeatures

The `FourierFeatures` class implements positional encoding using Fourier basis functions.

```python
class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_frequencies: int = 10,
        learnable: bool = True,
        use_phase: bool = True,
        use_gaussian: bool = False,
        freq_init: str = "linear",
        freq_scale: float = 10.0,
        use_layernorm: bool = True,
        dropout: float = 0.1,
        projector_layers: int = 1,
        time_dim: int = 1,
        activation: str = "silu"
    ):
```

#### Frequency Initialization Methods

1. **Linear**: Equally spaced frequencies from 1.0 to `freq_scale`
2. **Log**: Logarithmically spaced frequencies
3. **Geometric**: Geometrically spaced frequencies
4. **Random**: Uniformly random frequencies
5. **Gaussian**: Random frequencies sampled from a Gaussian distribution

#### Components

1. **Frequency Matrix**: Stores the frequencies for each input dimension
2. **Phase Shifts**: Optional learnable phase offsets
3. **Layer Normalization**: Stabilizes training by normalizing features
4. **Projection MLP**: Maps Fourier features to the desired output dimension

#### Mathematical Formulation

For input $x$ and time $t$:

1. Calculate signal: $s = 2\pi \cdot t \cdot f + \phi$, where $f$ are frequencies and $\phi$ are phase shifts
2. Apply sinusoidal encoding: $e = [\sin(s), \cos(s)]$
3. Flatten and normalize the encoding
4. Concatenate with original features: $[x, e]$
5. Apply projection MLP

### 6. AdaptiveFourierFeatures

The `AdaptiveFourierFeatures` class extends `FourierFeatures` with attention mechanisms to adaptively weight different frequency components.

```python
class AdaptiveFourierFeatures(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_frequencies: int = 16,
        learnable: bool = True,
        use_phase: bool = True,
        use_gaussian: bool = False,
        dropout: float = 0.1,
        freq_attention_heads: int = 4,
        attention_dim: int = 32,
    ):
```

#### Key Components

1. **Frequency Matrix and Phase**: Similar to `FourierFeatures`
2. **Frequency Scaling**: Learnable scaling factors for each frequency
3. **Attention Mechanism**: 
   - Query: Generated from input features
   - Key/Value: Generated from frequency embeddings
   - Multi-head attention: Captures relationships between input features and frequencies
4. **Gating Mechanism**: Controls information flow

#### Attention-Based Feature Selection

The attention mechanism allows the model to:
1. Focus on relevant frequencies for each input feature
2. Dynamically adjust the importance of frequency components based on input context
3. Learn different representations for different parts of the input

#### Mathematical Formulation

1. Generate queries from input: $Q = W_q \cdot x$
2. Generate keys and values from frequencies: $K = W_k \cdot f$, $V = W_v \cdot f$
3. Compute attention weights: $A = \text{softmax}(\frac{QK^T}{\sqrt{d}})$
4. Apply attention to sinusoidal features: $F = A \cdot [\sin(s), \cos(s)]$
5. Apply gating: $G = \sigma(W_g \cdot [x, F]) \odot \text{SiLU}(W_p \cdot [x, F])$

## Comparison and Applications

### Comparison of Approaches

| Class | Main Technique | Key Features |
|-------|---------------|-------------|
| `SpectralConv1D` | Basic spectral convolution | Direct implementation of complex-valued filtering in frequency domain |
| `FNO1DLayer` | Layer-based integration | Adds residual connections, normalization, and activation to spectral convolution |
| `FourierBlock` | Selective frequency processing | Processes specific frequency modes for efficiency |
| `FourierFeatures` | Fourier basis encoding | Transforms inputs using sinusoidal functions at various frequencies |
| `AdaptiveFourierFeatures` | Attention-weighted encoding | Dynamically weights frequency components using attention mechanisms |

### Applications in Time Series Analysis

These Fourier-based neural network components are particularly suitable for:

1. **Periodic Pattern Recognition**: Capturing cyclical patterns in data
2. **Multi-scale Analysis**: Simultaneously analyzing patterns at different time scales
3. **Long-range Dependencies**: Efficiently modeling dependencies across long sequences
4. **Continuous Time Series**: Handling irregularly sampled or continuous-time data

### Implementation Considerations

1. **Computational Efficiency**: FFT-based operations can be more efficient than conventional convolutions for long sequences
2. **Complex Number Handling**: Proper handling of complex numbers is essential
3. **Mode Selection**: The choice of frequency modes affects model capacity and efficiency
4. **Activation Functions**: Non-linearities are applied after returning to the time domain

## Conclusion

Fourier Neural Networks represent an approach to sequence modeling that leverages frequency domain processing. By transforming data to the frequency domain, applying learned filters, and transforming back, these networks can efficiently capture periodic patterns and long-range dependencies in time series data.