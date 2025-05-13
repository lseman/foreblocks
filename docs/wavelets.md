# Wavelet Neural Networks for Time Series

This document explains the implementation of wavelet-based neural network components for time series processing.

## 1. Mathematical Foundation

### 1.1 Wavelet Basis Functions

```python
def get_phi_psi(k, base):
    """
    Constructs wavelet basis functions.
    
    Args:
        k: Number of wavelet components
        base: 'legendre' or 'chebyshev' polynomial basis
        
    Returns:
        phi: Scaling functions
        psi1: Wavelet functions for [0, 0.5]
        psi2: Wavelet functions for [0.5, 1]
    """
```

This function creates orthogonal wavelet basis functions using:
- Legendre polynomials: Based on orthogonal polynomials defined on [-1, 1]
- Chebyshev polynomials: Known for minimizing approximation error

### 1.2 Filter Bank Generation

```python
def get_filter(base, k):
    """
    Generates wavelet filter banks.
    
    Args:
        base: 'legendre' or 'chebyshev'
        k: Filter size
        
    Returns:
        H0, H1: Decomposition filters
        G0, G1: Reconstruction filters
        PHI0, PHI1: Scaling matrices
    """
```

The filter banks consist of:
- Decomposition filters (H0, G0): Split signal into approximation and detail coefficients
- Reconstruction filters (H1, G1): Combine approximation and detail to recover the signal

## 2. Core Neural Network Components

### 2.1 sparseKernelFT1d

```python
class sparseKernelFT1d(nn.Module):
    """
    Applies Fourier Transform to process wavelet coefficients efficiently.
    
    Args:
        k: Wavelet dimension
        alpha: Number of Fourier modes
        c: Channel multiplier
    """
```

This module:
1. Transforms input to frequency domain
2. Applies learned complex-valued weights to selected modes
3. Transforms back to time domain

### 2.2 MWT_CZ1d (Multi-Wavelet Transform)

```python
class MWT_CZ1d(nn.Module):
    """
    Multi-level wavelet transform with learnable components.
    
    Args:
        k: Wavelet dimension
        alpha: Number of Fourier modes
        L: Coarsest wavelet decomposition level
        c: Channel multiplier
        base: Wavelet basis ('legendre' or 'chebyshev')
    """
```

This module implements:
1. Wavelet decomposition using filter banks
2. Learnable processing of wavelet coefficients using Fourier kernels
3. Wavelet reconstruction to recover the transformed signal

## 3. Practical Implementation: MultiWaveletFeatureExtractor

```python
class MultiWaveletFeatureExtractor(nn.Module):
    """
    Simplified wavelet-based feature extractor.
    
    Args:
        input_channels: Number of input features
        wavelet_dim: Dimensionality of wavelet space
        wavelet_modes: Number of wavelet modes
        alpha: Number of Fourier modes
        L: Coarsest decomposition level
        n_layers: Number of wavelet transform layers
        base: Wavelet basis function
        out_channels: Number of output features
    """
```

This high-level module provides a complete pipeline for wavelet-based feature extraction:

1. **Input Projection**: Maps input data to wavelet coefficient space
   ```python
   self.project_in = nn.Linear(input_channels, self.c * self.k)
   ```

2. **Multi-layer Wavelet Processing**: Applies sequential wavelet transforms
   ```python
   self.wavelet_blocks = nn.ModuleList([
       MWT_CZ1d(k=self.k, alpha=alpha, L=L, c=self.c, base=base)
       for _ in range(n_layers)
   ])
   ```

3. **Output Projection**: Maps processed wavelet features to output space
   ```python
   self.project_out = nn.Linear(self.c * self.k, out_channels)
   ```

### Forward Process

```python
def forward(self, x):
    """
    x: [B, L, C]
    Returns: [B, L, C] — extracted wavelet-based features
    """
    B, L, C = x.shape
    x_proj = self.project_in(x)             # [B, L, c * k]
    x_wavelet = x_proj.view(B, L, self.c, self.k)

    for i, layer in enumerate(self.wavelet_blocks):
        x_wavelet = layer(x_wavelet)
        if i < self.n_layers - 1:
            x_wavelet = F.relu(x_wavelet)

    x_flat = x_wavelet.view(B, L, -1)       # [B, L, c * k]
    out = self.project_out(x_flat)          # [B, L, C]
    return out
```

This forward process:
1. Projects input features to wavelet space
2. Applies multiple wavelet transform layers with non-linear activations
3. Projects processed wavelet features back to output space

## Applications

This wavelet-based feature extractor is particularly effective for:

1. **Time Series Classification**: Extracting multi-scale patterns from signals
2. **Anomaly Detection**: Identifying unusual patterns at different time scales
3. **Signal Compression**: Representing signals with fewer coefficients
4. **Noise Reduction**: Separating signal from noise at different scales