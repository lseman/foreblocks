import torch
import torch.nn as nn
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1 / math.sqrt(in_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x):
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        L_ft = x_ft.shape[-1]
        modes = min(self.modes, L_ft)

        x_r = x_ft[:, :, :modes].real
        x_i = x_ft[:, :, :modes].imag
        w_r = self.weight_real[:, :, :modes]
        w_i = self.weight_imag[:, :, :modes]

        out_r = torch.einsum('bcm,com->bom', x_r, w_r) - torch.einsum('bcm,com->bom', x_i, w_i)
        out_i = torch.einsum('bcm,com->bom', x_r, w_i) + torch.einsum('bcm,com->bom', x_i, w_r)

        out_ft = torch.zeros(B, self.out_channels, L_ft, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes] = torch.complex(out_r, out_i)

        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)
        return x_out.permute(0, 2, 1)

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

    def forward(self, x):
        # x: [B, L, C_in]
        residual = x
        x = x.permute(0, 2, 1)  # [B, C_in, L]
        x = self.spectral(x)  # [B, L, C_out]

        # Residual path (after projecting if needed)
        residual = self.residual_proj(residual.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + residual  # [B, L, C_out]
        x = self.act(self.norm(x))
        return x


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

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        x_ft = torch.fft.rfft(x, dim=-1)  # [B, C, L//2+1], complex

        # Prepare output FFT buffer
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)

        # Apply real-valued linear transformation in frequency domain
        for i, freq_idx in enumerate(self.index):
            if freq_idx >= x_ft.shape[-1]:
                continue

            xr = x_ft[:, :, freq_idx].real  # [B, in_channels]
            xi = x_ft[:, :, freq_idx].imag  # [B, in_channels]

            wr = self.weight_real[:, :, i]  # [in_channels, out_channels]
            wi = self.weight_imag[:, :, i]  # [in_channels, out_channels]

            # Re(Y) = Xr*Wr - Xi*Wi
            # Im(Y) = Xr*Wi + Xi*Wr
            real_part = torch.einsum('bi,io->bo', xr, wr) - torch.einsum('bi,io->bo', xi, wi)
            imag_part = torch.einsum('bi,io->bo', xr, wi) + torch.einsum('bi,io->bo', xi, wr)

            out_ft[:, :, freq_idx] = torch.complex(real_part, imag_part)

        # Inverse FFT
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)  # [B, out_channels, L]
        return x_out.permute(0, 2, 1)  # [B, L, out_channels]



class FourierFeatures(nn.Module):
    """
    Enhanced Fourier Features module for time series encoding.
    
    This module projects time series onto learned frequency components,
    capturing periodic patterns at different frequencies. Improvements include:
    - Better initialization of frequencies
    - Layer normalization for stability
    - Support for multiple positional encoding styles
    - Device-safe implementation
    - Auto-scale frequencies based on sequence length
    - Adjustable projector architecture
    """
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
        """
        Initialize the enhanced FourierFeatures module.
        
        Args:
            input_size: Number of features in input
            output_size: Dimension of output features
            num_frequencies: Number of frequency components to learn
            learnable: Whether frequencies are learnable parameters
            use_phase: Include learnable phase shifts
            use_gaussian: Use random Gaussian frequencies (Rahimi & Recht style)
            freq_init: Initialization method ("linear", "log", "geometric", "random")
            freq_scale: Scale factor for frequency initialization
            use_layernorm: Apply layer normalization to stabilize training
            dropout: Dropout rate for projection layers
            projector_layers: Number of layers in projection MLP
            time_dim: Dimension along which time flows (0, 1, or 2)
            activation: Activation function ("relu", "gelu", "silu", "tanh")
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_frequencies = num_frequencies
        self.use_phase = use_phase
        self.use_gaussian = use_gaussian
        self.freq_scale = freq_scale
        self.use_layernorm = use_layernorm
        self.time_dim = time_dim
        
        # Initialize frequency components
        if use_gaussian:
            # Gaussian random Fourier features (Rahimi & Recht)
            self.freq_matrix = nn.Parameter(
                torch.randn(input_size, num_frequencies) * freq_scale * 0.1,
                requires_grad=learnable
            )
        else:
            # Initialize frequencies based on specified method
            if freq_init == "linear":
                # Linear spacing (good for data with regular sampling)
                freqs = torch.linspace(1.0, freq_scale, num_frequencies)
            elif freq_init == "log":
                # Log spacing (better for capturing patterns across scales)
                freqs = torch.exp(torch.linspace(
                    0, math.log(freq_scale), num_frequencies
                ))
            elif freq_init == "geometric":
                # Geometric spacing (similar to log, but different distribution)
                freqs = torch.tensor([
                    freq_scale ** (i / (num_frequencies - 1)) 
                    for i in range(num_frequencies)
                ])
            elif freq_init == "random":
                # Uniformly random (more diverse frequency coverage)
                freqs = torch.rand(num_frequencies) * freq_scale
            else:
                raise ValueError(f"Unknown freq_init: {freq_init}")
                
            # Use the same frequencies for all input dimensions
            self.freq_matrix = nn.Parameter(
                freqs.repeat(input_size, 1),
                requires_grad=learnable
            )
        
        # Phase shifts
        if use_phase:
            self.phase = nn.Parameter(
                torch.zeros(input_size, num_frequencies),
                requires_grad=True
            )
        else:
            self.register_parameter('phase', None)
        
        # Pre-compute Fourier feature dimensionality
        self.fourier_dim = 2 * input_size * num_frequencies
        
        # Layer normalization
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(self.fourier_dim)
        else:
            self.layer_norm = nn.Identity()
        
        # Create projection MLP with the specified number of layers
        if projector_layers == 1:
            self.projection = nn.Sequential(
                nn.Linear(input_size + self.fourier_dim, output_size),
                self._get_activation(activation),
                nn.Dropout(dropout)
            )
        else:
            layers = []
            # Input layer
            layers.append(nn.Linear(input_size + self.fourier_dim, output_size * 2))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for _ in range(projector_layers - 2):
                layers.append(nn.Linear(output_size * 2, output_size * 2))
                layers.append(self._get_activation(activation))
                layers.append(nn.Dropout(dropout))
                
            # Output layer
            layers.append(nn.Linear(output_size * 2, output_size))
            
            self.projection = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation.lower(), nn.SiLU())
    
    def _normalize_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate normalized time indices based on sequence length.
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            
        Returns:
            Normalized time tensor suitable for frequency computation
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        # Create linear time sequence from 0 to 1
        time = torch.linspace(0, 1, seq_len, device=device)
        
        # Reshape based on time_dim
        if self.time_dim == 0:
            # Time flows along batch dimension (unusual, but supported)
            time = time.unsqueeze(1).unsqueeze(2)  # [seq_len, 1, 1]
            time = time.expand(-1, batch, self.input_size)  # [seq_len, batch, input_size]
            time = time.permute(1, 0, 2)  # [batch, seq_len, input_size]
        else:
            # Default: time flows along sequence dimension
            time = time.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1]
            time = time.expand(batch, -1, self.input_size)  # [batch, seq_len, input_size]
            
        return time
        
    def forward(self, x: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Fourier feature encoding to the input.
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            time: Optional explicit time tensor
            
        Returns:
            Encoded features [batch, seq_len, output_size]
        """
        batch, seq_len, in_dim = x.shape
        device = x.device
        
        assert in_dim == self.input_size, f"Expected input_size={self.input_size}, got {in_dim}"
        
        # Ensure parameters are on the correct device
        freq_matrix = self.freq_matrix.to(device)
        
        # Get time indices (either provided or generated)
        if time is None:
            time = self._normalize_time(x)
        
        # Add time dimension for broadcasting with frequencies
        time = time.unsqueeze(-1)  # [batch, seq_len, input_size, 1]
        
        # Apply frequency modulation
        # Reshape freq_matrix for broadcasting: [1, 1, input_size, num_frequencies]
        freqs = freq_matrix.unsqueeze(0).unsqueeze(0)
        
        # Compute time * frequency: [batch, seq_len, input_size, num_frequencies]
        signal = 2 * math.pi * time * freqs
        
        # Add phase shift if used
        if self.phase is not None:
            phase = self.phase.to(device)
            signal = signal + phase.unsqueeze(0).unsqueeze(0)
        
        # Apply sinusoidal encoding
        sin_feat = torch.sin(signal)
        cos_feat = torch.cos(signal)
        
        # Combine and reshape: [batch, seq_len, input_size * 2 * num_frequencies]
        fourier_encoded = torch.cat([sin_feat, cos_feat], dim=-1)
        fourier_encoded = fourier_encoded.flatten(start_dim=2)
        
        # Apply layer normalization if enabled
        if self.use_layernorm:
            fourier_encoded = self.layer_norm(fourier_encoded)
        
        # Concatenate with original features
        combined = torch.cat([x, fourier_encoded], dim=-1)
        
        # Apply projection MLP
        output = self.projection(combined)
        
        return output

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
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_frequencies = num_frequencies
        self.use_phase = use_phase

        # Frequency matrix
        if use_gaussian:
            self.freq_matrix = nn.Parameter(torch.randn(input_size, num_frequencies) * 10.0)
        else:
            freqs = torch.linspace(1.0, 10.0, num_frequencies)
            self.freq_matrix = nn.Parameter(freqs.repeat(input_size, 1), requires_grad=learnable)

        # Optional learnable phase
        self.phase = nn.Parameter(torch.randn(input_size, num_frequencies)) if use_phase else None

        # Frequency scaling
        self.freq_scale = nn.Parameter(torch.ones(input_size, num_frequencies))

        # Attention projections
        self.query_proj = nn.Linear(input_size, attention_dim)
        self.key_proj = nn.Linear(1, attention_dim)
        self.value_proj = nn.Linear(1, attention_dim)
        self.attn = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=freq_attention_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Final projection
        self.gate = nn.Sequential(
            nn.Linear(input_size + 2 * input_size * num_frequencies, output_size),
            nn.Sigmoid()
        )
        self.projection = nn.Sequential(
            nn.Linear(input_size + 2 * input_size * num_frequencies, output_size),
            nn.SiLU()
        )

        self.attn_weights_log = None  # Placeholder for attention weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, in_dim = x.shape
        device = x.device
        assert in_dim == self.input_size

        # Time indices
        time = torch.linspace(0, 1, seq_len, device=device).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        time = time.expand(batch, -1, in_dim)  # [batch, seq_len, input_size]

        queries = self.query_proj(x)  # [batch, seq_len, attn_dim]

        fourier_features_list = []

        for i in range(in_dim):
            freqs_i = self.freq_matrix[i] * self.freq_scale[i]  # [num_freq]
            phases_i = self.phase[i] if self.phase is not None else torch.zeros_like(freqs_i)
            time_i = time[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]

            # Signal = 2pi * freq * time + phase
            signal = 2 * math.pi * time_i * freqs_i.view(1, 1, -1) + phases_i.view(1, 1, -1)  # [B, T, F]
            sin_features = torch.sin(signal)
            cos_features = torch.cos(signal)

            # Prepare keys and values
            freq_embeds = freqs_i.view(-1, 1)  # [F, 1]
            keys = self.key_proj(freq_embeds)  # [F, attn_dim]
            values = self.value_proj(freq_embeds)  # [F, attn_dim]
            keys = keys.unsqueeze(0).expand(batch, -1, -1)  # [B, F, attn_dim]
            values = values.unsqueeze(0).expand(batch, -1, -1)  # [B, F, attn_dim]

            # Multihead attention
            attn_out, attn_weights = self.attn(queries, keys, values)  # attn_weights: [B, T, F]
            attn_weights = self.dropout(attn_weights)  # ✅ CORRECT

            # Weighted sinusoidal features
            sin_weighted = sin_features * attn_weights
            cos_weighted = cos_features * attn_weights
            combined = torch.cat([sin_weighted, cos_weighted], dim=-1)  # [B, T, 2F]

            fourier_features_list.append(combined)

        fourier_features = torch.cat(fourier_features_list, dim=2)  # [B, T, input_size * 2 * num_freq]
        combined_input = torch.cat([x, fourier_features], dim=2)  # [B, T, input_size + enriched]

        gated = self.gate(combined_input) * self.projection(combined_input)
        self.attn_weights_log = attn_weights
        return x + gated
