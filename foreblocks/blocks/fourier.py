import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List, Tuple, Union


# === Fourier Neural Operator (FNO1D) ===
class FNO1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.modes = modes
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.activation = nn.GELU()
        self.weight_real = nn.Parameter(torch.randn(out_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(torch.randn(out_channels, out_channels, modes))

    def compl_mul1d_real(self, a_r, a_i, b_r, b_i):
        return (
            torch.einsum("bcm,ocm->bom", a_r, b_r) - torch.einsum("bcm,ocm->bom", a_i, b_i),
            torch.einsum("bcm,ocm->bom", a_r, b_i) + torch.einsum("bcm,ocm->bom", a_i, b_r)
        )

    def forward(self, x):
        x = self.activation(self.fc1(x)).permute(0, 2, 1)
        x_ft = torch.fft.rfft(x, dim=-1)
        r, i = x_ft.real, x_ft.imag
        used_modes = min(self.modes, r.shape[-1])
        r, i = r[:, :, :used_modes], i[:, :, :used_modes]
        r_out, i_out = self.compl_mul1d_real(r, i, self.weight_real[:, :, :used_modes], self.weight_imag[:, :, :used_modes])
        out_ft = torch.zeros(x.shape[0], self.fc2.out_features, r.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :used_modes] = torch.complex(r_out, i_out)
        x = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1).permute(0, 2, 1)
        return self.activation(self.fc2(x))
    
# === Fourier Features Module ===
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
