import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SpectralConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Optimized: Use complex weights directly for better efficiency
        scale = 1 / math.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        L_ft = x_ft.shape[-1]
        modes = min(self.modes, L_ft)

        # Optimized: Vectorized complex multiplication
        x_ft_trunc = x_ft[:, :, :modes]  # [B, C, modes]
        out_ft_trunc = torch.einsum(
            "bcm,com->bom", x_ft_trunc, self.weights[:, :, :modes]
        )

        # Reconstruct full spectrum
        out_ft = torch.zeros(
            B, self.out_channels, L_ft, device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :modes] = out_ft_trunc

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


def get_frequency_modes(seq_len, modes=64, mode_select_method="random"):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == "random":
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class FourierBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, seq_len, modes=16, mode_select_method="random"
    ):
        super().__init__()
        print("FourierBlock (optimized, AMP-compatible) initialized.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.index = get_frequency_modes(seq_len, modes, mode_select_method)
        self.modes = len(self.index)

        # Optimized: Better initialization scale
        scale = 1 / math.sqrt(in_channels * out_channels)

        # Use real-valued weights for AMP compatibility
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes)
        )

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        x_ft = torch.fft.rfft(x, dim=-1)  # [B, C, L//2+1], complex

        # Prepare output FFT buffer
        out_ft = torch.zeros(
            B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat
        )

        # Optimized: Vectorized frequency processing
        if self.modes > 0:
            # Extract valid frequency indices
            valid_indices = [i for i in self.index if i < x_ft.shape[-1]]
            if valid_indices:
                # Batch process all valid frequencies
                freq_indices = torch.tensor(valid_indices, device=x.device)
                mode_indices = torch.arange(len(valid_indices), device=x.device)

                # Extract frequency components for all valid indices
                x_freq = x_ft[:, :, freq_indices]  # [B, in_channels, num_valid_freqs]
                xr = x_freq.real
                xi = x_freq.imag

                # Get corresponding weights
                wr = self.weight_real[
                    :, :, mode_indices
                ]  # [in_channels, out_channels, num_valid_freqs]
                wi = self.weight_imag[:, :, mode_indices]

                # Vectorized complex multiplication: (xr + j*xi) * (wr + j*wi)
                real_part = torch.einsum("bcf,cof->bof", xr, wr) - torch.einsum(
                    "bcf,cof->bof", xi, wi
                )
                imag_part = torch.einsum("bcf,cof->bof", xr, wi) + torch.einsum(
                    "bcf,cof->bof", xi, wr
                )

                # Assign results back
                out_ft[:, :, freq_indices] = torch.complex(real_part, imag_part)

        # Inverse FFT
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)  # [B, out_channels, L]
        return x_out.permute(0, 2, 1)  # [B, L, out_channels]


class FourierFeatures(nn.Module):
    """
    Enhanced Fourier Features module for time series encoding.
    Optimized version with better memory efficiency and numerical stability.
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
        activation: str = "silu",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_frequencies = num_frequencies
        self.use_phase = use_phase
        self.use_gaussian = use_gaussian
        self.freq_scale = freq_scale
        self.use_layernorm = use_layernorm
        self.time_dim = time_dim

        # Optimized frequency initialization
        if use_gaussian:
            # Better scaling for Gaussian random features
            self.freq_matrix = nn.Parameter(
                torch.randn(input_size, num_frequencies)
                * freq_scale
                / math.sqrt(input_size),
                requires_grad=learnable,
            )
        else:
            # Optimized frequency initialization with better numerical properties
            if freq_init == "linear":
                freqs = torch.linspace(1.0, freq_scale, num_frequencies)
            elif freq_init == "log":
                # Log spacing often works better for natural signals
                freqs = torch.logspace(0, math.log10(freq_scale), num_frequencies)
            elif freq_init == "geometric":
                freqs = freq_scale ** torch.linspace(0, 1, num_frequencies)
            elif freq_init == "random":
                freqs = torch.rand(num_frequencies) * freq_scale
            else:
                raise ValueError(f"Unknown freq_init: {freq_init}")

            self.freq_matrix = nn.Parameter(
                freqs.repeat(input_size, 1), requires_grad=learnable
            )

        # Phase shifts
        if use_phase:
            self.phase = nn.Parameter(
                torch.zeros(input_size, num_frequencies), requires_grad=True
            )
        else:
            self.register_parameter("phase", None)

        # Pre-compute Fourier feature dimensionality
        self.fourier_dim = 2 * input_size * num_frequencies

        # Optimized normalization
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(self.fourier_dim)
        else:
            self.layer_norm = nn.Identity()

        # Optimized projection layers
        if projector_layers == 1:
            self.projection = nn.Sequential(
                nn.Linear(input_size + self.fourier_dim, output_size),
                self._get_activation(activation),
                nn.Dropout(dropout),
            )
        else:
            # Better intermediate dimension sizing
            hidden_dim = max(output_size, (input_size + self.fourier_dim) // 2)
            layers = []
            layers.append(nn.Linear(input_size + self.fourier_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))

            for _ in range(projector_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self._get_activation(activation))
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_dim, output_size))
            self.projection = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(activation.lower(), nn.SiLU())

    def _normalize_time(self, x: torch.Tensor) -> torch.Tensor:
        """Generate normalized time indices based on sequence length."""
        batch, seq_len, _ = x.shape
        device = x.device

        # Create linear time sequence from 0 to 1
        time = torch.linspace(0, 1, seq_len, device=device)

        # Optimized broadcasting
        if self.time_dim == 0:
            time = (
                time.view(seq_len, 1, 1)
                .expand(-1, batch, self.input_size)
                .permute(1, 0, 2)
            )
        else:
            time = time.view(1, seq_len, 1).expand(batch, -1, self.input_size)

        return time

    def forward(
        self, x: torch.Tensor, time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Fourier feature encoding to the input.
        Optimized for memory efficiency and numerical stability.
        """
        batch, seq_len, in_dim = x.shape
        device = x.device

        assert (
            in_dim == self.input_size
        ), f"Expected input_size={self.input_size}, got {in_dim}"

        # Get time indices
        if time is None:
            time = self._normalize_time(x)

        # Optimized vectorized computation
        # Reshape for broadcasting: time [B, L, D, 1], freq [1, 1, D, F]
        time_exp = time.unsqueeze(-1)  # [B, L, D, 1]
        freq_exp = self.freq_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, D, F]

        # Compute angles
        signal = 2 * math.pi * time_exp * freq_exp

        # Add phase if used
        if self.phase is not None:
            signal = signal + self.phase.unsqueeze(0).unsqueeze(0)

        # Optimized: Compute sin and cos simultaneously
        sin_feat = torch.sin(signal)
        cos_feat = torch.cos(signal)

        # Flatten and concatenate: [B, L, 2*D*F]
        fourier_encoded = torch.cat([sin_feat, cos_feat], dim=-1).flatten(start_dim=2)

        # Apply normalization
        if self.use_layernorm:
            fourier_encoded = self.layer_norm(fourier_encoded)

        # Concatenate with original features and project
        combined = torch.cat([x, fourier_encoded], dim=-1)
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

        # Optimized frequency initialization
        if use_gaussian:
            scale = 1.0 / math.sqrt(input_size)
            self.freq_matrix = nn.Parameter(
                torch.randn(input_size, num_frequencies) * scale * 10.0
            )
        else:
            # Better frequency distribution
            freqs = torch.logspace(0, math.log10(10.0), num_frequencies)
            self.freq_matrix = nn.Parameter(
                freqs.repeat(input_size, 1), requires_grad=learnable
            )

        # Optional learnable phase
        self.phase = (
            nn.Parameter(torch.zeros(input_size, num_frequencies))
            if use_phase
            else None
        )

        # Learnable frequency scaling
        self.freq_scale = nn.Parameter(torch.ones(input_size, num_frequencies))

        # Optimized attention with proper dimensions
        self.query_proj = nn.Linear(input_size, attention_dim)
        self.key_proj = nn.Linear(1, attention_dim)
        self.value_proj = nn.Linear(1, attention_dim)

        # Use scaled dot-product attention for efficiency
        self.attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=freq_attention_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        # Optimized final projection with gating
        fourier_dim = 2 * input_size * num_frequencies
        self.gate = nn.Sequential(
            nn.Linear(input_size + fourier_dim, output_size),
            nn.Sigmoid(),
        )
        self.projection = nn.Sequential(
            nn.Linear(input_size + fourier_dim, output_size),
            nn.SiLU(),
        )

        self.attn_weights_log = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, in_dim = x.shape
        device = x.device
        assert in_dim == self.input_size

        # Optimized time generation
        time = torch.linspace(0, 1, seq_len, device=device)
        time = time.view(1, seq_len, 1).expand(batch, -1, in_dim)

        # Pre-compute attention components
        queries = self.query_proj(x)  # [B, L, attn_dim]

        # Process all dimensions efficiently
        fourier_features_list = []

        for i in range(in_dim):
            # Get frequency parameters for dimension i
            freqs_i = self.freq_matrix[i] * self.freq_scale[i]
            phases_i = (
                self.phase[i] if self.phase is not None else torch.zeros_like(freqs_i)
            )
            time_i = time[:, :, i : i + 1]  # [B, L, 1]

            # Compute signal
            signal = 2 * math.pi * time_i * freqs_i.view(1, 1, -1) + phases_i.view(
                1, 1, -1
            )
            sin_features = torch.sin(signal)
            cos_features = torch.cos(signal)

            # Attention over frequencies
            freq_embeds = freqs_i.view(-1, 1)
            keys = self.key_proj(freq_embeds).unsqueeze(0).expand(batch, -1, -1)
            values = self.value_proj(freq_embeds).unsqueeze(0).expand(batch, -1, -1)

            # Apply attention
            attn_out, attn_weights = self.attn(queries, keys, values)
            attn_weights = self.dropout(attn_weights)

            # Weight the sinusoidal features
            sin_weighted = sin_features * attn_weights
            cos_weighted = cos_features * attn_weights
            combined = torch.cat([sin_weighted, cos_weighted], dim=-1)

            fourier_features_list.append(combined)

        # Combine all Fourier features
        fourier_features = torch.cat(fourier_features_list, dim=2)
        combined_input = torch.cat([x, fourier_features], dim=2)

        # Gated output
        gate_weights = self.gate(combined_input)
        projection_out = self.projection(combined_input)
        gated_out = gate_weights * projection_out

        # Store attention weights for analysis
        self.attn_weights_log = attn_weights

        return x + gated_out
