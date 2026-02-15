import math

import torch
import torch.nn as nn


def get_lowest_modes(seq_len: int, num_modes: int) -> int:
    """Return the actual number of modes to keep (lowest frequencies)."""
    return min(num_modes, (seq_len // 2) + 1)


class SpectralConv1d(nn.Module):
    """
    1D Fourier layer. Does FFT -> linear transform in freq domain -> IFFT.
    Keeps only the lowest `modes` frequencies (standard FNO behavior).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of Fourier modes to multiply

        scale = 1 / math.sqrt(in_channels * out_channels)
        self.scale = scale
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, L]
        B, C_in, L = x.shape

        # Real FFT
        x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")  # [B, C_in, L//2+1]
        out_ft = torch.zeros(
            B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device
        )

        # Truncate modes
        modes = min(self.modes, x_ft.shape[-1])

        # Complex multiplication in frequency domain
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )

        # Inverse real FFT
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1, norm="ortho")  # [B, C_out, L]

        return x_out


class FNO1dLayer(nn.Module):
    """
    Standard FNO-style block with:
    - Spectral convolution
    - Skip connection (with 1×1 conv if channels change)
    - LayerNorm + activation
    Input & output are both [B, L, C]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        modes: int = 24,
        activation: str = "gelu",
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.norm = nn.LayerNorm(out_channels)

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() in ("silu", "swish"):
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        residual = x

        # To channel-first
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.spectral_conv(x)  # [B, C_out, L]
        skip = self.skip(residual.permute(0, 2, 1))  # [B, C_out, L]

        x = x + skip
        x = x.permute(0, 2, 1)  # [B, L, C_out]

        x = self.norm(x)
        x = self.act(x)

        return x


class FourierFeatures(nn.Module):
    """
    Classical Fourier Feature Encoding (à la Tancik et al.)
    for injecting high-frequency information into time-series / coordinate inputs.
    Output shape: [B, L, output_size]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_freqs: int = 10,
        scale: float = 10.0,
        learnable_freqs: bool = False,
        use_phase: bool = False,
        freq_init: str = "linear",
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs

        # Frequency initialization
        if freq_init == "linear":
            freqs = torch.linspace(1.0, scale, num_freqs)
        elif freq_init == "log":
            freqs = torch.logspace(0, math.log10(scale), num_freqs)
        else:
            freqs = torch.linspace(1.0, scale, num_freqs)  # fallback

        self.freqs = nn.Parameter(
            freqs.repeat(input_dim, 1), requires_grad=learnable_freqs
        )

        if use_phase:
            self.phase = nn.Parameter(torch.zeros(input_dim, num_freqs))
        else:
            self.phase = None

        enc_dim = 2 * input_dim * num_freqs  # sin + cos per channel per freq

        layers = [nn.Linear(input_dim + enc_dim, output_dim)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if activation.lower() == "gelu":
            layers.append(nn.GELU())
        elif activation.lower() in ("silu", "swish"):
            layers.append(nn.SiLU())
        else:
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape

        # Normalized time [0,1]
        t = torch.linspace(0, 1, L, device=x.device, dtype=x.dtype)
        t = t.view(1, L, 1).expand(B, L, D)

        # [B, L, D, F]
        angles = 2.0 * math.pi * t.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)

        if self.phase is not None:
            angles = angles + self.phase.unsqueeze(0).unsqueeze(0)

        sin_feats = torch.sin(angles)
        cos_feats = torch.cos(angles)

        fourier_feats = torch.cat([sin_feats, cos_feats], dim=-1)  # [B,L,D,2F]
        fourier_feats = fourier_feats.flatten(start_dim=2)  # [B,L, D*2F]

        combined = torch.cat([x, fourier_feats], dim=-1)
        out = self.net(combined)

        return out


# Optional small wrapper for stacking FNO layers
class FNO1dStack(nn.Module):
    def __init__(
        self,
        width: int = 64,
        depth: int = 6,
        modes: int = 24,
        in_channels: int = 1,
        out_channels: int = 1,
        lifting_channels: int | None = None,
    ):
        super().__init__()
        lifting_channels = lifting_channels or width

        self.lifting = (
            nn.Linear(in_channels, lifting_channels)
            if in_channels != lifting_channels
            else nn.Identity()
        )

        layers = []
        for _ in range(depth):
            layers.append(FNO1dLayer(lifting_channels, lifting_channels, modes=modes))

        self.fno_layers = nn.Sequential(*layers)

        self.projection = nn.Linear(lifting_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, in_channels]
        x = self.lifting(x)
        x = self.fno_layers(x)
        x = self.projection(x)
        return x
