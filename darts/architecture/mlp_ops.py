import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from foreblocks.ops.norms_triton import (
        TRITON_AVAILABLE,
        RMSNormTritonFunction,
        _should_use_triton,
    )
except Exception:  # pragma: no cover - foreblocks namespace may exclude transformer
    TRITON_AVAILABLE = False
    RMSNormTritonFunction = None

    def _should_use_triton(x, min_numel: int = 2048) -> bool:
        return False

from .norms import RMSNorm


class IdentityOp(nn.Module):
    """Identity operation with optional dimension projection"""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ResidualMLPOp(nn.Module):
    """MLP with residual connection and proper scaling"""

    def __init__(self, input_dim: int, latent_dim: int, expansion_factor: float = 2.67):
        super().__init__()
        hidden_dim = int(latent_dim * expansion_factor)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim, bias=False),
            nn.Dropout(0.05),
        )
        self.norm = RMSNorm(latent_dim)
        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        output = self.mlp(x)
        return self.norm(output + residual)


class MLPMixerOp(nn.Module):
    """TimeMixer/FITS-style MLP mixing across time and channels."""

    def __init__(self, input_dim: int, latent_dim: int, seq_length: int):
        super().__init__()
        self.seq_length = seq_length
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        self.token_mix = nn.Sequential(
            nn.Conv1d(
                latent_dim,
                latent_dim,
                kernel_size=5,
                padding=2,
                groups=latent_dim,
                bias=False,
            ),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.channel_mix = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim, bias=False),
            nn.Dropout(0.05),
        )

        self.norm1 = RMSNorm(latent_dim)
        self.norm2 = RMSNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        token_res = x
        token_out = self.token_mix(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = token_res + token_out

        channel_res = x
        channel_out = self.channel_mix(self.norm2(x))
        return channel_res + channel_out


