import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from foreblocks.transformer.norms.triton_backend import (
        TRITON_AVAILABLE,
        RMSNormTritonFunction,
        _should_use_triton,
    )
except Exception:  # pragma: no cover - foreblocks namespace may exclude transformer
    TRITON_AVAILABLE = False
    RMSNormTritonFunction = None

    def _should_use_triton(x, min_numel: int = 2048) -> bool:
        return False

from .norms import ChannelRMSNorm, RMSNorm


class ConvMixerOp(nn.Module):
    """ConvMixer-style operation with depthwise separable convolutions"""

    def __init__(self, input_dim: int, latent_dim: int, kernel_size: int = 9):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        self.depthwise = nn.Conv1d(
            latent_dim,
            latent_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=latent_dim,
            bias=False,
        )

        self.pointwise = nn.Conv1d(latent_dim, latent_dim, kernel_size=1, bias=False)

        self.norm1 = ChannelRMSNorm(latent_dim)
        self.norm2 = RMSNorm(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        residual = x

        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = self.norm1(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = self.pointwise(x_conv) + x_conv
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.dropout(x_conv)

        return self.norm2(x_conv + residual)


class GRNOp(nn.Module):
    """Gated Residual Network with proper gating"""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim, bias=False)
        self.fc2 = nn.Linear(latent_dim, latent_dim, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False), nn.Sigmoid()
        )

        self.norm = RMSNorm(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        h = self.activation(self.fc1(x))
        h = self.dropout(h)
        gated = self.gate(h)
        y = gated * self.fc2(h)

        return self.norm(y + residual)


class PatchEmbedOp(nn.Module):
    """PatchTST-style patch tokenization with interpolation back to sequence length."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        patch_size: int = 16,
        stride: int | None = None,
    ):
        super().__init__()
        self.patch_size = max(2, int(patch_size))
        self.stride = max(
            1, int(stride if stride is not None else self.patch_size // 2)
        )

        patch_dim = input_dim * self.patch_size
        self.patch_proj = nn.Linear(patch_dim, latent_dim, bias=False)
        self.mix = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim, bias=False),
            nn.Dropout(0.05),
        )
        self.norm = RMSNorm(latent_dim)
        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        residual = self.residual_proj(x)

        x_t = x.transpose(1, 2)
        if L < self.patch_size:
            x_t = F.pad(x_t, (0, self.patch_size - L))

        patches = x_t.unfold(dimension=2, size=self.patch_size, step=self.stride)
        patches = (
            patches.permute(0, 2, 1, 3).contiguous().reshape(B, -1, C * self.patch_size)
        )

        patch_tokens = self.patch_proj(patches)
        patch_tokens = self.mix(patch_tokens)

        y = F.interpolate(
            patch_tokens.transpose(1, 2),
            size=L,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        return self.norm(y + residual)


class InvertedAttentionOp(nn.Module):
    """iTransformer-style variate-dimension attention (attention across channels)."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.pre_norm = RMSNorm(latent_dim)

        self.channel_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.channel_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.channel_v = nn.Linear(latent_dim, latent_dim, bias=False)

        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim, bias=False),
            nn.Dropout(0.05),
        )
        self.norm = RMSNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        residual = x

        x_n = self.pre_norm(x)
        # Aggregate temporal evidence per variate/channel, then attend across variates.
        q = self.channel_q(x_n).transpose(1, 2)  # [B, D, L]
        k = self.channel_k(x_n).transpose(1, 2)
        v = self.channel_v(x_n).transpose(1, 2)

        scale = max(q.size(-1), 1) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, D, D]
        attn = F.softmax(attn, dim=-1)

        mixed = torch.matmul(attn, v).transpose(1, 2)  # [B, L, D]
        mixed = self.out_proj(mixed)

        y = residual + mixed
        y = y + self.ffn(self.norm(y))
        return self.norm(y)


class SwiGLUFFNOp(nn.Module):
    """SwiGLU FFN operation - state-of-the-art FFN variant with gated activation.

    First projects input_dim → latent_dim, then applies SwiGLU gating internally.
    SwiGLU(x) = SiLU(xW) ⊗ (xV) where W, V are split from a single projection.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Project to latent dim first
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.norm = nn.LayerNorm(latent_dim)

        # Gated FFN: split hidden into gate and value
        hidden = int(latent_dim * hidden_ratio)
        self.gate_value = nn.Linear(latent_dim, hidden * 2, bias=False)

        # Output projection
        self.output_proj = nn.Linear(hidden, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.norm(x)
        gv = self.gate_value(x)
        gate, value = gv.chunk(2, dim=-1)
        # SwiGLU: SiLU(gate) ⊗ value
        out = F.silu(gate) * value
        out = self.dropout(out)
        return self.output_proj(out) + x


class GeGLUFFNOp(nn.Module):
    """GeGLU FFN operation - gated variant with GELU activation.

    First projects input_dim → latent_dim, then applies GeGLU gating.
    GeGLU(x) = GELU(xW) ⊗ (xV)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Project to latent dim first
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.norm = nn.LayerNorm(latent_dim)

        # Gated FFN: split hidden into gate and value
        hidden = int(latent_dim * hidden_ratio)
        self.gate_value = nn.Linear(latent_dim, hidden * 2, bias=False)

        # Output projection
        self.output_proj = nn.Linear(hidden, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.norm(x)
        gv = self.gate_value(x)
        gate, value = gv.chunk(2, dim=-1)
        # GeGLU: GELU(gate) ⊗ value
        out = F.gelu(gate) * value
        out = self.dropout(out)
        return self.output_proj(out) + x


class GatedGeLUFFNOp(nn.Module):
    """Gated GELU FFN - a simpler gated variant.

    First projects input_dim → latent_dim, then applies GELU gating.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Project to latent dim first
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.norm = nn.LayerNorm(latent_dim)

        hidden = int(latent_dim * hidden_ratio)
        self.hidden_proj = nn.Linear(latent_dim, hidden, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.norm(x)
        hidden = self.hidden_proj(x)
        gate = self.gate(x)
        out = hidden * gate
        out = self.dropout(out)
        return self.output_proj(out) + x


