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

from .norms import ChannelRMSNorm
from .norms import RMSNorm


class CausalConv1d(nn.Module):
    """Length-preserving causal Conv1d (left padding only)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        depthwise: bool = False,
        channel_multiplier: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if depthwise:
            groups = in_channels
            out_channels = in_channels * max(1, int(channel_multiplier))

        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups})"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )

        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TimeConvOp(nn.Module):
    """Depthwise-separable temporal convolution"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        kernel_size: int = 3,
        causal: bool = True,
    ):
        super().__init__()
        self.depthwise: nn.Module
        if causal:
            self.depthwise = CausalConv1d(
                input_dim,
                input_dim,
                kernel_size,
                dilation=1,
                groups=input_dim,
                bias=False,
            )
        else:
            self.depthwise = nn.Conv1d(
                input_dim,
                input_dim,
                kernel_size,
                padding=((kernel_size - 1) // 2),
                groups=input_dim,
                bias=False,
            )
        self.pointwise = nn.Conv1d(input_dim, latent_dim, 1, bias=False)

        self.norm = RMSNorm(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        residual = self.residual_proj(x)

        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = self.pointwise(x_conv)

        x_conv = x_conv.transpose(1, 2)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)

        return self.norm(x_conv + residual)


class TCNOp(nn.Module):
    """Temporal Convolutional Network with depthwise-separable and dilations"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        kernel_size: int = 3,
        causal: bool = True,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        self.dilations = list(dilations) if dilations is not None else [1, 2, 4]
        if len(self.dilations) == 0:
            self.dilations = [1, 2, 4]

        self.depthwise_layers = nn.ModuleList()
        self.pointwise_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_ch = input_dim
        for dilation in self.dilations:
            if causal:
                depthwise = CausalConv1d(
                    in_ch,
                    in_ch,
                    kernel_size,
                    dilation=dilation,
                    depthwise=True,
                    bias=False,
                )
            else:
                depthwise = nn.Conv1d(
                    in_ch,
                    in_ch,
                    kernel_size,
                    padding=((kernel_size - 1) // 2) * dilation,
                    dilation=dilation,
                    groups=in_ch,
                    bias=False,
                )

            self.depthwise_layers.append(depthwise)
            self.pointwise_layers.append(nn.Conv1d(in_ch, latent_dim, 1, bias=False))
            self.norms.append(RMSNorm(latent_dim))
            in_ch = latent_dim

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        current = x.transpose(1, 2)
        for depthwise, pointwise, norm in zip(
            self.depthwise_layers, self.pointwise_layers, self.norms
        ):
            current = depthwise(current)
            current = pointwise(current)
            current = current.transpose(1, 2)
            current = self.activation(norm(current))
            current = self.dropout(current)
            current = current.transpose(1, 2)

        output = current.transpose(1, 2)
        return output + residual


class MultiScaleConvOp(nn.Module):
    """Multi-scale convolution with attention-based fusion"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        scales: list[int] | None = None,
    ):
        super().__init__()
        self.scales = scales or [1, 3, 5, 7]
        self.num_scales = len(self.scales)
        self.latent_dim = latent_dim

        self.scale_convs = nn.ModuleList()
        for kernel_size in self.scales:
            conv = nn.Sequential(
                nn.Conv1d(
                    input_dim,
                    input_dim,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=input_dim,
                    bias=False,
                ),
                nn.Conv1d(
                    input_dim, latent_dim // self.num_scales, kernel_size=1, bias=False
                ),
                ChannelRMSNorm(latent_dim // self.num_scales),
                nn.GELU(),
            )
            self.scale_convs.append(conv)

        self.attention = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(latent_dim // 4, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.final_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=1, bias=False)
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

        scale_features = [conv(x_t) for conv in self.scale_convs]
        multi_scale = torch.cat(scale_features, dim=1)

        attn_weights = self.attention(multi_scale)

        weighted_features = [
            feat * attn_weights[:, i : i + 1, :]
            for i, feat in enumerate(scale_features)
        ]

        # Weighted fusion across scales preserves intended channel partitioning.
        combined = torch.cat(weighted_features, dim=1)

        output = self.final_proj(combined).transpose(1, 2)
        return self.norm(output + residual)


class PyramidConvOp(nn.Module):
    """Pyramid convolution with progressive downsampling and upsampling"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        levels: int = 3,
        upsample_mode: str = "nearest",
    ):
        super().__init__()
        self.levels = min(levels, 3)
        self.upsample_mode = (
            upsample_mode if upsample_mode in {"nearest", "linear"} else "nearest"
        )

        base_channels = max(latent_dim // (2**self.levels), 8)

        self.input_proj = nn.Conv1d(
            input_dim, base_channels * (2**self.levels), kernel_size=1, bias=False
        )

        encoder_channels = [
            base_channels * (2 ** (self.levels - i)) for i in range(self.levels + 1)
        ]
        self.encoder_convs = nn.ModuleList()
        for i in range(self.levels):
            in_ch, out_ch = encoder_channels[i], encoder_channels[i + 1]
            conv = nn.Sequential(
                nn.Conv1d(
                    in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False
                ),
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                ChannelRMSNorm(out_ch),
                nn.GELU(),
                nn.Dropout(0.05),
            )
            self.encoder_convs.append(conv)

        decoder_channels = encoder_channels[::-1]
        self.decoder_convs = nn.ModuleList()
        for i in range(self.levels):
            in_ch, out_ch = decoder_channels[i], decoder_channels[i + 1]
            conv = nn.Sequential(
                nn.ConvTranspose1d(
                    in_ch, out_ch, 3, stride=2, padding=1, output_padding=1, bias=False
                ),
                ChannelRMSNorm(out_ch),
                nn.GELU(),
            )
            self.decoder_convs.append(conv)

        self.skip_fusions = nn.ModuleList(
            [
                nn.Conv1d(
                    decoder_channels[i + 1] + encoder_channels[self.levels - 1 - i],
                    decoder_channels[i + 1],
                    kernel_size=1,
                    bias=False,
                )
                for i in range(self.levels - 1)
            ]
        )
        self.early_skip_projs = nn.ModuleList(
            [
                nn.Conv1d(
                    encoder_channels[self.levels - 1 - i],
                    decoder_channels[i + 1],
                    kernel_size=1,
                    bias=False,
                )
                for i in range(self.levels)
            ]
        )

        self.final_proj = nn.Conv1d(
            decoder_channels[-1], latent_dim, kernel_size=1, bias=False
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

        x_proj = self.input_proj(x_t)

        encoder_features = [x_proj]
        encoder_lengths = [x_proj.shape[-1]]
        current = x_proj
        for conv in self.encoder_convs:
            current = conv(current)
            encoder_features.append(current)
            encoder_lengths.append(current.shape[-1])

        current = encoder_features[-1]
        for i, conv in enumerate(self.decoder_convs):
            current = conv(current)

            # Explicitly recover the mirrored encoder resolution at each stage.
            target_len = encoder_lengths[self.levels - 1 - i]
            if current.shape[-1] != target_len:
                interp_kwargs = {"size": target_len, "mode": self.upsample_mode}
                if self.upsample_mode == "linear":
                    interp_kwargs["align_corners"] = False
                current = F.interpolate(current, **interp_kwargs)

            skip_idx = self.levels - 1 - i
            skip = encoder_features[skip_idx]
            current = current + self.early_skip_projs[i](skip)

            if i < len(self.decoder_convs) - 1:
                fused = torch.cat([current, skip], dim=1)
                current = self.skip_fusions[i](fused)

        current = self.final_proj(current)
        if current.shape[-1] != L:
            interp_kwargs = {"size": L, "mode": self.upsample_mode}
            if self.upsample_mode == "linear":
                interp_kwargs["align_corners"] = False
            current = F.interpolate(current, **interp_kwargs)

        output = current.transpose(1, 2)
        return self.norm(output + residual)


