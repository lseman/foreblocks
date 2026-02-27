from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Simple RMS normalization"""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.scale * x / (norm + self.eps)


class ChannelRMSNorm(nn.Module):
    """RMSNorm over channel dimension for [B, C, L] tensors."""

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.norm = RMSNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


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


class TCNOp(nn.Module):
    """Temporal Convolutional Network with depthwise-separable and dilations"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        kernel_size: int = 3,
        causal: bool = True,
        dilations: Optional[List[int]] = None,
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


class FourierOp(nn.Module):
    """Fourier operation with learnable frequency weighting"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        num_frequencies: Optional[int] = None,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_frequencies = (
            min(seq_length // 2 + 1, 32) if num_frequencies is None else num_frequencies
        )

        self.low_freq_proj = nn.Sequential(
            nn.Linear(input_dim * 2, latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )
        self.high_freq_proj = nn.Sequential(
            nn.Linear(input_dim * 2, latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )

        self.low_cutoff_logit = nn.Parameter(torch.tensor(0.0))
        self.high_cutoff_logit = nn.Parameter(torch.tensor(0.0))
        self.filter_sharpness = nn.Parameter(torch.tensor(6.0))
        self.post_mix_logit = nn.Parameter(torch.tensor(0.0))

        self.freq_weights = nn.Parameter(torch.randn(self.num_frequencies) * 0.02)

        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False), nn.Sigmoid()
        )

        self.output_proj = nn.Linear(input_dim + latent_dim, latent_dim, bias=False)
        self.norm = RMSNorm(latent_dim)

    def _build_low_high_masks(
        self, n_freq: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.linspace(0.0, 1.0, steps=n_freq, device=device, dtype=dtype)
        sharpness = torch.clamp(self.filter_sharpness, min=1.0).to(dtype=dtype)

        low_cut = torch.sigmoid(self.low_cutoff_logit).to(dtype=dtype)
        high_cut = torch.sigmoid(self.high_cutoff_logit).to(dtype=dtype)

        low_mask = torch.sigmoid((low_cut - pos) * sharpness)
        high_mask = torch.sigmoid((pos - high_cut) * sharpness)
        return low_mask.view(1, -1, 1), high_mask.view(1, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # Build a fixed-length FFT reference window for stable frequency bins.
        if L < self.seq_length:
            x_padded = F.pad(x, (0, 0, 0, self.seq_length - L))
        else:
            x_padded = x[:, : self.seq_length]

        x_fft = torch.fft.rfft(x_padded, dim=1, norm="ortho")
        x_fft = x_fft[:, : self.num_frequencies]
        n_freq = x_fft.size(1)
        low_mask, high_mask = self._build_low_high_masks(
            n_freq, device=x_fft.device, dtype=x_fft.real.dtype
        )
        low_fft = x_fft * low_mask
        high_fft = x_fft * high_mask

        # Frequency tokenization + learnable low/high filtering before projection.
        low_feat = torch.cat([low_fft.real, low_fft.imag], dim=-1)
        high_feat = torch.cat([high_fft.real, high_fft.imag], dim=-1)

        low_feat = self.low_freq_proj(low_feat)
        high_feat = self.high_freq_proj(high_feat)

        post_mix = torch.sigmoid(self.post_mix_logit)
        freq_feat = post_mix * low_feat + (1.0 - post_mix) * high_feat

        # Global spectral summary (weighted) + local spectral context (interpolated).
        weights = F.softmax(self.freq_weights[:n_freq], dim=0).view(1, -1, 1)
        global_feat = (freq_feat * weights).sum(dim=1, keepdim=True).expand(-1, L, -1)

        local_feat = F.interpolate(
            freq_feat.transpose(1, 2),
            size=L,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        spectral_context = 0.5 * global_feat + 0.5 * local_feat
        gated = self.gate(spectral_context)

        combined = torch.cat([x, gated * spectral_context], dim=-1)
        return self.norm(self.output_proj(combined))


class WaveletOp(nn.Module):
    """Multi-scale wavelet-style operation using dilated convolutions"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_scales: int = 3,
        dilations: Optional[List[int]] = None,
    ):
        super().__init__()
        candidate_dilations = dilations or [1, 2, 4, 8]
        self.dilations = candidate_dilations[
            : max(1, min(num_scales, len(candidate_dilations)))
        ]
        self.num_scales = len(self.dilations)

        self.scale_layers = nn.ModuleList()
        for dilation in self.dilations:
            layer = nn.Sequential(
                nn.Conv1d(
                    input_dim,
                    input_dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=input_dim,
                    bias=False,
                ),
                nn.Conv1d(input_dim, input_dim, kernel_size=1, bias=False),
                ChannelRMSNorm(input_dim),
                nn.GELU(),
                nn.Dropout(0.05),
            )
            self.scale_layers.append(layer)

        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))

        self.fusion = nn.Conv1d(
            input_dim * self.num_scales, latent_dim, kernel_size=1, bias=False
        )
        self.norm = RMSNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape
        x_t = x.transpose(1, 2)

        scale_weights = F.softmax(self.scale_logits, dim=0)
        features = []
        for idx, layer in enumerate(self.scale_layers):
            feat = layer(x_t)
            if feat.shape[-1] != L:
                feat = F.adaptive_avg_pool1d(feat, L)
            features.append(feat * scale_weights[idx])

        fused = torch.cat(features, dim=1)
        output = self.fusion(fused).transpose(1, 2)
        return self.norm(output)


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


class MultiScaleConvOp(nn.Module):
    """Multi-scale convolution with attention-based fusion"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        scales: Optional[List[int]] = None,
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


class MambaOp(nn.Module):
    """Lightweight selective-SSM style block with gated recurrent state mixing."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim * 2, bias=False)
        self.causal_mix = CausalConv1d(
            latent_dim,
            latent_dim,
            kernel_size=3,
            groups=latent_dim,
            bias=False,
        )

        self.state_decay_logit = nn.Parameter(torch.zeros(latent_dim))
        self.state_input_scale = nn.Parameter(torch.ones(latent_dim))

        self.output_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.norm = RMSNorm(latent_dim)
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        residual = self.residual_proj(x)

        x_proj = self.input_proj(x)
        x_u, x_g = torch.chunk(x_proj, 2, dim=-1)

        x_u = self.causal_mix(x_u.transpose(1, 2)).transpose(1, 2)

        decay = torch.sigmoid(self.state_decay_logit).view(1, 1, -1)
        in_scale = self.state_input_scale.view(1, 1, -1)

        state = torch.zeros(B, x_u.size(-1), device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            state = decay.squeeze(1) * state + (1.0 - decay.squeeze(1)) * (
                in_scale.squeeze(1) * x_u[:, t, :]
            )
            outputs.append(state)

        y = torch.stack(outputs, dim=1)
        y = y * torch.sigmoid(x_g)
        y = self.dropout(self.output_proj(y))
        return self.norm(y + residual)


class PatchEmbedOp(nn.Module):
    """PatchTST-style patch tokenization with interpolation back to sequence length."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        patch_size: int = 16,
        stride: Optional[int] = None,
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
            patches.permute(0, 2, 1, 3).contiguous().view(B, -1, C * self.patch_size)
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


class DLinearOp(nn.Module):
    """DLinear-style trend/seasonal decomposition followed by linear projection."""

    def __init__(self, input_dim: int, latent_dim: int, trend_kernel: int = 25):
        super().__init__()
        self.trend_kernel = max(3, int(trend_kernel) | 1)  # enforce odd kernel

        self.seasonal_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.trend_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.output_norm = RMSNorm(latent_dim)

        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x_t = x.transpose(1, 2)

        trend = F.avg_pool1d(
            x_t,
            kernel_size=self.trend_kernel,
            stride=1,
            padding=self.trend_kernel // 2,
        ).transpose(1, 2)
        seasonal = x - trend

        y = self.seasonal_proj(seasonal) + self.trend_proj(trend)
        return self.output_norm(y + residual)


class FixedOp(nn.Module):
    """Simple wrapper for fixed operations"""

    def __init__(self, selected_op: nn.Module):
        super().__init__()
        self.op = selected_op

    def forward(self, x):
        return self.op(x)


__all__ = [
    "RMSNorm",
    "ChannelRMSNorm",
    "CausalConv1d",
    "IdentityOp",
    "TimeConvOp",
    "ResidualMLPOp",
    "TCNOp",
    "FourierOp",
    "WaveletOp",
    "ConvMixerOp",
    "GRNOp",
    "MultiScaleConvOp",
    "PyramidConvOp",
    "MambaOp",
    "PatchEmbedOp",
    "InvertedAttentionOp",
    "MLPMixerOp",
    "DLinearOp",
    "FixedOp",
]
