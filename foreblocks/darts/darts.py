import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DARTSConfig:
    """Configuration class for DARTS model"""

    input_dim: int = 3
    hidden_dim: int = 64
    latent_dim: int = 64
    forecast_horizon: int = 24
    seq_length: int = 48
    num_cells: int = 2
    num_nodes: int = 4
    dropout: float = 0.1
    initial_search: bool = False
    selected_ops: Optional[List[str]] = None
    loss_type: str = "huber"
    use_gradient_checkpointing: bool = False
    temperature: float = 1.0
    use_mixed_precision: bool = True
    use_compile: bool = False
    memory_efficient: bool = True

    # New optimization parameters
    arch_lr: float = 3e-4
    weight_lr: float = 1e-3
    alpha_l2_reg: float = 1e-3
    edge_normalization: bool = True
    progressive_shrinking: bool = True


class RMSNorm(nn.Module):
    """Simple RMS normalization"""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.scale * x / (norm + self.eps)


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

    def __init__(self, input_dim: int, latent_dim: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size,
            padding=kernel_size - 1,
            groups=input_dim,
            bias=False,
        )
        # Pointwise convolution
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

        # Depthwise-separable convolution
        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = self.pointwise(x_conv)

        # Causal truncation
        if x_conv.size(2) > L:
            x_conv = x_conv[:, :, :L]

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

    def __init__(self, input_dim: int, latent_dim: int, kernel_size: int = 3):
        super().__init__()
        # First dilated depthwise-separable block
        self.depthwise1 = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size,
            padding=kernel_size - 1,
            dilation=1,
            groups=input_dim,
            bias=False,
        )
        self.pointwise1 = nn.Conv1d(input_dim, latent_dim, 1, bias=False)

        # Second dilated depthwise-separable block
        self.depthwise2 = nn.Conv1d(
            latent_dim,
            latent_dim,
            kernel_size,
            padding=(kernel_size - 1) * 2,
            dilation=2,
            groups=latent_dim,
            bias=False,
        )
        self.pointwise2 = nn.Conv1d(latent_dim, latent_dim, 1, bias=False)

        self.norm1 = RMSNorm(latent_dim)
        self.norm2 = RMSNorm(latent_dim)
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

        # First block
        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise1(x_conv)
        x_conv = self.pointwise1(x_conv)
        if x_conv.size(2) > L:
            x_conv = x_conv[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.activation(self.norm1(x_conv))
        x_conv = self.dropout(x_conv)

        # Second block
        x_conv2 = x_conv.transpose(1, 2)
        x_conv2 = self.depthwise2(x_conv2)
        x_conv2 = self.pointwise2(x_conv2)
        if x_conv2.size(2) > L:
            x_conv2 = x_conv2[:, :, :L]
        x_conv2 = x_conv2.transpose(1, 2)
        x_conv2 = self.activation(self.norm2(x_conv2))

        return x_conv2 + residual


class FourierOp(nn.Module):
    """Fourier operation with learnable frequency weighting"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        num_frequencies: int = None,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_frequencies = (
            min(seq_length // 2 + 1, 32) if num_frequencies is None else num_frequencies
        )

        # Frequency processing
        self.freq_proj = nn.Sequential(
            nn.Linear(input_dim * 2, latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )

        # Learnable frequency weights
        self.freq_weights = nn.Parameter(torch.randn(self.num_frequencies) * 0.02)

        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False), nn.Sigmoid()
        )

        self.output_proj = nn.Linear(input_dim + latent_dim, latent_dim, bias=False)
        self.norm = RMSNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # Pad or truncate to target sequence length
        if L < self.seq_length:
            x_padded = F.pad(x, (0, 0, 0, self.seq_length - L))
        else:
            x_padded = x[:, : self.seq_length]

        # FFT processing
        x_fft = torch.fft.rfft(x_padded, dim=1, norm="ortho")
        x_fft = x_fft[:, : self.num_frequencies]

        # Apply learnable frequency weights
        weights = F.softmax(self.freq_weights, dim=0).view(1, -1, 1)
        real = x_fft.real * weights
        imag = x_fft.imag * weights

        # Process frequency features
        freq_feat = torch.cat([real, imag], dim=-1)
        freq_feat = self.freq_proj(freq_feat)

        # Global feature extraction and gating
        global_feat = freq_feat.mean(dim=1, keepdim=True).expand(-1, L, -1)
        gated = self.gate(global_feat)

        # Combine with input
        combined = torch.cat([x[:, :L], gated * global_feat], dim=-1)
        return self.norm(self.output_proj(combined))


class WaveletOp(nn.Module):
    """Multi-scale wavelet-style operation using dilated convolutions"""

    def __init__(self, input_dim: int, latent_dim: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        scales = [1, 2, 4][:num_scales]

        # Multi-scale depthwise-separable convolutions
        self.scale_layers = nn.ModuleList()
        for dilation in scales:
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
                nn.BatchNorm1d(input_dim),
                nn.GELU(),
                nn.Dropout(0.05),
            )
            self.scale_layers.append(layer)

        self.fusion = nn.Conv1d(
            input_dim * num_scales, latent_dim, kernel_size=1, bias=False
        )
        self.norm = RMSNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x_t = x.transpose(1, 2)

        # Multi-scale processing
        features = []
        for layer in self.scale_layers:
            feat = layer(x_t)
            # Ensure consistent length
            if feat.shape[-1] != L:
                feat = F.adaptive_avg_pool1d(feat, L)
            features.append(feat)

        # Fuse features
        fused = torch.cat(features, dim=1)
        output = self.fusion(fused).transpose(1, 2)
        return self.norm(output)


class ConvMixerOp(nn.Module):
    """ConvMixer-style operation with depthwise separable convolutions"""

    def __init__(self, input_dim: int, latent_dim: int, kernel_size: int = 9):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            latent_dim,
            latent_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=latent_dim,
            bias=False,
        )

        # Pointwise convolution
        self.pointwise = nn.Conv1d(latent_dim, latent_dim, kernel_size=1, bias=False)

        self.norm1 = nn.BatchNorm1d(latent_dim)
        self.norm2 = RMSNorm(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        residual = x

        # Depthwise-separable convolution
        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = self.norm1(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = self.pointwise(x_conv) + x_conv  # Inner residual
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

    def __init__(self, input_dim: int, latent_dim: int, scales: list = None):
        super().__init__()
        self.scales = scales or [1, 3, 5, 7]
        self.num_scales = len(self.scales)
        self.latent_dim = latent_dim

        # Multi-scale depthwise-separable convolutions
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
                nn.BatchNorm1d(latent_dim // self.num_scales),
                nn.GELU(),
            )
            self.scale_convs.append(conv)

        # Attention mechanism for scale fusion
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

        # Multi-scale feature extraction
        scale_features = [conv(x_t) for conv in self.scale_convs]
        multi_scale = torch.cat(scale_features, dim=1)

        # Attention-based fusion
        attn_weights = self.attention(multi_scale)

        # Apply attention and combine
        weighted_features = [
            feat * attn_weights[:, i : i + 1, :]
            for i, feat in enumerate(scale_features)
        ]

        combined = torch.stack(weighted_features, dim=0).sum(dim=0)
        # Expand to full dimension
        combined = combined.repeat(1, self.num_scales, 1)[:, : self.latent_dim, :]

        output = self.final_proj(combined).transpose(1, 2)
        return self.norm(output + residual)


class PyramidConvOp(nn.Module):
    """Pyramid convolution with progressive downsampling and upsampling"""

    def __init__(self, input_dim: int, latent_dim: int, levels: int = 3):
        super().__init__()
        self.levels = min(levels, 3)

        # Calculate channel dimensions for pyramid
        base_channels = max(latent_dim // (2**self.levels), 8)

        self.input_proj = nn.Conv1d(
            input_dim, base_channels * (2**self.levels), kernel_size=1, bias=False
        )

        # Encoder (downsampling) with depthwise-separable convolutions
        encoder_channels = [
            base_channels * (2 ** (self.levels - i)) for i in range(self.levels + 1)
        ]
        self.encoder_convs = nn.ModuleList()
        for i in range(self.levels):
            in_ch, out_ch = encoder_channels[i], encoder_channels[i + 1]
            conv = nn.Sequential(
                # Depthwise
                nn.Conv1d(
                    in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False
                ),
                # Pointwise
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(0.05),
            )
            self.encoder_convs.append(conv)

        # Decoder (upsampling)
        decoder_channels = encoder_channels[::-1]
        self.decoder_convs = nn.ModuleList()
        for i in range(self.levels):
            in_ch, out_ch = decoder_channels[i], decoder_channels[i + 1]
            conv = nn.Sequential(
                nn.ConvTranspose1d(
                    in_ch, out_ch, 3, stride=2, padding=1, output_padding=1, bias=False
                ),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            )
            self.decoder_convs.append(conv)

        # Skip connections
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

        # Project input
        x_proj = self.input_proj(x_t)

        # Encoder path
        encoder_features = [x_proj]
        current = x_proj
        for conv in self.encoder_convs:
            current = conv(current)
            encoder_features.append(current)

        # Decoder path with skip connections
        current = encoder_features[-1]
        for i, conv in enumerate(self.decoder_convs):
            current = conv(current)

            # Add skip connection if not last layer
            if i < len(self.decoder_convs) - 1:
                skip_idx = self.levels - 1 - i
                skip = encoder_features[skip_idx]

                # Handle dimension mismatches
                if current.shape[-1] != skip.shape[-1]:
                    target_len = min(current.shape[-1], skip.shape[-1])
                    current = current[:, :, :target_len]
                    skip = skip[:, :, :target_len]

                # Fuse skip connection
                fused = torch.cat([current, skip], dim=1)
                current = self.skip_fusions[i](fused)

        # Final projection and resize
        current = self.final_proj(current)
        if current.shape[-1] != L:
            current = F.interpolate(current, size=L, mode="linear", align_corners=False)

        output = current.transpose(1, 2)
        return self.norm(output + residual)


class FixedOp(nn.Module):
    """Simple wrapper for fixed operations"""

    def __init__(self, selected_op: nn.Module):
        super().__init__()
        self.op = selected_op

    def forward(self, x):
        return self.op(x)


from .darts_base import *


class MixedOp(nn.Module):
    """Enhanced MixedOp using your existing operators with better search strategy"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        available_ops: Optional[List[str]] = None,
        drop_prob: float = 0.1,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        num_nodes: int = 4,
        use_hierarchical: bool = True,
        adaptive_sampling: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.drop_prob = drop_prob
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.use_hierarchical = use_hierarchical
        self.adaptive_sampling = adaptive_sampling

        # Define operation map using your existing operators
        self.op_map = {
            "Identity": lambda: IdentityOp(input_dim, latent_dim),
            "TimeConv": lambda: TimeConvOp(input_dim, latent_dim),
            "ResidualMLP": lambda: ResidualMLPOp(input_dim, latent_dim),
            "Wavelet": lambda: WaveletOp(input_dim, latent_dim),
            "Fourier": lambda: FourierOp(input_dim, latent_dim, seq_length),
            "TCN": lambda: TCNOp(input_dim, latent_dim),
            "ConvMixer": lambda: ConvMixerOp(input_dim, latent_dim),
            "GRN": lambda: GRNOp(input_dim, latent_dim),
            "MultiScaleConv": lambda: MultiScaleConvOp(input_dim, latent_dim),
            "PyramidConv": lambda: PyramidConvOp(input_dim, latent_dim),
        }

        # Group operations by complexity/type for hierarchical search
        self.operation_groups = {
            "basic": ["Identity", "ResidualMLP"],
            "temporal": ["TimeConv", "TCN", "ConvMixer"],
            "frequency": ["Fourier", "Wavelet"],
            "advanced": ["GRN", "MultiScaleConv", "PyramidConv"],
        }

        # Initialize operations
        self.available_ops = self._validate_ops(available_ops)

        if use_hierarchical:
            self._init_hierarchical_search()
        else:
            self._init_flat_search()

        # Operation efficiency scores (for regularization)
        self.op_efficiency = {
            "Identity": 1.0,
            "ResidualMLP": 0.8,
            "TimeConv": 0.7,
            "TCN": 0.5,
            "ConvMixer": 0.6,
            "Fourier": 0.4,
            "Wavelet": 0.4,
            "GRN": 0.6,
            "MultiScaleConv": 0.3,
            "PyramidConv": 0.2,
        }

        # Adaptive sampling weights
        if adaptive_sampling:
            self.performance_tracker = nn.Parameter(
                torch.zeros(len(self.available_ops)), requires_grad=False
            )
            self.usage_counter = nn.Parameter(
                torch.zeros(len(self.available_ops)), requires_grad=False
            )

        # Fallback operation
        self.fallback_idx = (
            self.available_ops.index("Identity")
            if "Identity" in self.available_ops
            else 0
        )

        # Output projection for dimension mismatch
        self.output_proj = nn.Identity()  # Will be replaced if needed

    def _validate_ops(self, ops):
        """Validate and filter available operations"""
        if not ops:
            return ["Identity", "TimeConv", "ResidualMLP", "TCN"]

        # Filter valid operations
        valid_ops = [op for op in ops if op in self.op_map]

        # Ensure minimum operations
        if len(valid_ops) < 2:
            valid_ops = ["Identity", "TimeConv"]

        return valid_ops

    def _init_hierarchical_search(self):
        """Initialize hierarchical search with group and operation level alphas"""
        # Group level parameters
        active_groups = {}
        for group_name, group_ops in self.operation_groups.items():
            group_valid_ops = [op for op in group_ops if op in self.available_ops]
            if group_valid_ops:
                active_groups[group_name] = group_valid_ops

        self.active_groups = active_groups
        self.group_names = list(active_groups.keys())

        # Group selection parameters
        self.group_alphas = nn.Parameter(torch.randn(len(self.group_names)) * 0.1)

        # Operation modules and parameters for each group
        self.group_ops = nn.ModuleDict()
        self.op_alphas = nn.ParameterDict()

        self.op_to_group = {}  # Map operation to group index
        self.group_op_indices = {}  # Map group to operation indices

        all_ops = []
        for group_idx, (group_name, group_ops) in enumerate(active_groups.items()):
            # Create operations for this group
            group_modules = nn.ModuleList([self.op_map[op]() for op in group_ops])
            self.group_ops[group_name] = group_modules

            # Create alpha parameters for this group
            self.op_alphas[group_name] = nn.Parameter(torch.randn(len(group_ops)) * 0.1)

            # Track mappings
            start_idx = len(all_ops)
            for local_idx, op in enumerate(group_ops):
                global_idx = start_idx + local_idx
                self.op_to_group[global_idx] = (group_idx, local_idx)
                all_ops.append(op)

            self.group_op_indices[group_name] = list(
                range(start_idx, start_idx + len(group_ops))
            )

        self.ops = nn.ModuleList([self.op_map[op]() for op in all_ops])
        self.available_ops = all_ops

    def _init_flat_search(self):
        """Initialize flat search space"""
        self.ops = nn.ModuleList([self.op_map[op]() for op in self.available_ops])
        self._alphas = nn.Parameter(
            torch.randn(len(self.ops)) * 0.1
        )  # Use _alphas internally

    def _get_weights(self, top_k: Optional[int] = None):
        """Get operation weights with optional top-k selection"""
        if self.use_hierarchical:
            return self._get_hierarchical_weights(top_k)
        else:
            return self._get_flat_weights(top_k)

    def _get_hierarchical_weights(self, top_k: Optional[int] = None):
        """Get weights for hierarchical search"""
        if self.use_gumbel and self.training:
            group_weights = F.gumbel_softmax(
                self.group_alphas, tau=self.temperature, hard=False
            )
        else:
            group_weights = F.softmax(self.group_alphas / self.temperature, dim=0)

        final_weights = []
        selected_ops = []

        for group_idx, (group_name, group_weight) in enumerate(
            zip(self.group_names, group_weights)
        ):
            if group_weight.item() > 1e-6:
                # Get operation weights within group
                if self.use_gumbel and self.training:
                    op_weights = F.gumbel_softmax(
                        self.op_alphas[group_name], tau=self.temperature, hard=False
                    )
                else:
                    op_weights = F.softmax(
                        self.op_alphas[group_name] / self.temperature, dim=0
                    )

                # Get operation indices for this group
                op_indices = self.group_op_indices[group_name]

                for local_idx, (op_idx, op_weight) in enumerate(
                    zip(op_indices, op_weights)
                ):
                    if op_weight.item() > 1e-6:
                        final_weight = group_weight * op_weight
                        final_weights.append(final_weight)
                        selected_ops.append(op_idx)

        # Apply top-k selection
        if top_k is not None and len(final_weights) > top_k:
            weight_tensor = torch.stack(final_weights)
            _, top_indices = torch.topk(weight_tensor, top_k)
            final_weights = [final_weights[i] for i in top_indices]
            selected_ops = [selected_ops[i] for i in top_indices]

        return list(zip(selected_ops, final_weights))

    def _get_flat_weights(self, top_k: Optional[int] = None):
        """Get weights for flat search"""
        if self.use_gumbel and self.training:
            weights = F.gumbel_softmax(self._alphas, tau=self.temperature, hard=False)
        else:
            weights = F.softmax(self._alphas / self.temperature, dim=0)

        # Apply adaptive sampling if enabled
        if self.adaptive_sampling and self.training:
            # Boost weights of better performing operations
            performance_boost = torch.sigmoid(self.performance_tracker) * 0.1
            weights = weights + performance_boost
            weights = F.softmax(weights, dim=0)

        selected_ops = [(i, w) for i, w in enumerate(weights) if w.item() > 1e-6]

        # Apply top-k selection
        if top_k is not None and len(selected_ops) > top_k:
            selected_ops = sorted(
                selected_ops, key=lambda x: x[1].item(), reverse=True
            )[:top_k]

        return selected_ops

    def _ensure_output_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure output has correct dimensions"""
        if x.shape[-1] != self.latent_dim:
            if isinstance(self.output_proj, nn.Identity):
                self.output_proj = nn.Linear(x.shape[-1], self.latent_dim).to(x.device)
            x = self.output_proj(x)
        return x

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """Enhanced forward with better operation selection"""
        op_weights = self._get_weights(top_k)

        if not op_weights:
            # Fallback to identity or first operation
            fallback_out = self.ops[self.fallback_idx](x)
            return self._ensure_output_dim(fallback_out)

        outputs = []
        total_weight = 0
        efficiency_penalty = 0

        for op_idx, weight in op_weights:
            # DropPath (stochastic depth)
            if self.training and self.drop_prob > 0.0:
                if torch.bernoulli(torch.tensor(1.0 - self.drop_prob)).item() == 0:
                    continue

            try:
                out = self.ops[op_idx](x)
                out = self._ensure_output_dim(out)
                outputs.append(out * weight)
                total_weight += weight.item()

                # Track efficiency penalty
                op_name = self.available_ops[op_idx]
                if op_name in self.op_efficiency:
                    efficiency_penalty += weight.item() * (
                        1 - self.op_efficiency[op_name]
                    )

                # Update performance tracker if adaptive sampling
                if self.adaptive_sampling and self.training:
                    # Simple heuristic: operations that don't cause NaN/inf are "good"
                    if torch.isfinite(out).all():
                        with torch.no_grad():
                            self.performance_tracker[op_idx] += 0.01
                            self.usage_counter[op_idx] += 1
                    else:
                        with torch.no_grad():
                            self.performance_tracker[op_idx] -= 0.02

            except Exception as e:
                if self.training:
                    print(f"[MixedOp] Op '{self.available_ops[op_idx]}' failed: {e}")
                continue

        if outputs:
            result = sum(outputs) / max(total_weight, 1e-6)
            # Store efficiency penalty for regularization
            self._last_efficiency_penalty = efficiency_penalty * 0.01
            return result

        # Final fallback
        fallback_out = self.ops[self.fallback_idx](x)
        return self._ensure_output_dim(fallback_out)

    def get_alphas(self) -> torch.Tensor:
        """Get normalized architecture weights"""
        if self.use_hierarchical:
            # Combine group and operation weights
            group_weights = F.softmax(self.group_alphas, dim=0)
            all_weights = []

            for group_idx, (group_name, group_weight) in enumerate(
                zip(self.group_names, group_weights)
            ):
                op_weights = F.softmax(self.op_alphas[group_name], dim=0)
                final_weights = group_weight * op_weights
                all_weights.extend(final_weights.tolist())

            return torch.tensor(all_weights, device=self.group_alphas.device)
        else:
            return F.softmax(self.alphas.detach(), dim=0)

    @property
    def alphas(self):
        """Compatibility property for accessing alphas"""
        if self.use_hierarchical:
            return self.get_alphas()
        else:
            return self._alphas

    @alphas.setter
    def alphas(self, value):
        """Compatibility setter for alphas"""
        if self.use_hierarchical:
            # For hierarchical, we need to distribute the values
            # This is a simplified approach - in practice you might want more sophisticated logic
            if hasattr(self, "group_alphas"):
                with torch.no_grad():
                    # Update group alphas with first few values
                    num_groups = len(self.group_alphas)
                    if len(value) >= num_groups:
                        self.group_alphas.data = value[:num_groups]
        else:
            self._alphas = value

    def get_entropy_loss(self) -> torch.Tensor:
        """Get entropy loss for exploration"""
        total_entropy = 0

        if self.use_hierarchical:
            # Group entropy
            group_probs = F.softmax(self.group_alphas / self.temperature, dim=0)
            group_entropy = -(group_probs * torch.log(group_probs + 1e-8)).sum()
            total_entropy += group_entropy

            # Operation entropy within groups
            for group_name in self.group_names:
                op_probs = F.softmax(
                    self.op_alphas[group_name] / self.temperature, dim=0
                )
                op_entropy = -(op_probs * torch.log(op_probs + 1e-8)).sum()
                total_entropy += op_entropy
        else:
            probs = F.softmax(self._alphas / self.temperature, dim=0)
            total_entropy = -(probs * torch.log(probs + 1e-8)).sum()

        return -0.01 * total_entropy  # Encourage exploration

    def get_efficiency_penalty(self) -> torch.Tensor:
        """Get efficiency penalty for regularization"""
        return getattr(self, "_last_efficiency_penalty", torch.tensor(0.0))

    def set_temperature(self, temp: float):
        """Set temperature for Gumbel softmax"""
        self.temperature = max(temp, 0.1)  # Prevent too low temperature

    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about operation usage and performance"""
        stats = {}

        if self.adaptive_sampling:
            for i, op_name in enumerate(self.available_ops):
                usage = self.usage_counter[i].item()
                performance = self.performance_tracker[i].item()
                stats[op_name] = {
                    "usage_count": usage,
                    "avg_performance": performance / max(usage, 1),
                    "efficiency": self.op_efficiency.get(op_name, 0.5),
                }

        return stats

    def describe(self, top_k: int = 3) -> Dict[str, float]:
        """Return top-k operations and their weights for inspection"""
        alphas = self.get_alphas()
        topk_vals, topk_idx = torch.topk(alphas, min(top_k, len(alphas)))
        return {
            self.available_ops[i]: round(w.item(), 4)
            for i, w in zip(topk_idx.tolist(), topk_vals)
        }

    def get_raw_alphas(self):
        """Get raw alpha parameters for debugging/analysis"""
        if self.use_hierarchical:
            return {
                "group_alphas": self.group_alphas.detach(),
                "op_alphas": {
                    name: alpha.detach() for name, alpha in self.op_alphas.items()
                },
            }
        else:
            return {"alphas": self._alphas.detach()}


class DARTSCell(nn.Module):
    """Enhanced DARTS cell with progressive search and better aggregation"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        num_nodes: int = 4,
        initial_search: bool = False,
        selected_ops: Optional[List[str]] = None,
        aggregation: str = "weighted",
        temperature: float = 1.0,
        use_checkpoint: bool = False,
        progressive_stage: str = "basic",  # "basic", "intermediate", "advanced"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_nodes = num_nodes
        self.initial_search = initial_search
        self.aggregation = aggregation
        self.temperature = temperature
        self.use_checkpoint = use_checkpoint
        self.progressive_stage = progressive_stage

        # Progressive operation selection
        self.stage_operations = {
            "basic": ["Identity", "ResidualMLP", "TimeConv"],
            "intermediate": ["Identity", "TimeConv", "TCN", "ConvMixer", "GRN"],
            "advanced": [
                "Identity",
                "TimeConv",
                "TCN",
                "ConvMixer",
                "GRN",
                "Fourier",
                "Wavelet",
                "MultiScaleConv",
                "PyramidConv",
            ],
        }

        self.available_ops = self._select_operations(selected_ops)
        self.num_edges = sum(range(num_nodes))

        self._init_components()
        self._edge_indices = self._precompute_edge_indices()

    def _select_operations(self, selected_ops):
        """Select operations based on search stage"""
        if self.initial_search:
            return ["Identity", "TimeConv"]

        if selected_ops:
            return selected_ops

        return self.stage_operations.get(
            self.progressive_stage, self.stage_operations["basic"]
        )

    def _precompute_edge_indices(self):
        """Precompute edge indices for faster lookup"""
        return {
            (i, j): sum(range(i)) + j
            for i in range(1, self.num_nodes)
            for j in range(i)
        }

    def _init_components(self):
        """Initialize all components"""
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim, bias=False),
            RMSNorm(self.latent_dim),
            nn.GELU(),
        )

        # Enhanced mixed operations
        self.edges = nn.ModuleList(
            [
                MixedOp(
                    self.latent_dim,
                    self.latent_dim,
                    self.seq_length,
                    available_ops=self.available_ops,
                    temperature=self.temperature,
                    use_hierarchical=True,
                    adaptive_sampling=True,
                )
                for _ in range(self.num_edges)
            ]
        )

        # Learnable residual weights per node
        self.residual_weights = nn.Parameter(torch.full((self.num_nodes,), 0.2))

        # Edge importance weights
        self.edge_importance = nn.Parameter(torch.ones(self.num_edges) * 0.5)

        # Aggregation weights if using weighted aggregation
        if self.aggregation == "weighted":
            self.agg_weights = nn.Parameter(torch.ones(self.num_edges) * 0.1)
        else:
            self.agg_weights = None

        # Output normalization
        self.out_norm = RMSNorm(self.latent_dim)

        # Progressive search parameters
        self.stage_gates = nn.Parameter(
            torch.ones(3)
        )  # [basic, intermediate, advanced]

    def _get_edge_index(self, node_idx, input_idx):
        """Get edge index efficiently"""
        return self._edge_indices[(node_idx, input_idx)]

    def _aggregate_inputs(self, inputs, edge_indices):
        """Aggregate inputs with different strategies"""
        if len(inputs) == 1:
            return inputs[0]

        stacked = torch.stack(inputs, dim=0)

        if self.aggregation == "weighted" and self.agg_weights is not None:
            # Use edge-specific weights
            weights = F.softmax(
                torch.stack([self.agg_weights[i] for i in edge_indices]), dim=0
            )
            weights = weights.view(-1, 1, 1, 1)
            return (weights * stacked).sum(dim=0)
        elif self.aggregation == "attention":
            # Simple attention mechanism
            attention_scores = torch.mean(stacked, dim=[2, 3])  # [num_inputs, batch]
            attention_weights = F.softmax(attention_scores, dim=0)
            attention_weights = attention_weights.view(-1, 1, 1, 1)
            return (attention_weights * stacked).sum(dim=0)
        elif self.aggregation == "max":
            return torch.max(stacked, dim=0)[0]
        else:  # mean
            return torch.mean(stacked, dim=0)

    def _apply_residual(self, node_output, residual_input, node_idx):
        """Apply learnable residual connection with proper dimension handling"""
        residual_weight = torch.sigmoid(self.residual_weights[node_idx])

        # Handle dimension mismatches
        if node_output.shape != residual_input.shape:
            # Temporal alignment
            if node_output.shape[1] != residual_input.shape[1]:
                residual_input = F.interpolate(
                    residual_input.transpose(1, 2),
                    size=node_output.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            # Feature dimension alignment
            if node_output.shape[2] != residual_input.shape[2]:
                if not hasattr(self, f"_residual_proj_{node_idx}"):
                    proj = nn.Linear(residual_input.shape[2], node_output.shape[2]).to(
                        residual_input.device
                    )
                    setattr(self, f"_residual_proj_{node_idx}", proj)
                residual_input = getattr(self, f"_residual_proj_{node_idx}")(
                    residual_input
                )

        return residual_weight * node_output + (1 - residual_weight) * residual_input

    def forward(self, x):
        """Enhanced forward pass with progressive search"""
        if not isinstance(x, torch.Tensor):
            x = torch.stack(x) if isinstance(x, (list, tuple)) else x

        x_proj = self.input_proj(x)
        nodes = [x_proj]

        total_efficiency_penalty = 0

        for node_idx in range(1, self.num_nodes):
            node_inputs, edge_indices = [], []

            for input_idx in range(node_idx):
                edge_idx = self._get_edge_index(node_idx, input_idx)
                edge = self.edges[edge_idx]

                # Apply edge importance gating
                edge_weight = torch.sigmoid(self.edge_importance[edge_idx])

                if edge_weight.item() > 0.1:  # Skip unimportant edges
                    # Use gradient checkpointing if enabled
                    if self.training and self.use_checkpoint:
                        out = torch.utils.checkpoint.checkpoint(
                            edge, nodes[input_idx], use_reentrant=False
                        )
                    else:
                        out = edge(nodes[input_idx])

                    # Apply edge weight
                    out = out * edge_weight
                    node_inputs.append(out)
                    edge_indices.append(edge_idx)

                    # Accumulate efficiency penalty
                    # total_efficiency_penalty += edge.get_efficiency_penalty()

            if node_inputs:
                # Aggregate inputs
                agg = self._aggregate_inputs(node_inputs, edge_indices)
                # Apply residual connection
                out = self._apply_residual(agg, nodes[node_idx - 1], node_idx)
            else:
                # Fallback to previous node
                out = nodes[node_idx - 1]

            nodes.append(out)

        # Apply final residual and normalization
        final = self._apply_residual(nodes[-1], x_proj, 0)
        result = self.out_norm(final)

        # Store efficiency penalty for regularization
        self._last_efficiency_penalty = total_efficiency_penalty

        return result

    def advance_progressive_stage(self):
        """Advance to next progressive search stage"""
        stages = ["basic", "intermediate", "advanced"]
        current_idx = stages.index(self.progressive_stage)
        if current_idx < len(stages) - 1:
            self.progressive_stage = stages[current_idx + 1]
            new_ops = self.stage_operations[self.progressive_stage]

            # Update all edges with new operations
            for edge in self.edges:
                edge.available_ops = new_ops
                if hasattr(edge, "_init_hierarchical_search"):
                    edge._init_hierarchical_search()

    def get_alphas(self):
        """Get all edge alphas"""
        return [edge.get_alphas() for edge in self.edges]

    def get_entropy_loss(self):
        """Get total entropy loss for exploration"""
        total = sum(edge.get_entropy_loss() for edge in self.edges)

        # Add aggregation entropy if using weighted aggregation
        if self.agg_weights is not None:
            probs = F.softmax(self.agg_weights, dim=0)
            agg_entropy = -(probs * torch.log(probs + 1e-8)).sum()
            total -= 0.005 * agg_entropy

        return total

    def get_efficiency_penalty(self) -> torch.Tensor:
        """Get total efficiency penalty"""
        return getattr(self, "_last_efficiency_penalty", torch.tensor(0.0))

    def set_temperature(self, temp: float):
        """Update temperature for all edges"""
        self.temperature = temp
        for edge in self.edges:
            edge.set_temperature(temp)

    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about edge usage"""
        stats = {}
        for i, edge in enumerate(self.edges):
            edge_stats = edge.get_operation_statistics()
            edge_weight = torch.sigmoid(self.edge_importance[i]).item()
            stats[f"edge_{i}"] = {
                "importance_weight": edge_weight,
                "operations": edge_stats,
                "top_ops": edge.describe(top_k=2),
            }
        return stats


class TimeSeriesDARTS(nn.Module):
    """Simplified TimeSeriesDARTS with essential features"""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        forecast_horizon: int = 24,
        seq_length: int = 48,
        num_cells: int = 2,
        num_nodes: int = 4,
        dropout: float = 0.1,
        initial_search: bool = False,
        selected_ops: Optional[List] = None,
        loss_type: str = "huber",
        use_gradient_checkpointing: bool = False,
        temperature: float = 1.0,
        use_attention_bridge: bool = True,
        attention_layers: int = 2,
    ):
        super().__init__()

        self._config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "forecast_horizon": forecast_horizon,
            "seq_length": seq_length,
            "num_cells": num_cells,
            "num_nodes": num_nodes,
            "selected_ops": selected_ops,
        }

        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.forecast_horizon = forecast_horizon
        self.seq_length = seq_length
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.initial_search = initial_search
        self.selected_ops = selected_ops
        self.loss_type = loss_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.temperature = temperature
        self.use_attention_bridge = use_attention_bridge
        self.attention_layers = attention_layers

        # Pruning state tracking
        self.pruning_history = []
        self.operation_performance = {}
        self.pruned_operations = set()
        self._init_components()

    def get_config(self):
        return copy.deepcopy(self._config)

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def _init_components(self):
        """Initialize all model components"""
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # DARTS cells with projections and scaling
        self.cells = nn.ModuleList()
        self.cell_proj = nn.ModuleList()
        self.layer_scales = nn.ParameterList()

        for i in range(self.num_cells):
            # Temperature decay for deeper cells
            temp = self.temperature * (0.8**i)

            # Cell
            self.cells.append(
                DARTSCell(
                    input_dim=self.hidden_dim,
                    latent_dim=self.latent_dim,
                    seq_length=self.seq_length,
                    num_nodes=self.num_nodes,
                    initial_search=self.initial_search,
                    selected_ops=self.selected_ops,
                    aggregation="weighted",
                    temperature=temp,
                    use_checkpoint=self.use_gradient_checkpointing,
                )
            )

            # Projection layer
            self.cell_proj.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim, bias=False),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout * 0.5),
                )
            )

            # Layer scaling
            self.layer_scales.append(nn.Parameter(torch.ones(1) * 0.1))

        # Cell combination weights
        self.cell_weights = nn.Parameter(torch.ones(self.num_cells) * 0.5)
        self.cell_importance = nn.Parameter(torch.ones(self.num_cells) * 0.8)
        self.global_skip = nn.Parameter(torch.tensor(0.1))

        self.feature_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Sigmoid()
        )

        self.feature_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        # Encoder and decoder
        self.forecast_encoder = MixedEncoder(
            self.hidden_dim,
            self.latent_dim,
            seq_len=self.seq_length,
            dropout=self.dropout,
            temperature=self.temperature,
        )

        self.forecast_decoder = MixedDecoder(
            self.input_dim,
            self.latent_dim,
            seq_len=self.seq_length,
            dropout=self.dropout,
            temperature=self.temperature,
            use_attention_bridge=self.use_attention_bridge,
            attention_layers=self.attention_layers,
        )

        # Feature fusion
        self.gate_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
        )

        self.output_layer = nn.Linear(self.latent_dim, self.input_dim, bias=False)

        self.residual_weights = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(self.num_cells)]
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is float32 for consistent computation"""
        return tensor.float() if tensor.dtype != torch.float32 else tensor

    def forward(
        self,
        x_seq: torch.Tensor,
        x_future: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass"""
        B, L, _ = x_seq.shape

        # Ensure consistent dtype
        x_seq = self._ensure_dtype(x_seq)

        # Input embedding
        x_emb = self.input_embedding(x_seq)
        original_input = x_emb

        # Process through enhanced DARTS cells
        current_input = x_emb
        cell_outputs = []

        for i, (cell, proj, scale, res_weight, importance) in enumerate(
            zip(
                self.cells,
                self.cell_proj,
                self.layer_scales,
                self.residual_weights,
                self.cell_importance,
            )
        ):
            # Apply cell with optional checkpointing
            if self.training and self.use_gradient_checkpointing:
                cell_out = torch.utils.checkpoint.checkpoint(
                    cell, current_input, use_reentrant=False
                )
            else:
                cell_out = cell(current_input)

            # Project and scale
            projected = proj(cell_out) * scale * torch.sigmoid(importance)
            cell_outputs.append(projected)

            # Enhanced residual connection between cells
            if i > 0:
                residual_contrib = current_input * res_weight
                current_input = cell_out + residual_contrib
            else:
                current_input = cell_out

        # Enhanced cell feature combination
        if len(cell_outputs) > 1:
            # Learnable weighted combination
            cell_weights_norm = F.softmax(self.cell_weights[: len(cell_outputs)], dim=0)
            cell_importance_norm = torch.sigmoid(
                self.cell_importance[: len(cell_outputs)]
            )

            # Combine with both weights and importance
            final_weights = cell_weights_norm * cell_importance_norm
            final_weights = final_weights / final_weights.sum()

            combined_features = sum(w * f for w, f in zip(final_weights, cell_outputs))
        else:
            combined_features = cell_outputs[0]

        # Enhanced feature fusion with gating
        concatenated = torch.cat([combined_features, original_input], dim=-1)
        gate = self.feature_gate(concatenated)
        gated_features = gate * combined_features + (1 - gate) * original_input

        # Apply feature transformation
        final_features = self.feature_transform(gated_features)

        # Global skip connection
        final_features = final_features + self.global_skip * original_input

        # Encoding
        h_enc, context, encoder_state = self.forecast_encoder(final_features)

        # Decoding
        forecasts = []
        decoder_input = x_seq[:, -1:, :]
        decoder_hidden = encoder_state

        # Ensure consistent dtypes
        decoder_input = self._ensure_dtype(decoder_input)
        context = self._ensure_dtype(context)
        h_enc = self._ensure_dtype(h_enc)

        if isinstance(decoder_hidden, tuple):
            decoder_hidden = tuple(self._ensure_dtype(h) for h in decoder_hidden)
        else:
            decoder_hidden = self._ensure_dtype(decoder_hidden)

        for t in range(self.forecast_horizon):
            # Decoder step
            out, decoder_hidden = self.forecast_decoder(
                decoder_input, context, decoder_hidden, h_enc
            )

            # Post-processing
            prediction = self.output_layer(out)
            forecasts.append(prediction.squeeze(1))

            # Teacher forcing
            if (
                self.training
                and x_future is not None
                and t < x_future.size(1)
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                decoder_input = x_future[:, t : t + 1]
                decoder_input = self._ensure_dtype(decoder_input)
            else:
                decoder_input = prediction

        return torch.stack(forecasts, dim=1)

    # Analysis methods
    def get_all_alphas(self) -> Dict[str, torch.Tensor]:
        """Extract all architecture parameters"""
        alphas = {}

        # Cell alphas
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas"):
                        alphas[f"cell_{i}_edge_{j}"] = edge.get_alphas()

        # Encoder/decoder alphas
        alphas["encoder"] = self.forecast_encoder.get_alphas()
        alphas["decoder"] = self.forecast_decoder.get_alphas()

        # Attention alphas
        if (
            self.use_attention_bridge
            and hasattr(self.forecast_decoder, "attention_alphas")
            and self.forecast_decoder.attention_alphas is not None
        ):
            alphas["attention_bridge"] = F.softmax(
                self.forecast_decoder.attention_alphas, dim=0
            )

        return alphas

    def derive_discrete_architecture(self, threshold: float = 0.3) -> Dict[str, Any]:
        """Derive discrete architecture from continuous weights"""
        discrete_arch = {}
        weights = self.get_operation_weights()

        for component_name, component_weights in weights.items():
            if not component_weights:  # Skip empty weight dictionaries
                continue

            max_op = max(component_weights, key=component_weights.get)
            max_weight = component_weights[max_op]

            if component_name.startswith("cell_"):
                parts = component_name.split("_")
                if len(parts) >= 2:
                    cell_name = f"cell_{parts[1]}"
                    if cell_name not in discrete_arch:
                        discrete_arch[cell_name] = {}
                    edge_name = "_".join(parts[2:]) if len(parts) > 2 else "edge"
                    discrete_arch[cell_name][edge_name] = {
                        "operation": max_op,
                        "weight": max_weight,
                    }
            else:
                discrete_arch[component_name] = {"type": max_op, "weight": max_weight}

        return discrete_arch

    def get_operation_weights(self) -> Dict[str, Dict[str, float]]:
        """Get normalized operation weights"""
        weights = {}

        # Cell weights
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas") and hasattr(edge, "available_ops"):
                        try:
                            alphas = edge.get_alphas()
                            if alphas.numel() > 0:
                                weights[f"cell_{i}_edge_{j}"] = {
                                    op: weight.item()
                                    for op, weight in zip(
                                        edge.available_ops, F.softmax(alphas, dim=0)
                                    )
                                }
                        except Exception:
                            continue

        # Encoder/Decoder weights
        for component_name, component in [
            ("encoder", self.forecast_encoder),
            ("decoder", self.forecast_decoder),
        ]:
            if hasattr(component, "get_alphas"):
                try:
                    alphas = component.get_alphas()
                    if alphas.numel() > 0:
                        names = getattr(
                            component,
                            f"{component_name}_names",
                            [f"op_{i}" for i in range(len(alphas))],
                        )
                        soft_alphas = F.softmax(alphas, dim=0)
                        weights[component_name] = {
                            name: weight.item()
                            for name, weight in zip(names, soft_alphas[: len(names)])
                        }
                except Exception:
                    continue

        return weights

    def set_temperature(self, temp: float):
        """Update temperature for all components"""
        self.temperature = temp
        for cell in self.cells:
            if hasattr(cell, "set_temperature"):
                cell.set_temperature(temp)
        self.forecast_encoder.set_temperature(temp)
        self.forecast_decoder.set_temperature(temp)

    # PRUNING METHODS
    def prune_weak_operations(
        self, threshold: float = 0.1, strategy: str = "probability"
    ) -> Dict[str, Any]:
        """
        Prune weak operations based on their weights/importance

        Args:
            threshold: Minimum weight to keep an operation
            strategy: "probability" | "gradient" | "entropy" | "performance"

        Returns:
            Dict with pruning statistics
        """
        pruning_stats = {
            "operations_pruned": 0,
            "operations_kept": 0,
            "pruned_details": {},
            "threshold_used": threshold,
            "strategy": strategy,
        }

        if strategy == "probability":
            pruning_stats.update(self._prune_by_probability(threshold))
        elif strategy == "gradient":
            pruning_stats.update(self._prune_by_gradient_magnitude(threshold))
        elif strategy == "entropy":
            pruning_stats.update(self._prune_by_entropy(threshold))
        elif strategy == "performance":
            pruning_stats.update(self._prune_by_performance(threshold))
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

        # Store pruning history
        self.pruning_history.append(pruning_stats)
        print(
            f"Pruning completed with strategy '{strategy}' and threshold {threshold}."
        )
        return pruning_stats

    def _prune_by_probability(self, threshold: float) -> Dict[str, Any]:
        """Prune operations based on their probability weights"""
        stats = {"operations_pruned": 0, "operations_kept": 0, "pruned_details": {}}

        # Prune cell operations
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas") and hasattr(edge, "available_ops"):
                        alphas = edge.get_alphas()
                        probs = F.softmax(alphas, dim=0)

                        # Find operations below threshold
                        weak_ops = []
                        for k, (op_name, prob) in enumerate(
                            zip(edge.available_ops, probs)
                        ):
                            if prob.item() < threshold and op_name != "Identity":
                                weak_ops.append((k, op_name, prob.item()))

                        # Prune weak operations by setting very low alpha values
                        if weak_ops:
                            with torch.no_grad():
                                for k, op_name, prob in weak_ops:
                                    if hasattr(edge, "_alphas"):
                                        edge._alphas[k] = -10.0  # Very low value
                                    elif hasattr(edge, "alphas"):
                                        edge.alphas[k] = -10.0

                                    # Track pruned operation
                                    op_id = f"cell_{i}_edge_{j}_{op_name}"
                                    self.pruned_operations.add(op_id)
                                    stats["pruned_details"][op_id] = prob
                                    stats["operations_pruned"] += 1

                        # Count kept operations
                        kept_ops = len(edge.available_ops) - len(weak_ops)
                        stats["operations_kept"] += kept_ops

        return stats
