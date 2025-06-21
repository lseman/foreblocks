import functools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class FixedOp(nn.Module):
    """Optimized FixedOp with better efficiency"""

    def __init__(self, selected_op: nn.Module):
        super().__init__()
        self.op = selected_op

    def forward(self, x):
        return self.op(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.scale * x / (norm + self.eps)


class MemoryEfficientOp(nn.Module):
    """Base class for memory-efficient operations with lazy initialization"""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._initialized = False

    def _lazy_init(self, x: torch.Tensor):
        """Override in subclasses for lazy initialization"""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._lazy_init(x)
            self._initialized = True
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override in subclasses for actual forward logic"""
        raise NotImplementedError


class IdentityOp(nn.Module):
    """Optimized identity operation with optional dimension transformation"""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.transform = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class TimeConvOp(MemoryEfficientOp):
    """Causal temporal convolution with depthwise separable structure"""

    def __init__(self, input_dim: int, latent_dim: int, kernel_size: int = 3):
        super().__init__(input_dim, latent_dim)
        self.kernel_size = kernel_size

    def _lazy_init(self, x: torch.Tensor):
        device = x.device
        self.depthwise = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            self.kernel_size,
            padding=self.kernel_size - 1,
            groups=self.input_dim,
            bias=False,
        ).to(device)

        self.pointwise = nn.Conv1d(
            self.input_dim, self.latent_dim, kernel_size=1, bias=False
        ).to(device)

        self.norm = RMSNorm(self.latent_dim).to(device)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = self.pointwise(x_conv)

        # Causal truncation
        if x_conv.size(2) > residual.size(1):
            x_conv = x_conv[:, :, : residual.size(1)]

        x_conv = x_conv.transpose(1, 2)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        return self.norm(x_conv + residual)


class TCNOp(MemoryEfficientOp):
    """Temporal Convolutional Network with dilated convolutions"""

    def __init__(
        self, input_dim: int, latent_dim: int, kernel_size: int = 3, dilation: int = 1
    ):
        super().__init__(input_dim, latent_dim)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1)

    def _lazy_init(self, x: torch.Tensor):
        device = x.device
        self.depthwise = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.input_dim,
            bias=False,
        ).to(device)

        self.pointwise = nn.Conv1d(
            self.input_dim, self.latent_dim, kernel_size=1, bias=False
        ).to(device)

        self.norm = RMSNorm(self.latent_dim).to(device)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Conv1d(self.input_dim, self.latent_dim, kernel_size=1, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x_t = x.transpose(1, 2)
        residual = self.residual_proj(x_t).transpose(1, 2)

        x_conv = self.depthwise(x_t)
        x_conv = self.pointwise(x_conv)

        # Causal truncation
        if x_conv.size(2) > L:
            x_conv = x_conv[:, :, :L]

        x_conv = x_conv.transpose(1, 2)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        return self.norm(x_conv + residual)


class ResidualMLPOp(MemoryEfficientOp):
    """MLP with residual connection and RMSNorm"""

    def __init__(self, input_dim: int, latent_dim: int, expansion_factor: float = 2.67):
        super().__init__(input_dim, latent_dim)
        self.expansion_factor = expansion_factor

    def _lazy_init(self, x: torch.Tensor):
        device = x.device
        hidden_dim = int(self.latent_dim * self.expansion_factor)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim, bias=False),
            nn.Dropout(0.05),
        ).to(device)

        self.norm = RMSNorm(self.latent_dim).to(device)

        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.mlp(x)
        return self.norm(out + residual)


class FourierOp(MemoryEfficientOp):
    """Efficient real-valued FFT-based operator with learnable frequency weighting"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        num_frequencies: int = None,
    ):
        super().__init__(input_dim, latent_dim)
        self.seq_length = seq_length
        self.num_frequencies = (
            min(seq_length // 2 + 1, 32)
            if num_frequencies is None
            else min(num_frequencies, seq_length // 2 + 1)
        )

    def _lazy_init(self, x: torch.Tensor):
        device = x.device

        self.freq_proj = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=False),
        ).to(device)

        self.freq_weights = nn.Parameter(
            torch.randn(self.num_frequencies, device=device) * 0.02
        )

        self.gate = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=False), nn.Sigmoid()
        ).to(device)

        self.output_proj = nn.Linear(
            self.input_dim + self.latent_dim, self.latent_dim, bias=False
        ).to(device)

        self.norm = RMSNorm(self.latent_dim).to(device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # Pad or truncate to target sequence length
        x_padded = (
            F.pad(x, (0, 0, 0, self.seq_length - L))
            if L < self.seq_length
            else x[:, : self.seq_length]
        )

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


class WaveletOp(MemoryEfficientOp):
    """Efficient wavelet-style operation using dilated convolutions"""

    def __init__(self, input_dim: int, latent_dim: int, num_scales: int = 3):
        super().__init__(input_dim, latent_dim)
        self.num_scales = num_scales

    def _lazy_init(self, x: torch.Tensor):
        device = x.device

        self.dwconv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        self.input_dim,
                        self.input_dim,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        groups=self.input_dim,
                        bias=False,
                    ),
                    nn.Conv1d(
                        self.input_dim, self.input_dim, kernel_size=1, bias=False
                    ),
                    nn.BatchNorm1d(self.input_dim),
                    nn.GELU(),
                    nn.Dropout(0.05),
                )
                for d in [1, 2, 4][: self.num_scales]
            ]
        ).to(device)

        self.fusion = nn.Conv1d(
            self.input_dim * self.num_scales, self.latent_dim, kernel_size=1, bias=False
        ).to(device)

        self.norm = RMSNorm(self.latent_dim).to(device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x_t = x.transpose(1, 2)

        # Multi-scale processing
        features = [layer(x_t) for layer in self.dwconv_layers]
        features = [
            F.adaptive_avg_pool1d(f, L) if f.shape[-1] != L else f for f in features
        ]

        # Fuse features
        out = torch.cat(features, dim=1)
        out = self.fusion(out).transpose(1, 2)
        return self.norm(out)


class ConvMixerOp(MemoryEfficientOp):
    """ConvMixer-style operator with depthwise separable convolutions"""

    def __init__(self, input_dim: int, latent_dim: int, kernel_size: int = 9):
        super().__init__(input_dim, latent_dim)
        self.kernel_size = kernel_size

    def _lazy_init(self, x: torch.Tensor):
        device = x.device

        self.input_proj = nn.Linear(self.input_dim, self.latent_dim, bias=False).to(
            device
        )

        self.depthwise = nn.Conv1d(
            self.latent_dim,
            self.latent_dim,
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.latent_dim,
            bias=False,
        ).to(device)

        self.pointwise = nn.Conv1d(
            self.latent_dim, self.latent_dim, kernel_size=1, bias=False
        ).to(device)

        self.norm1 = nn.BatchNorm1d(self.latent_dim).to(device)
        self.norm2 = RMSNorm(self.latent_dim).to(device)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        residual = x

        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = self.norm1(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = self.pointwise(x_conv) + x_conv  # Inner residual
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.dropout(x_conv)

        return self.norm2(x_conv + residual)  # Outer residual


class GRNOp(MemoryEfficientOp):
    """Gated Residual Network with simplified linear structure"""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__(input_dim, latent_dim)

    def _lazy_init(self, x: torch.Tensor):
        device = x.device

        self.fc1 = nn.Linear(self.input_dim, self.latent_dim, bias=False).to(device)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim, bias=False).to(device)

        self.gate = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=False), nn.Sigmoid()
        ).to(device)

        self.norm = RMSNorm(self.latent_dim).to(device)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        h = self.activation(self.fc1(x))
        h = self.dropout(h)
        gated = self.gate(h)
        y = gated * self.fc2(h)
        return self.norm(y + residual)


class MultiScaleConvOp(MemoryEfficientOp):
    """Multi-scale convolutional operation with attention-based fusion"""

    def __init__(self, input_dim: int, latent_dim: int, scales: list = None):
        super().__init__(input_dim, latent_dim)
        self.scales = scales or [1, 3, 5, 7]
        self.num_scales = len(self.scales)

    def _lazy_init(self, x: torch.Tensor):
        device = x.device

        # Multi-scale convolutions
        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        self.input_dim,
                        self.input_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=self.input_dim,
                        bias=False,
                    ),
                    nn.Conv1d(
                        self.input_dim,
                        self.latent_dim // self.num_scales,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self.latent_dim // self.num_scales),
                    nn.GELU(),
                )
                for k in self.scales
            ]
        ).to(device)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.latent_dim // 4, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1),
        ).to(device)

        self.final_proj = nn.Conv1d(
            self.latent_dim, self.latent_dim, kernel_size=1, bias=False
        ).to(device)

        self.norm = RMSNorm(self.latent_dim).to(device)

        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
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
        combined = combined.repeat(1, self.num_scales, 1)[:, : self.latent_dim, :]

        output = self.final_proj(combined).transpose(1, 2)
        return self.norm(output + residual)


class PyramidConvOp(MemoryEfficientOp):
    """Pyramid convolution with progressive downsampling and upsampling"""

    def __init__(self, input_dim: int, latent_dim: int, levels: int = 3):
        super().__init__(input_dim, latent_dim)
        self.levels = min(levels, 3)

    def _lazy_init(self, x: torch.Tensor):
        device = x.device

        # Calculate channel dimensions
        base_channels = max(self.latent_dim // (2**self.levels), 8)

        self.input_proj = nn.Conv1d(
            self.input_dim, base_channels * (2**self.levels), kernel_size=1, bias=False
        ).to(device)

        # Encoder (downsampling)
        encoder_channels = [
            base_channels * (2 ** (self.levels - i)) for i in range(self.levels + 1)
        ]
        self.encoder_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        encoder_channels[i],
                        encoder_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(encoder_channels[i + 1]),
                    nn.GELU(),
                    nn.Dropout(0.05),
                )
                for i in range(self.levels)
            ]
        ).to(device)

        # Decoder (upsampling)
        decoder_channels = encoder_channels[::-1]
        self.decoder_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(decoder_channels[i + 1]),
                    nn.GELU(),
                )
                for i in range(self.levels)
            ]
        ).to(device)

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
        ).to(device)

        self.final_proj = nn.Conv1d(
            decoder_channels[-1], self.latent_dim, kernel_size=1, bias=False
        ).to(device)

        self.norm = RMSNorm(self.latent_dim).to(device)

        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
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


class SoftOpFusion(nn.Module):
    """Streamlined SoftOpFusion with better tensor handling"""

    def __init__(self, num_ops: int, feature_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_ops = num_ops
        self.feature_dim = feature_dim

        # Fusion network with residual connection
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * num_ops, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_ops),
        )

        # Lazy projection layers
        self.projections = nn.ModuleDict()

    def _get_projection(self, input_dim: int, device: torch.device) -> nn.Module:
        """Get or create projection layer for input dimension"""
        key = str(input_dim)
        if key not in self.projections:
            self.projections[key] = nn.Linear(input_dim, self.feature_dim).to(device)
        return self.projections[key]

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [B, T, D] format with correct feature dimension"""
        # Handle different input shapes
        if tensor.dim() == 2:  # [B, D] -> [B, 1, D]
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 4:  # [B, C, H, W] -> [B, C*H*W]
            B, C, H, W = tensor.shape
            tensor = tensor.view(B, C, H * W).transpose(1, 2)

        # Project to correct feature dimension if needed
        if tensor.shape[-1] != self.feature_dim:
            proj = self._get_projection(tensor.shape[-1], tensor.device)
            tensor = proj(tensor)

        return tensor

    def _align_all_tensors(self, op_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Align all tensors to consistent [B, T, D] format"""
        # Normalize all tensors first
        aligned_outputs = []
        for i, output in enumerate(op_outputs):
            try:
                aligned = self._normalize_tensor(output)
                aligned_outputs.append(aligned)
            except Exception as e:
                print(f"Warning: Failed to align tensor {i}: {e}")
                # Create fallback tensor
                if aligned_outputs:
                    fallback = torch.zeros_like(aligned_outputs[0])
                else:
                    B = output.shape[0]
                    fallback = torch.zeros(
                        B, 1, self.feature_dim, device=output.device, dtype=output.dtype
                    )
                aligned_outputs.append(fallback)

        if not aligned_outputs:
            raise RuntimeError("No valid tensors after alignment")

        # Ensure consistent dimensions across all tensors
        target_batch = aligned_outputs[0].shape[0]
        target_seq_len = min(t.shape[1] for t in aligned_outputs)

        processed_outputs = []
        for tensor in aligned_outputs:
            # Handle batch size mismatch
            if tensor.shape[0] != target_batch:
                if tensor.shape[0] > target_batch:
                    tensor = tensor[:target_batch]
                else:
                    # Repeat to match batch size
                    repeat_factor = (
                        target_batch + tensor.shape[0] - 1
                    ) // tensor.shape[0]
                    tensor = tensor.repeat(repeat_factor, 1, 1)[:target_batch]

            # Trim to consistent sequence length
            tensor = tensor[:, :target_seq_len, :]
            processed_outputs.append(tensor)

        return processed_outputs

    def _handle_num_ops_mismatch(
        self, op_outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Handle mismatch between expected and actual number of operations"""
        if len(op_outputs) == self.num_ops:
            return op_outputs

        if len(op_outputs) < self.num_ops:
            # Duplicate last output to fill
            while len(op_outputs) < self.num_ops:
                op_outputs.append(op_outputs[-1])
        else:
            # Take first num_ops outputs
            op_outputs = op_outputs[: self.num_ops]

        return op_outputs

    def forward(self, op_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Main forward pass with robust tensor handling"""
        if not op_outputs:
            raise ValueError("op_outputs cannot be empty")

        # Handle number of operations mismatch
        op_outputs = self._handle_num_ops_mismatch(op_outputs)

        # Align all tensors to consistent format
        processed_outputs = self._align_all_tensors(op_outputs)

        # Stack tensors for processing
        try:
            # For gating: concatenate features [B, T, D * num_ops]
            stacked_for_gating = torch.cat(processed_outputs, dim=-1)

            # For weighting: stack operations [B, T, D, num_ops]
            stacked_for_weighting = torch.stack(processed_outputs, dim=-1)

        except Exception as e:
            print(f"Error in tensor stacking: {e}")
            print(f"Tensor shapes: {[t.shape for t in processed_outputs]}")
            raise

        # Compute gating scores and apply weighted combination
        gate_logits = self.fusion(stacked_for_gating)  # [B, T, num_ops]
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Apply weighted combination: [B, T, D, num_ops] * [B, T, 1, num_ops] -> [B, T, D]
        weighted_output = (stacked_for_weighting * gate_scores.unsqueeze(-2)).sum(
            dim=-1
        )

        return weighted_output

    def get_gate_entropy(self, op_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute entropy of gating decisions for regularization"""
        if not op_outputs:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        try:
            with torch.no_grad():
                # Use same alignment process as forward
                op_outputs = self._handle_num_ops_mismatch(op_outputs)
                aligned_outputs = self._align_all_tensors(op_outputs)

                # Compute gate probabilities
                stacked = torch.cat(aligned_outputs, dim=-1)
                gate_logits = self.fusion(stacked)
                gate_probs = F.softmax(gate_logits, dim=-1)

                # Compute entropy: -sum(p * log(p))
                entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1)
                return entropy.mean()

        except Exception as e:
            print(f"Warning: Failed to compute gate entropy: {e}")
            return torch.tensor(0.0, device=next(self.parameters()).device)


class MixedOp(nn.Module):
    """Streamlined MixedOp with better integration and less bloat"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        seq_length,
        available_ops=None,
        drop_prob=0.1,
        normalize_outputs=True,
        temperature=1.0,
        use_gumbel=True,
        fuse_strategy: str = "soft",
    ):
        super().__init__()
        # Store all config
        for key, value in locals().items():
            if key not in ["self", "__class__"]:
                setattr(self, key, value)

        self.fuse_strategy = fuse_strategy.lower()

        # Operation mapping
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

        # Setup operations
        self._setup_operations()

        # Architecture parameters
        self.alphas = nn.Parameter(torch.randn(len(self.ops)) * 0.02)

        # Soft fusion if requested
        if self.fuse_strategy == "soft":
            self.fusion = SoftOpFusion(
                num_ops=len(self.ops), feature_dim=latent_dim, dropout_rate=drop_prob
            )

        # Caching for inference
        self._cached_weights = None

    def _setup_operations(self):
        """Setup available operations with validation"""
        if not self.available_ops:
            self.available_ops = ["Identity", "TimeConv", "ResidualMLP"]

        # Remove duplicates while preserving order
        seen = set()
        self.available_ops = [
            op
            for op in self.available_ops
            if op in self.op_map and not (op in seen or seen.add(op))
        ]

        # Ensure minimum operations
        if len(self.available_ops) < 2:
            self.available_ops = ["Identity", "TimeConv", "ResidualMLP"]

        # Create operation modules
        self.ops = nn.ModuleList([self.op_map[op]() for op in self.available_ops])

    def _get_weights(self):
        """Get operation weights with caching for inference"""
        if not self.training and self._cached_weights is not None:
            return self._cached_weights

        if self.use_gumbel and self.training:
            weights = F.gumbel_softmax(self.alphas, tau=self.temperature, hard=False)
        else:
            weights = F.softmax(self.alphas / self.temperature, dim=0)

        if not self.training:
            self._cached_weights = weights.detach()

        return weights

    def _ensure_correct_dims(
        self, output: torch.Tensor, op_idx: int, device: torch.device
    ) -> torch.Tensor:
        """Ensure output has correct dimensions with lazy projection creation"""
        if output.shape[-1] != self.latent_dim:
            # Create projection on demand
            if not hasattr(self, "output_projections"):
                self.output_projections = nn.ModuleDict()

            key = f"{op_idx}_{output.shape[-1]}"
            if key not in self.output_projections:
                self.output_projections[key] = nn.Linear(
                    output.shape[-1], self.latent_dim
                ).to(device)

            output = self.output_projections[key](output)

        return output

    def _apply_operations(self, x: torch.Tensor, weights: torch.Tensor) -> tuple:
        """Apply all operations and collect valid outputs"""
        device = x.device

        # Early exit for inference with dominant weight
        if not self.training and self.fuse_strategy in {"weighted", "hard"}:
            max_weight = weights.max()
            if max_weight > 0.95:
                dominant_op = self.ops[weights.argmax().item()]
                output = dominant_op(x)
                return [
                    self._ensure_correct_dims(output, weights.argmax().item(), device)
                ], [max_weight]

        # Apply dropout during training
        if self.training and self.drop_prob > 0:
            drop_mask = torch.rand(len(self.ops), device=device) < self.drop_prob
        else:
            drop_mask = torch.zeros(len(self.ops), dtype=torch.bool, device=device)

        op_outputs = []
        active_weights = []

        for i, (op, w) in enumerate(zip(self.ops, weights)):
            if drop_mask[i] or w.item() < 1e-4:  # Skip very small weights
                continue

            try:
                out = op(x)
                out = self._ensure_correct_dims(out, i, device)

                if self.normalize_outputs and self.training:
                    out = F.layer_norm(out, out.shape[-1:])

                op_outputs.append(out)
                active_weights.append(w)

            except Exception as e:
                if self.training:
                    print(f"Operation {self.available_ops[i]} failed: {e}")
                continue

        return op_outputs, active_weights

    def _weighted_fusion(
        self, op_outputs: List[torch.Tensor], active_weights: List[torch.Tensor]
    ) -> torch.Tensor:
        """Weighted fusion of operation outputs"""
        total_weight = sum(w.item() for w in active_weights)

        if total_weight < 1e-8:
            return op_outputs[0]  # Return first output if weights are too small

        result = sum(w * out for w, out in zip(active_weights, op_outputs))

        if self.normalize_outputs and abs(total_weight - 1.0) > 1e-4:
            result = result / total_weight

        return result

    def _fallback_to_identity(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback to Identity operation if all others fail"""
        identity_idx = (
            self.available_ops.index("Identity")
            if "Identity" in self.available_ops
            else 0
        )
        return self.ops[identity_idx](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main forward pass with unified fusion strategies"""
        weights = self._get_weights()

        # Apply operations
        op_outputs, active_weights = self._apply_operations(x, weights)

        # Fallback if no operations succeeded
        if not op_outputs:
            return self._fallback_to_identity(x)

        # Apply fusion strategy
        if self.fuse_strategy == "soft":
            try:
                return self.fusion(op_outputs)
            except Exception as e:
                print(f"SoftOpFusion failed: {e}, falling back to weighted fusion")
                return self._weighted_fusion(op_outputs, active_weights)

        elif self.fuse_strategy == "hard":
            # Hard selection using Gumbel
            one_hot = F.gumbel_softmax(
                self.alphas / self.temperature, tau=1.0, hard=True
            )
            idx = one_hot.argmax().item()
            return self.ops[idx](x)

        else:  # weighted fusion (default)
            return self._weighted_fusion(op_outputs, active_weights)

    # === ARCHITECTURE ANALYSIS ===

    def get_alphas(self) -> torch.Tensor:
        """Get normalized architecture weights"""
        return F.softmax(self.alphas.detach(), dim=0)

    def set_temperature(self, temp: float):
        """Set temperature and clear cached weights"""
        self.temperature = temp
        self._cached_weights = None

    def get_entropy_loss(self) -> torch.Tensor:
        """Regularization loss to encourage exploration"""
        probs = F.softmax(self.alphas / self.temperature, dim=0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        return -0.05 * entropy

    def get_gate_entropy(self) -> torch.Tensor:
        """Get entropy from fusion layer if using soft fusion"""
        if self.fuse_strategy == "soft" and hasattr(self, "fusion"):
            # Create dummy outputs to compute entropy
            dummy_outputs = []
            device = next(self.parameters()).device

            for op in self.ops:
                dummy_input = torch.randn(1, 10, self.latent_dim, device=device)
                with torch.no_grad():
                    dummy_out = op(dummy_input)
                dummy_outputs.append(dummy_out)

            return self.fusion.get_gate_entropy(dummy_outputs)

        return torch.tensor(0.0, device=next(self.parameters()).device)


class DARTSCell(nn.Module):
    """Optimized DARTS cell - streamlined version with all functionality preserved"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        num_nodes: int = 4,
        initial_search: bool = False,
        selected_ops: Optional[List[str]] = None,
        aggregation: str = "mean",
        temperature: float = 1.0,
        use_checkpoint: bool = False,
        enable_caching: bool = True,
    ):
        super().__init__()
        # Store all config
        for key, value in locals().items():
            if key not in ["self", "__class__"]:
                setattr(self, key, value)

        # Core setup
        self.available_ops = self._select_operations()
        self.num_edges = sum(range(num_nodes))

        # Initialize all components
        self._init_components()

        # Caching and buffers
        self._inference_cache = {} if enable_caching else None

    def _select_operations(self) -> List[str]:
        """Smart operation selection with fallbacks"""
        if self.initial_search:
            return ["Identity", "TimeConv"]

        if self.selected_ops:
            valid_ops = [
                "Identity",
                "TimeConv",
                "Fourier",
                "ResidualMLP",
                "TCN",
                "Wavelet",
                "ConvMixer",
                "GRN",
                "MultiScaleConv",
                "PyramidConv",
            ]
            filtered_ops = [op for op in self.selected_ops if op in valid_ops]
            return filtered_ops if len(filtered_ops) >= 2 else ["Identity", "TimeConv"]

        return ["Identity", "TimeConv", "Fourier", "ResidualMLP", "TCN"]

    def _init_components(self):
        """Initialize all model components"""
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim, bias=False),
            RMSNorm(self.latent_dim),
            nn.GELU(),
        )

        # Create edges
        self.edges = nn.ModuleList()
        for _ in range(self.num_edges):
            edge = MixedOp(
                input_dim=self.latent_dim,
                latent_dim=self.latent_dim,
                seq_length=self.seq_length,
                available_ops=self.available_ops,
                temperature=self.temperature,
                fuse_strategy="soft",
                drop_prob=0.1,
                normalize_outputs=True,
            )
            self.edges.append(edge)

        # Learnable weights
        self.residual_weights = nn.Parameter(torch.full((self.num_nodes,), 0.2))

        # Aggregation weights for weighted combination
        if self.aggregation == "weighted" and self.num_nodes > 2:
            self.agg_weights = nn.Parameter(torch.ones(self.num_nodes - 1))
        else:
            self.agg_weights = None

        # Learned temperature
        self.learned_temp = nn.Parameter(torch.tensor(1.0))

    def _get_edge_index(self, node_idx: int, input_idx: int) -> int:
        """Efficiently compute edge index"""
        return sum(range(node_idx)) + input_idx

    def _aggregate_inputs(
        self, inputs: List[torch.Tensor], node_idx: int
    ) -> torch.Tensor:
        """Optimized input aggregation with multiple strategies"""
        if not inputs:
            raise ValueError("No inputs to aggregate")

        # Ensure all inputs are tensors and handle various input types
        tensor_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                tensor_inputs.append(inp)
            elif isinstance(inp, (list, tuple)) and len(inp) > 0:
                if isinstance(inp[0], torch.Tensor):
                    tensor_inputs.append(torch.stack(inp) if len(inp) > 1 else inp[0])
                else:
                    raise TypeError(
                        f"Invalid input type in aggregation: {type(inp[0])}"
                    )
            else:
                raise TypeError(f"Invalid input type in aggregation: {type(inp)}")

        if len(tensor_inputs) == 1:
            return tensor_inputs[0]

        stacked_inputs = torch.stack(tensor_inputs, dim=0)

        # Aggregation strategies
        if self.aggregation == "weighted" and self.agg_weights is not None:
            num_inputs = len(tensor_inputs)
            if num_inputs <= self.agg_weights.size(0):
                weights = F.softmax(self.agg_weights[:num_inputs], dim=0)
                return torch.sum(weights.view(-1, 1, 1, 1) * stacked_inputs, dim=0)

        elif self.aggregation == "attention":
            # Simple attention-based aggregation
            query = stacked_inputs.mean(dim=0, keepdim=True)  # [1, B, T, D]
            scores = torch.sum(
                query * stacked_inputs, dim=-1, keepdim=True
            )  # [num_inputs, B, T, 1]
            weights = F.softmax(scores, dim=0)
            return torch.sum(weights * stacked_inputs, dim=0)

        elif self.aggregation == "max":
            return torch.max(stacked_inputs, dim=0)[0]

        else:  # mean (default)
            return torch.mean(stacked_inputs, dim=0)

    def _apply_residual(
        self, node_output: torch.Tensor, previous_node: torch.Tensor, node_idx: int
    ) -> torch.Tensor:
        """Efficient residual connection with learned gating"""
        residual_weight = torch.sigmoid(self.residual_weights[node_idx])

        # Handle dimension mismatches
        if node_output.shape != previous_node.shape:
            if node_output.shape[1] != previous_node.shape[1]:
                target_len = node_output.shape[1]
                previous_node = F.interpolate(
                    previous_node.transpose(1, 2),
                    size=target_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

        return residual_weight * node_output + (1 - residual_weight) * previous_node

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward computation with optimizations"""
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            if isinstance(x, (list, tuple)):
                x = torch.stack(x) if len(x) > 1 else x[0]
            else:
                raise TypeError(f"Expected tensor input, got {type(x)}")

        # Project input
        x_proj = self.input_proj(x)
        nodes = [x_proj]

        # Process nodes efficiently
        for node_idx in range(1, self.num_nodes):
            # Collect inputs from previous nodes
            node_inputs = []
            for input_idx in range(node_idx):
                edge_idx = self._get_edge_index(node_idx, input_idx)
                edge = self.edges[edge_idx]

                # Apply edge operation with optional checkpointing
                if self.use_checkpoint and self.training:
                    edge_output = torch.utils.checkpoint.checkpoint(
                        edge, nodes[input_idx], use_reentrant=False
                    )
                else:
                    edge_output = edge(nodes[input_idx])

                node_inputs.append(edge_output)

            # Aggregate inputs
            aggregated = self._aggregate_inputs(node_inputs, node_idx)

            # Apply residual connection
            if node_idx > 0:
                node_output = self._apply_residual(
                    aggregated, nodes[node_idx - 1], node_idx
                )
            else:
                node_output = aggregated

            nodes.append(node_output)

        # Final output with input residual
        final_residual_weight = torch.sigmoid(self.residual_weights[0])
        final_output = (
            final_residual_weight * nodes[-1] + (1 - final_residual_weight) * x_proj
        )

        return final_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main forward pass with optional caching"""
        # Caching for inference
        if (
            not self.training
            and self.enable_caching
            and self._inference_cache is not None
        ):
            input_hash = hash((x.shape, x.device, x.dtype))
            if input_hash in self._inference_cache:
                return self._inference_cache[input_hash]

        result = self._forward_core(x)

        # Cache result for inference
        if (
            not self.training
            and self.enable_caching
            and self._inference_cache is not None
        ):
            # Limit cache size to prevent memory bloat
            if len(self._inference_cache) > 100:
                self._inference_cache.clear()
            self._inference_cache[input_hash] = result.detach()

        return result

    # === ARCHITECTURE ANALYSIS ===

    def get_alphas(self) -> List[torch.Tensor]:
        """Get architecture weights from all edges"""
        return [edge.get_alphas() for edge in self.edges]

    def get_entropy_loss(self) -> torch.Tensor:
        """Compute total entropy loss across all edges"""
        total_entropy = sum(edge.get_entropy_loss() for edge in self.edges)

        # Add entropy for aggregation weights if applicable
        if self.agg_weights is not None:
            agg_probs = F.softmax(self.agg_weights, dim=0)
            agg_entropy = -(agg_probs * torch.log(agg_probs + 1e-8)).sum()
            total_entropy += -0.01 * agg_entropy  # Small weight for aggregation entropy

        return total_entropy

    def get_gate_entropy(self) -> torch.Tensor:
        """Get gate entropy from all edges for regularization"""
        total_gate_entropy = sum(edge.get_gate_entropy() for edge in self.edges)
        return total_gate_entropy / len(self.edges)

    def set_temperature(self, temp: float):
        """Set temperature for all edges and update learned temperature"""
        self.temperature = temp
        for edge in self.edges:
            edge.set_temperature(temp)
        # Update learned temperature
        with torch.no_grad():
            self.learned_temp.data.fill_(temp)

    def get_architecture_summary(self) -> Dict:
        """Get summary of current architecture choices"""
        # Get dominant operations for each edge
        edge_ops = []
        for i, edge in enumerate(self.edges):
            alphas = edge.get_alphas()
            dominant_op_idx = alphas.argmax().item()
            dominant_op = self.available_ops[dominant_op_idx]
            edge_ops.append(
                {
                    "edge_idx": i,
                    "dominant_op": dominant_op,
                    "confidence": alphas[dominant_op_idx].item(),
                }
            )

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "available_ops": self.available_ops,
            "aggregation": self.aggregation,
            "edge_operations": edge_ops,
        }

    # === UTILITY METHODS ===

    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        edge_params = sum(
            sum(p.numel() for p in edge.parameters()) for edge in self.edges
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "edge_parameters": edge_params,
            "edge_percentage": (
                (edge_params / total_params * 100) if total_params > 0 else 0
            ),
        }

    def enable_gradient_checkpointing(self, enabled: bool = True):
        """Enable or disable gradient checkpointing for memory efficiency"""
        self.use_checkpoint = enabled

    def clear_cache(self):
        """Clear inference cache"""
        if self._inference_cache is not None:
            self._inference_cache.clear()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        stats = {
            "cache_size": len(self._inference_cache) if self._inference_cache else 0
        }

        if torch.cuda.is_available():
            stats.update(
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "cached_mb": torch.cuda.memory_reserved() / 1024**2,
                }
            )

        return stats


from .darts_base import *


class TimeSeriesDARTS(nn.Module):
    """Enhanced TimeSeriesDARTS with cross-attention bridge - streamlined version"""

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
        memory_efficient: bool = True,
        use_compile: bool = False,
        use_attention_bridge: bool = True,
        attention_layers: int = 2,
    ):
        super().__init__()
        # Store all config as attributes
        for key, value in locals().items():
            if key not in ["self", "__class__"]:
                setattr(self, key, value)

        self._init_all_components()
        self._setup_compilation()

    def _init_all_components(self):
        """Initialize all model components in one place"""
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, self.hidden_dim) * 0.02
        )

        # DARTS cells with projections
        self.cells = nn.ModuleList()
        self.cell_proj = nn.ModuleList()
        self.layer_scales = nn.ParameterList()

        for i in range(self.num_cells):
            temp = self.temperature * (0.8**i)

            self.cells.append(
                DARTSCell(
                    input_dim=self.input_dim if i == 0 else self.latent_dim,
                    latent_dim=self.latent_dim,
                    seq_length=self.seq_length,
                    num_nodes=self.num_nodes,
                    initial_search=self.initial_search,
                    selected_ops=self.selected_ops,
                    aggregation="weighted" if i > 0 else "mean",
                    temperature=temp,
                )
            )

            self.cell_proj.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim, bias=False),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout * 0.5),
                )
            )

            self.layer_scales.append(nn.Parameter(torch.ones(1) * 1e-2))

        self.cell_weights = nn.Parameter(torch.ones(self.num_cells))

        # Encoder/Decoder
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

        # Fusion and output layers
        self.gate_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim, bias=False),
            nn.Dropout(self.dropout * 0.5),
        )

        self.output_layer = nn.Linear(self.latent_dim, self.input_dim, bias=False)
        self.forecast_norm = nn.LayerNorm(self.latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Consolidated weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _setup_compilation(self):
        """Setup torch.compile if requested"""
        self._compiled_forward = None
        if self.use_compile and hasattr(torch, "compile"):
            try:
                self._compiled_forward = torch.compile(
                    self._uncompiled_forward, mode="default"
                )
            except Exception as e:
                print(f"[Compile Warning] torch.compile failed: {e}")

    def forward(
        self,
        x_seq: torch.Tensor,
        x_future: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Main forward pass with optional compilation"""
        if self._compiled_forward:
            return self._compiled_forward(x_seq, x_future, teacher_forcing_ratio)
        return self._uncompiled_forward(x_seq, x_future, teacher_forcing_ratio)

    @torch.jit.ignore
    def _uncompiled_forward(
        self,
        x_seq: torch.Tensor,
        x_future: Optional[torch.Tensor],
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        """Core forward implementation"""
        B, L, _ = x_seq.shape

        # Input embedding
        x_emb = self.input_embedding(x_seq)

        # DARTS cell processing with fusion
        current_input = x_seq
        cell_features = []

        for i, (cell, proj, scale) in enumerate(
            zip(self.cells, self.cell_proj, self.layer_scales)
        ):
            # Cell forward with optional checkpointing
            if self.training and self.use_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint

                cell_out = checkpoint(cell, current_input, use_reentrant=False)
            else:
                cell_out = cell(current_input)

            # Project and scale
            projected = proj(cell_out) * scale
            cell_features.append(projected)
            current_input = cell_out + current_input * 0.1 if i > 0 else cell_out

        # Weighted combination of cell features
        if len(cell_features) > 1:
            weights = F.softmax(self.cell_weights[: len(cell_features)], dim=0)
            final_features = sum(w * f for w, f in zip(weights, cell_features))
        else:
            final_features = cell_features[0]

        # Feature fusion
        fuse_input = torch.cat([final_features, x_emb], dim=-1)
        alpha = self.gate_fuse(fuse_input)
        combined = alpha * final_features + (1 - alpha) * x_emb

        # Encoding
        h_enc, context, encoder_state = self.forecast_encoder(combined)

        # Decoding with attention bridge
        forecasts = []
        decoder_input = x_seq[:, -1:, :]
        decoder_hidden = encoder_state

        for t in range(self.forecast_horizon):
            # Decoder with cross-attention to encoder
            out, decoder_hidden = self.forecast_decoder(
                decoder_input, context, decoder_hidden, h_enc
            )

            # Post-processing
            out = self.forecast_norm(out + context)
            out = self.mlp(out) + out
            prediction = self.output_layer(out)
            forecasts.append(prediction.squeeze(1))

            # Teacher forcing decision
            if (
                self.training
                and x_future is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                decoder_input = x_future[:, t : t + 1]
            else:
                decoder_input = prediction

        return torch.stack(forecasts, dim=1)

    def calculate_loss(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Calculate total loss with all regularization terms"""
        pred = self.forward(x_seq, x_future, teacher_forcing_ratio)

        # Main prediction loss
        loss_functions = {
            "huber": lambda p, t: F.huber_loss(p, t, delta=0.1),
            "mse": F.mse_loss,
            "mae": F.l1_loss,
        }
        main_loss = loss_functions.get(self.loss_type, F.smooth_l1_loss)(pred, x_future)

        # All regularization in one place
        entropy_loss = self._compute_all_entropy_losses()
        alpha_l2 = self._compute_all_alpha_l2()
        attention_reg = self._compute_attention_regularization()

        total_loss = main_loss + entropy_loss + alpha_l2 + attention_reg

        if return_components:
            return {
                "total_loss": total_loss,
                "main_loss": main_loss,
                "entropy_loss": entropy_loss,
                "alpha_l2": alpha_l2,
                "attention_reg": attention_reg,
            }
        return total_loss

    def _compute_all_entropy_losses(self) -> torch.Tensor:
        """Compute all entropy regularization losses"""
        entropy_loss = sum(cell.get_entropy_loss() for cell in self.cells) * 1e-3
        entropy_loss += self.forecast_encoder.get_entropy_loss()
        entropy_loss += self.forecast_decoder.get_entropy_loss()
        return entropy_loss

    def _compute_all_alpha_l2(self) -> torch.Tensor:
        """Compute L2 regularization on all architecture parameters"""
        alpha_l2 = 0.0

        # Cell alphas
        for cell in self.cells:
            for edge in cell.edges:
                alpha_l2 += (edge.alphas**2).sum()

        # Encoder/decoder alphas
        alpha_l2 += (self.forecast_encoder.alphas**2).sum()
        alpha_l2 += (self.forecast_decoder.alphas**2).sum()

        # Attention alphas
        if hasattr(self.forecast_decoder, "attention_alphas"):
            alpha_l2 += (self.forecast_decoder.attention_alphas**2).sum()

        return alpha_l2 * 1e-4

    def _compute_attention_regularization(self) -> torch.Tensor:
        """Compute attention-specific regularization"""
        if not self.use_attention_bridge or not hasattr(
            self.forecast_decoder, "attention_alphas"
        ):
            return torch.tensor(0.0, device=next(self.parameters()).device)

        attn_probs = F.softmax(self.forecast_decoder.attention_alphas, dim=0)

        # Entropy + penalty for no attention
        attn_entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum()
        no_attention_penalty = attn_probs[-1] * 0.1

        return -attn_entropy * 1e-3 + no_attention_penalty

    # === ARCHITECTURE ANALYSIS (CONSOLIDATED) ===

    def get_all_alphas(self) -> Dict[str, torch.Tensor]:
        """Get all architecture parameters in one place"""
        alphas = {}

        # Cell alphas
        for i, cell in enumerate(self.cells):
            for j, edge in enumerate(cell.edges):
                alphas[f"cell_{i}_edge_{j}"] = edge.get_alphas()

        # Encoder/decoder alphas
        alphas["encoder"] = self.forecast_encoder.get_alphas()
        alphas["decoder"] = self.forecast_decoder.get_alphas()

        # Attention alphas
        if self.use_attention_bridge and hasattr(
            self.forecast_decoder, "attention_alphas"
        ):
            alphas["attention_bridge"] = F.softmax(
                self.forecast_decoder.attention_alphas, dim=0
            )

        return alphas

    def get_operation_weights(self) -> Dict[str, Dict[str, float]]:
        """Get current operation weights for analysis"""
        weights = {}

        # Cell weights
        for i, cell in enumerate(self.cells):
            for j, edge in enumerate(cell.edges):
                weights[f"cell_{i}_edge_{j}"] = {
                    op: weight.item()
                    for op, weight in zip(edge.available_ops, edge.get_alphas())
                }

        # Encoder/Decoder weights
        for component_name, component in [
            ("encoder", self.forecast_encoder),
            ("decoder", self.forecast_decoder),
        ]:
            names = getattr(
                component,
                (
                    f"{component_name}_names"
                    if component_name == "encoder"
                    else "decoder_names"
                ),
            )
            alphas = component.get_alphas()
            if component_name == "decoder":
                alphas = alphas[: len(names)]  # Only decoder type alphas
            weights[component_name] = {
                name: weight.item() for name, weight in zip(names, alphas)
            }

        # Attention bridge weights
        if self.use_attention_bridge and hasattr(
            self.forecast_decoder, "attention_alphas"
        ):
            attention_weights = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
            attention_names = [
                f"attention_layer_{i}" for i in range(len(attention_weights) - 1)
            ] + ["no_attention"]
            weights["attention_bridge"] = {
                name: weight.item()
                for name, weight in zip(attention_names, attention_weights)
            }

        return weights

    def derive_discrete_architecture(
        self, threshold: float = 0.3, top_k_fallback: int = 2
    ) -> Dict[str, Any]:
        """Derive discrete architecture from current weights"""
        discrete_arch = {}

        # Process all components uniformly
        weights = self.get_operation_weights()

        for component_name, component_weights in weights.items():
            if component_name.startswith("cell_"):
                # Cell architecture
                if (
                    component_name.replace("_edge_", "_").split("_")[1]
                    not in discrete_arch
                ):
                    cell_name = f"cell_{component_name.split('_')[1]}"
                    discrete_arch[cell_name] = {}

                max_op = max(component_weights, key=component_weights.get)
                max_weight = component_weights[max_op]

                if max_weight > threshold:
                    discrete_arch[cell_name][component_name.split("_")[-1]] = {
                        "operation": max_op,
                        "weight": max_weight,
                        "confidence": max_weight,
                    }
                else:
                    # Fallback to top-k
                    sorted_ops = sorted(
                        component_weights.items(), key=lambda x: x[1], reverse=True
                    )
                    discrete_arch[cell_name][component_name.split("_")[-1]] = {
                        "operation": sorted_ops[0][0],
                        "alternatives": sorted_ops[:top_k_fallback],
                        "weight": sorted_ops[0][1],
                        "confidence": sorted_ops[0][1],
                    }
            else:
                # Encoder/decoder/attention
                max_op = max(component_weights, key=component_weights.get)
                max_weight = component_weights[max_op]
                discrete_arch[component_name] = {"type": max_op, "weight": max_weight}

        return discrete_arch

    def validate_architecture_health(self) -> Dict[str, Any]:
        """Comprehensive architecture health validation"""
        issues = []
        total_identity_dominance = 0
        total_edges = 0

        # Check all alphas uniformly
        all_alphas = self.get_all_alphas()

        for component_name, alphas in all_alphas.items():
            entropy = -(alphas * torch.log(alphas + 1e-8)).sum().item()
            max_weight = alphas.max().item()

            # Common health checks
            if entropy < 0.5:
                issues.append(
                    f"{component_name}: Low diversity (entropy={entropy:.3f})"
                )

            if max_weight > 0.9:
                issues.append(f"{component_name}: Extreme dominance ({max_weight:.3f})")

            # Identity dominance for cells
            if component_name.startswith("cell_"):
                cell_idx = int(component_name.split("_")[1])
                edge_idx = int(component_name.split("_")[3])
                edge = self.cells[cell_idx].edges[edge_idx]

                if "Identity" in edge.available_ops:
                    identity_idx = edge.available_ops.index("Identity")
                    identity_weight = alphas[identity_idx].item()
                    total_identity_dominance += identity_weight

                    if identity_weight > 0.7:
                        issues.append(
                            f"{component_name}: Identity dominates ({identity_weight:.3f})"
                        )

                total_edges += 1

            # Attention-specific checks
            elif component_name == "attention_bridge":
                no_attention_weight = alphas[-1].item()
                if no_attention_weight > 0.8:
                    issues.append(
                        f"Attention bridge: No attention dominates ({no_attention_weight:.3f})"
                    )

        avg_identity_dominance = total_identity_dominance / max(total_edges, 1)
        health_score = max(
            0,
            1.0 - len(issues) / max((total_edges * 2 + 2), 1) - avg_identity_dominance,
        )

        return {
            "issues": issues,
            "avg_identity_dominance": avg_identity_dominance,
            "total_edges": total_edges,
            "health_score": health_score,
            "attention_enabled": self.use_attention_bridge,
        }

    def apply_architecture_fixes(self):
        """Apply automatic fixes for architecture issues"""
        health = self.validate_architecture_health()

        # Fix identity dominance
        if health["avg_identity_dominance"] > 0.5:
            print("Applying Identity dominance fix...")
            with torch.no_grad():
                for cell in self.cells:
                    for edge in cell.edges:
                        if "Identity" in edge.available_ops:
                            identity_idx = edge.available_ops.index("Identity")
                            current_weights = F.softmax(edge.alphas, dim=0)
                            if current_weights[identity_idx] > 0.6:
                                edge.alphas.data[identity_idx] -= 0.2
                                for i, op in enumerate(edge.available_ops):
                                    if op != "Identity":
                                        edge.alphas.data[i] += 0.05

        # Fix attention issues
        if self.use_attention_bridge and hasattr(
            self.forecast_decoder, "attention_alphas"
        ):
            attention_weights = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
            if attention_weights[-1].item() > 0.8:
                print("Encouraging attention bridge usage...")
                with torch.no_grad():
                    self.forecast_decoder.attention_alphas.data[-1] -= 0.5
                    for i in range(len(self.forecast_decoder.attention_alphas) - 1):
                        self.forecast_decoder.attention_alphas.data[i] += 0.1

        # Add noise for low entropy
        low_entropy_count = sum(
            1 for issue in health["issues"] if "Low diversity" in issue
        )
        if low_entropy_count > len(health["issues"]) * 0.5:
            print("Adding noise to increase exploration...")
            with torch.no_grad():
                for cell in self.cells:
                    for edge in cell.edges:
                        edge.alphas.data += torch.randn_like(edge.alphas) * 0.1

                if hasattr(self.forecast_decoder, "attention_alphas"):
                    self.forecast_decoder.attention_alphas.data += (
                        torch.randn_like(self.forecast_decoder.attention_alphas) * 0.05
                    )

    def print_architecture_summary(self):
        """Print comprehensive architecture summary"""
        print("=" * 80)
        print("ENHANCED DARTS ARCHITECTURE SUMMARY")
        print("=" * 80)

        discrete_arch = self.derive_discrete_architecture()
        health = self.validate_architecture_health()

        # Print all components
        for comp_name, comp_data in discrete_arch.items():
            if comp_name.startswith("cell_"):
                print(f"\n{comp_name.upper()}:")
                for edge_name, edge_data in comp_data.items():
                    print(
                        f"  {edge_name}: {edge_data['operation']} ({edge_data['weight']:.3f})"
                    )
            else:
                print(
                    f"\n{comp_name.upper()}: {comp_data['type']} ({comp_data['weight']:.3f})"
                )

        # Health summary
        print(f"\nARCHITECTURE HEALTH:")
        print(f"  Health Score: {health['health_score']:.3f}")
        print(f"  Identity Dominance: {health['avg_identity_dominance']:.3f}")
        print(f"  Issues: {len(health['issues'])}")

        if health["issues"]:
            for issue in health["issues"][-3:]:  # Show last 3 issues
                print(f"    - {issue}")

        print("=" * 80)

    # === UTILITY METHODS ===
    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage(self) -> Dict[str, float]:
        total_params = sum(p.numel() * p.element_size() for p in self.parameters())
        total_buffers = sum(b.numel() * b.element_size() for b in self.buffers())
        return {
            "parameters_mb": total_params / (1024 * 1024),
            "buffers_mb": total_buffers / (1024 * 1024),
            "total_mb": (total_params + total_buffers) / (1024 * 1024),
        }
