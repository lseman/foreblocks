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

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.scale * x / (norm + self.eps)


class MemoryEfficientOp(nn.Module):
    """Base class for memory-efficient operations with lazy initialization"""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._initialized = False

    def _lazy_init(self, x):
        """Override in subclasses for lazy initialization"""
        pass

    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x)
            self._initialized = True
        return self._forward(x)

    def _forward(self, x):
        """Override in subclasses for actual forward logic"""
        raise NotImplementedError


class IdentityOp(nn.Module):
    """Optimized identity operation"""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.transform = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x):
        return self.transform(x)



class TimeConvOp(MemoryEfficientOp):
    """Causal temporal convolution with depthwise separable structure"""

    def __init__(self, input_dim, latent_dim, kernel_size=3):
        super().__init__(input_dim, latent_dim)
        self.kernel_size = kernel_size

    def _lazy_init(self, x):
        self.depthwise = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            self.kernel_size,
            padding=self.kernel_size - 1,
            groups=self.input_dim,
            bias=False,
        ).to(x.device)
        self.pointwise = nn.Conv1d(
            self.input_dim, self.latent_dim, kernel_size=1, bias=False
        ).to(x.device)

        self.norm = RMSNorm(self.latent_dim).to(x.device)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(x.device)

    def _forward(self, x):
        residual = self.residual_proj(x)

        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = self.pointwise(x_conv)

        if x_conv.size(2) > residual.size(1):  # causal truncation
            x_conv = x_conv[:, :, : residual.size(1)]

        x_conv = x_conv.transpose(1, 2)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        return self.norm(x_conv + residual)


class TCNOp(MemoryEfficientOp):
    """Temporal Convolutional Network with dilated depthwise separable convs"""

    def __init__(self, input_dim, latent_dim, kernel_size=3, dilation=1):
        super().__init__(input_dim, latent_dim)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1)

    def _lazy_init(self, x):
        self.depthwise = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.input_dim,
            bias=False,
        ).to(x.device)
        self.pointwise = nn.Conv1d(
            self.input_dim, self.latent_dim, kernel_size=1, bias=False
        ).to(x.device)

        self.norm = RMSNorm(self.latent_dim).to(x.device)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Conv1d(self.input_dim, self.latent_dim, kernel_size=1, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(x.device)

    def _forward(self, x):
        B, L, C = x.shape
        x_t = x.transpose(1, 2)
        residual = self.residual_proj(x_t).transpose(1, 2)

        x_conv = self.depthwise(x_t)
        x_conv = self.pointwise(x_conv)

        if x_conv.size(2) > L:  # causal truncation
            x_conv = x_conv[:, :, :L]

        x_conv = x_conv.transpose(1, 2)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        return self.norm(x_conv + residual)


class ResidualMLPOp(MemoryEfficientOp):
    """MLP with residual and RMSNorm"""

    def __init__(self, input_dim, latent_dim, expansion_factor=2.67):
        super().__init__(input_dim, latent_dim)
        self.expansion_factor = expansion_factor

    def _lazy_init(self, x):
        hidden_dim = int(self.latent_dim * self.expansion_factor)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim, bias=False),
            nn.Dropout(0.05),
        ).to(x.device)

        self.norm = RMSNorm(self.latent_dim).to(x.device)
        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(x.device)

    def _forward(self, x):
        residual = self.residual_proj(x)
        out = self.mlp(x)
        return self.norm(out + residual)


class FourierOp(MemoryEfficientOp):
    """Efficient real-valued FFT-based operator with learnable frequency weighting"""

    def __init__(self, input_dim, latent_dim, seq_length, num_frequencies=None):
        super().__init__(input_dim, latent_dim)
        self.seq_length = seq_length
        self.num_frequencies = (
            min(seq_length // 2 + 1, 32)
            if num_frequencies is None
            else min(num_frequencies, seq_length // 2 + 1)
        )

    def _lazy_init(self, x):
        self.freq_proj = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=False),
        ).to(x.device)

        self.freq_weights = nn.Parameter(
            torch.randn(self.num_frequencies, device=x.device) * 0.02
        )
        self.gate = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=False), nn.Sigmoid()
        ).to(x.device)

        self.output_proj = nn.Linear(
            self.input_dim + self.latent_dim, self.latent_dim, bias=False
        ).to(x.device)
        self.norm = RMSNorm(self.latent_dim).to(x.device)

    def _forward(self, x):
        B, L, C = x.shape

        x_padded = (
            F.pad(x, (0, 0, 0, self.seq_length - L))
            if L < self.seq_length
            else x[:, : self.seq_length]
        )
        x_fft = torch.fft.rfft(x_padded, dim=1, norm="ortho")
        x_fft = x_fft[:, : self.num_frequencies]

        weights = F.softmax(self.freq_weights, dim=0).view(1, -1, 1)
        real = x_fft.real * weights
        imag = x_fft.imag * weights
        freq_feat = torch.cat([real, imag], dim=-1)
        freq_feat = self.freq_proj(freq_feat)
        global_feat = freq_feat.mean(dim=1, keepdim=True).expand(-1, L, -1)

        gated = self.gate(global_feat)
        combined = torch.cat([x[:, :L], gated * global_feat], dim=-1)
        return self.norm(self.output_proj(combined))


class WaveletOp(MemoryEfficientOp):
    """Efficient wavelet-style operation using dilated depthwise separable convolutions"""

    def __init__(self, input_dim, latent_dim, num_scales=3):
        super().__init__(input_dim, latent_dim)
        self.num_scales = num_scales

    def _lazy_init(self, x):
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
        ).to(x.device)

        self.fusion = nn.Conv1d(
            self.input_dim * self.num_scales, self.latent_dim, kernel_size=1, bias=False
        ).to(x.device)
        self.norm = RMSNorm(self.latent_dim).to(x.device)

    def _forward(self, x):
        B, L, C = x.shape
        x_t = x.transpose(1, 2)

        features = [layer(x_t) for layer in self.dwconv_layers]
        features = [
            F.adaptive_avg_pool1d(f, L) if f.shape[-1] != L else f for f in features
        ]

        out = torch.cat(features, dim=1)
        out = self.fusion(out).transpose(1, 2)
        return self.norm(out)


class ConvMixerOp(MemoryEfficientOp):
    """ConvMixer-style operator with depthwise separable conv and double residual"""

    def __init__(self, input_dim, latent_dim, kernel_size=9):
        super().__init__(input_dim, latent_dim)
        self.kernel_size = kernel_size

    def _lazy_init(self, x):
        self.input_proj = nn.Linear(self.input_dim, self.latent_dim, bias=False).to(
            x.device
        )

        self.depthwise = nn.Conv1d(
            self.latent_dim,
            self.latent_dim,
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.latent_dim,
            bias=False,
        ).to(x.device)
        self.pointwise = nn.Conv1d(
            self.latent_dim, self.latent_dim, kernel_size=1, bias=False
        ).to(x.device)

        self.norm1 = nn.BatchNorm1d(self.latent_dim).to(x.device)
        self.norm2 = RMSNorm(self.latent_dim).to(x.device)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

    def _forward(self, x):
        x = self.input_proj(x)  # [B, L, D]
        residual = x

        x_conv = x.transpose(1, 2)  # [B, D, L]
        x_conv = self.depthwise(x_conv)
        x_conv = self.norm1(x_conv)
        x_conv = self.activation(x_conv)

        x_conv = (
            self.pointwise(x_conv) + x_conv
        )  # local residual inside ConvMixer block
        x_conv = x_conv.transpose(1, 2)

        x_conv = self.dropout(x_conv)
        return self.norm2(x_conv + residual)  # outer residual


class GRNOp(MemoryEfficientOp):
    """Gated Residual Network with simplified linear structure"""

    def __init__(self, input_dim, latent_dim):
        super().__init__(input_dim, latent_dim)

    def _lazy_init(self, x):
        self.fc1 = nn.Linear(self.input_dim, self.latent_dim, bias=False).to(x.device)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim, bias=False).to(x.device)

        self.gate = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=False), nn.Sigmoid()
        ).to(x.device)

        self.norm = RMSNorm(self.latent_dim).to(x.device)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim
            else nn.Identity()
        ).to(x.device)

    def _forward(self, x):
        residual = self.residual_proj(x)

        h = self.activation(self.fc1(x))
        h = self.dropout(h)

        gated = self.gate(h)
        y = gated * self.fc2(h)

        return self.norm(y + residual)


class MultiScaleConvOp(MemoryEfficientOp):
    """Multi-scale convolutional operation with feature pyramid"""
    def __init__(self, input_dim, latent_dim, scales=[1, 3, 5, 7]):
        super().__init__(input_dim, latent_dim)
        self.scales = scales
        self.num_scales = len(scales)

    def _lazy_init(self, x):
        # Multi-scale depthwise separable convolutions
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.input_dim, self.input_dim, kernel_size=k,
                         padding=k//2, groups=self.input_dim, bias=False),
                nn.Conv1d(self.input_dim, self.latent_dim // self.num_scales, 
                         kernel_size=1, bias=False),
                nn.BatchNorm1d(self.latent_dim // self.num_scales),
                nn.GELU()
            ) for k in self.scales
        ]).to(x.device)
        
        # Feature fusion with attention
        self.attention = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.latent_dim // 4, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        ).to(x.device)
        
        # Final projection and normalization
        self.final_proj = nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1, bias=False).to(x.device)
        self.norm = RMSNorm(self.latent_dim).to(x.device)
        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim else nn.Identity()
        ).to(x.device)

    def _forward(self, x):
        B, L, C = x.shape
        residual = self.residual_proj(x)
        
        x_t = x.transpose(1, 2)  # [B, C, L]
        
        # Multi-scale feature extraction
        scale_features = []
        for conv in self.scale_convs:
            feat = conv(x_t)
            scale_features.append(feat)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=1)  # [B, latent_dim, L]
        
        # Attention-based fusion
        attn_weights = self.attention(multi_scale)  # [B, num_scales, L]
        
        # Apply attention to each scale
        weighted_features = []
        for i, feat in enumerate(scale_features):
            weighted = feat * attn_weights[:, i:i+1, :]
            weighted_features.append(weighted)
        
        # Combine weighted features
        combined = torch.stack(weighted_features, dim=0).sum(dim=0)  # [B, latent_dim//num_scales, L]
        
        # Expand to full dimension and project
        combined = combined.repeat(1, self.num_scales, 1)[:, :self.latent_dim, :]
        output = self.final_proj(combined).transpose(1, 2)  # [B, L, latent_dim]
        
        return self.norm(output + residual)


class PyramidConvOp(MemoryEfficientOp):
    """Pyramid convolution with progressive downsampling and upsampling"""
    def __init__(self, input_dim, latent_dim, levels=3):
        super().__init__(input_dim, latent_dim)
        self.levels = min(levels, 3)  # Limit levels to prevent too small channels

    def _lazy_init(self, x):
        # Calculate channel dimensions for each level
        base_channels = max(self.latent_dim // (2**self.levels), 8)  # Minimum 8 channels
        
        # Input projection to ensure we start with correct dimensions
        self.input_proj = nn.Conv1d(self.input_dim, base_channels * (2**self.levels), 
                                   kernel_size=1, bias=False).to(x.device)
        
        # Encoder pyramid (downsampling)
        encoder_channels = [base_channels * (2**(self.levels - i)) for i in range(self.levels + 1)]
        
        self.encoder_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(encoder_channels[i], encoder_channels[i+1], 
                         kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(encoder_channels[i+1]),
                nn.GELU(),
                nn.Dropout(0.05)
            ) for i in range(self.levels)
        ]).to(x.device)
        
        # Decoder pyramid (upsampling)
        decoder_channels = encoder_channels[::-1]  # Reverse for upsampling
        
        self.decoder_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(decoder_channels[i], decoder_channels[i+1],
                                 kernel_size=3, stride=2, padding=1, 
                                 output_padding=1, bias=False),
                nn.BatchNorm1d(decoder_channels[i+1]),
                nn.GELU()
            ) for i in range(self.levels)
        ]).to(x.device)
        
        # Skip connection fusion layers with correct channel dimensions
        self.skip_fusions = nn.ModuleList([
            nn.Conv1d(decoder_channels[i+1] + encoder_channels[self.levels-1-i], 
                     decoder_channels[i+1], kernel_size=1, bias=False)
            for i in range(self.levels - 1)
        ]).to(x.device)
        
        # Final projection to target dimension
        self.final_proj = nn.Conv1d(decoder_channels[-1], self.latent_dim, 
                                   kernel_size=1, bias=False).to(x.device)
        
        self.norm = RMSNorm(self.latent_dim).to(x.device)
        self.residual_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim else nn.Identity()
        ).to(x.device)

    def _forward(self, x):
        B, L, C = x.shape
        residual = self.residual_proj(x)
        
        x_t = x.transpose(1, 2)  # [B, C, L]
        
        # Project input to appropriate channel dimension
        x_proj = self.input_proj(x_t)
        
        # Encoder path - store features for skip connections
        encoder_features = [x_proj]
        current = x_proj
        
        for conv in self.encoder_convs:
            current = conv(current)
            encoder_features.append(current)
        
        # Decoder path with skip connections
        current = encoder_features[-1]  # Start from bottleneck
        
        for i, conv in enumerate(self.decoder_convs):
            current = conv(current)
            
            # Add skip connection if not the last layer
            if i < len(self.decoder_convs) - 1:
                # Get corresponding encoder feature (reverse order)
                skip_idx = self.levels - 1 - i
                skip = encoder_features[skip_idx]
                
                # Handle temporal dimension mismatches
                if current.shape[-1] != skip.shape[-1]:
                    target_len = min(current.shape[-1], skip.shape[-1])
                    current = current[:, :, :target_len]
                    skip = skip[:, :, :target_len]
                
                # Concatenate and fuse skip connection
                fused = torch.cat([current, skip], dim=1)
                current = self.skip_fusions[i](fused)
        
        # Final projection
        current = self.final_proj(current)
        
        # Handle final output size to match input length
        if current.shape[-1] != L:
            current = F.interpolate(current, size=L, mode='linear', align_corners=False)
        
        output = current.transpose(1, 2)  # [B, L, latent_dim]
        return self.norm(output + residual)


class SoftOpFusion(nn.Module):
    def __init__(self, num_ops: int, feature_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_ops = num_ops
        self.feature_dim = feature_dim

        # More robust fusion network with residual connection
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * num_ops, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_ops),
        )

        # Pre-create projection layers for common dimensions
        self.projections = nn.ModuleDict()

    def _get_or_create_projection(
        self, input_dim: int, device: torch.device
    ) -> nn.Module:
        """Get or create a projection layer for the given input dimension"""
        key = str(input_dim)
        if key not in self.projections:
            self.projections[key] = nn.Linear(input_dim, self.feature_dim)
            self.projections[key] = self.projections[key].to(device)
        return self.projections[key]

    def _align_tensor_dims(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is [B, T, D] format"""
        if tensor.dim() == 2:  # [B, D] -> [B, 1, D]
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 4:  # [B, C, H, W] -> flatten to [B, T, D]
            B, C, H, W = tensor.shape
            tensor = tensor.view(B, C, H * W).transpose(1, 2)

        # Ensure last dimension matches feature_dim
        if tensor.shape[-1] != self.feature_dim:
            proj = self._get_or_create_projection(tensor.shape[-1], tensor.device)
            tensor = proj(tensor)

        return tensor

    def forward(self, op_outputs: List[torch.Tensor]) -> torch.Tensor:
        if not op_outputs:
            raise ValueError("op_outputs cannot be empty")

        if len(op_outputs) != self.num_ops:
            # Pad or trim to match expected number of ops
            if len(op_outputs) < self.num_ops:
                # Duplicate last output to fill
                while len(op_outputs) < self.num_ops:
                    op_outputs.append(op_outputs[-1])
            else:
                # Take first num_ops outputs
                op_outputs = op_outputs[: self.num_ops]

        # Align all tensors to [B, T, D] format
        aligned_outputs = []
        for i, output in enumerate(op_outputs):
            try:
                aligned = self._align_tensor_dims(output)
                aligned_outputs.append(aligned)
            except Exception as e:
                print(
                    f"Warning: Failed to align tensor {i} with shape {output.shape}: {e}"
                )
                # Create a zero tensor as fallback
                if aligned_outputs:
                    fallback = torch.zeros_like(aligned_outputs[0])
                else:
                    # Create a basic fallback tensor
                    B = output.shape[0]
                    fallback = torch.zeros(
                        B, 1, self.feature_dim, device=output.device, dtype=output.dtype
                    )
                aligned_outputs.append(fallback)

        if not aligned_outputs:
            raise RuntimeError("No valid tensors after alignment")

        # Ensure all tensors have the same batch and sequence dimensions
        B = aligned_outputs[0].shape[0]
        min_seq_len = min(t.shape[1] for t in aligned_outputs)

        # Trim all tensors to same sequence length and ensure same batch size
        processed_outputs = []
        for t in aligned_outputs:
            if t.shape[0] != B:
                # Handle batch size mismatch
                if t.shape[0] > B:
                    t = t[:B]  # Trim
                else:
                    # Repeat to match batch size
                    repeat_factor = (B + t.shape[0] - 1) // t.shape[
                        0
                    ]  # Ceiling division
                    t = t.repeat(repeat_factor, 1, 1)[:B]

            t = t[:, :min_seq_len, :]  # Trim sequence length
            processed_outputs.append(t)

        # Stack and concatenate - now all tensors should have shape [B, min_seq_len, feature_dim]
        try:
            stacked_for_gating = torch.cat(
                processed_outputs, dim=-1
            )  # [B, T, D * num_ops]
            stacked_for_weighting = torch.stack(
                processed_outputs, dim=-1
            )  # [B, T, D, num_ops]
        except Exception as e:
            print(f"Error in tensor stacking: {e}")
            print(f"Tensor shapes: {[t.shape for t in processed_outputs]}")
            raise

        # Compute gating scores
        gate_logits = self.fusion(stacked_for_gating)  # [B, T, num_ops]
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Apply weighted combination
        # gate_scores: [B, T, num_ops] -> [B, T, 1, num_ops]
        # stacked_for_weighting: [B, T, D, num_ops]
        weighted_output = (stacked_for_weighting * gate_scores.unsqueeze(-2)).sum(
            dim=-1
        )

        return weighted_output  # [B, T, D]

    def get_gate_entropy(self, op_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute entropy of gating decisions for regularization"""
        if not op_outputs:
            return torch.tensor(0.0)

        with torch.no_grad():
            aligned_outputs = [self._align_tensor_dims(out) for out in op_outputs]
            min_seq_len = min(t.shape[1] for t in aligned_outputs)
            aligned_outputs = [t[:, :min_seq_len, :] for t in aligned_outputs]

            stacked = torch.cat(aligned_outputs, dim=-1)
            gate_logits = self.fusion(stacked)
            gate_probs = F.softmax(gate_logits, dim=-1)

            # Compute entropy: -sum(p * log(p))
            entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1)
            return entropy.mean()


class MixedOp(nn.Module):
    """Fixed MixedOp with better SoftOpFusion integration"""

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
        fuse_strategy: str = "weighted",
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.normalize_outputs = normalize_outputs
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.fuse_strategy = fuse_strategy.lower()
        self.latent_dim = latent_dim

        # Define operation mapping (you'll need to implement these)
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

        if not available_ops:
            available_ops = ["Identity", "TimeConv", "ResidualMLP"]

        # Remove duplicates while preserving order
        seen = set()
        self.available_ops = [
            op
            for op in available_ops
            if op in self.op_map and not (op in seen or seen.add(op))
        ]

        if len(self.available_ops) < 2:
            self.available_ops = ["Identity", "TimeConv", "ResidualMLP"]

        self.ops = nn.ModuleList([self.op_map[op]() for op in self.available_ops])
        self.alphas = nn.Parameter(torch.randn(len(self.ops)) * 0.02)

        if self.fuse_strategy == "soft":
            self.fusion = SoftOpFusion(
                num_ops=len(self.ops), feature_dim=latent_dim, dropout_rate=drop_prob
            )

        self._cached_weights = None

    def forward(self, x):
        weights = self._get_weights()
        device = x.device

        # Early exit for inference with dominant weight
        if not self.training and self.fuse_strategy in {"weighted", "hard"}:
            max_weight = weights.max()
            if max_weight > 0.95:
                return self.ops[weights.argmax().item()](x)

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

                # Ensure output has correct dimensions
                if out.shape[-1] != self.latent_dim:
                    # Use the projection system from SoftOpFusion
                    if not hasattr(self, "output_projections"):
                        self.output_projections = nn.ModuleDict()

                    key = f"{i}_{out.shape[-1]}"
                    if key not in self.output_projections:
                        self.output_projections[key] = nn.Linear(
                            out.shape[-1], self.latent_dim
                        ).to(device)

                    proj = self.output_projections[key]
                    out = proj(out)

                if self.normalize_outputs and self.training:
                    out = F.layer_norm(out, out.shape[-1:])

                op_outputs.append(out)
                active_weights.append(w)

            except Exception as e:
                if self.training:
                    print(f"Operation {self.available_ops[i]} failed: {e}")
                continue

        # Fallback to Identity if no operations succeeded
        if not op_outputs:
            identity_idx = (
                self.available_ops.index("Identity")
                if "Identity" in self.available_ops
                else 0
            )
            return self.ops[identity_idx](x)

        # Apply fusion strategy
        if self.fuse_strategy == "soft":
            try:
                return self.fusion(op_outputs)
            except Exception as e:
                print(f"SoftOpFusion failed: {e}, falling back to weighted fusion")
                # Fallback to weighted fusion
                return self._weighted_fusion(op_outputs, active_weights)

        elif self.fuse_strategy == "hard":
            # Hard selection using Gumbel
            one_hot = F.gumbel_softmax(
                self.alphas / self.temperature, tau=1.0, hard=True
            )
            idx = one_hot.argmax().item()
            return self.ops[idx](x)

        else:  # weighted fusion
            return self._weighted_fusion(op_outputs, active_weights)

    def _weighted_fusion(
        self, op_outputs: List[torch.Tensor], active_weights: List[torch.Tensor]
    ) -> torch.Tensor:
        """Fallback weighted fusion"""
        total_weight = sum(w.item() for w in active_weights)
        if total_weight < 1e-8:
            return op_outputs[0]  # Return first output if weights are too small

        result = sum(w * out for w, out in zip(active_weights, op_outputs))
        if self.normalize_outputs and abs(total_weight - 1.0) > 1e-4:
            result = result / total_weight
        return result

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

    def get_alphas(self):
        return F.softmax(self.alphas.detach(), dim=0)

    def set_temperature(self, temp: float):
        """Set temperature for softmax and clear cached weights"""
        self.temperature = temp
        self._cached_weights = None

    def get_entropy_loss(self):
        """Regularization loss to encourage exploration"""
        probs = F.softmax(self.alphas / self.temperature, dim=0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        return -0.05 * entropy

    def get_gate_entropy(self) -> torch.Tensor:
        """Get entropy from fusion layer if using soft fusion"""
        if self.fuse_strategy == "soft" and hasattr(self, "fusion"):
            # Create dummy outputs to compute entropy
            dummy_outputs = []
            for op in self.ops:
                dummy_input = torch.randn(
                    1, 10, self.latent_dim, device=next(op.parameters()).device
                )
                with torch.no_grad():
                    dummy_out = op(dummy_input)
                dummy_outputs.append(dummy_out)
            return self.fusion.get_gate_entropy(dummy_outputs)
        return torch.tensor(0.0)



class DARTSCell(nn.Module):
    """Highly optimized DARTS cell with memory efficiency and performance improvements"""
    
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
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.aggregation = aggregation
        self.temperature = temperature
        self.use_checkpoint = use_checkpoint
        self.enable_caching = enable_caching
        
        # Optimized input projection with better initialization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=False),
            RMSNorm(latent_dim),
            nn.GELU()
        )
        
        # Smart operation selection
        self.available_ops = self._select_operations(initial_search, selected_ops)
        
        # Pre-compute edge count for efficiency
        self.num_edges = sum(range(num_nodes))
        
        # Create optimized edges with shared components where possible
        self.edges = self._create_optimized_edges()
        
        # Efficient residual and aggregation weights
        self._init_learnable_weights()
        
        # Caching for inference optimization
        self._inference_cache = {} if enable_caching else None
        self._last_input_shape = None
        self._dominant_paths = None
        
        # Pre-allocate buffers for efficiency
        self._node_buffer = None
        self._edge_outputs_buffer = None
    
    def _select_operations(self, initial_search: bool, selected_ops: Optional[List[str]]) -> List[str]:
        """Smart operation selection with fallbacks"""
        if initial_search:
            return ["Identity", "TimeConv"]
        
        if selected_ops:
            # Validate selected operations
            valid_ops = ["Identity", "TimeConv", "Fourier", "ResidualMLP", "TCN", "Wavelet", "ConvMixer", "GRN", 
                         "MultiScaleConv", "PyramidConv"]
            filtered_ops = [op for op in selected_ops if op in valid_ops]
            return filtered_ops if len(filtered_ops) >= 2 else ["Identity", "TimeConv"]
        
        return ["Identity", "TimeConv", "Fourier", "ResidualMLP", "TCN"]
    
    def _create_optimized_edges(self) -> nn.ModuleList:
        """Create edges with potential parameter sharing for efficiency"""
        edges = nn.ModuleList()
        
        # Group edges by similar characteristics for potential sharing
        for edge_idx in range(self.num_edges):
            edge = MixedOp(
                input_dim=self.latent_dim,
                latent_dim=self.latent_dim,
                seq_length=self.seq_length,
                available_ops=self.available_ops,
                temperature=self.temperature,
                fuse_strategy="soft",  # Use soft fusion for better gradients
                drop_prob=0.1,
                normalize_outputs=True,
            )
            edges.append(edge)
        
        return edges
    
    def _init_learnable_weights(self):
        """Initialize learnable weights with better defaults"""
        # Residual weights with better initialization
        self.residual_weights = nn.Parameter(
            torch.full((self.num_nodes,), 0.2)  # Start with stronger residuals
        )
        
        # Aggregation weights for weighted combination
        if self.aggregation == "weighted" and self.num_nodes > 2:
            max_inputs = self.num_nodes - 1
            self.agg_weights = nn.Parameter(torch.ones(max_inputs))
        else:
            self.agg_weights = None
        
        # Optional learned temperature for adaptive exploration
        self.learned_temp = nn.Parameter(torch.tensor(1.0))
    
    def _get_edge_index(self, node_idx: int, input_idx: int) -> int:
        """Efficiently compute edge index"""
        return sum(range(node_idx)) + input_idx
    
    def _aggregate_inputs_efficient(self, inputs: List[torch.Tensor], node_idx: int) -> torch.Tensor:
        """Optimized input aggregation with multiple strategies"""
        if not inputs:
            raise ValueError("No inputs to aggregate")
        
        # Ensure all inputs are tensors
        tensor_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                tensor_inputs.append(inp)
            elif isinstance(inp, (list, tuple)) and len(inp) > 0:
                if isinstance(inp[0], torch.Tensor):
                    tensor_inputs.append(torch.stack(inp) if len(inp) > 1 else inp[0])
                else:
                    raise TypeError(f"Invalid input type in aggregation: {type(inp[0])}")
            else:
                raise TypeError(f"Invalid input type in aggregation: {type(inp)}")
        
        if len(tensor_inputs) == 1:
            return tensor_inputs[0]
        
        # Efficient aggregation strategies
        if self.aggregation == "weighted" and self.agg_weights is not None:
            num_inputs = len(tensor_inputs)
            if num_inputs <= self.agg_weights.size(0):
                weights = F.softmax(self.agg_weights[:num_inputs], dim=0)
                # Vectorized weighted sum
                stacked_inputs = torch.stack(tensor_inputs, dim=0)
                weighted_result = torch.sum(
                    weights.view(-1, 1, 1, 1) * stacked_inputs, dim=0
                )
                return weighted_result
        
        elif self.aggregation == "attention":
            # Simple attention-based aggregation
            stacked_inputs = torch.stack(tensor_inputs, dim=0)  # [num_inputs, B, T, D]
            
            # Compute attention weights
            query = stacked_inputs.mean(dim=0, keepdim=True)  # [1, B, T, D]
            scores = torch.sum(query * stacked_inputs, dim=-1, keepdim=True)  # [num_inputs, B, T, 1]
            weights = F.softmax(scores, dim=0)
            
            return torch.sum(weights * stacked_inputs, dim=0)
        
        elif self.aggregation == "max":
            stacked_inputs = torch.stack(tensor_inputs, dim=0)
            return torch.max(stacked_inputs, dim=0)[0]
        
        else:  # mean (default)
            stacked_inputs = torch.stack(tensor_inputs, dim=0)
            return torch.mean(stacked_inputs, dim=0)
    
    def _apply_residual_connection(self, node_output: torch.Tensor, 
                                 previous_node: torch.Tensor, node_idx: int) -> torch.Tensor:
        """Efficient residual connection with learned gating"""
        residual_weight = torch.sigmoid(self.residual_weights[node_idx])
        
        # Handle dimension mismatches efficiently
        if node_output.shape != previous_node.shape:
            # Simple interpolation for sequence length mismatch
            if node_output.shape[1] != previous_node.shape[1]:
                target_len = node_output.shape[1]
                previous_node = F.interpolate(
                    previous_node.transpose(1, 2), 
                    size=target_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
        
        return residual_weight * node_output + (1 - residual_weight) * previous_node
    
    def _forward_with_caching(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with intelligent caching for inference"""
        if not self.training and self.enable_caching:
            input_hash = hash((x.shape, x.device, x.dtype))
            
            if input_hash in self._inference_cache:
                return self._inference_cache[input_hash]
        
        result = self._forward_core(x)
        
        # Cache result for inference
        if not self.training and self.enable_caching:
            # Limit cache size to prevent memory bloat
            if len(self._inference_cache) > 100:
                self._inference_cache.clear()
            self._inference_cache[input_hash] = result.detach()
        
        return result
    
    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward computation with optimizations"""
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            if isinstance(x, (list, tuple)):
                x = torch.stack(x) if len(x) > 1 else x[0]
            else:
                raise TypeError(f"Expected tensor input, got {type(x)}")
        
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
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
                
                # Apply edge operation
                if self.use_checkpoint and self.training:
                    # Use gradient checkpointing for memory efficiency
                    edge_output = torch.utils.checkpoint.checkpoint(
                        edge, nodes[input_idx], use_reentrant=False
                    )
                else:
                    edge_output = edge(nodes[input_idx])
                
                node_inputs.append(edge_output)
            
            # Aggregate inputs
            aggregated = self._aggregate_inputs_efficient(node_inputs, node_idx)
            
            # Apply residual connection
            if node_idx > 0:
                node_output = self._apply_residual_connection(
                    aggregated, nodes[node_idx - 1], node_idx
                )
            else:
                node_output = aggregated
            
            nodes.append(node_output)
        
        # Final output with input residual
        final_residual_weight = torch.sigmoid(self.residual_weights[0])
        final_output = (final_residual_weight * nodes[-1] + 
                       (1 - final_residual_weight) * x_proj)
        
        return final_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main forward pass with optional caching"""
        if self.enable_caching:
            return self._forward_with_caching(x)
        else:
            return self._forward_core(x)
    
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
        return total_gate_entropy / len(self.edges)  # Average across edges
    
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
        summary = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'available_ops': self.available_ops,
            'aggregation': self.aggregation,
        }
        
        # Get dominant operations for each edge
        edge_ops = []
        for i, edge in enumerate(self.edges):
            alphas = edge.get_alphas()
            dominant_op_idx = alphas.argmax().item()
            dominant_op = self.available_ops[dominant_op_idx]
            edge_ops.append({
                'edge_idx': i,
                'dominant_op': dominant_op,
                'confidence': alphas[dominant_op_idx].item()
            })
        
        summary['edge_operations'] = edge_ops
        return summary
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        edge_params = sum(sum(p.numel() for p in edge.parameters()) for edge in self.edges)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'edge_parameters': edge_params,
            'edge_percentage': (edge_params / total_params * 100) if total_params > 0 else 0
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
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2  # MB
            return {
                'allocated_mb': allocated,
                'cached_mb': cached,
                'cache_size': len(self._inference_cache) if self._inference_cache else 0
            }
        return {'cache_size': len(self._inference_cache) if self._inference_cache else 0}
    
from .darts_base import *

class TimeSeriesDARTS(nn.Module):
    """Enhanced TimeSeriesDARTS with cross-attention bridge"""

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
        self._store_config(locals())
        self._init_components()
        self._init_weights()
        self._setup_compilation()

    def _store_config(self, config: Dict[str, Any]) -> None:
        """Store configuration parameters"""
        config.pop("self")
        config.pop("__class__", None)
        for key, value in config.items():
            setattr(self, key, value)

    def _init_components(self) -> None:
        """Initialize all model components"""
        self._init_embedding_layers()
        self._init_darts_cells()
        self._init_forecasting_components()
        self._init_output_layers()

    def _init_embedding_layers(self) -> None:
        """Initialize input embedding and positional encoding"""
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),  # Changed from RMSNorm for compatibility
            nn.GELU(),
        )
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, self.hidden_dim) * 0.02
        )

    def _init_darts_cells(self) -> None:
        """Initialize DARTS cells with projections and scaling"""
        self.cells = nn.ModuleList()
        self.cell_proj = nn.ModuleList()
        self.layer_scales = nn.ParameterList()

        for i in range(self.num_cells):
            temp = self.temperature * (0.8**i)

            # DARTS cell (assuming DARTSCell is defined elsewhere)
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
            self.layer_scales.append(nn.Parameter(torch.ones(1) * 1e-2))

        self.cell_weights = nn.Parameter(torch.ones(self.num_cells))

    def _init_forecasting_components(self) -> None:
        """Initialize encoder, decoder, and fusion components with attention bridge"""
        # Assuming MixedEncoder is defined elsewhere
        self.forecast_encoder = MixedEncoder(
            self.hidden_dim,
            self.latent_dim,
            seq_len=self.seq_length,
            dropout=self.dropout,
            temperature=self.temperature,
        )
        
        # Enhanced decoder with attention bridge
        self.forecast_decoder = MixedDecoder(
            self.input_dim,
            self.latent_dim,
            seq_len=self.seq_length,
            dropout=self.dropout,
            temperature=self.temperature,
            use_attention_bridge=self.use_attention_bridge,
            attention_layers=self.attention_layers
        )

        self.gate_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), 
            nn.Sigmoid()
        )

    def _init_output_layers(self) -> None:
        """Initialize MLP and output layers"""
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim, bias=False),
            nn.Dropout(self.dropout * 0.5),
        )
        self.output_layer = nn.Linear(self.latent_dim, self.input_dim, bias=False)
        self.forecast_norm = nn.LayerNorm(self.latent_dim)

    def _init_weights(self) -> None:
        """Initialize model weights with proper strategies"""
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

    def _setup_compilation(self) -> None:
        """Setup torch.compile if requested"""
        self._compiled_forward = None
        if self.use_compile and hasattr(torch, "compile"):
            try:
                self._compiled_forward = torch.compile(
                    self._uncompiled_forward, mode="default"
                )
            except Exception as e:
                print(f"[Compile Warning] torch.compile failed: {e}")

    def _forward_cells(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DARTS cells with feature fusion"""
        current_input = x
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

            # Residual connection for deeper cells
            current_input = cell_out + current_input * 0.1 if i > 0 else cell_out

        # Weighted combination of cell features
        if len(cell_features) > 1:
            weights = F.softmax(self.cell_weights[: len(cell_features)], dim=0)
            combined = sum(w * f for w, f in zip(weights, cell_features))
        else:
            combined = cell_features[0]

        return combined

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
        """Core forward implementation with attention bridge"""
        B, L, _ = x_seq.shape

        # Input embedding
        x_emb = self.input_embedding(x_seq)

        # DARTS cell processing
        final_features = self._forward_cells(x_seq)

        # Feature fusion
        fuse_input = torch.cat([final_features, x_emb], dim=-1)
        alpha = self.gate_fuse(fuse_input)
        combined = alpha * final_features + (1 - alpha) * x_emb

        # Encoding
        h_enc, context, encoder_state = self.forecast_encoder(combined)

        # Decoding with attention bridge
        return self._decode_forecasts(
            x_seq, x_future, context, encoder_state, h_enc, teacher_forcing_ratio
        )

    def _decode_forecasts(
        self,
        x_seq: torch.Tensor,
        x_future: Optional[torch.Tensor],
        context: torch.Tensor,
        encoder_state: torch.Tensor,
        encoder_output: torch.Tensor,  # Full encoder output for attention
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        """Decode forecasts with cross-attention bridge"""
        forecasts = []
        decoder_input = x_seq[:, -1:, :]
        decoder_hidden = encoder_state

        for t in range(self.forecast_horizon):
            # Decoder with cross-attention to encoder
            out, decoder_hidden = self.forecast_decoder(
                decoder_input, context, decoder_hidden, encoder_output
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

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights for analysis"""
        weights = {}
        
        if hasattr(self.forecast_decoder, 'attention_alphas'):
            attention_weights = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
            weights['attention_bridge'] = attention_weights
            
        decoder_weights = F.softmax(self.forecast_decoder.alphas, dim=0)
        weights['decoder_type'] = decoder_weights
        
        return weights

    def calculate_loss(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Calculate total loss with attention regularization"""
        pred = self.forward(x_seq, x_future, teacher_forcing_ratio)

        # Main prediction loss
        loss_functions = {
            "huber": lambda p, t: F.huber_loss(p, t, delta=0.1),
            "mse": F.mse_loss,
            "mae": F.l1_loss,
        }
        main_loss = loss_functions.get(self.loss_type, F.smooth_l1_loss)(pred, x_future)

        # Regularization losses
        entropy_loss = self._compute_entropy_loss()
        alpha_l2 = self._compute_alpha_l2()
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

    def _compute_entropy_loss(self) -> torch.Tensor:
        """Compute entropy regularization loss including attention"""
        entropy_loss = sum(cell.get_entropy_loss() for cell in self.cells) * 1e-3
        entropy_loss += self.forecast_encoder.get_entropy_loss()
        entropy_loss += self.forecast_decoder.get_entropy_loss()
        return entropy_loss

    def _compute_alpha_l2(self) -> torch.Tensor:
        """Compute L2 regularization on architecture parameters"""
        alpha_l2 = 0.0
        for cell in self.cells:
            for edge in cell.edges:
                alpha_l2 += (edge.alphas**2).sum()
        alpha_l2 += (self.forecast_encoder.alphas**2).sum()
        alpha_l2 += (self.forecast_decoder.alphas**2).sum()
        
        # Add attention alphas L2 regularization
        if hasattr(self.forecast_decoder, 'attention_alphas'):
            alpha_l2 += (self.forecast_decoder.attention_alphas**2).sum()
        
        return alpha_l2 * 1e-4

    def _compute_attention_regularization(self) -> torch.Tensor:
        """Compute attention-specific regularization"""
        if not self.use_attention_bridge or not hasattr(self.forecast_decoder, 'attention_alphas'):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Encourage diversity in attention choices
        attn_probs = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
        
        # Entropy regularization for attention diversity
        attn_entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum()
        
        # Penalty for always choosing "no attention" (last option)
        no_attention_penalty = attn_probs[-1] * 0.1
        
        return -attn_entropy * 1e-3 + no_attention_penalty

    # Architecture Analysis Methods
    def get_alphas(self) -> List[torch.Tensor]:
        """Get all architecture parameters including attention"""
        alphas = [cell.get_alphas() for cell in self.cells]
        alphas.extend([
            self.forecast_encoder.get_alphas(),
            self.forecast_decoder.get_alphas(),
        ])
        return alphas

    def get_alpha_dict(self) -> Dict[str, torch.Tensor]:
        """Get architecture parameters as named dictionary including attention"""
        alpha_map = {}
        
        # Cell alphas
        for idx, cell in enumerate(self.cells):
            edge_idx = 0
            for i in range(cell.num_nodes):
                for j in range(i + 1, cell.num_nodes):
                    if edge_idx < len(cell.edges):
                        alpha_map[f"cell{idx}_edge_{i}->{j}"] = cell.edges[
                            edge_idx
                        ].get_alphas()
                        edge_idx += 1

        alpha_map["encoder_type"] = self.forecast_encoder.get_alphas()
        
        # Decoder alphas (including attention)
        decoder_alphas = self.forecast_decoder.get_alphas()
        alpha_map["decoder_type"] = decoder_alphas[:len(self.forecast_decoder.decoder_names)]
        
        # Attention bridge alphas
        if self.use_attention_bridge and hasattr(self.forecast_decoder, 'attention_alphas'):
            alpha_map["attention_bridge"] = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
        
        return alpha_map

    def get_operation_weights(self) -> Dict[str, Dict[str, float]]:
        """Get current operation weights including attention for analysis"""
        weights = {}

        # Cell weights
        for i, cell in enumerate(self.cells):
            for j, edge in enumerate(cell.edges):
                weights[f"cell_{i}_edge_{j}"] = {
                    op: weight.item()
                    for op, weight in zip(edge.available_ops, edge.get_alphas())
                }

        # Encoder/Decoder weights
        for component, names_attr in [
            ("encoder", "encoder_names"),
            ("decoder", "decoder_names"),
        ]:
            comp_obj = getattr(self, f"forecast_{component}")
            names = getattr(comp_obj, names_attr)
            alphas = comp_obj.get_alphas()
            if component == "decoder":
                # Only take decoder type alphas, not attention alphas
                alphas = alphas[:len(names)]
            weights[component] = {
                name: weight.item() for name, weight in zip(names, alphas)
            }

        # Attention bridge weights
        if self.use_attention_bridge and hasattr(self.forecast_decoder, 'attention_alphas'):
            attention_weights = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
            attention_names = [f"attention_layer_{i}" for i in range(len(attention_weights)-1)] + ["no_attention"]
            weights["attention_bridge"] = {
                name: weight.item() for name, weight in zip(attention_names, attention_weights)
            }

        return weights

    def derive_discrete_architecture(
        self, threshold: float = 0.3, top_k_fallback: int = 2
    ) -> Dict[str, Any]:
        """Derive discrete architecture including attention choices"""
        discrete_arch = {}

        # Cell architecture
        for i, cell in enumerate(self.cells):
            cell_arch = {}
            for j, edge in enumerate(cell.edges):
                weights = edge.get_alphas()
                max_weight = weights.max().item()
                max_idx = weights.argmax().item()

                if max_weight > threshold:
                    cell_arch[f"edge_{j}"] = {
                        "operation": edge.available_ops[max_idx],
                        "weight": max_weight,
                        "confidence": max_weight,
                    }
                else:
                    # Fallback to top-k
                    topk_indices = torch.topk(weights, k=top_k_fallback).indices
                    topk_ops = [edge.available_ops[idx.item()] for idx in topk_indices]
                    topk_weights = [weights[idx].item() for idx in topk_indices]

                    cell_arch[f"edge_{j}"] = {
                        "operation": topk_ops[0],
                        "alternatives": list(zip(topk_ops, topk_weights)),
                        "weight": topk_weights[0],
                        "confidence": topk_weights[0],
                    }

            discrete_arch[f"cell_{i}"] = cell_arch

        # Encoder/Decoder selection
        for comp_name, comp_obj in [
            ("encoder", self.forecast_encoder),
            ("decoder", self.forecast_decoder),
        ]:
            if comp_name == "decoder":
                weights = F.softmax(comp_obj.alphas, dim=0)  # Only decoder type weights
                names = comp_obj.decoder_names
            else:
                weights = comp_obj.get_alphas()
                names = getattr(comp_obj, f"{comp_name}_names")
            
            idx = weights.argmax().item()
            discrete_arch[comp_name] = {
                "type": names[idx],
                "weight": weights.max().item(),
            }

        # Attention bridge selection
        if self.use_attention_bridge and hasattr(self.forecast_decoder, 'attention_alphas'):
            attention_weights = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
            max_idx = attention_weights.argmax().item()
            
            if max_idx == len(attention_weights) - 1:
                attention_choice = "no_attention"
            else:
                attention_choice = f"attention_layer_{max_idx}"
            
            discrete_arch["attention_bridge"] = {
                "type": attention_choice,
                "weight": attention_weights.max().item(),
                "num_layers": len(attention_weights) - 1,
            }

        return discrete_arch

    def validate_architecture_health(self) -> Dict[str, Any]:
        """Comprehensive architecture health validation including attention"""
        issues = []
        total_identity_dominance = 0
        total_edges = 0

        # Check cell health
        for i, cell in enumerate(self.cells):
            for j, edge in enumerate(cell.edges):
                weights = edge.get_alphas()
                entropy = -(weights * torch.log(weights + 1e-8)).sum().item()
                max_weight = weights.max().item()

                # Check for issues
                if entropy < 0.5:
                    issues.append(
                        f"Cell {i}, Edge {j}: Low diversity (entropy={entropy:.3f})"
                    )

                if max_weight > 0.9:
                    dominant_op = edge.available_ops[weights.argmax().item()]
                    issues.append(
                        f"Cell {i}, Edge {j}: {dominant_op} dominates ({max_weight:.3f})"
                    )

                # Track Identity dominance
                if "Identity" in edge.available_ops:
                    identity_idx = edge.available_ops.index("Identity")
                    identity_weight = weights[identity_idx].item()
                    total_identity_dominance += identity_weight

                    if identity_weight > 0.7:
                        issues.append(
                            f"Cell {i}, Edge {j}: Identity dominates ({identity_weight:.3f})"
                        )

                total_edges += 1

        # Check attention health
        if self.use_attention_bridge and hasattr(self.forecast_decoder, 'attention_alphas'):
            attention_weights = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
            attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum().item()
            
            if attention_entropy < 0.5:
                issues.append(f"Attention bridge: Low diversity (entropy={attention_entropy:.3f})")
            
            # Check if always choosing no attention
            no_attention_weight = attention_weights[-1].item()
            if no_attention_weight > 0.8:
                issues.append(f"Attention bridge: No attention dominates ({no_attention_weight:.3f})")

        avg_identity_dominance = total_identity_dominance / max(total_edges, 1)
        health_score = max(
            0, 1.0 - len(issues) / max((total_edges * 2 + 2), 1) - avg_identity_dominance
        )

        return {
            "issues": issues,
            "avg_identity_dominance": avg_identity_dominance,
            "total_edges": total_edges,
            "health_score": health_score,
            "attention_enabled": self.use_attention_bridge,
        }

    def apply_architecture_fixes(self) -> None:
        """Apply automatic fixes for architecture issues including attention"""
        health = self.validate_architecture_health()

        if health["avg_identity_dominance"] > 0.5:
            print("Applying Identity dominance fix...")
            self.encourage_diversity()

        # Fix attention issues
        if self.use_attention_bridge and hasattr(self.forecast_decoder, 'attention_alphas'):
            attention_weights = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
            no_attention_weight = attention_weights[-1].item()
            
            if no_attention_weight > 0.8:
                print("Encouraging attention bridge usage...")
                with torch.no_grad():
                    # Reduce no-attention weight and boost attention layers
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
                        noise = torch.randn_like(edge.alphas) * 0.1
                        edge.alphas.data += noise
                
                # Add noise to attention if needed
                if hasattr(self.forecast_decoder, 'attention_alphas'):
                    attn_noise = torch.randn_like(self.forecast_decoder.attention_alphas) * 0.05
                    self.forecast_decoder.attention_alphas.data += attn_noise

    def encourage_diversity(self) -> None:
        """Encourage diversity by reducing Identity dominance and encouraging attention"""
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

    def get_diversity_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed diversity metrics including attention"""
        metrics = {}

        # Cell metrics
        for i, cell in enumerate(self.cells):
            for j, edge in enumerate(cell.edges):
                weights = edge.get_alphas()
                entropy = -(weights * torch.log(weights + 1e-8)).sum().item()
                max_weight = weights.max().item()

                identity_dominance = 0.0
                if "Identity" in edge.available_ops:
                    identity_idx = edge.available_ops.index("Identity")
                    identity_dominance = weights[identity_idx].item()

                metrics[f"cell_{i}_edge_{j}"] = {
                    "entropy": entropy,
                    "max_weight": max_weight,
                    "identity_dominance": identity_dominance,
                    "num_active_ops": (weights > 0.1).sum().item(),
                }

        # Attention metrics
        if self.use_attention_bridge and hasattr(self.forecast_decoder, 'attention_alphas'):
            attention_weights = F.softmax(self.forecast_decoder.attention_alphas, dim=0)
            attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum().item()
            
            metrics["attention_bridge"] = {
                "entropy": attention_entropy,
                "max_weight": attention_weights.max().item(),
                "no_attention_dominance": attention_weights[-1].item(),
                "num_attention_layers": len(attention_weights) - 1,
            }

        return metrics

    # Utility Methods
    def get_model_size(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get estimated memory usage in MB"""
        total_params = sum(p.numel() * p.element_size() for p in self.parameters())
        total_buffers = sum(b.numel() * b.element_size() for b in self.buffers())
        return {
            "parameters_mb": total_params / (1024 * 1024),
            "buffers_mb": total_buffers / (1024 * 1024),
            "total_mb": (total_params + total_buffers) / (1024 * 1024),
        }

    def print_architecture_summary(self) -> None:
        """Print a comprehensive summary of the discovered architecture"""
        print("=" * 80)
        print("ENHANCED DARTS ARCHITECTURE SUMMARY")
        print("=" * 80)
        
        # Get current weights
        weights = self.get_operation_weights()
        discrete_arch = self.derive_discrete_architecture()
        
        # Print cell operations
        print("\nDARTS CELLS:")
        for cell_name, cell_ops in weights.items():
            if cell_name.startswith("cell_"):
                print(f"\n{cell_name.upper()}:")
                for edge_name, ops in cell_ops.items():
                    max_op = max(ops, key=ops.get)
                    max_weight = ops[max_op]
                    print(f"  {edge_name}: {max_op} ({max_weight:.3f})")
        
        # Print encoder/decoder
        print(f"\nENCODER: {discrete_arch['encoder']['type']} ({discrete_arch['encoder']['weight']:.3f})")
        print(f"DECODER: {discrete_arch['decoder']['type']} ({discrete_arch['decoder']['weight']:.3f})")
        
        # Print attention bridge
        if 'attention_bridge' in weights:
            print(f"\nATTENTION BRIDGE:")
            for attn_type, weight in weights['attention_bridge'].items():
                print(f"  {attn_type}: {weight:.3f}")
            
            if 'attention_bridge' in discrete_arch:
                chosen_attention = discrete_arch['attention_bridge']
                print(f"  -> CHOSEN: {chosen_attention['type']} ({chosen_attention['weight']:.3f})")
        
        # Print health metrics
        health = self.validate_architecture_health()
        print(f"\nARCHITECTURE HEALTH:")
        print(f"  Health Score: {health['health_score']:.3f}")
        print(f"  Identity Dominance: {health['avg_identity_dominance']:.3f}")
        print(f"  Total Issues: {len(health['issues'])}")
        
        if health['issues']:
            print(f"  Recent Issues:")
            for issue in health['issues'][-5:]:  # Show last 5 issues
                print(f"    - {issue}")
        
        print("=" * 80)


