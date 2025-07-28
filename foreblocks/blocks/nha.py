import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalAttention(nn.Module):
    """
    Optimized hierarchical attention with reduced computational complexity.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # Single linear layer for Q, K, V to reduce memory allocation
        self.qkv = nn.Linear(input_dim, hidden_dim * num_heads * 3, bias=False)
        self.output_projection = nn.Linear(hidden_dim * num_heads, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Single matrix multiplication for Q, K, V
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.hidden_dim)
        q, k, v = qkv.unbind(2)  # Split into Q, K, V

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, hidden_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with fused operations
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.output_projection(context)


class TemporalConvLayer(nn.Module):
    """
    Optimized temporal convolution with grouped convolutions for efficiency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = True,
        dropout: float = 0.1,
        groups: int = 1,  # Added groups for efficiency
    ):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Use groups to reduce parameters and computation
        self.groups = min(groups, min(in_channels, out_channels))

        if causal:
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=0,
                dilation=dilation,
                groups=self.groups,
                bias=False,
            )
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation,
                groups=self.groups,
                bias=False,
            )

        # Fused activation and normalization
        self.norm_act = nn.Sequential(
            nn.LayerNorm(out_channels), nn.GELU(), nn.Dropout(dropout, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In-place operations where possible
        x_conv = x.transpose(1, 2)

        if self.causal:
            x_conv = F.pad(x_conv, (self.padding, 0))

        y = self.conv(x_conv).transpose(1, 2)
        return self.norm_act(y)


class HierarchicalBlock(nn.Module):
    """
    Optimized hierarchical block with reduced memory allocations and
    computational complexity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_levels: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        pooling_kernel: int = 2,
        expressiveness_ratio: float = 1.0,
        residual_connections: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        self.residual_connections = residual_connections
        self.pooling_kernel = pooling_kernel

        # Single input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)

        # Pre-compute dilation factors
        self.dilations = [2**i for i in range(num_levels)]

        # Use ModuleList but with shared components where possible
        conv_groups = max(1, hidden_dim // 8)  # Adaptive grouping

        self.temporal_convs = nn.ModuleList(
            [
                TemporalConvLayer(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    groups=conv_groups,
                )
                for dilation in self.dilations
            ]
        )

        # Simplified attention - reduce heads for inner levels
        self.attention_layers = nn.ModuleList(
            [
                HierarchicalAttention(
                    hidden_dim,
                    hidden_dim // 2,
                    num_heads=max(1, attention_heads // (i + 1)),
                    dropout=dropout,
                )
                for i in range(num_levels)
            ]
        )

        # Fused transformations to reduce layer count
        self.level_gate_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid(),
                )
                for _ in range(num_levels)
            ]
        )

        # Simplified projections
        self.backcast_projection = nn.Linear(hidden_dim, input_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=False)

        # Reduced level projections - use shared weights
        self.level_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Lighter cross-level attention
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=max(1, attention_heads // 2),
            dropout=dropout,
            batch_first=True,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout, inplace=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def _apply_multi_rate_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized multi-rate sampling with minimal operations."""
        if self.pooling_kernel <= 1:
            return x

        # Use strided operations instead of adaptive pooling when possible
        if self.pooling_kernel == 2:
            # Simple 2x subsampling and upsampling
            x_pooled = x.transpose(1, 2)
            x_pooled = F.avg_pool1d(x_pooled, 2, stride=1, padding=0)
            # Pad to original length if needed
            if x_pooled.size(-1) < x.size(1):
                pad_size = x.size(1) - x_pooled.size(-1)
                x_pooled = F.pad(x_pooled, (0, pad_size), mode="replicate")
            elif x_pooled.size(-1) > x.size(1):
                x_pooled = x_pooled[..., : x.size(1)]
            return x_pooled.transpose(1, 2)
        else:
            # Fallback to adaptive pooling for other kernel sizes
            x_pooled = x.transpose(1, 2)
            x_pooled = F.adaptive_avg_pool1d(x_pooled, x.size(1))
            return x_pooled.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply multi-rate sampling
        x_sampled = self._apply_multi_rate_sampling(x)

        # Project input
        h = self.input_projection(x_sampled)

        # Pre-allocate level outputs to avoid dynamic list growth
        level_outputs = []
        current_h = h

        # Process levels with optimized operations
        for i in range(self.num_levels):
            # Temporal convolution
            conv_out = self.temporal_convs[i](current_h)

            # Attention (skip for some levels if performance critical)
            if i % 2 == 0 or self.num_levels <= 2:  # Apply attention selectively
                attn_out = self.attention_layers[i](conv_out)
            else:
                attn_out = conv_out  # Skip attention for some levels

            # Fused transformation and gating
            level_transform = self.level_gate_transforms[i]
            level_features = level_transform[0](attn_out)  # Transform

            gate_input = torch.cat([current_h, level_features], dim=-1)
            gate = level_transform[1:](gate_input)  # Gate computation

            # Update with residual
            if self.residual_connections:
                current_h = current_h + gate * level_features
            else:
                current_h = gate * level_features

            current_h = self.layer_norm(current_h)

            # Shared level projection
            level_proj = self.level_projection(current_h)
            level_outputs.append(level_proj)

        # Simplified cross-level attention (only if multiple levels)
        if len(level_outputs) > 1 and self.num_levels > 2:
            # Process only every other level for efficiency
            selected_levels = level_outputs[::2]  # Take every 2nd level
            if len(selected_levels) > 1:
                stacked = torch.stack(selected_levels, dim=2)  # [B, T, L, H]
                B, T, L, H = stacked.shape

                # Reshape efficiently
                reshaped = stacked.view(B * T, L, H)
                attended, _ = self.cross_level_attention(reshaped, reshaped, reshaped)
                attended = attended.view(B, T, L, H)

                # Update selected levels
                for i, level_idx in enumerate(range(0, len(level_outputs), 2)):
                    if level_idx < len(level_outputs):
                        level_outputs[level_idx] = attended[:, :, i, :]

        # Fast final representation
        if len(level_outputs) == 1:
            final_representation = level_outputs[0]
        else:
            final_representation = torch.stack(level_outputs, dim=0).mean(dim=0)

        final_representation = self.dropout(final_representation)

        # Output projections
        backcast = self.backcast_projection(final_representation)
        output_embedding = self.output_projection(final_representation)

        return backcast, output_embedding


class NHA(nn.Module):
    """
    Optimized Neural Hierarchical Architecture (NHA) with significant performance improvements:
    - Reduced memory allocations
    - Fused operations
    - Selective computation
    - Efficient attention mechanisms
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        num_levels_per_block: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        share_blocks: bool = False,
        pooling_kernels: list = None,
        expressiveness_ratios: list = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.share_blocks = share_blocks

        # Optimized default configurations
        if pooling_kernels is None:
            pooling_kernels = [
                max(1, 4 // (i + 1)) for i in range(num_blocks)
            ]  # Smaller kernels

        if expressiveness_ratios is None:
            expressiveness_ratios = [1.0] * num_blocks  # Simplified ratios

        self.pooling_kernels = pooling_kernels[:num_blocks]
        self.expressiveness_ratios = expressiveness_ratios[:num_blocks]

        # Create blocks with weight sharing when beneficial
        if share_blocks:
            self.shared_block = HierarchicalBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_levels=num_levels_per_block,
                kernel_size=kernel_size,
                attention_heads=attention_heads,
                dropout=dropout,
                pooling_kernel=self.pooling_kernels[0],
                expressiveness_ratio=self.expressiveness_ratios[0],
            )
            self.blocks = nn.ModuleList([self.shared_block] * num_blocks)
        else:
            self.blocks = nn.ModuleList(
                [
                    HierarchicalBlock(
                        input_dim=input_dim if i == 0 else hidden_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        num_levels=num_levels_per_block,
                        kernel_size=kernel_size,
                        attention_heads=max(
                            1, attention_heads // (i + 1)
                        ),  # Reduce heads in later blocks
                        dropout=dropout,
                        pooling_kernel=self.pooling_kernels[i],
                        expressiveness_ratio=self.expressiveness_ratios[i],
                    )
                    for i in range(num_blocks)
                ]
            )

        # Simplified final layers
        self.embedding_projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)

        # Pre-allocate projection layers to avoid dynamic creation
        self._backcast_projections = nn.ModuleDict()

    def _get_or_create_projection(
        self, i: int, input_size: int, output_size: int, device: torch.device
    ) -> nn.Module:
        """Efficiently manage projection layers."""
        key = f"proj_{i}_{input_size}_{output_size}"
        if key not in self._backcast_projections:
            self._backcast_projections[key] = nn.Linear(
                input_size, output_size, bias=False
            ).to(device)
        return self._backcast_projections[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass with minimal memory allocations."""
        batch_size, seq_len, _ = x.size()
        current_input = x

        # Process blocks with optimized residual handling
        for i, block in enumerate(self.blocks):
            backcast, block_output = block(current_input)

            # Efficient dimension matching
            if backcast.shape != current_input.shape:
                if backcast.size(1) != current_input.size(1):
                    # Use more efficient interpolation
                    backcast = F.interpolate(
                        backcast.transpose(1, 2),
                        size=current_input.size(1),
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)

                if backcast.size(-1) != current_input.size(-1):
                    proj = self._get_or_create_projection(
                        i, backcast.size(-1), current_input.size(-1), backcast.device
                    )
                    backcast = proj(backcast)

            # Optimized residual computation
            if i == 0:
                current_input = block_output
            else:
                current_input = current_input + block_output

        # Final processing
        sequence_embedding = self.embedding_projection(current_input)
        sequence_embedding = self.layer_norm(sequence_embedding)
        sequence_embedding = self.dropout(sequence_embedding)

        return sequence_embedding

    def get_pooled_embedding(
        self, x: torch.Tensor, pooling: str = "mean"
    ) -> torch.Tensor:
        """Optimized pooling operations."""
        sequence_embeddings = self.forward(x)

        if pooling == "mean":
            return sequence_embeddings.mean(dim=1)
        elif pooling == "max":
            return sequence_embeddings.max(dim=1)[0]
        elif pooling == "last":
            return sequence_embeddings[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def extract_hierarchical_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Lightweight feature extraction for interpretation."""
        current_input = x
        results = {
            "input": x,
            "block_outputs": [],
            "pooling_kernels": self.pooling_kernels,
            "expressiveness_ratios": self.expressiveness_ratios,
        }

        for i, block in enumerate(self.blocks):
            backcast, block_output = block(current_input)
            results["block_outputs"].append(block_output)

            if i == 0:
                current_input = block_output
            else:
                current_input = current_input + block_output

        sequence_embedding = self.embedding_projection(current_input)
        results["sequence_embedding"] = self.layer_norm(
            self.dropout(sequence_embedding)
        )

        return results
