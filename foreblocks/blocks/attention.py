import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List, Tuple, Union


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention for time series modeling.
    
    Processes time series at multiple time scales using a hierarchical structure.
    This approach is particularly effective for capturing both short-term and
    long-term dependencies in time series.
    
    Inspired by hierarchical architectures in natural language processing.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 8,
        num_levels: int = 3,
        pooling_kernel: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        positional_encoding: bool = True
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden feature dimension
            num_heads: Number of attention heads
            num_levels: Number of hierarchical levels
            pooling_kernel: Kernel size for downsampling between levels
            dropout: Dropout probability
            activation: Activation function
            positional_encoding: Whether to use positional encodings
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.pooling_kernel = pooling_kernel
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_size, dropout)
            
        # Multi-level attention blocks
        self.level_blocks = nn.ModuleList()
        for i in range(num_levels):
            level_block = nn.Sequential(
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            self.level_blocks.append(level_block)
            
        # Cross-level attention for information flow between levels
        self.cross_level_attentions = nn.ModuleList()
        for i in range(num_levels - 1):
            cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads // 2,  # Fewer heads for cross-level
                dropout=dropout,
                batch_first=True
            )
            self.cross_level_attentions.append(cross_attn)
            
        # Upsampling blocks for reconstruction
        self.upsample_blocks = nn.ModuleList()
        for i in range(num_levels - 1):
            up_block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * pooling_kernel),
                nn.Unfold(kernel_size=(pooling_kernel, 1), stride=1),
                self._get_activation(activation)
            )
            self.upsample_blocks.append(up_block)
            
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            self._get_activation(activation),
            nn.Linear(hidden_size, hidden_size)
        )
            
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh()
        }.get(activation.lower(), nn.GELU())
        
    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample sequence using average pooling"""
        # x shape: [batch, seq_len, hidden_size]
        # Transpose to [batch, hidden_size, seq_len]
        x_trans = x.transpose(1, 2)
        
        # Apply average pooling
        x_pooled = F.avg_pool1d(
            x_trans, 
            kernel_size=self.pooling_kernel,
            stride=self.pooling_kernel
        )
        
        # Transpose back to [batch, new_seq_len, hidden_size]
        return x_pooled.transpose(1, 2)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            mask: Optional padding mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Apply positional encoding if enabled
        if self.positional_encoding:
            x = self.pos_encoder(x)
            
        # Process each hierarchical level
        level_outputs = []
        current_x = x
        current_mask = mask
        
        # Downsampling pass - process increasingly coarser representations
        for i in range(self.num_levels):
            # Apply attention block at this level
            attn_block = self.level_blocks[i]
            
            # First part is MultiheadAttention
            mha = attn_block[0]
            residual = current_x
            
            # Apply attention with mask if provided
            if current_mask is not None:
                # Convert padding mask to attention mask
                attn_mask = current_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.to(torch.bool)
                attn_out, _ = mha(current_x, current_x, current_x, key_padding_mask=~current_mask)
            else:
                attn_out, _ = mha(current_x, current_x, current_x)
                
            current_x = attn_out + residual
            
            # Apply rest of the block (norm, FFN, etc.)
            for j in range(1, len(attn_block)):
                layer = attn_block[j]
                if isinstance(layer, nn.LayerNorm):
                    current_x = layer(current_x)
                elif j == 1:  # First linear after attention
                    residual = current_x
                    current_x = layer(current_x)
                elif j == len(attn_block) - 1:  # Last norm
                    current_x = layer(current_x + residual)
                else:
                    current_x = layer(current_x)
            
            # Store output at this level
            level_outputs.append(current_x)
            
            # Downsample for next level
            if i < self.num_levels - 1:
                current_x = self._downsample(current_x)
                if current_mask is not None:
                    # Downsample mask too (take every pooling_kernel-th value)
                    current_mask = current_mask[:, ::self.pooling_kernel]
                    if current_mask.size(1) > current_x.size(1):
                        current_mask = current_mask[:, :current_x.size(1)]
        
        # Upsampling pass with cross-level attention
        for i in range(self.num_levels - 2, -1, -1):
            # Get features from this level and the one above
            curr_features = level_outputs[i]
            higher_features = level_outputs[i + 1]
            
            # Apply cross-level attention
            # Higher level features attend to current level
            cross_attn = self.cross_level_attentions[i]
            
            # Upsample higher level features to match current level size
            if higher_features.size(1) < curr_features.size(1):
                # Simple repeat upsampling
                ratio = curr_features.size(1) // higher_features.size(1)
                higher_upsampled = higher_features.repeat_interleave(ratio, dim=1)
                
                # Cut to match exactly if needed
                if higher_upsampled.size(1) > curr_features.size(1):
                    higher_upsampled = higher_upsampled[:, :curr_features.size(1), :]
            else:
                higher_upsampled = higher_features
                
            # Cross-attention: higher features as query, current as key/value
            cross_out, _ = cross_attn(
                higher_upsampled, curr_features, curr_features
            )
            
            # Combine features with residual connection
            level_outputs[i] = curr_features + 0.5 * cross_out
        
        # Return the features from the first (most detailed) level
        output = self.output_proj(level_outputs[0])
        
        return output
class AutoCorrelationBlock(nn.Module):
    def __init__(self, d_model, factor=1, dropout=0.1):
        super().__init__()
        self.factor = factor
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def autocorrelation(self, query, key):
        # Ensure float32 for FFT compatibility
        query = query.float()
        key = key.float()

        B, T, D = query.shape

        # FFT over time
        q_fft = torch.fft.rfft(query, dim=1)
        k_fft = torch.fft.rfft(key, dim=1)
        corr_freq = q_fft * torch.conj(k_fft)
        corr_time = torch.fft.irfft(corr_freq, n=T, dim=1)  # [B, T, D]
        corr = corr_time.mean(dim=-1)  # [B, T]

        # Top-k delays (lags)
        topk = min(self.factor * int(math.log2(T)), T)
        _, delays = torch.topk(corr, topk, dim=1)  # [B, K]

        # Vectorized batched shifting
        out = torch.zeros_like(key)
        for i in range(topk):
            shift_i = delays[:, i]  # [B]
            shifted = torch.stack([
                torch.roll(key[b], shifts=-shift_i[b].item(), dims=0)
                for b in range(B)
            ], dim=0)  # [B, T, D]
            out += shifted

        return out / topk

    def forward(self, x):
        # x: [B, T, D]
        q = k = self.projection(x)
        v = x
        context = self.autocorrelation(q, k)
        return self.dropout(context + v)


class AutoCorrelationPreprocessor(nn.Module):
    def __init__(self, d_model=1, factor=1, dropout=0.1):
        super().__init__()
        self.block = AutoCorrelationBlock(d_model, factor=factor, dropout=dropout)

    def forward(self, x):
        return self.block(x)
