import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

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


class FixedOp(nn.Module):
    """Optimized FixedOp with better efficiency"""
    def __init__(self, selected_op: nn.Module):
        super().__init__()
        self.op = selected_op

    def forward(self, x):
        return self.op(x)


class RMSNorm(nn.Module):
    """RMS Normalization - more efficient than LayerNorm"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # More efficient computation
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.scale * x / (rms + self.eps)


# Monkey patch to add RMSNorm to nn module
nn.RMSNorm = RMSNorm


class TransformerOp(nn.Module):
    """Optimized Transformer operation with Flash Attention style improvements"""

    def __init__(self, input_dim, latent_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # Auto-adjust heads for optimal performance
        self.num_heads = min(num_heads, latent_dim // 8)
        while latent_dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        
        self.head_dim = latent_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.latent_dim = latent_dim

        # Efficient input projection
        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False) if input_dim != latent_dim else nn.Identity()

        # Fused QKV projection for better memory bandwidth
        self.qkv = nn.Linear(latent_dim, latent_dim * 3, bias=False)
        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.RMSNorm(latent_dim)
        
        # Optimized feedforward
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4, bias=False),
            nn.GELU(),  # Faster than SiLU
            nn.Linear(latent_dim * 4, latent_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.RMSNorm(latent_dim)

    def forward(self, x):
        B, L, _ = x.shape
        x_proj = self.input_proj(x)
        
        # Multi-head attention with optimized computation
        qkv = self.qkv(x_proj).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Use optimized attention based on sequence length
        if L <= 128:  # Standard attention for short sequences
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
            out = torch.matmul(attn, v)
        else:  # Linear attention for long sequences
            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0
            # More efficient einsum operations
            kv = torch.einsum("bhld,bhlm->bhdm", k, v)
            out = torch.einsum("bhld,bhdm->bhlm", q, kv)
            k_sum = k.sum(dim=2, keepdim=True)
            normalizer = torch.einsum("bhld,bhmd->bhlm", q, k_sum)
            out = out / (normalizer + 1e-8)

        out = out.reshape(B, L, self.latent_dim)
        out = self.dropout(self.out_proj(out))
        out = self.norm1(out + x_proj)
        
        # Feedforward with residual
        ffn_out = self.ffn(out)
        return self.norm2(ffn_out + out)


class IdentityOp(nn.Module):
    """Optimized identity operation"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.transform = nn.Linear(input_dim, latent_dim, bias=False) if input_dim != latent_dim else nn.Identity()

    def forward(self, x):
        return self.transform(x)


class WaveletOp(nn.Module):
    """Optimized wavelet operation with depthwise separable convolutions"""

    def __init__(self, input_dim, latent_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        # More efficient depthwise separable convolutions
        self.dwconv_layers = nn.ModuleList()
        for dilation in [1, 2, 4][:num_scales]:
            dwconv = nn.Sequential(
                # Depthwise convolution
                nn.Conv1d(input_dim, input_dim, 3, padding=dilation, dilation=dilation, 
                         groups=input_dim, bias=False),
                # Pointwise convolution
                nn.Conv1d(input_dim, input_dim, 1, bias=False),
                nn.BatchNorm1d(input_dim),
                nn.GELU(),
                nn.Dropout(0.05),
            )
            self.dwconv_layers.append(dwconv)

        # Efficient fusion
        self.fusion = nn.Conv1d(input_dim * num_scales, latent_dim, 1, bias=False)
        self.norm = nn.RMSNorm(latent_dim)

    def forward(self, x):
        B, L, C = x.shape
        x_conv = x.transpose(1, 2)

        # Process all scales in parallel
        features = [layer(x_conv) for layer in self.dwconv_layers]
        
        # Ensure all features have the same length
        for i, feat in enumerate(features):
            if feat.size(2) != L:
                features[i] = F.adaptive_avg_pool1d(feat, L)

        # Efficient concatenation and fusion
        concat = torch.cat(features, dim=1)
        out = self.fusion(concat).transpose(1, 2)
        return self.norm(out)


class FourierOp(nn.Module):
    """Optimized Fourier operation with cached FFT and learnable frequencies"""

    def __init__(self, input_dim, latent_dim, seq_length, num_frequencies=None):
        super().__init__()
        self.seq_length = seq_length
        self.num_frequencies = min(seq_length // 2 + 1, 32) if num_frequencies is None else min(num_frequencies, seq_length // 2 + 1)

        # Efficient frequency processing
        self.freq_proj = nn.Sequential(
            nn.Linear(input_dim * 2, latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )

        # Learnable frequency selection
        self.freq_weights = nn.Parameter(torch.randn(self.num_frequencies) * 0.02)
        
        # Simplified gating
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(input_dim + latent_dim, latent_dim, bias=False)
        self.norm = nn.RMSNorm(latent_dim)

    def forward(self, x):
        B, L, C = x.shape

        # Efficient FFT with proper padding
        if L != self.seq_length:
            x_padded = F.pad(x, (0, 0, 0, max(0, self.seq_length - L)))
            x_fft = torch.fft.rfft(x_padded, dim=1, norm="ortho")
        else:
            x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
            
        x_fft = x_fft[:, :self.num_frequencies, :]

        # Apply learnable frequency weighting
        freq_weights = F.softmax(self.freq_weights, dim=0)
        x_fft = x_fft * freq_weights.view(1, -1, 1)

        # Process frequency features
        real_imag = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        freq_features = self.freq_proj(real_imag)

        # Global-local fusion with gating
        global_feat = freq_features.mean(dim=1, keepdim=True).expand(-1, L, -1)
        gate = self.gate(global_feat)
        
        # Combine with input
        combined = torch.cat([x, gate * global_feat], dim=-1)
        return self.norm(self.output_proj(combined))


class AttentionOp(nn.Module):
    """Optimized self-attention with efficient computation"""

    def __init__(self, input_dim, latent_dim, num_heads=8, dropout=0.1):
        super().__init__()

        # Auto-adjust heads
        self.num_heads = min(num_heads, latent_dim // 8)
        while latent_dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.head_dim = latent_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False) if input_dim != latent_dim else nn.Identity()
        
        # Fused QKV for better memory efficiency
        self.qkv = nn.Linear(latent_dim, latent_dim * 3, bias=False)
        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(latent_dim)

    def forward(self, x):
        B, L, _ = x.shape
        x_proj = self.input_proj(x)

        # Efficient QKV computation
        qkv = self.qkv(x_proj).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Adaptive attention mechanism
        if L <= 64:  # Use standard attention for very short sequences
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
            out = torch.matmul(attn, v)
        else:  # Use linear attention
            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0
            kv = torch.einsum("bhld,bhlm->bhdm", k, v)
            out = torch.einsum("bhld,bhdm->bhlm", q, kv)
            k_sum = k.sum(dim=2, keepdim=True)
            normalizer = torch.einsum("bhld,bhmd->bhlm", q, k_sum)
            out = out / (normalizer + 1e-8)

        out = out.reshape(B, L, -1)
        out = self.dropout(self.out_proj(out))
        return self.norm(out + x_proj)


class TimeConvOp(nn.Module):
    """Optimized temporal convolution with causal padding"""

    def __init__(self, input_dim, latent_dim, kernel_size=3):
        super().__init__()

        # Efficient depthwise separable convolution
        self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size, padding=kernel_size-1, 
                                 groups=input_dim, bias=False)
        self.pointwise = nn.Conv1d(input_dim, latent_dim, 1, bias=False)

        self.norm = nn.BatchNorm1d(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        # Residual projection
        self.residual_proj = nn.Linear(input_dim, latent_dim, bias=False) if input_dim != latent_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)

        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        
        # Causal truncation
        if x_conv.size(2) > residual.size(1):
            x_conv = x_conv[:, :, :residual.size(1)]
        
        x_conv = self.pointwise(x_conv)
        x_conv = self.norm(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)

        return x_conv + residual


class TCNOp(nn.Module):
    """Optimized Temporal Convolutional Network"""

    def __init__(self, input_dim, latent_dim, kernel_size=3, dilation=1):
        super().__init__()

        padding = dilation * (kernel_size - 1)
        # Efficient depthwise separable convolution
        self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size, padding=padding, 
                                 dilation=dilation, groups=input_dim, bias=False)
        self.pointwise = nn.Conv1d(input_dim, latent_dim, 1, bias=False)
        self.norm = nn.BatchNorm1d(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        # Residual connection
        self.residual = nn.Conv1d(input_dim, latent_dim, 1, bias=False) if input_dim != latent_dim else nn.Identity()

    def forward(self, x):
        residual = x.transpose(1, 2)
        residual = self.residual(residual).transpose(1, 2)

        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        
        # Causal truncation
        if x_conv.size(2) > residual.size(1):
            x_conv = x_conv[:, :, :residual.size(1)]
        
        x_conv = self.pointwise(x_conv)
        x_conv = self.norm(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)

        return x_conv + residual


class ResidualMLPOp(nn.Module):
    """Optimized Residual MLP with GELU activation"""

    def __init__(self, input_dim, latent_dim, expansion_factor=2.67):
        super().__init__()
        
        hidden_dim = int(latent_dim * expansion_factor)
        
        # Efficient feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim, bias=False),
            nn.Dropout(0.05)
        )
        
        self.norm = nn.RMSNorm(latent_dim)
        self.residual_proj = nn.Linear(input_dim, latent_dim, bias=False) if input_dim != latent_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        output = self.mlp(x)
        return self.norm(output + residual)


class ConvMixerOp(nn.Module):
    """Optimized ConvMixer with efficient convolutions"""

    def __init__(self, input_dim, latent_dim, kernel_size=9):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        # Efficient depthwise + pointwise structure
        self.depthwise = nn.Conv1d(latent_dim, latent_dim, kernel_size, 
                                 groups=latent_dim, padding=kernel_size // 2, bias=False)
        self.pointwise = nn.Conv1d(latent_dim, latent_dim, 1, bias=False)

        self.norm1 = nn.BatchNorm1d(latent_dim)
        self.norm2 = nn.RMSNorm(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.input_proj(x)
        residual = x

        x_conv = x.transpose(1, 2)
        x_conv = self.depthwise(x_conv)
        x_conv = self.norm1(x_conv)
        x_conv = self.activation(x_conv)
        
        x_conv = self.pointwise(x_conv)
        x_conv = x_conv + x_conv  # Residual within conv
        x_conv = x_conv.transpose(1, 2)
        
        x_conv = self.dropout(x_conv)
        return self.norm2(x_conv + residual)


class GRNOp(nn.Module):
    """Optimized Gated Residual Network"""

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Efficient linear layers
        self.linear1 = nn.Linear(input_dim, latent_dim, bias=False)
        self.linear2 = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Simplified gating
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.Sigmoid(),
        )

        self.norm = nn.RMSNorm(latent_dim)
        self.dropout = nn.Dropout(0.05)
        self.activation = nn.GELU()

        # Residual projection
        self.residual_proj = nn.Linear(input_dim, latent_dim, bias=False) if input_dim != latent_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)

        h = self.activation(self.linear1(x))
        h = self.dropout(h)
        
        g = self.gate(h)
        y = g * self.linear2(h)

        return self.norm(y + residual)


class MixedOp(nn.Module):
    """Highly optimized mixed operation with efficient sampling"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        seq_length,
        available_ops=None,
        drop_prob=0.1,
        normalize_outputs=True,
        temperature=1.0,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.normalize_outputs = normalize_outputs
        self.temperature = temperature

        # Optimized operation map
        self.op_map = {
            "Identity": lambda: IdentityOp(input_dim, latent_dim),
            "Wavelet": lambda: WaveletOp(input_dim, latent_dim),
            "Fourier": lambda: FourierOp(input_dim, latent_dim, seq_length),
            "Attention": lambda: AttentionOp(input_dim, latent_dim),
            "TCN": lambda: TCNOp(input_dim, latent_dim),
            "ResidualMLP": lambda: ResidualMLPOp(input_dim, latent_dim),
            "ConvMixer": lambda: ConvMixerOp(input_dim, latent_dim),
            "Transformer": lambda: TransformerOp(input_dim, latent_dim),
            "GRN": lambda: GRNOp(input_dim, latent_dim),
            "TimeConv": lambda: TimeConvOp(input_dim, latent_dim),
        }

        if not available_ops:
            available_ops = ["Identity", "Attention", "TimeConv", "Fourier"]

        self.available_ops = list(available_ops)
        self.ops = nn.ModuleList([self.op_map[op]() for op in self.available_ops if op in self.op_map])

        # Better initialization
        self.alphas = nn.Parameter(torch.zeros(len(self.ops)), requires_grad=True)
        nn.init.normal_(self.alphas, std=0.01)
        
        # Cache for efficiency
        self._cached_weights = None
        self._cache_counter = 0

    def forward(self, x):
        # Use cached weights for efficiency during inference
        if not self.training and self._cached_weights is not None:
            weights = self._cached_weights
        else:
            if self.training:
                weights = F.gumbel_softmax(self.alphas, tau=self.temperature, hard=False)
            else:
                weights = F.softmax(self.alphas / self.temperature, dim=0)
            
            if not self.training:
                self._cached_weights = weights

        # Efficient operation execution with early stopping
        result = None
        total_weight = 0.0
        
        for i, (weight, op) in enumerate(zip(weights, self.ops)):
            # Skip operations with very low weights
            if weight < 1e-3:
                continue

            # Stochastic depth during training
            if self.training and torch.rand(1).item() < self.drop_prob:
                continue

            try:
                out = op(x)
                if self.normalize_outputs and self.training:  # Only normalize during training
                    out = F.layer_norm(out, out.shape[-1:])
                
                if result is None:
                    result = weight * out
                else:
                    result = result + weight * out
                total_weight += weight
                
            except Exception as e:
                continue

        if result is None:
            return x

        # Normalize by actual weights used
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            result = result / total_weight

        return result

    def get_alphas(self):
        return F.softmax(self.alphas, dim=0)

    def get_entropy_loss(self):
        probs = F.softmax(self.alphas, dim=0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        return -0.01 * entropy


class DARTSCell(nn.Module):
    """Optimized DARTS cell with better connectivity"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        seq_length,
        num_nodes=4,
        initial_search=False,
        selected_ops=None,
        aggregation="mean",
        temperature=1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.aggregation = aggregation

        # Efficient input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=False), 
            nn.RMSNorm(latent_dim), 
            nn.GELU()
        )

        # Operation selection
        if initial_search:
            self.available_ops = ["Identity", "Attention", "TimeConv"]
        elif selected_ops:
            self.available_ops = selected_ops
        else:
            self.available_ops = [
                "Identity", "TimeConv", "Attention", "Fourier",
                "GRN", "ResidualMLP", "TCN", "ConvMixer", "Transformer", "Wavelet",
            ]

        # Create edges efficiently
        num_edges = sum(range(num_nodes))
        self.edges = nn.ModuleList([
            MixedOp(latent_dim, latent_dim, seq_length, self.available_ops, temperature=temperature)
            for _ in range(num_edges)
        ])

        # Node processing
        self.node_norms = nn.ModuleList([nn.RMSNorm(latent_dim) for _ in range(num_nodes)])
        
        # Learnable aggregation weights
        if num_nodes > 2:
            self.agg_weights = nn.Parameter(torch.ones(num_nodes - 1))

    def forward(self, x):
        x_proj = self.input_proj(x)
        nodes = [x_proj]

        edge_idx = 0
        for i in range(1, self.num_nodes):
            # Collect inputs from previous nodes
            inputs = []
            for j in range(i):
                edge_out = self.edges[edge_idx + j](nodes[j])
                inputs.append(edge_out)
            edge_idx += i

            # Efficient aggregation
            if len(inputs) == 1:
                node_output = inputs[0]
            else:
                if self.aggregation == "weighted" and hasattr(self, 'agg_weights'):
                    weights = F.softmax(self.agg_weights[:len(inputs)], dim=0)
                    node_output = sum(w * inp for w, inp in zip(weights, inputs))
                else:
                    node_output = torch.stack(inputs, dim=0).mean(dim=0)

            # Apply normalization with small residual
            if i > 0:
                node_output = self.node_norms[i](node_output + nodes[i-1] * 0.1)
            else:
                node_output = self.node_norms[i](node_output)
            nodes.append(node_output)

        return nodes[-1] + x_proj * 0.1

    def get_alphas(self):
        return [edge.get_alphas() for edge in self.edges]

    def get_entropy_loss(self):
        return sum(edge.get_entropy_loss() for edge in self.edges)


class TimeSeriesDARTS(nn.Module):
    """Highly optimized main DARTS model"""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        latent_dim=64,
        forecast_horizon=24,
        seq_length=48,
        num_cells=2,
        num_nodes=4,
        dropout=0.1,
        initial_search=False,
        selected_ops=None,
        loss_type="huber",
        use_gradient_checkpointing=False,
        temperature=1.0,
        memory_efficient=True,
        use_compile=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.forecast_horizon = forecast_horizon
        self.seq_length = seq_length
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.memory_efficient = memory_efficient

        # Efficient input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False), 
            nn.RMSNorm(hidden_dim), 
            nn.GELU()
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, hidden_dim) * 0.02, requires_grad=True)

        # DARTS Cells with progressive temperature
        self.cells = nn.ModuleList()
        for i in range(num_cells):
            cell_temp = temperature * (0.8 ** i)
            cell = DARTSCell(
                input_dim=input_dim if i == 0 else latent_dim,
                latent_dim=latent_dim,
                seq_length=seq_length,
                num_nodes=num_nodes,
                initial_search=initial_search,
                selected_ops=selected_ops,
                aggregation="weighted" if i > 0 else "mean",
                temperature=cell_temp,
            )
            self.cells.append(cell)

        # Cell projections
        self.cell_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim, bias=False),
                nn.RMSNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            for _ in range(num_cells)
        ])

        # Efficient encoder-decoder
        self.forecast_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_cells > 1 else 0,
            bidirectional=False,
        )

        self.forecast_decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_cells > 1 else 0,
        )

        # Optimized MLP
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim, bias=False),
            nn.Dropout(dropout * 0.5),
        )

        # Simplified output layer
        self.output_layer = nn.Linear(latent_dim, input_dim, bias=False)
        self.forecast_norm = nn.RMSNorm(latent_dim)
        self.loss_type = loss_type

        self._init_weights()

        # Apply torch.compile if requested and available
        if use_compile and hasattr(torch, 'compile'):
            try:
                # Use default mode for better compatibility
                self.forward = torch.compile(self.forward, mode="default")
                print("Successfully applied torch.compile with default mode")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Using eager execution.")

    def _init_weights(self):
        """Optimized weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use truncated normal for better convergence
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
            elif isinstance(m, (nn.RMSNorm, nn.LayerNorm, nn.BatchNorm1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_cells(self, x):
        """Optimized forward through cells"""
        current_input = x
        cell_features = []

        for i, (cell, proj) in enumerate(zip(self.cells, self.cell_proj)):
            if self.use_gradient_checkpointing and self.training:
                cell_out = checkpoint(cell, current_input, use_reentrant=False)
            else:
                cell_out = cell(current_input)
            
            projected = proj(cell_out)
            cell_features.append(projected)
            
            # Efficient residual connection
            if i > 0:
                current_input = cell_out + current_input * 0.1
            else:
                current_input = cell_out

        # Efficient feature fusion
        if len(cell_features) > 1:
            # Simple weighted average instead of complex attention
            weights = F.softmax(torch.randn(len(cell_features), device=x.device), dim=0)
            combined_features = sum(w * feat for w, feat in zip(weights, cell_features))
        else:
            combined_features = cell_features[0]

        return combined_features

    @torch.jit.ignore
    def forward(self, x_seq, x_future=None, teacher_forcing_ratio=0.5):
        batch, seq_len, _ = x_seq.shape

        # Efficient input processing
        x_emb = self.input_embedding(x_seq)
        
        # Adaptive positional encoding
        if seq_len <= self.seq_length:
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            # More efficient interpolation
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        x_emb = x_emb + pos_enc

        # Process through cells
        final_features = self._forward_cells(x_seq)

        # Efficient feature combination
        alpha = torch.sigmoid((final_features * x_emb).mean(dim=-1, keepdim=True))
        combined = alpha * final_features + (1 - alpha) * x_emb

        # Encoder-decoder with optimized processing
        h_enc, hn_enc = self.forecast_encoder(combined)
        
        # Simplified context computation
        context = h_enc[:, -1:, :]  # Just use last state
        h_n = hn_enc

        forecasts = []
        decoder_input = x_seq[:, -1:, :]

        # Optimized decoder loop
        for t in range(self.forecast_horizon):
            output, h_n = self.forecast_decoder(decoder_input, h_n)

            # Simplified processing
            processed = self.forecast_norm(output + context)
            processed = self.mlp(processed) + output
            
            # Direct prediction
            prediction = self.output_layer(processed)
            forecasts.append(prediction.squeeze(1))

            # Efficient teacher forcing
            if (self.training and x_future is not None and 
                torch.rand(1).item() < teacher_forcing_ratio):
                decoder_input = x_future[:, t : t + 1, :]
            else:
                decoder_input = prediction

        return torch.stack(forecasts, dim=1)

    def calculate_loss(self, x_seq, x_future, teacher_forcing_ratio=0.5, return_components=False):
        """Optimized loss calculation"""
        pred = self.forward(x_seq, x_future, teacher_forcing_ratio)

        # Efficient loss computation
        if self.loss_type == "huber":
            main_loss = F.huber_loss(pred, x_future, delta=0.1, reduction='mean')
        elif self.loss_type == "mse":
            main_loss = F.mse_loss(pred, x_future, reduction='mean')
        elif self.loss_type == "mae":
            main_loss = F.l1_loss(pred, x_future, reduction='mean')
        else:
            main_loss = F.smooth_l1_loss(pred, x_future, reduction='mean')
        
        # Simplified regularization
        entropy_loss = sum(cell.get_entropy_loss() for cell in self.cells) * 0.001
        
        total_loss = main_loss + entropy_loss
        
        if return_components:
            return {
                'total_loss': total_loss,
                'main_loss': main_loss,
                'entropy_loss': entropy_loss,
            }
        
        return total_loss

    def get_alphas(self):
        """Get architecture parameters"""
        return [cell.get_alphas() for cell in self.cells]

    def get_alpha_dict(self):
        """Get detailed architecture parameters"""
        alpha_map = {}
        for idx, cell in enumerate(self.cells):
            edge_idx = 0
            for i in range(cell.num_nodes):
                for j in range(i + 1, cell.num_nodes):
                    alpha_map[f"cell{idx}_edge_{i}->{j}"] = cell.edges[edge_idx].get_alphas()
                    edge_idx += 1
        return alpha_map

    def get_model_size(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)