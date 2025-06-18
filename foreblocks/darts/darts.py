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
    """RMS Normalization - more efficient than LayerNorm"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.scale * x / (rms + self.eps)


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


class AttentionOp(MemoryEfficientOp):
    """Memory-efficient self-attention with automatic fallbacks"""
    def __init__(self, input_dim, latent_dim, num_heads=8, dropout=0.1):
        super().__init__(input_dim, latent_dim)
        self.num_heads = min(num_heads, max(1, latent_dim // 8))
        while latent_dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = latent_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

    def _lazy_init(self, x):
        self.input_proj = (
            nn.Linear(self.input_dim, self.latent_dim, bias=False)
            if self.input_dim != self.latent_dim else nn.Identity()
        ).to(x.device)
        self.qkv = nn.Linear(self.latent_dim, self.latent_dim * 3, bias=False).to(x.device)
        self.out_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False).to(x.device)
        self.norm = RMSNorm(self.latent_dim).to(x.device)

    def _forward(self, x):
        B, L, _ = x.shape
        x_proj = self.input_proj(x)
        
        qkv = self.qkv(x_proj).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if L <= 64:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.training:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ v
        else:
            # Linear attention for long sequences
            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0
            kv = torch.einsum("bhld,bhlm->bhdm", k, v)
            out = torch.einsum("bhld,bhdm->bhlm", q, kv)
            normalizer = torch.einsum("bhld,bhd->bhl", q, k.sum(dim=2))
            out = out / (normalizer.unsqueeze(-1) + 1e-8)

        out = out.transpose(1, 2).reshape(B, L, -1)
        out = F.dropout(self.out_proj(out), p=self.dropout, training=self.training)
        return self.norm(out + x_proj)


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
                    nn.Conv1d(self.input_dim, self.input_dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.input_dim),
                    nn.GELU(),
                    nn.Dropout(0.05),
                )
                for d in [1, 2, 4][:self.num_scales]
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
        self.input_proj = nn.Linear(self.input_dim, self.latent_dim, bias=False).to(x.device)

        self.depthwise = nn.Conv1d(
            self.latent_dim,
            self.latent_dim,
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.latent_dim,
            bias=False,
        ).to(x.device)
        self.pointwise = nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1, bias=False).to(x.device)

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


class MixedOp(nn.Module):
    """Optimized mixed operation with better efficiency and caching"""
    def __init__(
        self,
        input_dim,
        latent_dim,
        seq_length,
        available_ops=None,
        drop_prob=0.1,
        normalize_outputs=True,
        temperature=1.0,
        use_gumbel=False,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.normalize_outputs = normalize_outputs
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        self.op_map = {
            "Identity": lambda: IdentityOp(input_dim, latent_dim),
            "Wavelet": lambda: WaveletOp(input_dim, latent_dim),
            "Fourier": lambda: FourierOp(input_dim, latent_dim, seq_length),
            "Attention": lambda: AttentionOp(input_dim, latent_dim),
            "TCN": lambda: TCNOp(input_dim, latent_dim),
            "ResidualMLP": lambda: ResidualMLPOp(input_dim, latent_dim),
            "ConvMixer": lambda: ConvMixerOp(input_dim, latent_dim),
            "GRN": lambda: GRNOp(input_dim, latent_dim),
            "TimeConv": lambda: TimeConvOp(input_dim, latent_dim),
        }

        if not available_ops:
            available_ops = ["Identity", "Attention", "TimeConv", "ResidualMLP"]

        # Filter available ops to ensure we have meaningful operations
        self.available_ops = []
        for op in available_ops:
            if op in self.op_map:
                self.available_ops.append(op)
        
        # Ensure we have at least 2 operations for meaningful search
        if len(self.available_ops) < 2:
            self.available_ops = ["Identity", "Attention", "TimeConv", "ResidualMLP"]
            
        # Remove duplicates while preserving order
        seen = set()
        self.available_ops = [op for op in self.available_ops if not (op in seen or seen.add(op))]
        self.ops = nn.ModuleList([self.op_map[op]() for op in self.available_ops])
        
        # Initialize alphas with balanced weights, NO bias toward Identity
        init_alphas = torch.randn(len(self.ops)) * 0.02
        # Ensure no single operation dominates at initialization
        init_alphas = init_alphas - init_alphas.mean()
        self.alphas = nn.Parameter(init_alphas)

        self._cached_weights = None

    def set_temperature(self, temp: float):
        """Update temperature externally (annealing)"""
        self.temperature = temp
        self._cached_weights = None

    def _get_weights(self):
        if not self.training and self._cached_weights is not None:
            return self._cached_weights

        if self.use_gumbel and self.training:
            weights = F.gumbel_softmax(self.alphas, tau=self.temperature, hard=False)
        else:
            weights = F.softmax(self.alphas / self.temperature, dim=0)

        if not self.training:
            self._cached_weights = weights.detach()

        return weights

    def forward(self, x):
        weights = self._get_weights()
        device = x.device

        # Early exit for single dominant operation (but only in inference)
        if not self.training:
            max_weight = weights.max()
            if max_weight > 0.95:
                max_idx = weights.argmax()
                return self.ops[max_idx](x)

        # DropPath mask per op (only active during training)
        if self.training and self.drop_prob > 0:
            drop_mask = torch.rand(len(self.ops), device=device) < self.drop_prob
        else:
            drop_mask = torch.zeros(len(self.ops), dtype=torch.bool, device=device)

        outputs, total_weight = [], 0.0

        for i, (op, w) in enumerate(zip(self.ops, weights)):
            if drop_mask[i] or w.item() < 5e-4:  # Lowered threshold to include more ops
                continue

            try:
                out = op(x)
                if self.normalize_outputs and self.training:
                    out = F.layer_norm(out, out.shape[-1:])
                outputs.append(w * out)
                total_weight += w.item()
            except Exception as e:
                # Log the error but continue with other operations
                if self.training:
                    print(f"Operation {self.available_ops[i]} failed: {e}")
                continue

        if not outputs:
            # Fallback: use Identity if available, otherwise first operation
            if "Identity" in self.available_ops:
                identity_idx = self.available_ops.index("Identity")
                return self.ops[identity_idx](x)
            else:
                return self.ops[0](x)

        result = sum(outputs)
        if self.normalize_outputs and abs(total_weight - 1.0) > 1e-4:
            result = result / total_weight

        return result

    def get_alphas(self):
        return F.softmax(self.alphas.detach(), dim=0)

    def get_entropy_loss(self):
        probs = F.softmax(self.alphas / self.temperature, dim=0)
        # Encourage diversity - penalize concentration on single operation
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        # Encourage exploration by maximizing entropy (negative entropy loss)
        return -0.05 * entropy  # Negative to encourage high entropy (diversity)


class DARTSCell(nn.Module):
    """Optimized DARTS cell with memory efficiency improvements"""
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
        self.latent_dim = latent_dim
        self.aggregation = aggregation

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=False),
            RMSNorm(latent_dim),
            nn.GELU()
        )

        self.available_ops = (
            ["Identity", "Attention", "TimeConv"]
            if initial_search
            else (
                selected_ops
                if selected_ops
                else ["Identity", "TimeConv", "Attention", "Fourier", "ResidualMLP", "TCN"]
            )
        )

        # Optimized: fewer edges for efficiency
        self.edges = nn.ModuleList([
            MixedOp(
                latent_dim, latent_dim, seq_length,
                self.available_ops, temperature=temperature
            )
            for _ in range(sum(range(num_nodes)))
        ])

        # Simplified residual connections
        self.residual_weights = nn.Parameter(torch.ones(num_nodes) * 0.1)

        self.agg_weights = (
            nn.Parameter(torch.ones(num_nodes - 1))
            if (aggregation == "weighted" and num_nodes > 2)
            else None
        )

    def _aggregate(self, inputs, node_idx):
        if len(inputs) == 1:
            return inputs[0]
        if self.aggregation == "weighted" and self.agg_weights is not None:
            weights = F.softmax(self.agg_weights[: len(inputs)], dim=0)
            return sum(w * inp for w, inp in zip(weights, inputs))
        return torch.mean(torch.stack(inputs, dim=0), dim=0)

    def forward(self, x):
        x_proj = self.input_proj(x)
        nodes = [x_proj]
        edge_idx = 0

        for i in range(1, self.num_nodes):
            inputs = [self.edges[edge_idx + j](nodes[j]) for j in range(i)]
            edge_idx += i
            node_out = self._aggregate(inputs, i)

            # Simplified residual
            residual_weight = torch.sigmoid(self.residual_weights[i])
            node_out = residual_weight * node_out + (1 - residual_weight) * nodes[i - 1]
            nodes.append(node_out)

        # Final output with residual from input
        final_residual_weight = torch.sigmoid(self.residual_weights[0])
        return final_residual_weight * nodes[-1] + (1 - final_residual_weight) * x_proj

    def get_alphas(self):
        return [edge.get_alphas() for edge in self.edges]

    def get_entropy_loss(self):
        return sum(edge.get_entropy_loss() for edge in self.edges)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# Using the previously defined TransformerEncoderWrapper and TransformerDecoderWrapper
# from the mixed encoder/decoder implementation

class FixedEncoder(nn.Module):
    def __init__(
        self,
        rnn: Optional[Union[nn.LSTM, nn.GRU]] = None,
        *,
        rnn_type: Optional[str] = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rnn is not None:
            self.rnn = rnn
            # Better type detection for transformer wrappers
            rnn_class_name = type(rnn).__name__
            if 'TransformerEncoderWrapper' in rnn_class_name or 'TransformerEncoder' in rnn_class_name:
                self.rnn_type = "transformer"
            else:
                self.rnn_type = rnn_class_name.lower()
            
            if hasattr(rnn, "hidden_size"):
                self.latent_dim = rnn.hidden_size
            elif hasattr(rnn, "latent_dim"):
                self.latent_dim = rnn.latent_dim
            else:
                raise ValueError(f"Unsupported rnn module: {type(rnn).__name__}")
            
            self.num_layers = getattr(rnn, "num_layers", 1)

        else:
            if rnn_type is None:
                raise ValueError("Either 'rnn' or 'rnn_type' must be provided")
                
            self.rnn_type = rnn_type.lower()
            self.latent_dim = latent_dim
            self.num_layers = num_layers
            
            if self.rnn_type == "transformer":
                self.rnn = TransformerEncoderWrapper(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
            else:
                rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
                self.rnn = rnn_cls(
                    input_size=input_dim,
                    hidden_size=latent_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )

    def forward(self, x):
        # Check if this is a transformer wrapper by class name
        if (self.rnn_type == "transformer" or 
            'TransformerEncoderWrapper' in self.rnn.__class__.__name__ or
            'TransformerEncoder' in self.rnn.__class__.__name__):
            # TransformerEncoderWrapper returns (output, ctx, state)
            h, ctx, state = self.rnn(x)
            return h, ctx, state
        else:
            # RNN returns (output, hidden_state)
            h, state = self.rnn(x)
            ctx = h[:, -1:, :]  # Final timestep context [batch, 1, latent_dim]
            return h, ctx, state

    def get_alphas(self):
        """Return one-hot encoding for the active RNN type"""
        device = next(self.parameters()).device
        if self.rnn_type == "lstm":
            return torch.tensor([1.0, 0.0, 0.0], device=device)
        elif self.rnn_type == "gru":
            return torch.tensor([0.0, 1.0, 0.0], device=device)
        else:  # transformer
            return torch.tensor([0.0, 0.0, 1.0], device=device)

class FixedDecoder(nn.Module):
    def __init__(
        self,
        rnn: Optional[Union[nn.LSTM, nn.GRU]] = None,
        *,
        rnn_type: Optional[str] = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rnn is not None:
            self.rnn = rnn
            # Better type detection for transformer wrappers
            rnn_class_name = type(rnn).__name__
            if 'TransformerDecoderWrapper' in rnn_class_name or 'TransformerDecoder' in rnn_class_name:
                self.rnn_type = "transformer"
            else:
                self.rnn_type = rnn_class_name.lower()
            
            if hasattr(rnn, "hidden_size"):
                self.latent_dim = rnn.hidden_size
            elif hasattr(rnn, "latent_dim"):
                self.latent_dim = rnn.latent_dim
            else:
                raise ValueError(f"Unsupported rnn module: {type(rnn).__name__}")
            
            self.num_layers = getattr(rnn, "num_layers", 1)

        else:
            if rnn_type is None:
                raise ValueError("Either 'rnn' or 'rnn_type' must be provided")
                
            self.rnn_type = rnn_type.lower()
            self.latent_dim = latent_dim
            self.num_layers = num_layers
            
            if self.rnn_type == "transformer":
                self.rnn = TransformerDecoderWrapper(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
            else:
                rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
                self.rnn = rnn_cls(
                    input_size=input_dim,
                    hidden_size=latent_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )

    def _prepare_hidden_state(self, hidden_state, batch_size, device):
        """Prepare hidden state for the specific RNN type"""
        if hidden_state is None:
            # Initialize zero hidden state
            h_0 = torch.zeros(self.num_layers, batch_size, self.latent_dim, device=device)
            if self.rnn_type == "lstm":
                c_0 = torch.zeros_like(h_0)
                return (h_0, c_0)
            else:
                return h_0
        
        # Handle existing hidden state
        if self.rnn_type == "lstm":
            if not isinstance(hidden_state, tuple):
                # Convert single tensor to LSTM tuple format
                if hidden_state.dim() == 2:
                    h = hidden_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
                else:
                    h = hidden_state.contiguous()
                c = torch.zeros_like(h)
                return (h, c)
            else:
                # Ensure both h and c are properly shaped
                h, c = hidden_state
                if h.dim() == 2:
                    h = h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
                else:
                    h = h.contiguous()
                if c.dim() == 2:
                    c = c.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
                else:
                    c = c.contiguous()
                return (h, c)
        
        elif self.rnn_type == "gru":
            if isinstance(hidden_state, tuple):
                # Extract h from tuple if needed
                hidden_state = hidden_state[0]
            if hidden_state.dim() == 2:
                hidden_state = hidden_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
            else:
                hidden_state = hidden_state.contiguous()
            return hidden_state
        
        else:  # transformer
            # For transformer, we can pass through the hidden state as-is
            return hidden_state

    def forward(self, tgt, memory=None, hidden_state=None):
        batch_size = tgt.size(0)
        device = tgt.device

        if (self.rnn_type == "transformer" or 
            'TransformerDecoderWrapper' in self.rnn.__class__.__name__ or
            'TransformerDecoder' in self.rnn.__class__.__name__):
            # For TransformerDecoderWrapper, we need to pass memory and hidden_state
            # TransformerDecoderWrapper expects: forward(tgt, memory_or_hidden, hidden_state=None)
            # and returns: (output, hidden_state)
            out, new_state = self.rnn(tgt, memory, hidden_state)
        else:
            # Prepare hidden state for RNN
            prepared_state = self._prepare_hidden_state(hidden_state, batch_size, device)
            out, new_state = self.rnn(tgt, prepared_state)
        
        return out, new_state

    def get_alphas(self):
        """Return one-hot encoding for the active RNN type"""
        device = next(self.parameters()).device
        if self.rnn_type == "lstm":
            return torch.tensor([1.0, 0.0, 0.0], device=device)
        elif self.rnn_type == "gru":
            return torch.tensor([0.0, 1.0, 0.0], device=device)
        else:  # transformer
            return torch.tensor([0.0, 0.0, 1.0], device=device)
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class TransformerEncoderWrapper(nn.Module):
    """Transformer encoder wrapper compatible with RNN-style interface"""
    def __init__(self, input_dim, latent_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
        self.pos_encoder = PositionalEncoding(latent_dim)
        self.state_proj = nn.Linear(latent_dim, latent_dim)
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, hidden_state=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        output = self.transformer(x)

        # Extract context from last position
        ctx = output[:, -1:, :]  # Keep as [batch, 1, dim] for consistency
        
        # Create RNN-compatible hidden state
        state = self.state_proj(ctx.squeeze(1))
        h_state = state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = h_state.clone()
        
        return output, ctx, (h_state, c_state)

class TransformerDecoderWrapper(nn.Module):
    """Transformer decoder wrapper compatible with RNN-style interface"""
    def __init__(self, input_dim, latent_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
        self.pos_decoder = PositionalEncoding(latent_dim)
        self.state_proj = nn.Linear(latent_dim, latent_dim)
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory_or_hidden, hidden_state=None):
        # Handle both memory (for transformer) and hidden state (for RNN compatibility)
        if isinstance(memory_or_hidden, tuple):
            # This is a hidden state from RNN, we need memory for transformer
            # Use the hidden state as memory (this is a compatibility hack)
            h_state, c_state = memory_or_hidden
            memory = h_state.transpose(0, 1)  # Convert to [batch, seq, dim]
        else:
            # This is memory from encoder
            memory = memory_or_hidden
            
        tgt = self.input_proj(tgt)
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory)

        # Create RNN-compatible state output
        last_output = output[:, -1, :]
        state = self.state_proj(last_output)
        h_state = state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = h_state.clone()
        
        return output, (h_state, c_state)

class MixedEncoder(nn.Module):
    """Mixed encoder supporting LSTM, GRU, and Transformer"""
    def __init__(self, input_dim, latent_dim, seq_len, dropout=0.1, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = 2

        self.encoders = nn.ModuleList([
            nn.LSTM(
                input_dim, latent_dim, num_layers=self.num_layers,
                batch_first=True, dropout=dropout,
            ),
            nn.GRU(
                input_dim, latent_dim, num_layers=self.num_layers,
                batch_first=True, dropout=dropout,
            ),
            TransformerEncoderWrapper(
                input_dim=input_dim,
                latent_dim=latent_dim,
                num_layers=self.num_layers,
                dropout=dropout
            )
        ])
        self.encoder_names = ["lstm", "gru", "transformer"]
        self.alphas = nn.Parameter(torch.randn(len(self.encoders)) * 0.01)

    def forward(self, x):
        weights = F.softmax(self.alphas / self.temperature, dim=0)

        # Fast path for inference when one encoder dominates
        if not self.training and weights.max() > 0.9:
            max_idx = weights.argmax()
            encoder = self.encoders[max_idx]
            
            if self.encoder_names[max_idx] in ["lstm", "gru"]:
                # RNN encoders return (output, hidden_state)
                output, hidden_state = encoder(x)
                # Create context from last timestep
                ctx = output[:, -1:, :]
                return output, ctx, hidden_state
            else:
                # Transformer encoder returns (output, ctx, hidden_state)
                return encoder(x)

        outputs, contexts, states = [], [], []
        
        for i, encoder in enumerate(self.encoders):
            if weights[i].item() > 1e-3:
                if self.encoder_names[i] in ["lstm", "gru"]:
                    # RNN encoders
                    output, hidden_state = encoder(x)
                    ctx = output[:, -1:, :]  # Last timestep as context
                    outputs.append(output)
                    contexts.append(ctx)
                    states.append(hidden_state)
                else:
                    # Transformer encoder
                    output, ctx, hidden_state = encoder(x)
                    outputs.append(output)
                    contexts.append(ctx)
                    states.append(hidden_state)

        if not outputs:
            # Fallback to first encoder
            encoder = self.encoders[0]
            if self.encoder_names[0] in ["lstm", "gru"]:
                output, hidden_state = encoder(x)
                ctx = output[:, -1:, :]
                return output, ctx, hidden_state
            else:
                return encoder(x)

        # Weighted combination
        active_weights = weights[:len(outputs)]
        total_weight = active_weights.sum()
        norm_weights = active_weights / total_weight
        
        weighted_output = sum(norm_weights[i] * outputs[i] for i in range(len(outputs)))
        weighted_context = sum(norm_weights[i] * contexts[i] for i in range(len(contexts)))
        
        # Use state from the encoder with highest weight
        max_weight_idx = torch.argmax(active_weights).item()
        final_state = states[max_weight_idx]

        return weighted_output, weighted_context, final_state

    def get_alphas(self):
        return F.softmax(self.alphas, dim=0)

    def get_entropy_loss(self):
        probs = self.get_alphas()
        log_probs = torch.log(torch.clamp_min(probs, 1e-8))
        entropy = -(probs * log_probs).sum() * 0.01
        return torch.clamp(entropy, min=0.0, max=1.0)

class MixedDecoder(nn.Module):
    """Mixed decoder supporting LSTM, GRU, and Transformer"""
    def __init__(self, input_dim, latent_dim, seq_len, dropout=0.1, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = 2

        self.decoders = nn.ModuleList([
            nn.LSTM(
                input_dim, latent_dim, num_layers=self.num_layers,
                batch_first=True, dropout=dropout,
            ),
            nn.GRU(
                input_dim, latent_dim, num_layers=self.num_layers,
                batch_first=True, dropout=dropout,
            ),
            TransformerDecoderWrapper(
                input_dim=input_dim,
                latent_dim=latent_dim,
                num_layers=self.num_layers,
                dropout=dropout
            )
        ])
        self.decoder_names = ["lstm", "gru", "transformer"]
        self.alphas = nn.Parameter(torch.randn(len(self.decoders)) * 0.01)

    def _prepare_hidden_state(self, hidden_state, batch_size, device):
        """Prepare hidden state for RNN decoders"""
        if hidden_state is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.latent_dim, device=device)
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0), h_0

        if isinstance(hidden_state, tuple):
            h, c = hidden_state
            h = h if h.dim() == 3 else h.unsqueeze(0).expand(self.num_layers, -1, -1)
            c = c if c.dim() == 3 else c.unsqueeze(0).expand(self.num_layers, -1, -1)
            return (h.contiguous(), c.contiguous()), h
        else:
            h = hidden_state if hidden_state.dim() == 3 else hidden_state.unsqueeze(0).expand(self.num_layers, -1, -1)
            c = torch.zeros_like(h)
            return (h.contiguous(), c.contiguous()), h

    def forward(self, tgt, memory, hidden_state=None):
        batch_size, _, _ = tgt.size()
        device = tgt.device
        weights = F.softmax(self.alphas / self.temperature, dim=0)

        # Fast path for inference
        if not self.training and weights.max() > 0.9:
            max_idx = weights.argmax()
            decoder = self.decoders[max_idx]
            
            if self.decoder_names[max_idx] == "lstm":
                lstm_state, _ = self._prepare_hidden_state(hidden_state, batch_size, device)
                return decoder(tgt, lstm_state)
            elif self.decoder_names[max_idx] == "gru":
                _, gru_state = self._prepare_hidden_state(hidden_state, batch_size, device)
                return decoder(tgt, gru_state)
            else:  # transformer
                return decoder(tgt, memory, hidden_state)

        # Prepare states for RNN decoders
        lstm_state, gru_state = self._prepare_hidden_state(hidden_state, batch_size, device)
        outputs, new_states, active_weights = [], [], []

        for i, decoder in enumerate(self.decoders):
            if weights[i].item() > 1e-3:
                if self.decoder_names[i] == "lstm":
                    output, state = decoder(tgt, lstm_state)
                elif self.decoder_names[i] == "gru":
                    output, state = decoder(tgt, gru_state)
                else:  # transformer
                    output, state = decoder(tgt, memory, hidden_state)

                outputs.append(output)
                new_states.append(state)
                active_weights.append(weights[i])

        if not outputs:
            # Fallback to LSTM
            return self.decoders[0](tgt, lstm_state)

        # Weighted combination
        total_weight = sum(active_weights)
        norm_weights = [w / total_weight for w in active_weights]
        weighted_output = sum(w * out for w, out in zip(norm_weights, outputs))

        # Use state from decoder with highest weight
        max_idx = active_weights.index(max(active_weights))
        new_state = new_states[max_idx]

        return weighted_output, new_state

    def get_alphas(self):
        return F.softmax(self.alphas, dim=0)

    def get_entropy_loss(self):
        probs = self.get_alphas()
        log_probs = torch.log(torch.clamp_min(probs, 1e-8))
        entropy = -(probs * log_probs).sum() * 0.01
        return torch.clamp(entropy, min=0.0, max=1.0)
    
class TimeSeriesDARTS(nn.Module):
    """Enhanced TimeSeriesDARTS with improved code organization and all features"""
    
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
    ):
        super().__init__()
        self._store_config(locals())
        self._init_components()
        self._init_weights()
        self._setup_compilation()

    def _store_config(self, config: Dict[str, Any]) -> None:
        """Store configuration parameters"""
        config.pop('self')
        config.pop('__class__', None)
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
            RMSNorm(self.hidden_dim),
            nn.GELU()
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
            temp = self.temperature * (0.8 ** i)
            
            # DARTS cell
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
                    RMSNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout * 0.5),
                )
            )
            
            # Layer scaling
            self.layer_scales.append(nn.Parameter(torch.ones(1) * 1e-2))

        self.cell_weights = nn.Parameter(torch.ones(self.num_cells))

    def _init_forecasting_components(self) -> None:
        """Initialize encoder, decoder, and fusion components"""
        self.forecast_encoder = MixedEncoder(
            self.hidden_dim, self.latent_dim, seq_len=self.seq_length,
            dropout=self.dropout, temperature=self.temperature
        )
        self.forecast_decoder = MixedDecoder(
            self.input_dim, self.latent_dim, seq_len=self.seq_length,
            dropout=self.dropout, temperature=self.temperature
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
        self.forecast_norm = RMSNorm(self.latent_dim)

    def _init_weights(self) -> None:
        """Initialize model weights with proper strategies"""
        weight_init_strategies = {
            nn.Linear: self._init_linear,
            (nn.LSTM, nn.GRU): self._init_rnn,
            (RMSNorm, nn.LayerNorm): self._init_norm,
        }
        
        for module in self.modules():
            for types, init_fn in weight_init_strategies.items():
                if isinstance(module, types):
                    init_fn(module)
                    break

    def _init_linear(self, module: nn.Linear) -> None:
        """Initialize linear layer weights"""
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def _init_rnn(self, module: nn.Module) -> None:
        """Initialize RNN weights"""
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "lstm" in type(module).__name__.lower():
                    # Set forget gate bias to 1
                    param.data[param.size(0) // 4 : param.size(0) // 2].fill_(1.0)

    def _init_norm(self, module: nn.Module) -> None:
        """Initialize normalization layer weights"""
        if hasattr(module, "weight"):
            nn.init.ones_(module.weight)
        if hasattr(module, "bias"):
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

    @functools.lru_cache(maxsize=8)
    def _interpolate_pos_encoding(self, seq_len: int) -> torch.Tensor:
        """Cache and interpolate positional encodings"""
        return F.interpolate(
            self.pos_encoding.transpose(1, 2),
            size=seq_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _forward_cells(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DARTS cells with feature fusion"""
        current_input = x
        cell_features = []

        for i, (cell, proj, scale) in enumerate(zip(self.cells, self.cell_proj, self.layer_scales)):
            # Cell forward with optional checkpointing
            cell_out = (
                checkpoint(cell, current_input, use_reentrant=False)
                if (self.training and self.use_gradient_checkpointing)
                else cell(current_input)
            )
            
            # Project and scale
            projected = proj(cell_out) * scale
            cell_features.append(projected)
            
            # Residual connection for deeper cells
            current_input = cell_out + current_input * 0.1 if i > 0 else cell_out

        # Weighted combination of cell features
        if len(cell_features) > 1:
            weights = F.softmax(self.cell_weights[:len(cell_features)], dim=0)
            combined = sum(w * f for w, f in zip(weights, cell_features))
        else:
            combined = cell_features[0]

        return combined

    def forward(
        self, 
        x_seq: torch.Tensor, 
        x_future: Optional[torch.Tensor] = None, 
        teacher_forcing_ratio: float = 0.5
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
        teacher_forcing_ratio: float
    ) -> torch.Tensor:
        """Core forward implementation"""
        B, L, _ = x_seq.shape
        
        # Input embedding with optional positional encoding
        x_emb = self.input_embedding(x_seq)
        # pos_enc = self._interpolate_pos_encoding(L)  # Commented for efficiency
        # x_emb = x_emb + pos_enc

        # DARTS cell processing
        final_features = self._forward_cells(x_seq)

        # Feature fusion
        fuse_input = torch.cat([final_features, x_emb], dim=-1)
        alpha = self.gate_fuse(fuse_input)
        combined = alpha * final_features + (1 - alpha) * x_emb

        # Encoding
        h_enc, context, encoder_state = self.forecast_encoder(combined)

        # Decoding with teacher forcing
        return self._decode_forecasts(x_seq, x_future, context, encoder_state, teacher_forcing_ratio)

    def _decode_forecasts(
        self,
        x_seq: torch.Tensor,
        x_future: Optional[torch.Tensor],
        context: torch.Tensor,
        encoder_state: torch.Tensor,
        teacher_forcing_ratio: float
    ) -> torch.Tensor:
        """Decode forecasts with teacher forcing"""
        forecasts = []
        decoder_input = x_seq[:, -1:, :]
        decoder_hidden = encoder_state

        for t in range(self.forecast_horizon):
            out, decoder_hidden = self.forecast_decoder(
                decoder_input, context, decoder_hidden
            )
            out = self.forecast_norm(out + context)
            out = self.mlp(out) + out
            prediction = self.output_layer(out)
            forecasts.append(prediction.squeeze(1))

            # Teacher forcing decision
            if (self.training and x_future is not None and 
                torch.rand(1).item() < teacher_forcing_ratio):
                decoder_input = x_future[:, t:t+1]
            else:
                decoder_input = prediction

        return torch.stack(forecasts, dim=1)

    def calculate_loss(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        return_components: bool = False
    ) -> torch.Tensor:
        """Calculate total loss with components"""
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
        
        total_loss = main_loss + entropy_loss + alpha_l2

        if return_components:
            return {
                "total_loss": total_loss,
                "main_loss": main_loss,
                "entropy_loss": entropy_loss,
                "alpha_l2": alpha_l2,
            }
        return total_loss

    def _compute_entropy_loss(self) -> torch.Tensor:
        """Compute entropy regularization loss"""
        entropy_loss = sum(cell.get_entropy_loss() for cell in self.cells) * 1e-3
        entropy_loss += self.forecast_encoder.get_entropy_loss()
        entropy_loss += self.forecast_decoder.get_entropy_loss()
        return entropy_loss

    def _compute_alpha_l2(self) -> torch.Tensor:
        """Compute L2 regularization on architecture parameters"""
        alpha_l2 = 0.0
        for cell in self.cells:
            for edge in cell.edges:
                alpha_l2 += (edge.alphas ** 2).sum()
        alpha_l2 += (self.forecast_encoder.alphas ** 2).sum()
        alpha_l2 += (self.forecast_decoder.alphas ** 2).sum()
        return alpha_l2 * 1e-4

    # Architecture Analysis Methods
    def get_alphas(self) -> List[torch.Tensor]:
        """Get all architecture parameters"""
        return [cell.get_alphas() for cell in self.cells] + [
            self.forecast_encoder.get_alphas(),
            self.forecast_decoder.get_alphas(),
        ]

    def get_alpha_dict(self) -> Dict[str, torch.Tensor]:
        """Get architecture parameters as named dictionary"""
        alpha_map = {}
        for idx, cell in enumerate(self.cells):
            edge_idx = 0
            for i in range(cell.num_nodes):
                for j in range(i + 1, cell.num_nodes):
                    if edge_idx < len(cell.edges):
                        alpha_map[f"cell{idx}_edge_{i}->{j}"] = cell.edges[edge_idx].get_alphas()
                        edge_idx += 1
        
        alpha_map["encoder_type"] = self.forecast_encoder.get_alphas()
        alpha_map["decoder_type"] = self.forecast_decoder.get_alphas()
        return alpha_map

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
        for component, names_attr in [("encoder", "encoder_names"), ("decoder", "decoder_names")]:
            comp_obj = getattr(self, f"forecast_{component}")
            names = getattr(comp_obj, names_attr)
            alphas = comp_obj.get_alphas()
            weights[component] = {
                name: weight.item() for name, weight in zip(names, alphas)
            }
        
        return weights

    def derive_discrete_architecture(
        self, 
        threshold: float = 0.3, 
        top_k_fallback: int = 2
    ) -> Dict[str, Any]:
        """Derive discrete architecture from continuous search"""
        discrete_arch = {}

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
                        "confidence": max_weight
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
                        "confidence": topk_weights[0]
                    }

            discrete_arch[f"cell_{i}"] = cell_arch

        # Encoder/Decoder selection
        for comp_name, comp_obj in [("encoder", self.forecast_encoder), ("decoder", self.forecast_decoder)]:
            weights = comp_obj.get_alphas()
            idx = weights.argmax().item()
            names = getattr(comp_obj, f"{comp_name}_names")
            discrete_arch[comp_name] = {
                "type": names[idx],
                "weight": weights.max().item()
            }

        return discrete_arch

    # Optimization and Health Methods
    def set_temperature(self, temp: float) -> None:
        """Update temperature for all mixed operations"""
        for cell in self.cells:
            for edge in cell.edges:
                edge.set_temperature(temp)
        self.forecast_encoder.temperature = temp
        self.forecast_decoder.temperature = temp

    def prune_weak_operations(self, threshold: float = 0.1) -> None:
        """Remove operations with consistently low weights"""
        with torch.no_grad():
            for cell in self.cells:
                for edge in cell.edges:
                    alphas = edge.get_alphas()
                    
                    # Handle Identity dominance
                    if "Identity" in edge.available_ops:
                        identity_idx = edge.available_ops.index("Identity")
                        if alphas[identity_idx] > 0.7:
                            # Boost non-identity operations
                            for i, op in enumerate(edge.available_ops):
                                if op != "Identity":
                                    edge.alphas.data[i] += 0.5
                            continue
                    
                    # Prune weak operations
                    weak_ops = alphas < threshold
                    if weak_ops.any() and not weak_ops.all():
                        edge.alphas.data[weak_ops] = -5.0

    def encourage_diversity(self) -> None:
        """Encourage diversity by reducing Identity dominance"""
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

    def validate_architecture_health(self) -> Dict[str, Any]:
        """Comprehensive architecture health validation"""
        issues = []
        total_identity_dominance = 0
        total_edges = 0
        
        for i, cell in enumerate(self.cells):
            for j, edge in enumerate(cell.edges):
                weights = edge.get_alphas()
                entropy = -(weights * torch.log(weights + 1e-8)).sum().item()
                max_weight = weights.max().item()
                
                # Check for issues
                if entropy < 0.5:
                    issues.append(f"Cell {i}, Edge {j}: Low diversity (entropy={entropy:.3f})")
                
                if max_weight > 0.9:
                    dominant_op = edge.available_ops[weights.argmax().item()]
                    issues.append(f"Cell {i}, Edge {j}: {dominant_op} dominates ({max_weight:.3f})")
                
                # Track Identity dominance
                if "Identity" in edge.available_ops:
                    identity_idx = edge.available_ops.index("Identity")
                    identity_weight = weights[identity_idx].item()
                    total_identity_dominance += identity_weight
                    
                    if identity_weight > 0.7:
                        issues.append(f"Cell {i}, Edge {j}: Identity dominates ({identity_weight:.3f})")
                
                total_edges += 1
        
        avg_identity_dominance = total_identity_dominance / max(total_edges, 1)
        health_score = max(0, 1.0 - len(issues) / max(total_edges * 2, 1) - avg_identity_dominance)
        
        return {
            "issues": issues,
            "avg_identity_dominance": avg_identity_dominance,
            "total_edges": total_edges,
            "health_score": health_score
        }

    def apply_architecture_fixes(self) -> None:
        """Apply automatic fixes for architecture issues"""
        health = self.validate_architecture_health()
        
        if health["avg_identity_dominance"] > 0.5:
            print("Applying Identity dominance fix...")
            self.encourage_diversity()
        
        # Add noise for low entropy
        low_entropy_count = sum(1 for issue in health["issues"] if "Low diversity" in issue)
        if low_entropy_count > len(health["issues"]) * 0.5:
            print("Adding noise to increase exploration...")
            with torch.no_grad():
                for cell in self.cells:
                    for edge in cell.edges:
                        noise = torch.randn_like(edge.alphas) * 0.1
                        edge.alphas.data += noise

    def get_diversity_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed diversity metrics"""
        metrics = {}
        
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
                    "num_active_ops": (weights > 0.1).sum().item()
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
            "total_mb": (total_params + total_buffers) / (1024 * 1024)
        }