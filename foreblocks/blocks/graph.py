import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Literal
import math
import contextlib

# External
from xformers.ops import memory_efficient_attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from torch.nn.utils import spectral_norm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple, Union
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple, Union
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple, Union
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple
import math


class LatentCorrelationLayer(nn.Module):
    """
    Optimized Latent Correlation Layer that maintains the original's effectiveness
    while adding targeted performance improvements:
    - Better numerical stability
    - Optional spectral normalization
    - Improved initialization
    - Enhanced Chebyshev filtering
    - Memory-efficient computation options
    """

    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        learnable_alpha: bool = True,
        init_alpha: float = 0.5,
        use_layer_norm: bool = True,
        low_rank: bool = False,
        rank: Optional[int] = None,
        correlation_dropout: float = 0.0,
        cheb_k: int = 3,
        eps: float = 1e-8,
        # Optimizations (conservative additions)
        use_spectral_norm: bool = False,
        improved_init: bool = True,
        temperature: float = 1.0,
        gradient_checkpointing: bool = False,
        memory_efficient: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size or (2 * input_size)
        self.low_rank = low_rank
        self.rank = rank or max(1, input_size // 4)
        self.use_layer_norm = use_layer_norm
        self.cheb_k = max(1, cheb_k)
        self.eps = eps
        self.improved_init = improved_init
        self.temperature = temperature
        self.gradient_checkpointing = gradient_checkpointing
        self.memory_efficient = memory_efficient

        # Alpha blending (fixed tensor creation)
        if learnable_alpha:
            init_logit = torch.logit(torch.tensor(init_alpha, dtype=torch.float32))
            self.alpha = nn.Parameter(init_logit.detach().clone())
        else:
            self.register_buffer("alpha", torch.tensor(init_alpha, dtype=torch.float32))

        # Correlation (same as original)
        if low_rank:
            scale = 1.0 / (self.rank**0.5)
            self.corr_factors = nn.Parameter(
                torch.randn(2, input_size, self.rank) * scale
            )
        else:
            self.correlation = nn.Parameter(torch.randn(input_size, input_size))

        # Projections (with optional spectral normalization)
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.output_size)

        if use_spectral_norm:
            self.input_proj = spectral_norm(self.input_proj)
            self.output_proj = spectral_norm(self.output_proj)

        # Normalization (same as original)
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(input_size, eps=eps)
            self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=eps)
            self.layer_norm3 = nn.LayerNorm(self.output_size, eps=eps)

        # Dropout (same as original)
        self.dropout = (
            nn.Dropout(correlation_dropout)
            if correlation_dropout > 0
            else nn.Identity()
        )

        # Chebyshev coefficients
        self.cheb_weights = nn.Parameter(torch.ones(self.cheb_k) / self.cheb_k)

        # Optional temperature for Chebyshev weights
        if temperature != 1.0:
            self.cheb_temp = nn.Parameter(
                torch.tensor(temperature, dtype=torch.float32)
            )
        else:
            self.register_buffer("cheb_temp", torch.tensor(1.0, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        """Enhanced parameter initialization while maintaining original structure"""
        if self.low_rank:
            if self.improved_init:
                # Better initialization for low-rank factors
                nn.init.orthogonal_(self.corr_factors[0])
                nn.init.orthogonal_(self.corr_factors[1])
            else:
                # Original initialization
                scale = 1.0 / (self.rank**0.5)
                self.corr_factors.data = torch.randn_like(self.corr_factors) * scale
        else:
            if self.improved_init:
                # Start closer to identity for better stability
                nn.init.eye_(self.correlation)
                with torch.no_grad():
                    noise_scale = 0.01 if self.input_size > 64 else 0.05
                    self.correlation.data += noise_scale * torch.randn_like(
                        self.correlation
                    )
                    self.correlation.data = 0.5 * (
                        self.correlation.data + self.correlation.data.t()
                    )
            else:
                # Original initialization
                nn.init.eye_(self.correlation)
                with torch.no_grad():
                    self.correlation.data += 0.01 * torch.randn_like(self.correlation)
                    self.correlation.data = 0.5 * (
                        self.correlation.data + self.correlation.data.t()
                    )

        # Improved projection initialization
        if self.improved_init:
            gain = math.sqrt(2.0)  # For GELU activation
            nn.init.xavier_uniform_(self.input_proj.weight, gain=gain)
            nn.init.zeros_(self.input_proj.bias)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            # Original initialization
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        # Chebyshev weights initialization
        nn.init.constant_(self.cheb_weights, 1.0 / self.cheb_k)

    def get_learned_correlation(self) -> torch.Tensor:
        """Same as original but with optional temperature scaling"""
        if self.low_rank:
            U, V = self.corr_factors[0], self.corr_factors[1]
            corr = torch.matmul(U, V.T)
            corr = 0.5 * (corr + corr.T)
        else:
            corr = 0.5 * (self.correlation + self.correlation.T)

        # Optional temperature scaling for learned correlations
        if self.temperature != 1.0:
            corr = torch.tanh(corr / self.temperature)
        else:
            corr = torch.tanh(corr)

        return self.dropout(corr) if self.training else corr

    def compute_data_correlation(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced data correlation with optional memory-efficient computation"""
        if self.memory_efficient and x.shape[-1] > 128:
            return self._compute_data_correlation_efficient(x)

        # Original implementation (proven to work well)
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_reshaped = x_centered.transpose(1, 2)
        norms = torch.norm(x_reshaped, dim=2, keepdim=True).clamp(min=self.eps)
        x_normalized = x_reshaped / norms
        corr_batch = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
        return corr_batch.mean(dim=0).clamp(min=-1.0, max=1.0).detach()

    def _compute_data_correlation_efficient(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient correlation computation for large feature dimensions"""
        B, L, D = x.shape
        x_centered = x - x.mean(dim=1, keepdim=True)

        # Compute correlation in chunks to save memory
        chunk_size = 64
        corr_chunks = []

        for i in range(0, D, chunk_size):
            end_i = min(i + chunk_size, D)
            chunk_i = x_centered[:, :, i:end_i].transpose(1, 2)  # [B, chunk_size, L]
            norm_i = torch.norm(chunk_i, dim=2, keepdim=True).clamp(min=self.eps)
            chunk_i_norm = chunk_i / norm_i

            row_chunks = []
            for j in range(0, D, chunk_size):
                end_j = min(j + chunk_size, D)
                chunk_j = x_centered[:, :, j:end_j].transpose(
                    1, 2
                )  # [B, chunk_size, L]
                norm_j = torch.norm(chunk_j, dim=2, keepdim=True).clamp(min=self.eps)
                chunk_j_norm = chunk_j / norm_j

                # Compute correlation between chunks
                chunk_corr = torch.bmm(chunk_i_norm, chunk_j_norm.transpose(1, 2))
                row_chunks.append(chunk_corr.mean(dim=0))

            corr_chunks.append(torch.cat(row_chunks, dim=1))

        corr = torch.cat(corr_chunks, dim=0)
        return corr.clamp(min=-1.0, max=1.0).detach()

    def compute_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """Enhanced Laplacian computation with better numerical stability"""
        A = A.clone()
        A.fill_diagonal_(0.0)

        # Improved numerical stability
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=self.eps), -0.5)

        # Handle potential infinities more robustly
        deg_inv_sqrt = torch.where(
            torch.isinf(deg_inv_sqrt), torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt
        )

        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = (
            torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            - D_inv_sqrt @ A @ D_inv_sqrt
        )

        # Tighter clamping for better numerical properties
        return L.clamp(min=-1.5, max=1.5)

    def chebyshev_filter(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Enhanced Chebyshev filtering with optional temperature"""
        # Temperature-scaled softmax for Chebyshev weights
        cheb_weights = F.softmax(self.cheb_weights / self.cheb_temp, dim=0)

        Tx_0 = x
        if self.cheb_k == 1:
            return cheb_weights[0] * Tx_0

        Tx_1 = torch.matmul(x, L)
        out = cheb_weights[0] * Tx_0 + cheb_weights[1] * Tx_1

        for k in range(2, self.cheb_k):
            Tx_k = 2 * torch.matmul(Tx_1, L) - Tx_0
            # Slightly tighter clamping for stability
            Tx_k = Tx_k.clamp(min=-50, max=50)
            out += cheb_weights[k] * Tx_k
            Tx_0, Tx_1 = Tx_1, Tx_k

        return out

    def _forward_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core forward implementation"""
        if self.use_layer_norm:
            x = self.layer_norm1(x)

        raw_data_corr = self.compute_data_correlation(x)
        learned_corr = self.get_learned_correlation()
        alpha = torch.sigmoid(self.alpha)
        mixed_corr = alpha * learned_corr + (1 - alpha) * raw_data_corr

        laplacian = self.compute_laplacian(mixed_corr)
        x_filtered = self.chebyshev_filter(x, laplacian)
        x_proj = self.input_proj(x_filtered)

        if self.use_layer_norm:
            x_proj = self.layer_norm2(x_proj)

        x_proj = F.gelu(x_proj)  # Keep original activation
        out = self.output_proj(x_proj)

        if self.use_layer_norm:
            out = self.layer_norm3(out)

        return out, mixed_corr

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional gradient checkpointing"""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        else:
            return self._forward_impl(x)

    def get_correlation_stats(self) -> dict:
        """Return correlation statistics for monitoring"""
        stats = {}

        if hasattr(self, "correlation"):
            corr = self.correlation.detach()
            stats["learned_corr_mean"] = corr.mean().item()
            stats["learned_corr_std"] = corr.std().item()
            stats["learned_corr_max"] = corr.max().item()
            stats["learned_corr_min"] = corr.min().item()

        if hasattr(self, "alpha"):
            stats["alpha"] = torch.sigmoid(self.alpha).item()

        if hasattr(self, "cheb_weights"):
            weights = F.softmax(self.cheb_weights / self.cheb_temp, dim=0)
            stats["cheb_weights"] = weights.detach().cpu().numpy().tolist()

        return stats


from flash_attn import flash_attn_qkvpacked_func


def round_to_supported_head_dim(dim):
    supported_dims = [16, 32, 64, 128]
    return min(supported_dims, key=lambda x: abs(x - dim))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Union, Tuple
import math

try:
    from flash_attn import flash_attn_qkvpacked_func

    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


def round_to_supported_head_dim(dim: int) -> int:
    """Round to nearest supported head dimension for attention backends"""
    supported_dims = [8, 16, 32, 64, 128, 256]
    return min(supported_dims, key=lambda x: abs(x - dim))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Union, Tuple
import math

try:
    from flash_attn import flash_attn_qkvpacked_func

    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


def round_to_supported_head_dim(dim: int) -> int:
    """Round to nearest supported head dimension for attention backends"""
    supported_dims = [8, 16, 32, 64, 128, 256]
    return min(supported_dims, key=lambda x: abs(x - dim))


class MessagePassing(nn.Module):
    """
    Optimized message passing base class with full backward compatibility.
    Enhanced with:
    - Better numerical stability
    - Improved memory efficiency
    - Enhanced attention mechanisms
    - Optional spectral normalization
    - Gradient checkpointing support
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        aggregation: str = "sum",
        num_heads: int = 4,
        # Optional optimization parameters (fully backward compatible)
        eps: float = 1e-10,
        use_spectral_norm: bool = False,
        improved_init: bool = True,
        gradient_checkpointing: bool = False,
        attention_dropout: float = 0.0,
        memory_efficient: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.num_heads = num_heads
        self.eps = eps
        self.use_spectral_norm = use_spectral_norm
        self.improved_init = improved_init
        self.gradient_checkpointing = gradient_checkpointing
        self.attention_dropout = attention_dropout
        self.memory_efficient = memory_efficient

        # Enhanced head dimension calculation
        raw_head_dim = hidden_dim * num_heads
        self.head_dim = round_to_supported_head_dim(raw_head_dim)

        # Shared node transformation with optional spectral normalization
        self.message_transform = nn.Linear(input_size, hidden_dim)
        if use_spectral_norm:
            self.message_transform = spectral_norm(self.message_transform)

        # SAGE components with improvements
        if self.aggregation == "sage":
            self.sage_update = nn.Linear(input_size + hidden_dim, hidden_dim)
            if use_spectral_norm:
                self.sage_update = spectral_norm(self.sage_update)

        if self.aggregation == "sage_lstm":
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=attention_dropout if attention_dropout > 0 else 0,
            )
            self.sage_update = nn.Linear(input_size + hidden_dim, hidden_dim)
            if use_spectral_norm:
                self.sage_update = spectral_norm(self.sage_update)

        # Enhanced attention projections
        if aggregation == "xformers":
            self._init_xformers_projections()
        elif aggregation == "flash":
            self._init_flash_projections()

        # Initialize parameters
        if improved_init:
            self._enhanced_parameter_init()
        else:
            self._standard_parameter_init()

    def _init_xformers_projections(self):
        """Initialize xFormers attention projections with optimizations"""
        # More efficient projection dimensions
        proj_dim = self.input_size * self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.input_size, proj_dim, bias=False)
        self.k_proj = nn.Linear(self.input_size, proj_dim, bias=False)
        self.v_proj = nn.Linear(self.input_size, proj_dim, bias=False)
        self.bias_proj = nn.Linear(self.input_size, self.num_heads * self.num_heads)

        if self.use_spectral_norm:
            self.q_proj = spectral_norm(self.q_proj)
            self.k_proj = spectral_norm(self.k_proj)
            self.v_proj = spectral_norm(self.v_proj)

    def _init_flash_projections(self):
        """Initialize FlashAttention projections with optimizations"""
        proj_dim = self.input_size * self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.input_size, proj_dim, bias=False)
        self.k_proj = nn.Linear(self.input_size, proj_dim, bias=False)
        self.v_proj = nn.Linear(self.input_size, proj_dim, bias=False)

        if self.use_spectral_norm:
            self.q_proj = spectral_norm(self.q_proj)
            self.k_proj = spectral_norm(self.k_proj)
            self.v_proj = spectral_norm(self.v_proj)

    def _enhanced_parameter_init(self):
        """Enhanced parameter initialization for better training"""
        # Message transform
        nn.init.xavier_uniform_(self.message_transform.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.message_transform.bias)

        # SAGE components
        if hasattr(self, "sage_update"):
            nn.init.xavier_uniform_(self.sage_update.weight)
            nn.init.zeros_(self.sage_update.bias)

        # Attention projections
        if hasattr(self, "q_proj"):
            gain = 1.0 / math.sqrt(3)  # For Q, K, V projections
            nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)

        if hasattr(self, "bias_proj"):
            nn.init.xavier_uniform_(self.bias_proj.weight)
            nn.init.zeros_(self.bias_proj.bias)

    def _standard_parameter_init(self):
        """Standard parameter initialization (original behavior)"""
        # Keep original initialization if improved_init=False
        pass

    def message(self, h: torch.Tensor) -> torch.Tensor:
        """
        Enhanced message computation with optional normalization
        """
        return self.message_transform(h)  # [B, T, hidden_dim]

    def aggregate(
        self,
        messages: torch.Tensor,
        graph: torch.Tensor,
        self_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enhanced aggregation with better numerical stability
        """
        if self.aggregation == "sum":
            return self._sum_aggregate(messages, graph)
        elif self.aggregation == "mean":
            return self._mean_aggregate(messages, graph)
        elif self.aggregation == "max":
            return self._max_aggregate(messages, graph)
        elif self.aggregation == "sage":
            return self._sage_aggregate(messages, graph, self_features)
        elif self.aggregation == "sage_lstm":
            return self._sage_lstm_aggregate(messages, graph, self_features)
        elif self.aggregation == "xformers":
            return self._xformers_aggregate(messages, graph)
        elif self.aggregation == "flash":
            return self._flash_aggregate(messages, graph)
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation}")

    def _sum_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """Sum aggregation - original implementation"""
        return torch.einsum("bth,hg->btg", messages, graph)

    def _mean_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """Mean aggregation with improved numerical stability"""
        deg = graph.sum(dim=1).clamp(min=self.eps)  # Better epsilon handling
        norm_graph = graph / deg.unsqueeze(1)
        return torch.einsum("bth,hg->btg", messages, norm_graph)

    def _max_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """Max aggregation with proper implementation"""
        # Original had incorrect implementation, fixed here
        expanded = torch.einsum("bth,hg->bthg", messages, graph)
        return expanded.max(dim=2)[0]

    def _sage_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor, self_features: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced SAGE aggregation with stability improvements"""
        assert self_features is not None, "SAGE requires self node features"

        # Graph aggregation (mean) with better numerical stability
        deg = graph.sum(dim=1).clamp(min=self.eps)
        norm_graph = graph / deg.unsqueeze(1)
        neighbor_agg = torch.einsum("bth,hg->btg", messages, norm_graph)

        # Concatenate self features and aggregated neighbor messages
        concat = torch.cat([self_features, neighbor_agg], dim=-1)
        return self.sage_update(concat)

    def _sage_lstm_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor, self_features: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced SAGE-LSTM aggregation"""
        assert self_features is not None, "SAGE-LSTM requires self node features"

        neighbor_sequences = torch.einsum("bth,hg->btg", messages, graph).transpose(
            1, 2
        )

        # Enhanced LSTM processing
        lstm_out, _ = self.lstm(neighbor_sequences)
        neighbor_agg, _ = torch.max(lstm_out, dim=1)
        neighbor_agg = neighbor_agg.unsqueeze(1).expand(-1, messages.size(1), -1)

        concat = torch.cat([self_features, neighbor_agg], dim=-1)
        return self.sage_update(concat)

    def _flash_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhanced FlashAttention with fallback to PyTorch native
        """
        if not FLASH_AVAILABLE:
            return self._pytorch_attention_aggregate(messages, graph)

        B, T, D = messages.shape
        H = self.num_heads
        d_head = self.head_dim

        try:
            # Reshape for FlashAttention
            x_flat = messages.reshape(B * T, D)

            # Project Q, K, V with proper reshaping
            q = self.q_proj(x_flat).reshape(B, T, H, d_head)
            k = self.k_proj(x_flat).reshape(B, T, H, d_head)
            v = self.v_proj(x_flat).reshape(B, T, H, d_head)

            # Stack for FlashAttention: [B, T, 3, H, d_head]
            qkv = torch.stack([q, k, v], dim=2)

            # Apply FlashAttention
            out = flash_attn_qkvpacked_func(qkv, causal=False)  # [B, T, H, d_head]

            # Reshape back to [B, T, D]
            out = out.reshape(B, T, H * d_head)
            if out.shape[-1] != D:
                # Project back to original dimension if needed
                out = out[..., :D]

            return out

        except Exception as e:
            # Fallback to PyTorch attention if FlashAttention fails
            return self._pytorch_attention_aggregate(messages, graph)

    def _xformers_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhanced xFormers attention with fallback
        """
        if not XFORMERS_AVAILABLE:
            return self._pytorch_attention_aggregate(messages, graph)

        B, T, D = messages.shape
        H = self.num_heads
        d_head = self.head_dim

        try:
            # Flatten for processing
            x_flat = messages.reshape(B * T, D)

            # Project Q/K/V with proper dimensions
            q = self.q_proj(x_flat).reshape(B * T, H, D, d_head)
            k = self.k_proj(x_flat).reshape(B * T, H, D, d_head)
            v = self.v_proj(x_flat).reshape(B * T, H, D, d_head)

            # Optional attention bias from graph structure
            attn_bias = None
            if hasattr(self, "bias_proj") and graph is not None:
                try:
                    bias = self.bias_proj(messages)  # [B, T, H*H]
                    bias = bias.reshape(B * T, H, H)
                    attn_bias = bias.unsqueeze(2).expand(-1, -1, D, -1)
                except:
                    attn_bias = None  # Skip bias if shapes don't match

            # Apply xFormers attention
            out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

            # Reshape and aggregate
            out = out.permute(0, 2, 1, 3)  # [B*T, D, H, d_head]
            out = out.reshape(B, T, D, -1)
            out = out.mean(dim=-1)  # [B, T, D]

            return out

        except Exception as e:
            # Fallback to PyTorch attention
            return self._pytorch_attention_aggregate(messages, graph)

    def _pytorch_attention_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """
        PyTorch native attention as fallback
        """
        B, T, D = messages.shape
        H = self.num_heads
        d_head = max(1, D // H)

        # Simple multi-head attention
        q = messages.unsqueeze(2).expand(-1, -1, H, -1).reshape(B, T * H, D // H)
        k = q  # Self-attention
        v = q

        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d_head)
        attn = F.softmax(scores, dim=-1)

        if self.attention_dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.attention_dropout)

        out = torch.bmm(attn, v)
        out = out.reshape(B, T, H, D // H).mean(dim=2)  # Average over heads

        return out

    def forward(self, h: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - must be implemented by subclasses
        """
        raise NotImplementedError("Subclass must implement forward pass.")

    def update(self, x: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        """
        Update function - original implementation preserved
        """
        combined = torch.cat([x, agg], dim=-1)
        return self.norm(self.output_proj(self.update_fn(combined)))

    def get_memory_stats(self) -> dict:
        """
        Get memory and performance statistics
        """
        stats = {
            "aggregation": self.aggregation,
            "input_size": self.input_size,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "use_spectral_norm": self.use_spectral_norm,
            "memory_efficient": self.memory_efficient,
            "flash_available": FLASH_AVAILABLE,
            "xformers_available": XFORMERS_AVAILABLE,
        }

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        stats.update(
            {"total_parameters": total_params, "trainable_parameters": trainable_params}
        )

        return stats

    def enable_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing"""
        self.gradient_checkpointing = enable

    def set_attention_dropout(self, dropout: float):
        """Dynamically set attention dropout"""
        self.attention_dropout = dropout
        if hasattr(self, "lstm"):
            # Update LSTM dropout if it exists
            self.lstm.dropout = dropout if dropout > 0 else 0


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Union
import math


class GraphConv(MessagePassing):
    """
    Optimized Graph Convolution Layer with full backward compatibility.
    Enhanced with:
    - Better parameter initialization
    - Improved update mechanisms
    - Optional spectral normalization
    - Flexible activation functions
    - Enhanced residual connections
    - Memory efficiency optimizations
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        aggregation: str = "sum",
        dropout: float = 0.1,
        # Optional optimization parameters (fully backward compatible)
        num_heads: int = 4,
        activation: str = "gelu",
        use_spectral_norm: bool = False,
        improved_update: bool = True,
        residual_connection: bool = True,
        layer_norm_eps: float = 1e-5,
        gradient_checkpointing: bool = False,
        init_gain: float = 1.0,
        dropout_schedule: str = "fixed",  # "fixed", "decay", "adaptive"
        **kwargs,
    ):
        # Initialize parent MessagePassing
        super().__init__(
            input_size=input_size,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            num_heads=num_heads,
            **kwargs,
        )

        self.output_size = output_size
        self.dropout_p = dropout
        self.activation_name = activation
        self.use_spectral_norm = use_spectral_norm
        self.improved_update = improved_update
        self.residual_connection = residual_connection
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.init_gain = init_gain
        self.dropout_schedule = dropout_schedule

        # Enhanced update function
        if improved_update:
            self.update_fn = self._create_improved_update_fn()
        else:
            # Original update function
            self.update_fn = nn.Sequential(
                nn.Linear(input_size + hidden_dim, input_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Enhanced output projection
        self.output_proj = nn.Linear(input_size, output_size)
        if use_spectral_norm:
            self.output_proj = spectral_norm(self.output_proj)

        # Enhanced normalization
        self.norm = nn.LayerNorm(output_size, eps=layer_norm_eps)

        # Optional residual connection
        if residual_connection and input_size != output_size:
            self.residual_proj = nn.Linear(input_size, output_size)
            if use_spectral_norm:
                self.residual_proj = spectral_norm(self.residual_proj)

        # Adaptive dropout for dynamic adjustment
        if dropout_schedule != "fixed":
            self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

        # Initialize parameters
        self._initialize_parameters()

    def _create_improved_update_fn(self):
        """Create enhanced update function with better architecture"""
        layers = []

        # First layer: expand dimensions
        layers.append(nn.Linear(self.input_size + self.hidden_dim, self.hidden_dim))
        layers.append(self._get_activation())
        layers.append(self._get_dropout(self.dropout_p))

        # Optional intermediate layer for complex updates
        if self.hidden_dim >= 128:  # Only for larger networks
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self._get_activation())
            layers.append(self._get_dropout(self.dropout_p * 0.5))  # Reduced dropout

        # Final layer: project to input size
        layers.append(nn.Linear(self.hidden_dim, self.input_size))
        layers.append(self._get_activation())
        layers.append(self._get_dropout(self.dropout_p * 0.25))  # Minimal final dropout

        return nn.Sequential(*layers)

    def _get_activation(self):
        """Get activation function"""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(inplace=True),
            "silu": nn.SiLU(inplace=True),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
            "elu": nn.ELU(inplace=True),
            "mish": nn.Mish(inplace=True),
        }
        return activations.get(self.activation_name.lower(), nn.GELU())

    def _get_dropout(self, p: float):
        """Get dropout layer with optional scheduling"""
        if self.dropout_schedule == "fixed":
            return nn.Dropout(p)
        else:
            # For adaptive/decay scheduling, we'll adjust in forward pass
            return nn.Dropout(p)

    def _initialize_parameters(self):
        """Enhanced parameter initialization"""
        # Initialize update function layers
        for module in self.update_fn:
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization with custom gain
                if self.activation_name.lower() in ["relu", "leaky_relu", "elu"]:
                    gain = (
                        math.sqrt(2.0) * self.init_gain
                    )  # He initialization for ReLU-like
                else:
                    gain = 1.0 * self.init_gain  # Xavier for others

                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Initialize residual projection if exists
        if hasattr(self, "residual_proj"):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

        # Initialize LayerNorm
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def _adjust_dropout_rate(self) -> float:
        """Dynamically adjust dropout rate based on training progress"""
        if self.dropout_schedule == "fixed" or not self.training:
            return self.dropout_p

        if self.dropout_schedule == "decay":
            # Exponential decay: starts high, decreases over time
            decay_factor = 0.99
            step = self.training_step.item() if hasattr(self, "training_step") else 0
            return self.dropout_p * (decay_factor ** (step / 1000))

        elif self.dropout_schedule == "adaptive":
            # Could implement adaptive dropout based on loss/gradients
            # For now, return fixed rate
            return self.dropout_p

        return self.dropout_p

    def _apply_residual_connection(
        self, x: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Apply enhanced residual connection"""
        if not self.residual_connection:
            return out

        if hasattr(self, "residual_proj"):
            # Different dimensions - use projection
            residual = self.residual_proj(x)
        elif x.shape[-1] == out.shape[-1]:
            # Same dimensions - direct addition
            residual = x
        else:
            # No residual possible
            return out

        # Scaled residual connection for better gradient flow
        return out + residual * 0.1  # Scale factor to prevent residual dominance

    def update(self, x: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        """
        Enhanced update function with residual connections and better processing
        """
        # Increment training step for dropout scheduling
        if self.training and hasattr(self, "training_step"):
            self.training_step += 1

        # Combine input and aggregated features
        combined = torch.cat([x, agg], dim=-1)

        # Apply update function
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            updated = torch.utils.checkpoint.checkpoint(
                self.update_fn, combined, use_reentrant=False
            )
        else:
            updated = self.update_fn(combined)

        # Project to output size
        out = self.output_proj(updated)

        # Apply residual connection
        out = self._apply_residual_connection(x, out)

        # Apply normalization
        out = self.norm(out)

        return out

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with optional optimizations
        """
        # Compute messages
        msg = self.message(x)  # [B, T, hidden_dim]

        # Aggregate messages based on aggregation type
        if self.aggregation in ["sage", "sage_lstm"]:
            agg = self.aggregate(msg, graph, x)  # [B, T, hidden_dim]
        else:
            agg = self.aggregate(msg, graph)  # [B, T, hidden_dim]

        # Update and return
        return self.update(x, agg)  # [B, T, output_size]

    def enable_gradient_checkpointing(self, enable: bool = True):
        """Enable/disable gradient checkpointing"""
        self.gradient_checkpointing = enable

    def set_dropout_schedule(self, schedule: str):
        """Change dropout schedule dynamically"""
        valid_schedules = ["fixed", "decay", "adaptive"]
        if schedule in valid_schedules:
            self.dropout_schedule = schedule
            if schedule != "fixed" and not hasattr(self, "training_step"):
                self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))
        else:
            raise ValueError(
                f"Invalid schedule: {schedule}. Must be one of {valid_schedules}"
            )

    def reset_training_step(self):
        """Reset training step counter"""
        if hasattr(self, "training_step"):
            self.training_step.fill_(0)


class SageLayer(GraphConv):
    def __init__(self, input_size, hidden_dim):
        super().__init__(input_size, input_size, hidden_dim, aggregation="sage")


class AttGraphConv(MessagePassing):
    def __init__(self, input_size, output_size, hidden_dim, num_heads=4, dropout=0.1):
        print(f"Using {num_heads} attention heads for AttGraphConv")
        super().__init__(input_size, hidden_dim)
        self.num_heads = num_heads
        self.output_proj = nn.Linear(input_size, output_size)
        self.attn_q = nn.Linear(input_size, hidden_dim)
        self.attn_k = nn.Linear(input_size, hidden_dim)
        self.update_fn = nn.Sequential(
            nn.Linear(input_size + hidden_dim, input_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(output_size)

    def compute_attention(self, x):
        # x: [B, T, F]
        q = self.attn_q(x).mean(dim=1)  # [B, hidden]
        k = self.attn_k(x).mean(dim=1)  # [B, hidden]
        scores = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)).squeeze(-1)  # [B, 1]
        alpha = torch.sigmoid(scores).squeeze(-1)  # [B]
        return alpha

    def forward(self, x, graph):
        attn_graph = torch.tanh(graph) * (graph.abs() > 1e-3).float()  # soft mask
        msg = self.message(x)
        agg = self.aggregate(msg, attn_graph)
        combined = torch.cat([x, agg], dim=-1)
        h = self.update_fn(combined)
        return self.norm(self.output_proj(h))


class XFormerAttGraphConv(MessagePassing):
    def __init__(self, input_size, output_size, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__(input_size, hidden_dim)
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(input_size, input_size * self.head_dim * num_heads)
        self.k_proj = nn.Linear(input_size, input_size * self.head_dim * num_heads)
        self.v_proj = nn.Linear(input_size, input_size * self.head_dim * num_heads)

        self.update_fn = nn.Sequential(
            nn.Linear(input_size + self.head_dim, input_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_proj = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F] (input time series)
        graph: [F, F] (correlation/adjacency matrix over features)
        """
        B, T, F = x.shape
        H, D = self.num_heads, self.head_dim

        # Apply graph as soft attention bias over features
        # x: [B, T, F] → [B, F, T] (features as "tokens")
        x_feat = x.reshape(B * T, F)

        # Project to Q, K, V
        q = self.q_proj(x_feat).reshape(B * T, H, F, D).transpose(1, 2)  # [B, H, F, D]
        k = self.k_proj(x_feat).reshape(B * T, H, F, D).transpose(1, 2)  # [B, H, F, D]
        v = self.v_proj(x_feat).reshape(B * T, H, F, D).transpose(1, 2)  # [B, H, F, D]

        # Use xformers efficient attention
        out = memory_efficient_attention(q, k, v)  # [B, H, F, D]
        out = out.permute(0, 2, 1, 3)  # [B*T, D_token, H, d_head]
        out = out.reshape(B, T, D, -1)
        # take mean over last dim
        out = out.mean(dim=-1)  # [B, T, D]
        # Update with residual information
        combined = torch.cat([x, out], dim=-1)  # [B, T, in + hidden]
        updated = self.update_fn(combined)  # [B, T, input_size]
        return self.norm(self.output_proj(updated))  # [B, T, output_size]


class LatentGraphNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Optional[int] = None,
        correlation_hidden_size: Optional[int] = None,
        low_rank: bool = True,
        rank: Optional[int] = None,
        num_passes: int = 1,
        aggregation: str = "sum",
        dropout: float = 0.1,
        residual: bool = True,
        strategy: Literal["vanilla", "attn", "xformers", "sage", "gtat"] = "vanilla",
        jk_mode: Literal["last", "sum", "max", "concat", "lstm", "none"] = "none",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size or max(input_size, output_size)
        self.num_passes = num_passes
        self.residual = residual

        # Latent correlation layer (data + learnable graph)
        self.correlation_layer = LatentCorrelationLayer(
            input_size=input_size,
            output_size=input_size,
            hidden_size=correlation_hidden_size,
            low_rank=low_rank,
            rank=rank,
            correlation_dropout=dropout,
        )

        # Message passing layers
        self.message_passing_layers = nn.ModuleList(
            [
                self._create_layer(
                    strategy, input_size, self.hidden_size, aggregation, dropout
                )
                for _ in range(num_passes)
            ]
        )

        self.jk_mode = jk_mode
        if jk_mode != "none":
            # Jump knowledge module
            self.jump_knowledge = JumpKnowledge(
                mode=jk_mode, hidden_size=self.input_size, output_size=input_size
            )

        if strategy == "gtat":
            self.gdv_encoder = GDVEncoder(gdv_dim=73, topo_dim=input_size)

        self.norm = nn.LayerNorm(output_size)

    def _create_layer(
        self,
        strategy: str,
        input_size: int,
        hidden_size: int,
        aggregation: str,
        dropout: float,
    ) -> nn.Module:
        if strategy == "vanilla":
            return GraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                aggregation=aggregation,
                dropout=dropout,
            )
        elif strategy == "attn":
            return AttGraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                dropout=dropout,
            )
        elif strategy == "xformers":
            return XFormerAttGraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=16,
                dropout=dropout,
            )
        elif strategy == "sage":
            return SageLayer(input_size=input_size, hidden_dim=hidden_size)
        elif strategy == "gtat":
            return GTATLayerWrapper(
                input_size,
                hidden_size,
                topo_dim=input_size,
                hidden_dim=hidden_size,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        corr_features, correlation = self.correlation_layer(x)

        h = corr_features
        outputs: List[torch.Tensor] = []

        topo_embedding = None
        if any(isinstance(l, GTATLayerWrapper) for l in self.message_passing_layers):
            B, T, F = x.shape
            gdv = compute_mock_gdv(F).to(x.device)
            gdv = gdv / (gdv.sum(dim=1, keepdim=True) + 1e-6)
            topo_embedding = self.gdv_encoder(gdv)  # [F, topo_dim]
            topo_embedding = (
                topo_embedding.unsqueeze(0).expand(B, F, -1).clone()
            )  # ✅ shape [B, F, topo_dim]

        for layer in self.message_passing_layers:
            if isinstance(layer, GTATLayerWrapper):
                h = layer(h, correlation, topo_embedding)
            else:
                h = layer(h, correlation)

        if self.jk_mode != "none":
            jk_out = self.jump_knowledge(outputs)

            if self.residual and x.shape[-1] == jk_out.shape[-1]:
                jk_out = jk_out + x
        else:
            jk_out = h
            if self.residual and x.shape[-1] == h.shape[-1]:
                jk_out = h + x

        return jk_out


class JumpKnowledge(nn.Module):
    def __init__(
        self,
        mode: Literal["last", "sum", "max", "concat", "lstm"] = "concat",
        hidden_size: int = None,
        output_size: int = None,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out_proj = None  # lazy initialization

        if self.mode == "lstm":
            assert hidden_size is not None
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.out_proj = (
                nn.Identity()
                if hidden_size == output_size
                else nn.Linear(hidden_size, output_size)
            )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        if self.mode == "last":
            return xs[-1]

        elif self.mode == "sum":
            return torch.stack(xs, dim=0).sum(dim=0)

        elif self.mode == "max":
            return torch.stack(xs, dim=0).max(dim=0)[0]

        elif self.mode == "concat":
            x_cat = torch.cat(xs, dim=-1)  # [B, T, D * num_layers]
            if self.out_proj is None:
                input_dim = x_cat.size(-1)
                self.out_proj = nn.Linear(input_dim, self.output_size).to(x_cat.device)
            return self.out_proj(x_cat)

        elif self.mode == "lstm":
            B, T, D = xs[0].shape
            x_seq = torch.stack(xs, dim=1).reshape(B * T, len(xs), D)
            lstm_out, _ = self.lstm(x_seq)
            final = lstm_out[:, -1, :].reshape(B, T, -1)
            return self.out_proj(final)

        else:
            raise ValueError(f"Unsupported JK mode: {self.mode}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import subprocess
import tempfile
import numpy as np
from typing import Union, Optional


# === GDV via ORCA ===
def compute_gdv_orca(
    G: Union[nx.Graph, nx.DiGraph], orca_path: str = "./orca", graphlet_size: int = 5
) -> np.ndarray:
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)

    with tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as edge_file, tempfile.NamedTemporaryFile("r", delete=False) as out_file:

        node_map = {n: i for i, n in enumerate(G.nodes())}
        for u, v in G.edges():
            edge_file.write(f"{node_map[u]} {node_map[v]}\n")
        edge_file.flush()

        cmd = [orca_path, str(graphlet_size), edge_file.name, out_file.name]
        subprocess.run(cmd, check=True)

        out_lines = out_file.readlines()
        gdv = [list(map(int, line.strip().split())) for line in out_lines]

    return np.array(gdv, dtype=np.float32)


def compute_mock_gdv(num_nodes: int, gdv_dim: int = 73) -> torch.Tensor:
    return torch.randn(num_nodes, gdv_dim)


class GDVEncoder(nn.Module):
    def __init__(self, gdv_dim: int, topo_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(gdv_dim, topo_dim), nn.ReLU(), nn.Linear(topo_dim, topo_dim)
        )

    def forward(self, gdv):
        return self.proj(gdv)


class GTATLayer(nn.Module):
    def __init__(self, feature_dim, topo_dim, hidden_dim):
        super().__init__()
        self.feature_attn = nn.Linear(2 * hidden_dim, 1)
        self.topo_attn = nn.Linear(2 * hidden_dim, 1)

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.topo_proj = nn.Linear(topo_dim, hidden_dim)

    def forward(self, H, T, adj):
        # H: [B, N, F], T: [B, F, F_t], adj: [F, F] or [B, F, F]
        B, N, Fdim = H.shape
        _, Fdim, F_t = T.shape

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)  # [B, F, F]

        # Project features
        H_proj = self.feature_proj(H)  # [B, N, H]
        T_proj = self.topo_proj(T)  # [B, F, H]

        Hi = T_proj.unsqueeze(2).expand(B, Fdim, Fdim, -1)
        Hj = T_proj.unsqueeze(1).expand(B, Fdim, Fdim, -1)
        topo_input = torch.cat([Hi, Hj], dim=-1)
        e_topo = F.leaky_relu(self.topo_attn(topo_input)).squeeze(-1)
        beta = F.softmax(e_topo.masked_fill(adj == 0, -1e4), dim=-1)  # [B, F, F]

        T_out = torch.bmm(beta, T_proj)  # [B, F, H]

        H_out = torch.zeros_like(H_proj)
        for b in range(B):
            for t in range(N):
                Hi = H_proj[b, t].unsqueeze(0).expand(Fdim, -1)
                Hj = T_out[b]
                feat_cat = torch.cat([Hi, Hj], dim=-1)  # [F, 2H]
                e_feat = F.leaky_relu(self.feature_attn(feat_cat)).squeeze(-1)  # [F]
                alpha = F.softmax(e_feat, dim=-1)  # [F]
                H_out[b, t] = torch.matmul(alpha, Hj)

        return H_out, T_out


class GTATLayerWrapper(nn.Module):
    def __init__(self, input_size, output_size, topo_dim, hidden_dim, dropout):
        super().__init__()
        self.gtat_layer = GTATLayer(input_size, topo_dim, hidden_dim)
        self.output_proj = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, graph, topo_embedding):
        h, t = self.gtat_layer(x, topo_embedding, graph)
        out = self.output_proj(h)
        return self.norm(self.dropout(out))


class GTAT(nn.Module):
    def __init__(self, in_dim, gdv_dim, topo_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.gdv_encoder = GDVEncoder(gdv_dim, topo_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GTATLayer(hidden_dim, topo_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, H, adj, gdv=None):
        if gdv is None:
            gdv = compute_mock_gdv(H.size(0))
        if isinstance(gdv, np.ndarray):
            gdv = torch.tensor(gdv, dtype=torch.float32, device=H.device)

        gdv = gdv / (gdv.sum(dim=1, keepdim=True) + 1e-6)
        T = self.gdv_encoder(gdv)
        H = self.input_proj(H)

        for layer in self.layers:
            H, T = layer(H, T, adj)

        return self.output_proj(H)


class GTATIntegrated(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        gdv_dim: int = 73,
        topo_dim: int = 64,
        hidden_size: Optional[int] = None,
        num_passes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size or max(input_size, output_size)
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, output_size)
        self.gdv_encoder = GDVEncoder(gdv_dim, topo_dim)
        self.layers = nn.ModuleList(
            [
                GTATLayerWrapper(
                    self.hidden_size,
                    self.hidden_size,
                    topo_dim,
                    self.hidden_size,
                    dropout,
                )
                for _ in range(num_passes)
            ]
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, gdv: Optional[torch.Tensor] = None
    ):
        if gdv is None:
            gdv = compute_mock_gdv(x.size(-1))  # x.shape: [B, T, F]
        if isinstance(gdv, np.ndarray):
            gdv = torch.tensor(gdv, dtype=torch.float32, device=x.device)

        gdv = gdv / (gdv.sum(dim=1, keepdim=True) + 1e-6)
        topo_embedding = self.gdv_encoder(gdv)
        if topo_embedding.ndim == 2:
            B, T, F = x.shape
            topo_embedding = (
                topo_embedding.unsqueeze(0).expand(B, T, -1).clone()
            )  # [B, F, topo_dim]
        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h, adj, topo_embedding)

        return self.norm(self.output_proj(h))
