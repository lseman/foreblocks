import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Union, Tuple, List, Dict, Literal

# External libraries
from xformers.ops import memory_efficient_attention
from flash_attn import flash_attn_qkvpacked_func


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
    Clean message passing base class with full backward compatibility.
    Optimized for essential functionality without bloat.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        aggregation: str = "sum",
        num_heads: int = 4,
        # Backward compatibility parameters (kept but simplified)
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
        num_heads = round_to_supported_head_dim(num_heads)
        self.num_heads = num_heads
        self.eps = eps
        self.attention_dropout = attention_dropout

        # Check xFormers availability
        try:
            from xformers.ops import memory_efficient_attention

            self.xformers_available = True
        except ImportError:
            self.xformers_available = False

        # Simple head dimension calculation
        self.head_dim = round_to_supported_head_dim(hidden_dim)
        print(f"Using {num_heads} attention heads with head dimension {self.head_dim}")

        # Core message transformation
        self.message_transform = nn.Linear(input_size, hidden_dim)
        if use_spectral_norm:
            from torch.nn.utils import spectral_norm

            self.message_transform = spectral_norm(self.message_transform)

        # SAGE components (only if needed)
        if aggregation in ["sage", "sage_lstm"]:
            self.sage_update = nn.Linear(input_size + hidden_dim, hidden_dim)
            if use_spectral_norm:
                self.sage_update = spectral_norm(self.sage_update)

        if aggregation == "sage_lstm":
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=attention_dropout if attention_dropout > 0 else 0,
            )

        # Attention projections (only if needed)
        if aggregation in ["xformers", "flash"]:
            proj_dim = self.head_dim * num_heads
            self.q_proj = nn.Linear(input_size, proj_dim, bias=False)
            self.k_proj = nn.Linear(input_size, proj_dim, bias=False)
            self.v_proj = nn.Linear(input_size, proj_dim, bias=False)

            if use_spectral_norm:
                self.q_proj = spectral_norm(self.q_proj)
                self.k_proj = spectral_norm(self.k_proj)
                self.v_proj = spectral_norm(self.v_proj)

            self.bias_proj = nn.Linear(input_size, proj_dim, bias=False)
        # Initialize parameters
        if improved_init:
            self._init_parameters()

    def _init_parameters(self):
        """Clean parameter initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def message(self, h: torch.Tensor) -> torch.Tensor:
        """Compute messages from node features"""
        return self.message_transform(h)  # [B, T, hidden_dim]

    def aggregate(
        self,
        messages: torch.Tensor,
        graph: torch.Tensor,
        self_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Enhanced aggregation with better numerical stability"""
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
        """Mean aggregation with safe division"""
        deg = graph.sum(dim=1, keepdim=True).clamp(min=self.eps)

        # Use safe division to prevent NaN creation
        norm_graph = torch.where(deg > self.eps, graph / deg, torch.zeros_like(graph))

        result = messages @ norm_graph
        return result

    def _max_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """Max aggregation"""
        expanded = torch.einsum("bth,hg->bthg", messages, graph)
        return expanded.max(dim=2)[0]

    def _sage_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor, self_features: torch.Tensor
    ) -> torch.Tensor:
        """SAGE aggregation"""
        assert self_features is not None, "SAGE requires self node features"

        deg = graph.sum(dim=1).clamp(min=self.eps)
        norm_graph = graph / deg.unsqueeze(1)
        neighbor_agg = torch.einsum("bth,hg->btg", messages, norm_graph)

        concat = torch.cat([self_features, neighbor_agg], dim=-1)
        return self.sage_update(concat)

    def _sage_lstm_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor, self_features: torch.Tensor
    ) -> torch.Tensor:
        """SAGE-LSTM aggregation"""
        assert self_features is not None, "SAGE-LSTM requires self node features"

        neighbor_sequences = torch.einsum("bth,hg->btg", messages, graph).transpose(
            1, 2
        )
        lstm_out, _ = self.lstm(neighbor_sequences)
        neighbor_agg, _ = torch.max(lstm_out, dim=1)
        neighbor_agg = neighbor_agg.unsqueeze(1).expand(-1, messages.size(1), -1)

        concat = torch.cat([self_features, neighbor_agg], dim=-1)
        return self.sage_update(concat)

    def _xformers_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """xFormers memory-efficient attention with fallback"""
        if not self.xformers_available:
            return self._pytorch_attention_aggregate(messages, graph)

        try:
            from xformers.ops import memory_efficient_attention

            B, T, D = messages.shape
            # print(messages.shape)
            H = self.num_heads
            head_dim = self.head_dim

            # Project to Q, K, V
            q = self.q_proj(messages).view(B, T, H, head_dim)
            k = self.k_proj(messages).view(B, T, H, head_dim)
            v = self.v_proj(messages).view(B, T, H, head_dim)

            # Create attention bias from graph
            attn_bias = None
            if graph is not None:
                # Option 1: Project messages to bias space
                bias_proj = torch.nn.Linear(
                    D, T, device=messages.device, dtype=messages.dtype
                )
                attn_logits = bias_proj(messages)  # Shape: (B, T, T)

                # Apply graph mask
                mask = (
                    graph == 0
                    if graph.shape[-2:] == (T, T)
                    else torch.zeros(B, T, T, device=messages.device, dtype=torch.bool)
                )
                attn_bias = attn_logits.masked_fill(mask, float("-inf"))

                # Expand for all heads: (B, T, T) -> (B, H, T, T)
                attn_bias = attn_bias.unsqueeze(1).expand(B, H, T, T)

            # Apply xFormers attention
            out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            return out.reshape(B, T, H * head_dim)[:, :, : self.hidden_dim]

        except Exception:
            return self._pytorch_attention_aggregate(messages, graph)

    def _flash_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """FlashAttention with fallback"""
        # For now, fallback to xFormers or PyTorch attention
        return self._xformers_aggregate(messages, graph)

    def _pytorch_attention_aggregate(
        self, messages: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch native attention fallback"""
        B, T, D = messages.shape
        H = self.num_heads
        head_dim = max(1, self.hidden_dim // H)

        # Simple multi-head attention
        q = messages.view(B, T, H, head_dim)
        k = q
        v = q

        # Scaled dot-product attention
        scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)

        # Apply graph mask
        if graph is not None:
            mask = (graph == 0).unsqueeze(1).expand(-1, H, -1, -1)
            scores.masked_fill_(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.attention_dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.attention_dropout)

        out = torch.einsum("bhts,bshd->bthd", attn, v)
        return out.reshape(B, T, H * head_dim)[:, :, : self.hidden_dim]

    def forward(self, h: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError("Subclass must implement forward pass.")

    # Backward compatibility methods (simplified)
    def enable_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing"""
        self.gradient_checkpointing = enable


class GraphConv(MessagePassing):
    """
    Clean and optimized Graph Convolution Layer.
    Focuses on essential optimizations without bloat.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        aggregation: str = "sum",
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            **kwargs,
        )

        self.output_size = output_size
        self.dropout_p = dropout

        # Pre-compute activation for efficiency
        self.activation = self._get_activation(activation)

        # Streamlined update function
        self.update_fn = nn.Sequential(
            nn.Linear(input_size + hidden_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(output_size)

        # Residual projection only if dimensions differ
        self.residual_proj = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )

        self._initialize_parameters()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function efficiently"""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
        }
        return activations.get(activation.lower(), nn.GELU())

    def _initialize_parameters(self):
        """Simple but effective parameter initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def update(self, x: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        """Clean update function with residual connection"""
        # Combine and process
        combined = torch.cat([x, agg], dim=-1)
        out = self.update_fn(combined)

        # Add residual connection if possible
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        elif x.shape[-1] == out.shape[-1]:
            out = out + x

        # Normalize and return
        return self.norm(out)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        msg = self.message(x)

        if self.aggregation in ["sage", "sage_lstm"]:
            agg = self.aggregate(msg, graph, x)
        else:
            agg = self.aggregate(msg, graph)

        return self.update(x, agg)


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
        self.jk_mode = jk_mode
        self.strategy = strategy

        # Pre-compute strategy checks to avoid repeated isinstance calls
        self.uses_gtat = strategy == "gtat"
        self.uses_jk = jk_mode != "none"

        # print the info
        print(
            f"[LatentLayer] Using strategy: {strategy}, num_passes: {num_passes}, aggregation: {aggregation}"
            f", dropout: {dropout}, residual: {residual}, jk_mode: {jk_mode}"
        )

        # Latent correlation layer (data + learnable graph)
        self.correlation_layer = LatentCorrelationLayer(
            input_size=input_size,
            output_size=input_size,
            hidden_size=correlation_hidden_size,
            low_rank=low_rank,
            rank=rank,
            correlation_dropout=dropout,
        )

        # Message passing layers - use factory pattern for cleaner creation
        self.message_passing_layers = nn.ModuleList(
            [
                self._create_layer(
                    strategy, input_size, self.hidden_size, aggregation, dropout
                )
                for _ in range(num_passes)
            ]
        )

        # Jump knowledge module - only create if needed
        if self.uses_jk:
            self.jump_knowledge = JumpKnowledge(
                mode=jk_mode, hidden_size=self.input_size, output_size=input_size
            )

        # GTAT components - only create if needed
        if self.uses_gtat:
            self.gdv_encoder = GDVEncoder(gdv_dim=73, topo_dim=input_size)
            # Pre-compute and cache GDV to avoid recomputation
            self.register_buffer("cached_gdv", None)

        self.norm = nn.LayerNorm(output_size)

    def _create_layer(
        self,
        strategy: str,
        input_size: int,
        hidden_size: int,
        aggregation: str,
        dropout: float,
    ) -> nn.Module:
        # Use dictionary dispatch instead of if-elif chain for better performance
        layer_factory = {
            "vanilla": lambda: GraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                aggregation=aggregation,
                dropout=dropout,
            ),
            "attn": lambda: AttGraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                dropout=dropout,
            ),
            "xformers": lambda: XFormerAttGraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=16,
                dropout=dropout,
            ),
            "sage": lambda: SageLayer(input_size=input_size, hidden_dim=hidden_size),
            "gtat": lambda: GTATLayerWrapper(
                input_size,
                hidden_size,
                topo_dim=input_size,
                hidden_dim=hidden_size,
                dropout=dropout,
            ),
        }

        if strategy not in layer_factory:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return layer_factory[strategy]()

    def _get_topo_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Compute or retrieve cached topology embedding for GTAT."""
        B, T, F = x.shape

        # Check if we need to compute/cache GDV
        if self.cached_gdv is None or self.cached_gdv.size(0) != F:
            gdv = compute_mock_gdv(F).to(x.device)
            gdv = gdv / (gdv.sum(dim=1, keepdim=True) + 1e-6)
            self.cached_gdv = gdv

        topo_embedding = self.gdv_encoder(self.cached_gdv)  # [F, topo_dim]
        return topo_embedding.unsqueeze(0).expand(B, F, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get correlation features and adjacency matrix
        corr_features, correlation = self.correlation_layer(x)
        h = corr_features

        # Pre-compute topology embedding if needed (avoid redundant computation)
        topo_embedding = None
        if self.uses_gtat:
            topo_embedding = self._get_topo_embedding(x)

        # Store outputs for jump knowledge if needed
        outputs = [] if self.uses_jk else None

        # Message passing
        for layer in self.message_passing_layers:
            if self.uses_gtat and isinstance(layer, GTATLayerWrapper):
                h = layer(h, correlation, topo_embedding)
            else:
                h = layer(h, correlation)

            # Only append to outputs if jump knowledge is used
            if self.uses_jk:
                outputs.append(h)

        # Jump knowledge or direct output
        if self.uses_jk:
            jk_out = self.jump_knowledge(outputs)
            # Residual connection with shape compatibility check
            if self.residual and x.shape[-1] == jk_out.shape[-1]:
                jk_out = jk_out + x
            result = jk_out
        else:
            # Direct output with residual connection
            if self.residual and x.shape[-1] == h.shape[-1]:
                result = h + x
            else:
                result = h

        return result


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

        # Pre-compute mode-specific setup to avoid runtime checks
        self._setup_mode_components()

    def _setup_mode_components(self):
        """Initialize components based on the selected mode."""
        if self.mode == "lstm":
            if self.hidden_size is None:
                raise ValueError("hidden_size must be provided for LSTM mode")

            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
            self.out_proj = (
                nn.Identity()
                if self.hidden_size == self.output_size
                else nn.Linear(self.hidden_size, self.output_size)
            )
        elif self.mode == "concat":
            # For concat mode, we'll use lazy initialization but prepare the flag
            self.out_proj = None
            self._concat_proj_initialized = False
        else:
            # For other modes, no additional components needed
            self.out_proj = None

    def _init_concat_projection(self, input_dim: int, device: torch.device):
        """Initialize concatenation projection layer lazily."""
        if not self._concat_proj_initialized:
            self.out_proj = nn.Linear(input_dim, self.output_size)
            # Move to correct device
            self.out_proj = self.out_proj.to(device)
            self._concat_proj_initialized = True

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        if not xs:
            raise ValueError("Input list cannot be empty")

        # Use dictionary dispatch for better performance
        return self._forward_dispatch[self.mode](self, xs)

    def _forward_last(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return xs[-1]

    def _forward_sum(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # More memory-efficient than stacking then summing
        result = xs[0]
        for x in xs[1:]:
            result = result + x
        return result

    def _forward_max(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # More memory-efficient elementwise max
        result = xs[0]
        for x in xs[1:]:
            result = torch.maximum(result, x)
        return result

    def _forward_concat(self, xs: List[torch.Tensor]) -> torch.Tensor:
        x_cat = torch.cat(xs, dim=-1)  # [B, T, D * num_layers]

        if self.out_proj is None:
            input_dim = x_cat.size(-1)
            self._init_concat_projection(input_dim, x_cat.device)

        return self.out_proj(x_cat)

    def _forward_lstm(self, xs: List[torch.Tensor]) -> torch.Tensor:
        B, T, D = xs[0].shape
        # Stack along sequence dimension efficiently
        x_seq = torch.stack(xs, dim=1)  # [B, num_layers, T, D]
        x_seq = x_seq.transpose(1, 2).reshape(B * T, len(xs), D)  # [B*T, num_layers, D]

        lstm_out, _ = self.lstm(x_seq)
        # Get the last output for each sequence
        final = lstm_out[:, -1, :].reshape(B, T, -1)
        return self.out_proj(final)

    # Create dispatch dictionary as class attribute for efficiency
    @property
    def _forward_dispatch(self):
        if not hasattr(self, "__forward_dispatch"):
            self.__forward_dispatch = {
                "last": self._forward_last,
                "sum": self._forward_sum,
                "max": self._forward_max,
                "concat": self._forward_concat,
                "lstm": self._forward_lstm,
            }
        return self.__forward_dispatch


import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import subprocess
import tempfile
import numpy as np
from typing import Union, Optional


# === GDV Computation ===
def compute_gdv_orca(
    G: Union[nx.Graph, nx.DiGraph], orca_path: str = "./orca", graphlet_size: int = 5
) -> np.ndarray:
    """Compute GDV using ORCA - simplified but functional"""
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)

    with tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as edge_file, tempfile.NamedTemporaryFile("r", delete=False) as out_file:

        # Write edges with node mapping
        node_map = {n: i for i, n in enumerate(G.nodes())}
        for u, v in G.edges():
            edge_file.write(f"{node_map[u]} {node_map[v]}\n")
        edge_file.flush()

        # Run ORCA
        cmd = [orca_path, str(graphlet_size), edge_file.name, out_file.name]
        subprocess.run(cmd, check=True)

        # Parse output
        gdv = [list(map(int, line.strip().split())) for line in out_file.readlines()]

    return np.array(gdv, dtype=np.float32)


def compute_mock_gdv(num_nodes: int, gdv_dim: int = 73) -> torch.Tensor:
    """Generate mock GDV for testing/prototyping"""
    return torch.randn(num_nodes, gdv_dim)


class GDVEncoder(nn.Module):
    """Clean GDV encoder with residual connection"""

    def __init__(self, gdv_dim: int, topo_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(gdv_dim, topo_dim),
            nn.ReLU(inplace=True),
            nn.Linear(topo_dim, topo_dim),
        )
        # Add residual projection if dimensions differ
        self.residual_proj = (
            nn.Linear(gdv_dim, topo_dim) if gdv_dim != topo_dim else None
        )

    def forward(self, gdv: torch.Tensor) -> torch.Tensor:
        out = self.proj(gdv)
        if self.residual_proj is not None:
            out = out + self.residual_proj(gdv)
        elif gdv.shape[-1] == out.shape[-1]:
            out = out + gdv
        return out


def get_mask_value(tensor: torch.Tensor) -> float:
    """Get appropriate mask value based on tensor dtype"""
    if tensor.dtype == torch.float16:
        return -65504.0  # Safe value for float16
    elif tensor.dtype == torch.bfloat16:
        return -3.38e38  # Safe value for bfloat16 (but conservative)
    else:
        return -1e9  # Standard value for float32


class GTATLayer(nn.Module):
    """Optimized GTAT layer with vectorized operations and dtype-safe masking"""

    def __init__(self, feature_dim: int, topo_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Attention mechanisms
        self.feature_attn = nn.Linear(2 * hidden_dim, 1)
        self.topo_attn = nn.Linear(2 * hidden_dim, 1)

        # Projections
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.topo_proj = nn.Linear(topo_dim, hidden_dim)

    def forward(self, H: torch.Tensor, T: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        H: [B, N, F] - node features
        T: [B, F, F_t] - topology features
        adj: [F, F] or [B, F, F] - adjacency matrix
        """
        B, N, F_dim = H.shape

        # Ensure adjacency is 3D
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)

        # Project features
        H_proj = self.feature_proj(H)  # [B, N, H]
        T_proj = self.topo_proj(T)  # [B, F, H]

        # Vectorized topology attention
        T_expanded_i = T_proj.unsqueeze(2).expand(-1, -1, F_dim, -1)  # [B, F, F, H]
        T_expanded_j = T_proj.unsqueeze(1).expand(-1, F_dim, -1, -1)  # [B, F, F, H]
        topo_cat = torch.cat([T_expanded_i, T_expanded_j], dim=-1)  # [B, F, F, 2H]

        e_topo = self.topo_attn(topo_cat).squeeze(-1)  # [B, F, F]
        e_topo = F.leaky_relu(e_topo)

        # Apply adjacency mask and softmax with dtype-safe mask value
        mask_value = get_mask_value(e_topo)
        beta = F.softmax(e_topo.masked_fill(adj == 0, mask_value), dim=-1)
        T_out = torch.bmm(beta, T_proj)  # [B, F, H]

        # Vectorized feature attention (much more efficient)
        H_expanded = H_proj.unsqueeze(2).expand(-1, -1, F_dim, -1)  # [B, N, F, H]
        T_expanded = T_out.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, F, H]
        feat_cat = torch.cat([H_expanded, T_expanded], dim=-1)  # [B, N, F, 2H]

        e_feat = self.feature_attn(feat_cat).squeeze(-1)  # [B, N, F]
        e_feat = F.leaky_relu(e_feat)
        alpha = F.softmax(e_feat, dim=-1)  # [B, N, F]

        # Weighted aggregation
        H_out = torch.bmm(
            alpha.view(B * N, 1, F_dim),
            T_out.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, F_dim, -1),
        )
        H_out = H_out.squeeze(1).view(B, N, -1)  # [B, N, H]

        return H_out, T_out


class GTATLayerWrapper(nn.Module):
    """Clean wrapper with proper normalization"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        topo_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gtat_layer = GTATLayer(input_size, topo_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

        # Residual connection if dimensions match
        self.residual_proj = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )

    def forward(
        self, x: torch.Tensor, graph: torch.Tensor, topo_embedding: torch.Tensor
    ) -> torch.Tensor:
        h, _ = self.gtat_layer(x, topo_embedding, graph)
        out = self.dropout(self.output_proj(h))

        # Add residual connection
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        elif x.shape[-1] == out.shape[-1]:
            out = out + x

        return self.norm(out)


class GTATIntegrated(nn.Module):
    """Clean, optimized GTAT model"""

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
        self.topo_dim = topo_dim

        # Core components
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, output_size)
        self.gdv_encoder = GDVEncoder(gdv_dim, topo_dim)

        # GTAT layers
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

        # Cache for GDV computation
        self.register_buffer("cached_gdv", None)

    def _get_topo_embedding(
        self, x: torch.Tensor, gdv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get topology embedding with caching"""
        B, T, F = x.shape

        if gdv is None:
            # Check cache first
            if self.cached_gdv is None or self.cached_gdv.size(0) != F:
                gdv = compute_mock_gdv(F, 73).to(x.device)
                self.cached_gdv = gdv
            else:
                gdv = self.cached_gdv
        elif isinstance(gdv, np.ndarray):
            gdv = torch.tensor(gdv, dtype=torch.float32, device=x.device)

        # Normalize GDV
        gdv = gdv / (gdv.sum(dim=1, keepdim=True) + 1e-6)

        # Encode and expand for batch
        topo_embedding = self.gdv_encoder(gdv)  # [F, topo_dim]
        if topo_embedding.ndim == 2:
            topo_embedding = topo_embedding.unsqueeze(0).expand(B, -1, -1)

        return topo_embedding

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, gdv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optimized topology handling"""
        # Get topology embedding
        topo_embedding = self._get_topo_embedding(x, gdv)

        # Project input
        h = self.input_proj(x)

        # Apply GTAT layers
        for layer in self.layers:
            h = layer(h, adj, topo_embedding)

        # Final projection and normalization
        out = self.output_proj(h)
        return self.norm(out)


# Simplified standalone GTAT for backward compatibility
class GTAT(nn.Module):
    """Simplified GTAT for backward compatibility"""

    def __init__(
        self,
        in_dim: int,
        gdv_dim: int,
        topo_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.integrated = GTATIntegrated(
            input_size=in_dim,
            output_size=out_dim,
            gdv_dim=gdv_dim,
            topo_dim=topo_dim,
            hidden_size=hidden_dim,
            num_passes=num_layers,
        )

    def forward(
        self, H: torch.Tensor, adj: torch.Tensor, gdv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.integrated(H, adj, gdv)
