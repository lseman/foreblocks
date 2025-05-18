import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Literal
import math
import contextlib

# External
from xformers.ops import memory_efficient_attention



class LatentCorrelationLayer(nn.Module):
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
        eps: float = 1e-8  # Added epsilon parameter for numerical stability
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size or (2 * input_size)
        self.low_rank = low_rank
        self.rank = rank or max(1, input_size // 4)
        self.use_layer_norm = use_layer_norm
        self.correlation_dropout = correlation_dropout
        self.cheb_k = cheb_k
        self.eps = eps

        # Alpha blending - using sigmoid for improved stability
        if learnable_alpha:
            # Initialize with logit of init_alpha for better sigmoid behavior
            self.alpha = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_alpha))))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))

        # Correlation parameters
        if low_rank:
            # Initialize with scaled random values for better gradient flow
            scale = 1.0 / (self.rank ** 0.5)
            self.corr_factors = nn.Parameter(torch.randn(2, input_size, self.rank) * scale)
        else:
            self.correlation = nn.Parameter(torch.randn(input_size, input_size))

        # Projection layers
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.output_size)

        # Normalization layers
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(input_size)
            self.layer_norm2 = nn.LayerNorm(self.hidden_size)
            self.layer_norm3 = nn.LayerNorm(self.output_size)

        # Dropout
        self.dropout = nn.Dropout(correlation_dropout) if correlation_dropout > 0 else nn.Identity()
        
        # Chebyshev polynomial coefficients - learnable weights for each order
        self.cheb_weights = nn.Parameter(torch.ones(cheb_k) / cheb_k)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with numerically stable values"""
        if self.low_rank:
            # Better orthogonal initialization for low-rank factors
            nn.init.orthogonal_(self.corr_factors[0])
            nn.init.orthogonal_(self.corr_factors[1])
        else:
            # Initialize correlation matrix to be close to identity
            nn.init.eye_(self.correlation)
            with torch.no_grad():
                # Small random perturbation
                self.correlation.data += 0.01 * torch.randn_like(self.correlation)
                # Ensure symmetry
                self.correlation.data = 0.5 * (self.correlation.data + self.correlation.data.t())

        # Initialize projection layers
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize Chebyshev weights to be normalized
        nn.init.constant_(self.cheb_weights, 1.0 / self.cheb_k)

    def get_learned_correlation(self) -> torch.Tensor:
        """Get the learned correlation matrix with symmetry preservation"""
        if self.low_rank:
            # Low-rank factorization: C = UV^T
            U, V = self.corr_factors[0], self.corr_factors[1]
            corr = torch.matmul(U, V.T)
            # Ensure symmetry
            corr = 0.5 * (corr + corr.T)
        else:
            # Full-rank case: ensure symmetry
            corr = 0.5 * (self.correlation + self.correlation.T)

        # Apply tanh to constrain values to [-1, 1]
        corr = torch.tanh(corr)
        
        # Apply dropout during training
        return self.dropout(corr) if self.training else corr

    def compute_data_correlation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute correlation from data with improved numerical stability"""
        # Center the data
        x_centered = x - x.mean(dim=1, keepdim=True)
        
        # Reshape for batch correlation computation [B, F, T]
        x_reshaped = x_centered.transpose(1, 2)
        
        # Compute norms with epsilon for stability
        norms = torch.norm(x_reshaped, dim=2, keepdim=True).clamp(min=self.eps)
        x_normalized = x_reshaped / norms
        
        # Compute batch correlation matrices
        corr_batch = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
        
        # Average across batch and clamp to valid correlation range
        return corr_batch.mean(dim=0).clamp(min=-1.0, max=1.0)

    def compute_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """Compute normalized graph Laplacian with improved stability"""
        # Create a copy and zero out diagonal (self-loops)
        A = A.clone()
        A.fill_diagonal_(0.0)
        
        # Compute degree vector (sum of adjacency matrix rows)
        deg = A.sum(dim=1)
        
        # Compute D^{-1/2} with numerical stability
        deg_inv_sqrt = torch.pow(deg.clamp(min=self.eps), -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        # Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        L = torch.eye(A.size(0), device=A.device) - torch.matmul(
            torch.matmul(D_inv_sqrt, A), D_inv_sqrt
        )
        
        return L

    def chebyshev_filter(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Apply Chebyshev polynomial filtering with learnable weights.
        x: [B, T, F]
        L: [F, F] (symmetric Laplacian)
        """
        B, T, F = x.shape
        
        # Initialize Chebyshev polynomials T_0(L) and T_1(L)
        Tx_0 = x                   # T_0(L) = I (identity)
        Tx_1 = torch.matmul(x, L)  # T_1(L) = L
        
        # Use softmax for better weight normalization
        cheb_weights_norm = torch.nn.functional.softmax(self.cheb_weights, dim=0)
        
        # Initialize output with weighted first two orders
        out = cheb_weights_norm[0] * Tx_0 + cheb_weights_norm[1] * Tx_1
        
        # Previous two terms for the recurrence relation
        prev_terms = [Tx_0, Tx_1]
        
        # Recurrence relation for higher orders: T_k(L) = 2L·T_{k-1}(L) - T_{k-2}(L)
        for k in range(2, self.cheb_k):
            # Apply recurrence with numerical stability
            Tx_k = 2 * torch.matmul(prev_terms[-1], L) - prev_terms[-2]
            
            # Prevent exploding values
            Tx_k = torch.clamp(Tx_k, min=-1e2, max=1e2)
            
            # Add weighted contribution to output
            out = out + cheb_weights_norm[k] * Tx_k
            
            # Update previous terms (only need last two)
            prev_terms = [prev_terms[-1], Tx_k]
            
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the latent correlation layer.
        x: [B, T, F] - Batch, Time/Sequence, Features
        returns: [B, T, output_size], [F, F] correlation matrix
        """
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x = self.layer_norm1(x)

        # Compute data-driven and learned correlations
        data_corr = self.compute_data_correlation(x)
        learned_corr = self.get_learned_correlation()

        # Blend correlations with sigmoid-transformed alpha
        alpha = torch.sigmoid(self.alpha)
        mixed_corr = alpha * learned_corr + (1 - alpha) * data_corr
        
        # Compute graph Laplacian from correlation matrix
        laplacian = self.compute_laplacian(mixed_corr)
        
        # Prevent eigenvalue explosion for numerical stability
        laplacian = torch.clamp(laplacian, min=-2.0, max=2.0)

        # Apply Chebyshev filtering
        x_filtered = self.chebyshev_filter(x, laplacian)
        
        # Project to higher dimension
        x_proj = self.input_proj(x_filtered)

        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x_proj = self.layer_norm2(x_proj)

        # Apply non-linearity
        x_proj = F.gelu(x_proj)
        
        # Project to output dimension
        output = self.output_proj(x_proj)

        # Apply final layer normalization if enabled
        if self.use_layer_norm:
            output = self.layer_norm3(output)
            
        return output, mixed_corr
        

class MessagePassing(nn.Module):
    """
    Generic message passing base class.
    Supports xformers-based multi-head attention over the feature dimension.
    """
    def __init__(self, input_size: int, hidden_dim: int, aggregation: str = "sum", num_heads: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.num_heads = num_heads
        self.head_dim = hidden_dim * num_heads

        # Shared node transformation
        self.message_transform = nn.Linear(input_size, hidden_dim)

        # Projections for attention mode
        if aggregation == "xformers":
            self.q_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)
            self.k_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)
            self.v_proj = nn.Linear(input_size, input_size * num_heads * self.head_dim)

    def message(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply shared linear transformation to input features.
        """
        return self.message_transform(h)  # [B, T, hidden_dim]

    def aggregate(self, messages: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages based on specified strategy.
        messages: [B, T, hidden_dim]
        graph: [F, F] for sum/mean, or attention bias for xformers
        """
        if self.aggregation == "sum":
            return torch.einsum("bth,hg->btg", messages, graph)

        
        elif self.aggregation == "mean":
            B, T, H = messages.shape  # messages: [B, T, hidden]
            F = graph.shape[0]        # graph: [F, F]

            # STEP 1 — check for existing NaNs early
            if torch.isnan(messages).any():
                raise RuntimeError("NaNs detected in messages before aggregation")

            if torch.isnan(graph).any() or torch.isinf(graph).any():
                raise RuntimeError("NaNs or Infs detected in graph before aggregation")

            # STEP 2 — build identity matrix matching dtype/device
            identity = torch.eye(F, device=graph.device, dtype=graph.dtype)

            # STEP 3 — clone and fix isolated rows
            row_sums = graph.sum(dim=1, keepdim=True)
            isolated = row_sums.squeeze() == 0
            graph_safe = graph.clone()
            if isolated.any():
                graph_safe[isolated, :] = identity[isolated, :]

            # STEP 4 — normalize (row-wise mean)
            row_sums = graph_safe.sum(dim=1, keepdim=True).clamp(min=1e-6)
            normalized_graph = graph_safe / row_sums

            # STEP 5 — apply aggregation
            out = torch.einsum("bth,hg->btg", messages, normalized_graph)

            # STEP 6 — final check
            if torch.isnan(out).any() or torch.isinf(out).any():
                # Print only shape + stats for brevity
                print("NaNs in output despite normalization:")
                print("→ messages stats:", messages.mean().item(), messages.std().item())
                print("→ normalized_graph stats:", normalized_graph.mean().item(), normalized_graph.std().item())
                print("→ out stats:", out.mean().item(), out.std().item())
                raise RuntimeError("NaNs in aggregation output")

            return out

        elif self.aggregation == "xformers":
            return self._xformers_aggregate(messages, graph)

        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

    def _xformers_aggregate(self, messages: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Apply attention over features per time step, with bias shape [B*T, 12, 4, 4].
        """
        B, T, D = messages.shape
        H = self.num_heads
        d_head = self.head_dim

        # Flatten [B, T, D] → [B*T, D]
        x_flat = messages.reshape(B * T, D)

        # Project Q/K/V to [B*T, H, D, d_head]
        q = self.q_proj(x_flat).reshape(B * T, H, D, d_head)
        k = self.k_proj(x_flat).reshape(B * T, H, D, d_head)
        v = self.v_proj(x_flat).reshape(B * T, H, D, d_head)

        # Static bias: [H, H] → [B*T, D, H, H]
        # Goal: create [12288, 12, 4, 4] with alignment-compatible memory layout

        # Step 1: allocate a larger tensor with last dim padded to 8
        # Aligned and dtype-matching bias

    
        # Call xFormers
        out = memory_efficient_attention(q, k, v)
        #out = memory_efficient_attention(q, k, v, attn_bias=bias)  # [B*T, H, D, d_head]
        out = out.permute(0, 2, 1, 3)           # [B*T, D_token, H, d_head]
        #print(f"Xformers out: {out.shape=}")
        out = out.reshape(B, T, D, -1)
        # take mean over last dim
        out = out.mean(dim=-1)                 # [B, T, D]

        return out


    def forward(self, h: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement forward pass.")


class GraphConv(MessagePassing):
    def __init__(self, input_size, output_size, hidden_dim, aggregation="sum", dropout=0.1):
        super().__init__(input_size, hidden_dim, aggregation)
        self.update_fn = nn.Sequential(
            nn.Linear(input_size + hidden_dim, input_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.output_proj = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, graph):
        msg = self.message(x)                     # [B, T, hidden]
        agg = self.aggregate(msg, graph)          # [B, T, hidden]
        combined = torch.cat([x, agg], dim=-1)    # [B, T, in+hidden]
        updated = self.update_fn(combined)        # [B, T, in]
        return self.norm(self.output_proj(updated))

class AttGraphConv(MessagePassing):
    def __init__(self, input_size, output_size, hidden_dim, num_heads=1, dropout=0.1):
        super().__init__(input_size, hidden_dim)
        self.num_heads = num_heads
        self.output_proj = nn.Linear(input_size, output_size)
        self.attn_q = nn.Linear(input_size, hidden_dim)
        self.attn_k = nn.Linear(input_size, hidden_dim)
        self.update_fn = nn.Sequential(
            nn.Linear(input_size + hidden_dim, input_size),
            nn.GELU(),
            nn.Dropout(dropout)
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

from xformers.ops import memory_efficient_attention, AttentionOpBase

from xformers.ops import memory_efficient_attention, AttentionOpBase
import torch.nn.functional as F

class XFormerAttGraphConv(MessagePassing):
    def __init__(self, input_size, output_size, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__(input_size, hidden_dim)
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(input_size, hidden_dim)
        self.k_proj = nn.Linear(input_size, hidden_dim)
        self.v_proj = nn.Linear(input_size, hidden_dim)

        self.update_fn = nn.Sequential(
            nn.Linear(input_size + hidden_dim, input_size),
            nn.GELU(),
            nn.Dropout(dropout)
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
        x_feat = x.transpose(1, 2)  # [B, F, T]

        # Project to Q, K, V
        q = self.q_proj(x_feat).reshape(B, F, H, D).transpose(1, 2)  # [B, H, F, D]
        k = self.k_proj(x_feat).reshape(B, F, H, D).transpose(1, 2)  # [B, H, F, D]
        v = self.v_proj(x_feat).reshape(B, F, H, D).transpose(1, 2)  # [B, H, F, D]

        # Modulate attention scores using graph
        # graph: [F, F] → [1, 1, F, F] → broadcasted
        attn_bias = torch.tanh(graph).unsqueeze(0).unsqueeze(0)  # [1, 1, F, F]

        # Use xformers efficient attention
        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)  # [B, H, F, D]
        out = out.transpose(1, 2).reshape(B, F, H * D)  # [B, F, hidden_dim]

        # Bring back to [B, T, hidden_dim]
        out = out.transpose(1, 2)  # [B, T, hidden_dim]

        # Update with residual information
        combined = torch.cat([x, out], dim=-1)  # [B, T, in + hidden]
        updated = self.update_fn(combined)      # [B, T, input_size]
        return self.norm(self.output_proj(updated))  # [B, T, output_size]
import torch
import torch.nn as nn
from typing import Optional, List, Literal
from xformers.ops import memory_efficient_attention

# Assume these are defined elsewhere and imported:
# - LatentCorrelationLayer
# - GraphConv
# - AttGraphConv
# - XFormerAttGraphConv
# - JumpKnowledge

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
        aggregation: str = 'sum',
        dropout: float = 0.1,
        residual: bool = True,
        strategy: Literal['vanilla', 'attn', 'xformers'] = 'vanilla',
        jk_mode: Literal['last', 'sum', 'max', 'concat', 'lstm', 'none'] = 'none'
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
            correlation_dropout=dropout
        )

        # Message passing layers
        self.message_passing_layers = nn.ModuleList([
            self._create_layer(strategy, input_size, self.hidden_size, aggregation, dropout)
            for _ in range(num_passes)
        ])

        self.jk_mode = jk_mode
        if jk_mode != 'none':
            # Jump knowledge module
            self.jump_knowledge = JumpKnowledge(
                mode=jk_mode,
                hidden_size=self.input_size,
                output_size=input_size
            )

        self.norm = nn.LayerNorm(output_size)

    def _create_layer(
        self,
        strategy: str,
        input_size: int,
        hidden_size: int,
        aggregation: str,
        dropout: float
    ) -> nn.Module:
        if strategy == 'vanilla':
            return GraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                aggregation=aggregation,
                dropout=dropout
            )
        elif strategy == 'attn':
            return AttGraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                dropout=dropout
            )
        elif strategy == 'xformers':
            return XFormerAttGraphConv(
                input_size=input_size,
                output_size=hidden_size,
                hidden_dim=hidden_size,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        corr_features, correlation = self.correlation_layer(x)

        h = corr_features
        outputs: List[torch.Tensor] = []

        for layer in self.message_passing_layers:
            h = layer(h, correlation)
            outputs.append(h)

        if self.jk_mode != 'none':
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
        mode: Literal['last', 'sum', 'max', 'concat', 'lstm'] = 'concat',
        hidden_size: int = None,
        output_size: int = None,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out_proj = None  # lazy initialization

        if self.mode == 'lstm':
            assert hidden_size is not None
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.out_proj = nn.Identity() if hidden_size == output_size else nn.Linear(hidden_size, output_size)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        if self.mode == 'last':
            return xs[-1]

        elif self.mode == 'sum':
            return torch.stack(xs, dim=0).sum(dim=0)

        elif self.mode == 'max':
            return torch.stack(xs, dim=0).max(dim=0)[0]

        elif self.mode == 'concat':
            x_cat = torch.cat(xs, dim=-1)  # [B, T, D * num_layers]
            if self.out_proj is None:
                input_dim = x_cat.size(-1)
                self.out_proj = nn.Linear(input_dim, self.output_size).to(x_cat.device)
            return self.out_proj(x_cat)

        elif self.mode == 'lstm':
            B, T, D = xs[0].shape
            x_seq = torch.stack(xs, dim=1).reshape(B * T, len(xs), D)
            lstm_out, _ = self.lstm(x_seq)
            final = lstm_out[:, -1, :].reshape(B, T, -1)
            return self.out_proj(final)

        else:
            raise ValueError(f"Unsupported JK mode: {self.mode}")
