import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import contextlib


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
    Designed to work on [B, T, F] inputs with [F, F] feature graphs.
    """
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        aggregation: str = "sum"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        # Shared node transformation
        self.message_transform = nn.Linear(input_size, hidden_dim)

    def aggregate(self, messages: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages from neighbors (default: weighted sum).
        messages: [B, T, hidden_dim]
        graph: [F, F] (correlation/adjacency matrix)
        returns: [B, T, hidden_dim]
        """
        if self.aggregation == "sum":
            return torch.einsum("bfh,ij->bfj", messages, graph)
        elif self.aggregation == "mean":
            norm = graph.sum(dim=0, keepdim=True).clamp(min=1e-6)
            return torch.einsum("bfh,ij->bfj", messages, graph / norm)
        elif self.aggregation == "max":
            raise NotImplementedError("Max aggregation not supported yet.")
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

    def message(self, h: torch.Tensor) -> torch.Tensor:
        return self.message_transform(h)  # [B, T, hidden_dim]

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


class LatentGraphNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Optional[int] = None,
        correlation_hidden_size: Optional[int] = None,
        low_rank: bool = True,
        rank: Optional[int] = None,
        num_passes: int = 2,
        aggregation: str = 'sum',
        dropout: float = 0.1,
        edge_threshold: float = 0.01,
        residual: bool = True,
        strategy: str = 'vanilla'  # or 'attn'
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size or max(input_size, output_size)
        self.residual = residual

        # Correlation learning
        self.correlation_layer = LatentCorrelationLayer(
            input_size=input_size,
            output_size=input_size,
            hidden_size=correlation_hidden_size,
            low_rank=low_rank,
            rank=rank,
            correlation_dropout=dropout
        )

        # Select message passing strategy
        if strategy == 'vanilla':
            self.message_passing = GraphConv(
                input_size=input_size,
                output_size=self.hidden_size,
                hidden_dim=self.hidden_size,
                aggregation=aggregation,
                dropout=dropout
            )
        elif strategy == 'attn':
            self.message_passing = AttGraphConv(
                input_size=input_size,
                output_size=self.hidden_size,
                hidden_dim=self.hidden_size,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported graph conv strategy: {strategy}")

        self.output_proj = (
            nn.Linear(self.hidden_size, output_size)
            if self.hidden_size != output_size
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        corr_features, correlation = self.correlation_layer(x)
        mp_features = self.message_passing(corr_features, correlation)
        output = self.output_proj(mp_features)

        if self.residual and x.shape[-1] == output.shape[-1]:
            output = output + x

        return self.norm(output)
