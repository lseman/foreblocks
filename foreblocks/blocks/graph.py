import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional


class OptimizedGraphConv(nn.Module):
    """Base class for optimized graph convolution operations."""
    def __init__(self):
        super().__init__()
        self._filter_cache = {}
    
    def _clear_cache(self):
        """Clear the filter cache."""
        self._filter_cache.clear()


class SGConv(nn.Module):
    """
    Spectral Graph Convolutional layer for time series.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        k: Order of Chebyshev polynomials (default: 3)
        include_feature_edges: Whether to include edges between features (default: True)
        learned_adjacency: Whether to learn the adjacency matrix (default: True) 
        cache_size: Maximum sequence length to cache (default: 200)
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        k: int = 3,
        include_feature_edges: bool = True,
        learned_adjacency: bool = True,
        cache_size: int = 200
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.include_feature_edges = include_feature_edges
        self.cache_size = cache_size

        # Weights and bias
        self.weight = nn.Parameter(torch.Tensor(k, input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))

        # Learnable adjacency matrix
        if learned_adjacency:
            self.adj_param = nn.Parameter(torch.randn(input_size, input_size))
        else:
            self.register_parameter('adj_param', None)

        # Cache for graph filters
        self._filter_cache: Dict[int, List[torch.Tensor]] = {}
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
        if self.adj_param is not None:
            nn.init.xavier_normal_(self.adj_param)
            with torch.no_grad():
                self.adj_param.diagonal().fill_(2.0)

    def _build_adjacency_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build the adjacency matrix for the graph."""
        n = seq_len * self.input_size
        
        # Build sparse representation
        indices = []
        values = []
        
        # Temporal edges
        for i in range(seq_len - 1):
            for f in range(self.input_size):
                idx = i * self.input_size + f
                next_idx = (i + 1) * self.input_size + f
                indices.extend([[idx, next_idx], [next_idx, idx]])
                values.extend([1.0, 1.0])
        
        # Feature edges
        if self.include_feature_edges and self.adj_param is not None:
            feature_adj = torch.sigmoid(self.adj_param)
            
            for i in range(seq_len):
                for f1 in range(self.input_size):
                    for f2 in range(self.input_size):
                        if f1 != f2 and feature_adj[f1, f2] > 0.01:
                            idx1 = i * self.input_size + f1
                            idx2 = i * self.input_size + f2
                            indices.append([idx1, idx2])
                            values.append(feature_adj[f1, f2].item())
        
        # Self-loops
        for i in range(n):
            indices.append([i, i])
            values.append(1.0)
        
        # Create sparse tensor and convert to dense
        if indices:
            indices = torch.tensor(indices, device=device).t()
            values = torch.tensor(values, device=device)
            adj = torch.sparse_coo_tensor(indices, values, (n, n))
            return adj.to_dense()
        else:
            return torch.eye(n, device=device)

    def _compute_normalized_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        """Compute normalized Laplacian matrix."""
        d = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(d + 1e-10, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        return torch.eye(adj.size(0), device=adj.device) - torch.matmul(
            torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt
        )

    def _compute_chebyshev_polynomials(self, L: torch.Tensor) -> List[torch.Tensor]:
        """Compute Chebyshev polynomials up to order k."""
        n = L.size(0)
        device = L.device
        
        # Estimate spectral norm
        with torch.no_grad():
            x = torch.randn(n, 1, device=device)
            for _ in range(3):
                x = L @ x
                x = x / (torch.norm(x) + 1e-10)
            norm = torch.norm(L @ x) / (torch.norm(x) + 1e-10)
            norm = torch.clamp(norm, min=1.0)
        
        # Rescale Laplacian
        L_scaled = 2.0 * L / norm - torch.eye(n, device=device)
        
        # Compute polynomials
        polynomials = [torch.eye(n, device=device), L_scaled]
        for i in range(2, self.k):
            next_poly = 2.0 * torch.matmul(L_scaled, polynomials[-1]) - polynomials[-2]
            polynomials.append(next_poly)
            
        return polynomials

    def _get_graph_filter(self, seq_len: int) -> List[torch.Tensor]:
        """Get or compute the graph filter polynomials."""
        device = self.weight.device
        
        # Return from cache if available
        if seq_len in self._filter_cache:
            return self._filter_cache[seq_len]
        
        # Compute new filter
        adj = self._build_adjacency_matrix(seq_len, device)
        L = self._compute_normalized_laplacian(adj)
        polynomials = self._compute_chebyshev_polynomials(L)
        
        # Cache result if appropriate
        if self.cache_size > 0 and seq_len <= self.cache_size:
            if len(self._filter_cache) >= 5:
                min_len = min(self._filter_cache.keys())
                del self._filter_cache[min_len]
            self._filter_cache[seq_len] = polynomials
            
        return polynomials

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SGConv layer."""
        batch_size, seq_len, _ = x.shape
        
        # Get graph filter
        cheb_polynomials = self._get_graph_filter(seq_len)
        
        # Reshape input
        x_reshaped = x.reshape(batch_size, -1)
        
        # Apply filters
        outputs = []
        for k in range(self.k):
            x_k = torch.matmul(x_reshaped, cheb_polynomials[k])
            x_k = x_k.reshape(batch_size, seq_len, self.input_size)
            x_k = torch.matmul(x_k, self.weight[k])
            outputs.append(x_k)
        
        # Sum and add bias
        out = torch.stack(outputs).sum(dim=0) + self.bias
        
        return out


class AdaptiveGraphConv(OptimizedGraphConv):
    """
    Adaptive Graph Convolution that dynamically adjusts the graph structure.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        k: Order of Chebyshev polynomials (default: 2)
        heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
        alpha: LeakyReLU negative slope (default: 0.2)
        sparsity_threshold: Threshold for pruning connections (default: 0.01)
        cache_size: Size of the cache for graph structures (default: 5)
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        k: int = 2,
        heads: int = 4, 
        dropout: float = 0.1,
        alpha: float = 0.2,
        sparsity_threshold: float = 0.01,
        cache_size: int = 5
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.sparsity_threshold = sparsity_threshold
        self.cache_size = cache_size
        
        # Head dimensions
        self.head_output_size = output_size // heads
        assert self.head_output_size * heads == output_size, "Output size must be divisible by heads"
        
        # Parameters
        self.weight = nn.Parameter(torch.Tensor(k, heads, input_size, self.head_output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        
        # Attention mechanisms with reduced parameter count
        self.temporal_query = nn.Parameter(torch.Tensor(heads, self.head_output_size))
        self.temporal_key = nn.Parameter(torch.Tensor(heads, self.head_output_size))
        
        # Feature attention using low-rank approximation
        rank = min(16, input_size // 2)
        self.feature_factor = nn.Parameter(torch.Tensor(heads, 2, input_size, rank))
        
        # Activation and dropout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
        gain = nn.init.calculate_gain('leaky_relu', self.alpha)
        nn.init.xavier_normal_(self.temporal_query, gain=gain)
        nn.init.xavier_normal_(self.temporal_key, gain=gain)
        nn.init.xavier_normal_(self.feature_factor, gain=gain)
        
    def _compute_feature_attention(self, x: torch.Tensor, head: int) -> torch.Tensor:
        """Compute attention between features using low-rank approximation."""
        # Use low-rank factorization
        U = self.feature_factor[head, 0]
        V = self.feature_factor[head, 1]
        
        # Compute attention weights
        logits = torch.matmul(U, V.t())
        attn_logits = self.leakyrelu(logits)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # Apply dropout during training
        if self.training:
            attn_weights = self.attn_dropout(attn_weights)
        
        # Apply sparsity threshold
        attn_weights = F.threshold(attn_weights, self.sparsity_threshold, 0.0)
        
        return attn_weights
    
    def _compute_temporal_attention(self, x: torch.Tensor, head: int) -> torch.Tensor:
        """Compute attention between time steps."""
        # Compute queries and keys
        queries = torch.matmul(x, self.temporal_query[head].unsqueeze(1))
        keys = torch.matmul(x, self.temporal_key[head].unsqueeze(1))
        
        # Compute attention logits
        attn_logits = queries + keys.transpose(1, 2)
        attn_logits = self.leakyrelu(attn_logits)
        
        # Average across batch dimension
        attn_logits = attn_logits.mean(dim=0)
        
        # Apply softmax and threshold
        attn_weights = F.softmax(attn_logits, dim=1)
        if self.training:
            attn_weights = self.attn_dropout(attn_weights)
        attn_weights = F.threshold(attn_weights, self.sparsity_threshold, 0.0)
        
        return attn_weights
    
    def _build_sparse_graph(self, 
                           feature_attention: torch.Tensor, 
                           temporal_attention: torch.Tensor,
                           seq_len: int, 
                           device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build sparse adjacency matrix."""
        indices = []
        values = []
        
        # Add temporal connections
        temp_nonzero = temporal_attention > self.sparsity_threshold
        temp_indices = temp_nonzero.nonzero(as_tuple=True)
        for idx in range(len(temp_indices[0])):
            t1, t2 = temp_indices[0][idx].item(), temp_indices[1][idx].item()
            if t1 != t2:  # Skip self-loops for now
                weight = temporal_attention[t1, t2].item()
                for f in range(self.input_size):
                    idx1 = t1 * self.input_size + f
                    idx2 = t2 * self.input_size + f
                    indices.append([idx1, idx2])
                    values.append(weight)
        
        # Add feature connections
        feat_nonzero = feature_attention > self.sparsity_threshold
        feat_indices = feat_nonzero.nonzero(as_tuple=True)
        for idx in range(len(feat_indices[0])):
            f1, f2 = feat_indices[0][idx].item(), feat_indices[1][idx].item()
            if f1 != f2:  # Skip self-loops for now
                weight = feature_attention[f1, f2].item()
                for t in range(seq_len):
                    idx1 = t * self.input_size + f1
                    idx2 = t * self.input_size + f2
                    indices.append([idx1, idx2])
                    values.append(weight)
        
        # Add self-loops
        n = seq_len * self.input_size
        for i in range(n):
            indices.append([i, i])
            values.append(1.0)
            
        # Convert to tensors
        if indices:
            indices_tensor = torch.tensor(indices, device=device).t()
            values_tensor = torch.tensor(values, device=device)
            return indices_tensor, values_tensor
        else:
            return torch.tensor([[0, 0]], device=device).t(), torch.tensor([1.0], device=device)
    
    def _compute_normalized_laplacian(self, 
                                     indices: torch.Tensor, 
                                     values: torch.Tensor, 
                                     n: int, 
                                     device: torch.device) -> torch.Tensor:
        """Compute normalized Laplacian from sparse adjacency."""
        # Create sparse adjacency matrix
        adj_sparse = torch.sparse_coo_tensor(indices, values, (n, n))
        adj = adj_sparse.to_dense()
        
        # Compute degree matrix
        d = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(d + 1e-10, -0.5)
        
        # Compute normalized Laplacian
        d_inv_sqrt_mat = torch.diag(d_inv_sqrt)
        L = torch.eye(n, device=device) - torch.matmul(
            torch.matmul(d_inv_sqrt_mat, adj), d_inv_sqrt_mat
        )
        
        return L
    
    def _compute_cheb_polynomials(self, L: torch.Tensor) -> List[torch.Tensor]:
        """Compute Chebyshev polynomials."""
        n = L.size(0)
        device = L.device
        
        # Estimate spectral norm
        with torch.no_grad():
            x = torch.randn(n, 1, device=device)
            for _ in range(3):
                x = torch.matmul(L, x)
                x = x / (torch.norm(x) + 1e-10)
            spectral_norm = torch.matmul(x.t(), torch.matmul(L, x)).item()
            spectral_norm = max(abs(spectral_norm), 1.0)
        
        # Rescale
        L_scaled = 2.0 * L / spectral_norm - torch.eye(n, device=device)
        
        # Compute polynomials
        polynomials = [torch.eye(n, device=device), L_scaled]
        for i in range(2, self.k):
            next_poly = 2.0 * torch.matmul(L_scaled, polynomials[-1]) - polynomials[-2]
            polynomials.append(next_poly)
        
        return polynomials
    
    def _get_cache_key(self, seq_len: int, head: int) -> str:
        """Generate a cache key."""
        return f"{seq_len}_{head}"
    
    def _build_head_filters(self, x: torch.Tensor, head: int) -> List[torch.Tensor]:
        """Build filters for a specific head with caching."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        n = seq_len * self.input_size
        
        # Check cache first
        cache_key = self._get_cache_key(seq_len, head)
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]
        
        # Build attention matrices
        feature_attention = self._compute_feature_attention(x, head)
        x_proj = torch.matmul(x, self.weight[0, head])
        temporal_attention = self._compute_temporal_attention(x_proj, head)
        
        # Build graph and compute polynomials
        indices, values = self._build_sparse_graph(
            feature_attention, temporal_attention, seq_len, device
        )
        L = self._compute_normalized_laplacian(indices, values, n, device)
        polynomials = self._compute_cheb_polynomials(L)
        
        # Cache result
        if self.cache_size > 0:
            if len(self._filter_cache) >= self.cache_size:
                keys = list(self._filter_cache.keys())
                del self._filter_cache[keys[0]]
            self._filter_cache[cache_key] = polynomials
        
        return polynomials
    
    def _process_head(self, x: torch.Tensor, x_flat: torch.Tensor, head: int) -> torch.Tensor:
        """Process input through a single attention head."""
        batch_size, seq_len, _ = x.shape
        
        # Get filters for this head
        filters = self._build_head_filters(x, head)
        
        # Process with Chebyshev filters
        head_output = torch.zeros(
            batch_size, seq_len, self.head_output_size, 
            device=x.device, dtype=x.dtype
        )
        
        for k in range(self.k):
            x_k = torch.matmul(x_flat, filters[k])
            x_k = x_k.reshape(batch_size, seq_len, self.input_size)
            x_k = torch.matmul(x_k, self.weight[k, head])
            head_output += x_k
        
        return head_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optimized computation."""
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(batch_size, -1)
        
        # Process each head
        head_outputs = []
        for head in range(self.heads):
            head_output = self._process_head(x, x_flat, head)
            head_outputs.append(head_output)
        
        # Concatenate head outputs
        out = torch.cat(head_outputs, dim=2)
        
        # Add bias
        out = out + self.bias
        
        return out


class GraphConvMixture(nn.Module):
    """
    Memory-efficient Mixture of Graph Convolutions.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        num_experts: Number of specialist graph convolutions (default: 3)
        hidden_size: Hidden size for gate network (default: 16)
        k: Maximum order of Chebyshev polynomials (default: 3)
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int = 3,
        hidden_size: int = 16,
        k: int = 3
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.k = k
        
        # Create experts
        self.experts = nn.ModuleList([
            # Expert 1: Global patterns
            SGConv(
                input_size=input_size,
                output_size=output_size,
                k=k,
                include_feature_edges=True,
                learned_adjacency=True
            ),
            # Expert 2: Local patterns
            SGConv(
                input_size=input_size,
                output_size=output_size,
                k=2,
                include_feature_edges=False,
                learned_adjacency=False
            ),
            # Expert 3: Feature interactions
            AdaptiveGraphConv(
                input_size=input_size,
                output_size=output_size,
                k=2,
                heads=2
            )
        ])
        
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Output projection and normalization
        self.output_proj = nn.Linear(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory-efficient processing."""
        batch_size, seq_len, _ = x.shape
        
        # Compute gates
        seq_repr = x.mean(dim=1)
        gate_logits = self.gate_network(seq_repr)
        gates = F.softmax(gate_logits, dim=1)
        
        # Initialize output tensor
        combined = torch.zeros(
            batch_size, seq_len, self.output_size,
            device=x.device, dtype=x.dtype
        )
        
        # Process each expert separately
        for i, expert in enumerate(self.experts):
            expert_gates = gates[:, i]
            expert_output = expert(x)
            weighted_output = expert_gates.view(-1, 1, 1) * expert_output
            combined += weighted_output
        
        # Apply post-processing
        output = self.output_proj(combined)
        output = self.layer_norm(output)
        
        return output


class GraphConvFactory:
    """Factory for creating graph convolution layers."""
    @staticmethod
    def create(
        name: str,
        input_size: int,
        output_size: int,
        **kwargs
    ) -> nn.Module:
        """Create a graph convolution layer."""
        if name.lower() == 'sgconv':
            return SGConv(input_size, output_size, **kwargs)
        elif name.lower() == 'adaptive':
            return AdaptiveGraphConv(input_size, output_size, **kwargs)
        elif name.lower() == 'mixture':
            return GraphConvMixture(input_size, output_size, **kwargs)
        else:
            raise ValueError(f"Unknown graph convolution type: {name}")


class GraphConvProcessor(nn.Module):
    """
    Graph convolution preprocessor for time series data.
    
    Args:
        in_channels: Number of input features
        out_channels: Number of output features
        conv_type: Type of graph convolution ('sgconv', 'adaptive', 'mixture')
        **kwargs: Additional parameters for graph convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = 'mixture',
        **kwargs
    ):
        super().__init__()
        
        # Create graph convolution layer
        self.graph_conv = GraphConvFactory.create(
            name=conv_type,
            input_size=in_channels,
            output_size=out_channels,
            **kwargs
        )
        
        # Activation and normalization
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.graph_conv(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        return x
    

################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LatentCorrelationLayer(nn.Module):
    """
    High-performance Latent Correlation Layer for multivariate time series.
    
    This optimized implementation captures pairwise feature correlations with:
    - Vectorized operations to eliminate loops
    - Efficient correlation computation
    - Memory-optimized tensor operations
    - Low-rank correlation approximation option
    - Batch-parallel processing
    
    Args:
        input_size: Number of input features
        output_size: Number of output features (same as input_size by default)
        hidden_size: Size of hidden representation (default: 2*input_size)
        learnable_alpha: Whether alpha is learnable (default: True)
        init_alpha: Initial value for alpha parameter (default: 0.5)
        use_layer_norm: Whether to use layer normalization (default: True)
        low_rank: Whether to use low-rank approximation for correlation (default: False)
        rank: Rank for low-rank approximation (default: input_size//4)
        correlation_dropout: Dropout rate for correlation matrix (default: 0.0)
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
        correlation_dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size or (2 * input_size)
        self.use_layer_norm = use_layer_norm
        self.low_rank = low_rank
        self.correlation_dropout = correlation_dropout
        
        # Alpha parameter controls how much of the learned correlation to use
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))
        
        # Learnable correlation matrix
        if low_rank:
            # Low-rank approximation: Correlation ≈ U * V^T
            self.rank = rank or max(1, input_size // 4)
            self.corr_factors = nn.Parameter(torch.Tensor(2, input_size, self.rank))
        else:
            # Full correlation matrix
            self.correlation = nn.Parameter(torch.Tensor(input_size, input_size))
        
        # Projections to transform input and output
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.output_size)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(input_size)
            self.layer_norm2 = nn.LayerNorm(self.hidden_size)
            self.layer_norm3 = nn.LayerNorm(self.output_size)
        
        # Dropout for regularization
        if correlation_dropout > 0:
            self.dropout = nn.Dropout(correlation_dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with optimized values."""
        if self.low_rank:
            # Initialize factors to approximate identity matrix
            # U and V initialized so that U*V^T ≈ Identity
            nn.init.orthogonal_(self.corr_factors[0])
            nn.init.orthogonal_(self.corr_factors[1])
            
            # Scale to approximate identity matrix
            with torch.no_grad():
                # Compute U*V^T and adjust to make diagonal dominant
                approx = torch.matmul(self.corr_factors[0], self.corr_factors[1].t())
                diag_mean = torch.diagonal(approx).mean()
                if diag_mean != 0:
                    self.corr_factors.data *= 1.0 / diag_mean
        else:
            # Initialize correlation to identity with small random noise
            nn.init.eye_(self.correlation)
            with torch.no_grad():
                self.correlation.data += 0.01 * torch.randn_like(self.correlation)
                # Ensure it's symmetric
                self.correlation.data = 0.5 * (
                    self.correlation.data + self.correlation.data.t()
                )
        
        # Initialize projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def get_learned_correlation(self) -> torch.Tensor:
        """Get the current learned correlation matrix."""
        if self.low_rank:
            # Reconstruct from factors: Correlation ≈ U * V^T
            U, V = self.corr_factors[0], self.corr_factors[1]
            
            # Compute product and ensure symmetry
            corr = torch.matmul(U, V.t())
            corr = 0.5 * (corr + corr.t())
        else:
            # Get full correlation matrix and ensure symmetry
            corr = 0.5 * (self.correlation + self.correlation.t())
        
        # Apply tanh to ensure values in [-1, 1]
        corr = torch.tanh(corr)
        
        # Apply dropout during training if enabled
        if self.correlation_dropout > 0 and self.training:
            corr = self.dropout(corr)
        
        return corr
    
    def compute_data_correlation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute data-driven correlation matrix efficiently.
        
        Uses vectorized operations to eliminate loops.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Correlation matrix [input_size, input_size]
        """
        # Center data
        x_centered = x - x.mean(dim=1, keepdim=True)
        
        # Reshape to [batch_size, input_size, seq_len]
        x_reshaped = x_centered.transpose(1, 2)
        
        # Compute norms for each feature, shape: [batch_size, input_size, 1]
        norms = torch.norm(x_reshaped, dim=2, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero
        
        # Normalize features, shape: [batch_size, input_size, seq_len]
        x_normalized = x_reshaped / norms
        
        # Compute correlation matrices for entire batch at once
        # Shape: [batch_size, input_size, input_size]
        batch_correlations = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
        
        # Average correlations across batch
        correlation = batch_correlations.mean(dim=0)
        
        # Ensure values are in [-1, 1]
        correlation = torch.clamp(correlation, min=-1.0, max=1.0)
        
        return correlation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass with vectorized operations.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer normalization if enabled (improves stability and convergence)
        if self.use_layer_norm:
            x_norm = self.layer_norm1(x)
        else:
            x_norm = x
        
        # Compute data-driven correlation efficiently 
        data_corr = self.compute_data_correlation(x_norm)
        
        # Get learned correlation matrix
        learned_corr = self.get_learned_correlation()
        
        # Mix correlations using alpha with clamping for stability
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        mixed_corr = alpha_clamped * learned_corr + (1 - alpha_clamped) * data_corr
        
        # Apply correlation directly with batched operations
        # Reshape for efficient matrix multiplication
        x_flat = x_norm.reshape(-1, self.input_size)
        
        # Apply correlation
        x_corr = torch.matmul(x_flat, mixed_corr)
        
        # Reshape back to original dimensions
        x_corr = x_corr.reshape(batch_size, seq_len, self.input_size)
        
        # Apply non-linear transformation
        x_proj = self.input_proj(x_corr)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x_proj = self.layer_norm2(x_proj)
        
        # Apply activation
        x_proj = F.gelu(x_proj)
        
        # Apply output projection
        output = self.output_proj(x_proj)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            output = self.layer_norm3(output)
        
        return output

class SimpleStemGNNProcessor(nn.Module):
    """
    Simplified StemGNN-inspired processor compatible with foreblocks.
    
    Args:
        in_channels: Number of input features
        out_channels: Number of output features
        graph_conv_type: Type of graph convolution ('sgconv', 'adaptive', or 'mixture')
        use_layer_norm: Whether to use layer normalization (default: True)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        graph_conv_type: str = 'sgconv',
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Import needed modules
        
        # Latent correlation layer
        self.latent_corr = LatentCorrelationLayer(
            input_size=in_channels,
            output_size=in_channels,
            use_layer_norm=use_layer_norm
        )
        
        # Graph convolution
        self.graph_conv = GraphConvFactory.create(
            name=graph_conv_type,
            input_size=in_channels,
            output_size=out_channels
        )
        
        # Optional output normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_channels)
        else:
            self.layer_norm = nn.Identity()
        
        # Activation
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply latent correlation
        x = self.latent_corr(x)
        
        # Apply graph convolution
        x = self.graph_conv(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply activation
        x = self.activation(x)
        
        return x


# Example usage
# stemgnn_preprocessor = SimpleStemGNNProcessor(
#     in_channels=input_size,
#     out_channels=hidden_size,
#     graph_conv_type='sgconv'
# )
# 
# model = ForecastingModel(
#     encoder=LSTMEncoder(hidden_size, hidden_size, num_layers),
#     decoder=LSTMDecoder(output_size, hidden_size, output_size, num_layers),
#     target_len=target_len,
#     forecasting_strategy="seq2seq",
#     teacher_forcing_ratio=0.5,
#     output_size=output_size,
#     attention_module=attention_module,
#     input_preprocessor=stemgnn_preprocessor,
#     input_skip_connection=False,
# )