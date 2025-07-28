"""
Multi-Method Attention Layer with Optimized Backends

This module provides a flexible attention layer that supports various attention methods
with different backend optimizations for efficient computation.

Attention Methods:
- 'dot': Basic dot-product attention with softmax normalization
- 'mha': Multi-head attention (standard transformer attention)
- 'prob': Probabilistic attention using batch matrix multiplication
- 'temporal': Time-aware attention with temporal bias integration
- 'multiscale': Multi-scale attention with dilated temporal processing
- 'autocorr': Auto-correlation attention (Autoformer-style with FFT)

Backend Optimizations:
- 'torch': PyTorch's native scaled dot-product attention
- 'xformers': Memory-efficient attention using xformers library
- 'flash': FlashAttention for faster GPU computation

Dependencies:
- torch: Core PyTorch functionality
- xformers (optional): For memory-efficient attention backend
- flash_attn (optional): For FlashAttention backend
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional imports with fallback handling
try:
    import xformers
    import xformers.ops as xops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

try:
    import flash_attn
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class AttentionLayer(nn.Module):
    """
    Flexible attention layer supporting multiple attention methods and backend optimizations.
    
    Args:
        decoder_hidden_size (int): Hidden size of decoder
        encoder_hidden_size (int, optional): Hidden size of encoder. Defaults to decoder_hidden_size
        method (str): Attention method - 'dot', 'mha', 'prob', 'temporal', 'multiscale', 'autocorr'
        backend (str): Backend optimization - 'torch', 'xformers', 'flash'. Default: 'torch'
        nhead (int): Number of attention heads for multi-head methods. Default: 4
        dropout (float): Dropout probability. Default: 0.1
        time_embed_dim (int): Dimension for temporal embeddings (temporal method). Default: 8
        num_scales (int): Number of scales for multiscale method. Default: 3
        verbose (bool): Whether to print initialization info. Default: True
    """
    
    # Cache valid methods and backends as class attributes
    _VALID_METHODS = frozenset({"dot", "mha", "prob", "temporal", "multiscale", "autocorr"})
    _VALID_BACKENDS = frozenset({"torch", "xformers", "flash"})
    _MHA_METHODS = frozenset({"mha", "multiscale", "autocorr"})
    
    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: Optional[int] = None,
        method: str = "dot",
        attention_backend: str = "torch",
        nhead: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 8,
        num_scales: int = 3,
        verbose: bool = True,
    ):
        super().__init__()
        
        # Validate and store core parameters
        backend = attention_backend.lower()
        self._validate_inputs(method, backend, decoder_hidden_size, nhead)
        
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size or decoder_hidden_size
        self.method = method
        self.backend = backend
        self.nhead = nhead
        self.head_dim = decoder_hidden_size // nhead
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)  # Pre-compute scale factor
        
        # Build components efficiently
        self.dropout = nn.Dropout(dropout)
        self._build_projections()
        self._build_method_specific_layers(time_embed_dim, num_scales)
        
        if verbose:
            self._print_initialization_info()
    
    def _validate_inputs(self, method: str, backend: str, hidden_size: int, nhead: int):
        """Validate initialization parameters with efficient checks."""
        if method not in self._VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from {self._VALID_METHODS}")
        if backend not in self._VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Choose from {self._VALID_BACKENDS}")
        if hidden_size % nhead != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by nhead {nhead}")
        if backend == "flash" and not HAS_FLASH_ATTN:
            raise ImportError("FlashAttention not available but requested")
        if backend == "xformers" and not HAS_XFORMERS:
            raise ImportError("xformers not available but requested")
    
    def _build_projections(self):
        """Build core projection layers with optimized initialization."""
        # Combined layer for final output
        self.combined_layer = nn.Linear(
            self.decoder_hidden_size * 2, self.decoder_hidden_size
        )
        
        # Encoder projection - use Identity when possible to avoid unnecessary computation
        self.encoder_projection = (
            nn.Identity() if self.decoder_hidden_size == self.encoder_hidden_size
            else nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        )
        
        # Multi-head attention projections - only create when needed
        if self.method in self._MHA_METHODS:
            self.q_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.k_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)  
            self.v_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.out_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
        
        # Enhanced dot attention with projections for backend optimization
        elif self.method == "dot" and self.backend != "torch":
            self.q_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.k_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.v_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
        
        # Flash attention specific projection
        if self.backend == "flash":
            self.context_proj = nn.Linear(self.nhead * self.head_dim, self.decoder_hidden_size)
    
    def _build_method_specific_layers(self, time_embed_dim: int, num_scales: int):
        """Build method-specific layers efficiently."""
        if self.method == "temporal":
            self.time_bias = nn.Linear(time_embed_dim, self.decoder_hidden_size)
        
        elif self.method == "multiscale":
            # Use single ModuleList for efficiency
            self.scale_projections = nn.ModuleList([
                nn.ModuleDict({
                    'k': nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False),
                    'v': nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
                }) for _ in range(num_scales)
            ])
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.dilations = tuple(2**i for i in range(num_scales))  # Tuple for immutability
            self.scale_out_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
    
    def _print_initialization_info(self):
        """Print initialization information."""
        print(f"[Attention] Method: {self.method}, Backend: {self.backend}")
        if HAS_XFORMERS:
            print(f"[Attention] xformers version: {xformers.__version__}")
        if HAS_FLASH_ATTN:
            print(f"[Attention] FlashAttention version: {flash_attn.__version__}")
    
    def _prepare_decoder_hidden(self, h: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Prepare decoder hidden state for attention computation."""
        if isinstance(h, tuple):
            h = h[0]
        return h[-1] if h.dim() == 3 else h  # [B, H]
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into multiple attention heads."""
        B, T, D = x.shape
        return x.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine multiple attention heads back into single tensor."""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)
    
    def _apply_backend_optimization(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Apply backend-specific optimizations to scaled dot-product attention."""
        B = q.size(0)
        q, k, v = map(self._split_heads, (q, k, v))
        
        if self.backend == "flash" and HAS_FLASH_ATTN:
            # FlashAttention implementation
            q, k, v = [x.contiguous().to(torch.float16) for x in (q, k, v)]
            out = flash_attn_func(q, k, v, dropout_p=self.dropout.p, causal=False)
            out = out.transpose(1, 2).contiguous().view(B, -1, self.nhead * self.head_dim)
            return self.context_proj(out.to(torch.float32))
        
        elif self.backend == "xformers" and HAS_XFORMERS:
            # xformers memory-efficient attention
            out = xops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            return self._combine_heads(out.transpose(1, 2))
        
        else:  # torch backend
            # PyTorch native scaled dot-product attention
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
            return self._combine_heads(out)
    
    def _compute_attention_weights(self, query: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """Efficiently compute attention weights for methods that need them."""
        scores = torch.bmm(encoder_outputs, query.transpose(1, 2)).squeeze(2)
        return F.softmax(scores, dim=1)
        
    def _dot_attention(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized dot-product attention with backend optimization support."""
        query = decoder_hidden.unsqueeze(1)  # [B, 1, D]
        
        # Use backend optimization if projections are available
        if hasattr(self, 'q_proj'):
            q = self.q_proj(query)  # [B, 1, D]
            k = self.k_proj(encoder_outputs)  # [B, T, D]
            v = self.v_proj(encoder_outputs)  # [B, T, D]
            
            context = self._apply_backend_optimization(q, k, v)
            context = context[:, 0]  # Remove sequence dimension
            
            # Compute attention weights separately for compatibility
            with torch.no_grad():
                attn_weights = self._compute_attention_weights(query, encoder_outputs)
            
            return context, attn_weights
        
        else:
            # Optimized vanilla dot attention
            attn_weights = self._compute_attention_weights(query, encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            return context, attn_weights
        
    def _mha_attention(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Multi-head attention with optimized projections."""
        query = decoder_hidden.unsqueeze(1)  # [B, 1, D]
        
        # Project query, keys, and values
        q = self.q_proj(query)  # [B, 1, D]
        k = self.k_proj(encoder_outputs)  # [B, T, D]
        v = self.v_proj(encoder_outputs)  # [B, T, D]
        
        context = self._apply_backend_optimization(q, k, v)
        return context[:, 0], None  # Remove sequence dimension
    
    def _prob_attention(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Probabilistic attention using optimized batch matrix multiplication."""
        query = decoder_hidden.unsqueeze(1)  # [B, 1, D]
        scores = torch.bmm(query, encoder_outputs.transpose(1, 2)) * self.scale_factor
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
        return context, attn_weights.squeeze(1)
    
    def _temporal_attention(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
        encoder_timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Time-aware attention with temporal bias."""
        if encoder_timestamps is not None:
            # In-place addition for memory efficiency
            encoder_outputs = encoder_outputs + self.time_bias(encoder_timestamps)
        
        return self._dot_attention(decoder_hidden, encoder_outputs)
    
    def _multiscale_attention(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Optimized multi-scale attention with dilated temporal processing."""
        query = decoder_hidden.unsqueeze(1)  # [B, 1, D]
        q = self.q_proj(query)
        q = self._split_heads(q)
        
        outputs = []
        for i, dilation in enumerate(self.dilations):
            # Apply dilation to encoder outputs
            dilated_enc = encoder_outputs if dilation == 1 else encoder_outputs[:, ::dilation, :]
            
            # Use the optimized projection modules
            k_i = self.scale_projections[i]['k'](dilated_enc)
            v_i = self.scale_projections[i]['v'](dilated_enc)
            k_i, v_i = map(self._split_heads, (k_i, v_i))
            
            # Apply attention at this scale
            if self.backend == "xformers" and HAS_XFORMERS:
                out = xops.memory_efficient_attention(
                    q.transpose(1, 2), k_i.transpose(1, 2), v_i.transpose(1, 2)
                )
                out = self._combine_heads(out.transpose(1, 2))
            else:
                out = F.scaled_dot_product_attention(q, k_i, v_i, dropout_p=self.dropout.p)
                out = self._combine_heads(out)
            
            outputs.append(out)
        
        # Efficient weighted combination using stack and sum
        outputs = torch.stack(outputs, dim=0)  # [num_scales, B, 1, D]
        weights = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        combined = (outputs * weights).sum(dim=0)
        context = self.scale_out_proj(combined)
        
        return context[:, 0], None  # Remove sequence dimension
    
    def _autocorr_attention(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, topk: int = 3
    ) -> Tuple[torch.Tensor, None]:
        """Optimized auto-correlation attention using FFT."""
        query = decoder_hidden.unsqueeze(1)  # [B, 1, D]
        
        q = self.q_proj(query)  # [B, 1, D]
        
        # FFT-based autocorrelation with optimized dimension handling
        q_fft = torch.fft.rfft(q, dim=1)
        k_fft = torch.fft.rfft(encoder_outputs, dim=1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=1)
        
        # Find top-k correlations efficiently
        corr_mean = corr.abs().mean(dim=-1)  # [B, T]
        topk_values, topk_idx = torch.topk(corr_mean, k=min(topk, corr.size(1)), dim=1)
        
        # Vectorized aggregation using advanced indexing
        B, T, D = encoder_outputs.shape
        batch_idx = torch.arange(B, device=encoder_outputs.device).unsqueeze(1)
        
        # Create rolled versions more efficiently
        context_list = []
        for b in range(B):
            indices = topk_idx[b]  # [topk]
            rolled_outputs = []
            for idx in indices:
                rolled = torch.roll(encoder_outputs[b], -idx.item(), dims=0)
                rolled_outputs.append(rolled)
            context_list.append(torch.stack(rolled_outputs).mean(dim=0))
        
        context = torch.stack(context_list, dim=0)  # [B, T, D]
        return context.mean(dim=1), None  # Average over time
    
    def forward(
        self, 
        decoder_hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
        encoder_outputs: torch.Tensor,
        encoder_timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of attention layer.
        
        Args:
            decoder_hidden: Decoder hidden state [B, D] or tuple of states
            encoder_outputs: Encoder outputs [B, T, D]
            encoder_timestamps: Optional temporal embeddings [B, T, time_embed_dim]
        
        Returns:
            Tuple of (attended_output, attention_weights)
            - attended_output: [B, D] - Combined decoder and context representation
            - attention_weights: [B, T] or None - Attention weights if computed
        """
        decoder_hidden = self._prepare_decoder_hidden(decoder_hidden)
        encoder_outputs = self.encoder_projection(encoder_outputs)
        
        # Dispatch to appropriate attention method
        if self.method == "dot":
            context, attn_weights = self._dot_attention(decoder_hidden, encoder_outputs)
        elif self.method == "mha":
            context, attn_weights = self._mha_attention(decoder_hidden, encoder_outputs)
        elif self.method == "prob":
            context, attn_weights = self._prob_attention(decoder_hidden, encoder_outputs)
        elif self.method == "temporal":
            context, attn_weights = self._temporal_attention(decoder_hidden, encoder_outputs, encoder_timestamps)
        elif self.method == "multiscale":
            context, attn_weights = self._multiscale_attention(decoder_hidden, encoder_outputs)
        elif self.method == "autocorr":
            context, attn_weights = self._autocorr_attention(decoder_hidden, encoder_outputs)
        else:
            raise ValueError(f"Unknown attention method: {self.method}")
        
        # Combine context and decoder hidden state
        combined = self.combined_layer(torch.cat([context, decoder_hidden], dim=1))
        
        return torch.tanh(combined), attn_weights