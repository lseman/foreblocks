import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Optimized positional encoding with support for:
    - Standard sinusoidal encoding
    - Scaled injection
    - Dynamic dimension handling
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.max_len = max_len

        # Use nn.Dropout with inplace for memory efficiency
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else None

        # Precompute sinusoidal encoding with better numerical stability
        self._build_pe_cache(d_model, max_len)

        # Cache for dynamic dimensions
        self._pe_cache: Dict[int, torch.Tensor] = {}

    def _build_pe_cache(self, d_model: int, max_len: int):
        """Build positional encoding cache with optimized computation"""
        # Use log-space computation for better numerical stability
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # Optimized div_term computation
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # Pre-allocate with correct size
        sinusoid = torch.empty(max_len, d_model, dtype=torch.float32)

        # Vectorized sine/cosine computation
        angles = position * div_term
        sinusoid[:, 0::2] = torch.sin(angles)
        if d_model % 2 == 1:
            # Handle odd dimensions
            sinusoid[:, 1::2] = torch.cos(angles[:, :-1])
            sinusoid[:, -1] = 0  # Set last dimension to 0 for odd d_model
        else:
            sinusoid[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", sinusoid.unsqueeze(0), persistent=False)

    def _create_pe_for_dim(
        self, d_model: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create PE for different dimensions with caching"""
        cache_key = d_model

        if cache_key not in self._pe_cache:
            # Create PE for this dimension
            position = torch.arange(
                seq_len, dtype=torch.float32, device=device
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                * (-math.log(10000.0) / d_model)
            )

            pe = torch.empty(seq_len, d_model, dtype=torch.float32, device=device)
            angles = position * div_term
            pe[:, 0::2] = torch.sin(angles)

            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(angles[:, :-1])
                pe[:, -1] = 0
            else:
                pe[:, 1::2] = torch.cos(angles)

            # Cache only if reasonable size
            if d_model <= 2048:  # Prevent memory explosion
                self._pe_cache[cache_key] = pe.unsqueeze(0)

            return pe.unsqueeze(0)

        cached_pe = self._pe_cache[cache_key]
        if cached_pe.size(1) >= seq_len:
            return cached_pe[:, :seq_len]
        else:
            # Need to extend cache
            return self._create_pe_for_dim(d_model, seq_len, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward with minimal overhead"""
        B, T, D = x.shape

        if D == self.d_model and T <= self.max_len:
            # Fast path: use pre-computed PE
            pe = self.pe[:, :T]
        else:
            # Slow path: create PE on-demand
            pe = self._create_pe_for_dim(D, T, x.device)

        # Use fused operation
        x = x.add_(pe, alpha=self.scale)  # In-place addition with scaling

        return self.dropout(x) if self.dropout is not None else x


class InformerTimeEmbedding(nn.Module):
    """Optimized time embedding with better memory efficiency"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Use smaller embedding dimensions and project up
        embed_dim = min(d_model // 4, 64)  # Adaptive embedding size

        self.hour_embed = nn.Embedding(24, embed_dim)
        self.weekday_embed = nn.Embedding(7, embed_dim)
        self.day_embed = nn.Embedding(32, embed_dim)
        self.month_embed = nn.Embedding(13, embed_dim)

        # Project to full dimension if needed
        if embed_dim * 4 != d_model:
            self.projection = nn.Linear(embed_dim * 4, d_model)
        else:
            self.projection = None

        # Normalization factor
        self.norm_factor = 1.0 / math.sqrt(4.0)  # Pre-compute for efficiency

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Better initialization for time embeddings"""
        for embed in [
            self.hour_embed,
            self.weekday_embed,
            self.day_embed,
            self.month_embed,
        ]:
            nn.init.normal_(embed.weight, mean=0, std=0.02)

    def forward(self, time_feats: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward with vectorized operations
        time_feats: [B, T, 4] - month, weekday, hour, day
        """
        # Clamp inputs to valid ranges (safety check)
        month = torch.clamp(time_feats[:, :, 0].long(), 0, 12)
        weekday = torch.clamp(time_feats[:, :, 1].long(), 0, 6)
        hour = torch.clamp(time_feats[:, :, 2].long(), 0, 23)
        day = torch.clamp(time_feats[:, :, 3].long(), 0, 31)

        # Vectorized embedding lookup
        month_emb = self.month_embed(month)
        weekday_emb = self.weekday_embed(weekday)
        hour_emb = self.hour_embed(hour)
        day_emb = self.day_embed(day)

        # Concatenate embeddings
        embs = torch.cat([month_emb, weekday_emb, hour_emb, day_emb], dim=-1)

        # Project to target dimension if needed
        if self.projection is not None:
            embs = self.projection(embs)

        # Apply normalization
        return embs * self.norm_factor

class RotaryEmbedding(nn.Module):
    """
    Optimized Rotary position embeddings (RoPE) with better caching.
    """

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Pre-compute and cache frequencies for common sequence lengths
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, max_seq_len: int):
        """Pre-compute frequencies for better performance"""
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))
        seq_idx = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = seq_idx * theta

        # Pre-compute cos and sin
        cos_freqs = freqs.cos()
        sin_freqs = freqs.sin()

        # Store as buffers for automatic device handling
        self.register_buffer("cos_cached", cos_freqs, persistent=False)
        self.register_buffer("sin_cached", sin_freqs, persistent=False)

    def _get_freqs(self, seq_len: int, device: torch.device):
        """Get frequencies with fallback for longer sequences"""
        if seq_len <= self.max_seq_len:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        else:
            # Fallback for longer sequences
            theta = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=device) / self.dim)
            )
            seq_idx = torch.arange(seq_len, device=device).float().unsqueeze(1)
            freqs = seq_idx * theta
            return freqs.cos(), freqs.sin()

    @staticmethod
    def apply_rotary_pos_emb(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Optimized rotary embedding application"""
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split into even and odd
        x1, x2 = x[..., 0::2], x[..., 1::2]

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave back
        rotated_x = torch.empty_like(x)
        rotated_x[..., 0::2] = rotated_x1
        rotated_x[..., 1::2] = rotated_x2

        return rotated_x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_pos: Optional[torch.Tensor] = None,
        k_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass
        q, k: [batch, heads, seq_len, head_dim]
        """
        *_, q_len, head_dim = q.shape
        _, _, k_len, _ = k.shape

        # Only apply to rotary dimensions
        rotary_dim = min(head_dim, self.dim)

        if rotary_dim == head_dim:
            # Full rotation
            q_rot, k_rot = q, k
            q_pass = k_pass = None
        else:
            # Partial rotation
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        # Get frequencies
        max_len = max(q_len, k_len)
        cos_freqs, sin_freqs = self._get_freqs(max_len, q.device)

        # Handle custom positions
        if q_pos is not None:
            q_cos = cos_freqs[q_pos]
            q_sin = sin_freqs[q_pos]
        else:
            q_cos = cos_freqs[:q_len]
            q_sin = sin_freqs[:q_len]

        if k_pos is not None:
            k_cos = cos_freqs[k_pos]
            k_sin = sin_freqs[k_pos]
        else:
            k_cos = cos_freqs[:k_len]
            k_sin = sin_freqs[:k_len]

        # Apply rotary embedding
        q_rot = self.apply_rotary_pos_emb(q_rot, q_cos, q_sin)
        k_rot = self.apply_rotary_pos_emb(k_rot, k_cos, k_sin)

        # Concatenate with non-rotated parts if needed
        if q_pass is not None:
            q_out = torch.cat([q_rot, q_pass], dim=-1)
            k_out = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q_out, k_out = q_rot, k_rot

        return q_out, k_out

    def clear_cache(self):
        """Clear dynamic cache to free memory"""
        # Only persistent buffers remain
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math
from typing import Optional
from torch.nn.parameter import Parameter


@triton.jit
def fused_pe_addition_kernel(
    x_ptr,
    pe_ptr,
    output_ptr,
    scale_val: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for PE addition with scaling"""
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and PE
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    pe_vals = tl.load(pe_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: x + pe * scale
    result = x_vals + pe_vals * scale_val
    
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit  
def low_rank_matmul_kernel(
    U_ptr,
    V_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    low_rank_dim: tl.constexpr,
    d_model: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Optimized low-rank matrix multiplication kernel"""
    # 3D grid: (M_blocks, N_blocks, batch)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate ranges for this block
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Accumulator for this block
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k_start in range(0, low_rank_dim, BLOCK_K):
        k_end = tl.minimum(k_start + BLOCK_K, low_rank_dim)
        
        # Load U block [BLOCK_M, BLOCK_K]
        m_offsets = m_start + tl.arange(0, BLOCK_M)[:, None]
        k_offsets = k_start + tl.arange(0, BLOCK_K)[None, :]
        
        u_ptrs = U_ptr + pid_b * seq_len * low_rank_dim + m_offsets * low_rank_dim + k_offsets
        u_mask = (m_offsets < seq_len) & (k_offsets < k_end)
        U_block = tl.load(u_ptrs, mask=u_mask, other=0.0)
        
        # Load V block [BLOCK_K, BLOCK_N]  
        k_offsets_v = k_start + tl.arange(0, BLOCK_K)[:, None]
        n_offsets = n_start + tl.arange(0, BLOCK_N)[None, :]
        
        v_ptrs = V_ptr + pid_b * low_rank_dim * d_model + k_offsets_v * d_model + n_offsets
        v_mask = (k_offsets_v < k_end) & (n_offsets < d_model)
        V_block = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(U_block, V_block)
    
    # Store result
    m_offsets = m_start + tl.arange(0, BLOCK_M)[:, None]
    n_offsets = n_start + tl.arange(0, BLOCK_N)[None, :]
    
    output_ptrs = output_ptr + pid_b * seq_len * d_model + m_offsets * d_model + n_offsets
    output_mask = (m_offsets < seq_len) & (n_offsets < d_model)
    
    tl.store(output_ptrs, accumulator, mask=output_mask)


@triton.jit
def embedding_lookup_kernel(
    positions_ptr,
    embeddings_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    embed_dim: tl.constexpr,
    max_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized embedding lookup for positional indices"""
    pid = tl.program_id(axis=0)
    
    # Calculate which sequence position this block handles
    total_elements = batch_size * seq_len
    start_idx = pid * BLOCK_SIZE
    
    if start_idx >= total_elements:
        return
        
    # Process multiple positions in this block
    for i in range(BLOCK_SIZE):
        linear_idx = start_idx + i
        if linear_idx >= total_elements:
            break
            
        # Convert linear index to (batch, seq) coordinates
        batch_idx = linear_idx // seq_len
        seq_idx = linear_idx % seq_len
        
        # Load position index
        pos_ptr = positions_ptr + batch_idx * seq_len + seq_idx
        position = tl.load(pos_ptr)
        
        # Bounds check
        if position >= max_len or position < 0:
            continue
            
        # Copy embedding row
        for d in range(0, embed_dim, 32):  # Process 32 dims at a time
            d_end = tl.minimum(d + 32, embed_dim)
            d_offsets = d + tl.arange(0, 32)
            d_mask = d_offsets < d_end
            
            # Load from embedding table
            embed_ptrs = embeddings_ptr + position * embed_dim + d_offsets
            embed_vals = tl.load(embed_ptrs, mask=d_mask, other=0.0)
            
            # Store to output
            output_ptrs = output_ptr + linear_idx * embed_dim + d_offsets
            tl.store(output_ptrs, embed_vals, mask=d_mask)


class LearnablePositionalEncoding(nn.Module):
    """
    Highly optimized learnable positional encoding with:
    - Triton kernels for computation
    - Memory-efficient parameter storage
    - Fused operations
    - Optimized initialization
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        initialization: str = "normal",
        scale_strategy: str = "fixed",  # ["fixed", "learnable", "none"]
        scale_value: Optional[float] = None,
        use_layer_norm: bool = True,
        norm_strategy: str = "pre_add",  # or "post_add"
        low_rank_dim: Optional[int] = None,
        use_triton: bool = True,
        memory_efficient: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.low_rank_dim = low_rank_dim
        self.norm_strategy = norm_strategy
        self.use_triton = use_triton and torch.cuda.is_available()
        self.memory_efficient = memory_efficient
        
        # Optimized parameter initialization
        if low_rank_dim is None:
            # Standard learnable PE with memory optimization
            if memory_efficient:
                # Use half precision for PE if model allows it
                pe_data = self._init_pe_optimized(initialization, (max_len, d_model))
                self.pe = Parameter(pe_data)
            else:
                self.pe = Parameter(self._init_pe(initialization, (1, max_len, d_model)))
        else:
            # Low-rank factorization for memory efficiency
            self.pe_proj_U = Parameter(
                self._init_pe(initialization, (max_len, low_rank_dim))
            )
            self.pe_proj_V = Parameter(
                self._init_pe(initialization, (low_rank_dim, d_model))
            )
            
        # Optimized scaling
        if scale_strategy == "learnable":
            init_scale = scale_value or math.sqrt(d_model)
            self.scale = Parameter(torch.tensor(init_scale, dtype=torch.float32))
            self.scale_is_tensor = True
        elif scale_strategy == "fixed":
            scale_val = scale_value or math.sqrt(d_model)
            self.register_buffer('scale', torch.tensor(scale_val, dtype=torch.float32))
            self.scale_is_tensor = False
        else:
            self.register_buffer('scale', torch.tensor(1.0, dtype=torch.float32))
            self.scale_is_tensor = False
            
        # Optimized normalization
        if use_layer_norm:
            # Use fused layer norm when available
            if hasattr(F, 'layer_norm') and torch.cuda.is_available():
                self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
            else:
                self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.layer_norm = None
            
        # Optimized dropout
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else None
        
        # Cache for frequently used sequence lengths
        self._pe_cache = {}
        self._cache_size_limit = 8
        
    def _init_pe(self, mode: str, shape: tuple) -> torch.Tensor:
        """Standard initialization"""
        if mode == "normal":
            return torch.randn(shape) * math.sqrt(2.0 / shape[-1])
        elif mode == "uniform":
            bound = math.sqrt(6.0 / shape[-1])
            return torch.empty(shape).uniform_(-bound, bound)
        elif mode == "zero":
            return torch.zeros(shape)
        elif mode == "xavier":
            return torch.empty(shape).normal_(0, math.sqrt(2.0 / (shape[-2] + shape[-1])))
        else:
            return torch.randn(shape) * 0.02
            
    def _init_pe_optimized(self, mode: str, shape: tuple) -> torch.Tensor:
        """Memory-optimized initialization"""
        # Remove batch dimension for memory efficiency
        pe = self._init_pe(mode, shape)
        
        # Optional: quantize for even more memory savings (experimental)
        # pe = pe.half()  # Use FP16 if your model supports it
        
        return pe
        
    def _get_pe_standard(self, B: int, T: int, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get PE for standard (non-low-rank) case"""
        if positions is None:
            if self.memory_efficient:
                # Direct slicing without batch dimension
                pe = self.pe[:T].unsqueeze(0)  # [1, T, d_model]
                return pe.expand(B, -1, -1)
            else:
                return self.pe[:, :T].expand(B, -1, -1)
        else:
            # Embedding lookup - use Triton for large lookups
            if self.use_triton and positions.device.type == 'cuda' and B * T > 1024:
                return self._triton_embedding_lookup(positions)
            else:
                if self.memory_efficient:
                    return F.embedding(positions, self.pe)
                else:
                    return F.embedding(positions, self.pe.squeeze(0))
    
    def _triton_embedding_lookup(self, positions: torch.Tensor) -> torch.Tensor:
        """Triton-accelerated embedding lookup"""
        B, T = positions.shape
        output = torch.empty(B, T, self.d_model, device=positions.device, dtype=self.pe.dtype)
        
        total_elements = B * T
        BLOCK_SIZE = min(256, triton.next_power_of_2(total_elements))
        grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
        
        pe_data = self.pe if self.memory_efficient else self.pe.squeeze(0)
        
        embedding_lookup_kernel[(grid_size,)](
            positions,
            pe_data,
            output,
            batch_size=B,
            seq_len=T,
            embed_dim=self.d_model,
            max_len=self.max_len,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output
        
    def _get_pe_low_rank(self, B: int, T: int, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get PE for low-rank case with Triton optimization"""
        if positions is None:
            U = self.pe_proj_U[:T]  # [T, low_rank_dim]
            V = self.pe_proj_V      # [low_rank_dim, d_model]
            
            # Use Triton for large matrix multiplications
            if self.use_triton and U.device.type == 'cuda' and T * self.d_model > 4096:
                return self._triton_low_rank_matmul(U, V, B, T)
            else:
                # Standard PyTorch implementation
                pe = torch.matmul(U, V)  # [T, d_model]
                return pe.unsqueeze(0).expand(B, -1, -1)
        else:
            # Embedding lookup for both U and V
            U = F.embedding(positions, self.pe_proj_U)  # [B, T, low_rank_dim]
            V = self.pe_proj_V.unsqueeze(0).expand(B, -1, -1)  # [B, low_rank_dim, d_model]
            
            if self.use_triton and U.device.type == 'cuda' and B * T * self.d_model > 8192:
                return self._triton_batch_matmul(U, V)
            else:
                return torch.bmm(U, V)
                
    def _triton_low_rank_matmul(self, U: torch.Tensor, V: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Triton kernel for low-rank matrix multiplication"""
        output = torch.empty(B, T, self.d_model, device=U.device, dtype=U.dtype)
        
        # Expand U for batch processing
        U_batch = U.unsqueeze(0).expand(B, -1, -1).contiguous()
        V_batch = V.unsqueeze(0).expand(B, -1, -1).contiguous()
        
        BLOCK_M = min(64, triton.next_power_of_2(T))
        BLOCK_N = min(64, triton.next_power_of_2(self.d_model))
        BLOCK_K = min(32, triton.next_power_of_2(self.low_rank_dim))
        
        grid = (
            triton.cdiv(T, BLOCK_M),
            triton.cdiv(self.d_model, BLOCK_N),
            B
        )
        
        low_rank_matmul_kernel[grid](
            U_batch,
            V_batch,
            output,
            batch_size=B,
            seq_len=T,
            low_rank_dim=self.low_rank_dim,
            d_model=self.d_model,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        
        return output
        
    def _triton_batch_matmul(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Triton-optimized batch matrix multiplication"""
        B, T, _ = U.shape
        output = torch.empty(B, T, self.d_model, device=U.device, dtype=U.dtype)
        
        BLOCK_M = min(64, triton.next_power_of_2(T))
        BLOCK_N = min(64, triton.next_power_of_2(self.d_model))
        BLOCK_K = min(32, triton.next_power_of_2(self.low_rank_dim))
        
        grid = (
            triton.cdiv(T, BLOCK_M),
            triton.cdiv(self.d_model, BLOCK_N),
            B
        )
        
        low_rank_matmul_kernel[grid](
            U.contiguous(),
            V.contiguous(),
            output,
            batch_size=B,
            seq_len=T,
            low_rank_dim=self.low_rank_dim,
            d_model=self.d_model,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        
        return output
        
    def _triton_fused_addition(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """Triton kernel for fused PE addition with scaling"""
        output = torch.empty_like(x)
        
        total_elements = x.numel()
        BLOCK_SIZE = 1024
        grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
        
        # Convert scale to float for kernel
        if self.scale_is_tensor:
            scale_val = float(self.scale.item())
        else:
            scale_val = float(self.scale.item())
        
        fused_pe_addition_kernel[(grid_size,)](
            x.contiguous().view(-1),
            pe.contiguous().view(-1),
            output.view(-1),
            scale_val=scale_val,
            n_elements=total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Optimized forward pass"""
        B, T, _ = x.shape
        
        # Cache check for common sequence lengths
        cache_key = (T, positions is not None)
        if cache_key in self._pe_cache and positions is None:
            pe = self._pe_cache[cache_key]
            if pe.size(0) != B:
                pe = pe.expand(B, -1, -1)
        else:
            # Generate PE
            if self.low_rank_dim is None:
                pe = self._get_pe_standard(B, T, positions)
            else:
                pe = self._get_pe_low_rank(B, T, positions)
                
            # Cache if reasonable size and no custom positions
            if (len(self._pe_cache) < self._cache_size_limit and 
                positions is None and T <= 2048):
                self._pe_cache[cache_key] = pe[:1] if B > 1 else pe
        
        # Pre-norm if specified
        if self.norm_strategy == "pre_add" and self.layer_norm:
            x = self.layer_norm(x)
        
        # Fused PE addition with Triton for large tensors
        if (self.use_triton and x.device.type == 'cuda' and 
            B * T * self.d_model > 4096):
            x = self._triton_fused_addition(x, pe)
        else:
            # Standard addition
            if self.scale_is_tensor:
                x = x + pe * self.scale
            else:
                x = x.add_(pe, alpha=self.scale)  # In-place when possible
        
        # Post-norm if specified
        if self.norm_strategy == "post_add" and self.layer_norm:
            x = self.layer_norm(x)
            
        # Dropout
        return self.dropout(x) if self.dropout else x