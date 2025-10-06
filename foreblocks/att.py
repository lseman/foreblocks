"""
Enhanced SOTA Multi-Method Attention Layer

New Methods:
- ProbSparse: Efficient sparse attention from Informer
- Linear: Linear attention with kernel approximation
- Sliding Window: Local attention with configurable window
- LSH: Locality-sensitive hashing attention

Existing Improvements:
- GQA/MQA support for efficient KV caching
- Flash Attention 2 compatibility
- Proper mask handling for PyTorch 2.0+ SDPA
- xFormers integration
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Optional deps ----------
try:
    import xformers
    import xformers.ops as xops
    HAS_XFORMERS = True
except Exception:
    HAS_XFORMERS = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    HAS_FLASH_ATTN = False


class AttentionLayer(nn.Module):
    """
    Flexible attention supporting multiple methods and backends.
    
    Methods:
    - dot: Simple dot-product attention
    - mha: Multi-head attention
    - multiscale: Multi-scale temporal attention
    - probsparse: ProbSparse attention (Informer)
    - linear: Linear attention with kernel tricks
    - sliding_window: Local windowed attention
    - lsh: LSH attention (approximate)
    
    Backends: torch, xformers, flash
    """

    _VALID_METHODS = frozenset({
        "dot", "mha", "multiscale", "probsparse", 
        "linear", "sliding_window", "lsh"
    })
    _VALID_BACKENDS = frozenset({"torch", "xformers", "flash"})
    _MHA_METHODS = frozenset({"mha", "multiscale", "probsparse", "sliding_window", "lsh"})

    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: Optional[int] = None,
        method: str = "mha",
        attention_backend: str = "torch",
        nhead: int = 4,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        num_scales: int = 3,
        use_residual_ln: bool = False,
        use_swiglu: bool = True,
        # ProbSparse params
        sampling_factor: int = 5,
        top_k: Optional[int] = None,
        # Sliding window params
        window_size: int = 256,
        # LSH params
        n_hashes: int = 4,
        bucket_size: int = 64,
        verbose: bool = False,
    ):
        super().__init__()

        backend = attention_backend.lower()
        self._validate_inputs(method, backend, decoder_hidden_size, nhead)

        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size or decoder_hidden_size
        self.method = method
        self.backend = backend
        self.nhead = nhead
        self.head_dim = decoder_hidden_size // nhead
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)
        self.use_residual_ln = use_residual_ln

        # GQA setup
        self.n_kv_heads = n_kv_heads or nhead
        assert nhead % self.n_kv_heads == 0, "nhead must be divisible by n_kv_heads"
        self.n_rep = nhead // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Method-specific params
        self.sampling_factor = sampling_factor
        self.top_k = top_k
        self.window_size = window_size
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self.dropout = nn.Dropout(dropout)
        self._build_projections(use_swiglu)
        self._build_method_specific_layers(num_scales)

        if use_residual_ln:
            self.post_ln = nn.LayerNorm(self.decoder_hidden_size)

        if verbose:
            self._print_initialization_info()

    # ----- Setup -----

    def _validate_inputs(self, method: str, backend: str, hidden_size: int, nhead: int):
        if method not in self._VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from {self._VALID_METHODS}")
        if backend not in self._VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Choose from {self._VALID_BACKENDS}")
        if hidden_size % nhead != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by nhead {nhead}")
        if backend == "flash" and not HAS_FLASH_ATTN:
            raise ImportError("FlashAttention not available but requested")
        if backend == "xformers" and not HAS_XFORMERS:
            raise ImportError("xFormers not available but requested")

    def _build_projections(self, use_swiglu: bool):
        # Combine decoder hidden + context
        if use_swiglu:
            self.combined_layer = nn.Sequential(
                nn.Linear(self.decoder_hidden_size * 2, self.decoder_hidden_size * 4, bias=False),
                nn.SiLU(),
                nn.Linear(self.decoder_hidden_size * 4, self.decoder_hidden_size, bias=False)
            )
        else:
            self.combined_layer = nn.Linear(self.decoder_hidden_size * 2, self.decoder_hidden_size)

        # Align encoder dim to decoder dim if needed
        self.encoder_projection = (
            nn.Identity()
            if self.decoder_hidden_size == self.encoder_hidden_size
            else nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=False)
        )

        # QKV projections with GQA
        if self.method in self._MHA_METHODS or self.method == "dot":
            self.q_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)
            self.k_proj = nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False)
            
            # Output projection for backends that need it
            if self.backend in ["flash", "xformers"] or self.method in ["linear", "lsh"]:
                self.out_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)

    def _build_method_specific_layers(self, num_scales: int):
        if self.method == "multiscale":
            self.scale_projections = nn.ModuleList([
                nn.ModuleDict({
                    "k": nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False),
                    "v": nn.Linear(self.decoder_hidden_size, self.kv_dim, bias=False),
                })
                for _ in range(num_scales)
            ])
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.dilations = tuple(2 ** i for i in range(num_scales))
        
        elif self.method == "linear":
            # Feature map for linear attention
            self.feature_dim = self.head_dim
            self.feature_map = nn.Sequential(
                nn.Linear(self.head_dim, self.feature_dim),
                nn.ReLU()
            )
        
        elif self.method == "lsh":
            # LSH random projections
            self.register_buffer(
                "random_rotations",
                torch.randn(self.n_hashes, self.head_dim, self.head_dim // 2)
            )

    def _print_initialization_info(self):
        gqa_info = f"GQA({self.nhead}q/{self.n_kv_heads}kv)" if self.n_rep > 1 else "MHA"
        print(f"[Attention] {gqa_info}, Method: {self.method}, Backend: {self.backend}")
        if self.method == "probsparse":
            print(f"[Attention] ProbSparse: sampling_factor={self.sampling_factor}, top_k={self.top_k}")
        elif self.method == "sliding_window":
            print(f"[Attention] Sliding window size: {self.window_size}")
        elif self.method == "lsh":
            print(f"[Attention] LSH: n_hashes={self.n_hashes}, bucket_size={self.bucket_size}")
        if HAS_XFORMERS:
            print(f"[Attention] xFormers: {getattr(xformers, '__version__', 'unknown')}")
        if HAS_FLASH_ATTN:
            print(f"[Attention] FlashAttention 2 available")

    # ----- Helpers -----

    @staticmethod
    def _last_hidden(dec_h: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Extract last hidden state from GRU/LSTM output."""
        if isinstance(dec_h, tuple):
            dec_h = dec_h[0]
        return dec_h[-1]  # Last layer: [L,B,H] -> [B,H]

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA."""
        if self.n_rep == 1:
            return x
        
        B, n_kv_heads, T, head_dim = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_heads, self.n_rep, T, head_dim)
        return x.reshape(B, n_kv_heads * self.n_rep, T, head_dim)

    def _project_qkv(self, query: torch.Tensor, enc: torch.Tensor):
        """Project and reshape to multi-head format."""
        B, T, _ = enc.shape
        
        q = self.q_proj(query).view(B, 1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        return q, k, v

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, nH, T, Hd] -> [B, T, D]"""
        B, Hh, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, Hh * Hd)

    def _torch_sdpa(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """PyTorch 2.0+ scaled dot-product attention."""
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        B, _, Tq, _ = q.shape
        _, _, Tk, _ = k.shape
        
        combined_mask = None
        if attn_mask is not None or key_padding_mask is not None:
            combined_mask = torch.zeros(B, self.nhead, Tq, Tk, dtype=torch.bool, device=q.device)
            
            if attn_mask is not None:
                combined_mask = combined_mask | attn_mask.unsqueeze(0).unsqueeze(0)
            
            if key_padding_mask is not None:
                combined_mask = combined_mask | key_padding_mask.view(B, 1, 1, Tk)
        
        drop = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=combined_mask,
            dropout_p=drop,
            is_causal=False
        )
        return out

    def _flash_attn(self, q, k, v):
        """Flash Attention 2 wrapper."""
        assert HAS_FLASH_ATTN and flash_attn_func is not None
        
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        dtype = q.dtype if q.dtype in [torch.float16, torch.bfloat16] else torch.float16
        q = q.transpose(1, 2).contiguous().to(dtype)
        k = k.transpose(1, 2).contiguous().to(dtype)
        v = v.transpose(1, 2).contiguous().to(dtype)
        
        drop = self.dropout.p if self.training else 0.0
        out = flash_attn_func(q, k, v, dropout_p=drop, causal=False)
        
        out = out.transpose(1, 2).to(torch.float32)
        return out

    def _xformers_attn(self, q, k, v):
        """xFormers memory-efficient attention."""
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        drop = self.dropout.p if self.training else 0.0
        out = xops.memory_efficient_attention(q, k, v, p=drop)
        
        return out.transpose(1, 2)

    # ----- Attention Methods -----

    def _compute_attn_weights(self, query_1t: torch.Tensor, enc: torch.Tensor, key_padding_mask=None):
        """Vanilla attention weights for logging."""
        scores = torch.bmm(enc, query_1t.transpose(1, 2)).squeeze(2) * self.scale_factor
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        return F.softmax(scores, dim=1)

    def _dot_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """Simple dot-product attention with optional backend acceleration."""
        q1 = dec_h.unsqueeze(1)
        
        if hasattr(self, "q_proj"):
            q, k, v = self._project_qkv(q1, enc)
            
            if self.backend == "flash":
                out = self._flash_attn(q, k, v)
                ctx = self._combine_heads(out)[:, 0]
                ctx = self.out_proj(ctx)
            elif self.backend == "xformers":
                out = self._xformers_attn(q, k, v)
                ctx = self._combine_heads(out)[:, 0]
                ctx = self.out_proj(ctx)
            else:
                out = self._torch_sdpa(q, k, v, attn_mask, key_padding_mask)
                ctx = self._combine_heads(out)[:, 0]
            
            with torch.no_grad():
                w = self._compute_attn_weights(q1, enc, key_padding_mask)
            return ctx, w
        
        w = self._compute_attn_weights(q1, enc, key_padding_mask)
        ctx = torch.bmm(w.unsqueeze(1), enc).squeeze(1)
        return ctx, w

    def _mha_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """Multi-head attention with GQA support."""
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        
        if self.backend == "flash":
            out = self._flash_attn(q, k, v)
            ctx = self._combine_heads(out)[:, 0]
            ctx = self.out_proj(ctx)
        elif self.backend == "xformers":
            out = self._xformers_attn(q, k, v)
            ctx = self._combine_heads(out)[:, 0]
            ctx = self.out_proj(ctx)
        else:
            out = self._torch_sdpa(q, k, v, attn_mask, key_padding_mask)
            ctx = self._combine_heads(out)[:, 0]
        
        return ctx, None

    def _multiscale_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """Multi-scale attention: attend to encoder at different temporal resolutions."""
        q1 = dec_h.unsqueeze(1)
        q = self.q_proj(q1).view(-1, 1, self.nhead, self.head_dim).transpose(1, 2)
        
        outs = []
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                e = enc[:, ::dilation, :]
                kpm = key_padding_mask[:, ::dilation] if key_padding_mask is not None else None
            else:
                e = enc
                kpm = key_padding_mask
            
            B, T, _ = e.shape
            k_i = self.scale_projections[i]["k"](e).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v_i = self.scale_projections[i]["v"](e).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            
            if self.backend == "xformers":
                o = self._xformers_attn(q, k_i, v_i)
            else:
                o = self._torch_sdpa(q, k_i, v_i, attn_mask, kpm)
            
            outs.append(self._combine_heads(o))
        
        S = torch.stack(outs, dim=0)
        w = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        combined = (S * w).sum(dim=0)
        ctx = combined[:, 0]
        
        return ctx, None

    def _probsparse_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        ProbSparse attention from Informer.
        Selects top-k queries based on sparsity measurement.
        """
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        
        B, H, Tq, D = q.shape
        _, _, Tk, _ = k.shape
        
        # Repeat KV for GQA
        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)
        
        # Sample keys for sparsity measurement
        U_part = min(self.sampling_factor * int(math.log(Tk)), Tk)
        u = torch.randint(0, Tk, (U_part,), device=q.device)
        k_sample = k_full[:, :, u, :]  # [B, H, U_part, D]
        
        # Compute sparsity scores: M(q) = max(QK^T) - mean(QK^T)
        q_k_sample = torch.einsum('bhqd,bhkd->bhqk', q, k_sample)  # [B, H, Tq, U_part]
        M = q_k_sample.max(dim=-1)[0] - q_k_sample.mean(dim=-1)  # [B, H, Tq]
        
        # Select top-k queries (for decoder, usually just 1 query, so this is simplified)
        top_k = self.top_k or min(25, Tq)
        M_top, top_idx = M.topk(min(top_k, Tq), dim=-1, sorted=False)  # [B, H, top_k]
        
        # Gather top queries
        q_reduced = torch.gather(
            q, 2, 
            top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # [B, H, top_k, D]
        
        # Standard attention on reduced queries
        scores = torch.einsum('bhqd,bhkd->bhqk', q_reduced, k_full) * self.scale_factor
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(B, 1, 1, Tk), 
                float("-inf")
            )
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v_full)
        
        # For single query, just take first
        ctx = self._combine_heads(out)[:, 0]
        
        return ctx, None

    def _linear_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        Linear attention using kernel feature maps.
        Complexity: O(TDd) instead of O(T^2D)
        """
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        
        B, H, _, D = q.shape
        _, _, Tk, _ = k.shape
        
        # Repeat KV
        k = self._repeat_kv(k)  # [B, H, Tk, D]
        v = self._repeat_kv(v)
        
        # Apply feature map: φ(x) = elu(x) + 1
        q = F.elu(q) + 1  # [B, H, 1, D]
        k = F.elu(k) + 1  # [B, H, Tk, D]
        
        # Linear attention: (Q' * (K'^T * V)) / (Q' * K'^T * 1)
        # KV = sum_i k_i^T v_i
        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, H, D, D]
        
        # Z = sum_i k_i^T
        z = k.sum(dim=2)  # [B, H, D]
        
        # Output
        num = torch.einsum('bhqd,bhde->bhqe', q, kv)  # [B, H, 1, D]
        denom = torch.einsum('bhqd,bhd->bhq', q, z).unsqueeze(-1)  # [B, H, 1, 1]
        
        out = num / (denom + 1e-6)  # [B, H, 1, D]
        ctx = self._combine_heads(out)[:, 0]
        ctx = self.out_proj(ctx)
        
        return ctx, None

    def _sliding_window_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        Sliding window attention: each query only attends to local window.
        """
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        
        B, H, Tq, D = q.shape
        _, _, Tk, _ = k.shape
        
        # Repeat KV
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # For single query (Tq=1), attend to last window_size tokens
        win = min(self.window_size, Tk)
        k_win = k[:, :, -win:, :]
        v_win = v[:, :, -win:, :]
        
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k_win) * self.scale_factor
        
        if key_padding_mask is not None:
            mask_win = key_padding_mask[:, -win:]
            scores = scores.masked_fill(mask_win.view(B, 1, 1, -1), float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v_win)
        ctx = self._combine_heads(out)[:, 0]
        
        return ctx, None

    def _lsh_attention(self, dec_h: torch.Tensor, enc: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        LSH (Locality-Sensitive Hashing) attention.
        Uses random projections to hash queries and keys into buckets.
        """
        q1 = dec_h.unsqueeze(1)
        q, k, v = self._project_qkv(q1, enc)
        
        B, H, Tq, D = q.shape
        _, _, Tk, _ = k.shape
        
        # Repeat KV
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Simple LSH: use random rotation + sign
        # rotations: [n_hashes, D, D//2]
        q_rot = torch.einsum('bhqd,hdk->bhqhk', q, self.random_rotations)  # [B,H,1,n_hashes,D//2]
        k_rot = torch.einsum('bhkd,hdk->bhkhj', k, self.random_rotations)  # [B,H,Tk,n_hashes,D//2]
        
        # Hash: sign of rotations
        q_hash = (q_rot > 0).long()  # [B,H,1,n_hashes,D//2]
        k_hash = (k_rot > 0).long()  # [B,H,Tk,n_hashes,D//2]
        
        # Convert binary to bucket ID (simplified: just use first few bits)
        q_buckets = (q_hash * torch.tensor([2**i for i in range(min(8, D//2))], 
                                           device=q.device)).sum(-1)  # [B,H,1,n_hashes]
        k_buckets = (k_hash * torch.tensor([2**i for i in range(min(8, D//2))], 
                                           device=k.device)).sum(-1)  # [B,H,Tk,n_hashes]
        
        # Find matching buckets (attend only to same bucket)
        # For simplicity: attend to union of buckets across hashes
        matching_mask = torch.zeros(B, H, 1, Tk, dtype=torch.bool, device=q.device)
        
        for h in range(self.n_hashes):
            q_b = q_buckets[:, :, :, h]  # [B,H,1]
            k_b = k_buckets[:, :, :, h]  # [B,H,Tk]
            matching_mask |= (q_b.unsqueeze(-1) == k_b.unsqueeze(2))  # [B,H,1,Tk]
        
        # Standard attention with hash-based masking
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale_factor
        scores = scores.masked_fill(~matching_mask, float("-inf"))
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, Tk), float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        ctx = self._combine_heads(out)[:, 0]
        ctx = self.out_proj(ctx)
        
        return ctx, None

    # ----- Forward -----

    def forward(
        self,
        decoder_hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        encoder_outputs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            decoder_hidden: [L,B,H] or (h_n, c_n) from RNN
            encoder_outputs: [B,T,D]
            attn_mask: [1, T] boolean mask (True = masked)
            key_padding_mask: [B, T] boolean mask (True = masked)
        
        Returns:
            attended: [B,D]
            attn_weights: [B,T] or None
        """
        dec_h = self._last_hidden(decoder_hidden)
        enc = self.encoder_projection(encoder_outputs)

        # Route to appropriate attention method
        if self.method == "dot":
            ctx, w = self._dot_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "mha":
            ctx, w = self._mha_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "multiscale":
            ctx, w = self._multiscale_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "probsparse":
            ctx, w = self._probsparse_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "linear":
            ctx, w = self._linear_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "sliding_window":
            ctx, w = self._sliding_window_attention(dec_h, enc, attn_mask, key_padding_mask)
        elif self.method == "lsh":
            ctx, w = self._lsh_attention(dec_h, enc, attn_mask, key_padding_mask)
        else:
            raise ValueError(f"Unknown attention method: {self.method}")

        # Combine context with decoder hidden
        out = self.combined_layer(torch.cat([ctx, dec_h], dim=1))
        
        if self.use_residual_ln:
            out = self.post_ln(out + dec_h)
        
        return torch.tanh(out), w
