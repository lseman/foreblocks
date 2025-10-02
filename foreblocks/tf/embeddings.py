import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# ------------------------- Triton availability -------------------------
try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    triton = None
    tl = None


# =============================== PositionalEncoding (Sinusoidal) ===============================
class PositionalEncoding(nn.Module):
    """
    Optimized sinusoidal positional encoding with caching, scaling, and dynamic fallback.
    Mask/shape safe; dtype/device aligned with input.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 10000,
        scale: float = 1.0,
        cache_limit: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.max_len = max_len
        self.cache_limit = cache_limit

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self._build_pe_cache(d_model, max_len)

        # keyed by d_model, stores [1, T, D] tables
        self._pe_cache: Dict[int, torch.Tensor] = {}

    def _build_pe_cache(self, d_model: int, max_len: int):
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # [D/2]

        pe = torch.empty(max_len, d_model, dtype=torch.float32)
        angles = position * div_term  # [T,D/2]
        pe[:, 0::2] = torch.sin(angles)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(angles[:, :-1])
            pe[:, -1] = 0
        else:
            pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1,T,D]

    def _create_pe_for_dim(self, d_model: int, seq_len: int, device: torch.device) -> torch.Tensor:
        # cache longest built table per d_model
        if d_model in self._pe_cache:
            cached = self._pe_cache[d_model]
            if cached.size(1) >= seq_len:
                return cached[:, :seq_len].to(device)

        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model)
        )
        pe = torch.empty(seq_len, d_model, dtype=torch.float32, device=device)
        angles = position * div_term
        pe[:, 0::2] = torch.sin(angles)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(angles[:, :-1]); pe[:, -1] = 0
        else:
            pe[:, 1::2] = torch.cos(angles)

        if d_model <= 2048:
            if len(self._pe_cache) >= self.cache_limit:
                self._pe_cache.pop(next(iter(self._pe_cache)))
            self._pe_cache[d_model] = pe.unsqueeze(0)  # [1,T,D]
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D)
        pos: Optional positions [T] or [B,T] (long). If provided, uses embedding lookup.
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # choose / build table
        if D == self.d_model and T <= self.max_len:
            table = self.pe[:, :T].to(device=device)
        else:
            table = self._create_pe_for_dim(D, T, device)

        if pos is not None:
            # normalize pos -> [B,T]
            if pos.dim() == 1:
                pos = pos.unsqueeze(0).expand(B, -1)
            # base table for lookup
            base_table = (self.pe.squeeze(0) if D == self.d_model and self.pe.size(1) >= self.max_len
                          else table.squeeze(0))  # [Tmax, D]
            max_idx = int(pos.max().item())
            if base_table.size(0) <= max_idx:
                base_table = self._create_pe_for_dim(D, max_idx + 1, device).squeeze(0)
            pe = F.embedding(pos.to(device=device, dtype=torch.long), base_table)  # [B,T,D]
        else:
            pe = table.expand(B, -1, -1)  # [B,T,D]

        if pe.dtype != dtype:
            pe = pe.to(dtype)
        x = x.add(pe, alpha=self.scale)
        return self.dropout(x) if self.dropout else x


# =============================== InformerTimeEmbedding ===============================
class InformerTimeEmbedding(nn.Module):
    """Compact time embedding with projection, sane init, and simple normalization."""

    def __init__(self, d_model: int):
        super().__init__()
        embed_dim = min(d_model // 4, 64)
        self.hour_embed = nn.Embedding(24, embed_dim)
        self.weekday_embed = nn.Embedding(7, embed_dim)
        self.day_embed = nn.Embedding(32, embed_dim)
        self.month_embed = nn.Embedding(13, embed_dim)
        self.projection = nn.Linear(embed_dim * 4, d_model) if embed_dim * 4 != d_model else None
        self.norm_factor = 1.0 / math.sqrt(4.0)
        # init
        for emb in [self.hour_embed, self.weekday_embed, self.day_embed, self.month_embed]:
            nn.init.normal_((emb.weight), mean=0.0, std=0.02)

    def forward(self, time_feats: torch.Tensor) -> torch.Tensor:
        """
        time_feats: [B, T, 4] -> [month, weekday, hour, day]
        """
        month = torch.clamp(time_feats[..., 0].long(), 0, 12)
        weekday = torch.clamp(time_feats[..., 1].long(), 0, 6)
        hour = torch.clamp(time_feats[..., 2].long(), 0, 23)
        day = torch.clamp(time_feats[..., 3].long(), 0, 31)
        embs = torch.cat(
            [self.month_embed(month), self.weekday_embed(weekday),
             self.hour_embed(hour), self.day_embed(day)],
            dim=-1
        )
        if self.projection is not None:
            embs = self.projection(embs)
        return embs * self.norm_factor


# =============================== RotaryEmbedding / RoPE wrapper ===============================
class RotaryEmbedding(nn.Module):
    """
    RoPE with cached cos/sin, shape-safe broadcasting.
    """

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, max_seq_len: int):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        seq_idx = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        freqs = seq_idx * theta
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)  # [T, D/2]
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _get_freqs(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= self.max_seq_len:
            return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        seq_idx = torch.arange(seq_len, device=device).float().unsqueeze(1)
        freqs = seq_idx * theta
        return freqs.cos(), freqs.sin()

    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        x: [B,H,T,D], cos/sin: [T, D/2] or broadcastable to [B,H,T,D/2]
        """
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,D/2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_pos: Optional[torch.Tensor] = None,
        k_pos: Optional[torch.Tensor] = None,
        rotary_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: [B, H, T, D]
        q_pos/k_pos: Optional indices [T] or [B,T]
        """
        *_, q_len, head_dim = q.shape
        _, _, k_len, _ = k.shape
        rotary_dim = rotary_dim or min(self.dim, head_dim)
        assert rotary_dim % 2 == 0, "Rotary dimension must be even"

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        cos, sin = self._get_freqs(max(q_len, k_len), q.device)

        def idx(pos, cos_, sin_, L):
            if pos is not None:
                pos = pos.long()
                return cos_[pos], sin_[pos]
            return cos_[:L], sin_[:L]

        q_cos, q_sin = idx(q_pos, cos, sin, q_len)
        k_cos, k_sin = idx(k_pos, cos, sin, k_len)

        q_rot = self.apply_rotary_pos_emb(q_rot, q_cos, q_sin)
        k_rot = self.apply_rotary_pos_emb(k_rot, k_cos, k_sin)

        q_out = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() else q_rot
        k_out = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() else k_rot
        return q_out, k_out

    def clear_cache(self, full: bool = False):
        if full:
            if hasattr(self, "cos_cached"): del self.cos_cached
            if hasattr(self, "sin_cached"): del self.sin_cached


class RoPEPositionalEncoding(nn.Module):
    """
    Drop-in "positional encoding" interface that simply scales x.
    RoPE should be applied inside attention modules.
    """
    def __init__(self, d_model: int, max_len: int = 5000, scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.rope = RotaryEmbedding(d_model, max_seq_len=max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# =============================== Triton Kernels (keep) ===============================
if HAVE_TRITON:

    @triton.jit
    def fused_pe_addition_kernel(
        x_ptr, pe_ptr, output_ptr,
        scale_val: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused: out = x + pe * scale
        """
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        pe_vals = tl.load(pe_ptr + offsets, mask=mask, other=0.0)
        out = x_vals + pe_vals * scale_val
        tl.store(output_ptr + offsets, out, mask=mask)

    @triton.jit
    def low_rank_matmul_kernel(
        U_ptr, V_ptr, output_ptr,
        batch_size: tl.constexpr,
        seq_len: tl.constexpr,
        low_rank_dim: tl.constexpr,
        d_model: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Batched low-rank matmul:
          U: [B,T,r], V: [B,r,D] (or same per-batch), out: [B,T,D]
        3D grid: (M_blocks, N_blocks, B)
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        m0 = pid_m * BLOCK_M
        n0 = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, low_rank_dim, BLOCK_K):
            m_idx = m0 + tl.arange(0, BLOCK_M)[:, None]
            k_idx = k0 + tl.arange(0, BLOCK_K)[None, :]
            n_idx = n0 + tl.arange(0, BLOCK_N)[None, :]

            # U: [B,T,r] laid as contiguous B blocks of [T,r]
            U_ptrs = U_ptr + pid_b * (seq_len * low_rank_dim) + m_idx * low_rank_dim + k_idx
            umask = (m_idx < seq_len) & (k_idx < low_rank_dim)
            Ublk = tl.load(U_ptrs, mask=umask, other=0.0)

            # V: [B,r,D] laid as contiguous B blocks of [r,D]
            V_ptrs = V_ptr + pid_b * (low_rank_dim * d_model) + k_idx * d_model + n_idx
            vmask = (k_idx < low_rank_dim) & (n_idx < d_model)
            Vblk = tl.load(V_ptrs, mask=vmask, other=0.0)

            acc += tl.dot(Ublk, Vblk)

        m_idx = m0 + tl.arange(0, BLOCK_M)[:, None]
        n_idx = n0 + tl.arange(0, BLOCK_N)[None, :]
        out_ptrs = output_ptr + pid_b * (seq_len * d_model) + m_idx * d_model + n_idx
        omask = (m_idx < seq_len) & (n_idx < d_model)
        tl.store(out_ptrs, acc, mask=omask)

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
        """
        Gather rows of embeddings_ptr by positions_ptr (B,T) -> out (B,T,D)
        """
        pid = tl.program_id(axis=0)
        total = batch_size * seq_len
        start = pid * BLOCK_SIZE
        if start >= total:
            return

        for i in range(BLOCK_SIZE):
            linear = start + i
            if linear >= total:
                break
            b = linear // seq_len
            t = linear % seq_len

            pos = tl.load(positions_ptr + b * seq_len + t)
            if (pos < 0) | (pos >= max_len):
                continue

            # copy row in chunks of 32
            for d0 in range(0, embed_dim, 32):
                d_end = tl.minimum(d0 + 32, embed_dim)
                d_off = d0 + tl.arange(0, 32)
                d_mask = d_off < d_end

                e_ptrs = embeddings_ptr + pos * embed_dim + d_off
                vals = tl.load(e_ptrs, mask=d_mask, other=0.0)

                out_ptrs = output_ptr + linear * embed_dim + d_off
                tl.store(out_ptrs, vals, mask=d_mask)


# =============================== LearnablePositionalEncoding ===============================
class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding with optional low-rank factorization and Triton fast paths.
    Triton paths preserved and used with sane thresholds; purely drop-in.
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
        self.use_triton = use_triton and HAVE_TRITON and torch.cuda.is_available()
        self.memory_efficient = memory_efficient

        if use_triton and not HAVE_TRITON:
            print("Warning: Triton requested but not available. Falling back to PyTorch ops.")

        # Parameters
        if low_rank_dim is None:
            if memory_efficient:
                pe_data = self._init_pe_optimized(initialization, (max_len, d_model))  # [T,D]
                self.pe = Parameter(pe_data)
            else:
                self.pe = Parameter(self._init_pe(initialization, (1, max_len, d_model)))  # [1,T,D]
        else:
            self.pe_proj_U = Parameter(self._init_pe(initialization, (max_len, low_rank_dim)))  # [T,r]
            self.pe_proj_V = Parameter(self._init_pe(initialization, (low_rank_dim, d_model)))  # [r,D]

        # Scaling
        if scale_strategy == "learnable":
            init_scale = scale_value or math.sqrt(d_model)
            self.scale = Parameter(torch.tensor(init_scale, dtype=torch.float32))
            self.scale_is_tensor = True
        elif scale_strategy == "fixed":
            scale_val = scale_value or math.sqrt(d_model)
            self.register_buffer("scale", torch.tensor(scale_val, dtype=torch.float32))
            self.scale_is_tensor = False
        else:
            self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
            self.scale_is_tensor = False

        # Norm & dropout
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else None

        # Small cache
        self._pe_cache = {}
        self._cache_size_limit = 8

    # ---------------- inits ----------------
    def _init_pe(self, mode: str, shape: tuple) -> torch.Tensor:
        if mode == "normal":
            return torch.randn(shape) * math.sqrt(2.0 / shape[-1])
        if mode == "uniform":
            bound = math.sqrt(6.0 / shape[-1]); return torch.empty(shape).uniform_(-bound, bound)
        if mode == "zero":
            return torch.zeros(shape)
        if mode == "xavier":
            return torch.empty(shape).normal_(0, math.sqrt(2.0 / (shape[-2] + shape[-1])))
        return torch.randn(shape) * 0.02

    def _init_pe_optimized(self, mode: str, shape: tuple) -> torch.Tensor:
        return self._init_pe(mode, shape)

    # ---------------- standard/low-rank getters ----------------
    def _get_pe_standard(self, B: int, T: int, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if positions is None:
            if self.memory_efficient:
                pe = self.pe[:T].unsqueeze(0)  # [1,T,D]
                return pe.expand(B, -1, -1)
            return self.pe[:, :T].expand(B, -1, -1)  # [1,T,D] -> [B,T,D]
        # positions provided
        if self.use_triton and positions.is_cuda and B * T > 1024 and HAVE_TRITON:
            return self._triton_embedding_lookup(positions)
        if self.memory_efficient:
            return F.embedding(positions, self.pe)  # weight [Tmax,D]
        return F.embedding(positions, self.pe.squeeze(0))

    def _get_pe_low_rank(self, B: int, T: int, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if positions is None:
            U = self.pe_proj_U[:T]    # [T,r]
            V = self.pe_proj_V        # [r,D]
            if self.use_triton and U.is_cuda and T * self.d_model > 4096 and HAVE_TRITON:
                return self._triton_low_rank_matmul(U, V, B, T)  # [B,T,D]
            pe = U @ V  # [T,D]
            return pe.unsqueeze(0).expand(B, -1, -1)
        # positions provided
        U = F.embedding(positions, self.pe_proj_U)          # [B,T,r]
        V = self.pe_proj_V.unsqueeze(0).expand(B, -1, -1)   # [B,r,D]
        if self.use_triton and U.is_cuda and B * T * self.d_model > 8192 and HAVE_TRITON:
            return self._triton_batch_matmul(U, V)          # [B,T,D]
        return torch.bmm(U, V)

    # ---------------- Triton helpers ----------------
    def _triton_embedding_lookup(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Triton-accelerated embedding lookup for standard learnable PE.
        positions: [B,T]
        """
        if not HAVE_TRITON:
            return F.embedding(positions, self.pe if self.memory_efficient else self.pe.squeeze(0))

        B, T = positions.shape
        out = torch.empty(B, T, self.d_model, device=positions.device, dtype=self.pe.dtype)
        total = B * T
        BLOCK = min(256, triton.next_power_of_2(total))
        grid = (triton.cdiv(total, BLOCK),)
        pe_data = self.pe if self.memory_efficient else self.pe.squeeze(0)  # [Tmax, D]

        embedding_lookup_kernel[grid](
            positions,
            pe_data,
            out,
            batch_size=B,
            seq_len=T,
            embed_dim=self.d_model,
            max_len=self.max_len,
            BLOCK_SIZE=BLOCK,
        )
        return out

    def _triton_low_rank_matmul(self, U: torch.Tensor, V: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Use 3D-grid Triton kernel for batched (B,T,D) from U[T,r], V[r,D].
        We expand U to [B,T,r] and duplicate V per batch to match kernel layout.
        """
        if not HAVE_TRITON:
            pe = U @ V
            return pe.unsqueeze(0).expand(B, -1, -1)

        # Expand logically; .contiguous() to ensure flat addressing in kernel
        U_b = U.unsqueeze(0).expand(B, -1, -1).contiguous()           # [B,T,r]
        V_b = V.unsqueeze(0).expand(B, -1, -1).contiguous()           # [B,r,D]
        out = torch.empty(B, T, self.d_model, device=U.device, dtype=U.dtype)

        BLOCK_M = min(64, triton.next_power_of_2(T))
        BLOCK_N = min(64, triton.next_power_of_2(self.d_model))
        BLOCK_K = min(32, triton.next_power_of_2(self.low_rank_dim))

        grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(self.d_model, BLOCK_N), B)

        low_rank_matmul_kernel[grid](
            U_b, V_b, out,
            batch_size=B,
            seq_len=T,
            low_rank_dim=self.low_rank_dim,
            d_model=self.d_model,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return out

    def _triton_batch_matmul(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Batched matmul U[B,T,r] @ V[B,r,D] via the same kernel.
        """
        if not HAVE_TRITON:
            return torch.bmm(U, V)

        B, T, r = U.shape
        out = torch.empty(B, T, self.d_model, device=U.device, dtype=U.dtype)

        BLOCK_M = min(64, triton.next_power_of_2(T))
        BLOCK_N = min(64, triton.next_power_of_2(self.d_model))
        BLOCK_K = min(32, triton.next_power_of_2(r))

        grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(self.d_model, BLOCK_N), B)

        low_rank_matmul_kernel[grid](
            U.contiguous(),
            V.contiguous(),
            out,
            batch_size=B,
            seq_len=T,
            low_rank_dim=r,
            d_model=self.d_model,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return out

    def _triton_fused_addition(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """
        Fused add: x + pe * scale (Triton).
        """
        if not HAVE_TRITON:
            # PyTorch fallback with identical semantics
            scale_val = float(self.scale.item()) if isinstance(self.scale, torch.Tensor) else float(self.scale)
            return x + pe * scale_val

        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        scale_val = float(self.scale.item()) if isinstance(self.scale, torch.Tensor) else float(self.scale)

        fused_pe_addition_kernel[grid](
            x.contiguous().view(-1),
            pe.contiguous().view(-1),
            out.view(-1),
            scale_val=scale_val,
            n_elements=n,
            BLOCK_SIZE=BLOCK,
        )
        return out

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        # cache for common sizes without custom positions
        cache_key = (T, positions is not None)
        if cache_key in self._pe_cache and positions is None:
            pe = self._pe_cache[cache_key]
            if pe.size(0) != B:
                pe = pe.expand(B, -1, -1)
        else:
            if self.low_rank_dim is None:
                pe = self._get_pe_standard(B, T, positions)  # [B,T,D]
            else:
                pe = self._get_pe_low_rank(B, T, positions)  # [B,T,D]
            if len(self._pe_cache) < self._cache_size_limit and positions is None and T <= 2048:
                self._pe_cache[cache_key] = pe[:1] if B > 1 else pe

        if self.norm_strategy == "pre_add" and self.layer_norm:
            x = self.layer_norm(x)

        # align dtype/device
        if pe.device != device:
            pe = pe.to(device)
        if pe.dtype != dtype:
            pe = pe.to(dtype)

        # Triton fused add for big tensors
        if self.use_triton and x.is_cuda and (B * T * self.d_model > 4096) and HAVE_TRITON:
            x = self._triton_fused_addition(x, pe)
        else:
            if self.scale_is_tensor:
                x = x + pe * self.scale.to(dtype=dtype, device=device)
            else:
                x = x + pe * float(self.scale.item())

        if self.norm_strategy == "post_add" and self.layer_norm:
            x = self.layer_norm(x)

        return self.dropout(x) if self.dropout else x
