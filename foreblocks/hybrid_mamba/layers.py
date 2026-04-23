from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import causal_depthwise_conv1d
from .ops import dt_prep
from .ops import fused_out
from .ops import grouped_ssd_scan
from .ops import selective_scan


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, d_inner: int, kernel_size: int):
        super().__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=kernel_size,
            groups=d_inner,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        weight = self.conv.weight.view(self.d_inner, self.kernel_size).contiguous()
        bias = self.conv.bias.contiguous() if self.conv.bias is not None else None
        x = causal_depthwise_conv1d(x, weight, bias)
        return x.transpose(1, 2)


class RMSNormWeightOnly(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        raise RuntimeError("Use fused_out(...) instead.")


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al. 2021, RoFormer).

    Caches cos/sin tables up to ``max_seq_len`` and extends on demand.
    Applied to Q and K before scaled dot-product attention.

    Args:
        head_dim: Dimension of each attention head. Must be even.
        base: Frequency base (default 10 000, as in the original paper).
        max_seq_len: Pre-built cache length. Extended automatically if exceeded.
    """

    def __init__(self, head_dim: int, base: int = 10_000, max_seq_len: int = 8192):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._seq_len_cached:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos()[None, None]   # (1, 1, T, head_dim)
        self._sin_cached = emb.sin()[None, None]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q and K tensors of shape ``(B, H, T, head_dim)``."""
        seqlen = q.shape[2]
        self._build_cache(seqlen)
        cos = self._cos_cached[:, :, :seqlen].to(dtype=q.dtype, device=q.device)
        sin = self._sin_cached[:, :, :seqlen].to(dtype=q.dtype, device=q.device)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot

    def apply_at_pos(
        self, q: torch.Tensor, k: torch.Tensor, pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to single-token tensors ``(B, H, head_dim)`` at position *pos*.

        Used during recurrent (step-by-step) inference with a KV cache.
        """
        self._build_cache(pos + 1)
        cos = self._cos_cached[0, 0, pos].to(dtype=q.dtype, device=q.device)
        sin = self._sin_cached[0, 0, pos].to(dtype=q.dtype, device=q.device)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Sliding-Window Attention with RoPE + GQA + attention sink
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Causal sliding-window self-attention with RoPE, optional GQA, and optional
    attention sink tokens.

    Args:
        d_model: Model dimension.
        num_heads: Number of query heads.
        n_kv_heads: Number of key/value heads for Grouped Query Attention.
            ``None`` (default) means standard MHA (``n_kv_heads == num_heads``).
            Must evenly divide ``num_heads``.
        window_size: Each token attends to at most this many past tokens
            (including itself).
        dropout: Attention dropout probability during training.
        bias: Whether Q/K/V/out projections use bias.
        rope_base: Frequency base for RoPE.
        max_seq_len: Pre-built RoPE cache length.
        n_sink_tokens: Number of leading sequence positions that every token may
            attend to in addition to its local window (StreamingLLM, 2023).
            Set to 1–4 to prevent information starvation at window boundaries.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        n_kv_heads: int | None = None,
        window_size: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
        rope_base: int = 10_000,
        max_seq_len: int = 8192,
        n_sink_tokens: int = 0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        n_kv_heads = n_kv_heads or num_heads
        if num_heads % n_kv_heads != 0:
            raise ValueError("num_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = num_heads // n_kv_heads
        self.window_size = window_size
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.n_sink_tokens = n_sink_tokens

        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base, max_seq_len=max_seq_len)

    def _sliding_mask(
        self, seqlen: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        rows = torch.arange(seqlen, device=device)
        cols = torch.arange(seqlen, device=device)
        rel = rows[:, None] - cols[None, :]
        allowed = (rel >= 0) & (rel < self.window_size)
        if self.n_sink_tokens > 0:
            sink = cols[None, :] < self.n_sink_tokens
            allowed = allowed | sink
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
        mask.masked_fill_(allowed, 0.0)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn_mask = self._sliding_mask(T, x.device, q.dtype)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# FeedForward block (SwiGLU)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise SwiGLU feed-forward block with pre-norm.

    Parameter count is set by ``expansion`` (default ``8/3``) rounded up to the
    nearest multiple of 64, matching the LLaMA / Mistral convention.

    Args:
        d_model: Input/output dimension.
        expansion: Inner dim multiplier (default ``8/3``).
        dropout: Dropout after the output projection.
    """

    def __init__(self, d_model: int, expansion: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        d_inner = max(int(d_model * expansion), 64)
        d_inner = (d_inner + 63) // 64 * 64
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_inner, bias=False)
        self.w2 = nn.Linear(d_model, d_inner, bias=False)
        self.out = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.drop(self.out(F.silu(self.w1(h)) * self.w2(h)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_dt_rank(d_model: int) -> int:
    return max(4, math.ceil(d_model / 16))


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _fused_out_2d(
    y: torch.Tensor,
    z: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CPU-safe fused_out for 2-D tensors ``(B, D)`` used in step() methods."""
    x = y * F.silu(z) + residual
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms * norm_weight


def _conv_step(
    u_raw: torch.Tensor,
    state_conv: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of a causal depthwise conv from a ring buffer.

    Args:
        u_raw: ``(B, D)`` current input.
        state_conv: ``(B, D, K-1)`` ring buffer of previous inputs.
        weight: ``(D, K)`` conv weights.
        bias: ``(D,)`` or ``None``.

    Returns:
        ``(u_out, new_state)`` where ``u_out`` is ``(B, D)``.
    """
    full = torch.cat([state_conv, u_raw.unsqueeze(-1)], dim=-1)  # (B, D, K)
    u = (full * weight.unsqueeze(0)).sum(-1)
    if bias is not None:
        u = u + bias
    return u, full[:, :, 1:]


# ---------------------------------------------------------------------------
# HybridMambaBlock — pure SSM block with optional pre-norm
# ---------------------------------------------------------------------------

class HybridMambaBlock(nn.Module):
    """Mamba-style SSM block (selective scan, dense per-feature A/D matrices).

    Args:
        d_model: Input/output dimension.
        d_inner: Expanded inner dimension (default ``2 * d_model``).
        d_state: SSM state dimension.
        d_conv: Causal depthwise conv kernel size.
        dt_rank: Low-rank size for Δt projection. ``None`` → auto.
        dt_min / dt_max: Softplus clamp range for the time-step.
        use_cuda_scan: Use the CUDA selective-scan kernel when available.
        use_pre_norm: Apply a LayerNorm on the input before projection.
            Recommended ``True`` (default) for training stability in deep stacks.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        dt_min: float = 1e-4,
        dt_max: float = 1.0,
        use_cuda_scan: bool = True,
        use_pre_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner or 2 * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or _auto_dt_rank(d_model)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_cuda_scan = use_cuda_scan

        self.pre_norm = nn.LayerNorm(d_model) if use_pre_norm else nn.Identity()

        total_out = 2 * self.d_inner + self.dt_rank + 2 * self.d_inner * self.d_state
        self.in_proj = nn.Linear(d_model, total_out, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner, kernel_size=d_conv)

        self.A_log = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        self.Dskip = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))

        self.norm = RMSNormWeightOnly(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.residual_proj = (
            nn.Identity()
            if self.d_model == self.d_inner
            else nn.Linear(self.d_model, self.d_inner, bias=False)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
        with torch.no_grad():
            base = torch.arange(1, self.d_state + 1, device=self.A_log.device, dtype=self.A_log.dtype)
            self.A_log.copy_(base.log().unsqueeze(0).expand(self.d_inner, -1))
            self.Dskip.fill_(1.0)
            dt = torch.rand(self.d_inner, device=self.dt_bias.device, dtype=self.dt_bias.dtype)
            dt = dt * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
            self.dt_bias.copy_(_inverse_softplus(dt.exp()))

    def _split_proj(self, p: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size, seqlen, _ = p.shape
        D = self.d_inner
        N = self.d_state
        z, u, dt_hidden, Bflat, Cflat = torch.split(
            p, [D, D, self.dt_rank, D * N, D * N], dim=-1
        )
        Bpar = Bflat.reshape(batch_size, seqlen, D, N)
        Cpar = Cflat.reshape(batch_size, seqlen, D, N)
        return z, u, dt_hidden, Bpar, Cpar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        p = self.in_proj(self.pre_norm(x))
        z, u, dt_hidden, Bpar, Cpar = self._split_proj(p)
        u = self.conv(u)

        dt_raw = self.dt_proj(dt_hidden)
        dt = dt_prep(dt_raw, self.dt_bias, dt_min=self.dt_min, dt_max=self.dt_max)
        A = -torch.exp(self.A_log)

        y = selective_scan(
            u=u, dt=dt, A=A, Bpar=Bpar, Cpar=Cpar,
            Dskip=self.Dskip, use_cuda_kernel=self.use_cuda_scan,
        )

        residual_inner = self.residual_proj(residual)
        y = fused_out(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y)

    # ------------------------------------------------------------------
    # Recurrent inference
    # ------------------------------------------------------------------

    def make_state(self, batch: int, device=None, dtype=None) -> dict[str, torch.Tensor]:
        """Return a fresh recurrent state for *batch* sequences."""
        return {
            "conv": torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype),
            "ssm":  torch.zeros(batch, self.d_inner, self.d_state,    device=device, dtype=dtype),
        }

    def step(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Single-token recurrent forward.

        Args:
            x: ``(B, d_model)``
            state: dict from :meth:`make_state`, mutated in-place.

        Returns:
            ``(B, d_model)``
        """
        B = x.shape[0]
        D, N = self.d_inner, self.d_state

        p = self.in_proj(self.pre_norm(x))                          # (B, total_out)
        z, u_raw, dt_hidden, Bflat, Cflat = torch.split(
            p, [D, D, self.dt_rank, D * N, D * N], dim=-1
        )
        Bpar = Bflat.reshape(B, D, N)
        Cpar = Cflat.reshape(B, D, N)

        weight = self.conv.conv.weight.view(D, self.d_conv)
        u, state["conv"] = _conv_step(u_raw, state["conv"], weight, self.conv.conv.bias)

        dt_raw = self.dt_proj(dt_hidden)                            # (B, D)
        dt = F.softplus(dt_raw + self.dt_bias).clamp(self.dt_min, self.dt_max)

        A = -torch.exp(self.A_log)                                  # (D, N)
        A_disc = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))       # (B, D, N)

        h = state["ssm"].to(dtype=A_disc.dtype)
        h = A_disc * h + dt.unsqueeze(-1) * Bpar * u.unsqueeze(-1)  # (B, D, N)
        y = (Cpar * h).sum(-1) + self.Dskip * u                     # (B, D)
        state["ssm"] = h.detach()

        residual_inner = self.residual_proj(x)
        y_normed = _fused_out_2d(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y_normed)


# ---------------------------------------------------------------------------
# StructuredStateSpaceDualityBranch — multi-head SSD (Mamba-2 style)
# ---------------------------------------------------------------------------

class StructuredStateSpaceDualityBranch(nn.Module):
    """Multi-head Structured State Space Duality (SSD) branch.

    Uses per-head A and D matrices with a grouped scan (Dao & Gu 2024).
    Intended to be used as the SSM sub-module inside ``HybridMamba2Block``,
    which handles pre-normalisation before calling this module.

    Args:
        d_model: Input/output dimension.
        d_inner: Expanded inner dimension (default ``2 * d_model``).
        d_state: SSM state dimension per head.
        d_conv: Causal depthwise conv kernel size.
        dt_rank: Low-rank size for Δt. ``None`` → auto.
        num_heads: Number of SSM heads. ``d_inner`` must be divisible by this.
        dt_min / dt_max: Softplus clamp range for the time-step.
        use_gated_delta: Gate the state update delta per head with a learned
            sigmoid (adds ``num_heads`` extra parameters). Off by default.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        num_heads: int = 8,
        dt_min: float = 1e-4,
        dt_max: float = 1.0,
        use_gated_delta: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner or 2 * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or _auto_dt_rank(d_model)
        self.num_heads = num_heads
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_gated_delta = use_gated_delta
        if self.d_inner % num_heads != 0:
            raise ValueError("d_inner must be divisible by num_heads")

        self.head_dim = self.d_inner // self.num_heads
        extra = self.num_heads if use_gated_delta else 0
        total_out = 2 * self.d_inner + self.dt_rank + 2 * self.num_heads * self.d_state + extra
        self.in_proj = nn.Linear(d_model, total_out, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.num_heads, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner, kernel_size=d_conv)

        self.A_log = nn.Parameter(torch.empty(self.num_heads, self.d_state))
        self.Dskip = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))

        self.norm = RMSNormWeightOnly(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.residual_proj = (
            nn.Identity()
            if self.d_model == self.d_inner
            else nn.Linear(self.d_model, self.d_inner, bias=False)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
        with torch.no_grad():
            base = torch.arange(1, self.d_state + 1, device=self.A_log.device, dtype=self.A_log.dtype)
            self.A_log.copy_(base.log().unsqueeze(0).expand(self.num_heads, -1))
            self.Dskip.fill_(1.0)
            dt = torch.rand(self.num_heads, device=self.dt_bias.device, dtype=self.dt_bias.dtype)
            dt = dt * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
            self.dt_bias.copy_(_inverse_softplus(dt.exp()))

    def _split_proj(self, p: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size, seqlen, _ = p.shape
        h = self.num_heads
        n = self.d_state
        split_sizes = [self.d_inner, self.d_inner, self.dt_rank, h * n, h * n]
        if self.use_gated_delta:
            split_sizes.append(h)
        parts = torch.split(p, split_sizes, dim=-1)
        z, u, dt_hidden, Bflat, Cflat = parts[:5]
        delta_gate = parts[5] if self.use_gated_delta else None
        Bpar = Bflat.reshape(batch_size, seqlen, h, n)
        Cpar = Cflat.reshape(batch_size, seqlen, h, n)
        return z, u, dt_hidden, Bpar, Cpar, delta_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, seqlen, _ = x.shape

        p = self.in_proj(x)
        z, u, dt_hidden, Bpar, Cpar, delta_gate = self._split_proj(p)
        u = self.conv(u)

        dt_raw = self.dt_proj(dt_hidden)
        dt = dt_prep(dt_raw, self.dt_bias, dt_min=self.dt_min, dt_max=self.dt_max)
        A = -torch.exp(self.A_log)

        u_heads = u.reshape(batch_size, seqlen, self.num_heads, self.head_dim)
        y = grouped_ssd_scan(
            u=u_heads, dt=dt, A=A, Bpar=Bpar, Cpar=Cpar,
            Dskip=self.Dskip, delta_gate=delta_gate,
        ).reshape(batch_size, seqlen, self.d_inner)

        residual_inner = self.residual_proj(residual)
        y = fused_out(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y)

    # ------------------------------------------------------------------
    # Recurrent inference
    # ------------------------------------------------------------------

    def make_state(self, batch: int, device=None, dtype=None) -> dict[str, torch.Tensor]:
        """Return a fresh recurrent state for *batch* sequences."""
        return {
            "conv": torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype),
            "ssm":  torch.zeros(batch, self.num_heads, self.head_dim, self.d_state, device=device, dtype=dtype),
        }

    def step(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Single-token recurrent forward.

        Args:
            x: ``(B, d_model)``
            state: dict from :meth:`make_state`, mutated in-place.

        Returns:
            ``(B, d_model)``
        """
        B = x.shape[0]
        H, P, N = self.num_heads, self.head_dim, self.d_state
        D = self.d_inner

        split_sizes = [D, D, self.dt_rank, H * N, H * N]
        if self.use_gated_delta:
            split_sizes.append(H)
        parts = torch.split(self.in_proj(x), split_sizes, dim=-1)
        z, u_raw, dt_hidden, Bflat, Cflat = parts[:5]
        delta_gate_raw = parts[5] if self.use_gated_delta else None

        Bpar = Bflat.reshape(B, H, N)
        Cpar = Cflat.reshape(B, H, N)

        weight = self.conv.conv.weight.view(D, self.d_conv)
        u, state["conv"] = _conv_step(u_raw, state["conv"], weight, self.conv.conv.bias)

        dt_raw = self.dt_proj(dt_hidden)                             # (B, H)
        dt = F.softplus(dt_raw + self.dt_bias).clamp(self.dt_min, self.dt_max)

        A = -torch.exp(self.A_log)                                   # (H, N)
        h = state["ssm"].to(dtype=dt.dtype)                          # (B, H, P, N)

        decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))         # (B, H, N)
        decayed = decay.unsqueeze(-2) * h                            # (B, H, P, N)

        u_heads = u.reshape(B, H, P)
        # outer product → (B, H, P, N)
        delta = dt.unsqueeze(-1).unsqueeze(-1) * Bpar.unsqueeze(-2) * u_heads.unsqueeze(-1)

        if delta_gate_raw is not None:
            gate = torch.sigmoid(delta_gate_raw).unsqueeze(-1).unsqueeze(-1)
            h_new = decayed + gate * delta
        else:
            h_new = decayed + delta

        y_heads = (Cpar.unsqueeze(-2) * h_new).sum(-1) + self.Dskip.unsqueeze(0) * u_heads
        y = y_heads.reshape(B, D)
        state["ssm"] = h_new.detach()

        residual_inner = self.residual_proj(x)
        y_normed = _fused_out_2d(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y_normed)


# ---------------------------------------------------------------------------
# HybridMamba2Block — SSD branch + sliding-window attention, learned gate
# ---------------------------------------------------------------------------

class HybridMamba2Block(nn.Module):
    """Hybrid block that fuses an SSD (Mamba-2) branch with sliding-window
    attention via a learned per-channel gate.

    Architecture::

        ssm_out  = SSD( LayerNorm(x) )
        attn_out = SlidingWindowAttention( LayerNorm(x) )
        gate     = sigmoid( Linear( LayerNorm(x) ) )
        mixed    = gate * ssm_out + (1 − gate) * attn_out
        output   = out_proj( LayerNorm(mixed) )

    Args:
        d_model: Model dimension.
        d_inner: SSM inner (expanded) dimension.
        d_state: SSM state dimension.
        d_conv: Causal conv kernel size for the SSM branch.
        dt_rank: Low-rank Δt size for the SSM branch.
        num_heads: Heads for both the SSM branch and attention.
        n_kv_heads: KV heads for GQA in the attention branch. ``None`` → MHA.
        window_size: Attention causal window (tokens).
        attn_dropout: Attention dropout during training.
        use_gated_delta: Enable per-head delta gating in the SSD branch.
        rope_base: RoPE frequency base for the attention branch.
        max_seq_len: Pre-built RoPE cache length.
        n_sink_tokens: Attention sink tokens in the sliding-window attention.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        num_heads: int = 8,
        n_kv_heads: int | None = None,
        window_size: int = 128,
        attn_dropout: float = 0.0,
        use_gated_delta: bool = False,
        use_cuda_scan: bool = True,
        rope_base: int = 10_000,
        max_seq_len: int = 8192,
        n_sink_tokens: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        del use_cuda_scan  # kept for API compatibility; SSD branch ignores it

        self.ssm_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.mix_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)

        self.ssm = StructuredStateSpaceDualityBranch(
            d_model=d_model,
            d_inner=d_inner,
            d_state=d_state,
            d_conv=d_conv,
            dt_rank=dt_rank,
            num_heads=num_heads,
            use_gated_delta=use_gated_delta,
        )
        self.attn = SlidingWindowAttention(
            d_model=d_model,
            num_heads=num_heads,
            n_kv_heads=n_kv_heads,
            window_size=window_size,
            dropout=attn_dropout,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
            n_sink_tokens=n_sink_tokens,
        )
        self.mix_gate = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.mix_gate.weight)
        nn.init.zeros_(self.mix_gate.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ssm_out = self.ssm(self.ssm_norm(x))
        attn_out = self.attn(self.attn_norm(x))
        gate = torch.sigmoid(self.mix_gate(self.mix_norm(x)))
        mixed = gate * ssm_out + (1.0 - gate) * attn_out
        return self.out_proj(self.out_norm(mixed))

    # ------------------------------------------------------------------
    # Recurrent inference
    # ------------------------------------------------------------------

    def make_state(self, batch: int, device=None, dtype=None) -> dict:
        """Return a fresh recurrent state for *batch* sequences.

        The state contains SSM conv/hidden state plus a circular KV cache and
        an absolute position counter for RoPE.
        """
        state = self.ssm.make_state(batch, device=device, dtype=dtype)
        nkv = self.attn.n_kv_heads
        ws  = self.attn.window_size
        hd  = self.attn.head_dim
        state["kv_k"]   = torch.zeros(batch, nkv, ws, hd, device=device, dtype=dtype)
        state["kv_v"]   = torch.zeros(batch, nkv, ws, hd, device=device, dtype=dtype)
        state["kv_pos"] = 0   # next write slot in circular buffer
        state["kv_len"] = 0   # valid entries (capped at window_size)
        state["abs_pos"] = 0  # absolute position counter for RoPE
        return state

    def _attn_step(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        B = x.shape[0]
        attn = self.attn

        q = attn.q_proj(x).reshape(B, attn.num_heads,  attn.head_dim)
        k = attn.k_proj(x).reshape(B, attn.n_kv_heads, attn.head_dim)
        v = attn.v_proj(x).reshape(B, attn.n_kv_heads, attn.head_dim)

        q, k = attn.rope.apply_at_pos(q, k, state["abs_pos"])
        state["abs_pos"] += 1

        buf = state["kv_pos"]
        state["kv_k"][:, :, buf] = k.detach()
        state["kv_v"][:, :, buf] = v.detach()
        state["kv_pos"] = (buf + 1) % attn.window_size
        state["kv_len"] = min(state["kv_len"] + 1, attn.window_size)

        valid = state["kv_len"]
        k_ctx = state["kv_k"] if valid == attn.window_size else state["kv_k"][:, :, :valid]
        v_ctx = state["kv_v"] if valid == attn.window_size else state["kv_v"][:, :, :valid]

        if attn.n_rep > 1:
            k_ctx = k_ctx.repeat_interleave(attn.n_rep, dim=1)
            v_ctx = v_ctx.repeat_interleave(attn.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q.unsqueeze(2), k_ctx, v_ctx, is_causal=False
        )  # (B, Hq, 1, D)
        y = y.squeeze(2).reshape(B, attn.d_model)
        return attn.out_proj(y)

    def step(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        """Single-token recurrent forward.

        Args:
            x: ``(B, d_model)``
            state: dict from :meth:`make_state`, mutated in-place.

        Returns:
            ``(B, d_model)``
        """
        ssm_out  = self.ssm.step(self.ssm_norm(x), state)
        attn_out = self._attn_step(self.attn_norm(x), state)
        gate     = torch.sigmoid(self.mix_gate(self.mix_norm(x)))
        mixed    = gate * ssm_out + (1.0 - gate) * attn_out
        return self.out_proj(self.out_norm(mixed))


# ---------------------------------------------------------------------------
# Tiny language model wrappers
# ---------------------------------------------------------------------------

class TinyHybridMambaLM(nn.Module):
    """Small language model built from stacked ``HybridMambaBlock`` layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        tie_embeddings: bool = True,
        use_pre_norm: bool = True,
        mlp_every_n: int = 0,
        ffn_expansion: float = 8 / 3,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                HybridMambaBlock(
                    d_model=d_model,
                    d_inner=2 * d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                    use_cuda_scan=True,
                    use_pre_norm=use_pre_norm,
                )
                for _ in range(n_layers)
            ]
        )
        self._has_ffn: list[bool] = [
            mlp_every_n > 0 and (i + 1) % mlp_every_n == 0
            for i in range(n_layers)
        ]
        self.ffns = nn.ModuleList(
            FeedForward(d_model, expansion=ffn_expansion, dropout=ffn_dropout)
            if has else nn.Identity()
            for has in self._has_ffn
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk, ffn, has_ffn in zip(self.blocks, self.ffns, self._has_ffn):
            x = x + blk(x)
            if has_ffn:
                x = x + ffn(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def make_states(self, batch: int, device=None, dtype=None) -> list[dict]:
        """Return per-block recurrent states for token-by-token inference."""
        return [blk.make_state(batch, device=device, dtype=dtype) for blk in self.blocks]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Autoregressive generation using recurrent :meth:`step` mode.

        Args:
            input_ids: ``(B, T_prompt)`` token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Softmax temperature. ``0.0`` → greedy argmax.
            top_k: If > 0, restrict sampling to the top-k logits.

        Returns:
            ``(B, T_prompt + max_new_tokens)`` token IDs.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.parameters()).dtype
        states = self.make_states(B, device=device, dtype=dtype)

        def _step(tok: torch.Tensor) -> torch.Tensor:
            x = self.embed(tok)
            for i, (blk, ffn, has_ffn) in enumerate(zip(self.blocks, self.ffns, self._has_ffn)):
                x = x + blk.step(x, states[i])
                if has_ffn:
                    x = x + ffn(x)
            return self.final_norm(x)

        for t in range(input_ids.shape[1]):
            last = _step(input_ids[:, t])

        generated: list[torch.Tensor] = []
        logits = self.lm_head(last)
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_tok = logits.argmax(-1)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                next_tok = torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(1)
            generated.append(next_tok)
            logits = self.lm_head(_step(next_tok))

        if not generated:
            return input_ids
        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)


class TinyHybridMamba2LM(nn.Module):
    """Small language model alternating ``HybridMambaBlock`` (pure SSM) and
    ``HybridMamba2Block`` (SSM + sliding-window attention) layers.

    Attention blocks are placed every ``attn_every_n`` layers (0-indexed).
    Feed-forward (SwiGLU) blocks can optionally be interleaved with
    ``mlp_every_n``.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimension.
        n_layers: Total number of blocks.
        d_state: SSM state dimension.
        d_conv: Causal conv kernel size.
        dt_rank: Low-rank Δt size (``None`` → auto).
        num_heads: Attention and SSM head count.
        n_kv_heads: KV heads for GQA (``None`` → MHA). Passed to hybrid blocks.
        window_size: Attention causal window.
        attn_every_n: Place a hybrid (SSM + attention) block every N layers.
        tie_embeddings: Share embedding and LM-head weights.
        use_gated_delta: Enable per-head delta gating in SSD branches.
        use_pre_norm: Pre-norm in pure SSM blocks.
        rope_base: RoPE frequency base.
        max_seq_len: Pre-built RoPE cache length.
        n_sink_tokens: Attention sink tokens in hybrid blocks.
        mlp_every_n: Insert a SwiGLU FFN after every N-th block. ``0`` disables.
        ffn_expansion: FFN inner dim multiplier (default ``8/3``).
        ffn_dropout: Dropout inside FFN blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        num_heads: int = 8,
        n_kv_heads: int | None = None,
        window_size: int = 128,
        attn_every_n: int = 2,
        tie_embeddings: bool = True,
        use_gated_delta: bool = False,
        use_cuda_scan: bool = True,
        use_pre_norm: bool = True,
        rope_base: int = 10_000,
        max_seq_len: int = 8192,
        n_sink_tokens: int = 0,
        mlp_every_n: int = 0,
        ffn_expansion: float = 8 / 3,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        if attn_every_n < 1:
            raise ValueError("attn_every_n must be >= 1")

        self.embed = nn.Embedding(vocab_size, d_model)
        blocks = []
        for idx in range(n_layers):
            if idx % attn_every_n == 0:
                block = HybridMamba2Block(
                    d_model=d_model,
                    d_inner=2 * d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                    num_heads=num_heads,
                    n_kv_heads=n_kv_heads,
                    window_size=window_size,
                    use_gated_delta=use_gated_delta,
                    use_cuda_scan=use_cuda_scan,
                    rope_base=rope_base,
                    max_seq_len=max_seq_len,
                    n_sink_tokens=n_sink_tokens,
                )
            else:
                block = HybridMambaBlock(
                    d_model=d_model,
                    d_inner=2 * d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                    use_cuda_scan=use_cuda_scan,
                    use_pre_norm=use_pre_norm,
                )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self._has_ffn: list[bool] = [
            mlp_every_n > 0 and (i + 1) % mlp_every_n == 0
            for i in range(n_layers)
        ]
        self.ffns = nn.ModuleList(
            FeedForward(d_model, expansion=ffn_expansion, dropout=ffn_dropout)
            if has else nn.Identity()
            for has in self._has_ffn
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk, ffn, has_ffn in zip(self.blocks, self.ffns, self._has_ffn):
            x = x + blk(x)
            if has_ffn:
                x = x + ffn(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def make_states(self, batch: int, device=None, dtype=None) -> list[dict]:
        """Return per-block recurrent states for token-by-token inference."""
        return [blk.make_state(batch, device=device, dtype=dtype) for blk in self.blocks]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Autoregressive generation using recurrent :meth:`step` mode.

        Args:
            input_ids: ``(B, T_prompt)`` token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Softmax temperature. ``0.0`` → greedy argmax.
            top_k: If > 0, restrict sampling to the top-k logits.

        Returns:
            ``(B, T_prompt + max_new_tokens)`` token IDs.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.parameters()).dtype
        states = self.make_states(B, device=device, dtype=dtype)

        def _step(tok: torch.Tensor) -> torch.Tensor:
            x = self.embed(tok)
            for i, (blk, ffn, has_ffn) in enumerate(zip(self.blocks, self.ffns, self._has_ffn)):
                x = x + blk.step(x, states[i])
                if has_ffn:
                    x = x + ffn(x)
            return self.final_norm(x)

        for t in range(input_ids.shape[1]):
            last = _step(input_ids[:, t])

        generated: list[torch.Tensor] = []
        logits = self.lm_head(last)
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_tok = logits.argmax(-1)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                next_tok = torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(1)
            generated.append(next_tok)
            logits = self.lm_head(_step(next_tok))

        if not generated:
            return input_ids
        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)
