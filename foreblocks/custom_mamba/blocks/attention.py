from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm
from .rotary import RotaryEmbedding


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
            Set to 1-4 to prevent information starvation at window boundaries.
        qk_norm: Apply per-head RMSNorm to Q and K before RoPE. This improves
            attention-logit stability in deep or long-context stacks.
        qk_norm_eps: Epsilon used by Q/K RMSNorm.
        logit_softcap: If set, apply ``softcap * tanh(logits / softcap)`` before
            the attention softmax.
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
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        logit_softcap: float | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if logit_softcap is not None and logit_softcap <= 0:
            raise ValueError("logit_softcap must be positive when set")

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
        self.qk_norm = qk_norm
        self.logit_softcap = logit_softcap

        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_norm = (
            RMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        )

        self.rope = RotaryEmbedding(
            self.head_dim, base=rope_base, max_seq_len=max_seq_len
        )

    def _sliding_mask(
        self, seqlen: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        rows = torch.arange(seqlen, device=device)
        cols = torch.arange(seqlen, device=device)
        rel = rows[:, None] - cols[None, :]
        allowed = (rel >= 0) & (rel < self.window_size)
        if self.n_sink_tokens > 0:
            sink = (cols[None, :] < self.n_sink_tokens) & (rel >= 0)
            allowed = allowed | sink
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
        mask.masked_fill_(allowed, 0.0)
        return mask

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.logit_softcap is None:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        return torch.matmul(probs, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn_mask = self._sliding_mask(T, x.device, q.dtype)
        y = self._attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(y)
