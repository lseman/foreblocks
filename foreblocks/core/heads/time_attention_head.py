from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class _RoPE1D(nn.Module):
    """Vectorized 1D rotary positional embedding over last dim (head_dim must be even)."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        head_dim = int(head_dim)
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, T, H] where H=head_dim
        """
        N, T, H = x.shape
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)  # [T, H/2]
        sin = freqs.sin().to(dtype=x.dtype, device=x.device)  # [T, H/2]
        cos = freqs.cos().to(dtype=x.dtype, device=x.device)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)  # interleave


class TimeAttention(nn.Module):
    """
    Per-feature Transformer block over time:
      - treats each feature f as its own sequence (length T)
      - attention is computed over time independently per feature stream
      - maps back to [B,T,F] via learned scalar projection + residual

    This is "channel-independent" in the sense that features do not attend to each other;
    each feature attends over its own history.

    NOTE: This head is shape-preserving but increases compute by F (batch becomes B*F).
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        causal: bool = False,
        window: Optional[int] = None,  # if set, mask attention to |i-j| <= window
        use_rope: bool = True,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        d_model = int(d_model)
        n_heads = int(n_heads)

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = bool(causal)
        self.window = int(window) if window is not None else None
        self.use_rope = bool(use_rope)

        # Lift each feature stream (scalar per timestep) -> d_model using grouped 1x1 conv:
        # Input:  [B, F, T]
        # Output: [B, F*d_model, T]  with groups=F means each feature has its own Linear(1->d_model)
        self.lift = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim * d_model,
            kernel_size=1,
            groups=feature_dim,
            bias=True,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=float(dropout), batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        hidden = int(ffn_mult) * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d_model),
            nn.Dropout(float(dropout)),
        )

        self.drop = nn.Dropout(float(dropout))

        # Project back to scalar per timestep for each feature stream
        self.to_scalar = nn.Linear(d_model, 1, bias=True)

        head_dim = d_model // n_heads
        self.rope = _RoPE1D(head_dim) if self.use_rope else None

    def _attn_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if (not self.causal) and (self.window is None):
            return None

        # float mask: 0 allowed, -inf disallowed
        mask = torch.zeros((T, T), device=device, dtype=dtype)

        if self.causal:
            mask = mask + torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), diagonal=1)

        if self.window is not None:
            idx = torch.arange(T, device=device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            win_mask = dist.abs() > self.window
            mask = torch.where(win_mask, torch.full_like(mask, float("-inf")), mask)

        return mask

    def _apply_rope_qk(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        # q,k: [N, T, d_model] -> [N, T, n_heads, head_dim]
        N, T, _ = q.shape
        head_dim = self.d_model // self.n_heads
        qh = q.view(N, T, self.n_heads, head_dim)
        kh = k.view(N, T, self.n_heads, head_dim)

        # apply RoPE per head (vectorized by merging N*n_heads)
        qh2 = qh.permute(0, 2, 1, 3).contiguous().view(N * self.n_heads, T, head_dim)
        kh2 = kh.permute(0, 2, 1, 3).contiguous().view(N * self.n_heads, T, head_dim)
        qh2 = self.rope(qh2)
        kh2 = self.rope(kh2)
        qh = qh2.view(N, self.n_heads, T, head_dim).permute(0, 2, 1, 3).contiguous()
        kh = kh2.view(N, self.n_heads, T, head_dim).permute(0, 2, 1, 3).contiguous()

        q = qh.view(N, T, self.d_model)
        k = kh.view(N, T, self.d_model)
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,F] -> y: [B,T,F]
        """
        if x.dim() != 3:
            raise ValueError(f"TimeAttention expects [B,T,F], got {tuple(x.shape)}")

        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != configured feature_dim={self.feature_dim}")

        # Build per-feature streams: [B,F,T] -> lift -> [B, F*d_model, T]
        xt = x.permute(0, 2, 1).contiguous()                 # [B,F,T]
        lifted = self.lift(xt)                               # [B, F*d_model, T]
        lifted = lifted.view(B, self.feature_dim, self.d_model, T).permute(0, 1, 3, 2).contiguous()
        # lifted: [B, F, T, d_model] -> merge B*F
        h = lifted.view(B * self.feature_dim, T, self.d_model)

        h1 = self.norm1(h)
        q = k = v = h1
        q, k = self._apply_rope_qk(q, k)

        attn_mask = self._attn_mask(T, device=x.device, dtype=torch.float32)  # keep float32 for stability
        y, _ = self.attn(q, k, v, attn_mask=attn_mask)

        h2 = h + self.drop(y)
        h3 = self.norm2(h2)
        h4 = h2 + self.ffn(h3)

        # back to scalar per timestep per feature
        s = self.to_scalar(h4).squeeze(-1)                   # [B*F, T]
        s = s.view(B, self.feature_dim, T).permute(0, 2, 1).contiguous()  # [B,T,F]

        return x + s  # residual


@node(
    type_id="timeattn_head",
    name="TimeAttentionHead",
    category="Preprocessing",
    outputs=["timeattn_head"],
    color="bg-gradient-to-r from-indigo-400 to-cyan-500",
)
class TimeAttentionHead(BaseHead):
    """BaseHead wrapper for TimeAttention. Forward -> [B,T,F]."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        causal: bool = False,
        window: Optional[int] = None,
        use_rope: bool = True,
    ):
        super().__init__(
            module=TimeAttention(
                feature_dim=feature_dim,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                ffn_mult=ffn_mult,
                causal=causal,
                window=window,
                use_rope=use_rope,
            ),
            name="timeattn",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Chronos-2 Embeddings Head
# ──────────────────────────────────────────────────────────────────────────────
