from __future__ import annotations

import torch
import torch.nn as nn

from ..ops import rotary_apply


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
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        self._cos_cached = freqs.cos()[None, None]  # (1, 1, T, head_dim // 2)
        self._sin_cached = freqs.sin()[None, None]

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
        q_rot = rotary_apply(q, cos, sin)
        k_rot = rotary_apply(k, cos, sin)
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
        cos_full = torch.cat([cos, cos], dim=-1)
        sin_full = torch.cat([sin, sin], dim=-1)
        q_rot = q * cos_full + self._rotate_half(q) * sin_full
        k_rot = k * cos_full + self._rotate_half(k) * sin_full
        return q_rot, k_rot
