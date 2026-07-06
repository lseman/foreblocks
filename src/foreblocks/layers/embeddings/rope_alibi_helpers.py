"""foreblocks.layers.embeddings.rope_alibi_helpers.

Positional encoding utility functions for attention modules.

Provides helpers to apply RoPE to Q/K tensors and ALiBi bias to attention
scores. Used by linear attention, gated delta net, and Kimi attention
implementations.

Core API:
- apply_rope_qkv: apply rotary embeddings to query/key tensors
- apply_alibi_bias: apply ALiBi positional bias to attention scores

"""

from __future__ import annotations

import torch
import torch.nn as nn


def apply_rope_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    rotary_emb: nn.Module,
    seqlen_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K tensors.

    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D]
        rotary_emb: RotaryEmbedding module
        seqlen_offset: Offset for KV cache decoding

    Returns:
        (q_rotated, k_rotated) same shapes as input
    """
    B, H, L, D = q.shape

    # RotaryEmbedding expects [B, L, H, D] format
    q_4d = q.transpose(1, 2)  # [B, L, H, D]
    k_4d = k.transpose(1, 2)  # [B, L, H, D]

    # Apply RoPE using the torch-native path (works regardless of triton)
    cos, sin = rotary_emb._cos_cached, rotary_emb._sin_cached
    cos = cos[:L].unsqueeze(0).expand(B, -1, -1)  # [B, L, D/2]
    sin = sin[:L].unsqueeze(0).expand(B, -1, -1)  # [B, L, D/2]

    # Rotate halves
    ro_dim = cos.shape[-1] * 2
    q_rot = _rotate_half(q_4d[..., :ro_dim])
    q_out = q_4d * cos + q_rot * sin
    # Keep rest unchanged
    if ro_dim < D:
        q_out = torch.cat([q_out, q_4d[..., ro_dim:]], dim=-1)

    k_rot = _rotate_half(k_4d[..., :ro_dim])
    k_out = k_4d * cos + k_rot * sin
    if ro_dim < D:
        k_out = torch.cat([k_out, k_4d[..., ro_dim:]], dim=-1)

    # Transpose back to [B, H, L, D]
    return q_out.transpose(1, 2), k_out.transpose(1, 2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate every pair of dimensions by 90 degrees."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_alibi_bias(
    attn_scores: torch.Tensor,
    alibi_bias: nn.Module,
) -> torch.Tensor:
    """
    Add ALiBi positional bias to attention scores.

    Args:
        attn_scores: Attention scores [B, H, Lq, Lk]
        alibi_bias: ALiBiPositionalBias module

    Returns:
        attn_scores with bias added [B, H, Lq, Lk]
    """
    B, H, Lq, Lk = attn_scores.shape
    bias = alibi_bias(Lq, Lk, device=attn_scores.device)
    return attn_scores + bias


def create_rotary_embedding(
    head_dim: int,
    max_seq_len: int = 4096,
    base: float = 10000.0,
) -> nn.Module:
    """
    Create a RotaryEmbedding module.

    Args:
        head_dim: Per-head dimension
        max_seq_len: Maximum sequence length
        base: Base frequency for RoPE

    Returns:
        RotaryEmbedding module
    """
    # Import here to avoid circular imports
    from foreblocks.layers.embeddings.rotary import RotaryEmbedding

    return RotaryEmbedding(dim=head_dim, base=base)


def create_alibi_bias(
    num_heads: int,
    max_seq_len: int = 4096,
) -> nn.Module:
    """
    Create an ALiBiPositionalBias module.

    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length

    Returns:
        ALiBiPositionalBias module
    """
    from foreblocks.layers.embeddings.alibi_bias import ALiBiPositionalBias

    return ALiBiPositionalBias(num_heads=num_heads, max_seq_len=max_seq_len)
