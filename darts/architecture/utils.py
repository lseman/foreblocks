"""Attention modules: SelfAttention, AttentionBridge, LearnedPoolingBridge."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_positional import RotaryPositionalEncoding
from .bb_primitives import RMSNorm

__all__ = [
    "SelfAttention",
    "AttentionBridge",
    "LearnedPoolingBridge",
]


def _make_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Canonical ALiBi slopes from Press et al. (2021): 1 / 2^(8 i / H)."""
    num_heads = max(1, int(num_heads))

    # For non-power-of-two head counts, fall back to interpolated slopes
    # (same recipe as the reference ALiBi implementation).
    def _pow2_slopes(n: int) -> list[float]:
        start = 2.0 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = _pow2_slopes(num_heads)
    else:
        closest = 2 ** int(math.floor(math.log2(num_heads)))
        slopes = _pow2_slopes(closest)
        extra = _pow2_slopes(2 * closest)[0::2][: num_heads - closest]
        slopes = slopes + extra
    return torch.tensor(slopes, dtype=torch.float32)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular bool mask (True = block), cached per (device, len)."""
    key = (device, seq_len)
    cache = _causal_mask._cache  # type: ignore[attr-defined]
    tensor = cache.get(key)
    if tensor is None or tensor.size(0) != seq_len:
        tensor = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        cache[key] = tensor
    return tensor


_causal_mask._cache = {}  # type: ignore[attr-defined]


def _sinusoidal_features(
    seq_len: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> torch.Tensor:
    seq_len = int(max(1, seq_len))
    dim = int(max(1, dim))
    half = max(1, dim // 2)
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, half, device=device, dtype=dtype)
        * -(torch.log(torch.tensor(base, device=device, dtype=dtype)) / max(half, 1))
    )
    angles = position * div_term.unsqueeze(0)
    feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if feats.size(-1) < dim:
        feats = F.pad(feats, (0, dim - feats.size(-1)))
    return feats[:, :dim]


def _seasonal_relative_bias(
    query_len: int,
    key_len: int,
    num_heads: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    q_pos = torch.arange(query_len, device=device, dtype=dtype).unsqueeze(1)
    k_pos = torch.arange(key_len, device=device, dtype=dtype).unsqueeze(0)
    rel = q_pos - k_pos
    periods = torch.tensor([4.0, 8.0, 16.0, 24.0, 48.0], device=device, dtype=dtype)
    bias = torch.stack(
        [torch.cos(2.0 * torch.pi * rel / p) for p in periods], dim=0
    ).mean(dim=0)
    slopes = (
        _make_alibi_slopes(num_heads)
        .to(device=device, dtype=dtype)
        .reshape(1, num_heads, 1, 1)
    )
    return 0.1 * slopes * bias.reshape(1, 1, query_len, key_len)


