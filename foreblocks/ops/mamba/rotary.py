from __future__ import annotations

import torch

try:
    from foreblocks.ops.attention.fused_rope import (
        _TRITON_AVAILABLE as ROTARY_TRITON_AVAILABLE,
        triton_apply_rope_bthd,
    )
except Exception:
    ROTARY_TRITON_AVAILABLE = False
    triton_apply_rope_bthd = None  # type: ignore[assignment]


def rotary_apply_fallback(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    half = x.shape[-1] // 2
    if cos.shape[-1] == half:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


def _cos_sin_half_tables(
    cos: torch.Tensor,
    sin: torch.Tensor,
    T: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(dtype=cos.dtype)
    sin = sin.to(dtype=sin.dtype)
    if cos.ndim == 4:
        cos = cos[0, 0, :T]
        sin = sin[0, 0, :T]
    elif cos.ndim == 3:
        cos = cos[0, :T]
        sin = sin[0, :T]
    elif cos.ndim == 2:
        cos = cos[:T]
        sin = sin[:T]
    else:
        raise ValueError("cos and sin must have rank 2, 3, or 4")
    return cos.contiguous(), sin.contiguous()


def rotary_apply(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply GPT-NeoX-style RoPE over ``x`` shaped ``[B, H, T, D]``."""
    if (
        ROTARY_TRITON_AVAILABLE
        and triton_apply_rope_bthd is not None
        and x.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and x.ndim == 4
        and x.shape[-1] % 2 == 0
        and x.dtype == cos.dtype
        and x.dtype == sin.dtype
        and cos.shape[-1] == x.shape[-1] // 2
        and sin.shape[-1] == x.shape[-1] // 2
    ):
        B, H, T, D = x.shape
        cos_half, sin_half = _cos_sin_half_tables(cos, sin, T, D)
        out_bthd = triton_apply_rope_bthd(
            x.transpose(1, 2),
            cos_half,
            sin_half,
        )
        return out_bthd.transpose(1, 2)
    return rotary_apply_fallback(x, cos, sin)
