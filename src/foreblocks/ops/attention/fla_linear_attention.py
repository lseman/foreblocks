"""foreblocks.ops.attention.fla_linear_attention.

Wrap upstream FLA fused recurrent linear attention with Foreblocks layout.

FLA provides a fused recurrent implementation of linear attention (kernel-based O(T)
attention). This module bridges FLA's internal layout to Foreblocks' `[B, H, T, D]`
convention, exposing a runtime availability check and a single entry point. Use when
you need FLA-backed linear attention inside a Foreblocks model without manual tensor
transposes.

Core API:
- can_use_fla_linear_attn: runtime capability check (FLA installed, CUDA, shape/dtype valid)
- fla_recurrent_linear_attn_forward: fused recurrent linear attention with Foreblocks layout

"""

import os

import torch

from foreblocks.ops.attention.fla_backend import (
    fla_fused_recurrent_linear_attn,
    is_fla_available,
)


def can_use_fla_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    if os.environ.get("FOREBLOCKS_DISABLE_FLA_LINEAR_ATTN", "") == "1":
        return False
    if not is_fla_available("fla.ops.linear_attn"):
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    if q.shape[:3] != v.shape[:3]:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    return True


@torch.compiler.disable
def fla_recurrent_linear_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not can_use_fla_linear_attn(q, k, v):
        raise RuntimeError("FLA fused recurrent linear attention is not available")
    fn = fla_fused_recurrent_linear_attn()
    out, _ = fn(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        scale=1.0,
        normalize=True,
    )
    return out.transpose(1, 2).contiguous()


__all__ = ["can_use_fla_linear_attn", "fla_recurrent_linear_attn_forward"]
