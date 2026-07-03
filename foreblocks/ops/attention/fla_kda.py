"""foreblocks.ops.attention.fla_kda.

This module implements the fla kda pieces for its package.
It belongs to the attention modules, variants, caches, and utilities area of Foreblocks.
It exposes functions such as can_use_fla_kda, fla_kda_forward.
"""

import os

import torch

from foreblocks.ops.attention.fla_backend import (
    fla_chunk_kda,
    fla_fused_recurrent_kda,
    is_fla_available,
)


def can_use_fla_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
    *,
    recurrent: bool = False,
) -> bool:
    if os.environ.get("FOREBLOCKS_DISABLE_FLA_KDA", "") == "1":
        return False
    if not is_fla_available("fla.ops.kda"):
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    if v.shape[:3] != q.shape[:3]:
        return False
    if g.shape != q.shape or beta.shape != q.shape[:3]:
        return False
    if initial_state.shape != q.shape[:2] + (q.shape[-1], v.shape[-1]):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if not recurrent and chunk_size not in (32, 64):
        return False
    if q.shape[-1] > 256:
        return False
    return all(t.is_cuda for t in (q, k, v, g, beta, initial_state))


@torch.compiler.disable
def fla_kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    chunk_size: int,
    *,
    recurrent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run upstream FLA KDA with [B, H, T, *] layout."""
    if not can_use_fla_kda(
        q, k, v, g, beta, initial_state, chunk_size, recurrent=recurrent
    ):
        raise RuntimeError("FLA KDA is not available")
    fn = fla_fused_recurrent_kda() if recurrent else fla_chunk_kda()
    kwargs = {}
    if not recurrent:
        kwargs["chunk_size"] = chunk_size
    out, final_state = fn(
        q=q.transpose(1, 2).contiguous(),
        k=k.transpose(1, 2).contiguous(),
        v=v.transpose(1, 2).contiguous(),
        g=g.transpose(1, 2).contiguous(),
        beta=beta.transpose(1, 2).contiguous(),
        scale=float(scale),
        initial_state=initial_state.contiguous().to(torch.float32),
        output_final_state=True,
        **kwargs,
    )
    return out.transpose(1, 2).contiguous(), final_state.contiguous()


__all__ = ["can_use_fla_kda", "fla_kda_forward"]
