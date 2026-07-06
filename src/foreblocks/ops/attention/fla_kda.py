"""foreblocks.ops.attention.fla_kda.

Wrap upstream FLA kernel-discretised attention with Foreblocks tensor layout.

Kernel Discretised Attention (KDA) extends gated-linear attention with a kernel-based
recurrent state update. This module bridges FLA's internal layout to Foreblocks'
`[B, H, T, D]` convention, exposing runtime availability checks and a single entry
point. Use when you need FLA-backed KDA inside a Foreblocks model without manual
tensor transposes.

Core API:
- can_use_fla_kda: runtime capability check (FLA installed, CUDA, shape/dtype valid)
- fla_kda_forward: chunk or recurrent KDA with Foreblocks layout

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
    """Run upstream FLA KDA with [B, H, T, *] layout.

    Args:
        q, k, v: query, key, value tensors [B, H, T, D].
        g: gate tensor [B, H, T, D].
        beta: kernel discretisation parameter [B, H, T].
        initial_state: per-head state [B, H, D, D].
        scale, chunk_size: attention scale and chunk size (chunk mode only).
        recurrent: if True, use recurrent mode instead of chunked.

    Returns:
        (output, final_state): output [B, H, T, D], state [B, H, D, D].
    """
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
