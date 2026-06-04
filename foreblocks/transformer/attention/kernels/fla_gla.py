import os

import torch

from .fla_backend import fla_chunk_gla, fla_fused_recurrent_gla, is_fla_available


def can_use_fla_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> bool:
    if os.environ.get("FOREBLOCKS_DISABLE_FLA_GLA", "") == "1":
        return False
    if not is_fla_available("fla.ops.gla"):
        return False
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        return False
    if g.shape != q.shape:
        return False
    if initial_state.shape != q.shape[:2] + (q.shape[-1], v.shape[-1]):
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and initial_state.is_cuda):
        return False
    return True


@torch.compiler.disable
def fla_gla_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    mode: str = "chunk",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run upstream FLA GLA with Foreblocks [B, H, T, D] layout."""
    if not can_use_fla_gla(q, k, v, g, initial_state):
        raise RuntimeError("FLA GLA is not available for these tensors")
    q_fla = q.transpose(1, 2).contiguous()
    k_fla = k.transpose(1, 2).contiguous()
    v_fla = v.transpose(1, 2).contiguous()
    g_fla = g.transpose(1, 2).contiguous()
    state = initial_state.contiguous()
    if mode == "recurrent":
        fn = fla_fused_recurrent_gla()
    else:
        fn = fla_chunk_gla()
    out, final_state = fn(
        q_fla,
        k_fla,
        v_fla,
        g_fla,
        scale=float(scale),
        initial_state=state,
        output_final_state=True,
    )
    return out.transpose(1, 2).contiguous(), final_state.contiguous()


__all__ = ["can_use_fla_gla", "fla_gla_forward"]
