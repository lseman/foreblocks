import os

import torch

from .fla_backend import fla_chunk_gdn2, fla_fused_recurrent_gdn2, is_fla_available


def can_use_fla_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
    *,
    recurrent: bool = False,
) -> bool:
    if os.environ.get("FOREBLOCKS_DISABLE_FLA_GDN2", "") == "1":
        return False
    if not recurrent and chunk_size != 64:
        return False
    if not is_fla_available("fla.ops.gdn2"):
        return False
    if q.ndim != 4 or k.shape != q.shape or g.shape != q.shape:
        return False
    if b.shape != q.shape or v.ndim != 4 or w.shape != v.shape:
        return False
    if v.shape[:3] != q.shape[:3]:
        return False
    if initial_state.shape != q.shape[:2] + (q.shape[-1], v.shape[-1]):
        return False
    return all(t.is_cuda for t in (q, k, v, g, b, w, initial_state))


def can_use_fla_gdn2_chunk(*args, **kwargs) -> bool:
    return can_use_fla_gdn2(*args, **kwargs, recurrent=False)


@torch.compiler.disable
def fla_gdn2_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    chunk_size: int,
    *,
    recurrent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run upstream FLA GDN-2 kernels with [B, H, T, *] layout."""
    if not can_use_fla_gdn2(
        q, k, v, g, b, w, initial_state, chunk_size, recurrent=recurrent
    ):
        raise RuntimeError("FLA GDN-2 chunk kernel is not available")
    fn = fla_fused_recurrent_gdn2() if recurrent else fla_chunk_gdn2()
    kwargs = {}
    if not recurrent:
        kwargs["chunk_size"] = chunk_size
    out, final_state = fn(
        q=q.transpose(1, 2).contiguous(),
        k=k.transpose(1, 2).contiguous(),
        v=v.transpose(1, 2).contiguous(),
        g=g.transpose(1, 2).contiguous(),
        b=b.transpose(1, 2).contiguous(),
        w=w.transpose(1, 2).contiguous(),
        scale=float(scale),
        initial_state=initial_state.contiguous().to(torch.float32),
        output_final_state=True,
        **kwargs,
    )
    return out.transpose(1, 2).contiguous(), final_state.contiguous()


def fla_gdn2_chunk_forward(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    return fla_gdn2_forward(*args, **kwargs, recurrent=False)


__all__ = [
    "can_use_fla_gdn2",
    "can_use_fla_gdn2_chunk",
    "fla_gdn2_forward",
    "fla_gdn2_chunk_forward",
]
