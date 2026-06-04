import os

import torch

from .fla_backend import (
    fla_chunk_gated_delta_rule,
    fla_fused_recurrent_gated_delta_rule,
    is_fla_available,
)


def can_use_fla_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int = 0,
    *,
    recurrent: bool = True,
) -> bool:
    if os.environ.get("FOREBLOCKS_DISABLE_FLA_GATED_DELTA_RULE", "") == "1":
        return False
    if not recurrent and chunk_size != 64:
        return False
    if not is_fla_available("fla.ops.gated_delta_rule"):
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    if q.shape[:3] != v.shape[:3]:
        return False
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        return False
    if initial_state.shape != q.shape[:2] + (q.shape[-1], v.shape[-1]):
        return False
    return all(t.is_cuda for t in (q, k, v, g, beta, initial_state))


@torch.compiler.disable
def fla_gated_delta_rule_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    chunk_size: int = 0,
    *,
    recurrent: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run upstream FLA Gated Delta Rule with [B, H, T, *] layout."""
    if not can_use_fla_gated_delta_rule(
        q, k, v, g, beta, initial_state, chunk_size, recurrent=recurrent
    ):
        raise RuntimeError("FLA gated delta rule is not available")
    fn = fla_fused_recurrent_gated_delta_rule() if recurrent else fla_chunk_gated_delta_rule()
    out, final_state = fn(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        g=g.transpose(1, 2).contiguous(),
        beta=beta.transpose(1, 2).contiguous(),
        scale=float(scale),
        initial_state=initial_state.contiguous().to(torch.float32),
        output_final_state=True,
    )
    return out.transpose(1, 2).contiguous(), final_state.contiguous()


__all__ = ["can_use_fla_gated_delta_rule", "fla_gated_delta_rule_forward"]
