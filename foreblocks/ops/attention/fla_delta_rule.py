import os

import torch

from foreblocks.ops.attention.fla_backend import (
    fla_chunk_delta_rule,
    fla_fused_recurrent_delta_rule,
    is_fla_available,
)


def can_use_fla_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> bool:
    if os.environ.get("FOREBLOCKS_DISABLE_FLA_DELTA_RULE", "") == "1":
        return False
    if not is_fla_available("fla.ops.delta_rule"):
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda and beta.is_cuda and initial_state.is_cuda):
        return False
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        return False
    if beta.shape != q.shape[:3] + (1,):
        return False
    if initial_state.shape != q.shape[:2] + (q.shape[-1], q.shape[-1]):
        return False
    return True


def can_use_fla_recurrent_delta_rule(*args, **kwargs) -> bool:
    return can_use_fla_delta_rule(*args, **kwargs)


@torch.compiler.disable
def fla_delta_rule_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float | None = None,
    *,
    recurrent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run upstream FLA delta rule kernels with Foreblocks layouts.

    Foreblocks uses ``[B, H, T, D]`` and state ``[B, H, V, K]``.  FLA uses
    ``[B, T, H, D]`` and state ``[B, H, K, V]``.
    """
    if not can_use_fla_delta_rule(q, k, v, beta, initial_state):
        raise RuntimeError("FLA delta rule is not available")
    fn = fla_fused_recurrent_delta_rule() if recurrent else fla_chunk_delta_rule()
    q_fla = q.transpose(1, 2).contiguous()
    k_fla = k.transpose(1, 2).contiguous()
    v_fla = v.transpose(1, 2).contiguous()
    beta_fla = beta.squeeze(-1).transpose(1, 2).contiguous()
    state_fla = initial_state.transpose(-1, -2).contiguous()
    out, final_state = fn(
        q_fla,
        k_fla,
        v_fla,
        beta=beta_fla,
        scale=(q.shape[-1] ** -0.5 if scale is None else float(scale)),
        initial_state=state_fla,
        output_final_state=True,
    )
    return out.transpose(1, 2).contiguous(), final_state.transpose(-1, -2).contiguous()


def fla_recurrent_delta_rule(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    return fla_delta_rule_forward(*args, **kwargs, recurrent=True)


__all__ = [
    "can_use_fla_delta_rule",
    "can_use_fla_recurrent_delta_rule",
    "fla_delta_rule_forward",
    "fla_recurrent_delta_rule",
]
