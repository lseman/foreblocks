import math
import os

import torch

from .fla_backend import fla_fused_recurrent_delta_rule, is_fla_available

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _delta_rule_recurrent_fwd_kernel(
        Q,
        K,
        V,
        BETA,
        S0,
        O,
        ST,
        T: tl.constexpr,
        D: tl.constexpr,
        s_qb: tl.constexpr,
        s_qh: tl.constexpr,
        s_qt: tl.constexpr,
        s_qd: tl.constexpr,
        s_bb: tl.constexpr,
        s_bh: tl.constexpr,
        s_bt: tl.constexpr,
        s_sb: tl.constexpr,
        s_sh: tl.constexpr,
        s_sv: tl.constexpr,
        s_sk: tl.constexpr,
        scale: tl.constexpr,
        BD: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, BD)
        mask = offs < D

        q_base = Q + b * s_qb + h * s_qh
        k_base = K + b * s_qb + h * s_qh
        v_base = V + b * s_qb + h * s_qh
        beta_base = BETA + b * s_bb + h * s_bh
        s_base = S0 + b * s_sb + h * s_sh
        st_base = ST + b * s_sb + h * s_sh
        o_base = O + b * s_qb + h * s_qh

        state = tl.load(
            s_base + offs[:, None] * s_sv + offs[None, :] * s_sk,
            mask=mask[:, None] & mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for t in range(0, T):
            q = tl.load(q_base + t * s_qt + offs * s_qd, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(k_base + t * s_qt + offs * s_qd, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(v_base + t * s_qt + offs * s_qd, mask=mask, other=0.0).to(tl.float32)
            beta = tl.load(beta_base + t * s_bt).to(tl.float32)

            pred = tl.sum(state * k[None, :], axis=1)
            delta = beta * (v - pred)
            state += delta[:, None] * k[None, :]
            out = tl.sum(state * (q[None, :] * scale), axis=1)
            tl.store(o_base + t * s_qt + offs * s_qd, out, mask=mask)

        tl.store(
            st_base + offs[:, None] * s_sv + offs[None, :] * s_sk,
            state,
            mask=mask[:, None] & mask[None, :],
        )


def can_use_fused_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> bool:
    if not HAS_TRITON:
        return False
    if os.environ.get("FOREBLOCKS_DISABLE_TRITON_DELTA_RULE", "") == "1":
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda and beta.is_cuda and initial_state.is_cuda):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype or initial_state.dtype != q.dtype:
        return False
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        return False
    if beta.shape != q.shape[:3] + (1,):
        return False
    if initial_state.shape != q.shape[:2] + (q.shape[-1], q.shape[-1]):
        return False
    return q.shape[-1] <= 128


def can_use_fla_recurrent_delta_rule(
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


@torch.compiler.disable
def fla_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run upstream FLA fused recurrent delta rule with Foreblocks layouts.

    Foreblocks uses ``[B, H, T, D]`` and state ``[B, H, V, K]``.  FLA uses
    ``[B, T, H, D]`` and state ``[B, H, K, V]``.
    """
    if not can_use_fla_recurrent_delta_rule(q, k, v, beta, initial_state):
        raise RuntimeError("FLA fused recurrent delta rule is not available")
    fn = fla_fused_recurrent_delta_rule()
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


@torch.compiler.disable
def fused_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused recurrent DeltaNet forward for [B, H, T, D] tensors."""
    if not can_use_fused_recurrent_delta_rule(q, k, v, beta, initial_state):
        raise RuntimeError("fused_recurrent_delta_rule is not available")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    initial_state = initial_state.contiguous()
    B, H, T, D = q.shape
    scale = D ** -0.5 if scale is None else float(scale)
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    block_d = triton.next_power_of_2(D)
    _delta_rule_recurrent_fwd_kernel[(B, H)](
        q,
        k,
        v,
        beta,
        initial_state,
        out,
        final_state,
        T,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        initial_state.stride(0),
        initial_state.stride(1),
        initial_state.stride(2),
        initial_state.stride(3),
        scale,
        BD=block_d,
        num_warps=4,
    )
    return out, final_state


__all__ = [
    "can_use_fla_recurrent_delta_rule",
    "can_use_fused_recurrent_delta_rule",
    "fla_recurrent_delta_rule",
    "fused_recurrent_delta_rule",
]
