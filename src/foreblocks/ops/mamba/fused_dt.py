"""foreblocks.ops.mamba.fused_dt.

Fused dt_proj + softplus + clamp kernel.

Fuses the optional low-rank linear projection (dt_hidden → dt) with softplus
discretisation and range clamping into a single Triton kernel. When
`dt_proj_weight is None`, treats dt_hidden as already projected [B, T, H] —
only bias + softplus + clamp. Use when building Mamba2/Mamba3 models that
need the dt projection step with maximum throughput.

Core API:
- fused_dt: main entry, auto-selects Triton or fallback
- fused_dt_triton: Triton forward (falls back to PyTorch if no projection)
- fused_dt_fallback: PyTorch reference

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    FUSED_DT_TRITON_AVAILABLE = True
except Exception:
    triton = tl = None
    FUSED_DT_TRITON_AVAILABLE = False


if FUSED_DT_TRITON_AVAILABLE:

    @triton.jit
    def _fused_dt_kernel(
        dt_hidden_ptr,
        dt_proj_weight_ptr,
        dt_bias_ptr,
        out_ptr,
        B,
        T,
        dt_rank,
        num_heads,
        dt_min,
        dt_max,
        BLOCK_DT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        stride = T * num_heads
        b = pid // stride
        rem = pid % stride
        t = rem // num_heads
        h = rem % num_heads

        b_mask = b < B
        t_mask = t < T
        h_mask = h < num_heads
        if not (b_mask and t_mask and h_mask):
            return

        bt_idx = b * T + t

        j = tl.arange(0, BLOCK_DT)
        mask_j = j < dt_rank
        dt_h = tl.load(
            dt_hidden_ptr + (bt_idx * dt_rank + j), mask=mask_j, other=0.0
        ).to(tl.float32)
        w = tl.load(dt_proj_weight_ptr + (h * dt_rank + j), mask=mask_j, other=0.0).to(
            tl.float32
        )

        v = tl.sum(dt_h * w)
        bias = tl.load(dt_bias_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
        v += bias

        sp = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
        v = tl.maximum(sp, dt_min)
        v = tl.minimum(v, dt_max)

        tl.store(out_ptr + pid, v, mask=(b_mask & t_mask & h_mask))


def fused_dt_fallback(
    dt_hidden: torch.Tensor,
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    if dt_proj_weight is None:
        dt_raw = dt_hidden  # already [B, T, H]
    else:
        dt_raw = F.linear(dt_hidden, dt_proj_weight)
    return F.softplus(dt_raw + dt_bias.view(1, 1, -1)).clamp(dt_min, dt_max)


def fused_dt_bwd_fallback(
    grad_out: torch.Tensor,
    dt_hidden: torch.Tensor,
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
    compute_w_grad: bool = True,
    compute_bias_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if dt_proj_weight is None:
        dt_raw = dt_hidden  # already [B, T, H]
    else:
        dt_raw = F.linear(dt_hidden, dt_proj_weight)
    v = dt_raw + dt_bias.view(1, 1, -1)
    sp = F.softplus(v)
    pass_through = ((sp > dt_min) & (sp < dt_max)).to(grad_out.dtype)
    d_dt_raw = grad_out * pass_through * torch.sigmoid(v)

    if dt_proj_weight is None:
        d_dt_hidden = d_dt_raw
        d_dt_proj_weight = None
    else:
        d_dt_hidden = F.linear(d_dt_raw, dt_proj_weight.T) if compute_w_grad else None
        d_dt_proj_weight = (
            torch.einsum("bth,bhj->hj", d_dt_raw, dt_hidden) if compute_w_grad else None
        )
    d_dt_bias = d_dt_raw.sum(dim=(0, 1)) if compute_bias_grad else None

    return d_dt_hidden, d_dt_proj_weight, d_dt_bias


def fused_dt_triton(
    dt_hidden: torch.Tensor,
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    if dt_proj_weight is None:
        return fused_dt_fallback(dt_hidden, None, dt_bias, dt_min, dt_max)
    if not (
        FUSED_DT_TRITON_AVAILABLE
        and dt_hidden.is_cuda
        and dt_proj_weight.is_cuda
        and dt_bias.is_cuda
    ):
        return fused_dt_fallback(dt_hidden, dt_proj_weight, dt_bias, dt_min, dt_max)

    B, T, dt_rank = dt_hidden.shape
    num_heads = dt_proj_weight.shape[0]
    out = torch.empty(B, T, num_heads, device=dt_hidden.device, dtype=dt_hidden.dtype)

    n_elements = B * T * num_heads
    BLOCK_DT = min(max(triton.next_power_of_2(dt_rank), 16), 256)
    num_warps = 4 if BLOCK_DT >= 64 else 2
    _fused_dt_kernel[(n_elements,)](
        dt_hidden,
        dt_proj_weight,
        dt_bias,
        out,
        B,
        T,
        dt_rank,
        num_heads,
        dt_min,
        dt_max,
        BLOCK_DT=BLOCK_DT,
        num_warps=num_warps,
    )
    return out


def fused_dt_bwd_triton(
    grad_out: torch.Tensor,
    dt_hidden: torch.Tensor,
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
    compute_w_grad: bool = True,
    compute_bias_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return fused_dt_bwd_fallback(
        grad_out,
        dt_hidden,
        dt_proj_weight,
        dt_bias,
        dt_min,
        dt_max,
        compute_w_grad,
        compute_bias_grad,
    )


class _FusedDtFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt_hidden, dt_proj_weight, dt_bias, dt_min, dt_max):
        ctx.dt_min = dt_min
        ctx.dt_max = dt_max
        ctx.save_for_backward(dt_hidden, dt_proj_weight, dt_bias)
        return fused_dt_triton(dt_hidden, dt_proj_weight, dt_bias, dt_min, dt_max)

    @staticmethod
    def backward(ctx, grad_out):
        dt_hidden, dt_proj_weight, dt_bias = ctx.saved_tensors
        needs = ctx.needs_input_grad
        d_dt_hidden, d_dt_proj_weight, d_dt_bias = fused_dt_bwd_triton(
            grad_out,
            dt_hidden,
            dt_proj_weight,
            dt_bias,
            dt_min=ctx.dt_min,
            dt_max=ctx.dt_max,
            compute_w_grad=needs[1],
            compute_bias_grad=needs[2],
        )
        return (
            d_dt_hidden if needs[0] else None,
            d_dt_proj_weight if needs[1] else None,
            d_dt_bias if needs[2] else None,
            None,
            None,
        )


def fused_dt(
    dt_hidden: torch.Tensor,
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    if (
        dt_proj_weight is not None
        and FUSED_DT_TRITON_AVAILABLE
        and dt_hidden.is_cuda
        and dt_proj_weight.is_cuda
        and dt_bias.is_cuda
    ):
        return _FusedDtFn.apply(dt_hidden, dt_proj_weight, dt_bias, dt_min, dt_max)
    return fused_dt_fallback(dt_hidden, dt_proj_weight, dt_bias, dt_min, dt_max)
