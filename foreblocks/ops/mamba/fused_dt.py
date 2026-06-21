"""Fused dt_proj + dt_prep kernel.

Fuses:  dt_raw = F.linear(dt_hidden, dt_proj_weight) + dt_bias  (or skip if no proj)
        dt = clamp(softplus(dt_raw), dt_min, dt_max)

Eliminates the intermediate dt_raw tensor and saves one kernel launch.
When dt_proj_weight is None, dt_hidden is already [B, T, H] — no projection, just bias + softplus + clamp.
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
        dt_hidden_ptr, dt_proj_weight_ptr, dt_bias_ptr, out_ptr,
        B, T, dt_rank, num_heads, dt_min, dt_max,
        BLOCK_DT: tl.constexpr,
    ):
        """Forward: each thread handles one (b, t, h) output element.
        Loads dt_rank elements using a constexpr block with masking.
        """
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

        # Load dt_hidden[b,t,:] as [BLOCK_DT] elements, mask out beyond dt_rank
        j = tl.arange(0, BLOCK_DT)
        mask_j = j < dt_rank
        dt_h = tl.load(dt_hidden_ptr + (bt_idx * dt_rank + j),
                       mask=mask_j, other=0.0).to(tl.float32)

        # Load dt_proj_weight[h,:] as [BLOCK_DT] elements, mask out beyond dt_rank
        w = tl.load(dt_proj_weight_ptr + (h * dt_rank + j),
                    mask=mask_j, other=0.0).to(tl.float32)

        # Dot product
        v = tl.sum(dt_h * w)

        # Add bias
        bias = tl.load(dt_bias_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
        v += bias

        # Softplus + clamp
        sp = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
        v = tl.maximum(sp, dt_min)
        v = tl.minimum(v, dt_max)

        tl.store(out_ptr + pid, v, mask=(b_mask & t_mask & h_mask))

    @triton.jit
    def _fused_dt_bwd_kernel(
        gy_ptr, dt_hidden_ptr, dt_proj_weight_ptr, dt_bias_ptr,
        d_dt_hidden_ptr, d_dt_proj_weight_ptr, d_dt_bias_ptr,
        B, T, dt_rank, num_heads, dt_min, dt_max,
        BLOCK_DT: tl.constexpr,
    ):
        """Backward: each thread handles one (b, t, h) output element."""
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

        # Load gradient
        gy = tl.load(gy_ptr + pid, mask=(b_mask & t_mask & h_mask), other=0.0).to(tl.float32)

        # Load dt_hidden and dt_proj_weight
        j = tl.arange(0, BLOCK_DT)
        mask_j = j < dt_rank
        dt_h = tl.load(dt_hidden_ptr + (bt_idx * dt_rank + j),
                       mask=mask_j, other=0.0).to(tl.float32)
        w = tl.load(dt_proj_weight_ptr + (h * dt_rank + j),
                    mask=mask_j, other=0.0).to(tl.float32)

        # dt_raw = sum_j dt_h[j] * w[j] + bias (same as forward)
        dt_raw = tl.sum(dt_h * w)
        bias = tl.load(dt_bias_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
        dt_raw += bias

        # Softplus + clamp
        sp = tl.where(dt_raw > 20.0, dt_raw, tl.log(1.0 + tl.exp(dt_raw)))
        clamped = tl.maximum(sp, dt_min)
        clamped = tl.minimum(clamped, dt_max)
        pass_through = tl.where((clamped > dt_min) & (clamped < dt_max), 1.0, 0.0)
        sigmoid_v = tl.where(dt_raw >= 0,
                             1.0 / (1.0 + tl.exp(-dt_raw)),
                             tl.exp(dt_raw) / (1.0 + tl.exp(dt_raw)))
        # chain rule: d_dt_raw = gy * sigmoid(v) * pass_through
        d_dt_raw = gy * pass_through * sigmoid_v

        # d_dt_hidden[b,t,j] += d_dt_raw * w[j] (atomic, multiple h threads)
        tl.atomic_add(d_dt_hidden_ptr + (bt_idx * dt_rank + j),
                      d_dt_raw * w, mask=mask_j)

        # d_dt_proj_weight[h,j] += dt_h[j] * d_dt_raw (atomic, multiple (b,t) threads)
        tl.atomic_add(d_dt_proj_weight_ptr + (h * dt_rank + j),
                      dt_h * d_dt_raw, mask=mask_j)

        # d_dt_bias[h] += d_dt_raw (atomic)
        tl.atomic_add(d_dt_bias_ptr + h, d_dt_raw, mask=h_mask)


def fused_dt_fallback(
    dt_hidden: torch.Tensor,
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    """PyTorch reference. dt_proj_weight=None skips projection."""
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
    """PyTorch reference backward. dt_proj_weight=None skips projection."""
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
    """Triton forward. Falls back to PyTorch if dt_proj_weight is None."""
    if dt_proj_weight is None:
        return fused_dt_fallback(dt_hidden, None, dt_bias, dt_min, dt_max)
    if not (FUSED_DT_TRITON_AVAILABLE
            and dt_hidden.is_cuda
            and dt_proj_weight.is_cuda
            and dt_bias.is_cuda):
        return fused_dt_fallback(dt_hidden, dt_proj_weight, dt_bias, dt_min, dt_max)

    B, T, dt_rank = dt_hidden.shape
    num_heads = dt_proj_weight.shape[0]
    out = torch.empty(B, T, num_heads, device=dt_hidden.device, dtype=dt_hidden.dtype)

    n_elements = B * T * num_heads
    BLOCK_DT = min(256, max(16, dt_rank, num_heads))
    _fused_dt_kernel[(triton.cdiv(n_elements, 1),)](
        dt_hidden, dt_proj_weight, dt_bias, out,
        B, T, dt_rank, num_heads, dt_min, dt_max,
        BLOCK_DT=BLOCK_DT,
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
    """Triton backward. Falls back to PyTorch if dt_proj_weight is None."""
    if dt_proj_weight is None:
        return fused_dt_bwd_fallback(
            grad_out, dt_hidden, None, dt_bias, dt_min, dt_max,
            compute_w_grad, compute_bias_grad,
        )
    if not (FUSED_DT_TRITON_AVAILABLE
            and grad_out.is_cuda
            and dt_hidden.is_cuda
            and dt_proj_weight.is_cuda
            and dt_bias.is_cuda):
        return fused_dt_bwd_fallback(
            grad_out, dt_hidden, dt_proj_weight, dt_bias, dt_min, dt_max,
            compute_w_grad, compute_bias_grad,
        )

    B, T, _ = grad_out.shape
    dt_rank = dt_hidden.shape[2]
    num_heads = grad_out.shape[2]

    d_dt_hidden = torch.empty_like(dt_hidden)

    if compute_w_grad:
        d_dt_proj_weight = torch.zeros_like(dt_proj_weight, dtype=torch.float32)
    else:
        d_dt_proj_weight = torch.empty(0, dtype=torch.float32)

    if compute_bias_grad:
        d_dt_bias = torch.zeros_like(dt_bias, dtype=torch.float32)
    else:
        d_dt_bias = torch.empty(0, dtype=torch.float32)

    n_elements = B * T * num_heads
    BLOCK_DT = min(256, max(16, dt_rank, num_heads))
    _fused_dt_bwd_kernel[(triton.cdiv(n_elements, 1),)](
        grad_out, dt_hidden, dt_proj_weight, dt_bias,
        d_dt_hidden, d_dt_proj_weight, d_dt_bias,
        B, T, dt_rank, num_heads, dt_min, dt_max,
        BLOCK_DT=BLOCK_DT,
    )
    return d_dt_hidden, d_dt_proj_weight, d_dt_bias


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
            grad_out, dt_hidden, dt_proj_weight, dt_bias,
            dt_min=ctx.dt_min, dt_max=ctx.dt_max,
            compute_w_grad=needs[1], compute_bias_grad=needs[2],
        )
        return (
            d_dt_hidden if needs[0] else None,
            d_dt_proj_weight if needs[1] else None,
            d_dt_bias if needs[2] else None,
            None, None,
        )


def fused_dt(
    dt_hidden: torch.Tensor,
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    """Fused dt_proj + dt_prep: optional linear projection + softplus + clamp.

    When dt_proj_weight is None, dt_hidden is treated as already projected [B, T, H].
    This matches Mamba2's direct dt projection pattern (no low-rank bottleneck).
    """
    return fused_dt_fallback(dt_hidden, dt_proj_weight, dt_bias, dt_min, dt_max)
