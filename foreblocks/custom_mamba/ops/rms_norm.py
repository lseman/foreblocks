from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    RMS_NORM_TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    RMS_NORM_TRITON_AVAILABLE = False


if RMS_NORM_TRITON_AVAILABLE:

    @triton.jit
    def _rms_norm_fwd_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        M,
        D,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        offs = row * D + cols

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / D
        inv_rms = tl.rsqrt(mean_sq + eps)
        tl.store(out_ptr + offs, x * inv_rms * w, mask=mask)

    @triton.jit
    def _rms_norm_bwd_kernel(
        dy_ptr,
        x_ptr,
        w_ptr,
        dx_ptr,
        dw_ptr,
        M,
        D,
        eps,
        COMPUTE_DW: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        offs = row * D + cols

        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)

        mean_sq = tl.sum(x * x, axis=0) / D
        inv_rms = tl.rsqrt(mean_sq + eps)
        q = dy * w
        sum_qx = tl.sum(q * x, axis=0)
        dx = q * inv_rms - x * inv_rms * inv_rms * inv_rms * sum_qx / D

        if COMPUTE_DW:
            dw = dy * x * inv_rms
            tl.atomic_add(dw_ptr + cols, dw, mask=mask)

        tl.store(dx_ptr + offs, dx, mask=mask)


def rms_norm_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * weight


def _rms_norm_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not (RMS_NORM_TRITON_AVAILABLE and x.is_cuda and weight.is_cuda):
        return rms_norm_fallback(x, weight, eps)

    D = x.shape[-1]
    M = x.numel() // D
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    out = torch.empty_like(x_contig)
    block_d = triton.next_power_of_2(D)
    _rms_norm_fwd_kernel[(M,)](
        x_contig,
        weight_contig,
        out,
        M,
        D,
        eps,
        BLOCK_D=block_d,
    )
    return out.view_as(x)


def _rms_norm_backward_triton(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compute_weight_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not (
        RMS_NORM_TRITON_AVAILABLE
        and grad_out.is_cuda
        and x.is_cuda
        and weight.is_cuda
    ):
        x_ref = x.detach().requires_grad_(True)
        w_ref = weight.detach().requires_grad_(compute_weight_grad)
        with torch.enable_grad():
            y = rms_norm_fallback(x_ref, w_ref, eps)
        grads = torch.autograd.grad(
            y,
            (x_ref, w_ref) if compute_weight_grad else (x_ref,),
            grad_out,
            allow_unused=True,
        )
        return grads[0], grads[1] if compute_weight_grad else None

    D = x.shape[-1]
    M = x.numel() // D
    grad_contig = grad_out.contiguous()
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    dx = torch.empty_like(x_contig)
    dw = torch.zeros_like(weight_contig) if compute_weight_grad else torch.empty_like(weight_contig)
    block_d = triton.next_power_of_2(D)
    _rms_norm_bwd_kernel[(M,)](
        grad_contig,
        x_contig,
        weight_contig,
        dx,
        dw,
        M,
        D,
        eps,
        COMPUTE_DW=compute_weight_grad,
        BLOCK_D=block_d,
    )
    return dx.view_as(x), dw if compute_weight_grad else None


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps: float):
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        return _rms_norm_forward_triton(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        dx, dw = _rms_norm_backward_triton(
            grad_out,
            x,
            weight,
            ctx.eps,
            ctx.needs_input_grad[1],
        )
        return dx, dw, None


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if RMS_NORM_TRITON_AVAILABLE and x.is_cuda and weight.is_cuda:
        return _RMSNormFn.apply(x, weight, eps)
    return rms_norm_fallback(x, weight, eps)
