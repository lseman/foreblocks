from __future__ import annotations

import torch
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _dt_prep_kernel(
        x_ptr,
        bias_ptr,
        y_ptr,
        n_elements,
        D,
        dt_min,
        dt_max,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        d_idx = offs % D
        b = tl.load(bias_ptr + d_idx, mask=mask, other=0.0).to(tl.float32)

        v = x + b
        y = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
        y = tl.maximum(y, dt_min)
        y = tl.minimum(y, dt_max)

        tl.store(y_ptr + offs, y, mask=mask)

    @triton.jit
    def _fused_out_kernel(
        y_ptr,
        z_ptr,
        res_ptr,
        w_ptr,
        out_ptr,
        M,
        D,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_D)
        mask = col < D

        row_off = row * D + col

        y = tl.load(y_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col, mask=mask, other=1.0).to(tl.float32)

        g = z / (1.0 + tl.exp(-z))
        x = y * g + r

        mean_sq = tl.sum(x * x, axis=0) / D
        inv_rms = tl.rsqrt(mean_sq + eps)

        out = x * inv_rms * w
        tl.store(out_ptr + row_off, out, mask=mask)


def dt_prep_fallback(
    dt_raw: torch.Tensor,
    bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    return F.softplus(dt_raw + bias.view(1, 1, -1)).clamp(dt_min, dt_max)


def fused_out_fallback(
    y: torch.Tensor,
    z: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    x = y * F.silu(z) + residual
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * norm_weight.view(1, 1, -1)


class _DtPrepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt_raw, bias, dt_min: float, dt_max: float):
        ctx.dt_min = dt_min
        ctx.dt_max = dt_max
        ctx.save_for_backward(dt_raw, bias)
        return dt_prep_triton(dt_raw, bias, dt_min=dt_min, dt_max=dt_max)

    @staticmethod
    def backward(ctx, grad_out):
        dt_raw, bias = ctx.saved_tensors

        with torch.enable_grad():
            dt_raw_ = dt_raw.detach().requires_grad_(dt_raw.requires_grad)
            bias_ = bias.detach().requires_grad_(bias.requires_grad)
            out = dt_prep_fallback(
                dt_raw_,
                bias_,
                dt_min=ctx.dt_min,
                dt_max=ctx.dt_max,
            )
            all_inputs = (dt_raw_, bias_)
            grad_inputs = tuple(t for t in all_inputs if t.requires_grad)
            grads_required = torch.autograd.grad(
                outputs=out,
                inputs=grad_inputs,
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grads_iter = iter(grads_required)
        grads = tuple(next(grads_iter) if t.requires_grad else None for t in all_inputs)
        d_dt_raw, d_bias = grads
        return d_dt_raw, d_bias, None, None


class _FusedOutFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, z, residual, norm_weight, eps: float):
        ctx.eps = eps
        ctx.save_for_backward(y, z, residual, norm_weight)
        return fused_out_triton(y, z, residual, norm_weight, eps=eps)

    @staticmethod
    def backward(ctx, grad_out):
        y, z, residual, norm_weight = ctx.saved_tensors

        with torch.enable_grad():
            y_ = y.detach().requires_grad_(y.requires_grad)
            z_ = z.detach().requires_grad_(z.requires_grad)
            residual_ = residual.detach().requires_grad_(residual.requires_grad)
            norm_weight_ = norm_weight.detach().requires_grad_(norm_weight.requires_grad)
            out = fused_out_fallback(
                y_,
                z_,
                residual_,
                norm_weight_,
                eps=ctx.eps,
            )
            all_inputs = (y_, z_, residual_, norm_weight_)
            grad_inputs = tuple(t for t in all_inputs if t.requires_grad)
            grads_required = torch.autograd.grad(
                outputs=out,
                inputs=grad_inputs,
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grads_iter = iter(grads_required)
        grads = tuple(next(grads_iter) if t.requires_grad else None for t in all_inputs)
        dy, dz, d_residual, d_norm_weight = grads
        return dy, dz, d_residual, d_norm_weight, None


def dt_prep_triton(
    dt_raw: torch.Tensor,
    bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("dt_prep_triton called but Triton is not available")
    assert dt_raw.is_cuda and bias.is_cuda
    assert dt_raw.ndim == 3
    B, T, D = dt_raw.shape
    del B, T
    out = torch.empty_like(dt_raw)
    n = dt_raw.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _dt_prep_kernel[grid](
        dt_raw,
        bias,
        out,
        n,
        D,
        dt_min,
        dt_max,
        BLOCK=BLOCK,
    )
    return out


def fused_out_triton(
    y: torch.Tensor,
    z: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("fused_out_triton called but Triton is not available")
    assert y.is_cuda and z.is_cuda and residual.is_cuda and norm_weight.is_cuda
    assert y.shape == z.shape == residual.shape
    assert y.ndim == 3

    B, T, D = y.shape
    M = B * T

    y2 = y.contiguous().view(M, D)
    z2 = z.contiguous().view(M, D)
    r2 = residual.contiguous().view(M, D)

    out = torch.empty_like(y2)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_D = min(max(BLOCK_D, 16), 4096)

    grid = (M,)
    _fused_out_kernel[grid](
        y2,
        z2,
        r2,
        norm_weight,
        out,
        M,
        D,
        eps,
        BLOCK_D=BLOCK_D,
    )
    return out.view(B, T, D)


def dt_prep(
    dt_raw: torch.Tensor,
    bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    if TRITON_AVAILABLE and dt_raw.is_cuda and bias.is_cuda:
        return _DtPrepFn.apply(dt_raw, bias, dt_min, dt_max)
    return dt_prep_fallback(dt_raw, bias, dt_min=dt_min, dt_max=dt_max)


def fused_out(
    y: torch.Tensor,
    z: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if TRITON_AVAILABLE and y.is_cuda and z.is_cuda and residual.is_cuda and norm_weight.is_cuda:
        return _FusedOutFn.apply(y, z, residual, norm_weight, eps)
    return fused_out_fallback(y, z, residual, norm_weight, eps=eps)
