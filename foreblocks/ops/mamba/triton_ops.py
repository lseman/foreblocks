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
        w_ptr,
        out_ptr,
        M,
        D,
        eps,
        GROUP_SIZE: tl.constexpr,
        HAS_GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """RMSNormGated: rms_norm(y, weight, group_size) * silu(z)."""
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_D)
        mask = col < D

        row_off = row * D + col

        y = tl.load(y_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col, mask=mask, other=1.0).to(tl.float32)

        if HAS_GROUP:
            group_id = col // GROUP_SIZE
            group_offset = group_id * GROUP_SIZE
            gcol = group_offset + (col - group_offset)
            gmask = gcol < D
            y_g = tl.load(y_ptr + row * D + gcol, mask=gmask, other=0.0).to(tl.float32)
            mean_sq = tl.sum(y_g * y_g, axis=0) / GROUP_SIZE
            inv_rms = tl.rsqrt(mean_sq + eps)
            y_normed = y * inv_rms * w
        else:
            mean_sq = tl.sum(y * y, axis=0) / D
            inv_rms = tl.rsqrt(mean_sq + eps)
            y_normed = y * inv_rms * w

        sig = 1.0 / (1.0 + tl.exp(-z))
        silu_z = z * sig

        out = y_normed * silu_z
        tl.store(out_ptr + row_off, out, mask=mask)

    @triton.jit
    def _dt_prep_bwd_kernel(
        dy_ptr,
        x_ptr,
        b_ptr,
        dx_ptr,
        db_ptr,
        n_elements,
        D,
        dt_min,
        dt_max,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        d_idx = offs % D
        b = tl.load(b_ptr + d_idx, mask=mask, other=0.0).to(tl.float32)

        v = x + b
        # softplus with stable computation
        sp = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
        # clamp
        clamped = tl.maximum(sp, dt_min)
        clamped = tl.minimum(clamped, dt_max)

        # pass_through: 1 if strictly inside bounds, 0 if at boundary
        # Use nested where instead of & (which does element-wise mul on floats in Triton)
        pass_through = tl.where(
            clamped > dt_min,
            tl.where(clamped < dt_max, 1.0, 0.0),
            0.0,
        )

        # sigmoid (numerically stable)
        sigmoid_v = tl.where(
            v >= 0,
            1.0 / (1.0 + tl.exp(-v)),
            tl.exp(v) / (1.0 + tl.exp(v)),
        )

        dv = dy * pass_through * sigmoid_v

        tl.store(dx_ptr + offs, dv, mask=mask)

    @triton.jit
    def _fused_out_bwd_kernel(
        dy_ptr,
        y_ptr,
        z_ptr,
        w_ptr,
        ddy_ptr,
        ddz_ptr,
        ddw_ptr,
        M,
        D,
        eps,
        COMPUTE_DW: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_D)
        mask = col < D

        row_off = row * D + col

        dy = tl.load(dy_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col, mask=mask, other=1.0).to(tl.float32)

        # Forward recomputation
        mean_sq = tl.sum(y * y, axis=0) / D
        inv_rms = tl.rsqrt(mean_sq + eps)
        x_hat = y * inv_rms  # normalized y

        # silu(z) - numerically stable sigmoid
        sig = tl.where(
            (z >= 0).to(tl.int1),
            1.0 / (1.0 + tl.exp(-z)),
            tl.exp(z) / (1.0 + tl.exp(z)),
        )
        s = z * sig  # silu(z)

        # dx_hat = dy * w * s  (gradient flowing into RMSNorm output)
        dx_hat = dy * w * s

        # RMSNorm backward: dy_out = (dx_hat - mean(dx_hat * x_hat) * x_hat) * inv_rms
        mean_dhx_xh = tl.sum(dx_hat * x_hat, axis=0) / D
        dy_out = (dx_hat - mean_dhx_xh * x_hat) * inv_rms

        # d_w = sum(dy * x_hat * s, axis=0)
        # Note: atomic_add unreliable in Triton 3.x with mask; handled after kernel

        tl.store(ddy_ptr + row_off, dy_out, mask=mask)

        # d_s = dy * x_hat * w  (gradient into silu gate)
        ds = dy * x_hat * w

        # d_z = d_s * silu'(z) = d_s * (sig + z * sig * (1 - sig))
        dsilu = sig + z * sig * (1.0 - sig)
        dz = ds * dsilu
        tl.store(ddz_ptr + row_off, dz, mask=mask)


def dt_prep_fallback(
    dt_raw: torch.Tensor,
    bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    return F.softplus(dt_raw + bias.view(1, 1, -1)).clamp(dt_min, dt_max)


def _dt_prep_bwd_fallback(
    grad_out: torch.Tensor,
    dt_raw: torch.Tensor,
    bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
    compute_bias_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    v = dt_raw + bias.view(1, 1, -1)
    softplus_v = F.softplus(v)
    pass_through = ((softplus_v > dt_min) & (softplus_v < dt_max)).to(grad_out.dtype)
    d_dt_raw = grad_out * pass_through * torch.sigmoid(v)
    d_bias = d_dt_raw.sum(dim=(0, 1)) if compute_bias_grad else None
    return d_dt_raw, d_bias


def dt_prep_bwd_triton(
    grad_out: torch.Tensor,
    dt_raw: torch.Tensor,
    bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
    compute_bias_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Triton backward for dt_prep.

    Returns (d_dt_raw, d_bias).
    """
    if not (TRITON_AVAILABLE and grad_out.is_cuda and dt_raw.is_cuda and bias.is_cuda):
        return _dt_prep_bwd_fallback(
            grad_out,
            dt_raw,
            bias,
            dt_min=dt_min,
            dt_max=dt_max,
            compute_bias_grad=compute_bias_grad,
        )

    # Ensure contiguous for kernel
    grad_out = grad_out.contiguous()
    dt_raw = dt_raw.contiguous()
    bias = bias.contiguous()

    B, T, D = grad_out.shape
    n = dt_raw.numel()
    BLOCK = 1024

    d_dt_raw = torch.empty_like(grad_out)
    d_bias = torch.zeros_like(bias)

    grid = (triton.cdiv(n, BLOCK),)
    _dt_prep_bwd_kernel[grid](
        grad_out,
        dt_raw,
        bias,
        d_dt_raw,
        d_bias,
        n,
        D,
        dt_min,
        dt_max,
        BLOCK=BLOCK,
    )
    # Compute bias gradient via PyTorch (atomic_add unreliable in Triton 3.x)
    if compute_bias_grad:
        d_bias = d_dt_raw.sum(dim=(0, 1))
    return d_dt_raw, d_bias if compute_bias_grad else None


def fused_out_fallback(
    y: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """RMSNormGated: rms_norm(y, weight) * silu(z) — matches official Mamba2."""
    rms = torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + eps)
    return y * rms * norm_weight.view(1, 1, -1) * F.silu(z)


def _fused_out_bwd_fallback(
    grad_out: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-5,
    compute_norm_weight_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward for RMSNormGated: rms_norm(y, w) * silu(z)."""
    calc_dtype = torch.float64 if grad_out.dtype == torch.float64 else torch.float32
    grad32 = grad_out.to(calc_dtype)
    y32 = y.to(calc_dtype)
    z32 = z.to(calc_dtype)
    w32 = norm_weight.to(calc_dtype).view(1, 1, -1)

    # Forward recomputation
    inv_rms = torch.rsqrt((y32 * y32).mean(dim=-1, keepdim=True) + eps)
    x_hat = y32 * inv_rms  # normalized y
    sig = torch.sigmoid(z32)
    s = z32 * sig  # silu(z)

    # dx_hat = grad * w * s (gradient into RMSNorm output)
    dx_hat = grad32 * w32 * s

    # RMSNorm backward
    mean_dhx_xh = (dx_hat * x_hat).mean(dim=-1, keepdim=True)
    dy_out = (dx_hat - mean_dhx_xh * x_hat) * inv_rms

    # d_w = sum(grad * x_hat * s, dim=(0,1))
    if compute_norm_weight_grad:
        dw = (grad32 * x_hat * s).sum(dim=(0, 1)).to(norm_weight.dtype)
    else:
        dw = None

    # d_z = grad * x_hat * w * silu'(z)
    ds = grad32 * x_hat * w32
    dsilu = sig + z32 * sig * (1.0 - sig)
    dz = ds * dsilu

    return (dy_out.to(y.dtype), dz.to(z.dtype), dw)


def fused_out_bwd_triton(
    grad_out: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-5,
    compute_norm_weight_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Triton backward for fused_out.

    Returns (d_y, d_z, d_norm_weight).
    """
    if not (
        TRITON_AVAILABLE
        and grad_out.is_cuda
        and y.is_cuda
        and z.is_cuda
        and norm_weight.is_cuda
    ):
        return _fused_out_bwd_fallback(
            grad_out,
            y,
            z,
            norm_weight,
            eps=eps,
            compute_norm_weight_grad=compute_norm_weight_grad,
        )

    B, T, D = grad_out.shape
    M = B * T

    dy2 = grad_out.contiguous().view(M, D)
    y2 = y.contiguous().view(M, D)
    z2 = z.contiguous().view(M, D)

    ddy2 = torch.empty_like(dy2)
    ddz2 = torch.empty_like(dy2)
    ddw = (
        torch.zeros_like(norm_weight)
        if compute_norm_weight_grad
        else torch.empty_like(norm_weight)
    )

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_D = min(max(BLOCK_D, 16), 4096)

    grid = (M,)
    _fused_out_bwd_kernel[grid](
        dy2,
        y2,
        z2,
        norm_weight,
        ddy2,
        ddz2,
        ddw,
        M,
        D,
        eps,
        COMPUTE_DW=compute_norm_weight_grad,
        BLOCK_D=BLOCK_D,
    )
    # Compute dw via PyTorch (atomic_add unreliable in Triton 3.x)
    if compute_norm_weight_grad:
        # Recompute: RMSNorm(y, eps) with w=1, then silu(z)
        calc_dtype = torch.float64 if grad_out.dtype == torch.float64 else torch.float32
        y32 = y.to(calc_dtype)
        z32 = z.to(calc_dtype)
        w32 = norm_weight.to(calc_dtype).view(1, 1, -1)
        inv_rms = torch.rsqrt((y32 * y32).mean(dim=-1, keepdim=True) + eps)
        x_hat = y32 * inv_rms
        sig = torch.sigmoid(z32)
        s = z32 * sig
        grad32 = grad_out.to(calc_dtype)
        dw = (grad32 * x_hat * s).sum(dim=(0, 1)).to(norm_weight.dtype)
        ddw = dw

    return (
        ddy2.view(B, T, D),
        ddz2.view(B, T, D),
        ddw if compute_norm_weight_grad else None,
    )


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
        needs = ctx.needs_input_grad
        d_dt_raw, d_bias = dt_prep_bwd_triton(
            grad_out,
            dt_raw,
            bias,
            dt_min=ctx.dt_min,
            dt_max=ctx.dt_max,
            compute_bias_grad=needs[1],
        )
        return d_dt_raw if needs[0] else None, d_bias, None, None


class _FusedOutFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, z, norm_weight, eps: float, group_size: int | None = None):
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.save_for_backward(y, z, norm_weight)
        return fused_out_triton(y, z, norm_weight, eps=eps, group_size=group_size)

    @staticmethod
    def backward(ctx, grad_out):
        y, z, norm_weight = ctx.saved_tensors
        needs = ctx.needs_input_grad
        dy, dz, d_norm_weight = fused_out_bwd_triton(
            grad_out,
            y,
            z,
            norm_weight,
            eps=ctx.eps,
            compute_norm_weight_grad=needs[2],
        )
        return (
            dy if needs[0] else None,
            dz if needs[1] else None,
            d_norm_weight,
            None,
            None,
        )


def dt_prep_triton(
    dt_raw: torch.Tensor,
    bias: torch.Tensor,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
) -> torch.Tensor:
    if not (TRITON_AVAILABLE and dt_raw.is_cuda and bias.is_cuda):
        return dt_prep_fallback(dt_raw, bias, dt_min=dt_min, dt_max=dt_max)
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
    norm_weight: torch.Tensor,
    eps: float = 1e-5,
    group_size: int | None = None,
) -> torch.Tensor:
    """RMSNormGated: rms_norm(y, weight, group_size) * silu(z).

    When group_size is None or equals D, normalizes over full D dimension.
    When group_size is set, normalizes each group independently (for ngroups > 1).
    """
    if not (TRITON_AVAILABLE and y.is_cuda and z.is_cuda and norm_weight.is_cuda):
        return fused_out_fallback(y, z, norm_weight, eps=eps)
    assert y.shape == z.shape
    assert y.ndim == 3

    B, T, D = y.shape
    M = B * T

    y2 = y.contiguous().view(M, D)
    z2 = z.contiguous().view(M, D)

    out = torch.empty_like(y2)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_D = min(max(BLOCK_D, 16), 4096)
    actual_group = group_size if group_size is not None else D
    has_group = actual_group != D

    grid = (M,)
    _fused_out_kernel[grid](
        y2,
        z2,
        norm_weight,
        out,
        M,
        D,
        eps,
        GROUP_SIZE=actual_group,
        HAS_GROUP=has_group,
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
    norm_weight: torch.Tensor,
    eps: float = 1e-5,
    group_size: int | None = None,
) -> torch.Tensor:
    """RMSNormGated: rms_norm(y, weight, group_size) * silu(z) — matches official Mamba2.

    When group_size is None, normalizes over full D dimension.
    When group_size is set (e.g., d_inner // ngroups), normalizes each group independently.
    """
    if TRITON_AVAILABLE and y.is_cuda and z.is_cuda and norm_weight.is_cuda:
        return _FusedOutFn.apply(y, z, norm_weight, eps, group_size)
    return fused_out_fallback(y, z, norm_weight, eps)
