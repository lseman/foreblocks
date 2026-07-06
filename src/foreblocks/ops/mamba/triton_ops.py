"""foreblocks.ops.mamba.triton_ops.

Triton kernels for dt preparation and RMSNormGated output — Mamba2 primitives.

Provides Triton-accelerated dt_prep (softplus + clamp for discretisation) and
fused_out (RMSNormGated: RMSNorm(y, weight, group_size) * silu(z)) with full
autograd support. Both include PyTorch fallbacks. Use when building Mamba2
blocks that need these primitives with maximum throughput.

Core API:
- dt_prep: softplus + clamp time-step preparation (Triton or fallback)
- dt_prep_triton: Triton forward only
- fused_out: RMSNormGated — rms_norm(y, weight, group_size) * silu(z)
- fused_out_triton: Triton forward only

"""

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
        BLOCK_G: tl.constexpr,
    ):
        """RMSNormGated: rms_norm(y, weight, group_size) * silu(z).

        Grid (M, D // GROUP_SIZE): one program per (row, group), so the
        RMS statistic is computed over exactly one group. GROUP_SIZE == D
        recovers full-row RMSNorm.
        """
        row = tl.program_id(0)
        g = tl.program_id(1)
        lane = tl.arange(0, BLOCK_G)
        col = g * GROUP_SIZE + lane
        mask = (lane < GROUP_SIZE) & (col < D)

        row_off = row * D + col

        y = tl.load(y_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col, mask=mask, other=1.0).to(tl.float32)

        mean_sq = tl.sum(y * y, axis=0) / GROUP_SIZE
        inv_rms = tl.rsqrt(mean_sq + eps)
        y_normed = y * inv_rms * w

        sig = 1.0 / (1.0 + tl.exp(-z))
        out = y_normed * z * sig
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
        dw_partial_ptr,  # [n_row_blocks, D] fp32
        M,
        D,
        eps,
        GROUP_SIZE: tl.constexpr,
        BLOCK_G: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        COMPUTE_DW: tl.constexpr,
    ):
        """Backward for RMSNormGated with group support.

        Grid (n_row_blocks, D // GROUP_SIZE). Each program walks
        ROWS_PER_PROGRAM rows for one group and accumulates its dw slice
        into a per-row-block partial buffer (summed on the host) — no
        atomics, no separate forward recomputation pass.
        """
        row_block = tl.program_id(0)
        g = tl.program_id(1)
        lane = tl.arange(0, BLOCK_G)
        col = g * GROUP_SIZE + lane
        cmask = (lane < GROUP_SIZE) & (col < D)

        w = tl.load(w_ptr + col, mask=cmask, other=1.0).to(tl.float32)
        if COMPUTE_DW:
            dw = tl.zeros((BLOCK_G,), dtype=tl.float32)

        row_start = row_block * ROWS_PER_PROGRAM
        for r in range(ROWS_PER_PROGRAM):
            row = row_start + r
            rmask = row < M
            mask = rmask & cmask
            row_off = row * D + col

            dy = tl.load(dy_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + row_off, mask=mask, other=0.0).to(tl.float32)

            mean_sq = tl.sum(y * y, axis=0) / GROUP_SIZE
            inv_rms = tl.rsqrt(mean_sq + eps)
            x_hat = y * inv_rms

            sig = tl.sigmoid(z)
            s = z * sig  # silu(z)

            # gradient into RMSNorm output
            dx_hat = dy * w * s
            mean_dhx_xh = tl.sum(dx_hat * x_hat, axis=0) / GROUP_SIZE
            dy_out = (dx_hat - mean_dhx_xh * x_hat) * inv_rms
            tl.store(ddy_ptr + row_off, dy_out, mask=mask)

            # gradient into silu gate
            ds = dy * x_hat * w
            dz = ds * (sig + z * sig * (1.0 - sig))
            tl.store(ddz_ptr + row_off, dz, mask=mask)

            if COMPUTE_DW:
                dw += tl.where(mask, dy * x_hat * s, 0.0)

        if COMPUTE_DW:
            tl.store(dw_partial_ptr + row_block * D + col, dw, mask=cmask)


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
    group_size: int | None = None,
) -> torch.Tensor:
    """RMSNormGated: rms_norm(y, weight, group_size) * silu(z) — matches official Mamba2."""
    if group_size is not None and group_size != y.shape[-1]:
        *lead, D = y.shape
        yg = y.view(*lead, D // group_size, group_size)
        rms = torch.rsqrt((yg * yg).mean(dim=-1, keepdim=True) + eps)
        y_normed = (yg * rms).view(*lead, D)
    else:
        rms = torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + eps)
        y_normed = y * rms
    return y_normed * norm_weight.view(1, 1, -1) * F.silu(z)


def _fused_out_bwd_fallback(
    grad_out: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-5,
    compute_norm_weight_grad: bool = True,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward for RMSNormGated: rms_norm(y, w, group_size) * silu(z)."""
    calc_dtype = torch.float64 if grad_out.dtype == torch.float64 else torch.float32
    grad32 = grad_out.to(calc_dtype)
    y32 = y.to(calc_dtype)
    z32 = z.to(calc_dtype)
    w32 = norm_weight.to(calc_dtype).view(1, 1, -1)

    # Forward recomputation (per-group RMS statistic when group_size set)
    D = y32.shape[-1]
    gs = group_size if (group_size is not None and group_size != D) else D
    yg = y32.view(*y32.shape[:-1], D // gs, gs)
    inv_rms_g = torch.rsqrt((yg * yg).mean(dim=-1, keepdim=True) + eps)
    inv_rms = inv_rms_g.expand_as(yg).reshape(y32.shape)
    x_hat = y32 * inv_rms  # normalized y
    sig = torch.sigmoid(z32)
    s = z32 * sig  # silu(z)

    # dx_hat = grad * w * s (gradient into RMSNorm output)
    dx_hat = grad32 * w32 * s

    # RMSNorm backward (per-group mean when grouped)
    dhx = (dx_hat * x_hat).view(*y32.shape[:-1], D // gs, gs)
    mean_dhx_xh = dhx.mean(dim=-1, keepdim=True).expand_as(dhx).reshape(y32.shape)
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
    group_size: int | None = None,
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
            group_size=group_size,
        )

    B, T, D = grad_out.shape
    M = B * T
    gs = group_size if (group_size is not None and group_size != D) else D
    if D % gs != 0:
        raise ValueError(f"group_size {gs} must divide D {D}")
    n_groups = D // gs

    dy2 = grad_out.contiguous().view(M, D)
    y2 = y.contiguous().view(M, D)
    z2 = z.contiguous().view(M, D)

    ddy2 = torch.empty_like(dy2)
    ddz2 = torch.empty_like(dy2)

    sm_count = torch.cuda.get_device_properties(y.device).multi_processor_count
    n_row_blocks = max(1, min(sm_count, M))
    rows_per_program = triton.cdiv(M, n_row_blocks)
    dw_partial = torch.empty(
        (n_row_blocks, D) if compute_norm_weight_grad else (1, 1),
        dtype=torch.float32,
        device=y.device,
    )

    BLOCK_G = min(max(triton.next_power_of_2(gs), 16), 4096)

    _fused_out_bwd_kernel[(n_row_blocks, n_groups)](
        dy2,
        y2,
        z2,
        norm_weight,
        ddy2,
        ddz2,
        dw_partial,
        M,
        D,
        eps,
        GROUP_SIZE=gs,
        BLOCK_G=BLOCK_G,
        ROWS_PER_PROGRAM=rows_per_program,
        COMPUTE_DW=compute_norm_weight_grad,
    )

    ddw = (
        dw_partial.sum(dim=0).to(norm_weight.dtype)
        if compute_norm_weight_grad
        else None
    )
    return (ddy2.view(B, T, D), ddz2.view(B, T, D), ddw)


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
            group_size=ctx.group_size,
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
        return fused_out_fallback(y, z, norm_weight, eps=eps, group_size=group_size)
    assert y.shape == z.shape
    assert y.ndim == 3

    B, T, D = y.shape
    M = B * T
    gs = group_size if (group_size is not None and group_size != D) else D
    if D % gs != 0:
        raise ValueError(f"group_size {gs} must divide D {D}")
    n_groups = D // gs

    y2 = y.contiguous().view(M, D)
    z2 = z.contiguous().view(M, D)

    out = torch.empty_like(y2)
    BLOCK_G = min(max(triton.next_power_of_2(gs), 16), 4096)

    _fused_out_kernel[(M, n_groups)](
        y2,
        z2,
        norm_weight,
        out,
        M,
        D,
        eps,
        GROUP_SIZE=gs,
        BLOCK_G=BLOCK_G,
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
    return fused_out_fallback(y, z, norm_weight, eps, group_size=group_size)
