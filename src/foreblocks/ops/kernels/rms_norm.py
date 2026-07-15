"""foreblocks.ops.kernels.rms_norm.

Triton RMSNorm, fused add+RMSNorm, and scale/bias kernels with full autograd.

Provides forward and backward for RMS normalization, a fused kernel that combines
residual addition with RMSNorm (saving one intermediate tensor), and scale/bias
helpers. Use when building Mamba2-style blocks that need RMSNorm with fused
residual connections, or when you need scale+bias on top of normalized output.

Core API:
- RMSNormTritonFunction: autograd RMSNorm with optional weight
- FusedAddRMSNormFunction: fused (residual + update) + RMSNorm in one kernel
- fused_add_rmsnorm: convenience wrapper for FusedAddRMSNormFunction
- triton_scale_bias: element-wise y = x * alpha + beta
- triton_fused_rmsnorm_scale_bias: RMSNorm + scale + bias in one kernel

"""

# rms_norm.py
# -----------------------------------------------------------------------------
# Triton kernels and autograd functions for RMS Normalization, fused
# add+RMSNorm, and scale/bias helpers.
# -----------------------------------------------------------------------------

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def _should_use_triton(x: torch.Tensor, min_numel: int = 4096) -> bool:
    if not _supports_fused_row_width(x.shape[-1], x.element_size()):
        return False
    return (
        TRITON_AVAILABLE
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() >= min_numel
        and not torch.jit.is_scripting()
    )


def _supports_fused_row_width(n_cols: int, element_size: int) -> bool:
    if not TRITON_AVAILABLE:
        return False
    return n_cols <= 65536 // element_size


def _block_size_for_num_cols(n_cols: int, element_size: int) -> int:
    max_fused_size = 65536 // element_size
    block_size = triton.next_power_of_2(n_cols)
    if block_size > max_fused_size:
        raise RuntimeError(
            f"Triton RMSNorm only supports feature dims up to {max_fused_size} "
            f"for element size {element_size}; got {n_cols}."
        )
    return block_size


def _num_warps_for_block(block_size: int) -> int:
    if block_size <= 64:
        return 2
    if block_size <= 512:
        return 4
    if block_size <= 2048:
        return 8
    return 16


# =============================== Triton kernels ===============================

if TRITON_AVAILABLE:

    @triton.jit
    def rmsnorm_fwd_kernel(
        X,
        Y,
        W,
        Rms,
        stride_x_row,
        stride_y_row,
        n_cols,
        eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        x_ptrs = X + row_idx * stride_x_row + col
        x_row = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        ms = tl.sum(x_row * x_row, axis=0) / n_cols
        rms = tl.math.rsqrt(ms + eps)

        y = x_row * rms

        if HAS_WEIGHT:
            w = tl.load(W + col, mask=mask, other=1.0).to(tl.float32)
            y = y * w

        y_ptrs = Y + row_idx * stride_y_row + col
        tl.store(y_ptrs, y, mask=mask)
        tl.store(Rms + row_idx, rms)

    @triton.jit
    def fused_add_rmsnorm_fwd_kernel(
        Residual,
        Update,
        Y,
        XSum,
        W,
        Rms,
        stride_row,
        stride_y_row,
        n_cols,
        eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        r = tl.load(Residual + row_idx * stride_row + col, mask=mask, other=0.0).to(
            tl.float32
        )
        u = tl.load(Update + row_idx * stride_row + col, mask=mask, other=0.0).to(
            tl.float32
        )
        x = r + u
        # Store the residual sum here so the host never re-reads r and u
        # for a separate elementwise add.
        tl.store(XSum + row_idx * stride_row + col, x, mask=mask)

        ms = tl.sum(x * x, axis=0) / n_cols
        rms = tl.math.rsqrt(ms + eps)
        y = x * rms

        if HAS_WEIGHT:
            w = tl.load(W + col, mask=mask, other=1.0).to(tl.float32)
            y = y * w

        tl.store(Y + row_idx * stride_y_row + col, y, mask=mask)
        tl.store(Rms + row_idx, rms)

    @triton.jit
    def rmsnorm_bwd_kernel(
        DY,
        X,
        W,
        Rms,
        DX,
        DWPartial,
        stride_dy_row,
        stride_x_row,
        stride_dx_row,
        stride_dw_row,
        n_rows,
        n_cols,
        eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        COMPUTE_DW: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        row_start = row_block * ROWS_PER_PROGRAM
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        if HAS_WEIGHT:
            w = tl.load(W + col, mask=mask, other=1.0).to(tl.float32)
        if COMPUTE_DW:
            dw = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for row_offset in range(0, ROWS_PER_PROGRAM):
            row_idx = row_start + row_offset
            row_mask = row_idx < n_rows

            dy = tl.load(
                DY + row_idx * stride_dy_row + col,
                mask=row_mask & mask,
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                X + row_idx * stride_x_row + col,
                mask=row_mask & mask,
                other=0.0,
            ).to(tl.float32)
            rms = tl.load(Rms + row_idx, mask=row_mask, other=0.0).to(tl.float32)

            dy_scaled = dy * w if HAS_WEIGHT else dy
            sum_dy_x = tl.sum(dy_scaled * x, axis=0)
            dx = rms * (dy_scaled - x * sum_dy_x * rms * rms / n_cols)

            tl.store(
                DX + row_idx * stride_dx_row + col,
                dx,
                mask=row_mask & mask,
            )
            if COMPUTE_DW:
                dw += tl.where(row_mask & mask, dy * x * rms, 0.0)

        if COMPUTE_DW:
            tl.store(DWPartial + row_block * stride_dw_row + col, dw, mask=mask)

    @triton.jit
    def scale_bias_kernel(
        X,
        Y,
        Alpha,
        Beta,
        n_cols,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        base = row_idx * n_cols
        x = tl.load(X + base + col, mask=mask, other=0.0)
        alpha = tl.load(Alpha + col, mask=mask, other=1.0)

        y = x * alpha

        if HAS_BIAS:
            beta = tl.load(Beta + col, mask=mask, other=0.0)
            y = y + beta

        tl.store(Y + base + col, y, mask=mask)

    @triton.jit
    def fused_rmsnorm_scale_bias_kernel(
        X,
        Y,
        W,
        Alpha,
        Beta,
        stride_x_row,
        stride_y_row,
        n_cols,
        eps,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        x_ptrs = X + row_idx * stride_x_row + col
        x_row = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        ms = tl.sum(x_row * x_row, axis=0) / n_cols
        rms = tl.math.rsqrt(ms + eps)
        y = x_row * rms

        if HAS_WEIGHT:
            w = tl.load(W + col, mask=mask, other=1.0).to(tl.float32)
            y = y * w

        alpha = tl.load(Alpha + col, mask=mask, other=1.0).to(tl.float32)
        y = y * alpha

        if HAS_BIAS:
            beta = tl.load(Beta + col, mask=mask, other=0.0).to(tl.float32)
            y = y + beta

        y_ptrs = Y + row_idx * stride_y_row + col
        tl.store(y_ptrs, y, mask=mask)


# ============================== Autograd functions ===========================


def _new_grad_weight_partials(
    weight: torch.Tensor,
    n_row_blocks: int,
) -> torch.Tensor:
    return torch.empty(
        (n_row_blocks, weight.numel()),
        device=weight.device,
        dtype=torch.float32,
    )


def _num_backward_row_blocks(x: torch.Tensor, n_rows: int) -> int:
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    return max(1, min(sm_count, n_rows))


class RMSNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "RMSNormTritonFunction called but Triton is not available."
            )
        if weight is None:
            raise RuntimeError("RMSNormTritonFunction expects non-None weight.")

        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x.shape

        y = torch.empty_like(x)
        rms = torch.empty((M,), dtype=torch.float32, device=x.device)

        BLOCK_SIZE = _block_size_for_num_cols(N, x.element_size())

        rmsnorm_fwd_kernel[(M,)](
            x,
            y,
            weight,
            rms,
            x.stride(0),
            y.stride(0),
            N,
            eps,
            True,
            BLOCK_SIZE,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        ctx.save_for_backward(x, weight, rms)
        ctx.orig_shape = orig_shape
        ctx.eps = eps
        ctx.N = N
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "RMSNormTritonFunction backward called but Triton is not available."
            )

        x, weight, rms = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        eps = ctx.eps
        N = ctx.N

        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        M = grad_output.shape[0]

        grad_input = torch.empty_like(x)

        BLOCK_SIZE = _block_size_for_num_cols(N, x.element_size())
        n_row_blocks = _num_backward_row_blocks(x, M)
        rows_per_program = triton.cdiv(M, n_row_blocks)
        grad_weight_partials = _new_grad_weight_partials(weight, n_row_blocks)
        rmsnorm_bwd_kernel[(n_row_blocks,)](
            grad_output,
            x,
            weight,
            rms,
            grad_input,
            grad_weight_partials,
            grad_output.stride(0),
            x.stride(0),
            grad_input.stride(0),
            grad_weight_partials.stride(0),
            M,
            N,
            eps,
            True,
            BLOCK_SIZE,
            rows_per_program,
            True,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        grad_weight = grad_weight_partials.sum(dim=0).to(weight.dtype)
        return grad_input.reshape(orig_shape), grad_weight, None


class FusedAddRMSNormFunction(torch.autograd.Function):
    """
    Single-kernel fused (residual + update) + RMSNorm.

    Saves one [*, D] intermediate tensor vs. the two-kernel sequence.
    Only valid for CUDA dtypes in {float16, bfloat16, float32} and rows
    that fit in Triton's 64KB fused-row limit.
    """

    @staticmethod
    def forward(ctx, residual, update, weight, eps):
        if not TRITON_AVAILABLE:
            raise RuntimeError("FusedAddRMSNormFunction requires Triton.")
        if residual.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(
                f"FusedAddRMSNormFunction only supports float16/bfloat16/float32, got {residual.dtype}"
            )

        orig_shape = residual.shape
        residual_2d = residual.reshape(-1, orig_shape[-1]).contiguous()
        update_2d = update.reshape(-1, orig_shape[-1]).contiguous()
        M, N = residual_2d.shape

        y = torch.empty_like(residual_2d)
        out_2d = torch.empty_like(residual_2d)
        rms = torch.empty((M,), dtype=torch.float32, device=residual.device)

        BLOCK_SIZE = _block_size_for_num_cols(N, residual.element_size())

        fused_add_rmsnorm_fwd_kernel[(M,)](
            residual_2d,
            update_2d,
            y,
            out_2d,
            weight,
            rms,
            residual_2d.stride(0),
            y.stride(0),
            N,
            eps,
            weight is not None,
            BLOCK_SIZE,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        ctx.save_for_backward(out_2d, weight, rms)
        ctx.orig_shape = orig_shape
        ctx.N = N
        ctx.eps = eps
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        out_2d, weight, rms = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        N = ctx.N
        eps = ctx.eps

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        M = grad_output_2d.shape[0]

        grad_input = torch.empty_like(out_2d)

        BLOCK_SIZE = _block_size_for_num_cols(N, out_2d.element_size())
        compute_dw = weight is not None
        n_row_blocks = _num_backward_row_blocks(out_2d, M)
        rows_per_program = triton.cdiv(M, n_row_blocks)
        grad_weight_partials = (
            _new_grad_weight_partials(weight, n_row_blocks) if compute_dw else out_2d
        )

        rmsnorm_bwd_kernel[(n_row_blocks,)](
            grad_output_2d,
            out_2d,
            weight,
            rms,
            grad_input,
            grad_weight_partials,
            grad_output_2d.stride(0),
            out_2d.stride(0),
            grad_input.stride(0),
            grad_weight_partials.stride(0),
            M,
            N,
            eps,
            weight is not None,
            BLOCK_SIZE,
            rows_per_program,
            compute_dw,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        grad_weight = (
            grad_weight_partials.sum(dim=0).to(weight.dtype) if compute_dw else None
        )
        dx = grad_input.reshape(orig_shape)
        return dx, dx, grad_weight, None


# ============================== Python API ====================================


def rms_norm_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm using native PyTorch operations."""
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * weight


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm using Triton when supported, otherwise use PyTorch."""
    if _should_use_triton(x) and weight.is_cuda:
        return RMSNormTritonFunction.apply(x, weight, eps)
    return rms_norm_fallback(x, weight, eps)


def fused_add_rmsnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Single-kernel fused (residual + update) + RMSNorm for supported CUDA row widths."""
    return FusedAddRMSNormFunction.apply(residual, update, weight, eps)


def triton_scale_bias(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor | None
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton_scale_bias called but Triton is not available.")

    *lead, H = x.shape
    x_flat = x.reshape(-1, H).contiguous()
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = _block_size_for_num_cols(H, x.element_size())
    grid = (x_flat.shape[0],)

    scale_bias_kernel[grid](
        x_flat,
        y_flat,
        alpha,
        beta,
        H,
        beta is not None,
        BLOCK_SIZE,
        num_warps=_num_warps_for_block(BLOCK_SIZE),
    )

    return y_flat.reshape(*lead, H)


def triton_fused_rmsnorm_scale_bias(
    x: torch.Tensor,
    rms_weight: torch.Tensor | None,
    alpha: torch.Tensor,
    beta: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "triton_fused_rmsnorm_scale_bias called but Triton is not available."
        )

    *lead, H = x.shape
    x_flat = x.reshape(-1, H).contiguous()
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = _block_size_for_num_cols(H, x.element_size())
    grid = (x_flat.shape[0],)

    fused_rmsnorm_scale_bias_kernel[grid](
        x_flat,
        y_flat,
        rms_weight,
        alpha,
        beta,
        x_flat.stride(0),
        y_flat.stride(0),
        H,
        eps,
        rms_weight is not None,
        beta is not None,
        BLOCK_SIZE,
        num_warps=_num_warps_for_block(BLOCK_SIZE),
    )

    return y_flat.reshape(*lead, H)


__all__ = [
    "TRITON_AVAILABLE",
    "_should_use_triton",
    "_supports_fused_row_width",
    "RMSNormTritonFunction",
    "rms_norm",
    "rms_norm_fallback",
    "FusedAddRMSNormFunction",
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_fwd_kernel",
    "triton_scale_bias",
    "triton_fused_rmsnorm_scale_bias",
]
