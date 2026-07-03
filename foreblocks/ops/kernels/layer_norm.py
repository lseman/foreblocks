"""foreblocks.ops.kernels.layer_norm.

This module implements the layer norm pieces for its package.
It belongs to the low-level optimized operations and kernel wrappers area of Foreblocks.
It exposes classes such as LayerNormTritonFunction.
"""

# layer_norm.py
# -----------------------------------------------------------------------------
# Triton kernels and autograd function for Layer Normalization.
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


def _block_size_for_num_cols(n_cols: int, element_size: int) -> int:
    max_fused_size = 65536 // element_size
    block_size = triton.next_power_of_2(n_cols)
    if block_size > max_fused_size:
        raise RuntimeError(
            f"Triton LayerNorm only supports feature dims up to {max_fused_size} "
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
    def layernorm_fwd_kernel(
        X,
        Y,
        W,
        B,
        Mean,
        Rstd,
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

        mean = tl.sum(x_row, axis=0) / n_cols
        centered = x_row - mean
        var = tl.sum(centered * centered, axis=0) / n_cols
        rstd = tl.math.rsqrt(var + eps)

        y = centered * rstd

        if HAS_WEIGHT:
            w = tl.load(W + col, mask=mask, other=1.0).to(tl.float32)
            y = y * w
        if HAS_BIAS:
            b = tl.load(B + col, mask=mask, other=0.0).to(tl.float32)
            y = y + b

        y_ptrs = Y + row_idx * stride_y_row + col
        tl.store(y_ptrs, y, mask=mask)

        tl.store(Mean + row_idx, mean)
        tl.store(Rstd + row_idx, rstd)

    @triton.jit
    def layernorm_bwd_dx_kernel(
        DY,
        X,
        W,
        Mean,
        Rstd,
        DX,
        stride_dy_row,
        stride_x_row,
        stride_dx_row,
        n_cols,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        dy_ptrs = DY + row_idx * stride_dy_row + col
        x_ptrs = X + row_idx * stride_x_row + col

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        mean = tl.load(Mean + row_idx)
        rstd = tl.load(Rstd + row_idx)

        x_centered = x - mean
        x_normed = x_centered * rstd

        if HAS_WEIGHT:
            w = tl.load(W + col, mask=mask, other=1.0).to(tl.float32)
            dy_scaled = dy * w
        else:
            dy_scaled = dy

        c1 = tl.sum(dy_scaled, axis=0) / n_cols
        c2 = tl.sum(dy_scaled * x_normed, axis=0) / n_cols

        dx = rstd * (dy_scaled - c1 - x_normed * c2)

        dx_ptrs = DX + row_idx * stride_dx_row + col
        tl.store(dx_ptrs, dx, mask=mask)

    @triton.jit
    def layernorm_bwd_dwdb_partial_kernel(
        DY,
        X,
        Mean,
        Rstd,
        DWPartial,
        DBPartial,
        stride_dy_row,
        stride_x_row,
        n_rows,
        n_cols,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        block_m = tl.program_id(0)

        row_offsets = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = tl.arange(0, BLOCK_N)
        row_mask = row_offsets < n_rows
        col_mask = col_offsets < n_cols

        dy = tl.load(
            DY + row_offsets[:, None] * stride_dy_row + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            X + row_offsets[:, None] * stride_x_row + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        mean = tl.load(Mean + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
        x_normed = (x - mean[:, None]) * rstd[:, None]

        if HAS_WEIGHT:
            partial_dw = tl.sum(dy * x_normed, axis=0)
            tl.store(
                DWPartial + block_m * n_cols + col_offsets,
                partial_dw,
                mask=col_mask,
            )
        if HAS_BIAS:
            partial_db = tl.sum(dy, axis=0)
            tl.store(
                DBPartial + block_m * n_cols + col_offsets,
                partial_db,
                mask=col_mask,
            )

    @triton.jit
    def layernorm_bwd_col_reduce_kernel(
        Partial,
        Grad,
        n_blocks,
        n_cols,
        BLOCK_R: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < n_cols
        row_offsets = tl.arange(0, BLOCK_R)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        r0 = 0
        while r0 < n_blocks:
            rows = r0 + row_offsets
            row_mask = rows < n_blocks
            vals = tl.load(
                Partial + rows[:, None] * n_cols + col_offsets[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(vals, axis=0)
            r0 += BLOCK_R

        tl.store(Grad + col_offsets, acc, mask=col_mask)

    layernorm_bwd_dwdb_row_kernel = layernorm_bwd_dwdb_partial_kernel

    @triton.jit
    def layernorm_bwd_kernel(
        DY,
        X,
        W,
        Mean,
        Rstd,
        DX,
        DWPartial,
        DBPartial,
        stride_dy_row,
        stride_x_row,
        stride_dx_row,
        stride_dw_row,
        stride_db_row,
        n_rows,
        n_cols,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        row_start = row_block * ROWS_PER_PROGRAM
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        if HAS_WEIGHT:
            w = tl.load(W + col, mask=mask, other=1.0).to(tl.float32)
            dw = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        if HAS_BIAS:
            db = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

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
            mean = tl.load(Mean + row_idx, mask=row_mask, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd + row_idx, mask=row_mask, other=0.0).to(tl.float32)
            x_normed = tl.where(mask, (x - mean) * rstd, 0.0)

            dy_scaled = dy * w if HAS_WEIGHT else dy
            c1 = tl.sum(dy_scaled, axis=0) / n_cols
            c2 = tl.sum(dy_scaled * x_normed, axis=0) / n_cols
            dx = rstd * (dy_scaled - c1 - x_normed * c2)

            tl.store(
                DX + row_idx * stride_dx_row + col,
                dx,
                mask=row_mask & mask,
            )
            if HAS_WEIGHT:
                dw += tl.where(row_mask & mask, dy * x_normed, 0.0)
            if HAS_BIAS:
                db += tl.where(row_mask & mask, dy, 0.0)

        if HAS_WEIGHT:
            tl.store(DWPartial + row_block * stride_dw_row + col, dw, mask=mask)
        if HAS_BIAS:
            tl.store(DBPartial + row_block * stride_db_row + col, db, mask=mask)


# ============================== Autograd function ============================


def _layernorm_grad_weight_bias_block_reduce(
    weight: torch.Tensor,
    bias: torch.Tensor,
    n_row_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    partial_dw = torch.empty(
        (n_row_blocks, weight.numel()),
        device=weight.device,
        dtype=torch.float32,
    )
    partial_db = torch.empty(
        (n_row_blocks, bias.numel()),
        device=bias.device,
        dtype=torch.float32,
    )
    return partial_dw, partial_db


def _num_backward_row_blocks(x: torch.Tensor, n_rows: int) -> int:
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    return max(1, min(sm_count, n_rows))


class LayerNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "LayerNormTritonFunction called but Triton is not available."
            )
        if weight is None or bias is None:
            raise RuntimeError(
                "LayerNormTritonFunction expects non-None weight and bias."
            )

        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x.shape

        y = torch.empty_like(x)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        BLOCK_SIZE = _block_size_for_num_cols(N, x.element_size())

        layernorm_fwd_kernel[(M,)](
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            x.stride(0),
            y.stride(0),
            N,
            eps,
            True,
            True,
            BLOCK_SIZE,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.orig_shape = orig_shape
        ctx.N = N
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "LayerNormTritonFunction backward called but Triton is not available."
            )

        x, weight, bias, mean, rstd = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        N = ctx.N

        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        M = grad_output.shape[0]

        grad_input = torch.empty_like(x)
        BLOCK_SIZE = _block_size_for_num_cols(N, x.element_size())
        n_row_blocks = _num_backward_row_blocks(x, M)
        rows_per_program = triton.cdiv(M, n_row_blocks)
        grad_weight_partials, grad_bias_partials = _layernorm_grad_weight_bias_block_reduce(
            weight,
            bias,
            n_row_blocks,
        )
        layernorm_bwd_kernel[(n_row_blocks,)](
            grad_output,
            x,
            weight,
            mean,
            rstd,
            grad_input,
            grad_weight_partials,
            grad_bias_partials,
            grad_output.stride(0),
            x.stride(0),
            grad_input.stride(0),
            grad_weight_partials.stride(0),
            grad_bias_partials.stride(0),
            M,
            N,
            True,
            True,
            BLOCK_SIZE,
            rows_per_program,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        grad_weight = grad_weight_partials.sum(dim=0).to(weight.dtype)
        grad_bias = grad_bias_partials.sum(dim=0).to(bias.dtype)
        return grad_input.reshape(orig_shape), grad_weight, grad_bias, None


__all__ = [
    "TRITON_AVAILABLE",
    "LayerNormTritonFunction",
    "_block_size_for_num_cols",
    "layernorm_fwd_kernel",
    "layernorm_bwd_dx_kernel",
    "layernorm_bwd_kernel",
    "layernorm_bwd_dwdb_partial_kernel",
    "layernorm_bwd_dwdb_row_kernel",
    "layernorm_bwd_col_reduce_kernel",
]
