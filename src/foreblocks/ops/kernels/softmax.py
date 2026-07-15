"""foreblocks.ops.kernels.softmax.

Triton softmax kernel with full autograd.

Provides forward and backward for row-wise softmax. Use when you need
softmax with maximum throughput on CUDA tensors, particularly for
attention weights or probability distributions.

Core API:
- SoftmaxTritonFunction: autograd softmax with full backward
- triton_softmax: apply row-wise softmax

"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON_SOFTMAX = True
except Exception:
    HAS_TRITON_SOFTMAX = False
    triton = None
    tl = None


def _should_use_triton_softmax(
    x: torch.Tensor, min_numel: int = 4096, dim: int = -1
) -> bool:
    if x.ndim == 0:
        return False
    if not _supports_fused_row_width(x.shape[dim], x.element_size()):
        return False
    return (
        HAS_TRITON_SOFTMAX
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() >= min_numel
        and not torch.jit.is_scripting()
    )


def _supports_fused_row_width(n_cols: int, element_size: int) -> bool:
    if not HAS_TRITON_SOFTMAX:
        return False
    return n_cols <= 65536 // element_size


def _block_size_for_num_cols(n_cols: int, element_size: int) -> int:
    max_fused_size = 65536 // element_size
    block_size = triton.next_power_of_2(n_cols)
    if block_size > max_fused_size:
        raise RuntimeError(
            f"Triton softmax only supports feature dims up to {max_fused_size} "
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

if HAS_TRITON_SOFTMAX:

    @triton.jit
    def softmax_fwd_kernel(
        X,
        Y,
        stride_x_row,
        stride_y_row,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        x_ptrs = X + row_idx * stride_x_row + col
        x_row = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

        # Compute max for numerical stability
        row_max = tl.max(x_row, axis=0)
        row_max = tl.broadcast_to(row_max, [BLOCK_SIZE])

        # Compute exp(x - max)
        x_shifted = x_row - row_max
        exp_x = tl.exp(x_shifted)

        # Compute sum
        sum_exp = tl.sum(exp_x, axis=0)

        # Normalize
        y_row = exp_x / sum_exp

        y_ptrs = Y + row_idx * stride_y_row + col
        tl.store(y_ptrs, y_row, mask=mask)

    @triton.jit
    def softmax_bwd_kernel(
        DY,
        Y,
        DX,
        stride_dy_row,
        stride_y_row,
        stride_dx_row,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        row_start = row_block * ROWS_PER_PROGRAM
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        for row_offset in range(0, ROWS_PER_PROGRAM):
            row_idx = row_start + row_offset
            row_mask = row_idx < n_rows

            dy = tl.load(
                DY + row_idx * stride_dy_row + col,
                mask=row_mask & mask,
                other=0.0,
            ).to(tl.float32)
            y = tl.load(
                Y + row_idx * stride_y_row + col,
                mask=row_mask & mask,
                other=0.0,
            ).to(tl.float32)

            # dy * y - sum(dy * y) * y
            dy_y = dy * y
            sum_dy_y = tl.sum(dy_y, axis=0)
            sum_dy_y = tl.broadcast_to(sum_dy_y, [BLOCK_SIZE])

            dx = dy_y - sum_dy_y * y

            tl.store(
                DX + row_idx * stride_dx_row + col,
                dx,
                mask=row_mask & mask,
            )

    @triton.jit
    def softmax_bwd_kernel_fused(
        DY,
        Y,
        DX,
        stride_dy_row,
        stride_y_row,
        stride_dx_row,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        row_start = row_block * ROWS_PER_PROGRAM
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        for row_offset in range(0, ROWS_PER_PROGRAM):
            row_idx = row_start + row_offset
            row_mask = row_idx < n_rows

            dy = tl.load(
                DY + row_idx * stride_dy_row + col,
                mask=row_mask & mask,
                other=0.0,
            ).to(tl.float32)
            y = tl.load(
                Y + row_idx * stride_y_row + col,
                mask=row_mask & mask,
                other=0.0,
            ).to(tl.float32)

            dy_y = dy * y
            sum_dy_y = tl.sum(dy_y, axis=0)
            sum_dy_y = tl.broadcast_to(sum_dy_y, [BLOCK_SIZE])

            dx = dy_y - sum_dy_y * y

            tl.store(
                DX + row_idx * stride_dx_row + col,
                dx,
                mask=row_mask & mask,
            )


# ============================== Autograd functions ===========================


class SoftmaxTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        if not HAS_TRITON_SOFTMAX:
            raise RuntimeError("SoftmaxTritonFunction requires Triton.")

        if dim < -x.ndim or dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{x.ndim}, {x.ndim - 1}], but got {dim})"
            )
        dim = dim % x.ndim
        orig_shape = x.shape
        moved_shape = x.movedim(dim, -1).shape
        x_flat = x.movedim(dim, -1).reshape(-1, x.shape[dim]).contiguous()
        M, N = x_flat.shape

        y_flat = torch.empty_like(x_flat)

        BLOCK_SIZE = _block_size_for_num_cols(N, x_flat.element_size())

        softmax_fwd_kernel[(M,)](
            x_flat,
            y_flat,
            x_flat.stride(0),
            y_flat.stride(0),
            N,
            BLOCK_SIZE,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        ctx.save_for_backward(y_flat)
        ctx.orig_shape = orig_shape
        ctx.moved_shape = moved_shape
        ctx.dim = dim
        ctx.N = N
        return y_flat.reshape(moved_shape).movedim(-1, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if not HAS_TRITON_SOFTMAX:
            raise RuntimeError("SoftmaxTritonFunction backward requires Triton.")

        y_flat = ctx.saved_tensors[0]
        orig_shape = ctx.orig_shape
        moved_shape = ctx.moved_shape
        dim = ctx.dim
        N = ctx.N

        grad_output_flat = (
            grad_output.movedim(dim, -1).reshape(-1, N).contiguous()
        )
        M = grad_output_flat.shape[0]

        grad_input_flat = torch.empty_like(y_flat)

        BLOCK_SIZE = _block_size_for_num_cols(N, y_flat.element_size())
        n_row_blocks = max(1, min(torch.cuda.get_device_properties(y_flat.device).multi_processor_count, M))
        rows_per_program = triton.cdiv(M, n_row_blocks)

        softmax_bwd_kernel[(n_row_blocks,)](
            grad_output_flat,
            y_flat,
            grad_input_flat,
            grad_output_flat.stride(0),
            y_flat.stride(0),
            grad_input_flat.stride(0),
            M,
            N,
            BLOCK_SIZE,
            rows_per_program,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        grad_input = grad_input_flat.reshape(moved_shape).movedim(-1, dim)
        return grad_input.reshape(orig_shape), None


# ============================== Python API ====================================


def softmax_fallback(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax using native PyTorch operations."""
    return torch.softmax(x, dim=dim)


def triton_softmax(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Apply row-wise softmax using Triton when supported, otherwise use PyTorch."""
    if _should_use_triton_softmax(x, dim=dim):
        return SoftmaxTritonFunction.apply(x, dim)
    return softmax_fallback(x, dim)


__all__ = [
    "HAS_TRITON_SOFTMAX",
    "_should_use_triton_softmax",
    "_supports_fused_row_width",
    "SoftmaxTritonFunction",
    "triton_softmax",
    "softmax_fallback",
]
