"""foreblocks.ops.kernels.gelu.

Triton GELU kernel with full autograd.

Provides forward and backward for GELU activation. Use when you need
GELU with maximum throughput on CUDA tensors, particularly for
transformer feed-forward layers.

Core API:
- GeluTritonFunction: autograd GELU with full backward
- triton_gelu: apply GELU activation

"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON_GELU = True
except Exception:
    HAS_TRITON_GELU = False
    triton = None
    tl = None


def _should_use_triton_gelu(x: torch.Tensor, min_numel: int = 4096) -> bool:
    if not _supports_fused_row_width(x.shape[-1], x.element_size()):
        return False
    return (
        HAS_TRITON_GELU
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() >= min_numel
        and not torch.jit.is_scripting()
    )


def _supports_fused_row_width(n_cols: int, element_size: int) -> bool:
    if not HAS_TRITON_GELU:
        return False
    return n_cols <= 65536 // element_size


def _block_size_for_num_cols(n_cols: int, element_size: int) -> int:
    max_fused_size = 65536 // element_size
    block_size = triton.next_power_of_2(n_cols)
    if block_size > max_fused_size:
        raise RuntimeError(
            f"Triton GELU only supports feature dims up to {max_fused_size} "
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

if HAS_TRITON_GELU:

    @triton.jit
    def gelu_fwd_kernel(
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
        x_row = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        INV_SQRT_2 = 0.7071067811865476

        erf_val = tl.math.erf(x_row * INV_SQRT_2)
        y_row = 0.5 * x_row * (1.0 + erf_val)

        y_ptrs = Y + row_idx * stride_y_row + col
        tl.store(y_ptrs, y_row, mask=mask)

    @triton.jit
    def gelu_bwd_kernel(
        DY,
        X,
        DX,
        stride_dy_row,
        stride_x_row,
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
            x = tl.load(
                X + row_idx * stride_x_row + col,
                mask=row_mask & mask,
                other=0.0,
            ).to(tl.float32)

            # erf part: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            erf_val = tl.math.erf(x * 0.7071067811865476)

            # exp(-x^2/2) part
            x_sq = x * x
            exp_neg_x_sq_2 = tl.exp(-0.5 * x_sq)

            # dGELU/dx = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
            dgelu_dx = 0.5 * (1.0 + erf_val) + x * 0.3989422804014327 * exp_neg_x_sq_2

            dx = dy * dgelu_dx

            tl.store(
                DX + row_idx * stride_dx_row + col,
                dx,
                mask=row_mask & mask,
            )


# ============================== Autograd functions ===========================


class GeluTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not HAS_TRITON_GELU:
            raise RuntimeError("GeluTritonFunction requires Triton.")

        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x_flat.shape

        y_flat = torch.empty_like(x_flat)

        BLOCK_SIZE = _block_size_for_num_cols(N, x_flat.element_size())

        gelu_fwd_kernel[(M,)](
            x_flat,
            y_flat,
            x_flat.stride(0),
            y_flat.stride(0),
            N,
            BLOCK_SIZE,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        ctx.save_for_backward(x_flat)
        ctx.orig_shape = orig_shape
        ctx.N = N
        return y_flat.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        if not HAS_TRITON_GELU:
            raise RuntimeError("GeluTritonFunction backward requires Triton.")

        x_flat = ctx.saved_tensors[0]
        orig_shape = ctx.orig_shape
        N = ctx.N

        # Forward flattens all leading dimensions into rows, so the upstream
        # gradient must use the same layout before entering the row-wise kernel.
        grad_output_flat = grad_output.reshape(-1, N).contiguous()
        M = grad_output_flat.shape[0]
        grad_input_flat = torch.empty_like(x_flat)

        BLOCK_SIZE = _block_size_for_num_cols(N, x_flat.element_size())
        n_row_blocks = max(
            1,
            min(
                torch.cuda.get_device_properties(x_flat.device).multi_processor_count,
                M,
            ),
        )
        rows_per_program = triton.cdiv(M, n_row_blocks)

        gelu_bwd_kernel[(n_row_blocks,)](
            grad_output_flat,
            x_flat,
            grad_input_flat,
            grad_output_flat.stride(0),
            x_flat.stride(0),
            grad_input_flat.stride(0),
            M,
            N,
            BLOCK_SIZE,
            rows_per_program,
            num_warps=_num_warps_for_block(BLOCK_SIZE),
        )

        return grad_input_flat.reshape(orig_shape)


# ============================== Python API ====================================


def gelu_fallback(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU using native PyTorch operations."""
    return torch.nn.functional.gelu(x, approximate='tanh')


def triton_gelu(
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply GELU using Triton when supported, otherwise use PyTorch."""
    if _should_use_triton_gelu(x):
        return GeluTritonFunction.apply(x)
    return gelu_fallback(x)


__all__ = [
    "HAS_TRITON_GELU",
    "_should_use_triton_gelu",
    "_supports_fused_row_width",
    "GeluTritonFunction",
    "triton_gelu",
    "gelu_fallback",
]
