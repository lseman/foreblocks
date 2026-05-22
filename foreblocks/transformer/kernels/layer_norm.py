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
    def layernorm_bwd_dwdb_row_kernel(
        DY,
        X,
        Mean,
        Rstd,
        DW,
        DB,
        stride_dy_row,
        stride_x_row,
        n_cols,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
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

        x_normed = (x - mean) * rstd

        if HAS_WEIGHT:
            tl.atomic_add(DW + col, tl.where(mask, dy * x_normed, 0.0))
        if HAS_BIAS:
            tl.atomic_add(DB + col, tl.where(mask, dy, 0.0))


# ============================== Autograd function ============================


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

        BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)

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
        BLOCK_DX = min(triton.next_power_of_2(N), 2048)
        layernorm_bwd_dx_kernel[(M,)](
            grad_output,
            x,
            weight,
            mean,
            rstd,
            grad_input,
            grad_output.stride(0),
            x.stride(0),
            grad_input.stride(0),
            N,
            True,
            BLOCK_DX,
        )

        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        BLOCK_WB = min(triton.next_power_of_2(N), 1024)
        layernorm_bwd_dwdb_row_kernel[(M,)](
            grad_output,
            x,
            mean,
            rstd,
            grad_weight,
            grad_bias,
            grad_output.stride(0),
            x.stride(0),
            N,
            True,
            True,
            BLOCK_WB,
        )

        return grad_input.reshape(orig_shape), grad_weight, grad_bias, None


__all__ = [
    "TRITON_AVAILABLE",
    "LayerNormTritonFunction",
    "layernorm_fwd_kernel",
    "layernorm_bwd_dx_kernel",
    "layernorm_bwd_dwdb_row_kernel",
]
