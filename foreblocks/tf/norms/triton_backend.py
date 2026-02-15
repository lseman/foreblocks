import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def _should_use_triton(x: torch.Tensor, min_numel: int = 4096) -> bool:
    return (
        TRITON_AVAILABLE
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() >= min_numel
        and not torch.jit.is_scripting()
    )


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
    def rmsnorm_bwd_kernel(
        DY,
        X,
        W,
        Rms,
        DX,
        DW,
        stride_dy_row,
        stride_x_row,
        stride_dx_row,
        n_cols,
        eps,
        HAS_WEIGHT: tl.constexpr,
        COMPUTE_DW: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        dy_ptrs = DY + row_idx * stride_dy_row + col
        x_ptrs = X + row_idx * stride_x_row + col

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        rms = tl.load(Rms + row_idx)

        if HAS_WEIGHT:
            w = tl.load(W + col, mask=mask, other=1.0).to(tl.float32)
            dy_scaled = dy * w
        else:
            dy_scaled = dy

        sum_dy_x = tl.sum(dy_scaled * x, axis=0)
        dx = rms * (dy_scaled - x * sum_dy_x * rms * rms / n_cols)

        dx_ptrs = DX + row_idx * stride_dx_row + col
        tl.store(dx_ptrs, dx, mask=mask)

        if COMPUTE_DW:
            x_normed = x * rms
            tl.atomic_add(DW + col, tl.where(mask, dy * x_normed, 0.0))

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


class LayerNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        if not TRITON_AVAILABLE:
            raise RuntimeError("LayerNormTritonFunction called but Triton is not available.")
        if weight is None or bias is None:
            raise RuntimeError("LayerNormTritonFunction expects non-None weight and bias.")

        orig_shape = x.shape
        x = x.view(-1, x.shape[-1]).contiguous()
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
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        if not TRITON_AVAILABLE:
            raise RuntimeError("LayerNormTritonFunction backward called but Triton is not available.")

        x, weight, bias, mean, rstd = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        N = ctx.N

        grad_output = grad_output.view(-1, grad_output.shape[-1]).contiguous()
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

        return grad_input.view(orig_shape), grad_weight, grad_bias, None


class RMSNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if not TRITON_AVAILABLE:
            raise RuntimeError("RMSNormTritonFunction called but Triton is not available.")
        if weight is None:
            raise RuntimeError("RMSNormTritonFunction expects non-None weight.")

        orig_shape = x.shape
        x = x.view(-1, x.shape[-1]).contiguous()
        M, N = x.shape

        y = torch.empty_like(x)
        rms = torch.empty((M,), dtype=torch.float32, device=x.device)

        BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)

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
        )

        ctx.save_for_backward(x, weight, rms)
        ctx.orig_shape = orig_shape
        ctx.eps = eps
        ctx.N = N
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        if not TRITON_AVAILABLE:
            raise RuntimeError("RMSNormTritonFunction backward called but Triton is not available.")

        x, weight, rms = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        eps = ctx.eps
        N = ctx.N

        grad_output = grad_output.view(-1, grad_output.shape[-1]).contiguous()
        M = grad_output.shape[0]

        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros_like(weight)

        BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)
        rmsnorm_bwd_kernel[(M,)](
            grad_output,
            x,
            weight,
            rms,
            grad_input,
            grad_weight,
            grad_output.stride(0),
            x.stride(0),
            grad_input.stride(0),
            N,
            eps,
            True,
            True,
            BLOCK_SIZE,
        )

        return grad_input.view(orig_shape), grad_weight, None


def triton_scale_bias(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor | None) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton_scale_bias called but Triton is not available.")

    *lead, H = x.shape
    x_flat = x.reshape(-1, H).contiguous()
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = min(triton.next_power_of_2(H), 2048)
    grid = (x_flat.shape[0],)

    scale_bias_kernel[grid](
        x_flat,
        y_flat,
        alpha,
        beta,
        H,
        beta is not None,
        BLOCK_SIZE,
    )

    return y_flat.view(*lead, H)


def triton_fused_rmsnorm_scale_bias(
    x: torch.Tensor,
    rms_weight: torch.Tensor | None,
    alpha: torch.Tensor,
    beta: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton_fused_rmsnorm_scale_bias called but Triton is not available.")

    *lead, H = x.shape
    x_flat = x.reshape(-1, H).contiguous()
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = min(triton.next_power_of_2(H), 2048)
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
    )

    return y_flat.view(*lead, H)


__all__ = [
    "TRITON_AVAILABLE",
    "LayerNormTritonFunction",
    "RMSNormTritonFunction",
    "_should_use_triton",
    "triton_scale_bias",
    "triton_fused_rmsnorm_scale_bias",
]
