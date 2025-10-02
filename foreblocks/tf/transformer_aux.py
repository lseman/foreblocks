import math

# norms.py - High-Performance Normalization Library with Triton Acceleration
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Triton availability ----------------
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _should_use_triton(x: torch.Tensor, min_numel: int = 4096) -> bool:
    """Determine if we should use Triton based on tensor properties."""
    return (
        TRITON_AVAILABLE
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() >= min_numel
        and not torch.jit.is_scripting()
        and x.is_contiguous()
    )


# ================================================================
# Triton Kernels - LayerNorm
# ================================================================
if TRITON_AVAILABLE:
    @triton.jit
    def layernorm_fwd_kernel(
        X, Y, W, B, Mean, Rstd,
        stride_x_row, stride_y_row,
        n_cols, eps,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Optimized LayerNorm forward kernel."""
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
        DY, X, W, Mean, Rstd, DX,
        stride_dy_row, stride_x_row, stride_dx_row,
        n_cols,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """LayerNorm backward kernel for input gradients."""
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        dy_ptrs = DY + row_idx * stride_dy_row + col
        x_ptrs  = X  + row_idx * stride_x_row  + col

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x  = tl.load(x_ptrs,  mask=mask, other=0.0).to(tl.float32)

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

    # ---- NEW: row-wise dW/dB kernel using atomics (no Python-side loops) ----
    @triton.jit
    def layernorm_bwd_dwdb_row_kernel(
        DY, X, Mean, Rstd, DW, DB,
        stride_dy_row, stride_x_row,
        n_cols,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Each program handles one row: accumulates dW/dB contributions and atomically adds to global DW/DB.
        """
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        dy_ptrs = DY + row_idx * stride_dy_row + col
        x_ptrs  = X  + row_idx * stride_x_row  + col

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x  = tl.load(x_ptrs,  mask=mask, other=0.0).to(tl.float32)

        mean = tl.load(Mean + row_idx)
        rstd = tl.load(Rstd + row_idx)

        x_normed = (x - mean) * rstd

        if HAS_WEIGHT:
            tl.atomic_add(DW + col, tl.where(mask, dy * x_normed, 0.0))
        if HAS_BIAS:
            tl.atomic_add(DB + col, tl.where(mask, dy, 0.0))

    # ================================================================
    # Triton Kernels - RMSNorm
    # ================================================================
    @triton.jit
    def rmsnorm_fwd_kernel(
        X, Y, W, Rms,  # Added Rms parameter
        stride_x_row, stride_y_row,
        n_cols, eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """RMSNorm forward kernel - now saves RMS for backward."""
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
        
        # Save RMS for backward pass
        tl.store(Rms + row_idx, rms)

    @triton.jit
    def rmsnorm_bwd_kernel(
        DY, X, W, Rms, DX, DW,  # Added Rms parameter
        stride_dy_row, stride_x_row, stride_dx_row,
        n_cols, eps,
        HAS_WEIGHT: tl.constexpr,
        COMPUTE_DW: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """RMSNorm backward kernel - uses saved RMS."""
        row_idx = tl.program_id(0)

        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        dy_ptrs = DY + row_idx * stride_dy_row + col
        x_ptrs  = X  + row_idx * stride_x_row  + col

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x  = tl.load(x_ptrs,  mask=mask, other=0.0).to(tl.float32)

        # Use saved RMS instead of recomputing
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

    # ================================================================
    # Triton Kernels - Scale and Bias
    # ================================================================
    @triton.jit
    def scale_bias_kernel(
        X, Y, Alpha, Beta,
        n_cols,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scale and bias kernel."""
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

    # ================================================================
    # Triton Kernels - Fused RMSNorm + Scale + Bias
    # ================================================================
    @triton.jit
    def fused_rmsnorm_scale_bias_kernel(
        X, Y, W, Alpha, Beta,
        stride_x_row, stride_y_row,
        n_cols, eps,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMSNorm + scale + bias kernel."""
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


# ================================================================
# Autograd Functions
# ================================================================
class LayerNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1]).contiguous()
        M, N = x.shape

        y = torch.empty_like(x)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)

        layernorm_fwd_kernel[(M,)](
            x, y, weight, bias, mean, rstd,
            x.stride(0), y.stride(0),
            N, eps,
            weight is not None,
            bias is not None,
            BLOCK_SIZE,
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.orig_shape = orig_shape
        ctx.N = N
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        N = ctx.N

        grad_output = grad_output.view(-1, grad_output.shape[-1]).contiguous()
        M = grad_output.shape[0]

        # dX
        grad_input = torch.empty_like(x)
        BLOCK_DX = min(triton.next_power_of_2(N), 2048)
        layernorm_bwd_dx_kernel[(M,)](
            grad_output, x, weight, mean, rstd, grad_input,
            grad_output.stride(0), x.stride(0), grad_input.stride(0),
            N,
            weight is not None,
            BLOCK_DX,
        )

        grad_weight = None
        grad_bias = None

        # dW/dB via row-wise atomics
        if weight is not None or bias is not None:
            # always provide valid storage for pointers; use temp buffers if param is None
            if weight is not None:
                grad_weight = torch.zeros_like(weight)
                dw_ptr = grad_weight
            else:
                dw_ptr = x.new_zeros((N,))  # dummy
            if bias is not None:
                grad_bias = torch.zeros_like(bias)
                db_ptr = grad_bias
            else:
                db_ptr = x.new_zeros((N,))  # dummy

            BLOCK_WB = min(triton.next_power_of_2(N), 1024)
            layernorm_bwd_dwdb_row_kernel[(M,)](
                grad_output, x, mean, rstd,
                dw_ptr, db_ptr,
                grad_output.stride(0), x.stride(0),
                N,
                weight is not None,
                bias is not None,
                BLOCK_WB,
            )

        return grad_input.view(orig_shape), grad_weight, grad_bias, None


# Fix 3: RMSNormTritonFunction.forward - save RMS
class RMSNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1]).contiguous()
        M, N = x.shape

        y = torch.empty_like(x)
        rms = torch.empty((M,), dtype=torch.float32, device=x.device)  # Added
        
        BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)

        rmsnorm_fwd_kernel[(M,)](
            x, y, weight, rms,  # Added rms
            x.stride(0), y.stride(0),
            N, eps,
            weight is not None,
            BLOCK_SIZE,
        )

        ctx.save_for_backward(x, weight, rms)  # Save rms
        ctx.orig_shape = orig_shape
        ctx.eps = eps
        ctx.N = N
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rms = ctx.saved_tensors  # Load rms
        orig_shape = ctx.orig_shape
        eps = ctx.eps
        N = ctx.N

        grad_output = grad_output.view(-1, grad_output.shape[-1]).contiguous()
        M = grad_output.shape[0]

        grad_input = torch.empty_like(x)
        
        # Fixed: Always provide valid pointer for grad_weight
        if weight is not None:
            grad_weight = torch.zeros_like(weight)
            dw_ptr = grad_weight
        else:
            grad_weight = None
            dw_ptr = x.new_zeros((N,))  # Dummy buffer

        BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)
        rmsnorm_bwd_kernel[(M,)](
            grad_output, x, weight, rms, grad_input, dw_ptr,  # Pass rms
            grad_output.stride(0), x.stride(0), grad_input.stride(0),
            N, eps,
            weight is not None,
            weight is not None,
            BLOCK_SIZE,
        )

        return grad_input.view(orig_shape), grad_weight, None

# ================================================================
# Helpers for scale/bias and fused RMS
# ================================================================
def triton_scale_bias(x: torch.Tensor, alpha: torch.Tensor, beta: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply scale and optional bias using Triton."""
    *lead, H = x.shape
    x_flat = x.reshape(-1, H).contiguous()
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = min(triton.next_power_of_2(H), 2048)
    grid = (x_flat.shape[0],)

    scale_bias_kernel[grid](
        x_flat, y_flat, alpha, beta,
        H,
        beta is not None,
        BLOCK_SIZE,
    )

    return y_flat.view(*lead, H)


def triton_fused_rmsnorm_scale_bias(
    x: torch.Tensor,
    rms_weight: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: Optional[torch.Tensor],
    eps: float
) -> torch.Tensor:
    """Fused RMSNorm + scale + bias using Triton."""
    *lead, H = x.shape
    x_flat = x.reshape(-1, H).contiguous()
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = min(triton.next_power_of_2(H), 2048)
    grid = (x_flat.shape[0],)

    fused_rmsnorm_scale_bias_kernel[grid](
        x_flat, y_flat, rms_weight, alpha, beta,
        x_flat.stride(0), y_flat.stride(0),
        H, eps,
        rms_weight is not None,
        beta is not None,
        BLOCK_SIZE,
    )

    return y_flat.view(*lead, H)


# ================================================================
# Main Normalization Classes
# ================================================================
class FastLayerNorm(nn.Module):
    """High-performance LayerNorm with Triton acceleration."""

    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input validation
        if input.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(
                f"Expected last dimensions to be {self.normalized_shape}, "
                f"got {input.shape[-len(self.normalized_shape):]}"
            )
        
        if _should_use_triton(input, min_numel=4096):
            return LayerNormTritonFunction.apply(input, self.weight, self.bias, self.eps)
        else:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class FastRMSNorm(nn.Module):
    """High-performance RMSNorm with Triton acceleration."""

    def __init__(self, d_model: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension to be {self.d_model}, got {x.shape[-1]}"
            )
        
        if _should_use_triton(x, min_numel=2048):
            return RMSNormTritonFunction.apply(x, self.weight, self.eps)
        else:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight is not None:
                x = x * self.weight
            return x

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class AdaptiveLayerNorm(nn.Module):
    """LayerNorm followed by learnable scale and bias."""

    def __init__(self, d_model: int, eps: float = 1e-5, use_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.use_bias = use_bias

        self.norm = FastLayerNorm(d_model, eps=eps, elementwise_affine=True)

        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if use_bias else None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if _should_use_triton(x, min_numel=8192):
            x = triton_scale_bias(x, self.alpha, self.beta)
        else:
            x = x * self.alpha
            if self.beta is not None:
                x = x + self.beta
        return self.dropout(x)

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, eps={self.eps}, use_bias={self.use_bias}'


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm with additional learnable scale and bias parameters."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        use_bias: bool = False,
        dropout: float = 0.0,
        global_rms: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.use_bias = use_bias
        self.global_rms = global_rms

        self.weight = nn.Parameter(torch.ones(d_model))
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if use_bias else None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.global_rms and _should_use_triton(x, min_numel=2048):
            y = triton_fused_rmsnorm_scale_bias(x, self.weight, self.alpha, self.beta, self.eps)
        else:
            if self.global_rms:
                variance = x.pow(2).mean(dim=(-2, -1), keepdim=True)
            else:
                variance = x.pow(2).mean(dim=-1, keepdim=True)
            y = x * torch.rsqrt(variance + self.eps)
            if self.weight is not None:
                y = y * self.weight
            y = y * self.alpha
            if self.beta is not None:
                y = y + self.beta
        
        return self.dropout(y)

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, eps={self.eps}, use_bias={self.use_bias}, global_rms={self.global_rms}'


# ---------- Channel-last GroupNorm for [B, T, D] tensors ----------
class ChannelLastGroupNorm(nn.Module):
    """
    GroupNorm for channel-last tensors [B, ..., C].
    Internally permutes to [B, C, ...] → group_norm → permutes back.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f'num_channels ({num_channels}) must be divisible by num_groups ({num_groups})')
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("ChannelLastGroupNorm expects at least 2D input with channels in the last dim.")
        # permute last dim (C) to dim=1 for F.group_norm
        perm = list(range(x.dim()))
        perm.insert(1, perm.pop(-1))        # [B, C, T, ...]
        x_perm = x.permute(perm).contiguous()
        y_perm = F.group_norm(x_perm, self.num_groups, self.weight, self.bias, self.eps)
        # invert permutation
        inv = list(range(x.dim()))
        inv.append(inv.pop(1))              # [B, T, ..., C] back
        return y_perm.permute(inv)


# ================================================================
# Factory Function
# ================================================================
def create_norm_layer(
    norm_type: str,
    d_model: int,
    eps: float = 1e-5,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating normalization layers.

    Args:
        norm_type: Type ('layer', 'rms', 'adaptive_layer', 'adaptive_rms', 'group')
        d_model: Model dimension (channels for [B, T, D] tensors)
        eps: Epsilon for numerical stability
    """
    norm_type = norm_type.lower().replace('_', '').replace('-', '')

    if norm_type in ('layer', 'layernorm'):
        return FastLayerNorm(d_model, eps=eps, **kwargs)
    elif norm_type in ('rms', 'rmsnorm'):
        return FastRMSNorm(d_model, eps=eps, **kwargs)
    elif norm_type in ('adaptivelayer', 'adaptivelayernorm'):
        return AdaptiveLayerNorm(d_model, eps=eps, **kwargs)
    elif norm_type in ('adaptiverms', 'adaptivermsnorm'):
        return AdaptiveRMSNorm(d_model, eps=eps, **kwargs)
    elif norm_type in ('group', 'groupnorm'):
        num_groups = kwargs.pop('num_groups', 32)
        return ChannelLastGroupNorm(num_groups, d_model, eps=eps, **kwargs)
    else:
        raise ValueError(
            f"Unsupported norm type: {norm_type}. "
            f"Supported types: layer, rms, adaptive_layer, adaptive_rms, group"
        )
