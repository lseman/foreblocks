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


class RMSNorm(nn.Module):
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
# TemporalNorm (normalize along time axis per (B, D))
# ================================================================
class TemporalNorm(nn.Module):
    """
    Temporal normalization for sequences [B, T, D] (channel-last).
    - Computes per-sample, per-channel statistics across time T.
    - Supports standard (mean/std) or robust (median/IQR) normalization.
    - Optional causal/rolling window to avoid leakage into the future.
    - Optional mask to ignore missing timesteps in statistics.
    - Optional affine parameters (γ, β) like LayerNorm.
    
    Args:
        d_model:   number of channels (D)
        eps:       numerical stability
        affine:    learnable scale/bias after normalization
        mode:      'standard' (mean/std) or 'robust' (median/IQR*0.741)
        causal:    if True, uses left-causal windows; else full (or centered if window_size provided)
        window_size: int or None. If None -> full-window statistics over T.
                     If int -> rolling window size over time dimension.
        center:    if False, only scales (no mean/median subtraction)
        scale:     if False, only centers (no std/IQR division)
    Inputs:
        x:    [B, T, D]
        mask: optional [B, T, 1] or [B, T, D] (1=valid, 0=missing)
        return_stats: if True, returns (y, stats) where stats={'loc':..., 'scale':...}
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        affine: bool = True,
        mode: str = "standard",
        causal: bool = False,
        window_size: Optional[int] = None,
        center: bool = True,
        scale: bool = True,
    ):
        super().__init__()
        assert mode in {"standard", "robust"}
        self.d_model = d_model
        self.eps = eps
        self.affine = affine
        self.mode = mode
        self.causal = causal
        self.window_size = window_size
        self.center = center
        self.scale = scale

        if affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias   = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mask is None:
            return x, None
        # broadcast mask to [B,T,D]
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.expand(-1, -1, x.size(-1))
        return x * mask, mask

    def _reduce_full(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute location & scale over time axis (dim=1), per (B, D)
        if self.mode == "standard":
            if mask is None:
                loc = x.mean(dim=1, keepdim=True) if self.center else torch.zeros_like(x[:, :1, :])
                var = x.var(dim=1, unbiased=False, keepdim=True) if self.scale else torch.ones_like(x[:, :1, :])
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                loc = (x.sum(dim=1, keepdim=True) / denom) if self.center else torch.zeros_like(x[:, :1, :])
                diff = x - loc
                var = ((diff * diff * mask).sum(dim=1, keepdim=True) / denom) if self.scale else torch.ones_like(loc)
            scale = (var + self.eps).sqrt()
        else:  # robust
            if mask is not None:
                # replace missing by NaN and use nan-median; torch.nanmedian available on newer PyTorch
                x_ = torch.where(mask > 0, x, torch.nan)
                loc = torch.nanmedian(x_, dim=1, keepdim=True).values if self.center else torch.zeros_like(x[:, :1, :])
                diff = torch.abs(x - loc)
                mad  = torch.nanmedian(torch.where(mask > 0, diff, torch.nan), dim=1, keepdim=True).values if self.scale else torch.ones_like(loc)
            else:
                loc = torch.median(x, dim=1, keepdim=True).values if self.center else torch.zeros_like(x[:, :1, :])
                mad = torch.median(torch.abs(x - loc), dim=1, keepdim=True).values if self.scale else torch.ones_like(loc)
            # IQR-like scaling; 0.741 ~ 1/Φ^{-1}(0.75) for normal consistency
            scale = (mad * (1.0 / 0.741) + self.eps)
        return loc, scale

    def _reduce_rolling(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rolling (causal or centered) window stats over time.
        Returns loc, scale shaped [B, T, D].
        """
        B, T, D = x.shape
        W = int(self.window_size)
        # For efficiency and numerical stability we use cumulative sums (standard mode).
        if self.mode == "standard":
            if mask is None:
                ones = x.new_ones(B, T, D)
                m = ones
                sx = x.cumsum(dim=1)                                # sum
                sx2 = (x * x).cumsum(dim=1)                         # sum of squares
            else:
                m = mask
                sx = (x * m).cumsum(dim=1)
                sx2 = (x * x * m).cumsum(dim=1)

            # helper to get windowed cumulative difference
            def win_cumsum(cs):
                if self.causal:
                    # causal window [t-W+1, t]
                    pad = cs.new_zeros(B, 1, D)
                    cs_pad = torch.cat([pad, cs], dim=1)            # align for easy diff
                    prev = torch.roll(cs_pad, W, dims=1)[:, :T, :]
                    return cs_pad[:, 1:, :] - prev
                else:
                    # centered: [t - W//2, t + W//2] clipped to bounds
                    # build indices per t (vectorized gather)
                    half = W // 2
                    idx_r = torch.arange(T, device=x.device).clamp_max(T-1)
                    l = (idx_r - half).clamp_min(0)
                    r = (idx_r + (W - half - 1)).clamp_max(T-1)
                    # prefix sums for variable windows:
                    pad = cs.new_zeros(B, 1, D)
                    cs_pad = torch.cat([pad, cs], dim=1)            # shape [B,T+1,D]
                    # gather r+1 and l
                    r1 = cs_pad.gather(1, (r+1)[None, :, None].expand(B, -1, D))
                    l0 = cs_pad.gather(1, l[None, :, None].expand(B, -1, D))
                    return r1 - l0

            win_m  = win_cumsum(m.cumsum(dim=1) if mask is not None else m.cumsum(dim=1))
            win_sx = win_cumsum(sx)
            win_sx2= win_cumsum(sx2)

            denom = win_m.clamp_min(1.0)
            loc = win_sx / denom if self.center else x.new_zeros(B, T, D)
            var = (win_sx2 / denom - (loc * loc)) if self.scale else x.new_ones(B, T, D)
            scale = (var + self.eps).sqrt()
            return loc, scale

        else:
            # robust rolling (median/MAD) – heavier; compute with unfold on time
            # We do causal or centered by slicing windows per t.
            # For performance on long T you may prefer approximate robust stats.
            loc = []
            scale = []
            half = (W // 2)
            for t in range(T):
                if self.causal:
                    a, b = max(0, t - W + 1), t + 1
                else:
                    a, b = max(0, t - half), min(T, t + (W - half))
                xw = x[:, a:b, :]          # [B, w, D]
                if mask is not None:
                    mw = mask[:, a:b, :].bool()
                    # replace missing with NaN then nanmedian
                    xw = torch.where(mw, xw, torch.nan)
                    med = torch.nanmedian(xw, dim=1, keepdim=True).values if self.center else xw.new_zeros(B, 1, D)
                    mad = torch.nanmedian(torch.abs(xw - med), dim=1, keepdim=True).values if self.scale else xw.new_ones(B, 1, D)
                else:
                    med = torch.median(xw, dim=1, keepdim=True).values if self.center else xw.new_zeros(B, 1, D)
                    mad = torch.median(torch.abs(xw - med), dim=1, keepdim=True).values if self.scale else xw.new_ones(B, 1, D)
                loc.append(med.squeeze(1))
                scale.append((mad * (1.0 / 0.741) + self.eps).squeeze(1))
            loc = torch.stack(loc, dim=1)    # [B,T,D]
            scale = torch.stack(scale, dim=1)
            return loc, scale

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"TemporalNorm expects [B, T, D={self.d_model}], got {tuple(x.shape)}")
        B, T, D = x.shape

        x_masked, m = self._apply_mask(x, mask)

        if self.window_size is None:
            loc, scale = self._reduce_full(x_masked, m)
            # broadcast to [B,T,D]
            loc = loc.expand(B, T, D)
            scale = scale.expand(B, T, D)
        else:
            loc, scale = self._reduce_rolling(x_masked, m)

        y = x
        if self.center:
            y = y - loc
        if self.scale:
            y = y / scale

        if self.affine:
            y = y * self.weight.view(1, 1, D)
            if self.bias is not None:
                y = y + self.bias.view(1, 1, D)

        if return_stats:
            return y, {"loc": loc.detach(), "scale": scale.detach()}
        return y

    @torch.no_grad()
    def inverse(self, y: torch.Tensor, stats: dict) -> torch.Tensor:
        """
        Invert normalization using previously returned stats from forward(return_stats=True).
        """
        loc, scale = stats["loc"], stats["scale"]
        x = y
        D = self.d_model
        if self.affine:
            x = x / self.weight.view(1, 1, D)
            if self.bias is not None:
                x = x - self.bias.view(1, 1, D)
        if self.scale:
            x = x * scale
        if self.center:
            x = x + loc
        return x

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., NeurIPS 2021)
    Normalizes each instance (sample) independently over time, then reverses it during inference.
    """

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self.training_mode = True

    def forward(self, x: torch.Tensor, mode: str = "norm", stats: Optional[dict] = None):
        """
        x: [B, L, C]
        mode: "norm" (normalize) or "denorm" (reverse normalization)
        stats: {mean, std} optional external stats for reverse step
        """
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x_norm = (x - self.mean) / self.std
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm

        elif mode == "denorm":
            assert hasattr(self, "mean") and hasattr(self, "std"), "Must call norm() before denorm()"
            mean = stats.get("mean", self.mean) if stats else self.mean
            std = stats.get("std", self.std) if stats else self.std
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            return x * std + mean

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def reset_stats(self):
        if hasattr(self, "mean"): del self.mean
        if hasattr(self, "std"): del self.std
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
    elif norm_type in ('temporal', 'temporalnorm'):
        return TemporalNorm(d_model, eps=eps, **kwargs)
    elif norm_type in ("revin"):
        return RevIN(d_model, eps=eps, **kwargs)
    elif norm_type in ('rms', 'rmsnorm'):
        return RMSNorm(d_model, eps=eps, **kwargs)
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
