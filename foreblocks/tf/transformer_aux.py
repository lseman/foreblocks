from typing import Optional

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def fused_norm_scale_kernel(
        x_ptr,
        alpha_ptr,
        beta_ptr,
        out_ptr,
        batch_seq_size,
        hidden_size,
        eps,
        use_bias: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Optimized fused normalization + scaling kernel"""
        row_id = tl.program_id(0)

        # Load data
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size

        x_row_ptr = x_ptr + row_id * hidden_size
        out_row_ptr = out_ptr + row_id * hidden_size

        # Load input row
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)

        # Compute RMS normalization
        x_squared = x * x
        mean_x_squared = tl.sum(x_squared, axis=0) / hidden_size
        inv_rms = tl.rsqrt(mean_x_squared + eps)

        # Load scale parameters
        alpha = tl.load(alpha_ptr + col_offsets, mask=mask, other=1.0)

        # Apply normalization and scaling
        out = x * inv_rms * alpha

        if use_bias:
            beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
            out = out + beta

        # Store result
        tl.store(out_row_ptr + col_offsets, out, mask=mask)

    def triton_fused_norm_scale(
        x: torch.Tensor,
        alpha: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Optimized Triton-based fused normalization + scaling"""
        if not TRITON_AVAILABLE or not x.is_cuda or x.numel() < 4096:
            # Fallback to optimized PyTorch implementation
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + eps)
            out = x * inv_rms * alpha
            if beta is not None:
                out = out + beta
            return out

        try:
            batch_size, seq_len, hidden_size = x.shape
            batch_seq_size = batch_size * seq_len

            # Reshape for kernel processing
            x_flat = x.view(batch_seq_size, hidden_size)
            out = torch.empty_like(x_flat)

            # Launch kernel
            grid = (batch_seq_size,)
            BLOCK_SIZE = triton.next_power_of_2(min(hidden_size, 1024))

            fused_norm_scale_kernel[grid](
                x_flat,
                alpha,
                beta,
                out,
                batch_seq_size,
                hidden_size,
                eps=eps,
                use_bias=beta is not None,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out.view_as(x)

        except Exception:
            # Fallback to PyTorch if Triton fails
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + eps)
            out = x * inv_rms * alpha
            if beta is not None:
                out = out + beta
            return out


class AdaptiveRMSNorm(nn.Module):
    """Optimized Adaptive RMSNorm with automatic backend selection"""

    def __init__(self, d_model, eps=1e-5, use_bias=False):
        super().__init__()
        self.eps = eps
        self.use_bias = use_bias
        self.d_model = d_model

        # Parameters (keeping original names for compatibility)
        self.alpha = nn.Parameter(torch.ones(d_model))
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(d_model))
        else:
            self.beta = None

        # Cache configuration for faster runtime checks
        self._triton_threshold = 4096

        print(f"[Normalization] AdaptiveRMSNorm (Triton available: {TRITON_AVAILABLE})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward with smart backend selection"""
        # Use optimized backend selection logic
        if (
            TRITON_AVAILABLE
            and x.is_cuda
            and x.numel() > self._triton_threshold
            and not torch.jit.is_scripting()
        ):
            # Try Triton path
            return triton_fused_norm_scale(x, self.alpha, self.beta, self.eps)

        # Optimized PyTorch fallback
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps)
        out = x * inv_rms * self.alpha

        if self.use_bias and self.beta is not None:
            out = out + self.beta

        return out


class AdaptiveLayerNorm(nn.Module):
    """Optimized Adaptive LayerNorm with better performance"""

    def __init__(self, d_model, eps=1e-5, use_bias=False):
        super().__init__()

        # Use PyTorch's highly optimized LayerNorm as base
        self.norm = nn.LayerNorm(d_model, eps=eps)

        # Adaptive scaling parameters (keeping original names)
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.use_bias = use_bias

        if use_bias:
            self.beta = nn.Parameter(torch.zeros(d_model))
        else:
            self.beta = None

        # Cache for faster runtime checks
        self._triton_threshold = (
            8192  # Higher threshold for LayerNorm since it's already optimized
        )

        print(
            f"[Normalization] Adaptive LayerNorm (Triton available: {TRITON_AVAILABLE})"
        )

    def forward(self, x):
        """Optimized forward with efficient scaling"""
        # Apply standard LayerNorm (already highly optimized)
        normed = self.norm(x)

        # Apply adaptive scaling efficiently
        if (
            TRITON_AVAILABLE
            and x.is_cuda
            and x.numel() > self._triton_threshold
            and not torch.jit.is_scripting()
        ):
            # Use Triton for scaling if beneficial
            try:
                return triton_fused_norm_scale(
                    normed, self.alpha, self.beta if self.use_bias else None, eps=0.0
                )
            except Exception:
                pass  # Fall through to PyTorch version

        # Optimized PyTorch scaling
        out = normed * self.alpha
        if self.use_bias and self.beta is not None:
            out = out + self.beta

        return out


def create_norm_layer(norm_type: str, d_model: int, eps: float) -> nn.Module:
    """
    Optimized normalization layer factory (keeping original interface).
    Now with better performance and smarter backend selection.
    """

    if norm_type == "layer":
        return nn.LayerNorm(d_model, eps=eps)
    elif norm_type == "rms":
        return AdaptiveRMSNorm(d_model, eps=eps)
    elif norm_type == "adaptive_layer":
        return AdaptiveLayerNorm(d_model, eps=eps)
    elif norm_type == "adaptive_rms":
        return AdaptiveRMSNorm(d_model, eps=eps)
    else:
        # Default fallback
        return nn.LayerNorm(d_model, eps=eps)
