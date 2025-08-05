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
        """Triton kernel: Fused RMSNorm + scaling + optional bias"""
        row_id = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size

        x_row = x_ptr + row_id * hidden_size
        out_row = out_ptr + row_id * hidden_size

        x = tl.load(x_row + col_offsets, mask=mask, other=0.0)
        x_squared = x * x
        mean_x_squared = tl.sum(x_squared, axis=0) / hidden_size
        inv_rms = tl.rsqrt(mean_x_squared + eps)

        alpha = tl.load(alpha_ptr + col_offsets, mask=mask, other=1.0)
        out = x * inv_rms * alpha

        if use_bias:
            beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
            out += beta

        tl.store(out_row + col_offsets, out, mask=mask)

    def triton_fused_norm_scale(
        x: torch.Tensor,
        alpha: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Torch wrapper for Triton fused norm kernel"""
        if not TRITON_AVAILABLE or not x.is_cuda or x.numel() < 2048:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(var + eps)
            out = x * inv_rms * alpha
            if beta is not None:
                out += beta
            return out

        batch_size, seq_len, hidden_size = x.shape
        flat = x.view(-1, hidden_size).contiguous()
        out = torch.empty_like(flat)

        BLOCK_SIZE = triton.next_power_of_2(min(hidden_size, 1024))
        grid = (flat.shape[0],)

        fused_norm_scale_kernel[grid](
            flat,
            alpha,
            beta,
            out,
            flat.shape[0],
            hidden_size,
            eps=eps,
            use_bias=beta is not None,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out.view_as(x)


def should_use_triton(x: torch.Tensor, threshold: int = 2048) -> bool:
    return (
        TRITON_AVAILABLE
        and x.is_cuda
        and x.numel() > threshold
        and not torch.jit.is_scripting()
    )


class AdaptiveRMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        use_bias: bool = False,
        dropout: float = 0.0,
        global_rms: bool = False,
    ):
        """
        Adaptive RMSNorm with optional bias and Triton acceleration.
        Args:
            d_model: number of features
            eps: numerical stability
            use_bias: add beta parameter
            dropout: apply dropout after norm (0 = no dropout)
            global_rms: use global RMSNorm across B, T
        """
        super().__init__()
        self.eps = torch.tensor(eps, dtype=torch.float32)
        self.global_rms = global_rms

        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if use_bias else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if should_use_triton(x):
            out = triton_fused_norm_scale(x, self.alpha, self.beta, self.eps.item())
            return self.dropout(out)

        if self.global_rms:
            var = x.pow(2).mean(dim=(-2, -1), keepdim=True)
        else:
            var = x.pow(2).mean(dim=-1, keepdim=True)

        inv_rms = torch.rsqrt(var + self.eps)
        out = x * inv_rms * self.alpha
        if self.beta is not None:
            out += self.beta
        return self.dropout(out)

    def forward_export(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        use_bias: bool = False,
        dropout: float = 0.0,
    ):
        """
        Adaptive LayerNorm wrapper with optional bias and dropout.
        Applies standard LayerNorm first, then scale/bias with Triton fallback.
        """
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if use_bias else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        if should_use_triton(x, threshold=8192):
            try:
                x = triton_fused_norm_scale(x, self.alpha, self.beta, self.eps)
                return self.dropout(x)
            except Exception:
                pass

        out = x * self.alpha
        if self.beta is not None:
            out += self.beta
        return self.dropout(out)

    def forward_export(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

if TRITON_AVAILABLE:
        
    @triton.jit
    def rms_norm_kernel(
        x_ptr, out_ptr, weight_ptr,
        eps, n_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        x_row_ptr = x_ptr + row_idx * n_cols + col_offsets
        out_row_ptr = out_ptr + row_idx * n_cols + col_offsets

        mask = col_offsets < n_cols

        x = tl.load(x_row_ptr, mask=mask, other=0.0)
        x2 = x * x
        x2 = tl.where(mask, x2, 0.0)
        rms = tl.sqrt(tl.sum(x2, axis=0) / n_cols + eps)
        x_norm = x / rms

        weight = tl.load(weight_ptr + col_offsets, mask=mask)
        y = x_norm * weight
        tl.store(out_row_ptr, y, mask=mask)


    class RMSNorm(torch.nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.d_model = d_model
            self.weight = nn.Parameter(torch.ones(d_model))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.is_cuda, "Triton RMSNorm requires CUDA input"

            orig_shape = x.shape
            last_dim = orig_shape[-1]
            x_flat = x.reshape(-1, last_dim)
            y_flat = torch.empty_like(x_flat)

            grid = lambda meta: (x_flat.shape[0],)
            rms_norm_kernel[grid](
                x_flat, y_flat, self.weight,
                self.eps, last_dim,
                BLOCK_SIZE=triton.next_power_of_2(last_dim),
            )
            return y_flat.view(orig_shape)
else:
    class RMSNorm(nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            normed = x * torch.rsqrt(var + self.eps)
            return normed * self.weight


def create_norm_layer(norm_type: str, d_model: int, eps: float = 1e-5) -> nn.Module:
    """
    Factory for creating SOTA normalization layers.
    Supports 'layer', 'rms', 'adaptive_layer', 'adaptive_rms'.
    """
    if norm_type == "layer":
        return nn.LayerNorm(d_model, eps=eps)
    elif norm_type == "rms":
        return RMSNorm(d_model, eps=eps)
    elif norm_type == "adaptive_layer":
        return AdaptiveLayerNorm(d_model, eps=eps)
    elif norm_type == "adaptive_rms":
        return AdaptiveRMSNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
