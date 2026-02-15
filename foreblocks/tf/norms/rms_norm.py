import torch
import torch.nn as nn

from .triton_backend import (
    TRITON_AVAILABLE,
    RMSNormTritonFunction,
    _should_use_triton,
)


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
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension to be {self.d_model}, got {x.shape[-1]}"
            )

        if (
            self.elementwise_affine
            and _should_use_triton(x, min_numel=2048)
            and TRITON_AVAILABLE
        ):
            return RMSNormTritonFunction.apply(x, self.weight, self.eps)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x_norm = x_norm * self.weight
        return x_norm

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm with additional learnable scale and bias parameters."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        use_bias: bool = False,
        dropout: float = 0.0,
        global_rms: bool = False,
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
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension to be {self.d_model}, got {x.shape[-1]}"
            )

        if not self.global_rms:
            if _should_use_triton(x, min_numel=2048) and TRITON_AVAILABLE:
                base = RMSNormTritonFunction.apply(x, self.weight, self.eps)
            else:
                var = x.pow(2).mean(dim=-1, keepdim=True)
                base = x * torch.rsqrt(var + self.eps)
                base = base * self.weight
        else:
            var = x.pow(2).mean(dim=(-2, -1), keepdim=True)
            base = x * torch.rsqrt(var + self.eps)
            base = base * self.weight

        y = base * self.alpha
        if self.beta is not None:
            y = y + self.beta

        return self.dropout(y)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, eps={self.eps}, use_bias={self.use_bias}, "
            f"global_rms={self.global_rms}"
        )


__all__ = ["RMSNorm", "AdaptiveRMSNorm"]
