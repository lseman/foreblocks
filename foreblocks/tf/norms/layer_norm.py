from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_backend import (
    TRITON_AVAILABLE,
    LayerNormTritonFunction,
    _should_use_triton,
    triton_scale_bias,
)


class FastLayerNorm(nn.Module):
    """High-performance LayerNorm with Triton acceleration."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
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
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-len(self.normalized_shape) :] != self.normalized_shape:
            raise ValueError(
                f"Expected last dimensions to be {self.normalized_shape}, "
                f"got {input.shape[-len(self.normalized_shape):]}"
            )

        if (
            self.elementwise_affine
            and _should_use_triton(input, min_numel=4096)
            and TRITON_AVAILABLE
        ):
            return LayerNormTritonFunction.apply(input, self.weight, self.bias, self.eps)
        return F.layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class AdaptiveLayerNorm(nn.Module):
    """LayerNorm followed by learnable scale and bias."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
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

        if (
            TRITON_AVAILABLE
            and not torch.is_grad_enabled()
            and _should_use_triton(x, min_numel=8192)
        ):
            x = triton_scale_bias(x, self.alpha, self.beta)
        else:
            x = x * self.alpha
            if self.beta is not None:
                x = x + self.beta

        return self.dropout(x)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}, use_bias={self.use_bias}"


__all__ = ["FastLayerNorm", "AdaptiveLayerNorm"]
