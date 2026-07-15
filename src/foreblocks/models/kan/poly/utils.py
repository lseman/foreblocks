"""foreblocks.models.kan.poly.utils.

Shared utility functions for polynomial basis functions.

"""

import torch
from torch import Tensor, nn

from foreblocks.models.kan.poly.types import POLY_FAMILIES


def _resolve_family_name(family: str) -> str:
    family_name = str(family).strip().lower()
    if family_name not in POLY_FAMILIES:
        available = ", ".join(POLY_FAMILIES)
        raise ValueError(f"Unknown family '{family}'. Available: {available}")
    return family_name


def _reshape_in_out(x: Tensor, input_dim: int) -> tuple[Tensor, tuple[int, ...]]:
    if x.shape[-1] != input_dim:
        raise ValueError(f"Expected last dim {input_dim}, got {x.shape[-1]}")
    orig = x.shape
    return x.reshape(-1, input_dim), orig


def _restore_shape(y: Tensor, orig_shape: tuple[int, ...], output_dim: int) -> Tensor:
    return y.reshape(*orig_shape[:-1], output_dim)


def _tanh_to_unit(x: Tensor) -> Tensor:
    return torch.tanh(x)


def _tanh_to_positive(x: Tensor) -> Tensor:
    return 0.5 * (torch.tanh(x) + 1.0)


def _init_coeffs(input_dim: int, output_dim: int, basis_size: int) -> nn.Parameter:
    coeffs = nn.Parameter(torch.empty(input_dim, output_dim, basis_size))
    nn.init.normal_(coeffs, mean=0.0, std=1.0 / (input_dim * basis_size))
    return coeffs
