"""foreblocks.models.kan.poly.fourier.

Fourier KAN basis function.

"""

import math

import torch
from torch import Tensor, nn

from foreblocks.models.kan.poly.utils import _init_coeffs, _reshape_in_out, _restore_shape, _tanh_to_unit


class FourierKAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int = 5,
        base_freq: float = 1.0,
        learn_freq: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.num_terms = 2 * degree + 1
        self.freq = nn.Parameter(
            torch.full((input_dim,), base_freq), requires_grad=learn_freq
        )
        self.coeffs = _init_coeffs(input_dim, output_dim, self.num_terms)

    def _fourier_basis(self, x: Tensor) -> Tensor:
        freq = self.freq.unsqueeze(0).unsqueeze(-1)
        omega_x = 2.0 * math.pi * freq * x.unsqueeze(-1)
        k = torch.arange(0, self.degree + 1, device=x.device, dtype=x.dtype)
        angle = k.unsqueeze(0).unsqueeze(0) * omega_x
        cos_terms = torch.cos(angle)
        sin_terms = torch.sin(angle[..., 1:])
        return torch.cat([cos_terms, sin_terms], dim=-1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)
        basis = self._fourier_basis(x2)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y
