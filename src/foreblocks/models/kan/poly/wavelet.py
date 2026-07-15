"""foreblocks.models.kan.poly.wavelet.

Wavelet KAN basis function.

"""

import torch
from torch import Tensor, nn

from foreblocks.models.kan.poly.utils import _init_coeffs, _reshape_in_out, _restore_shape, _tanh_to_unit


class WaveletKAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_wavelets: int = 8,
        base_freq: float = 1.0,
        learn_freq: bool = True,
        learn_scale: bool = True,
        learn_shift: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_wavelets = num_wavelets

        self.scales = nn.Parameter(
            torch.ones(input_dim, num_wavelets), requires_grad=learn_scale
        )
        self.shifts = nn.Parameter(
            torch.linspace(-1.5, 1.5, num_wavelets).repeat(input_dim, 1),
            requires_grad=learn_shift,
        )
        self.freqs = nn.Parameter(
            torch.ones(input_dim, num_wavelets) * base_freq, requires_grad=learn_freq
        )
        self.coeffs = _init_coeffs(input_dim, output_dim, num_wavelets + 1)

    def _morlet(self, x: Tensor) -> Tensor:
        diff = x.unsqueeze(-1) - self.shifts
        gauss = torch.exp(-0.5 * (diff / (self.scales + 1e-6)) ** 2)
        oscil = torch.cos(self.freqs * diff)
        return gauss * oscil

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        basis = self._morlet(x2)
        basis = torch.cat([basis, torch.ones_like(basis[..., :1])], dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y
