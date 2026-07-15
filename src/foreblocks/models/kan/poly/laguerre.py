"""foreblocks.models.kan.poly.laguerre.

Laguerre polynomials basis function.

"""

import torch
from torch import Tensor, nn

from foreblocks.models.kan.poly.utils import _init_coeffs, _reshape_in_out, _restore_shape, _tanh_to_positive


class LaguerrePolynomials(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int,
        alpha: float = 0.0,
    ):
        super().__init__()
        if alpha <= -1.0:
            raise ValueError("Laguerre alpha must be > -1.0")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = float(alpha)
        self.coeffs = _init_coeffs(input_dim, output_dim, degree + 1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_positive(x2)

        alpha = self.alpha
        terms: list[Tensor] = [torch.ones_like(x2)]
        if self.degree >= 1:
            terms.append(1.0 + alpha - x2)

        for n in range(2, self.degree + 1):
            nf = float(n)
            terms.append(
                (
                    (2.0 * nf - 1.0 + alpha - x2) * terms[-1]
                    - (nf - 1.0 + alpha) * terms[-2]
                )
                / nf
            )

        basis = torch.stack(terms, dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y
