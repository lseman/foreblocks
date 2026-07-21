"""foreblocks.models.kan.poly.gegenbauer.

Gegenbauer polynomials basis function.

"""

import torch
from torch import Tensor, nn

from foreblocks.models.kan.poly.utils import (
    _init_coeffs,
    _reshape_in_out,
    _restore_shape,
    _tanh_to_unit,
)


class GegenbauerPolynomials(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int,
        alpha: float = 1.0,
    ):
        super().__init__()
        if alpha <= -0.5:
            raise ValueError("Gegenbauer alpha must be > -0.5")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = float(alpha)
        self.coeffs = _init_coeffs(input_dim, output_dim, degree + 1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        alpha = self.alpha
        terms: list[Tensor] = [torch.ones_like(x2)]
        if self.degree >= 1:
            terms.append(2.0 * alpha * x2)

        for n in range(2, self.degree + 1):
            nf = float(n)
            terms.append(
                (
                    2.0 * (nf + alpha - 1.0) * x2 * terms[-1]
                    - (nf + 2.0 * alpha - 2.0) * terms[-2]
                )
                / nf
            )

        basis = torch.stack(terms, dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y
