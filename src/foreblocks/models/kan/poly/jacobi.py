"""foreblocks.models.kan.poly.jacobi.

Jacobi polynomials basis function.

"""

import torch
from torch import Tensor, nn

from foreblocks.models.kan.poly.utils import (
    _init_coeffs,
    _reshape_in_out,
    _restore_shape,
    _tanh_to_unit,
)


class JacobiPolynomials(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int,
        alpha: float = 0.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.coeffs = _init_coeffs(input_dim, output_dim, degree + 1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        a = self.alpha
        b = self.beta

        terms: list[Tensor] = [torch.ones_like(x2)]
        if self.degree >= 1:
            terms.append(0.5 * ((a - b) + (a + b + 2.0) * x2))

        for n in range(2, self.degree + 1):
            nf = float(n)
            A = 2.0 * nf * (nf + a + b) * (2.0 * nf + a + b - 2.0)
            Bc = (
                (2.0 * nf + a + b - 1.0) * (2.0 * nf + a + b) * (2.0 * nf + a + b - 2.0)
            )
            C = (2.0 * nf + a + b - 1.0) * (a * a - b * b)
            D = 2.0 * (nf + a - 1.0) * (nf + b - 1.0) * (2.0 * nf + a + b)
            terms.append(((Bc * x2 + C) * terms[-1] - D * terms[-2]) / A)

        basis = torch.stack(terms, dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y
