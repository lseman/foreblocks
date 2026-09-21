"""foreblocks.models.kan.poly.hahn.

Hahn polynomials basis function.

"""

import torch
from torch import Tensor, nn

from foreblocks.models.kan.poly.utils import (
    _init_coeffs,
    _reshape_in_out,
    _restore_shape,
    _tanh_to_unit,
)


class HahnPolynomials(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int,
        alpha: float,
        beta: float,
        N: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.a = float(alpha)
        self.b = float(beta)
        self.N = int(N)
        self.coeffs = _init_coeffs(input_dim, output_dim, degree + 1)

    def forward(
        self, x: Tensor, return_basis: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x2, orig = _reshape_in_out(x, self.input_dim)
        x2 = _tanh_to_unit(x2)

        terms: list[Tensor] = []
        T0 = torch.ones(x2.shape[0], self.input_dim, device=x2.device, dtype=x2.dtype)
        terms.append(T0)

        if self.degree >= 1:
            T1 = 1.0 - (
                ((self.a + self.b + 2.0) * x2) / ((self.a + 1.0) * float(self.N))
            )
            terms.append(T1)

        for n in range(2, self.degree + 1):
            m = float(n - 1)
            A = (m + self.a + self.b + 1.0) * (m + self.a + 1.0) * (float(self.N) - m)
            A /= 2.0 * m + self.a + self.b + 1.0
            A /= 2.0 * m + self.a + self.b + 2.0

            C = m * (m + self.a + self.b + float(self.N) + 1.0) * (m + self.b)
            C /= 2.0 * m + self.a + self.b
            C /= 2.0 * m + self.a + self.b + 1.0

            Tn = ((A + C - x2) * terms[-1] - (C * terms[-2])) / A
            terms.append(Tn)

        basis = torch.stack(terms, dim=-1)
        y2 = torch.einsum("nid,iod->no", basis, self.coeffs)
        y = _restore_shape(y2, orig, self.output_dim)
        if return_basis:
            return y, basis
        return y
