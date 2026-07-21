"""foreblocks.models.kan.poly.

Polynomial basis functions for Kolmogorov-Arnold Network layers.

Implements 9 polynomial families: Hahn, Chebyshev, Jacobi, Legendre,
Gegenbauer, Laguerre, Fourier, Wavelet, and Probabilist Hermite. Each family
is a trainable nn.Module that maps input tensors through a polynomial basis
with learnable coefficients. Includes PolyLayerConfig dataclass and factory.

Core API:
- PolyLayerConfig: frozen dataclass for polynomial layer hyperparameters
- PolyFamily: literal type for selecting polynomial families
- build_poly_layer: factory function to create a polynomial layer from a family name
- list_poly_families: return available polynomial family names

"""

from __future__ import annotations

from foreblocks.models.kan.poly.chebyshev import ChebyshevPolynomials
from foreblocks.models.kan.poly.fourier import FourierKAN
from foreblocks.models.kan.poly.gegenbauer import GegenbauerPolynomials
from foreblocks.models.kan.poly.hahn import HahnPolynomials
from foreblocks.models.kan.poly.hermite import ProbHermitePolynomials
from foreblocks.models.kan.poly.jacobi import JacobiPolynomials
from foreblocks.models.kan.poly.laguerre import LaguerrePolynomials
from foreblocks.models.kan.poly.types import (
    DEFAULT_POLY_FAMILIES,
    POLY_FAMILIES,
    PolyFamily,
    PolyLayerConfig,
)
from foreblocks.models.kan.poly.utils import _resolve_family_name
from foreblocks.models.kan.poly.wavelet import WaveletKAN


def build_poly_layer(
    family: PolyFamily,
    input_dim: int,
    output_dim: int,
    config: PolyLayerConfig | None = None,
) -> nn.Module:
    from torch import nn

    cfg = config or PolyLayerConfig()
    family_name = _resolve_family_name(family)

    if family_name == "hahn":
        return HahnPolynomials(
            input_dim=input_dim,
            output_dim=output_dim,
            degree=cfg.degree,
            alpha=cfg.hahn_alpha,
            beta=cfg.hahn_beta,
            N=cfg.hahn_N,
        )
    if family_name == "chebyshev":
        return ChebyshevPolynomials(input_dim, output_dim, cfg.degree)
    if family_name == "jacobi":
        return JacobiPolynomials(
            input_dim=input_dim,
            output_dim=output_dim,
            degree=cfg.degree,
            alpha=cfg.jacobi_alpha,
            beta=cfg.jacobi_beta,
        )
    if family_name == "legendre":
        return JacobiPolynomials(input_dim, output_dim, cfg.degree, 0.0, 0.0)
    if family_name == "gegenbauer":
        return GegenbauerPolynomials(
            input_dim=input_dim,
            output_dim=output_dim,
            degree=cfg.degree,
            alpha=cfg.gegen_alpha,
        )
    if family_name == "laguerre":
        return LaguerrePolynomials(
            input_dim=input_dim,
            output_dim=output_dim,
            degree=cfg.degree,
            alpha=cfg.laguerre_alpha,
        )
    if family_name == "wavelet":
        num_wavelets = cfg.degree if cfg.degree > 4 else cfg.wavelet_num
        return WaveletKAN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_wavelets=num_wavelets,
            base_freq=cfg.wavelet_base_freq,
            learn_freq=cfg.wavelet_learn_freq,
            learn_scale=cfg.wavelet_learn_scale,
            learn_shift=cfg.wavelet_learn_shift,
        )
    if family_name == "fourier":
        return FourierKAN(
            input_dim=input_dim,
            output_dim=output_dim,
            degree=cfg.degree,
            base_freq=cfg.fourier_base_freq,
            learn_freq=cfg.fourier_learn_freq,
        )
    if family_name == "hermite":
        return ProbHermitePolynomials(input_dim, output_dim, cfg.degree)
    raise AssertionError(f"Unhandled family '{family_name}'")


def list_poly_families() -> tuple[PolyFamily, ...]:
    return POLY_FAMILIES


__all__ = [
    "ChebyshevPolynomials",
    "DEFAULT_POLY_FAMILIES",
    "FourierKAN",
    "GegenbauerPolynomials",
    "HahnPolynomials",
    "JacobiPolynomials",
    "LaguerrePolynomials",
    "POLY_FAMILIES",
    "PolyFamily",
    "PolyLayerConfig",
    "ProbHermitePolynomials",
    "WaveletKAN",
    "build_poly_layer",
    "list_poly_families",
]
