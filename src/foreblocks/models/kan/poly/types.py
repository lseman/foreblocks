"""foreblocks.models.kan.poly.types.

Type definitions and configuration for polynomial basis functions.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

PolyFamily = Literal[
    "hahn",
    "chebyshev",
    "jacobi",
    "legendre",
    "gegenbauer",
    "laguerre",
    "fourier",
    "wavelet",
    "hermite",
]


POLY_FAMILIES: tuple[PolyFamily, ...] = (
    "hahn",
    "chebyshev",
    "jacobi",
    "legendre",
    "gegenbauer",
    "laguerre",
    "fourier",
    "wavelet",
    "hermite",
)

DEFAULT_POLY_FAMILIES: tuple[PolyFamily, ...] = (
    "hahn",
    "chebyshev",
    "jacobi",
    "legendre",
    "gegenbauer",
)


@dataclass(frozen=True)
class PolyLayerConfig:
    degree: int = 3
    hahn_alpha: float = 1.0
    hahn_beta: float = 1.0
    hahn_N: int = 7
    jacobi_alpha: float = 0.0
    jacobi_beta: float = 0.0
    gegen_alpha: float = 1.0
    laguerre_alpha: float = 0.0
    wavelet_num: int = 8
    wavelet_base_freq: float = 1.0
    wavelet_learn_freq: bool = True
    wavelet_learn_scale: bool = True
    wavelet_learn_shift: bool = True
    fourier_base_freq: float = 1.0
    fourier_learn_freq: bool = False

    def with_updates(self, **kwargs) -> PolyLayerConfig:
        return replace(self, **kwargs)
