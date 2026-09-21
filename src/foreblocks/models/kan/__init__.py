"""foreblocks.models.kan.

Kolmogorov-Arnold Network (KAN) time series forecasting models.

Provides a patch-based KAN architecture with polynomial basis functions (Hahn,
Chebyshev, Jacobi, Legendre, Gegenbauer, Laguerre, Fourier, Wavelet, Hermite),
MoE expert routing, RevIN normalization, and configurable polynomial families.

Core API:
- KANModel: core KAN forecasting model with RevIN and patch-based architecture
- Backbone: patch-based KAN encoder with MoE expert layers
- PolyFamily: literal type for polynomial family selection
- PolyLayerConfig: configuration for polynomial layer parameters
- TokenRouter: token-level routing across MoE experts
- RouterConfig: router configuration dataclass
- build_poly_layer: factory to build a polynomial layer from a family name
- RevIN: reversible instance normalization for time series

"""

from foreblocks.models.kan.backbone import (
    BackBone,
    Backbone,
    Flatten_Head,
    FlattenHead,
    HeteroMoKANLayer,
    PolyKAN,
    PolyKANBlock,
)
from foreblocks.models.kan.model import KANModel, Model, RevIN, compute_patch_num
from foreblocks.models.kan.poly import (
    DEFAULT_POLY_FAMILIES,
    POLY_FAMILIES,
    ChebyshevPolynomials,
    FourierKAN,
    GegenbauerPolynomials,
    HahnPolynomials,
    JacobiPolynomials,
    LaguerrePolynomials,
    PolyFamily,
    PolyLayerConfig,
    ProbHermitePolynomials,
    WaveletKAN,
    build_poly_layer,
    list_poly_families,
)
from foreblocks.models.kan.router import RouterConfig, TokenRouter

__all__ = [
    "BackBone",
    "Backbone",
    "ChebyshevPolynomials",
    "DEFAULT_POLY_FAMILIES",
    "Flatten_Head",
    "FlattenHead",
    "FourierKAN",
    "GegenbauerPolynomials",
    "HahnPolynomials",
    "HeteroMoKANLayer",
    "JacobiPolynomials",
    "KANModel",
    "LaguerrePolynomials",
    "Model",
    "POLY_FAMILIES",
    "PolyFamily",
    "PolyKAN",
    "PolyKANBlock",
    "PolyLayerConfig",
    "ProbHermitePolynomials",
    "RevIN",
    "RouterConfig",
    "TokenRouter",
    "WaveletKAN",
    "build_poly_layer",
    "compute_patch_num",
    "list_poly_families",
]
