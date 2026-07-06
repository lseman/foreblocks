"""foreblocks.models.kan.kan.

KAN model definitions, RevIN normalization, and patching utilities.

Includes the core KAN forecasting model — a patch-based architecture with
optional polynomial, Jacobi, and wavelet basis components. Provides RevIN
reversible instance normalization and patch-number computation.

Core API:
- KANModel: core KAN forecasting model with RevIN and patch-based architecture
- RevIN: reversible instance normalization for time series
- compute_patch_num: compute number of patches given context window, patch length, and stride

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
from foreblocks.models.kan.poly import PolyFamily, PolyLayerConfig
from foreblocks.models.kan.router import RouterConfig, TokenRouter

__all__ = [
    "BackBone",
    "Backbone",
    "Flatten_Head",
    "FlattenHead",
    "HeteroMoKANLayer",
    "KANModel",
    "Model",
    "PolyFamily",
    "PolyKAN",
    "PolyKANBlock",
    "PolyLayerConfig",
    "RevIN",
    "RouterConfig",
    "TokenRouter",
    "compute_patch_num",
]
