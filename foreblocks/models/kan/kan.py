"""foreblocks.models.kan.kan.

This module implements the kan pieces for its package.
It belongs to the Kolmogorov-Arnold Network model components area of Foreblocks.
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
