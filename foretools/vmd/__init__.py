# -*- coding: utf-8 -*-
"""VMD package API."""

from .common import (
    BoundaryHandler,
    EMDVariants,
    FFTWManager,
    FractalDimension,
    HierarchicalParameters,
    ModeProcessor,
    SignalAnalyzer,
    VMDParameters,
    _energy,
    box_counting_dimension,
)
from .core import (
    CrossModeRefiner,
    InformerRefiner,
    VMDCore,
    refine_modes_cross_nn,
    refine_modes_nn,
)
from .pipeline import FastVMD, HierarchicalVMD, VMDOptimizer

__all__ = [
    "_energy",
    "FractalDimension",
    "box_counting_dimension",
    "VMDParameters",
    "HierarchicalParameters",
    "FFTWManager",
    "BoundaryHandler",
    "SignalAnalyzer",
    "EMDVariants",
    "ModeProcessor",
    "VMDCore",
    "CrossModeRefiner",
    "InformerRefiner",
    "refine_modes_nn",
    "refine_modes_cross_nn",
    "VMDOptimizer",
    "HierarchicalVMD",
    "FastVMD",
]
