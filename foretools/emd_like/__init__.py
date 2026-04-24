"""EMD-like decomposition package API."""

from .common import (
    BoundaryHandler,
    FFTWManager,
    FractalDimension,
    ModeProcessor,
    SignalAnalyzer,
    _energy,
    box_counting_dimension,
    fractal_dimension,
)
from .config import HierarchicalParameters, VMDParameters
from .core import (
    CrossModeRefiner,
    InformerRefiner,
    VMDCore,
    refine_modes_cross_nn,
    refine_modes_nn,
)
from .emd import EMDVariants
from .pipeline import FastVMD, HierarchicalVMD, VMDOptimizer
from .variants import VariationalVariants


__all__ = [
    "_energy",
    "FractalDimension",
    "fractal_dimension",
    "box_counting_dimension",
    "VMDParameters",
    "HierarchicalParameters",
    "FFTWManager",
    "BoundaryHandler",
    "SignalAnalyzer",
    "EMDVariants",
    "ModeProcessor",
    "VMDCore",
    "VariationalVariants",
    "CrossModeRefiner",
    "InformerRefiner",
    "refine_modes_nn",
    "refine_modes_cross_nn",
    "VMDOptimizer",
    "HierarchicalVMD",
    "FastVMD",
]
