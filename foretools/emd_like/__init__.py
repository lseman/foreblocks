"""EMD-like decomposition package API."""

from .common import BoundaryHandler
from .common import FFTWManager
from .common import FractalDimension
from .common import ModeProcessor
from .common import SignalAnalyzer
from .common import _energy
from .common import box_counting_dimension
from .common import fractal_dimension
from .config import HierarchicalParameters
from .config import VMDParameters
from .core import CrossModeRefiner
from .core import InformerRefiner
from .core import VMDCore
from .core import refine_modes_cross_nn
from .core import refine_modes_nn
from .emd import EMDVariants
from .pipeline import FastVMD
from .pipeline import HierarchicalVMD
from .pipeline import VMDOptimizer
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
