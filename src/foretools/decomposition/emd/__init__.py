"""Signal decomposition package: EMD variants and Variational Mode Decomposition (VMD).

This package provides two families of adaptive signal decomposition methods:

**Empirical Mode Decomposition (EMD) family**
- ``EMDVariants.emd`` — Classic Hilbert-Huang EMD with envelope sift
- ``EMDVariants.ceemdan`` — Complete EMD with adaptive noise
- ``EMDVariants.iceemdan`` — Improved CEEMDAN with better mode separation

**Variational Mode Decomposition (VMD) family**
- ``VMDCore.decompose`` — Core VMD solver with ADMM optimisation
- ``VMDCore.decompose_vncmd`` / ``decompose_chirp`` — Non-stationary IF tracking
- ``VMDCore.decompose_multivariate`` — Joint multi-channel MVMD
- ``VMDOptimizer.optimize`` — Full pipeline with Optuna hyperparameter search
- ``HierarchicalVMD.decompose`` — Multi-scale hierarchical decomposition

**Analysis utilities**
- ``SignalAnalyzer.assess_complexity`` — Auto-parameter selection from signal properties
- ``ModeProcessor.cost_signal`` — Composite quality metric for mode sets
- ``FractalDimension.*`` — Fractal/complexity dimension estimators

**Configuration**
- ``VMDParameters`` — Dataclass for all VMD hyperparameters
- ``VMDOptions`` — Bundled options for core decomposition calls
- ``HierarchicalParameters`` — Settings for hierarchical VMD

Quick start
-----------
>>> import numpy as np
>>> from foretools.decomposition.emd import EMDVariants, VariationalVariants
>>> sig = np.sin(2*np.pi*5*np.linspace(0, 1, 500)) + 0.5*np.sin(2*np.pi*12*np.linspace(0, 1, 500))
>>> imfs = EMDVariants.emd(sig)
>>> vmd = VariationalVariants()
>>> u, uh, omega = vmd.vmd(sig, alpha=2000, K=3)  # VMD decomposition
"""

from .emd import EMDVariants
from .variants import VariationalVariants
from .config import HierarchicalParameters, VMDOptions, VMDParameters
from .core import (
    CrossModeRefiner,
    InformerRefiner,
    VMDCore,
    refine_modes_cross_nn,
    refine_modes_nn,
)
from .pipeline import FastVMD, HierarchicalVMD, VMDOptimizer
from .analysis.signal_analysis import SignalAnalyzer
from .analysis.mode_processor import ModeProcessor
from .analysis.fractal import (
    FractalDimension,
    box_counting_dimension,
    fractal_dimension,
)
from .support.boundary import BoundaryHandler
from .support.fft import FFTWManager, TORCH_AVAILABLE, torch
from .support.utils import _energy


__all__ = [
    # EMD family
    "EMDVariants",
    # VMD family
    "VariationalVariants",
    "VMDCore",
    "VMDOptimizer",
    "HierarchicalVMD",
    "FastVMD",
    "refine_modes_nn",
    "refine_modes_cross_nn",
    "InformerRefiner",
    "CrossModeRefiner",
    # Configuration
    "VMDParameters",
    "VMDOptions",
    "HierarchicalParameters",
    # Analysis utilities
    "SignalAnalyzer",
    "ModeProcessor",
    "FractalDimension",
    "fractal_dimension",
    "box_counting_dimension",
    # Support
    "FFTWManager",
    "BoundaryHandler",
    "TORCH_AVAILABLE",
    "torch",
    "_energy",
]
