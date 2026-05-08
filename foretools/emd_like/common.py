"""EMD-like decomposition package API."""

from __future__ import annotations

from .boundary import BoundaryHandler
from .fft import FFTWManager, TORCH_AVAILABLE, torch
from .fractal import FractalDimension, box_counting_dimension, fractal_dimension
from .mode_processor import ModeProcessor
from .signal_analysis import SignalAnalyzer
from .utils import _energy

__all__ = [
    "_energy",
    "FractalDimension",
    "fractal_dimension",
    "box_counting_dimension",
    "FFTWManager",
    "TORCH_AVAILABLE",
    "torch",
    "BoundaryHandler",
    "SignalAnalyzer",
    "ModeProcessor",
]
