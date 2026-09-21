"""EMD-like decomposition package API."""

from __future__ import annotations

from .analysis.fractal import (
    FractalDimension,
    box_counting_dimension,
    fractal_dimension,
)
from .analysis.mode_processor import ModeProcessor
from .analysis.signal_analysis import SignalAnalyzer
from .support.boundary import BoundaryHandler
from .support.fft import TORCH_AVAILABLE, FFTWManager, torch
from .support.utils import _energy


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
