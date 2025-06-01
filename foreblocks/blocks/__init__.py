"""
ForeBlocks custom neural network blocks for time series modeling.

This package contains a collection of state-of-the-art neural network building
blocks specifically designed for time series forecasting and analysis.
"""

# Attention-based blocks
from .attention import (AutoCorrelationBlock, AutoCorrelationPreprocessor,
                        HierarchicalAttention)
# Preprocessing blocks
from .famous import N_BEATS, TimesBlock, TimesBlockPreprocessor
# Fourier-based blocks
from .fourier import AdaptiveFourierFeatures, FNO1DLayer, FourierFeatures
# Graph-based blocks
from .graph import LatentGraphNetwork
# Mamba blocks
from .mamba import MambaBlock, MambaDecoder, MambaEncoder
# Multiscale processing blocks
from .multiscale import MultiScaleTemporalConv
# NHA
from .nha import NHA
# ODE-based blocks
from .ode import NeuralODE
# Simple blocks
from .simple import GRU

__all__ = [
    # Simple blocks
    "GRU",
    "NHA",
    # Fourier-based blocks
    "FourierFeatures",
    "AdaptiveFourierFeatures",
    "FNO1DLayer",
    # Attention-based blocks
    "HierarchicalAttention",
    "AutoCorrelationBlock",
    "AutoCorrelationPreprocessor",
    # Graph-based blocks
    "LatentGraphNetwork",
    # Multiscale processing blocks
    "MultiScaleTemporalConv",
    # ODE-based blocks
    "NeuralODE",
    # Preprocessing blocks
    "N_BEATS",
    "TimesBlock",
    "TimesBlockPreprocessor"
    # Mamba blocks
    "MambaBlock",
    "MambaEncoder",
    "MambaDecoder",
]

