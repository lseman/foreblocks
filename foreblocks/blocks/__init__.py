"""
ForeBlocks custom neural network blocks for time series modeling.

This package contains a collection of state-of-the-art neural network building
blocks specifically designed for time series forecasting and analysis.
"""

from ..core.att import AttentionLayer

# Attention-based blocks
from .attention import (
    AutoCorrelationBlock,
    AutoCorrelationPreprocessor,
    HierarchicalAttention,
)
from .enc_dec import (
    GRUDecoder,
    GRUEncoder,
    LSTMDecoder,
    LSTMEncoder,
)

# Preprocessing blocks
# from .famous import N_BEATS, TimesBlock
# Fourier-based blocks
from .fourier import FNO1dLayer, FourierFeatures

# Graph-based blocks
from .graph import GraphPreprocessorNTF

# ODE-based blocks
from .ode import NeuralODE

# Multiscale processing blocks
# from .multiscale import MultiScaleTemporalConv
# NHA
# from .popular.nha import NHA
# Simple blocks
from .simple import GRN

# Mamba blocks
# from .mamba import MambaDecoder


__all__ = [
    "AttentionLayer",
    "LSTMEncoder",
    "LSTMDecoder",
    "GRUEncoder",
    "GRUDecoder",
    # Simple blocks
    "GRN",
    # "NHA",
    # Fourier-based blocks
    "FourierFeatures",
    "FNO1dLayer",
    # Attention-based blocks
    "HierarchicalAttention",
    "AutoCorrelationBlock",
    "AutoCorrelationPreprocessor",
    # Graph-based blocks
    "GraphPreprocessorNTF",
    # Multiscale processing blocks
    # "MultiScaleTemporalConv",
    # ODE-based blocks
    "NeuralODE",
    # Preprocessing blocks
    # "N_BEATS",
    # "TimesBlock",
    # "TimesBlockPreprocessor"
    # Mamba blocks
    # "MambaEncoder",
    # "MambaDecoder",
]
