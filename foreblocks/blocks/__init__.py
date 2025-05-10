"""
ForeBlocks custom neural network blocks for time series modeling.

This package contains a collection of state-of-the-art neural network building 
blocks specifically designed for time series forecasting and analysis.
"""

# Simple blocks
from .simple import GRU

# Fourier-based blocks
from .fourier import FourierFeatures, AdaptiveFourierFeatures

# Attention-based blocks
from .attention import HierarchicalAttention

# Graph-based blocks
from .graph import SGConv

# Multiscale processing blocks
from .multiscale import MultiScaleTemporalConv

# ODE-based blocks
from .ode import NeuralODE

# Preprocessing blocks
from .famous import N_BEATS

# Mamba blocks
from .mamba import MambaBlock

__all__ = [
    # Simple blocks
    'GRU',
    
    # Fourier-based blocks
    'FourierFeatures', 'AdaptiveFourierFeatures',
    
    # Attention-based blocks
    'HierarchicalAttention',
    
    # Graph-based blocks
    'SGConv',
    
    # Multiscale processing blocks
    'MultiScaleTemporalConv',
    
    # ODE-based blocks
    'NeuralODE',
    
    # Preprocessing blocks
    'N_BEATS',
    
    # Mamba blocks
    'MambaBlock',
]