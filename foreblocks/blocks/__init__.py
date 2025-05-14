"""
ForeBlocks custom neural network blocks for time series modeling.

This package contains a collection of state-of-the-art neural network building 
blocks specifically designed for time series forecasting and analysis.
"""

# Simple blocks
from .simple import GRU

# Fourier-based blocks
from .fourier import FourierFeatures, AdaptiveFourierFeatures, FNO1DLayer

# Attention-based blocks
from .attention import HierarchicalAttention, AutoCorrelationBlock, AutoCorrelationPreprocessor

# Graph-based blocks
from .graph import SGConv, AdaptiveGraphConv, GraphConvFactory, GraphConvProcessor, SimpleStemGNNProcessor

# Multiscale processing blocks
from .multiscale import MultiScaleTemporalConv

# ODE-based blocks
from .ode import NeuralODE

# Preprocessing blocks
from .famous import N_BEATS, TimesBlock, TimesBlockPreprocessor

# Mamba blocks
from .mamba import MambaBlock

# NHA
from .nha import NHA

__all__ = [
    # Simple blocks
    'GRU',
    
    'NHA',
    # Fourier-based blocks
    'FourierFeatures', 'AdaptiveFourierFeatures', 'FNO1DLayer',
    
    # Attention-based blocks
    'HierarchicalAttention', 'AutoCorrelationBlock', 'AutoCorrelationPreprocessor',
    
    # Graph-based blocks
    'SGConv', 'AdaptiveGraphConv', 'GraphConvFactory', 'GraphConvProcessor',
    'SimpleStemGNNProcessor',
    
    # Multiscale processing blocks
    'MultiScaleTemporalConv',
    
    # ODE-based blocks
    'NeuralODE',
    
    # Preprocessing blocks
    'N_BEATS', 'TimesBlock', 'TimesBlockPreprocessor'
    
    # Mamba blocks
    'MambaBlock',
]