"""
ForeBlocks DARTS: Neural Architecture Search for Time Series Forecasting
"""

# Neural Building Blocks
# Core DARTS Components
from .darts import (  # Temporal Operations; Attention Mechanisms; Spectral Analysis; Advanced Neural Blocks
    AttentionOp,
    ConvMixerOp,
    DARTSCell,
    FixedOp,
    FourierOp,
    GRNOp,
    IdentityOp,
    MixedOp,
    ResidualMLPOp,
    TCNOp,
    TimeConvOp,
    TimeSeriesDARTS,
    TransformerOp,
    WaveletOp,
)

# Zero-Cost Metrics and Search Functions
from .darts_run import *

__version__ = "1.0.0"
__author__ = "ForeBlocks Team"

__all__ = [
    # Core Components
    "TimeSeriesDARTS",
    "DARTSCell",
    "MixedOp",
    "FixedOp",
    # Neural Operations
    "TimeConvOp",
    "TCNOp",
    "ConvMixerOp",
    "AttentionOp",
    "TransformerOp",
    "WaveletOp",
    "FourierOp",
    "GRNOp",
    "ResidualMLPOp",
    "IdentityOp",
]
