"""
ForeBlocks DARTS: Neural Architecture Search for Time Series Forecasting
"""

# Neural Building Blocks
# Core DARTS Components
from .darts import (  # Temporal Operations; Attention Mechanisms; Spectral Analysis; Advanced Neural Blocks
    DARTSCell, TimeSeriesDARTS)

# Zero-Cost Metrics and Search Functions

__version__ = "1.0.0"
__author__ = "ForeBlocks Team"

__all__ = [
    # Core Components
    "TimeSeriesDARTS",
    "DARTSCell",
]
