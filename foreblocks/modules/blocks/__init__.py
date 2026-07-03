"""foreblocks.modules.blocks.

Neural network building blocks for time-series modeling.

Aggregates attention, recurrent, spectral, graph, ODE, and normalization blocks
into a single importable package. Use this submodule to access the full library
of forecasting-ready layers without importing from individual implementation files.

Core API (re-exports):
- AttentionLayer: multi-head attention wrapper
- LSTMEncoder / LSTMDecoder: LSTM encoder-decoder pair
- GRUEncoder / GRUDecoder: GRU encoder-decoder pair
- AutoCorrelationBlock: auto-correlation based attention block
- HierarchicalAttention: hierarchical multi-scale attention
- GRN: Gated Residual Network building block
- FourierFeatures / FNO1dLayer: Fourier feature and spectral convolution blocks
- NeuralODE: ODE-based continuous-time layer

"""

from foreblocks.core.att import AttentionLayer

# Attention-based blocks
from foreblocks.modules.blocks.attention import (
    AutoCorrelationBlock,
    AutoCorrelationPreprocessor,
    HierarchicalAttention,
)
from foreblocks.modules.blocks.enc_dec import (
    GRUDecoder,
    GRUEncoder,
    LSTMDecoder,
    LSTMEncoder,
)

# Preprocessing blocks
# from .famous import N_BEATS, TimesBlock
# Fourier-based blocks
from foreblocks.modules.blocks.fourier import FNO1dLayer, FourierFeatures

# ODE-based blocks
from foreblocks.modules.blocks.ode import NeuralODE

# Multiscale processing blocks
# from .multiscale import MultiScaleTemporalConv
# NHA
# from .popular.nha import NHA
# Simple blocks
from foreblocks.modules.blocks.simple import GRN

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
