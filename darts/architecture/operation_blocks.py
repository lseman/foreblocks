"""
Operation blocks - re-exports from split modules.

This module re-exports all operation classes from their dedicated
modules for backward compatibility.
"""

from __future__ import annotations

from .advanced_ops import ConvMixerOp
from .advanced_ops import GRNOp
from .advanced_ops import InvertedAttentionOp
from .advanced_ops import PatchEmbedOp
from .conv_ops import CausalConv1d
from .conv_ops import MultiScaleConvOp
from .conv_ops import PyramidConvOp
from .conv_ops import TCNOp
from .conv_ops import TimeConvOp
from .decomposition_ops import DLinearOp
from .decomposition_ops import NBeatsOp
from .decomposition_ops import TimesNetOp
from .fixed_ops import FixedOp
from .mlp_ops import IdentityOp
from .mlp_ops import MLPMixerOp
from .mlp_ops import ResidualMLPOp
from .norms import ChannelRMSNorm
from .norms import RMSNorm
from .spectral_ops import FourierOp
from .spectral_ops import WaveletOp

__all__ = [
    "RMSNorm",
    "ChannelRMSNorm",
    "CausalConv1d",
    "IdentityOp",
    "TimeConvOp",
    "ResidualMLPOp",
    "TCNOp",
    "FourierOp",
    "WaveletOp",
    "ConvMixerOp",
    "GRNOp",
    "MultiScaleConvOp",
    "PyramidConvOp",
    "PatchEmbedOp",
    "InvertedAttentionOp",
    "MLPMixerOp",
    "DLinearOp",
    "NBeatsOp",
    "TimesNetOp",
    "FixedOp",
]
