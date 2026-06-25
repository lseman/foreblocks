"""
Operation blocks - re-exports from split modules.

This module re-exports all operation classes from their dedicated
modules for backward compatibility.
"""

from __future__ import annotations

from .advanced_ops import ConvMixerOp, GRNOp, InvertedAttentionOp, PatchEmbedOp
from .conv_ops import CausalConv1d, MultiScaleConvOp, PyramidConvOp, TCNOp, TimeConvOp
from .decomposition_ops import DLinearOp, NBeatsOp, TimesNetOp
from .fixed_ops import FixedOp
from .mlp_ops import IdentityOp, MLPMixerOp, ResidualMLPOp
from .norms import ChannelRMSNorm, RMSNorm
from .spectral_ops import FourierOp, WaveletOp


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
