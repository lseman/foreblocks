"""
Attention blocks - re-exports from split modules.

This module re-exports all attention-related classes and functions
from their dedicated modules for backward compatibility.
"""

from __future__ import annotations

from .bridges import AttentionBridge
from .bridges import LearnedPoolingBridge
from .self_attention import SelfAttention
from .utils import _causal_mask
from .utils import _make_alibi_slopes
from .utils import _seasonal_relative_bias
from .utils import _sinusoidal_features

__all__ = [
    "SelfAttention",
    "AttentionBridge",
    "LearnedPoolingBridge",
    "_make_alibi_slopes",
    "_causal_mask",
    "_sinusoidal_features",
    "_seasonal_relative_bias",
]
