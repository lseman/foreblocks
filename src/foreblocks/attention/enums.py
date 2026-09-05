"""Closed attention configuration choices.

Backend and algorithm names deliberately remain strings because their registries
are extensible. These enums cover only choices owned by Foreblocks itself.
"""

from enum import StrEnum


class PositionEncoding(StrEnum):
    NONE = "none"
    ROPE = "rope"
    ALIBI = "alibi"
    SINUSOIDAL = "sinusoidal"


class RopeScaling(StrEnum):
    NONE = "none"
    LINEAR = "linear"
    NTK = "ntk"
    YARN = "yarn"


class QKNorm(StrEnum):
    RMS = "rms"
    L2 = "l2"


class GatedAttentionMode(StrEnum):
    PER_HEAD = "per_head"
    SHARED = "shared"


class AttentionOutputNorm(StrEnum):
    RMS = "rms"
    LAYER = "layer"


class SubqueryNorm(StrEnum):
    LEARNED = "learned"
    RMS = "rms"


__all__ = [
    "AttentionOutputNorm",
    "GatedAttentionMode",
    "PositionEncoding",
    "QKNorm",
    "RopeScaling",
    "SubqueryNorm",
]
