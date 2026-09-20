"""
Central registry for DARTS search-space operations.

Single source of truth for which time-series operators exist, which
family/group each belongs to, how to construct them, and their static
efficiency prior (used before, or in the absence of, FLOPs profiling).

Before this module existed, the same information was duplicated across four
independent literals that had already drifted out of sync:
``MixedOp.op_map``, ``MixedOp.operation_groups`` and
``MixedOp.op_efficiency``/``_apply_static_efficiency_priors`` in
``mixed_op.py``, plus ``DEFAULT_OPS``/``DEFAULT_OP_FAMILIES`` in
``config.py``. Notably, ``SwiGLU``/``GeGLU``/``GatedGELU`` were reachable
through ``MixedOp`` but absent from ``config.py``'s search-space defaults
and from the efficiency table. Anything that needs "the list of ops" should
derive it from ``OP_REGISTRY``/``FAMILY_TO_OPS`` here instead of keeping a
parallel literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .advanced import (
    ConvMixerOp,
    GatedGeLUFFNOp,
    GeGLUFFNOp,
    GRNOp,
    InvertedAttentionOp,
    PatchEmbedOp,
    SwiGLUFFNOp,
)
from .conv import MultiScaleConvOp, PyramidConvOp, TCNOp, TimeConvOp
from .decomposition import DLinearOp, NBeatsOp, TimesNetOp
from .mlp import IdentityOp, MLPMixerOp, ResidualMLPOp
from .spectral import FourierOp, WaveletOp
from .ssm import SSMOp

__all__ = ["OpSpec", "OP_REGISTRY", "FAMILY_TO_OPS", "DEFAULT_OP_NAMES", "build_op"]

# ctor(input_dim, latent_dim, seq_length) -> nn.Module
OpCtor = Callable[[int, int, int], nn.Module]


@dataclass(frozen=True)
class OpSpec:
    name: str
    family: str
    ctor: OpCtor
    efficiency: float  # static prior in [0, 1], higher = cheaper
    description: str = ""


def _ctor(cls, *, needs_seq_length: bool = False, **kwargs) -> OpCtor:
    if needs_seq_length:
        return lambda input_dim, latent_dim, seq_length: cls(
            input_dim, latent_dim, seq_length, **kwargs
        )
    return lambda input_dim, latent_dim, seq_length: cls(
        input_dim, latent_dim, **kwargs
    )


# Ordered to match the historical per-family grouping in MixedOp exactly
# (mlp / conv / frequency / attention / gated_ffn), with the new "ssm" family
# appended.
_SPECS: list[OpSpec] = [
    # mlp
    OpSpec("Identity", "mlp", _ctor(IdentityOp), 1.0, "No-op passthrough/projection."),
    OpSpec("ResidualMLP", "mlp", _ctor(ResidualMLPOp), 0.80, "Residual MLP block."),
    OpSpec("GRN", "mlp", _ctor(GRNOp), 0.65, "Gated residual network."),
    OpSpec("TimeMixer", "mlp", _ctor(MLPMixerOp, needs_seq_length=True), 0.68, "MLP-Mixer over time and channel axes."),
    OpSpec("NBeats", "mlp", _ctor(NBeatsOp), 0.78, "N-BEATS basis-expansion block."),
    # conv
    OpSpec("TimeConv", "conv", _ctor(TimeConvOp), 0.60, "Depthwise-separable temporal convolution."),
    OpSpec("TCN", "conv", _ctor(TCNOp), 0.50, "Dilated temporal convolutional network."),
    OpSpec("ConvMixer", "conv", _ctor(ConvMixerOp), 0.58, "ConvMixer-style depthwise-separable conv."),
    OpSpec("MultiScaleConv", "conv", _ctor(MultiScaleConvOp), 0.35, "Multi-scale dilated convolutions."),
    OpSpec("PyramidConv", "conv", _ctor(PyramidConvOp), 0.25, "Pyramidal multi-resolution convolutions."),
    # frequency
    OpSpec("Fourier", "frequency", _ctor(FourierOp, needs_seq_length=True), 0.45, "Learnable low/high-pass Fourier mixing."),
    OpSpec("Wavelet", "frequency", _ctor(WaveletOp), 0.45, "Multi-scale dilated wavelet-style convolution."),
    OpSpec("DLinear", "frequency", _ctor(DLinearOp), 0.95, "Trend/seasonal decomposition with linear heads."),
    OpSpec("TimesNet", "frequency", _ctor(TimesNetOp), 0.62, "TimesNet 2-D temporal-variation block."),
    # attention
    OpSpec("PatchEmbed", "attention", _ctor(PatchEmbedOp, patch_size=16), 0.68, "PatchTST-style patch tokenization."),
    OpSpec("InvertedAttention", "attention", _ctor(InvertedAttentionOp), 0.55, "iTransformer-style variate-dimension attention."),
    # gated_ffn
    OpSpec("SwiGLU", "gated_ffn", _ctor(SwiGLUFFNOp), 0.75, "SwiGLU-gated feed-forward block."),
    OpSpec("GeGLU", "gated_ffn", _ctor(GeGLUFFNOp), 0.75, "GeGLU-gated feed-forward block."),
    OpSpec("GatedGELU", "gated_ffn", _ctor(GatedGeLUFFNOp), 0.75, "GELU-gated feed-forward block."),
    # ssm — previously a placeholder family with no members (see
    # search/candidate_config.py's special-cased "ssm" handling).
    OpSpec("SSM", "ssm", _ctor(SSMOp, needs_seq_length=True), 0.55, "Lightweight S4D-style diagonal state-space block."),
]

OP_REGISTRY: dict[str, OpSpec] = {spec.name: spec for spec in _SPECS}

FAMILY_TO_OPS: dict[str, list[str]] = {}
for _spec in _SPECS:
    FAMILY_TO_OPS.setdefault(_spec.family, []).append(_spec.name)

# Preserves registration order, matching the historical DEFAULT_OPS ordering
# (Identity + the pre-existing 15 ops), with SwiGLU/GeGLU/GatedGELU/SSM
# appended rather than silently missing.
DEFAULT_OP_NAMES: list[str] = list(OP_REGISTRY.keys())


def build_op(name: str, input_dim: int, latent_dim: int, seq_length: int) -> nn.Module:
    """Instantiate operation *name* for the given dimensions."""
    return OP_REGISTRY[name].ctor(input_dim, latent_dim, seq_length)
