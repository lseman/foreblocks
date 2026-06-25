"""
ModernLinearAttention — modular linear attention with swappable backends.

Backends
--------
1. "rda"      : RDA (Riemannian Distance Attention) — original ELU+1 kernel.
                  Configurable feature_map: "elu", "relu", "silu", "leaky_relu".
                  O(L·d²) global, supports incremental recurrent decode.

2. "gla"      : Gated Linear Attention (GLA, Yang et al. 2023).
                  Per-timestep decay gate gk = log-sigmoid(low-rank) / τ.
                  Parallel chunk mode + exact recurrent mode.
                  O(L·d²) global, supports incremental decode.

3. "deltanet" : DeltaNet (Yang et al. 2024).
                  L2-normalised Q, K with causal conv pre-processing and
                  learnable β gate. Parallel WY chunk + exact recurrent.
                  O(L·d²) global, supports incremental decode.

4. "gated_delta" / "gated_deltanet"
                : Gated DeltaNet / Mamba-2 implementation exposed through the
                  wrapper without duplicating ``gated_delta.GatedDeltaNet``.

All backends implement the same drop-in API:
    (query, key, value, attn_mask, key_padding_mask, is_causal, layer_state)
    → (out, None, updated_state)

layer_state dict carries recurrent state under key "<backend>_state":
    { "rda_state": {"k_sum": ..., "kv_sum": ...} }
    { "gla_state": {"S": ...} }
    { "deltanet_state": {"S": ...} }
    { "gdn_state": ... }
"""

from __future__ import annotations

from foreblocks.modules.attention.modules.linear_att.base import (
    FeatureMapRegistry,
    RoPEMixin,
)
from foreblocks.modules.attention.modules.linear_att.deltanet import DeltaNetBackend
from foreblocks.modules.attention.modules.linear_att.gated_delta import GatedDeltaNet
from foreblocks.modules.attention.modules.linear_att.gated_deltanet2 import (
    GatedDeltaNet2,
)
from foreblocks.modules.attention.modules.linear_att.gla import GLABackend
from foreblocks.modules.attention.modules.linear_att.kimi import KimiAttention
from foreblocks.modules.attention.modules.linear_att.rda import RDABackend
from foreblocks.modules.attention.modules.linear_att.wrapper import (
    GatedDeltaBackend,
    KimiBackend,
    ModernLinearAttention,
)


__all__ = [
    "FeatureMapRegistry",
    "RoPEMixin",
    "RDABackend",
    "GLABackend",
    "DeltaNetBackend",
    "GatedDeltaBackend",
    "GatedDeltaNet",
    "GatedDeltaNet2",
    "KimiBackend",
    "KimiAttention",
    "ModernLinearAttention",
]
