from __future__ import annotations

from foreblocks.ops.raven.backend import fla_import_path, fla_path, get_fla_attr, has_fla_checkout, import_fla_module


with fla_import_path():
    from fla.layers.attn import Attention
    from fla.layers.utils import (
        get_layer_cache,
        get_unpad_data,
        index_first_axis,
        pad_input,
        update_layer_cache,
    )
    from fla.models.utils import Cache
    from fla.modules import FusedRMSNormGated, RMSNorm, RotaryEmbedding
    from fla.modules import GatedMLP as RavenMLP
    from fla.modules.activations import ACT2FN
    from fla.modules.feature_map import ReLUFeatureMap, SwishFeatureMap, T2RFeatureMap
    from fla.modules.layernorm import rms_norm_linear
    from fla.ops.attnres import fused_attnres
    from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa
    from fla.ops.utils.index import prepare_lens_from_mask


__all__ = [
    "ACT2FN",
    "Attention",
    "Cache",
    "FusedRMSNormGated",
    "RMSNorm",
    "RotaryEmbedding",
    "RavenMLP",
    "ReLUFeatureMap",
    "SwishFeatureMap",
    "T2RFeatureMap",
    "chunk_gsa",
    "fla_import_path",
    "fla_path",
    "fused_attnres",
    "fused_recurrent_gsa",
    "get_fla_attr",
    "get_layer_cache",
    "get_unpad_data",
    "has_fla_checkout",
    "import_fla_module",
    "index_first_axis",
    "pad_input",
    "prepare_lens_from_mask",
    "rms_norm_linear",
    "update_layer_cache",
]
