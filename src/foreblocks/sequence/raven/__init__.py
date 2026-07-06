"""foreblocks.sequence.raven.

Raven sequence-model integration with FLA-backed attention and gated sparse attention.

Top-level package for Raven: a sequence mixer using gated sparse attention (GSA)
with chunked or recurrent scan, Mamba2/GLA decay, and multi-slot routing. Provides
the Raven block, Raven model block wrapper, configuration, and re-exports FLA ops
for direct use.

Core API:
- Raven: gated sparse attention sequence mixer
- RavenBlock: transformer-style block wrapping Raven + MLP
- RavenConfig: HuggingFace-compatible model configuration
- chunk_gsa: chunked gated sparse attention scan
- fused_recurrent_gsa: fused recurrent GSA for decoding

"""

from foreblocks.ops.raven import (
    ACT2FN,
    Attention,
    Cache,
    FusedRMSNormGated,
    RavenMLP,
    ReLUFeatureMap,
    RMSNorm,
    RotaryEmbedding,
    SwishFeatureMap,
    T2RFeatureMap,
    chunk_gsa,
    fla_import_path,
    fla_path,
    fused_attnres,
    fused_recurrent_gsa,
    get_layer_cache,
    get_unpad_data,
    has_fla_checkout,
    import_fla_module,
    index_first_axis,
    pad_input,
    prepare_lens_from_mask,
    rms_norm_linear,
    update_layer_cache,
)
from foreblocks.sequence.raven.blocks import Raven, RavenBlock
from foreblocks.sequence.raven.configuration import RavenConfig

__all__ = [
    "ACT2FN",
    "Attention",
    "Cache",
    "FusedRMSNormGated",
    "RMSNorm",
    "Raven",
    "RavenBlock",
    "RavenConfig",
    "RavenMLP",
    "ReLUFeatureMap",
    "RotaryEmbedding",
    "SwishFeatureMap",
    "T2RFeatureMap",
    "chunk_gsa",
    "fla_import_path",
    "fla_path",
    "fused_attnres",
    "fused_recurrent_gsa",
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
