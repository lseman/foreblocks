"""foreblocks.ops.raven.

Package initializer that exposes the public symbols for this namespace.
It belongs to the Raven sequence-model integration helpers area of Foreblocks.
"""

from __future__ import annotations

from typing import Any

from foreblocks.ops.raven.backend import (
    fla_import_path,
    fla_path,
    get_fla_attr,
    has_fla_checkout,
    import_fla_module,
)


class _LazyFLAAttr:
    """Proxy an upstream FLA attribute without importing FLA on package import."""

    def __init__(self, module_name: str, attr_name: str):
        self._module_name = module_name
        self._attr_name = attr_name
        self.__name__ = attr_name

    def _resolve(self) -> Any:
        return get_fla_attr(self._module_name, self._attr_name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __getitem__(self, key: Any) -> Any:
        return self._resolve()[key]

    def __contains__(self, item: Any) -> bool:
        return item in self._resolve()

    def __iter__(self):
        return iter(self._resolve())

    def __repr__(self) -> str:
        return f"<lazy FLA attr {self._module_name}.{self._attr_name}>"


Attention = _LazyFLAAttr("fla.layers.attn", "Attention")
get_layer_cache = _LazyFLAAttr("fla.layers.utils", "get_layer_cache")
get_unpad_data = _LazyFLAAttr("fla.layers.utils", "get_unpad_data")
index_first_axis = _LazyFLAAttr("fla.layers.utils", "index_first_axis")
pad_input = _LazyFLAAttr("fla.layers.utils", "pad_input")
update_layer_cache = _LazyFLAAttr("fla.layers.utils", "update_layer_cache")
Cache = _LazyFLAAttr("fla.models.utils", "Cache")
FusedRMSNormGated = _LazyFLAAttr("fla.modules", "FusedRMSNormGated")
RMSNorm = _LazyFLAAttr("fla.modules", "RMSNorm")
RotaryEmbedding = _LazyFLAAttr("fla.modules", "RotaryEmbedding")
RavenMLP = _LazyFLAAttr("fla.modules", "GatedMLP")
ACT2FN = _LazyFLAAttr("fla.modules.activations", "ACT2FN")
ReLUFeatureMap = _LazyFLAAttr("fla.modules.feature_map", "ReLUFeatureMap")
SwishFeatureMap = _LazyFLAAttr("fla.modules.feature_map", "SwishFeatureMap")
T2RFeatureMap = _LazyFLAAttr("fla.modules.feature_map", "T2RFeatureMap")
rms_norm_linear = _LazyFLAAttr("fla.modules.layernorm", "rms_norm_linear")
fused_attnres = _LazyFLAAttr("fla.ops.attnres", "fused_attnres")
chunk_gsa = _LazyFLAAttr("fla.ops.gsa", "chunk_gsa")
fused_recurrent_gsa = _LazyFLAAttr("fla.ops.gsa", "fused_recurrent_gsa")
prepare_lens_from_mask = _LazyFLAAttr("fla.ops.utils.index", "prepare_lens_from_mask")


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
