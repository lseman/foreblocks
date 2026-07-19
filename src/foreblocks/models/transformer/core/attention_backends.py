"""Layer-level self-attention backend registry.

Maps a ``layer_attention_type`` name (as produced by
``BaseTransformer._get_layer_attention_type``) to the module class and
constructor kwargs used to build it. Shared by encoder and decoder layers so
the two never drift on which backends exist (see runtime/execution.py's
``LazyAttentionBackendMixin`` for the lazy-instantiation cache that uses it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from foreblocks.modules.attention.modules.linear_att import ModernLinearAttention
from foreblocks.modules.attention.modules.linear_att.gated_delta import GatedDeltaNet
from foreblocks.modules.attention.modules.linear_att.kimi import KimiAttention
from foreblocks.modules.attention.multi_att import MultiAttention


@dataclass(frozen=True)
class LayerAttentionBackendSpec:
    name: str
    module_cls: type[nn.Module]
    extra_kwargs: Callable[[dict], dict] = field(default=lambda cfg: {})

    def build(self, attn_init_kwargs: dict, pos_encoding_type: str, cfg: dict) -> nn.Module:
        return self.module_cls(
            **attn_init_kwargs,
            **self.extra_kwargs(cfg),
            pos_encoding_type=pos_encoding_type,
        )


def _std_kwargs(cfg: dict) -> dict:
    return dict(
        attention_type=cfg["att_type"],
        attn_implementation=cfg["attn_implementation"],
        freq_modes=cfg["freq_modes"],
        use_mla=not cfg["use_attention_matching_compaction"],
        use_attention_matching_compaction=cfg["use_attention_matching_compaction"],
        attention_matching_keep_ratio=cfg["attention_matching_keep_ratio"],
        attention_matching_trigger_len=cfg["attention_matching_trigger_len"],
        attention_matching_min_keep=cfg["attention_matching_min_keep"],
        attention_matching_query_budget=cfg["attention_matching_query_budget"],
        attention_matching_force_single_step=cfg["attention_matching_force_single_step"],
        moba_block_size=cfg["moba_block_size"],
        moba_topk=cfg["moba_topk"],
        rope_base=cfg["rope_base"],
        rope_scaling_type=cfg["rope_scaling_type"],
        rope_scaling_factor=cfg["rope_scaling_factor"],
    )


def _sype_kwargs(cfg: dict) -> dict:
    return dict(
        attention_type="sype",
        freq_modes=cfg["freq_modes"],
        use_mla=not cfg["use_attention_matching_compaction"],
        use_attention_matching_compaction=cfg["use_attention_matching_compaction"],
        attention_matching_keep_ratio=cfg["attention_matching_keep_ratio"],
        attention_matching_trigger_len=cfg["attention_matching_trigger_len"],
        attention_matching_min_keep=cfg["attention_matching_min_keep"],
        attention_matching_query_budget=cfg["attention_matching_query_budget"],
        attention_matching_force_single_step=cfg["attention_matching_force_single_step"],
        moba_block_size=cfg["moba_block_size"],
        moba_topk=cfg["moba_topk"],
        rope_base=cfg["rope_base"],
        rope_scaling_type=cfg["rope_scaling_type"],
        rope_scaling_factor=cfg["rope_scaling_factor"],
    )


def _linear_kwargs(cfg: dict) -> dict:
    return dict(backend="rda", state="elu")


def _gla_kwargs(cfg: dict) -> dict:
    return dict(backend="gla", mode="chunk", chunk_size=64)


def _deltanet_kwargs(cfg: dict) -> dict:
    return dict(backend="deltanet", mode="chunk", chunk_size=64)


def _gdn_modern_kwargs(cfg: dict) -> dict:
    return dict(chunk_size=64)


LAYER_ATTENTION_BACKENDS: dict[str, LayerAttentionBackendSpec] = {
    "standard": LayerAttentionBackendSpec("standard", MultiAttention, _std_kwargs),
    "sype": LayerAttentionBackendSpec("sype", MultiAttention, _sype_kwargs),
    "linear": LayerAttentionBackendSpec("linear", ModernLinearAttention, _linear_kwargs),
    "gla": LayerAttentionBackendSpec("gla", ModernLinearAttention, _gla_kwargs),
    "deltanet": LayerAttentionBackendSpec("deltanet", ModernLinearAttention, _deltanet_kwargs),
    "gated_deltanet": LayerAttentionBackendSpec("gated_deltanet", GatedDeltaNet, _gdn_modern_kwargs),
    "kimi": LayerAttentionBackendSpec("kimi", KimiAttention),
    "gated_delta": LayerAttentionBackendSpec("gated_delta", GatedDeltaNet),
}


def build_layer_attention_backend(
    name: str, attn_init_kwargs: dict, pos_encoding_type: str, cfg: dict
) -> nn.Module:
    spec = LAYER_ATTENTION_BACKENDS.get(name, LAYER_ATTENTION_BACKENDS["standard"])
    return spec.build(attn_init_kwargs, pos_encoding_type, cfg)


__all__ = [
    "LAYER_ATTENTION_BACKENDS",
    "LayerAttentionBackendSpec",
    "build_layer_attention_backend",
]
