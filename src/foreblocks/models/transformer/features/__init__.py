"""Optional transformer features such as patching, fusion, mHC, and SyPE."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.models.transformer.features.fusions import (
        fused_dropout_add,
        fused_dropout_add_norm,
        fused_dropout_gateskip_norm,
        get_dropout_p,
    )
    from foreblocks.models.transformer.features.mhc import (
        MHCHyperConnection,
        mhc_apply_norm_streamwise,
        mhc_collapse_streams,
        mhc_init_streams,
        sinkhorn_doubly_stochastic,
    )
    from foreblocks.models.transformer.features.patching import (
        PatchDetokenizer,
        PatchInfo,
        PatchTokenizer,
        patchify_padding_mask,
    )
    from foreblocks.models.transformer.features.residuals import (
        AttentionResidual,
        BlockAttentionResidual,
        normalize_attention_residual_mode,
    )
    from foreblocks.models.transformer.features.sype import (
        AdaptiveWarp,
        SyPERotator,
    )

__all__ = [
    "AdaptiveWarp",
    "AttentionResidual",
    "BlockAttentionResidual",
    "MHCHyperConnection",
    "PatchDetokenizer",
    "PatchInfo",
    "PatchTokenizer",
    "SyPERotator",
    "fused_dropout_add",
    "fused_dropout_add_norm",
    "fused_dropout_gateskip_norm",
    "get_dropout_p",
    "mhc_apply_norm_streamwise",
    "mhc_collapse_streams",
    "mhc_init_streams",
    "normalize_attention_residual_mode",
    "patchify_padding_mask",
    "sinkhorn_doubly_stochastic",
]

_MODULE_BY_NAME = {
    "fused_dropout_add": "fusions",
    "fused_dropout_add_norm": "fusions",
    "fused_dropout_gateskip_norm": "fusions",
    "get_dropout_p": "fusions",
    "MHCHyperConnection": "mhc",
    "mhc_apply_norm_streamwise": "mhc",
    "mhc_collapse_streams": "mhc",
    "mhc_init_streams": "mhc",
    "sinkhorn_doubly_stochastic": "mhc",
    "PatchDetokenizer": "patching",
    "PatchInfo": "patching",
    "PatchTokenizer": "patching",
    "patchify_padding_mask": "patching",
    "AdaptiveWarp": "sype",
    "SyPERotator": "sype",
    "AttentionResidual": "residuals",
    "BlockAttentionResidual": "residuals",
    "normalize_attention_residual_mode": "residuals",
}


def __getattr__(name: str):
    # Lazy re-export (PEP 562): foreblocks.modules.attention.multi_att imports
    # foreblocks.models.transformer.features.sype directly, which is part of a
    # transformer <-> attention import cycle. Eager imports here would deadlock it.
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
