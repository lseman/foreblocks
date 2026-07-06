"""Searchable mixed blocks and fixed deployment wrappers."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_attention import LearnedPoolingBridge
from .bb_sequence import (
    ArchitectureNormalizer,
    BaseFixedSequenceBlock,
    SearchableDecomposition,
    SequenceStateAdapter,
)
from .bb_transformers import (
    LightweightTransformerDecoder,
    LightweightTransformerEncoder,
)
from .helpers import _collect_layer_components, _mean_component_mode_probs


__all__ = [
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
]


def _resolve_searchable_self_attention_type(module_obj, fallback: str = "sdp") -> str:
    if module_obj is None:
        return fallback

    direct = getattr(module_obj, "self_attention_type", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    components = _collect_layer_components(module_obj, "self_attn")
    if components:
        modes = getattr(
            components[0], "MODES", ("sdp", "linear", "probsparse", "cosine", "local")
        )
        probs = _mean_component_mode_probs(
            components,
            direct_attr="attention_type",
            logits_attr="attn_alphas",
            mode_names=modes,
        )
        if probs is not None:
            idx = int(torch.argmax(probs).item())
            if 0 <= idx < len(modes):
                return str(modes[idx])

    return direct if isinstance(direct, str) and direct else fallback


def _freeze_transformer_self_attention(module_obj, attention_type: str) -> None:
    if module_obj is None:
        return

    resolved = str(attention_type).lower()
    if hasattr(module_obj, "self_attention_type"):
        module_obj.self_attention_type = resolved

    layers = getattr(module_obj, "layers", None)
    if not layers:
        return

    for layer in layers:
        self_attn = None
        if isinstance(layer, dict):
            self_attn = layer.get("self_attn")
        elif hasattr(layer, "get"):
            self_attn = layer.get("self_attn")
        elif hasattr(layer, "__contains__") and "self_attn" in layer:
            self_attn = layer["self_attn"]
        if self_attn is None:
            continue
        self_attn.attention_type = resolved
        self_attn.searchable = False
        if hasattr(self_attn, "attn_alphas"):
            self_attn._parameters.pop("attn_alphas", None)
            try:
                delattr(self_attn, "attn_alphas")
            except AttributeError:
                pass


def _resolve_searchable_self_attention_position(
    module_obj, fallback: str = "rope"
) -> str:
    if module_obj is None:
        return fallback
    components = _collect_layer_components(module_obj, "self_attn")
    if not components:
        return fallback
    modes = getattr(
        components[0], "POSITION_MODES",
        ("rope", "alibi", "none", "seasonal", "sinusoidal", "learned", "relative"),
    )
    probs = _mean_component_mode_probs(
        components,
        direct_attr="position_mode",
        logits_attr="position_alphas",
        mode_names=modes,
    )
    if probs is not None:
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(modes):
            return str(modes[idx])
    return fallback


def _freeze_transformer_self_attention_position(module_obj, position_mode: str) -> None:
    if module_obj is None:
        return
    valid_modes = ("rope", "alibi", "none", "seasonal", "sinusoidal", "learned", "relative")
    resolved = str(position_mode).lower()
    if resolved not in valid_modes:
        resolved = "rope"
    if hasattr(module_obj, "self_attention_position_mode"):
        module_obj.self_attention_position_mode = resolved
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return
    for layer in layers:
        self_attn = None
        if isinstance(layer, dict):
            self_attn = layer.get("self_attn")
        elif hasattr(layer, "get"):
            self_attn = layer.get("self_attn")
        elif hasattr(layer, "__contains__") and "self_attn" in layer:
            self_attn = layer["self_attn"]
        if self_attn is None:
            continue
        freezer = getattr(self_attn, "freeze_position_mode", None)
        if callable(freezer):
            freezer(resolved)
        else:
            self_attn.position_mode = resolved


def _resolve_searchable_cross_attention_type(module_obj, fallback: str = "sdp") -> str:
    if module_obj is None:
        return fallback

    direct = getattr(module_obj, "cross_attention_type", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    components = _collect_layer_components(module_obj, "cross_attn")
    if components:
        modes = getattr(
            components[0],
            "MODES",
            ("none", "sdp", "linear", "probsparse", "cosine", "local"),
        )
        probs = _mean_component_mode_probs(
            components,
            direct_attr="attention_type",
            logits_attr="attn_alphas",
            mode_names=modes,
        )
        if probs is not None:
            idx = int(torch.argmax(probs).item())
            if 0 <= idx < len(modes):
                return str(modes[idx])

    return direct if isinstance(direct, str) and direct else fallback


def _freeze_transformer_cross_attention(module_obj, attention_type: str) -> None:
    if module_obj is None:
        return

    resolved = str(attention_type).lower()
    if hasattr(module_obj, "cross_attention_type"):
        module_obj.cross_attention_type = resolved

    layers = getattr(module_obj, "layers", None)
    if not layers:
        return

    for layer in layers:
        cross_attn = None
        if isinstance(layer, dict):
            cross_attn = layer.get("cross_attn")
        elif hasattr(layer, "get"):
            cross_attn = layer.get("cross_attn")
        elif hasattr(layer, "__contains__") and "cross_attn" in layer:
            cross_attn = layer["cross_attn"]
        if cross_attn is None:
            continue
        cross_attn.attention_type = resolved
        cross_attn.searchable = False
        if hasattr(cross_attn, "attn_alphas"):
            cross_attn._parameters.pop("attn_alphas", None)
            try:
                delattr(cross_attn, "attn_alphas")
            except AttributeError:
                pass


def _resolve_searchable_cross_attention_position(
    module_obj, fallback: str = "rope"
) -> str:
    if module_obj is None:
        return fallback
    components = _collect_layer_components(module_obj, "cross_attn")
    if not components:
        return fallback
    modes = getattr(
        components[0], "POSITION_MODES",
        ("rope", "alibi", "none", "seasonal", "sinusoidal", "learned", "relative"),
    )
    probs = _mean_component_mode_probs(
        components,
        direct_attr="position_mode",
        logits_attr="position_alphas",
        mode_names=modes,
    )
    if probs is not None:
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(modes):
            return str(modes[idx])
    return fallback


def _freeze_transformer_cross_attention_position(
    module_obj, position_mode: str
) -> None:
    if module_obj is None:
        return
    valid_modes = ("rope", "alibi", "none", "seasonal", "sinusoidal", "learned", "relative")
    resolved = str(position_mode).lower()
    if resolved not in valid_modes:
        resolved = "rope"
    if hasattr(module_obj, "cross_attention_position_mode"):
        module_obj.cross_attention_position_mode = resolved
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return
    for layer in layers:
        cross_attn = None
        if isinstance(layer, dict):
            cross_attn = layer.get("cross_attn")
        elif hasattr(layer, "get"):
            cross_attn = layer.get("cross_attn")
        elif hasattr(layer, "__contains__") and "cross_attn" in layer:
            cross_attn = layer["cross_attn"]
        if cross_attn is None:
            continue
        freezer = getattr(cross_attn, "freeze_position_mode", None)
        if callable(freezer):
            freezer(resolved)
        else:
            cross_attn.position_mode = resolved


def _resolve_searchable_ffn_mode(module_obj, fallback: str = "swiglu") -> str:
    if module_obj is None:
        return fallback
    components = _collect_layer_components(module_obj, "ffn")
    if not components:
        return fallback
    modes = getattr(components[0], "MODE_NAMES", ("swiglu", "moe"))
    probs = _mean_component_mode_probs(
        components,
        direct_attr="ffn_mode",
        logits_attr="ffn_alphas",
        mode_names=modes,
    )
    if probs is not None:
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(modes):
            return str(modes[idx])
    return fallback


def _freeze_transformer_ffn_mode(module_obj, ffn_mode: str) -> None:
    if module_obj is None:
        return
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return
    resolved = "moe" if str(ffn_mode).lower() == "moe" else "swiglu"
    if hasattr(module_obj, "ffn_variant"):
        module_obj.ffn_variant = resolved
    if hasattr(module_obj, "use_moe"):
        module_obj.use_moe = resolved == "moe"
    for layer in layers:
        ffn = None
        if isinstance(layer, dict):
            ffn = layer.get("ffn")
        elif hasattr(layer, "get"):
            ffn = layer.get("ffn")
        elif hasattr(layer, "__contains__") and "ffn" in layer:
            ffn = layer["ffn"]
        if ffn is None:
            continue
        freezer = getattr(ffn, "freeze_ffn_mode", None)
        if callable(freezer):
            freezer(resolved)
        else:
            ffn.ffn_mode = resolved


def _resolve_searchable_patch_mode(module_obj, fallback: str = "direct") -> str:
    if module_obj is None:
        return fallback

    direct = getattr(module_obj, "patching_mode", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    logits = getattr(module_obj, "patch_alpha_logits", None)
    mode_names = getattr(module_obj, "patch_mode_names", ("direct", "patch"))
    if isinstance(logits, torch.Tensor) and logits.numel() == len(mode_names):
        probs = F.softmax(logits.detach(), dim=0)
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(mode_names):
            return str(mode_names[idx])

    resolver = getattr(module_obj, "resolve_patch_mode", None)
    if callable(resolver):
        try:
            return str(resolver())
        except Exception:
            pass

    return direct if isinstance(direct, str) and direct else fallback


def _freeze_transformer_patch_mode(module_obj, patch_mode: str) -> None:
    if module_obj is None:
        return

    resolved = str(patch_mode).lower()
    if resolved == "patch":
        resolved = "patch_16"
    valid_modes = tuple(
        getattr(
            module_obj,
            "patch_mode_names",
            (
                "direct",
                "patch_8",
                "patch_16",
                "patch_32",
                "multi_scale_patch",
                "variate_tokens",
            ),
        )
    )
    if resolved not in valid_modes:
        resolved = "direct"
    if hasattr(module_obj, "freeze_patch_mode"):
        try:
            module_obj.freeze_patch_mode(resolved)
            return
        except Exception:
            pass

    if hasattr(module_obj, "patching_mode"):
        module_obj.patching_mode = resolved
    if hasattr(module_obj, "patch_alpha_logits"):
        module_obj._parameters.pop("patch_alpha_logits", None)
        try:
            delattr(module_obj, "patch_alpha_logits")
        except AttributeError:
            pass


DECODE_STYLE_NAMES = ("autoregressive", "informer", "autoformer")


def _resolve_searchable_decoder_style(
    module_obj, fallback: str = "autoregressive"
) -> str:
    if module_obj is None:
        return fallback

    direct = getattr(module_obj, "decode_style", None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct

    logits = getattr(module_obj, "decode_style_alphas", None)
    style_names = getattr(
        module_obj, "decode_style_names", DECODE_STYLE_NAMES
    )
    if isinstance(logits, torch.Tensor) and logits.numel() == len(style_names):
        probs = F.softmax(logits.detach(), dim=0)
        idx = int(torch.argmax(probs).item())
        if 0 <= idx < len(style_names):
            return str(style_names[idx])

    resolver = getattr(module_obj, "resolve_decode_style", None)
    if callable(resolver):
        try:
            return str(resolver())
        except Exception:
            pass

    return direct if isinstance(direct, str) and direct else fallback


def _freeze_transformer_decoder_style(module_obj, decode_style: str) -> None:
    if module_obj is None:
        return

    valid_styles = DECODE_STYLE_NAMES
    resolved = str(decode_style).lower()
    if resolved not in valid_styles:
        resolved = "autoregressive"
    if hasattr(module_obj, "freeze_decode_style"):
        try:
            module_obj.freeze_decode_style(resolved)
            return
        except Exception:
            pass

    if hasattr(module_obj, "decode_style"):
        module_obj.decode_style = resolved
    if hasattr(module_obj, "decode_style_alphas"):
        module_obj._parameters.pop("decode_style_alphas", None)
        try:
            delattr(module_obj, "decode_style_alphas")
        except AttributeError:
            pass


