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


__all__ = [
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
]


def _layer_component(layer, key: str):
    if isinstance(layer, dict):
        return layer.get(key)
    if hasattr(layer, "get"):
        try:
            return layer.get(key)
        except Exception:
            pass
    if hasattr(layer, "__contains__") and key in layer:
        return layer[key]
    return None


def _collect_layer_components(module_obj, key: str) -> list[nn.Module]:
    layers = getattr(module_obj, "layers", None)
    if not layers:
        return []
    out: list[nn.Module] = []
    for layer in layers:
        component = _layer_component(layer, key)
        if component is not None:
            out.append(component)
    return out


def _mean_component_mode_probs(
    components: list[nn.Module],
    *,
    direct_attr: str,
    logits_attr: str,
    mode_names,
) -> torch.Tensor | None:
    probs = []
    mode_names = tuple(mode_names)
    for component in components:
        direct = getattr(component, direct_attr, None)
        if isinstance(direct, str) and direct in mode_names and direct != "auto":
            ref = next(component.parameters(), None)
            p = (
                ref.new_zeros(len(mode_names))
                if ref is not None
                else torch.zeros(len(mode_names))
            )
            p[mode_names.index(direct)] = 1.0
            probs.append(p)
            continue

        logits = getattr(component, logits_attr, None)
        if isinstance(logits, torch.Tensor) and logits.numel() == len(mode_names):
            probs.append(F.softmax(logits.detach(), dim=0))

    if not probs:
        return None
    return torch.stack(probs, dim=0).mean(dim=0)


