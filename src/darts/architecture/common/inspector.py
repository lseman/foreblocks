"""
Shared architecture inspector for extracting trained-architecture choices.

Both the multi-fidelity pipeline (phase-5 printing) and the architecture
derivation logic need to read the same model attributes (attention types,
FFN modes, patching modes, decoder styles, etc.) to produce a human-readable
summary of what the search converged to.

This module provides :class:`ArchitectureInspector` so that both call sites
use a single implementation.

Usage::

    inspector = ArchitectureInspector(model)
    print(inspector.summary())
    encoder_attn = inspector.extract_self_attention_type("encoder")
    decoder_style = inspector.extract_decoder_style()
    ...
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_first_layer_attribute(module: nn.Module, attr: str) -> Any:
    """Get *attr* from the first transformer layer (dict-based or attribute-based)."""
    layers = getattr(module, "layers", None)
    if not layers:
        return None
    first = layers[0]
    if isinstance(first, dict) or hasattr(first, "get"):
        return first.get(attr)
    if hasattr(first, "__contains__") and attr in first:
        return first[attr]
    return getattr(first, attr, None)


def _softmax_top(logits: torch.Tensor, names: tuple[str, ...]) -> str | None:
    """Return the argmax mode name from softmax(logits).  Returns None if logits is invalid."""
    if not isinstance(logits, torch.Tensor) or logits.numel() != len(names):
        return None
    probs = F.softmax(logits.detach(), dim=0)
    idx = int(torch.argmax(probs).item())
    if 0 <= idx < len(names):
        return str(names[idx])
    return None


def _mean_softmax_top(
    components: list[Any],
    *,
    direct_attr: str,
    logits_attr: str,
    mode_names: tuple[str, ...],
) -> str | None:
    """
    Average softmax over *components*, return top mode name.

    Used for per-layer choices where we want the average decision across
    all transformer layers.
    """
    probs_list: list[torch.Tensor] = []
    for comp in components:
        direct = getattr(comp, direct_attr, None)
        if isinstance(direct, str) and direct in mode_names and direct != "auto":
            ref = next(comp.parameters(), None)
            p = ref.new_zeros(len(mode_names)) if ref is not None else torch.zeros(len(mode_names))
            p[mode_names.index(direct)] = 1.0
            probs_list.append(p)
            continue
        logits = getattr(comp, logits_attr, None)
        if isinstance(logits, torch.Tensor) and logits.numel() == len(mode_names):
            probs_list.append(F.softmax(logits.detach(), dim=0))
    if not probs_list:
        return None
    mean_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    idx = int(torch.argmax(mean_probs).item())
    if 0 <= idx < len(mode_names):
        return str(mode_names[idx])
    return None


def _resolve_direct_or_logits(
    obj: Any,
    *,
    direct_attr: str,
    logits_attr: str,
    names: tuple[str, ...],
    fallback: str = "unknown",
) -> str:
    """Try direct attribute first, then softmax on logits."""
    direct = getattr(obj, direct_attr, None)
    if isinstance(direct, str) and direct and direct != "auto":
        return direct
    logits = getattr(obj, logits_attr, None)
    top = _softmax_top(logits, names)
    return top if top is not None else fallback


class ArchitectureInspector:
    """
    Extract architecture choices from a searched/finalized model.

    Wraps all the *_choice and *_extract logic that was previously duplicated
    across ``multi_fidelity.py`` (phase-5 printing) and ``finalization.py``.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    # ── Helpers ───────────────────────────────────────────────────────────

    def _forecast_component(self, role: str) -> nn.Module | None:
        return getattr(self.model, f"forecast_{role}", None)

    def _transformer_or_rnn(self, component: nn.Module | None) -> nn.Module | None:
        if component is None:
            return None
        transformer = getattr(component, "transformer", None)
        if transformer is not None:
            return transformer
        rnn = getattr(component, "rnn", None)
        return rnn

    def _encoder_layer_components(self, transformer: nn.Module | None, attr: str) -> list[Any]:
        if transformer is None:
            return []
        layers = getattr(transformer, "layers", None)
        if not layers:
            return []
        out = []
        for layer in layers:
            if isinstance(layer, dict) or hasattr(layer, "get"):
                val = layer.get(attr)
            elif hasattr(layer, "__contains__") and attr in layer:
                val = layer[attr]
            else:
                val = None
            if val is not None:
                out.append(val)
        return out

    # ── Attention type ────────────────────────────────────────────────────

    def extract_self_attention_type(self, role: str) -> str:
        """Extract the dominant self-attention type for encoder or decoder."""
        comp = self._forecast_component(role)
        if comp is None:
            return "not used"

        # Check if transformer is active in this role
        enc_choice = self._resolve_enc_dec_type(comp).lower()
        if role == "encoder":
            if "transformer" not in enc_choice and "patch" not in enc_choice:
                return "not applicable"
        else:
            if "transformer" not in enc_choice:
                return "not applicable"

        sub = self._transformer_or_rnn(comp)
        if sub is None:
            return "unknown"

        direct = getattr(sub, "self_attention_type", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct

        components = self._encoder_layer_components(sub, "self_attn")
        if components:
            modes = getattr(components[0], "MODES", ("sdp", "linear", "probsparse", "cosine", "local"))
            top = _mean_softmax_top(
                components,
                direct_attr="attention_type",
                logits_attr="attn_alphas",
                mode_names=modes,
            )
            if top:
                return top

        return str(getattr(self.model, "transformer_self_attention_type", "unknown"))

    # ── Attention position ────────────────────────────────────────────────

    def extract_attention_position(self, role: str) -> str:
        """Extract the dominant attention position encoding mode."""
        comp = self._forecast_component(role)
        if comp is None:
            return "not used"
        sub = self._transformer_or_rnn(comp)
        if sub is None:
            return "unknown"
        layers = getattr(sub, "layers", None)
        if not layers:
            return "unknown"
        first = layers[0]
        attn = _get_first_layer_attribute(sub, "self_attn")
        if attn is None:
            return "unknown"
        return _resolve_direct_or_logits(
            attn,
            direct_attr="position_mode",
            logits_attr="position_alphas",
            names=getattr(attn, "POSITION_MODES", ("rope", "alibi", "none", "seasonal")),
            fallback="unknown",
        )

    # ── Tokenizer / patching mode ─────────────────────────────────────────

    def extract_tokenizer_mode(self, role: str = "encoder") -> str:
        """Extract the encoder tokenizer / patching mode."""
        comp = self._forecast_component(role)
        if comp is None:
            return "not used"

        direct = getattr(comp, "patching_mode", None)
        if isinstance(direct, str) and direct:
            if direct != "auto":
                return direct

        logits = getattr(comp, "patch_alpha_logits", None)
        mode_names = getattr(comp, "patch_mode_names", ("direct", "patch"))
        top = _softmax_top(logits, mode_names)
        if top:
            return top

        resolver = getattr(comp, "resolve_patch_mode", None)
        if callable(resolver):
            try:
                return str(resolver())
            except Exception:
                pass
        return "unknown"

    # ── FFN mode ──────────────────────────────────────────────────────────

    def extract_ffn_mode(self, role: str) -> str:
        """Extract the dominant FFN mode for encoder or decoder."""
        comp = self._forecast_component(role)
        if comp is None:
            return "not used"
        sub = self._transformer_or_rnn(comp)
        if sub is None:
            return "unknown"
        layers = getattr(sub, "layers", None)
        if not layers:
            return "unknown"
        components = self._encoder_layer_components(sub, "ffn")
        if not components:
            return "unknown"
        return _mean_softmax_top(
            components,
            direct_attr="ffn_mode",
            logits_attr="ffn_alphas",
            mode_names=getattr(components[0], "MODE_NAMES", ("swiglu", "moe")),
        ) or "unknown"

    # ── Cross-attention ───────────────────────────────────────────────────

    def extract_cross_attention_type(self) -> str:
        comp = self._forecast_component("decoder")
        if comp is None:
            return "not used"

        direct = getattr(comp, "cross_attention_type", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct

        sub = self._transformer_or_rnn(comp)
        if sub is None:
            return "unknown"

        components = self._encoder_layer_components(sub, "cross_attn")
        if not components:
            return "unknown"
        modes = getattr(components[0], "MODES", ("none", "sdp", "linear", "probsparse", "cosine", "local"))
        return _mean_softmax_top(
            components,
            direct_attr="attention_type",
            logits_attr="attn_alphas",
            mode_names=modes,
        ) or "unknown"

    def extract_cross_attention_position(self) -> str:
        comp = self._forecast_component("decoder")
        if comp is None:
            return "not used"
        sub = self._transformer_or_rnn(comp)
        if sub is None:
            return "unknown"
        components = self._encoder_layer_components(sub, "cross_attn")
        if not components:
            return "unknown"
        return _resolve_direct_or_logits(
            components[0],
            direct_attr="position_mode",
            logits_attr="position_alphas",
            names=("rope", "alibi", "none", "seasonal"),
            fallback="unknown",
        )

    # ── Decoder style ─────────────────────────────────────────────────────

    def extract_decoder_style(self) -> str:
        comp = self._forecast_component("decoder")
        if comp is None:
            return "not used"
        direct = getattr(comp, "decode_style", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct
        logits = getattr(comp, "decode_style_alphas", None)
        style_names = getattr(comp, "decode_style_names", ("autoregressive", "informer"))
        top = _softmax_top(logits, style_names)
        if top:
            return top
        resolver = getattr(comp, "resolve_decode_style", None)
        if callable(resolver):
            try:
                return str(resolver())
            except Exception:
                pass
        return "unknown"

    # ── Decoder query mode ────────────────────────────────────────────────

    def extract_decoder_query_mode(self) -> str:
        direct = getattr(self.model, "decoder_query_mode", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct
        logits = getattr(self.model, "decoder_query_alphas", None)
        names = getattr(
            self.model,
            "decoder_query_mode_names",
            ("repeat_last", "zeros", "learned_horizon_queries", "shifted_target", "future_covariate_queries"),
        )
        top = _softmax_top(logits, names)
        if top:
            return top
        resolver = getattr(self.model, "resolve_decoder_query_mode", None)
        if callable(resolver):
            try:
                return str(resolver())
            except Exception:
                pass
        return "unknown"

    # ── Encoder/Decoder type ──────────────────────────────────────────────

    def _resolve_enc_dec_type(self, component: nn.Module) -> str:
        """Resolve the encoder/decoder type from a forecast component."""
        rnn = getattr(component, "rnn", None)
        if rnn is not None:
            return type(rnn).__name__
        transformer = getattr(component, "transformer", None)
        if transformer is not None:
            return type(transformer).__name__
        return type(component).__name__

    def extract_enc_dec_type(self, role: str) -> str:
        comp = self._forecast_component(role)
        if comp is None:
            return "not used"
        return self._resolve_enc_dec_type(comp)

    # ── Normalization ─────────────────────────────────────────────────────

    def extract_normalization(self) -> str:
        alpha = getattr(self.model, "norm_alpha", None)
        if isinstance(alpha, torch.Tensor) and alpha.numel() >= 3:
            names = ["revin", "instance_norm", "identity"]
            idx = int(torch.argmax(alpha.detach()).item())
            return names[idx] if 0 <= idx < len(names) else f"norm_{idx}"
        return "unknown"

    def selected_norm(self) -> str | None:
        return getattr(self.model, "selected_norm", None)

    # ── Cell operations ───────────────────────────────────────────────────

    def extract_cell_ops(self) -> list[str]:
        out: list[str] = []
        for ci, cell in enumerate(getattr(self.model, "cells", [])):
            edge_ops: list[str] = []
            for edge in getattr(cell, "edges", []):
                op_name = None
                fixed_op = getattr(edge, "op", None)
                if fixed_op is not None:
                    op_name = type(fixed_op).__name__.replace("Op", "")
                if op_name is None:
                    edge_ops.append("mixed")
                else:
                    edge_ops.append(op_name)
            if edge_ops:
                counts: dict[str, int] = {}
                for n in edge_ops:
                    counts[n] = counts.get(n, 0) + 1
                counts_txt = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                out.append(f"cell_{ci}: {counts_txt}")
        return out

    # ── Decomposition ─────────────────────────────────────────────────────

    def extract_decomposition(self, role: str) -> str:
        comp = self._forecast_component(role)
        decomp = getattr(comp, "searchable_decomp", None) if comp else None
        if decomp is None:
            return f"{role}: disabled"
        logits = getattr(decomp, "alpha_logits", None)
        if logits is None or logits.numel() == 0:
            return f"{role}: enabled (weights unavailable)"
        with torch.no_grad():
            weights = F.softmax(logits.detach(), dim=0)
            top_idx = int(torch.argmax(weights).item())
            top_weight = float(weights[top_idx].item())
        mode_names = ["none", "moving_avg_trend", "seasonal_residual", "learnable_filter"]
        mode = mode_names[top_idx] if top_idx < len(mode_names) else f"mode_{top_idx}"
        enabled = mode != "none"
        return f"{role}: {'enabled' if enabled else 'disabled'} ({mode}, weight: {top_weight:.3f})"

    # ── Transformer summary ───────────────────────────────────────────────

    def transformer_summary(self) -> str:
        """Build a one-line transformer summary for encoder + decoder."""
        enc = self.extract_enc_dec_type("encoder").lower()
        dec = self.extract_enc_dec_type("decoder").lower()
        uses_transformer = ("transformer" in enc) or ("transformer" in dec)
        if not uses_transformer:
            return "not active"
        attn_type = self.extract_self_attention_type("encoder")
        if attn_type in {"not used", "not applicable", "unknown"}:
            attn_type = self.extract_self_attention_type("decoder")
        ffn_variant = self.extract_ffn_mode("encoder")
        if ffn_variant in {"not used", "unknown"}:
            ffn_variant = self.extract_ffn_mode("decoder")
        tokenizer_mode = self.extract_tokenizer_mode("encoder")
        return f"attn:{attn_type} ffn:{ffn_variant} enc_tok:{tokenizer_mode}"

    # ── Full summary ──────────────────────────────────────────────────────

    def summary(self) -> dict[str, str]:
        """Return a flat dict of all extracted choices.

        Example::

            {
                "arch_mode": "encoder_decoder",
                "attention_encoder": "sdp",
                "attention_decoder": "sdp",
                "attention_position_encoder": "rope",
                "attention_position_decoder": "rope",
                "ffn_encoder": "swiglu",
                "ffn_decoder": "swiglu",
                "tokenizer": "patch",
                "decoder_style": "autoregressive",
                "decoder_query": "repeat_last",
                "cross_attention": "none",
                "cross_position": "rope",
                "normalization": "revin",
                "cells": ["cell_0: TimeConv:1, Fourier:1, Identity:2"],
            }
        """
        result: dict[str, str] = {}
        result["arch_mode"] = str(getattr(self.model, "arch_mode", "unknown"))
        result["hidden_dim"] = str(getattr(self.model, "hidden_dim", "unknown"))
        result["cells"] = str(getattr(self.model, "num_cells", "unknown"))
        result["nodes"] = str(getattr(self.model, "num_nodes", "unknown"))

        enc_sa = self.extract_self_attention_type("encoder")
        dec_sa = self.extract_self_attention_type("decoder")
        if enc_sa not in {"not used", "not applicable"}:
            result["attention_encoder"] = enc_sa
            result["attention_position_encoder"] = self.extract_attention_position("encoder")
            result["tokenizer"] = self.extract_tokenizer_mode("encoder")
            result["ffn_encoder"] = self.extract_ffn_mode("encoder")
        if dec_sa not in {"not used", "not applicable"}:
            result["attention_decoder"] = dec_sa
            result["attention_position_decoder"] = self.extract_attention_position("decoder")
            result["ffn_decoder"] = self.extract_ffn_mode("decoder")

        result["encoder_type"] = self.extract_enc_dec_type("encoder")
        result["decoder_type"] = self.extract_enc_dec_type("decoder")
        result["cross_attention"] = self.extract_cross_attention_type()
        result["cross_position"] = self.extract_cross_attention_position()
        result["decoder_style"] = self.extract_decoder_style()
        result["decoder_query"] = self.extract_decoder_query_mode()

        norm = self.selected_norm() or self.extract_normalization()
        result["normalization"] = norm

        result["decomposition_encoder"] = self.extract_decomposition("encoder")
        result["decomposition_decoder"] = self.extract_decomposition("decoder")

        cells = self.extract_cell_ops()
        if cells:
            result["cells"] = cells

        return result
