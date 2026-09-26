"""TransformerDecoderLayer: self-attention, cross-attention, and feed-forward."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from foreblocks.nn.attention.multihead import MultiAttention
from foreblocks.nn.attention.variants.registry import ATTENTION_VARIANTS
from foreblocks.nn.transformer.attention_backends import LazyAttentionBackendMixin
from foreblocks.nn.transformer.config import TransformerConfig
from foreblocks.nn.transformer.layers.base import BaseTransformerLayer
from foreblocks.nn.transformer.runtime.execution import (
    MHCExecutionMixin,
    ResidualBlockMixin,
)
from foreblocks.nn.transformer.runtime.forward import validate_memory_padding_mask
from foreblocks.nn.transformer.runtime.residual_state import AttentionResidualState
from foreblocks.nn.transformer.runtime.state import (
    AttentionCacheState,
    DecoderLayerState,
)


class TransformerDecoderLayer(
    ResidualBlockMixin,
    MHCExecutionMixin,
    LazyAttentionBackendMixin,
    BaseTransformerLayer,
):
    def __init__(self, config: TransformerConfig | None = None, **overrides: Any):
        super().__init__(config, **overrides)
        config = self.config
        self.layer_attention_type = config.attention
        self.is_causal = not config.informer

        # Recurrent/linear backends are self-attention only, so cross-attention
        # falls back to standard softmax attention for them.
        cross_name = (
            config.attention
            if config.attention in ATTENTION_VARIANTS.names()
            else "standard"
        )
        self.cross_attn = MultiAttention(
            replace(config, attention=cross_name).attention_config(cross=True)
        )

        self.self_attn_norm = self._norm_wrapper()
        self.cross_attn_norm = self._norm_wrapper()
        self.ff_norm = self._norm_wrapper()
        self.gate_self = self._gate()
        self.gate_cross = self._gate()
        self.gate_ff = self._gate()
        self.mhc_conn_self = self._mhc_connection()
        self.mhc_conn_cross = self._mhc_connection()
        self.mhc_conn_ff = self._mhc_connection()
        self.self_input_residual = self._input_residual()
        self.cross_input_residual = self._input_residual()
        self.ff_input_residual = self._input_residual()
        self.materialize_attention_type()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        *,
        layer_state: DecoderLayerState | None = None,
        prev_layer_state: DecoderLayerState | None = None,
        gate_budget: float | None = None,
        streams: torch.Tensor | None = None,
        mtp_targets: torch.Tensor | None = None,
        attention_residual_state: AttentionResidualState | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecoderLayerState, torch.Tensor | None]:
        """Return ``(output, layer_state, mhc_streams)``."""
        self._reset_aux_loss()
        if self.config.residual == "mhc" and layer_state is not None:
            raise RuntimeError(
                "residual='mhc' does not support KV-cached decoding; decode full "
                "sequences instead"
            )
        validate_memory_padding_mask(memory, memory_key_padding_mask)
        if active_mask is not None:
            active_mask = active_mask.to(dtype=torch.bool)
        cfg = self._residual_run_cfg(gate_budget)
        aux_l2_terms: list[torch.Tensor] = []

        if layer_state is not None:
            layer_state = DecoderLayerState.from_mapping(layer_state)
            state = {
                "self_attn": layer_state.self_attention,
                "cross_attn": layer_state.cross_attention,
            }
        else:
            state = {"self_attn": None, "cross_attn": None}

        self_attn_mod = self._self_attn()
        # A single cached step attends to every cached key; no mask is needed.
        self_attn_mask = (
            None if layer_state is not None and tgt.size(1) == 1 else tgt_mask
        )
        strategy = self._make_exec_strategy(
            tgt, streams=streams, attention_residual_state=attention_residual_state
        )

        def self_core(x_in: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
            out, _, updated = self_attn_mod(
                x_in,
                x_in,
                x_in,
                self_attn_mask,
                tgt_key_padding_mask,
                is_causal=self.is_causal,
                layer_state=state["self_attn"],
            )
            return out, updated

        def cross_core(x_in: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
            out, _, updated = self.cross_attn(
                x_in,
                memory,
                memory,
                memory_mask,
                memory_key_padding_mask,
                layer_state=state["cross_attn"],
            )
            return out, updated

        def ff_core(x_in: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
            return self._ff_forward_with_aux(
                x_in,
                mtp_targets=mtp_targets,
                padding_mask=tgt_key_padding_mask,
            ), None

        updated_self, _ = strategy.run_block(
            normw=self.self_attn_norm,
            gate=self.gate_self,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=self_core,
            hyper_conn=self.mhc_conn_self,
            prev_layer_state=prev_layer_state,
            kv_key="self_attn",
            active_mask=active_mask,
            residual_module=self.self_input_residual,
        )
        if updated_self is not None:
            state["self_attn"] = updated_self

        updated_cross, _ = strategy.run_block(
            normw=self.cross_attn_norm,
            gate=self.gate_cross,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=cross_core,
            hyper_conn=self.mhc_conn_cross,
            prev_layer_state=prev_layer_state,
            kv_key="cross_attn",
            active_mask=active_mask,
            residual_module=self.cross_input_residual,
        )
        if updated_cross is not None:
            state["cross_attn"] = updated_cross

        strategy.run_block(
            normw=self.ff_norm,
            gate=self.gate_ff,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=ff_core,
            hyper_conn=self.mhc_conn_ff,
            active_mask=active_mask,
            residual_module=self.ff_input_residual,
        )

        self._finalize_gateskip_aux(cfg, aux_l2_terms)
        output, out_streams = strategy.collapse(self.config.mhc_collapse)
        next_state = DecoderLayerState(
            self_attn=AttentionCacheState.from_mapping(state["self_attn"]),
            cross_attn=AttentionCacheState.from_mapping(state["cross_attn"]),
        )
        return output, next_state, out_streams


__all__ = ["TransformerDecoderLayer"]
