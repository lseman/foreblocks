"""TransformerEncoderLayer: self-attention and feed-forward sublayers."""

from __future__ import annotations

from typing import Any

import torch

from foreblocks.nn.transformer.attention_backends import LazyAttentionBackendMixin
from foreblocks.nn.transformer.config import TransformerConfig
from foreblocks.nn.transformer.layers.base import BaseTransformerLayer
from foreblocks.nn.transformer.runtime.execution import (
    MHCExecutionMixin,
    ResidualBlockMixin,
)
from foreblocks.nn.transformer.runtime.residual_state import AttentionResidualState


class TransformerEncoderLayer(
    ResidualBlockMixin,
    MHCExecutionMixin,
    LazyAttentionBackendMixin,
    BaseTransformerLayer,
):
    def __init__(self, config: TransformerConfig | None = None, **overrides: Any):
        super().__init__(config, **overrides)
        self.layer_attention_type = self.config.attention

        self.attn_norm = self._norm_wrapper()
        self.ff_norm = self._norm_wrapper()
        self.gate_attn = self._gate()
        self.gate_ff = self._gate()
        self.mhc_conn_attn = self._mhc_connection()
        self.mhc_conn_ff = self._mhc_connection()
        self.attn_input_residual = self._input_residual()
        self.ff_input_residual = self._input_residual()
        self.materialize_attention_type()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        *,
        gate_budget: float | None = None,
        streams: torch.Tensor | None = None,
        mtp_targets: torch.Tensor | None = None,
        attention_residual_state: AttentionResidualState | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(output, mhc_streams)``; streams are ``None`` unless mHC."""
        self._reset_aux_loss()
        if active_mask is not None:
            active_mask = active_mask.to(dtype=torch.bool)
        cfg = self._residual_run_cfg(gate_budget)
        aux_l2_terms: list[torch.Tensor] = []
        attn_mod = self._self_attn()
        strategy = self._make_exec_strategy(
            src, streams=streams, attention_residual_state=attention_residual_state
        )

        def attn_core(x_in: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
            out, _, _ = attn_mod(x_in, x_in, x_in, src_mask, src_key_padding_mask)
            return out, None

        def ff_core(x_in: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
            return self._ff_forward_with_aux(
                x_in,
                mtp_targets=mtp_targets,
                padding_mask=src_key_padding_mask,
            ), None

        strategy.run_block(
            normw=self.attn_norm,
            gate=self.gate_attn,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=attn_core,
            hyper_conn=self.mhc_conn_attn,
            active_mask=active_mask,
            residual_module=self.attn_input_residual,
        )
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
        return strategy.collapse(self.config.mhc_collapse)


__all__ = ["TransformerEncoderLayer"]
