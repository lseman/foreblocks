"""Normalization, residual, mHC, and layer-invocation execution strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from foreblocks.layers.norms import create_norm_layer
from foreblocks.models.transformer.features.fusions import (
    fused_dropout_add, fused_dropout_add_norm, fused_dropout_gateskip_norm,
    get_dropout_p,
)
from foreblocks.models.transformer.features.mhc import (
    mhc_apply_norm_streamwise, mhc_collapse_streams,
)
from foreblocks.modules.skip.gateskip import apply_skip_to_kv


class NormWrapper(nn.Module):
    """Own a normalization strategy and dropout without hiding residual flow."""

    def __init__(self, d_model, norm_type="rms", strategy="pre_norm", dropout=0.0, eps=1e-5):
        super().__init__()
        if strategy not in {"pre_norm", "post_norm", "sandwich_norm"}:
            raise ValueError("invalid norm strategy")
        self.norm = create_norm_layer(norm_type, d_model, eps)
        self.strategy = strategy
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, *_, **__):
        raise RuntimeError("NormWrapper is a holder; invoke .norm and .dropout explicitly")


@dataclass(frozen=True)
class ResidualRunCfg:
    use_gateskip: bool
    gate_budget: Optional[float]
    gate_lambda: float
    training: bool


class ResidualBlockMixin:
    @staticmethod
    def _drop_p(normw):
        return get_dropout_p(getattr(normw, "dropout", None))

    def _residual_apply(
        self, x, update, normw, p, gate, cfg, aux_l2_terms,
        updated_kv=None, prev_layer_state=None, kv_key=None, active_mask=None,
    ):
        if cfg.use_gateskip:
            x2, skip_mask = fused_dropout_gateskip_norm(
                residual=x, update=update, gate=gate, use_gateskip=True,
                gate_budget=cfg.gate_budget, aux_l2_terms=aux_l2_terms,
                gate_lambda=cfg.gate_lambda,
                norm_layer=(normw if normw.strategy != "pre_norm" else None),
                p=p, training=cfg.training, active_mask=active_mask,
            )
            if skip_mask is not None and updated_kv is not None and kv_key is not None:
                updated_kv = apply_skip_to_kv(updated_kv, skip_mask, prev_layer_state, kv_key)
            return x2, updated_kv, skip_mask
        if normw.strategy in ("pre_norm", "sandwich_norm"):
            x2 = fused_dropout_add(x, update, p=p, training=cfg.training)
            if normw.strategy == "sandwich_norm":
                x2 = normw.norm(x2)
        else:
            x2 = fused_dropout_add_norm(
                residual=x, update=update, norm_layer=normw,
                p=p, training=cfg.training,
            )
        return x2, updated_kv, None

    def _run_sublayer_nonmhc(
        self, x, normw, core_fn, gate, cfg, aux_l2_terms,
        prev_layer_state=None, kv_key=None, active_mask=None,
    ):
        x_in = normw.norm(x) if normw.strategy in ("pre_norm", "sandwich_norm") else x
        update, updated_kv = core_fn(x_in)
        return self._residual_apply(
            x, update, normw, self._drop_p(normw), gate, cfg, aux_l2_terms,
            updated_kv, prev_layer_state, kv_key, active_mask,
        )


class MHCBlockMixin:
    def _mhc_run_block(self, streams, normw, hyper_conn, core_fn):
        x_in, maps = hyper_conn.pre_aggregate(streams)
        if normw.strategy in ("pre_norm", "sandwich_norm"):
            x_in = normw.norm(x_in)
        streams = hyper_conn.combine(streams, normw.dropout(core_fn(x_in)), maps=maps)
        if normw.strategy in ("post_norm", "sandwich_norm"):
            streams = mhc_apply_norm_streamwise(normw.norm, streams)
        return streams


@dataclass
class LayerExecutionStrategy:
    owner: object
    use_mhc: bool
    x: Optional[torch.Tensor] = None
    streams: Optional[torch.Tensor] = None

    def run_block(
        self, *, normw, gate, cfg, aux_l2_terms, core_fn=None, mhc_core=None,
        hyper_conn=None, prev_layer_state=None, kv_key=None, active_mask=None,
    ):
        if not self.use_mhc:
            if self.x is None or core_fn is None:
                raise RuntimeError("non-mHC execution requires x and core_fn")
            self.x, updated, skipped = self.owner._run_sublayer_nonmhc(
                self.x, normw, core_fn, gate, cfg, aux_l2_terms,
                prev_layer_state, kv_key, active_mask,
            )
            return updated, skipped
        if self.streams is None or mhc_core is None or hyper_conn is None:
            raise RuntimeError("mHC execution requires streams, core, and connection")
        self.streams = self.owner._mhc_run_block(self.streams, normw, hyper_conn, mhc_core)
        return None, None

    def collapse(self, mode):
        if not self.use_mhc:
            if self.x is None:
                raise RuntimeError("execution has no tensor")
            return self.x, None
        if self.streams is None:
            raise RuntimeError("execution has no streams")
        return mhc_collapse_streams(self.streams, mode=mode), self.streams


@dataclass(frozen=True)
class ModelLayerInvokeStrategy:
    owner: object
    use_checkpoint: bool

    def run_encoder_layer(
        self, *, layer, x, src_mask, src_key_padding_mask, budget, streams,
        attention_residual_state, gateskip_active_mask,
    ):
        if self.use_checkpoint:
            def checkpointed(value):
                result, _ = layer(
                    value, src_mask, src_key_padding_mask, gate_budget=budget,
                    gate_lambda=self.owner.gate_lambda,
                    use_gateskip=self.owner.use_gateskip, streams=None,
                    use_mhc=self.owner.use_mhc, mhc_n_streams=self.owner.mhc_n_streams,
                    mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
                    mhc_collapse=self.owner.mhc_collapse,
                    gateskip_active_mask=gateskip_active_mask,
                )
                return result
            return self.owner._run_with_checkpoint(
                checkpointed, x, use_checkpoint=True
            ), streams
        return layer(
            x, src_mask, src_key_padding_mask, gate_budget=budget,
            gate_lambda=self.owner.gate_lambda, use_gateskip=self.owner.use_gateskip,
            streams=streams, use_mhc=self.owner.use_mhc,
            mhc_n_streams=self.owner.mhc_n_streams,
            mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
            mhc_collapse=self.owner.mhc_collapse,
            attention_residual_state=attention_residual_state,
            gateskip_active_mask=gateskip_active_mask,
        )

    def run_decoder_layer(
        self, *, layer, x, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
        memory_key_padding_mask, layer_state, prev_state, budget, streams,
        mtp_targets, attention_residual_state, gateskip_active_mask,
    ):
        args = (
            memory, tgt_mask, memory_mask, tgt_key_padding_mask,
            memory_key_padding_mask,
        )
        common = dict(
            incremental_state=layer_state, prev_layer_state=prev_state,
            gate_budget=budget, gate_lambda=self.owner.gate_lambda,
            use_gateskip=self.owner.use_gateskip, use_mhc=self.owner.use_mhc,
            mhc_n_streams=self.owner.mhc_n_streams,
            mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
            mhc_collapse=self.owner.mhc_collapse, mtp_targets=mtp_targets,
            gateskip_active_mask=gateskip_active_mask,
        )
        if self.use_checkpoint:
            def checkpointed(value):
                result, _, _ = layer(value, *args, streams=None, **common)
                return result
            result = self.owner._run_with_checkpoint(
                checkpointed, x, use_checkpoint=True
            )
            return result, layer_state, streams
        return layer(
            x, *args, streams=streams,
            attention_residual_state=attention_residual_state, **common,
        )


_LayerExecutionStrategy = LayerExecutionStrategy
_ModelLayerInvokeStrategy = ModelLayerInvokeStrategy

__all__ = [
    "LayerExecutionStrategy", "MHCBlockMixin", "ModelLayerInvokeStrategy",
    "NormWrapper", "ResidualBlockMixin", "ResidualRunCfg",
]
