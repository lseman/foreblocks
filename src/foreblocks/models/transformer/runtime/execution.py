"""Normalization, residual, mHC, and layer-invocation execution strategies."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn as nn

from foreblocks.layers.norms import create_norm_layer
from foreblocks.models.transformer.core.attention_backends import (
    build_layer_attention_backend,
)
from foreblocks.models.transformer.features.fusions import (
    fused_dropout_add, fused_dropout_add_norm, fused_dropout_gateskip_norm,
    get_dropout_p,
)
from foreblocks.models.transformer.features.mhc import (
    mhc_apply_norm_streamwise, mhc_collapse_streams,
)
from foreblocks.models.transformer.runtime.residual_state import (
    _append_attention_residual_update, _attention_residual_input,
)
from foreblocks.modules.skip.gateskip import apply_skip_to_kv


class NormWrapper(nn.Module):
    """Holder for a normalization layer, strategy, and dropout.

    Must be an nn.Module (not a plain dataclass) so that .norm/.dropout are
    registered submodules — otherwise .to()/.cuda()/.parameters()/
    state_dict() silently skip them (they previously did, when this was a
    frozen dataclass: the weights never moved off CPU and were absent from
    the optimizer's parameter list). forward() still raises: callers must
    invoke .norm(...)/.dropout(...) explicitly, never the wrapper itself.
    """

    def __init__(self, norm: nn.Module, strategy: str, dropout: nn.Module) -> None:
        super().__init__()
        if strategy not in {"pre_norm", "post_norm", "sandwich_norm"}:
            raise ValueError("invalid norm strategy")
        self.norm = norm
        self.strategy = strategy
        self.dropout = dropout

    @staticmethod
    def make(
        d_model: int,
        norm_type: str = "rms",
        strategy: str = "pre_norm",
        dropout: float = 0.0,
        eps: float = 1e-5,
    ) -> "NormWrapper":
        norm = create_norm_layer(norm_type, d_model, eps)
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        return NormWrapper(norm=norm, strategy=strategy, dropout=dropout_layer)

    def forward(self, *_args, **_kwargs):
        raise RuntimeError(
            "NormWrapper is a holder, not callable — use .norm(...) or .dropout(...)"
        )


@dataclass(frozen=True)
class ResidualRunCfg:
    use_gateskip: bool
    gate_budget: float | None
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


class LazyAttentionBackendMixin:
    """Lazily instantiate + cache the self-attention module for a layer.

    Shared by encoder and decoder layers so both route through the same
    ``LAYER_ATTENTION_BACKENDS`` registry (see core/attention_backends.py) and
    never drift on which backends are supported.
    """

    def _ensure_attn_backend(self, name: str) -> nn.Module:
        cache = self.__dict__.setdefault("_attn_backends", {})
        module = cache.get(name)
        if module is None:
            dev = next(self.parameters()).device
            module = build_layer_attention_backend(
                name, self._attn_init_kwargs, self._pos_encoding_type,
                self._attn_backend_cfg,
            ).to(dev)
            cache[name] = module
            self.add_module(f"_attn_backend_{name}", module)
        return module

    def _self_attn(self) -> nn.Module:
        return self._ensure_attn_backend(self.layer_attention_type)

    def set_layer_attention_type(self, layer_attention_type: str) -> None:
        self.layer_attention_type = str(layer_attention_type)

    def materialize_attention_type(
        self, layer_attention_type: str | None = None
    ) -> nn.Module:
        previous = self.layer_attention_type
        if layer_attention_type is not None:
            self.layer_attention_type = str(layer_attention_type)
        try:
            return self._self_attn()
        finally:
            self.layer_attention_type = previous


class ResidualExecutionPolicy(Protocol):
    """Execution contract shared by standard and stream residual policies."""

    def run_block(
        self, strategy: "LayerExecutionStrategy", context: "ResidualBlockContext"
    ) -> tuple[Any | None, torch.Tensor | None]: ...


CoreFn = Callable[[torch.Tensor], tuple[torch.Tensor, Any | None]]
MhcCoreFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ResidualBlockContext:
    """Typed inputs shared by the residual execution policies."""

    normw: NormWrapper
    gate: nn.Module | None
    cfg: ResidualRunCfg
    aux_l2_terms: list[torch.Tensor]
    core_fn: CoreFn | None = None
    mhc_core: MhcCoreFn | None = None
    hyper_conn: nn.Module | None = None
    prev_layer_state: Any | None = None
    kv_key: str | None = None
    active_mask: torch.Tensor | None = None
    residual_module: nn.Module | None = None


@dataclass(frozen=True)
class StandardResidualPolicy:
    def run_block(self, strategy, context):
        core_fn = context.core_fn
        if strategy.x is None or core_fn is None:
            raise RuntimeError("standard residual execution requires x and core_fn")
        strategy.x, updated, skipped = strategy.owner._run_sublayer_nonmhc(
            strategy.x,
            context.normw,
            core_fn,
            context.gate,
            context.cfg,
            context.aux_l2_terms,
            context.prev_layer_state,
            context.kv_key,
            context.active_mask,
        )
        return updated, skipped


@dataclass(frozen=True)
class MHCResidualPolicy:
    def run_block(self, strategy, context):
        mhc_core = context.mhc_core
        hyper_conn = context.hyper_conn
        if strategy.streams is None or mhc_core is None or hyper_conn is None:
            raise RuntimeError("mHC residual execution requires streams, core, and connection")
        strategy.streams = strategy.owner._mhc_run_block(
            strategy.streams, context.normw, hyper_conn, mhc_core
        )
        return None, None


@dataclass(frozen=True)
class AttentionResidualPolicy:
    """Residual-replacement policy used by paper-style depth attention.

    Unlike the standard/mHC policies, this one replaces the residual stream
    entirely with an externally-carried ``attention_residual_state`` (see
    ``runtime/residual_state.py``) rather than accumulating onto ``strategy.x``.
    """

    def run_block(self, strategy, context):
        core_fn = context.core_fn
        state = strategy.attention_residual_state
        if state is None or core_fn is None:
            raise RuntimeError(
                "attention-residual execution requires attention_residual_state and core_fn"
            )
        residual_module = context.residual_module
        if residual_module is None:
            raise RuntimeError("attention-residual execution requires residual_module")
        x_in = _attention_residual_input(state["current"], state, residual_module)
        out, updated_kv = strategy.owner._run_attnres_core(
            x_in, context.normw, core_fn
        )
        _append_attention_residual_update(state, out)
        strategy.x = state["current"]
        return updated_kv, None


@dataclass
class LayerExecutionStrategy:
    owner: object
    use_mhc: bool
    x: torch.Tensor | None = None
    streams: torch.Tensor | None = None
    use_attention_residual: bool = False
    attention_residual_state: dict | None = None

    @property
    def policy(self) -> ResidualExecutionPolicy:
        if self.use_attention_residual:
            return AttentionResidualPolicy()
        return MHCResidualPolicy() if self.use_mhc else StandardResidualPolicy()

    def run_block(
        self, *, normw, gate, cfg, aux_l2_terms, core_fn=None, mhc_core=None,
        hyper_conn=None, prev_layer_state=None, kv_key=None, active_mask=None,
        residual_module=None,
    ):
        return self.policy.run_block(
            self,
            ResidualBlockContext(
                normw=normw,
                gate=gate,
                cfg=cfg,
                aux_l2_terms=aux_l2_terms,
                core_fn=core_fn,
                mhc_core=mhc_core,
                hyper_conn=hyper_conn,
                prev_layer_state=prev_layer_state,
                kv_key=kv_key,
                active_mask=active_mask,
                residual_module=residual_module,
            ),
        )

    def collapse(self, mode):
        if self.use_attention_residual:
            if self.x is None:
                raise RuntimeError("execution has no tensor")
            return self.x, None
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
    "AttentionResidualPolicy", "LayerExecutionStrategy", "MHCBlockMixin",
    "MHCResidualPolicy", "ModelLayerInvokeStrategy", "NormWrapper",
    "ResidualBlockContext", "ResidualBlockMixin", "ResidualExecutionPolicy", "ResidualRunCfg",
    "StandardResidualPolicy",
]
