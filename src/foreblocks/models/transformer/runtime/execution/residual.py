"""Normalization and residual execution for transformer sublayers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn

from foreblocks.layers.norms import create_norm_layer
from foreblocks.models.transformer.features.fusions import (
    fused_dropout_add,
    fused_dropout_add_norm,
    fused_dropout_gateskip_norm,
    get_dropout_p,
)
from foreblocks.models.transformer.features.mhc import (
    mhc_apply_norm_streamwise,
    mhc_collapse_streams,
)
from foreblocks.models.transformer.runtime.residual_state import (
    AttentionResidualState,
    append_attention_residual_update,
    attention_residual_input,
)
from foreblocks.modules.skip.gateskip import apply_skip_to_kv


class NormWrapper(nn.Module):
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
    ) -> NormWrapper:
        norm = create_norm_layer(norm_type, d_model, eps)
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        return NormWrapper(norm=norm, strategy=strategy, dropout=dropout_layer)

    def forward(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError(
            "NormWrapper is a holder, not callable — use .norm(...) or .dropout(...)"
        )


@dataclass(frozen=True)
class ResidualRunCfg:
    use_gateskip: bool
    gate_budget: float | None
    gate_lambda: float
    training: bool


class ExecutionOwner(Protocol):
    """Operations required by residual execution policies."""

    def _run_sublayer_nonmhc(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]: ...
    def _mhc_run_block(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def _run_attnres_core(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]: ...
    def _residual_apply(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]: ...
    def _drop_p(self, normw: NormWrapper) -> float: ...


class MHCConnection(Protocol):
    def pre_aggregate(self, streams: torch.Tensor) -> tuple[torch.Tensor, Any]: ...
    def combine(
        self, streams: torch.Tensor, update: torch.Tensor, *, maps: Any
    ) -> torch.Tensor: ...


class ResidualBlockMixin:
    @staticmethod
    def _drop_p(normw: NormWrapper) -> float:
        return get_dropout_p(getattr(normw, "dropout", None))

    def _residual_apply(
        self: ExecutionOwner,
        x: torch.Tensor,
        update: torch.Tensor,
        normw: NormWrapper,
        p: float,
        gate: nn.Module | None,
        cfg: ResidualRunCfg,
        aux_l2_terms: list[torch.Tensor],
        updated_kv: Any = None,
        prev_layer_state: Any = None,
        kv_key: str | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
        if cfg.use_gateskip:
            x2, skip_mask = fused_dropout_gateskip_norm(
                residual=x,
                update=update,
                gate=gate,
                use_gateskip=True,
                gate_budget=cfg.gate_budget,
                aux_l2_terms=aux_l2_terms,
                gate_lambda=cfg.gate_lambda,
                norm_layer=(normw if normw.strategy != "pre_norm" else None),
                p=p,
                training=cfg.training,
                active_mask=active_mask,
            )
            if skip_mask is not None and updated_kv is not None and kv_key is not None:
                updated_kv = apply_skip_to_kv(
                    updated_kv, skip_mask, prev_layer_state, kv_key
                )
            return x2, updated_kv, skip_mask
        if normw.strategy in ("pre_norm", "sandwich_norm"):
            x2 = fused_dropout_add(x, update, p=p, training=cfg.training)
            if normw.strategy == "sandwich_norm":
                x2 = normw.norm(x2)
        else:
            x2 = fused_dropout_add_norm(
                residual=x,
                update=update,
                norm_layer=normw,
                p=p,
                training=cfg.training,
            )
        return x2, updated_kv, None

    def _run_sublayer_nonmhc(
        self: ExecutionOwner,
        x: torch.Tensor,
        normw: NormWrapper,
        core_fn: CoreFn,
        gate: nn.Module | None,
        cfg: ResidualRunCfg,
        aux_l2_terms: list[torch.Tensor],
        prev_layer_state: Any = None,
        kv_key: str | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
        x_in = normw.norm(x) if normw.strategy in ("pre_norm", "sandwich_norm") else x
        update, updated_kv = core_fn(x_in)
        return self._residual_apply(
            x,
            update,
            normw,
            self._drop_p(normw),
            gate,
            cfg,
            aux_l2_terms,
            updated_kv,
            prev_layer_state,
            kv_key,
            active_mask,
        )


class MHCBlockMixin:
    def _mhc_run_block(
        self,
        streams: torch.Tensor,
        normw: NormWrapper,
        hyper_conn: MHCConnection,
        core_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        x_in, maps = hyper_conn.pre_aggregate(streams)
        if normw.strategy in ("pre_norm", "sandwich_norm"):
            x_in = normw.norm(x_in)
        streams = hyper_conn.combine(streams, normw.dropout(core_fn(x_in)), maps=maps)
        if normw.strategy in ("post_norm", "sandwich_norm"):
            streams = mhc_apply_norm_streamwise(normw.norm, streams)
        return streams


CoreFn = Callable[[torch.Tensor], tuple[torch.Tensor, Any | None]]
MhcCoreFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class LayerExecutionStrategy:
    owner: ExecutionOwner
    use_mhc: bool
    x: torch.Tensor | None = None
    streams: torch.Tensor | None = None
    use_attention_residual: bool = False
    attention_residual_state: AttentionResidualState | None = None

    def run_block(
        self,
        *,
        normw: NormWrapper,
        gate: nn.Module | None,
        cfg: ResidualRunCfg,
        aux_l2_terms: list[torch.Tensor],
        core_fn: CoreFn | None = None,
        mhc_core: MhcCoreFn | None = None,
        hyper_conn: MHCConnection | None = None,
        prev_layer_state: Any | None = None,
        kv_key: str | None = None,
        active_mask: torch.Tensor | None = None,
        residual_module: nn.Module | None = None,
    ) -> tuple[Any | None, torch.Tensor | None]:
        if self.use_attention_residual:
            state = self.attention_residual_state
            if state is None or core_fn is None:
                raise RuntimeError(
                    "attention-residual execution requires state and core_fn"
                )
            if residual_module is None:
                raise RuntimeError(
                    "attention-residual execution requires residual_module"
                )
            x_in = attention_residual_input(state.current, state, residual_module)
            out, updated = self.owner._run_attnres_core(x_in, normw, core_fn)
            append_attention_residual_update(state, out)
            self.x = state.current
            return updated, None

        if self.use_mhc:
            if self.streams is None or mhc_core is None or hyper_conn is None:
                raise RuntimeError(
                    "mHC residual execution requires streams, core, and connection"
                )
            self.streams = self.owner._mhc_run_block(
                self.streams, normw, hyper_conn, mhc_core
            )
            return None, None

        if self.x is None or core_fn is None:
            raise RuntimeError("standard residual execution requires x and core_fn")
        self.x, updated, skipped = self.owner._run_sublayer_nonmhc(
            self.x,
            normw,
            core_fn,
            gate,
            cfg,
            aux_l2_terms,
            prev_layer_state,
            kv_key,
            active_mask,
        )
        return updated, skipped

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


__all__ = [
    "ExecutionOwner",
    "LayerExecutionStrategy",
    "MHCBlockMixin",
    "NormWrapper",
    "ResidualBlockMixin",
    "ResidualRunCfg",
]
