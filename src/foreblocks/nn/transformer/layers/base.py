"""BaseTransformerLayer: feed-forward and residual-policy plumbing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from foreblocks.nn.moe.feedforward import FeedForwardBlock
from foreblocks.nn.residual.attention_residual import (
    AttentionResidual,
    BlockAttentionResidual,
)
from foreblocks.nn.residual.hyper_connections import (
    MHCHyperConnection,
    mhc_init_streams,
)
from foreblocks.nn.routing.gateskip import ResidualGate
from foreblocks.nn.transformer.config import TransformerConfig
from foreblocks.nn.transformer.runtime.execution import (
    LayerExecutionStrategy,
    NormWrapper,
    ResidualRunCfg,
)
from foreblocks.nn.transformer.runtime.residual_state import (
    AttentionResidualState,
    init_attention_residual_state,
)


class BaseTransformerLayer(nn.Module):
    """Shared construction for encoder and decoder layers.

    Accepts a ``TransformerConfig``, keyword overrides, or both. A layer runs
    one attention backend, so ``attention_pattern`` must be ``"uniform"``;
    stacks pass each layer ``config.for_layer(index)``.
    """

    def __init__(self, config: TransformerConfig | None = None, **overrides: Any):
        super().__init__()
        config = TransformerConfig.resolve(config, **overrides)
        if config.attention_pattern != "uniform":
            raise ValueError(
                "attention_pattern applies to stacks; build layers from "
                "config.for_layer(index)"
            )
        self.config = config
        self.feed_forward = FeedForwardBlock(
            d_model=config.d_model,
            dim_ff=config.ff_dim,
            dropout=config.dropout,
            use_swiglu=config.swiglu,
            activation=config.activation,
            use_moe=config.use_moe,
            num_experts=max(config.moe_experts, 1),
            top_k=config.moe_top_k,
            moe_use_latent=config.moe_latent,
            moe_latent_dim=config.moe_latent_dim,
            moe_latent_d_ff=config.moe_latent_ff_dim,
            **config.moe_options,
        )
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    # ---- Per-sublayer modules ----------------------------------------------------
    def _norm_wrapper(self) -> NormWrapper:
        config = self.config
        return NormWrapper.build(
            config.d_model,
            config.norm,
            config.norm_placement,
            config.dropout,
            config.norm_eps,
        )

    def _gate(self) -> ResidualGate | None:
        if self.config.residual != "gateskip":
            return None
        return ResidualGate(self.config.d_model)

    def _mhc_connection(self) -> MHCHyperConnection | None:
        if self.config.residual != "mhc":
            return None
        return MHCHyperConnection(
            d_model=self.config.d_model,
            n_streams=self.config.mhc_streams,
            sinkhorn_iters=self.config.mhc_sinkhorn_iters,
        )

    def _input_residual(self) -> nn.Module | None:
        if self.config.residual != "attention":
            return None
        if self.config.attention_residual_mode == "block":
            return BlockAttentionResidual(self.config.d_model)
        return AttentionResidual(self.config.d_model)

    # ---- Forward helpers ---------------------------------------------------------
    def _reset_aux_loss(self) -> None:
        self.aux_loss = self.aux_loss.new_zeros(())

    def _update_aux_loss(self, new_loss: float | torch.Tensor) -> None:
        self.aux_loss = self.aux_loss + new_loss

    def _residual_run_cfg(self, gate_budget: float | None) -> ResidualRunCfg:
        config = self.config
        return ResidualRunCfg(
            use_gateskip=config.residual == "gateskip",
            gate_budget=config.gate_budget if gate_budget is None else gate_budget,
            gate_lambda=config.gate_aux_weight,
            training=self.training,
        )

    def _make_exec_strategy(
        self,
        x: torch.Tensor,
        *,
        streams: torch.Tensor | None,
        attention_residual_state: AttentionResidualState | None,
    ) -> LayerExecutionStrategy:
        residual = self.config.residual
        if residual == "attention":
            if attention_residual_state is None:
                attention_residual_state = init_attention_residual_state(
                    x,
                    self.config.attention_residual_mode,
                    self.config.attention_residual_block_size,
                )
            return LayerExecutionStrategy(
                owner=self,
                use_mhc=False,
                x=attention_residual_state.current,
                use_attention_residual=True,
                attention_residual_state=attention_residual_state,
            )
        if residual != "mhc":
            return LayerExecutionStrategy(owner=self, use_mhc=False, x=x)
        n_streams = self.config.mhc_streams
        if streams is None:
            streams = mhc_init_streams(x, n_streams)
        elif streams.dim() != 4 or streams.shape[1] != n_streams:
            raise ValueError(
                f"mHC streams must be [B,{n_streams},T,D], got {tuple(streams.shape)}"
            )
        return LayerExecutionStrategy(owner=self, use_mhc=True, streams=streams)

    def _finalize_gateskip_aux(
        self,
        cfg: ResidualRunCfg,
        aux_l2_terms: list[torch.Tensor],
    ) -> None:
        if cfg.use_gateskip and cfg.gate_lambda > 0 and aux_l2_terms:
            self._update_aux_loss(cfg.gate_lambda * torch.stack(aux_l2_terms).mean())

    def _ff_forward_with_aux(
        self,
        x: torch.Tensor,
        mtp_targets: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.use_moe:
            out, aux = self.feed_forward(
                x,
                return_aux_loss=True,
                mtp_targets=mtp_targets,
                padding_mask=padding_mask,
            )
            self._update_aux_loss(aux)
            return out
        return self.feed_forward(x)

    def _run_attnres_core(
        self,
        x: torch.Tensor,
        normw: NormWrapper,
        core_fn: Callable[[torch.Tensor], tuple[torch.Tensor, dict | None]],
    ) -> tuple[torch.Tensor, dict | None]:
        x_in = normw.norm(x) if normw.placement in ("pre", "sandwich") else x
        out, updated = core_fn(x_in)
        out = normw.dropout(out)
        if normw.placement in ("post", "sandwich"):
            out = normw.norm(out)
        return out, updated


__all__ = ["BaseTransformerLayer"]
