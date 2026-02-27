# -*- coding: utf-8 -*-
# Transformer with Toggleable GateSkip + Hybrid Attention (+ PatchTST-style OPTIONAL patching)
# Adds: Kimi linear attention as a selectable option:
#   attention_mode ∈ {"standard","linear","hybrid","kimi","hybrid_kimi","kimi_3to1"}
#
# NEW: Optional PatchTST-style patching
#   - Best default for encoder-decoder forecasting: PATCH ENCODER ONLY.
#   - Encoder: [B,T,D] -> [B,Np,D] (patch tokens) and returns memory tokens (NO unpatch in encoder).
#   - Decoder (default): stays timestep-level [B,T,D], cross-attends to patch-memory [B,Np,D].
#   - Optionally you can patch decoder too (non-incremental only), then unpatch at the end.
#
# NEW: Optional mHC (manifold-constrained Hyper-Connections)
#   - Maintains N parallel residual streams internally: [B, N, T, D]
#   - After each residual block (attn/ffn, and cross-attn in decoder), mixes streams with
#     a Sinkhorn-projected doubly-stochastic matrix (Birkhoff polytope).
#   - Collapses streams back to [B, T, D] at output for backward compatibility.
#   - For simplicity and safety, GateSkip remains applied on the collapsed stream only.
#   - Decoder mHC: disabled for incremental_state (KV-cached decoding) in this sketch.
#
# NEW: Mixture-of-Depths (MoD) / Dynamic Layer Skipping
#   - LayerGate predicts keep probability per layer (token-wise or seq-wise)
#   - LayerBudgetScheduler provides per-layer keep-rate target (annealed)
#   - Seq-wise skipping SAVES compute (skip whole layer).
#   - Token-wise MoD is BEHAVIORAL here (mix output vs bypass), not compute-saving.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ui.node_spec import node

from .attention.kimi_att import KimiAttention
from .attention.lin_att import LinearAttention
from .attention.multi_att import MultiAttention
from .embeddings import InformerTimeEmbedding, PositionalEncoding
from .experts.moe import *  # FeedForwardBlock (and optionally MoE)
from .fusions import (
    fused_dropout_add,  # fused helpers
    fused_dropout_add_norm,
    fused_dropout_gateskip_norm,
    get_dropout_p,
)
from .mhc import *
from .norms import *  # create_norm_layer, RMSNorm, etc.
from .patching import *  # PatchTokenizer, PatchDetokenizer, PatchInfo, patchify_padding_mask, etc.
from .skip.gateskip import *  # ResidualGate, gateskip_apply, apply_skip_to_kv, BudgetScheduler
from .skip.layerskip import *  # LayerGate, LayerBudgetScheduler


# ──────────────────────────────────────────────────────────────────────────────
# NormWrapper (holder to avoid accidental double residuals)
# ──────────────────────────────────────────────────────────────────────────────
class NormWrapper(nn.Module):
    """
    Holder for a norm layer + dropout; do NOT call forward().

    Use:
        x_norm = self.attn_norm.norm(x)
        out = sublayer(x_norm)
        out = self.attn_norm.dropout(out)
        x = x + out
    """

    def __init__(
        self,
        d_model: int,
        norm_type: str = "rms",
        strategy: str = "pre_norm",
        dropout: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if strategy not in {"pre_norm", "post_norm", "sandwich_norm"}:
            raise ValueError(
                "norm strategy must be one of {'pre_norm','post_norm','sandwich_norm'}"
            )
        self.norm = create_norm_layer(norm_type, d_model, eps)
        self.strategy = strategy
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, *_, **__):
        raise RuntimeError(
            "NormWrapper.forward() disabled. Use `.norm(x)` then apply sublayer, "
            "then `.dropout(y)` and residual yourself."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Small configs / mixins to DRY repeated residual logic
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ResidualRunCfg:
    use_gateskip: bool
    gate_budget: Optional[float]
    gate_lambda: float
    training: bool


class ResidualBlockMixin:
    """
    Eliminates repetitive code for non-mHC residual blocks:
            - pre_norm vs post_norm vs sandwich_norm
      - gateskip vs plain residual
      - fused add vs fused add+norm
      - optional KV "skip propagation" (decoder)
    """

    @staticmethod
    def _drop_p(normw: NormWrapper) -> float:
        return get_dropout_p(getattr(normw, "dropout", None))

    def _residual_apply(
        self,
        x: torch.Tensor,
        update: torch.Tensor,
        normw: NormWrapper,
        p: float,
        gate: "ResidualGate",
        cfg: ResidualRunCfg,
        aux_l2_terms: List[torch.Tensor],
        # decoder-only KV handling
        updated_kv: Optional[dict] = None,
        prev_layer_state: Optional[dict] = None,
        kv_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[dict], Optional[torch.Tensor]]:
        """
        Returns:
          x_out,
          updated_kv (possibly modified by skip-mask),
          skip_mask (only when gateskip enabled; else None)
        """
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
            )
            if skip_mask is not None and updated_kv is not None and kv_key is not None:
                updated_kv = apply_skip_to_kv(
                    updated_kv, skip_mask, prev_layer_state, kv_key
                )
            return x2, updated_kv, skip_mask

        # non-gateskip
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
        self,
        x: torch.Tensor,
        normw: NormWrapper,
        core_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[dict]]],
        gate: "ResidualGate",
        cfg: ResidualRunCfg,
        aux_l2_terms: List[torch.Tensor],
        # KV skip support (decoder)
        prev_layer_state: Optional[dict] = None,
        kv_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[dict], Optional[torch.Tensor]]:
        """
        core_fn(x_in) -> (update, updated_kv_or_none)
        Returns: (x_out, updated_kv, skip_mask)
        """
        p = self._drop_p(normw)
        x_in = normw.norm(x) if normw.strategy in ("pre_norm", "sandwich_norm") else x
        update, updated_kv = core_fn(x_in)
        x_out, updated_kv, skip_mask = self._residual_apply(
            x=x,
            update=update,
            normw=normw,
            p=p,
            gate=gate,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            updated_kv=updated_kv,
            prev_layer_state=prev_layer_state,
            kv_key=kv_key,
        )
        return x_out, updated_kv, skip_mask


class MHCBlockMixin:
    """
    Eliminates repetitive stream-wise mHC block code:
      - (optional) streamwise norm
      - streamwise sublayer run
      - residual add dropout
            - (optional) streamwise norm (post_norm / sandwich_norm)
      - mixer
    """

    def _mhc_run_block(
        self,
        streams: torch.Tensor,  # [B,N,T,D]
        normw: NormWrapper,
        p: float,
        training: bool,
        mixer: nn.Module,
        flat_core: Callable[[torch.Tensor], torch.Tensor],  # x_flat -> upd_flat
    ) -> torch.Tensor:
        if normw.strategy in ("pre_norm", "sandwich_norm"):
            s_norm = mhc_apply_norm_streamwise(normw.norm, streams)
            upd = mhc_run_sublayer_streamwise(flat_core, s_norm)
            streams = fused_dropout_add(streams, upd, p=p, training=training)
            if normw.strategy == "sandwich_norm":
                streams = mhc_apply_norm_streamwise(normw.norm, streams)
        else:
            upd = mhc_run_sublayer_streamwise(flat_core, streams)
            streams = fused_dropout_add(streams, upd, p=p, training=training)
            streams = mhc_apply_norm_streamwise(normw.norm, streams)
        return mixer(streams)


@dataclass
class _LayerExecutionStrategy:
    owner: "BaseTransformerLayer"
    use_mhc: bool
    x: Optional[torch.Tensor] = None
    streams: Optional[torch.Tensor] = None

    def run_block(
        self,
        *,
        normw: NormWrapper,
        gate: "ResidualGate",
        cfg: ResidualRunCfg,
        aux_l2_terms: List[torch.Tensor],
        core_fn: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[dict]]]
        ] = None,
        flat_core: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mixer: Optional[nn.Module] = None,
        prev_layer_state: Optional[dict] = None,
        kv_key: Optional[str] = None,
    ) -> Tuple[Optional[dict], Optional[torch.Tensor]]:
        if not self.use_mhc:
            if self.x is None or core_fn is None:
                raise RuntimeError(
                    "Non-mHC strategy requires current tensor and core_fn."
                )
            self.x, updated_kv, skip_mask = self.owner._run_sublayer_nonmhc(
                x=self.x,
                normw=normw,
                core_fn=core_fn,
                gate=gate,
                cfg=cfg,
                aux_l2_terms=aux_l2_terms,
                prev_layer_state=prev_layer_state,
                kv_key=kv_key,
            )
            return updated_kv, skip_mask

        if self.streams is None or flat_core is None or mixer is None:
            raise RuntimeError("mHC strategy requires streams, flat_core, and mixer.")
        p = self.owner._drop_p(normw)
        self.streams = self.owner._mhc_run_block(
            streams=self.streams,
            normw=normw,
            p=p,
            training=self.owner.training,
            mixer=mixer,
            flat_core=flat_core,
        )
        return None, None

    def collapse(self, mode: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_mhc:
            if self.x is None:
                raise RuntimeError("Non-mHC strategy has no tensor to return.")
            return self.x, None
        if self.streams is None:
            raise RuntimeError("mHC strategy has no streams to collapse.")
        return mhc_collapse_streams(self.streams, mode=mode), self.streams


@dataclass(frozen=True)
class _ModelLayerInvokeStrategy:
    owner: "BaseTransformer"
    use_checkpoint: bool

    def run_encoder_layer(
        self,
        *,
        layer: nn.Module,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor],
        budget: Optional[float],
        streams: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_checkpoint:

            def ckpt_fn(_x, layer_mod=layer, b=budget):
                out_x, _ = layer_mod(
                    _x,
                    src_mask,
                    src_key_padding_mask,
                    gate_budget=b,
                    gate_lambda=self.owner.gate_lambda,
                    use_gateskip=self.owner.use_gateskip,
                    streams=None,
                    use_mhc=self.owner.use_mhc,
                    mhc_n_streams=self.owner.mhc_n_streams,
                    mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
                    mhc_temperature=self.owner.mhc_temperature,
                    mhc_collapse=self.owner.mhc_collapse,
                )
                return out_x

            x = self.owner._run_with_checkpoint(ckpt_fn, x, use_checkpoint=True)
            return x, streams

        return layer(
            x,
            src_mask,
            src_key_padding_mask,
            gate_budget=budget,
            gate_lambda=self.owner.gate_lambda,
            use_gateskip=self.owner.use_gateskip,
            streams=streams,
            use_mhc=self.owner.use_mhc,
            mhc_n_streams=self.owner.mhc_n_streams,
            mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
            mhc_temperature=self.owner.mhc_temperature,
            mhc_collapse=self.owner.mhc_collapse,
        )

    def run_decoder_layer(
        self,
        *,
        layer: nn.Module,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
        layer_state: Optional[dict],
        prev_state: Optional[dict],
        budget: Optional[float],
        streams: Optional[torch.Tensor],
        mtp_targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[dict], Optional[torch.Tensor]]:
        if self.use_checkpoint:

            def ckpt_fn(_x, layer_mod=layer, b=budget, ps=prev_state, ls=layer_state):
                out_x, _, _ = layer_mod(
                    _x,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    incremental_state=ls,
                    prev_layer_state=ps,
                    gate_budget=b,
                    gate_lambda=self.owner.gate_lambda,
                    use_gateskip=self.owner.use_gateskip,
                    streams=None,
                    use_mhc=self.owner.use_mhc,
                    mhc_n_streams=self.owner.mhc_n_streams,
                    mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
                    mhc_temperature=self.owner.mhc_temperature,
                    mhc_collapse=self.owner.mhc_collapse,
                    mtp_targets=mtp_targets,
                )
                return out_x

            x = self.owner._run_with_checkpoint(ckpt_fn, x, use_checkpoint=True)
            return x, layer_state, streams

        return layer(
            x,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            incremental_state=layer_state,
            prev_layer_state=prev_state,
            gate_budget=budget,
            gate_lambda=self.owner.gate_lambda,
            use_gateskip=self.owner.use_gateskip,
            streams=streams,
            use_mhc=self.owner.use_mhc,
            mhc_n_streams=self.owner.mhc_n_streams,
            mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
            mhc_temperature=self.owner.mhc_temperature,
            mhc_collapse=self.owner.mhc_collapse,
            mtp_targets=mtp_targets,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Base layer (shared by encoder/decoder layers)
# ──────────────────────────────────────────────────────────────────────────────
class BaseTransformerLayer(nn.Module):
    """
    Base layer with a FeedForwardBlock + MoE/GateSkip plumbing and aux_loss buffer.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        use_gateskip: bool = False,
        gate_budget: Optional[float] = None,
        gate_lambda: float = 0.1,
        # mHC
        use_mhc: bool = False,
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_temperature: float = 1.0,
        mhc_collapse: str = "first",
    ):
        super().__init__()
        self.use_moe = use_moe
        self.use_gateskip = use_gateskip
        self.gate_budget = gate_budget
        self.gate_lambda = gate_lambda

        # mHC knobs (layer-level)
        self.use_mhc = bool(use_mhc)
        self.mhc_n_streams = int(mhc_n_streams)
        self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)
        self.mhc_temperature = float(mhc_temperature)
        self.mhc_collapse = str(mhc_collapse)

        self.feed_forward = FeedForwardBlock(
            d_model=d_model,
            dim_ff=dim_feedforward,
            dropout=dropout,
            use_swiglu=use_swiglu,
            activation=activation,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
        )

        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def _reset_aux_loss(self) -> None:
        self.aux_loss.zero_()

    def _update_aux_loss(self, new_loss) -> None:
        if torch.is_tensor(new_loss):
            self.aux_loss += new_loss
        elif new_loss != 0:
            self.aux_loss += torch.tensor(new_loss, device=self.aux_loss.device)

    def _make_exec_strategy(
        self,
        *,
        x: torch.Tensor,
        streams: Optional[torch.Tensor],
    ) -> _LayerExecutionStrategy:
        if not self.use_mhc:
            return _LayerExecutionStrategy(owner=self, use_mhc=False, x=x)

        self._ensure_mhc_mixers()
        if streams is None:
            streams = mhc_init_streams(x, self.mhc_n_streams)
        else:
            if streams.dim() != 4:
                raise ValueError(f"mHC streams must be [B,N,T,D], got {streams.shape}")
            if streams.shape[1] != self.mhc_n_streams:
                raise ValueError(
                    f"mHC streams N={streams.shape[1]} != configured {self.mhc_n_streams}"
                )

        return _LayerExecutionStrategy(owner=self, use_mhc=True, streams=streams)

    def _apply_runtime_mhc_overrides(
        self,
        *,
        use_mhc: Optional[bool],
        mhc_n_streams: Optional[int],
        mhc_sinkhorn_iters: Optional[int],
        mhc_temperature: Optional[float],
        mhc_collapse: Optional[str],
    ) -> None:
        """Apply per-call mHC overrides (for backward compatibility)."""
        if use_mhc is not None:
            self.use_mhc = bool(use_mhc)
        if mhc_n_streams is not None:
            self.mhc_n_streams = int(mhc_n_streams)
        if mhc_sinkhorn_iters is not None:
            self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)
        if mhc_temperature is not None:
            self.mhc_temperature = float(mhc_temperature)
        if mhc_collapse is not None:
            self.mhc_collapse = str(mhc_collapse)

    def _build_residual_cfg(
        self,
        *,
        use_gateskip: Optional[bool],
        gate_budget: Optional[float],
        gate_lambda: Optional[float],
        training: bool,
    ) -> ResidualRunCfg:
        """Resolve runtime GateSkip knobs into a single config object."""
        _use_gk = self.use_gateskip if use_gateskip is None else bool(use_gateskip)
        _budget = self.gate_budget if gate_budget is None else gate_budget
        _lambda = self.gate_lambda if gate_lambda is None else float(gate_lambda)
        return ResidualRunCfg(
            use_gateskip=_use_gk,
            gate_budget=_budget,
            gate_lambda=_lambda,
            training=training,
        )

    def _finalize_gateskip_aux(
        self,
        cfg: ResidualRunCfg,
        aux_l2_terms: List[torch.Tensor],
    ) -> None:
        """Accumulate GateSkip regularization term once per layer forward."""
        if cfg.use_gateskip and cfg.gate_lambda > 0 and aux_l2_terms:
            self._update_aux_loss(cfg.gate_lambda * torch.stack(aux_l2_terms).mean())

    def _ff_forward_with_aux(
        self, x: torch.Tensor, mtp_targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Shared FFN / MoE forward with auxiliary loss accounting."""
        if self.use_moe:
            out, aux = self.feed_forward(
                x, return_aux_loss=True, mtp_targets=mtp_targets
            )
            self._update_aux_loss(aux)
            return out
        return self.feed_forward(x)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder layer
# ──────────────────────────────────────────────────────────────────────────────
class TransformerEncoderLayer(ResidualBlockMixin, MHCBlockMixin, BaseTransformerLayer):
    """
    Encoder layer with:
      - MultiAttention / LinearAttention / KimiAttention
      - Optional MoE FFN
      - Optional GateSkip on both attention and FFN
      - Pre-/Post-norm via NormWrapper
      - Optional mHC stream mixing (Sinkhorn doubly-stochastic)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        freq_modes: int = 16,
        use_swiglu: bool = True,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        use_gateskip: bool = False,
        gate_budget: Optional[float] = None,
        gate_lambda: float = 0.1,
        layer_attention_type: str = "standard",  # "standard" | "linear" | "kimi"
        # mHC
        use_mhc: bool = False,
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_temperature: float = 1.0,
        mhc_collapse: str = "first",
    ):
        super().__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_swiglu=use_swiglu,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            use_gateskip=use_gateskip,
            gate_budget=gate_budget,
            gate_lambda=gate_lambda,
            use_mhc=use_mhc,
            mhc_n_streams=mhc_n_streams,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_temperature=mhc_temperature,
            mhc_collapse=mhc_collapse,
        )

        # Keep 3 attention modules so shared-layer routing can work without rebuilding modules.
        self.self_attn_std = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
        )
        self.self_attn_lin = LinearAttention(
            d_model=d_model, n_heads=nhead, dropout=dropout
        )
        self.self_attn_sype = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type="sype",
            freq_modes=freq_modes,
        )
        self.self_attn_kimi = KimiAttention(
            d_model=d_model, n_heads=nhead, dropout=dropout
        )

        self.layer_attention_type = str(layer_attention_type)

        self.attn_norm = NormWrapper(
            d_model, custom_norm, norm_strategy, dropout, layer_norm_eps
        )
        self.ff_norm = NormWrapper(
            d_model, custom_norm, norm_strategy, dropout, layer_norm_eps
        )

        self.gate_attn = ResidualGate(d_model)
        self.gate_ff = ResidualGate(d_model)

        # mHC mixers (attn and ffn)
        self.mhc_mixer_attn: Optional[nn.Module] = None
        self.mhc_mixer_ff: Optional[nn.Module] = None
        if self.use_mhc:
            self._ensure_mhc_mixers()

    def set_layer_attention_type(self, layer_attention_type: str) -> None:
        self.layer_attention_type = str(layer_attention_type)

    def _self_attn(self) -> nn.Module:
        t = self.layer_attention_type
        if t == "linear":
            return self.self_attn_lin
        if t == "sype":
            return self.self_attn_sype
        if t == "kimi":
            return self.self_attn_kimi
        return self.self_attn_std

    def _ensure_mhc_mixers(self) -> None:
        if not self.use_mhc:
            self.mhc_mixer_attn = None
            self.mhc_mixer_ff = None
            return

        if (self.mhc_mixer_attn is None) or (
            getattr(self.mhc_mixer_attn, "n", None) != self.mhc_n_streams
        ):
            self.mhc_mixer_attn = MHCResidualMixer(
                n_streams=self.mhc_n_streams,
                sinkhorn_iters=self.mhc_sinkhorn_iters,
                init="identity",
                temperature=self.mhc_temperature,
            )
        if (self.mhc_mixer_ff is None) or (
            getattr(self.mhc_mixer_ff, "n", None) != self.mhc_n_streams
        ):
            self.mhc_mixer_ff = MHCResidualMixer(
                n_streams=self.mhc_n_streams,
                sinkhorn_iters=self.mhc_sinkhorn_iters,
                init="identity",
                temperature=self.mhc_temperature,
            )

    def forward(
        self,
        src: torch.Tensor,  # [B,T,D] (T may be patch tokens)
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        gate_budget: Optional[float] = None,
        gate_lambda: Optional[float] = None,
        use_gateskip: Optional[bool] = None,
        # mHC: carry streams across layers
        streams: Optional[torch.Tensor] = None,  # [B,N,T,D] if enabled
        use_mhc: Optional[bool] = None,
        mhc_n_streams: Optional[int] = None,
        mhc_sinkhorn_iters: Optional[int] = None,
        mhc_temperature: Optional[float] = None,
        mhc_collapse: Optional[str] = None,
        mtp_targets: Optional[torch.Tensor] = None,  # [B,T,H,D]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          out_src: [B,T,D] (collapsed if mHC enabled else standard)
          out_streams: [B,N,T,D] if mHC enabled else None
        """
        self._reset_aux_loss()

        # runtime overrides (kept for backward compatibility; prefer model-level setters)
        self._apply_runtime_mhc_overrides(
            use_mhc=use_mhc,
            mhc_n_streams=mhc_n_streams,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_temperature=mhc_temperature,
            mhc_collapse=mhc_collapse,
        )
        cfg = self._build_residual_cfg(
            use_gateskip=use_gateskip,
            gate_budget=gate_budget,
            gate_lambda=gate_lambda,
            training=self.training,
        )
        aux_l2_terms: List[torch.Tensor] = []

        attn_mod = self._self_attn()

        strategy = self._make_exec_strategy(x=src, streams=streams)
        if self.use_mhc:
            assert self.mhc_mixer_attn is not None and self.mhc_mixer_ff is not None

        def attn_flat(x_flat: torch.Tensor) -> torch.Tensor:
            out, _, _ = attn_mod(x_flat, x_flat, x_flat, src_mask, src_key_padding_mask)
            return out

        def attn_core(x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
            out, _, _ = attn_mod(x_in, x_in, x_in, src_mask, src_key_padding_mask)
            return out, None

        strategy.run_block(
            normw=self.attn_norm,
            gate=self.gate_attn,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=attn_core,
            flat_core=attn_flat,
            mixer=self.mhc_mixer_attn,
        )

        def ff_flat(x_flat: torch.Tensor) -> torch.Tensor:
            flat_mtp = None
            if mtp_targets is not None:
                flat_mtp = mtp_targets.reshape(
                    mtp_targets.size(0) * mtp_targets.size(1),
                    mtp_targets.size(2),
                    mtp_targets.size(3),
                )
            return self._ff_forward_with_aux(x_flat, mtp_targets=flat_mtp)

        def ff_core(x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
            return self._ff_forward_with_aux(x_in, mtp_targets=mtp_targets), None

        strategy.run_block(
            normw=self.ff_norm,
            gate=self.gate_ff,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=ff_core,
            flat_core=ff_flat,
            mixer=self.mhc_mixer_ff,
        )

        self._finalize_gateskip_aux(cfg, aux_l2_terms)

        out_src, out_streams = strategy.collapse(self.mhc_collapse)
        return out_src, out_streams


# ──────────────────────────────────────────────────────────────────────────────
# Decoder layer
# ──────────────────────────────────────────────────────────────────────────────
class TransformerDecoderLayer(ResidualBlockMixin, MHCBlockMixin, BaseTransformerLayer):
    """
    Decoder layer with:
      - (Kimi|Linear|Multi) self-attention
      - Multi cross-attention
      - Optional GateSkip and MoE
      - Incremental state for autoregressive decoding
      - Optional mHC stream mixing (disabled when incremental_state is used)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        freq_modes: int = 32,
        use_swiglu: bool = True,
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        informer_like: bool = False,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        use_gateskip: bool = False,
        gate_budget: Optional[float] = None,
        gate_lambda: float = 0.1,
        layer_attention_type: str = "standard",  # "standard" | "linear" | "kimi"
        # mHC
        use_mhc: bool = False,
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_temperature: float = 1.0,
        mhc_collapse: str = "first",
    ):
        super().__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_swiglu=use_swiglu,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            use_gateskip=use_gateskip,
            gate_budget=gate_budget,
            gate_lambda=gate_lambda,
            use_mhc=use_mhc,
            mhc_n_streams=mhc_n_streams,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_temperature=mhc_temperature,
            mhc_collapse=mhc_collapse,
        )

        # Self-attention variants kept simultaneously (for shared-layer routing).
        self.self_attn_std = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
            cross_attention=False,
        )
        self.self_attn_lin = LinearAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            cross_attention=False,
        )
        self.self_attn_sype = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type="sype",
            freq_modes=freq_modes,
            cross_attention=False,
        )
        self.self_attn_kimi = KimiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            cross_attention=False,
        )
        self.layer_attention_type = str(layer_attention_type)

        # Cross-attention: always standard MultiAttention
        self.cross_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
            cross_attention=True,
        )

        self.is_causal = not informer_like

        self.self_attn_norm = NormWrapper(
            d_model, custom_norm, norm_strategy, dropout, layer_norm_eps
        )
        self.cross_attn_norm = NormWrapper(
            d_model, custom_norm, norm_strategy, dropout, layer_norm_eps
        )
        self.ff_norm = NormWrapper(
            d_model, custom_norm, norm_strategy, dropout, layer_norm_eps
        )

        self.gate_self = ResidualGate(d_model)
        self.gate_cross = ResidualGate(d_model)
        self.gate_ff = ResidualGate(d_model)

        # mHC mixers (self, cross, ffn)
        self.mhc_mixer_self: Optional[nn.Module] = None
        self.mhc_mixer_cross: Optional[nn.Module] = None
        self.mhc_mixer_ff: Optional[nn.Module] = None
        if self.use_mhc:
            self._ensure_mhc_mixers()

    def set_layer_attention_type(self, layer_attention_type: str) -> None:
        self.layer_attention_type = str(layer_attention_type)

    def _self_attn(self) -> nn.Module:
        t = self.layer_attention_type
        if t == "linear":
            return self.self_attn_lin
        if t == "sype":
            return self.self_attn_sype
        if t == "kimi":
            return self.self_attn_kimi
        return self.self_attn_std

    def _ensure_mhc_mixers(self) -> None:
        if not self.use_mhc:
            self.mhc_mixer_self = None
            self.mhc_mixer_cross = None
            self.mhc_mixer_ff = None
            return

        if (self.mhc_mixer_self is None) or (
            getattr(self.mhc_mixer_self, "n", None) != self.mhc_n_streams
        ):
            self.mhc_mixer_self = MHCResidualMixer(
                n_streams=self.mhc_n_streams,
                sinkhorn_iters=self.mhc_sinkhorn_iters,
                init="identity",
                temperature=self.mhc_temperature,
            )
        if (self.mhc_mixer_cross is None) or (
            getattr(self.mhc_mixer_cross, "n", None) != self.mhc_n_streams
        ):
            self.mhc_mixer_cross = MHCResidualMixer(
                n_streams=self.mhc_n_streams,
                sinkhorn_iters=self.mhc_sinkhorn_iters,
                init="identity",
                temperature=self.mhc_temperature,
            )
        if (self.mhc_mixer_ff is None) or (
            getattr(self.mhc_mixer_ff, "n", None) != self.mhc_n_streams
        ):
            self.mhc_mixer_ff = MHCResidualMixer(
                n_streams=self.mhc_n_streams,
                sinkhorn_iters=self.mhc_sinkhorn_iters,
                init="identity",
                temperature=self.mhc_temperature,
            )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[dict] = None,
        prev_layer_state: Optional[dict] = None,  # for KV copy on skip
        gate_budget: Optional[float] = None,
        gate_lambda: Optional[float] = None,
        use_gateskip: Optional[bool] = None,
        # mHC
        streams: Optional[torch.Tensor] = None,  # [B,N,T,D]
        use_mhc: Optional[bool] = None,
        mhc_n_streams: Optional[int] = None,
        mhc_sinkhorn_iters: Optional[int] = None,
        mhc_temperature: Optional[float] = None,
        mhc_collapse: Optional[str] = None,
        mtp_targets: Optional[torch.Tensor] = None,  # [B,T,H,D]
    ) -> Tuple[torch.Tensor, Optional[dict], Optional[torch.Tensor]]:
        """
        Returns:
          out_tgt: [B,T,D] (collapsed if mHC enabled else standard)
          ret_state: incremental state dict (or None)
          out_streams: [B,N,T,D] if mHC enabled else None
        """
        self._reset_aux_loss()

        # runtime overrides (kept for backward compatibility; prefer model-level setters)
        self._apply_runtime_mhc_overrides(
            use_mhc=use_mhc,
            mhc_n_streams=mhc_n_streams,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_temperature=mhc_temperature,
            mhc_collapse=mhc_collapse,
        )
        cfg = self._build_residual_cfg(
            use_gateskip=use_gateskip,
            gate_budget=gate_budget,
            gate_lambda=gate_lambda,
            training=self.training,
        )
        aux_l2_terms: List[torch.Tensor] = []

        # mHC sketch limitation: no incremental_state support (KV cache shape coupling)
        if self.use_mhc and incremental_state is not None:
            raise RuntimeError(
                "mHC decoder sketch does not support incremental_state/KV-cached decoding yet. "
                "Disable mHC for autoregressive decoding."
            )

        # Safety: if caller passes a timestep-level memory mask while memory is patch-token length,
        # catch it early (common silent bug when patch_encoder=True).
        if memory_key_padding_mask is not None:
            if (
                memory_key_padding_mask.shape[0] != memory.shape[0]
                or memory_key_padding_mask.shape[1] != memory.shape[1]
            ):
                raise ValueError(
                    f"memory_key_padding_mask shape {tuple(memory_key_padding_mask.shape)} must match memory [B,Tm]=[{memory.shape[0]},{memory.shape[1]}]"
                )

        state = {
            "self_attn": incremental_state.get("self_attn")
            if incremental_state
            else None,
            "cross_attn": incremental_state.get("cross_attn")
            if incremental_state
            else None,
        }

        self_attn_mod = self._self_attn()

        strategy = self._make_exec_strategy(x=tgt, streams=streams)
        if self.use_mhc:
            assert (
                self.mhc_mixer_self is not None
                and self.mhc_mixer_cross is not None
                and self.mhc_mixer_ff is not None
            )

        def self_flat(x_flat: torch.Tensor) -> torch.Tensor:
            kpm_flat = mhc_repeat_kpm(tgt_key_padding_mask, self.mhc_n_streams)
            am_flat = mhc_repeat_attn_mask(tgt_mask, self.mhc_n_streams)
            out, _, _ = self_attn_mod(
                x_flat,
                x_flat,
                x_flat,
                am_flat,
                kpm_flat,
                is_causal=self.is_causal,
                layer_state=None,
            )
            return out

        def self_core(x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
            out, _, updated = self_attn_mod(
                x_in,
                x_in,
                x_in,
                tgt_mask,
                tgt_key_padding_mask,
                is_causal=self.is_causal,
                layer_state=state["self_attn"],
            )
            return out, updated

        updated_self, _ = strategy.run_block(
            normw=self.self_attn_norm,
            gate=self.gate_self,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=self_core,
            flat_core=self_flat,
            mixer=self.mhc_mixer_self,
            prev_layer_state=prev_layer_state,
            kv_key="self_attn",
        )
        if updated_self is not None:
            state["self_attn"] = updated_self

        def cross_flat(x_flat: torch.Tensor) -> torch.Tensor:
            mkpm_flat = mhc_repeat_kpm(memory_key_padding_mask, self.mhc_n_streams)
            mm_flat = mhc_repeat_attn_mask(memory_mask, self.mhc_n_streams)

            mem_flat = (
                memory.repeat_interleave(self.mhc_n_streams, dim=0)
                if memory.dim() == 3
                else memory
            )
            out, _, _ = self.cross_attn(
                x_flat,
                mem_flat,
                mem_flat,
                mm_flat,
                mkpm_flat,
                layer_state=None,
            )
            return out

        def cross_core(x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
            out, _, updated = self.cross_attn(
                x_in,
                memory,
                memory,
                memory_mask,
                memory_key_padding_mask,
                layer_state=state["cross_attn"],
            )
            return out, updated

        updated_cross, _ = strategy.run_block(
            normw=self.cross_attn_norm,
            gate=self.gate_cross,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=cross_core,
            flat_core=cross_flat,
            mixer=self.mhc_mixer_cross,
            prev_layer_state=prev_layer_state,
            kv_key="cross_attn",
        )
        if updated_cross is not None:
            state["cross_attn"] = updated_cross

        def ff_flat(x_flat: torch.Tensor) -> torch.Tensor:
            flat_mtp = None
            if mtp_targets is not None:
                flat_mtp = mtp_targets.reshape(
                    mtp_targets.size(0) * mtp_targets.size(1),
                    mtp_targets.size(2),
                    mtp_targets.size(3),
                )
            return self._ff_forward_with_aux(x_flat, mtp_targets=flat_mtp)

        def ff_core(x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
            return self._ff_forward_with_aux(x_in, mtp_targets=mtp_targets), None

        strategy.run_block(
            normw=self.ff_norm,
            gate=self.gate_ff,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=ff_core,
            flat_core=ff_flat,
            mixer=self.mhc_mixer_ff,
        )

        self._finalize_gateskip_aux(cfg, aux_l2_terms)

        out_tgt, out_streams = strategy.collapse(self.mhc_collapse)
        ret_state = {"self_attn": state["self_attn"], "cross_attn": state["cross_attn"]}
        return out_tgt, ret_state, out_streams


# ──────────────────────────────────────────────────────────────────────────────
# Base transformer
# ──────────────────────────────────────────────────────────────────────────────
class BaseTransformer(nn.Module, ABC):
    """
    Base class shared by TransformerEncoder and TransformerDecoder.

    Handles:
      - input projection + positional / time embedding hook
      - layer construction with attention_mode routing
      - GateSkip runtime knobs and BudgetScheduler
      - gradient checkpointing (per-layer)
      - aux_loss aggregation (FIXED: uses executed layer indices)
      - Optional mHC knobs and stream propagation
      - Optional PatchTST-style patching knobs
      - Optional Mixture-of-Depths / Dynamic Layer Skipping
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        max_seq_len: int = 5000,
        pos_encoding_scale: float = 1.0,
        pos_encoder: Optional[nn.Module] = None,
        use_gradient_checkpointing: bool = False,
        share_layers: bool = False,
        use_final_norm: bool = True,
        use_swiglu: bool = True,
        freq_modes: int = 32,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        # GateSkip toggles
        use_gateskip: bool = False,
        gate_budget: Optional[float] = None,
        gate_lambda: float = 0.1,
        # Attention mode
        attention_mode: Literal[
            "standard",
            "linear",
            "sype",
            "hybrid",
            "kimi",
            "hybrid_kimi",
            "kimi_3to1",
        ] = "standard",
        # mHC toggles
        use_mhc: bool = False,
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_temperature: float = 1.0,
        mhc_collapse: str = "first",  # "first" or "mean"
        # PatchTST-style patching (best default: patch encoder only)
        patch_encoder: bool = True,
        patch_decoder: bool = False,
        patch_len: int = 16,
        patch_stride: int = 8,
        patch_pad_end: bool = True,
        # MoD / Dynamic depth skipping
        use_layer_skipping: bool = False,
        layer_skip_mode: Literal["token", "seq"] = "seq",
        layer_skip_temperature: float = 1.0,
        layer_skip_hard: bool = True,
        layer_skip_lambda: float = 0.05,
        layer_budget_scheduler: Optional[LayerBudgetScheduler] = None,
        # Global scaling for MoE aux loss (FIX)
        moe_aux_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gateskip = use_gateskip
        self.gate_budget = gate_budget
        self.gate_lambda = gate_lambda
        # Backward-compatible behavior:
        # if caller sets att_type to a routed mode but leaves attention_mode default,
        # promote attention_mode so the intended path is used.
        if attention_mode == "standard" and att_type in {"linear", "sype", "kimi"}:
            attention_mode = att_type
        self.attention_mode = attention_mode
        self.att_type = att_type
        self.budget_scheduler: Optional[BudgetScheduler] = None

        # MoE aux scaling (FIX)
        self.moe_aux_lambda = float(moe_aux_lambda)

        # mHC model-level
        self.use_mhc = bool(use_mhc)
        self.mhc_n_streams = int(mhc_n_streams)
        self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)
        self.mhc_temperature = float(mhc_temperature)
        self.mhc_collapse = str(mhc_collapse)

        # PatchTST-style patching knobs (subclasses decide whether to apply)
        self.patch_encoder = bool(patch_encoder)
        self.patch_decoder = bool(patch_decoder)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.patch_pad_end = bool(patch_pad_end)

        # MoD / layer skipping knobs
        self.use_layer_skipping = bool(use_layer_skipping)
        self.layer_skip_mode = str(layer_skip_mode)
        self.layer_skip_temperature = float(layer_skip_temperature)
        self.layer_skip_hard = bool(layer_skip_hard)
        self.layer_skip_lambda = float(layer_skip_lambda)
        self.layer_budget_scheduler = layer_budget_scheduler

        # Modules
        self.patcher = PatchTokenizer(
            self.d_model, self.patch_len, self.patch_stride, pad_end=self.patch_pad_end
        )
        self.unpatcher = PatchDetokenizer(
            self.d_model, self.patch_len, self.patch_stride
        )

        self.input_adapter = nn.Linear(input_size, self.d_model)
        self.pos_encoder = pos_encoder or PositionalEncoding(
            self.d_model, max_len=max_seq_len, scale=pos_encoding_scale
        )

        self.register_buffer("_causal_mask", torch.empty(0, 0), persistent=False)

        # Build layers (per-layer attention type decided by attention_mode)
        if share_layers:
            layer_attn_type = self._get_layer_attention_type(0)
            layer_kwargs = self._build_layer_kwargs(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                att_type,
                norm_strategy,
                custom_norm,
                layer_norm_eps,
                use_swiglu,
                freq_modes,
                use_moe,
                num_experts,
                top_k,
                use_gateskip,
                gate_budget,
                gate_lambda,
                layer_attn_type,
                use_mhc=self.use_mhc,
                mhc_n_streams=self.mhc_n_streams,
                mhc_sinkhorn_iters=self.mhc_sinkhorn_iters,
                mhc_temperature=self.mhc_temperature,
                mhc_collapse=self.mhc_collapse,
                **kwargs,
            )
            self.shared_layer = self._make_layer(**layer_kwargs)
            self.layers = None
        else:
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                layer_attn_type = self._get_layer_attention_type(i)
                layer_kwargs = self._build_layer_kwargs(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    activation,
                    att_type,
                    norm_strategy,
                    custom_norm,
                    layer_norm_eps,
                    use_swiglu,
                    freq_modes,
                    use_moe,
                    num_experts,
                    top_k,
                    use_gateskip,
                    gate_budget,
                    gate_lambda,
                    layer_attn_type,
                    use_mhc=self.use_mhc,
                    mhc_n_streams=self.mhc_n_streams,
                    mhc_sinkhorn_iters=self.mhc_sinkhorn_iters,
                    mhc_temperature=self.mhc_temperature,
                    mhc_collapse=self.mhc_collapse,
                    **kwargs,
                )
                self.layers.append(self._make_layer(**layer_kwargs))
            self.shared_layer = None

        self.final_norm = (
            create_norm_layer(custom_norm, d_model, layer_norm_eps)
            if use_final_norm
            else nn.Identity()
        )

        # Per-layer gates (even if share_layers=True)
        self.layer_gates = nn.ModuleList(
            [
                LayerGate(
                    d_model=self.d_model,
                    mode=self.layer_skip_mode,
                    hidden=0,
                    init_bias=2.0,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("layer_skip_aux", torch.tensor(0.0), persistent=False)
        self.apply(self._init_weights)
        self._print_init_summary(
            att_type=att_type,
            custom_norm=custom_norm,
            norm_strategy=norm_strategy,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            share_layers=share_layers,
            use_final_norm=use_final_norm,
        )

    def _generate_causal_mask(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        size = int(size)
        if size <= 0:
            return torch.empty(0, 0, device=device, dtype=dtype)

        mask = self._causal_mask
        needs_rebuild = (
            mask.numel() == 0
            or mask.device != device
            or mask.size(0) < size
            or mask.size(1) < size
        )
        if needs_rebuild:
            mask = torch.triu(
                torch.full(
                    (size, size), float("-inf"), device=device, dtype=torch.float32
                ),
                diagonal=1,
            )
            self._causal_mask = mask
        else:
            mask = mask[:size, :size]

        if mask.dtype != dtype:
            mask = mask.to(dtype=dtype)
        return mask

    # ---- Attention mode routing ------------------------------------------------
    def _get_layer_attention_type(self, layer_idx: int) -> str:
        mode = self.attention_mode
        if mode == "standard":
            return "standard"
        if mode == "linear":
            return "linear"
        if mode == "sype":
            return "sype"
        if mode == "kimi":
            return "kimi"
        if mode in ("hybrid", "hybrid_linear"):
            return "linear" if layer_idx < (self.num_layers - 1) else "standard"
        if mode in ("hybrid_kimi", "kimi_hybrid"):
            return "kimi" if layer_idx < (self.num_layers - 1) else "standard"
        if mode == "kimi_3to1":
            return "kimi" if (layer_idx % 4) < 3 else "standard"
        raise ValueError(f"Unknown attention_mode: {mode}")

    def _build_layer_kwargs(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        att_type,
        norm_strategy,
        custom_norm,
        layer_norm_eps,
        use_swiglu,
        freq_modes,
        use_moe,
        num_experts,
        top_k,
        use_gateskip,
        gate_budget,
        gate_lambda,
        layer_attention_type,
        # mHC
        use_mhc: bool,
        mhc_n_streams: int,
        mhc_sinkhorn_iters: int,
        mhc_temperature: float,
        mhc_collapse: str,
        **kwargs,
    ):
        return {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "att_type": att_type,
            "norm_strategy": norm_strategy,
            "custom_norm": custom_norm,
            "layer_norm_eps": layer_norm_eps,
            "use_swiglu": use_swiglu,
            "freq_modes": freq_modes,
            "use_moe": use_moe,
            "num_experts": num_experts,
            "top_k": top_k,
            "use_gateskip": use_gateskip,
            "gate_budget": gate_budget,
            "gate_lambda": gate_lambda,
            "layer_attention_type": layer_attention_type,
            # mHC
            "use_mhc": use_mhc,
            "mhc_n_streams": mhc_n_streams,
            "mhc_sinkhorn_iters": mhc_sinkhorn_iters,
            "mhc_temperature": mhc_temperature,
            "mhc_collapse": mhc_collapse,
            **kwargs,
        }

    # ---- GateSkip runtime setters ---------------------------------------------
    def _set_layer_gateskip_attrs(self, layer: nn.Module) -> None:
        if hasattr(layer, "use_gateskip"):
            layer.use_gateskip = bool(self.use_gateskip)
        if hasattr(layer, "gate_budget"):
            layer.gate_budget = self.gate_budget
        if hasattr(layer, "gate_lambda"):
            layer.gate_lambda = float(self.gate_lambda)

    def set_use_gateskip(self, flag: bool) -> None:
        self.use_gateskip = bool(flag)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_gateskip_attrs(layer)

    def set_gate_budget(self, budget: Optional[float]) -> None:
        self.gate_budget = budget
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_gateskip_attrs(layer)

    def set_gate_lambda(self, lam: float) -> None:
        self.gate_lambda = float(lam)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_gateskip_attrs(layer)

    def set_budget_scheduler(self, scheduler: BudgetScheduler) -> None:
        self.budget_scheduler = scheduler

    # ---- mHC runtime setters ---------------------------------------------------
    def _set_layer_mhc_attrs(self, layer: nn.Module) -> None:
        if hasattr(layer, "use_mhc"):
            layer.use_mhc = bool(self.use_mhc)
        if hasattr(layer, "mhc_n_streams"):
            layer.mhc_n_streams = int(self.mhc_n_streams)
        if hasattr(layer, "mhc_sinkhorn_iters"):
            layer.mhc_sinkhorn_iters = int(self.mhc_sinkhorn_iters)
        if hasattr(layer, "mhc_temperature"):
            layer.mhc_temperature = float(self.mhc_temperature)
        if hasattr(layer, "mhc_collapse"):
            layer.mhc_collapse = str(self.mhc_collapse)
        if hasattr(layer, "_ensure_mhc_mixers"):
            layer._ensure_mhc_mixers()

    def set_use_mhc(self, flag: bool) -> None:
        self.use_mhc = bool(flag)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_mhc_attrs(layer)

    def set_mhc_params(
        self,
        n_streams: Optional[int] = None,
        sinkhorn_iters: Optional[int] = None,
        temperature: Optional[float] = None,
        collapse: Optional[str] = None,
    ) -> None:
        if n_streams is not None:
            self.mhc_n_streams = int(n_streams)
        if sinkhorn_iters is not None:
            self.mhc_sinkhorn_iters = int(sinkhorn_iters)
        if temperature is not None:
            self.mhc_temperature = float(temperature)
        if collapse is not None:
            self.mhc_collapse = str(collapse)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_mhc_attrs(layer)

    # ---- MoD / layer skipping setters -----------------------------------------
    def set_use_layer_skipping(self, flag: bool) -> None:
        self.use_layer_skipping = bool(flag)

    def set_layer_budget_scheduler(self, scheduler: LayerBudgetScheduler) -> None:
        self.layer_budget_scheduler = scheduler

    # ---- Layer factory ---------------------------------------------------------
    @abstractmethod
    def _make_layer(self, **kwargs) -> nn.Module: ...

    # ---- Init & helpers --------------------------------------------------------
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _get_layer(self, idx: int) -> nn.Module:
        return self.shared_layer if self.layers is None else self.layers[idx]

    def _should_print_init_summary(self) -> bool:
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def _print_init_summary(
        self,
        *,
        att_type: str,
        custom_norm: str,
        norm_strategy: str,
        use_moe: bool,
        num_experts: int,
        top_k: int,
        share_layers: bool,
        use_final_norm: bool,
    ) -> None:
        if not self._should_print_init_summary():
            return

        per_layer_attn = [
            self._get_layer_attention_type(i) for i in range(self.num_layers)
        ]
        attn_mix = ", ".join(sorted(set(per_layer_attn)))
        print(
            f"[{self.__class__.__name__}] "
            f"att_type={att_type} | "
            f"attention_mode={self.attention_mode} (layers={attn_mix}) | "
            f"norm={custom_norm}/{norm_strategy} final_norm={use_final_norm} | "
            f"moe={use_moe} experts={num_experts} top_k={top_k} moe_aux_lambda={self.moe_aux_lambda} | "
            f"gateskip={self.use_gateskip} budget={self.gate_budget} gate_lambda={self.gate_lambda} | "
            f"mhc={self.use_mhc} streams={self.mhc_n_streams} collapse={self.mhc_collapse} | "
            f"patch(enc={self.patch_encoder},dec={self.patch_decoder},len={self.patch_len},stride={self.patch_stride}) | "
            f"layer_skip={self.use_layer_skipping}/{self.layer_skip_mode} | "
            f"shared_layers={share_layers} grad_ckpt={self.use_gradient_checkpointing}"
        )

    # FIX: aggregate aux loss over executed layer indices (supports skipping)
    def _aggregate_aux_loss(self, used_indices: List[int]) -> None:
        total_aux = self.aux_loss.new_zeros(())
        for i in used_indices:
            layer = self._get_layer(i)
            if hasattr(layer, "aux_loss"):
                total_aux = total_aux + layer.aux_loss
        denom = max(len(used_indices), 1)
        self.aux_loss = (total_aux / denom) * self.moe_aux_lambda
        # Add MoD/layer-skip auxiliary after layer aggregation
        self.aux_loss = self.aux_loss + self.layer_skip_aux

    @staticmethod
    def _run_with_checkpoint(fn, *inputs: torch.Tensor, use_checkpoint: bool):
        if not use_checkpoint:
            return fn(*inputs)
        return torch.utils.checkpoint.checkpoint(fn, *inputs, use_reentrant=False)

    def _get_runtime_budget(self) -> Optional[float]:
        if self.training and (self.budget_scheduler is not None):
            return self.budget_scheduler.get_budget()
        return self.gate_budget

    def _accum_layer_skip_aux(
        self, keep_prob: torch.Tensor, target_keep: float
    ) -> None:
        if self.use_layer_skipping and self.training:
            exp_keep = keep_prob.mean()
            self.layer_skip_aux = (
                self.layer_skip_aux
                + self.layer_skip_lambda * (exp_keep - target_keep) ** 2
            )

    def _prepare_layer_step(
        self,
        layer_idx: int,
        x: torch.Tensor,
        runtime_budget: Optional[float] = None,
    ) -> Tuple[nn.Module, Optional[float], bool, Optional[torch.Tensor]]:
        """
        Common per-layer orchestration:
          - shared-layer attention routing
          - runtime layer budget
          - skip decision + aux accumulation
        Returns:
          layer, budget, skip_layer_compute, keep_mask
        """
        layer = self._get_layer(layer_idx)
        if (self.shared_layer is not None) and hasattr(
            layer, "set_layer_attention_type"
        ):
            layer.set_layer_attention_type(self._get_layer_attention_type(layer_idx))

        budget = (
            runtime_budget if runtime_budget is not None else self._get_runtime_budget()
        )
        if (not self.use_layer_skipping) or (not self.training):
            return layer, budget, False, None

        skip_layer_compute, keep_prob, keep_mask, target_keep = self._maybe_skip_layer(
            layer_idx, x
        )
        self._accum_layer_skip_aux(keep_prob, target_keep)
        return layer, budget, skip_layer_compute, keep_mask

    def _maybe_apply_token_skip_mix(
        self,
        x_new: torch.Tensor,
        x_old: torch.Tensor,
        keep_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            keep_mask is not None
            and self.use_layer_skipping
            and self.training
            and (self.layer_skip_mode == "token")
        ):
            return self._token_mode_mix(x_new, x_old, keep_mask)
        return x_new

    def _finalize_layer_stack(self, used_indices: List[int]) -> None:
        self._aggregate_aux_loss(used_indices)
        if self.budget_scheduler is not None:
            self.budget_scheduler.step()
        if self.layer_budget_scheduler is not None:
            self.layer_budget_scheduler.step()

    @staticmethod
    def _token_mode_mix(
        x_new: torch.Tensor, x_old: torch.Tensor, keep_mask: torch.Tensor
    ) -> torch.Tensor:
        return keep_mask * x_new + (1.0 - keep_mask) * x_old

    # ---- MoD: decide skip / keep mask -----------------------------------------
    def _maybe_skip_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,  # [B,T,D]
    ) -> Tuple[bool, torch.Tensor, torch.Tensor, float]:
        """
        Returns:
          skip_layer_compute: bool   # meaningful only for layer_skip_mode=="seq"
          keep_prob: Tensor         # [B,T,1] or [B,1,1]
          keep_mask: Tensor         # same shape (hard ST or soft)
          target_keep: float
        """
        if (not self.use_layer_skipping) or (not self.training):
            keep_prob = torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
            keep_mask = keep_prob
            return False, keep_prob, keep_mask, 1.0

        gate = self.layer_gates[layer_idx]
        logits = gate(x)

        target_keep = (
            float(self.layer_budget_scheduler.get_keep_rate(layer_idx))
            if self.layer_budget_scheduler
            else 1.0
        )

        eps = 1e-6
        target_logit = math.log((target_keep + eps) / (1.0 - target_keep + eps))
        logits = logits + target_logit

        keep_prob, keep_mask = LayerGate.sample_straight_through(
            logits,
            temperature=self.layer_skip_temperature,
            hard=self.layer_skip_hard,
        )

        skip_layer_compute = False
        if self.layer_skip_mode == "seq":
            skip_layer_compute = bool((keep_mask.mean() < 0.5).item())

        return skip_layer_compute, keep_prob, keep_mask, target_keep


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────
@node(
    type_id="transformer_encoder",
    name="Transformer Encoder",
    category="Encoder",
    outputs=["encoder"],
    color="bg-gradient-to-br from-green-700 to-green-800",
)
class TransformerEncoder(BaseTransformer):
    def __init__(
        self,
        input_size: int = 1,
        use_time_encoding: bool = False,
        model_type: str = "transformer",
        ct_patchtst: bool = False,
        ct_patch_len: int = 16,
        ct_patch_stride: int = 8,
        ct_patch_pad_end: bool = True,
        ct_patch_fuse: Literal["mean", "linear"] = "linear",
        **kwargs,
    ):
        self.model_type = model_type
        # Auto-configure based on model_type
        if model_type == "informer-like":
            use_time_encoding = kwargs.get("use_time_encoding", True)
            # Informers often use patching or specific attention,
            # but usually handled by kwargs or specific layers.

        self.use_time_encoding = use_time_encoding
        self.ct_patchtst = bool(ct_patchtst)
        self.ct_patch_len = int(ct_patch_len)
        self.ct_patch_stride = int(ct_patch_stride)
        self.ct_patch_pad_end = bool(ct_patch_pad_end)
        self.ct_patch_fuse = str(ct_patch_fuse)
        if self.ct_patch_fuse not in ("mean", "linear"):
            raise ValueError("ct_patch_fuse must be 'mean' or 'linear'")
        if self.ct_patchtst:
            # CT-PatchTST is an alternative encoder tokenization path.
            kwargs["patch_encoder"] = False
        super().__init__(input_size, **kwargs)
        self.input_size = input_size
        self.time_encoder = (
            InformerTimeEmbedding(self.d_model) if use_time_encoding else None
        )
        self.ct_patch_embed = (
            nn.Linear(self.ct_patch_len, self.d_model) if self.ct_patchtst else None
        )
        self.ct_channel_fuse = (
            nn.Linear(self.input_size * self.d_model, self.d_model)
            if (self.ct_patchtst and self.ct_patch_fuse == "linear")
            else None
        )

        # FIX: stash last patched mask + patch info to avoid decoder mismatch bugs
        self.last_memory_key_padding_mask: Optional[torch.Tensor] = None
        self.last_patch_info: Optional[PatchInfo] = None

    @staticmethod
    def _compute_ct_patch_pad(T: int, P: int, S: int) -> int:
        if T <= 0:
            return 0
        if T < P:
            return P - T
        n_patches = math.ceil((T - P) / S) + 1
        T_pad = (n_patches - 1) * S + P
        return max(0, T_pad - T)

    def _ct_patchify(self, src: torch.Tensor) -> Tuple[torch.Tensor, PatchInfo]:
        if self.ct_patch_embed is None:
            raise RuntimeError("CT-PatchTST is not enabled.")
        if src.dim() != 3:
            raise ValueError(f"CT-PatchTST expects [B,T,C], got {tuple(src.shape)}")

        B, T, C = src.shape
        if C != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {C}")

        x = src.transpose(1, 2).contiguous()  # [B,C,T]
        pad = (
            self._compute_ct_patch_pad(T, self.ct_patch_len, self.ct_patch_stride)
            if self.ct_patch_pad_end
            else 0
        )
        if pad > 0:
            x = F.pad(x, (0, pad))
        T_pad = x.size(-1)

        patches = x.unfold(
            dimension=2, size=self.ct_patch_len, step=self.ct_patch_stride
        ).contiguous()  # [B,C,Np,P]
        Np = patches.size(2)

        tok = self.ct_patch_embed(patches)  # [B,C,Np,D]
        if self.ct_patch_fuse == "mean":
            tokens = tok.mean(dim=1)  # [B,Np,D]
        else:
            if self.ct_channel_fuse is None:
                raise RuntimeError("ct_channel_fuse is not initialized.")
            tok = tok.permute(0, 2, 1, 3).reshape(B, Np, C * self.d_model)
            tokens = self.ct_channel_fuse(tok)  # [B,Np,D]

        info = PatchInfo(
            T_orig=T,
            T_pad=T_pad,
            n_patches=Np,
            patch_len=self.ct_patch_len,
            stride=self.ct_patch_stride,
        )
        return tokens, info

    def _make_layer(self, **kwargs) -> nn.Module:
        kwargs.pop("informer_like", None)
        return TransformerEncoderLayer(**kwargs)

    def forward(
        self,
        src: torch.Tensor,  # [B, T, C]
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B,T] bool
        time_features: Optional[torch.Tensor] = None,  # [B, T, F_tf]
    ) -> torch.Tensor:
        B, T, C = src.shape
        if C != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {C}")
        if T > self.max_seq_len and (not self.patch_encoder) and (not self.ct_patchtst):
            raise ValueError(f"Sequence length {T} exceeds max {self.max_seq_len}")

        self.layer_skip_aux.zero_()

        patch_info: Optional[PatchInfo] = None
        if self.ct_patchtst:
            x, patch_info = self._ct_patchify(src)  # [B, Np, D]
            if x.shape[1] > self.max_seq_len:
                raise ValueError(
                    f"Encoder CT-patch token length {x.shape[1]} exceeds max_seq_len={self.max_seq_len}. "
                    f"Increase max_seq_len or adjust ct_patch_len/ct_patch_stride."
                )
            src_key_padding_mask = patchify_padding_mask(
                src_key_padding_mask,
                T=T,
                patch_len=self.ct_patch_len,
                stride=self.ct_patch_stride,
                pad_end=self.ct_patch_pad_end,
            )
        else:
            x = self.input_adapter(src)  # [B, T, D]
        if self.patch_encoder:
            x, patch_info = self.patcher(x)  # [B, Np, D]
            if x.shape[1] > self.max_seq_len:
                raise ValueError(
                    f"Encoder patch token length {x.shape[1]} exceeds max_seq_len={self.max_seq_len}. "
                    f"Increase max_seq_len or adjust patch_len/patch_stride."
                )
            src_key_padding_mask = patchify_padding_mask(
                src_key_padding_mask,
                T=T,
                patch_len=self.patch_len,
                stride=self.patch_stride,
                pad_end=self.patch_pad_end,
            )

        # FIX: stash for downstream decoder
        self.last_memory_key_padding_mask = src_key_padding_mask
        self.last_patch_info = patch_info

        x = self.pos_encoder(x)

        # Time encoding only for timestep-space by default (encoder patching skips it)
        if (
            (not self.patch_encoder)
            and (not self.ct_patchtst)
            and (self.time_encoder is not None)
            and (time_features is not None)
        ):
            time_emb = self.time_encoder(time_features)  # [B, T, D]
            if time_emb.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"Time features shape {time_emb.shape} incompatible with input {x.shape}"
                )
            x = x + time_emb

        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)

        # mHC streams
        streams: Optional[torch.Tensor] = (
            mhc_init_streams(x, self.mhc_n_streams) if self.use_mhc else None
        )

        # safest: checkpoint only when not using mHC (streams complicate checkpoint)
        use_ckpt = (
            self.training and self.use_gradient_checkpointing and (not self.use_mhc)
        )
        invoke = _ModelLayerInvokeStrategy(owner=self, use_checkpoint=use_ckpt)
        runtime_budget = self._get_runtime_budget()

        used_indices: List[int] = []

        for i in range(self.num_layers):
            layer, budget, skip_layer_compute, keep_mask = self._prepare_layer_step(
                i, x, runtime_budget=runtime_budget
            )

            if skip_layer_compute and (self.layer_skip_mode == "seq"):
                continue

            used_indices.append(i)

            if not use_ckpt:
                x_before = x

            x, streams = invoke.run_encoder_layer(
                layer=layer,
                x=x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                budget=budget,
                streams=streams,
            )

            if not use_ckpt:
                x = self._maybe_apply_token_skip_mix(x, x_before, keep_mask)

        # FIX: aggregate only over executed layers
        self._finalize_layer_stack(used_indices)

        x = self.final_norm(x)
        # IMPORTANT: if patch_encoder=True, we DO NOT unpatch here.
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────
@node(
    type_id="transformer_decoder",
    name="Transformer Decoder",
    category="Decoder",
    outputs=["decoder"],
    color="bg-gradient-to-br from-purple-700 to-purple-800",
)
class TransformerDecoder(BaseTransformer):
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        label_len: int = 0,
        informer_like: bool = True,
        use_time_encoding: bool = False,
        model_type: str = "transformer",
        **kwargs,
    ):
        self.model_type = model_type
        # Auto-configure for informer-like
        if model_type == "informer-like":
            informer_like = True
            use_time_encoding = True
            if label_len == 0:
                # Default label_len for Informer is often half of target_len (implicit)
                pass

        self.output_size = output_size
        self.label_len = label_len
        self.informer_like = informer_like
        self.use_time_encoding = use_time_encoding
        super().__init__(input_size, informer_like=informer_like, **kwargs)

        self.time_encoder = (
            InformerTimeEmbedding(self.d_model) if use_time_encoding else None
        )

        self.output_projection = (
            nn.Identity()
            if output_size == self.d_model
            else nn.Linear(self.d_model, output_size)
        )

    def _make_layer(self, **kwargs) -> nn.Module:
        return TransformerDecoderLayer(**kwargs)

    def _create_informer_padding_mask(
        self, B: int, T: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if not self.informer_like or self.label_len >= T:
            return None
        label_len = max(0, min(self.label_len, T))
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        mask[:, label_len:] = True
        return mask

    def _infer_mtp_num_heads(self) -> int:
        n = 0
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            ff = getattr(layer, "feed_forward", None)
            block = getattr(ff, "block", None) if ff is not None else None
            n_i = int(getattr(block, "mtp_num_heads", 0) or 0)
            n = max(n, n_i)
        return n

    def _prepare_mtp_base(self, base: torch.Tensor, seq_len: int) -> torch.Tensor:
        if base.dim() != 3:
            raise ValueError(
                f"mtp_targets base must be [B,T,F] before shifting, got {tuple(base.shape)}"
            )
        if base.size(1) != seq_len:
            raise ValueError(
                f"mtp_targets length {base.size(1)} must match decoder length {seq_len}"
            )
        if base.size(-1) != self.d_model:
            in_features = int(getattr(self.input_adapter, "in_features", -1))
            if base.size(-1) != in_features:
                raise ValueError(
                    f"mtp_targets last dim {base.size(-1)} must be decoder input {in_features} or d_model {self.d_model}"
                )
            base = self.input_adapter(base)
        return base

    def _build_shifted_mtp_targets(
        self, base: torch.Tensor, n_heads: int
    ) -> torch.Tensor:
        # base: [B,T,D] -> shifted targets [B,T,H,D], where head h predicts t+(h+1)
        if n_heads <= 0:
            return base.new_zeros(base.size(0), base.size(1), 0, base.size(2))
        B, T, D = base.shape
        tgt = base.new_zeros(B, T, n_heads, D)
        for h in range(n_heads):
            s = h + 1
            if s >= T:
                continue
            tgt[:, : T - s, h, :] = base[:, s:, :]
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,  # [B, T_tgt, C_tgt]
        memory: torch.Tensor,  # [B, T_src_or_Np, D]
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict] = None,
        return_incremental_state: bool = False,
        time_features: Optional[torch.Tensor] = None,
        mtp_targets: Optional[torch.Tensor] = None,  # [B,T,F] or [B,T,H,D]
    ):
        B, T, _ = tgt.shape
        device = tgt.device

        self.layer_skip_aux.zero_()

        if self.patch_decoder and incremental_state is not None:
            raise RuntimeError(
                "patch_decoder=True is not compatible with incremental_state/KV-cached decoding. "
                "Set patch_decoder=False (recommended) for autoregressive decoding."
            )
        if self.use_layer_skipping and incremental_state is not None:
            raise RuntimeError(
                "Dynamic layer skipping is not supported with incremental_state/KV-cached decoding. "
                "Disable layer skipping for autoregressive decoding (forward_one_step)."
            )

        # FIX: Validate memory + mask alignment early (patch_encoder common pitfall)
        if memory_key_padding_mask is not None:
            if (
                memory_key_padding_mask.shape[0] != memory.shape[0]
                or memory_key_padding_mask.shape[1] != memory.shape[1]
            ):
                raise ValueError(
                    f"memory_key_padding_mask shape {tuple(memory_key_padding_mask.shape)} must match memory [B,Tm]=[{memory.shape[0]},{memory.shape[1]}]"
                )

        x = self.input_adapter(tgt)  # [B, T, D]

        patch_info: Optional[PatchInfo] = None
        if self.patch_decoder:
            x, patch_info = self.patcher(x)  # [B, Np, D]
            if x.shape[1] > self.max_seq_len:
                raise ValueError(
                    f"Decoder patch token length {x.shape[1]} exceeds max_seq_len={self.max_seq_len}. "
                    f"Increase max_seq_len or adjust patch_len/patch_stride."
                )
            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = patchify_padding_mask(
                    tgt_key_padding_mask,
                    T=T,
                    patch_len=self.patch_len,
                    stride=self.patch_stride,
                    pad_end=self.patch_pad_end,
                )

        x = self.pos_encoder(x)

        # Time encoding only when decoder is timestep-level
        if (
            (not self.patch_decoder)
            and (self.time_encoder is not None)
            and (time_features is not None)
        ):
            time_emb = self.time_encoder(time_features)  # [B, T, D]
            if time_emb.shape[:2] == x.shape[:2]:
                x = x + time_emb

        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)

        # Optional MTP target prep for MoE FFNs in decoder layers.
        # Accepted:
        # - [B, T, F] (base sequence in decoder input dim or d_model)
        # - [B, T, H, D] (already shifted/aligned)
        layer_mtp_targets = None
        mtp_heads = self._infer_mtp_num_heads()
        if self.training and (mtp_heads > 0) and (mtp_targets is not None):
            if mtp_targets.dim() == 4:
                if (
                    mtp_targets.size(0) != x.size(0)
                    or mtp_targets.size(1) != x.size(1)
                    or mtp_targets.size(-1) != self.d_model
                ):
                    raise ValueError(
                        f"mtp_targets [B,T,H,D] must align with decoder x [B={x.size(0)},T={x.size(1)},D={self.d_model}], got {tuple(mtp_targets.shape)}"
                    )
                layer_mtp_targets = mtp_targets
            elif mtp_targets.dim() == 3:
                base = self._prepare_mtp_base(mtp_targets, seq_len=x.size(1))
                layer_mtp_targets = self._build_shifted_mtp_targets(base, mtp_heads)
            else:
                raise ValueError(
                    f"mtp_targets must be [B,T,F] or [B,T,H,D], got {tuple(mtp_targets.shape)}"
                )

        if tgt_mask is None:
            L = x.shape[1]
            tgt_mask = self._generate_causal_mask(L, device, dtype=x.dtype)

        if tgt_key_padding_mask is None and (not self.patch_decoder):
            tgt_key_padding_mask = self._create_informer_padding_mask(B, T, device)

        layer_states: List[Optional[dict]] = (
            incremental_state.get("layers", [None] * self.num_layers)
            if incremental_state
            else [None] * self.num_layers
        )

        streams: Optional[torch.Tensor] = None
        if self.use_mhc:
            if incremental_state is not None:
                raise RuntimeError(
                    "Decoder mHC sketch does not support incremental_state/KV-cached decoding. "
                    "Disable mHC for autoregressive decoding."
                )
            streams = mhc_init_streams(x, self.mhc_n_streams)

        use_ckpt = (
            self.training and self.use_gradient_checkpointing and (not self.use_mhc)
        )
        invoke = _ModelLayerInvokeStrategy(owner=self, use_checkpoint=use_ckpt)
        runtime_budget = self._get_runtime_budget()

        used_indices: List[int] = []

        for i in range(self.num_layers):
            layer, budget, skip_layer_compute, keep_mask = self._prepare_layer_step(
                i, x, runtime_budget=runtime_budget
            )

            prev_state = layer_states[i - 1] if i > 0 else None

            if skip_layer_compute and (self.layer_skip_mode == "seq"):
                continue

            used_indices.append(i)

            if not use_ckpt:
                x_before = x

            x, layer_states[i], streams = invoke.run_decoder_layer(
                layer=layer,
                x=x,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                layer_state=layer_states[i],
                prev_state=prev_state,
                budget=budget,
                streams=streams,
                mtp_targets=layer_mtp_targets,
            )

            if not use_ckpt:
                x = self._maybe_apply_token_skip_mix(x, x_before, keep_mask)

        # FIX: aggregate only over executed layers
        self._finalize_layer_stack(used_indices)

        x = self.final_norm(x)

        if self.patch_decoder:
            assert patch_info is not None
            x = self.unpatcher(x, patch_info)  # [B, T, D]

        out = self.output_projection(x)  # [B, T, output_size]

        if return_incremental_state:
            if incremental_state is None:
                incremental_state = {}
            incremental_state["layers"] = layer_states
            return out, incremental_state

        return out

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: Optional[Dict] = None,
        time_features: Optional[torch.Tensor] = None,
    ):
        if self.patch_decoder:
            raise RuntimeError(
                "forward_one_step requires patch_decoder=False (recommended default)."
            )
        if self.use_layer_skipping:
            raise RuntimeError(
                "forward_one_step does not support dynamic layer skipping with KV caching. "
                "Disable layer skipping for autoregressive decoding."
            )
        return self.forward(
            tgt,
            memory,
            incremental_state=incremental_state or {},
            return_incremental_state=True,
            time_features=time_features,
        )
