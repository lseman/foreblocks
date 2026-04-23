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
#   - Each residual block dynamically reads from the stream, applies the sublayer once,
#     then writes the update back with Sinkhorn-constrained stream mixing.
#   - Collapses streams back to [B, T, D] at output for backward compatibility.
#   - For simplicity and safety, GateSkip remains applied on the collapsed stream only.
#   - Decoder mHC: disabled for incremental_state (KV-cached decoding).
#
# NEW: Mixture-of-Depths (MoD) / Dynamic Layer Skipping
#   - MoDRouter predicts a scalar router score per token.
#   - MoDBudgetScheduler provides per-layer routed capacity as a keep-rate.
#   - The top-k routed tokens are packed, processed by the block, then scattered
#     back with a residual bypass for the rest.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from .attention.utils.residuals import AttentionResidual
from .attention.utils.residuals import normalize_attention_residual_mode
from .embeddings import LearnablePositionalEncoding
from .embeddings import PositionalEncoding
from .ff import FeedForwardBlock
from .fusions import fused_dropout_add  # fused helpers
from .fusions import fused_dropout_add_norm
from .fusions import fused_dropout_gateskip_norm
from .fusions import get_dropout_p
from .mhc import MHCHyperConnection
from .mhc import mhc_apply_norm_streamwise
from .mhc import mhc_collapse_streams
from .mhc import mhc_init_streams
from .norms import RMSNorm
from .norms import create_norm_layer
from .patching import PatchDetokenizer
from .patching import PatchTokenizer
from .patching import patchify_padding_mask
from .skip.gateskip import BudgetScheduler
from .skip.gateskip import ResidualGate
from .skip.gateskip import apply_skip_to_kv
from .skip.mod import MoDBudgetScheduler
from .skip.mod import MoDRouter
from .skip.mod import mod_capacity
from .skip.mod import mod_routed_indices
from .skip.mod import mod_router_aux_loss
from .skip.mod import mod_topk_mask


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
        active_mask: Optional[torch.Tensor] = None,
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
                active_mask=active_mask,
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
        active_mask: Optional[torch.Tensor] = None,
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
            active_mask=active_mask,
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
        hyper_conn: "MHCHyperConnection",
        core_fn: Callable[[torch.Tensor], torch.Tensor],  # x_in -> upd
    ) -> torch.Tensor:
        x_in, maps = hyper_conn.pre_aggregate(streams)
        if normw.strategy in ("pre_norm", "sandwich_norm"):
            x_in = normw.norm(x_in)

        upd = core_fn(x_in)
        upd = normw.dropout(upd)
        streams = hyper_conn.combine(streams, upd, maps=maps)

        if normw.strategy in ("post_norm", "sandwich_norm"):
            streams = mhc_apply_norm_streamwise(normw.norm, streams)
        return streams


def _init_attention_residual_state(x: torch.Tensor, mode: str, block_size: int) -> dict:
    state = {"mode": mode, "current": x}
    if mode == "full":
        state["history"] = [x]
        state["running_sum"] = x
        return state

    state["blocks"] = [x]
    state["partial"] = None
    state["block_size"] = int(block_size)
    state["sub_layers_in_block"] = 0
    return state


def _attention_residual_input(
    carrier: torch.Tensor, state: Optional[dict], module: Optional[nn.Module]
) -> torch.Tensor:
    if state is None or module is None:
        return carrier
    if state["mode"] == "full":
        return module(state["history"])
    return module(state["blocks"], state["partial"])


def _append_attention_residual_update(
    state: Optional[dict], update: torch.Tensor
) -> None:
    if state is None:
        return

    if state["mode"] == "full":
        state["history"].append(update)
        state["running_sum"] = state["running_sum"] + update
        state["current"] = state["running_sum"]
        return

    partial = update if state["partial"] is None else state["partial"] + update
    state["partial"] = partial
    state["sub_layers_in_block"] += 1
    if state["sub_layers_in_block"] >= state["block_size"]:
        state["blocks"].append(partial)
        state["partial"] = None
        state["sub_layers_in_block"] = 0
    state["current"] = partial


def _attention_residual_values(state: dict) -> List[torch.Tensor]:
    if state["mode"] == "full":
        return list(state["history"])

    values = list(state["blocks"])
    if state["partial"] is not None:
        values.append(state["partial"])
    return values


def _gateskip_active_mask_from_padding(
    padding_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if padding_mask is None:
        return None
    return ~padding_mask.to(dtype=torch.bool)


def _patchify_gateskip_active_mask(
    active_mask: Optional[torch.Tensor],
    *,
    T: int,
    patch_len: int,
    stride: int,
    pad_end: bool,
) -> Optional[torch.Tensor]:
    if active_mask is None:
        return None
    pad_mask = ~active_mask.to(dtype=torch.bool)
    patch_pad_mask = patchify_padding_mask(
        pad_mask,
        T=T,
        patch_len=patch_len,
        stride=stride,
        pad_end=pad_end,
    )
    if patch_pad_mask is None:
        return None
    return ~patch_pad_mask


def _gather_sequence_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return x[:, :0]
    view_shape = [indices.size(0), indices.size(1)] + [1] * (x.dim() - 2)
    expand_shape = [indices.size(0), indices.size(1)] + list(x.shape[2:])
    gather_idx = indices.view(*view_shape).expand(*expand_shape)
    return x.gather(1, gather_idx)


def _gather_padding_mask(
    mask: Optional[torch.Tensor],
    indices: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    if indices.numel() == 0:
        return slot_mask.new_ones(slot_mask.shape)
    if mask is None:
        return ~slot_mask
    gathered = mask.to(dtype=torch.bool).gather(1, indices)
    return gathered | (~slot_mask)


def _gather_square_mask(
    mask: Optional[torch.Tensor],
    indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if indices.numel() == 0:
        shape = (indices.size(0), 0, 0)
        return mask.new_zeros(shape)
    B, C = indices.shape
    if mask.dim() == 2:
        base = mask.unsqueeze(0).expand(B, -1, -1)
        gathered = base.gather(1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1)))
        return gathered.gather(2, indices.unsqueeze(1).expand(-1, C, -1))
    if mask.dim() == 3:
        if mask.size(0) == 1 and B > 1:
            base = mask.expand(B, -1, -1)
        elif mask.size(0) != B:
            raise ValueError(
                f"Batch attention mask shape {tuple(mask.shape)} incompatible with "
                f"batch size {B}"
            )
        else:
            base = mask
        gathered = base.gather(1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1)))
        return gathered.gather(2, indices.unsqueeze(1).expand(-1, C, -1))
    if mask.dim() == 4:
        if mask.size(0) == 1 and B > 1:
            base = mask.expand(B, -1, -1, -1)
        elif mask.size(0) != B:
            raise ValueError(
                f"Batch attention mask shape {tuple(mask.shape)} incompatible with "
                f"batch size {B}"
            )
        else:
            base = mask
        gathered = base.gather(
            2, indices[:, None, :, None].expand(-1, base.size(1), -1, base.size(-1))
        )
        return gathered.gather(
            3, indices[:, None, None, :].expand(-1, base.size(1), C, -1)
        )
    raise ValueError(
        f"Unsupported square attention mask shape {tuple(mask.shape)} for MoD routing"
    )


def _gather_query_mask(
    mask: Optional[torch.Tensor],
    indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if indices.numel() == 0:
        shape = (indices.size(0), 0, mask.size(-1))
        return mask.new_zeros(shape)
    B = indices.size(0)
    if mask.dim() == 2:
        base = mask.unsqueeze(0).expand(B, -1, -1)
        return base.gather(1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1)))
    if mask.dim() == 3:
        if mask.size(0) == 1 and B > 1:
            base = mask.expand(B, -1, -1)
        elif mask.size(0) != B:
            raise ValueError(
                f"Batch attention mask shape {tuple(mask.shape)} incompatible with "
                f"batch size {B}"
            )
        else:
            base = mask
        return base.gather(1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1)))
    if mask.dim() == 4:
        if mask.size(0) == 1 and B > 1:
            base = mask.expand(B, -1, -1, -1)
        elif mask.size(0) != B:
            raise ValueError(
                f"Batch attention mask shape {tuple(mask.shape)} incompatible with "
                f"batch size {B}"
            )
        else:
            base = mask
        return base.gather(
            2, indices[:, None, :, None].expand(-1, base.size(1), -1, base.size(-1))
        )
    raise ValueError(
        f"Unsupported query attention mask shape {tuple(mask.shape)} for MoD routing"
    )


def _scatter_mixture_of_depths_output(
    x_base: torch.Tensor,
    x_routed_in: torch.Tensor,
    x_routed_out: torch.Tensor,
    indices: torch.Tensor,
    slot_mask: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    if indices.numel() == 0:
        return x_base

    routed_delta = x_routed_out - x_routed_in
    routed_scale = router_logits.gather(1, indices).unsqueeze(-1)
    routed_update = routed_scale * routed_delta

    out = x_base.clone()
    for b in range(out.size(0)):
        valid = slot_mask[b]
        if not bool(valid.any()):
            continue
        idx = indices[b, valid]
        out[b, idx] = x_base[b, idx] + routed_update[b, valid]
    return out


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
        mhc_core: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        hyper_conn: Optional[nn.Module] = None,
        prev_layer_state: Optional[dict] = None,
        kv_key: Optional[str] = None,
        active_mask: Optional[torch.Tensor] = None,
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
                active_mask=active_mask,
            )
            return updated_kv, skip_mask

        if self.streams is None or mhc_core is None or hyper_conn is None:
            raise RuntimeError(
                "mHC strategy requires streams, mhc_core, and hyper_conn."
            )
        self.streams = self.owner._mhc_run_block(
            streams=self.streams,
            normw=normw,
            hyper_conn=hyper_conn,
            core_fn=mhc_core,
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
        attention_residual_state: Optional[dict],
        gateskip_active_mask: Optional[torch.Tensor],
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
                    mhc_collapse=self.owner.mhc_collapse,
                    gateskip_active_mask=gateskip_active_mask,
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
            mhc_collapse=self.owner.mhc_collapse,
            attention_residual_state=attention_residual_state,
            gateskip_active_mask=gateskip_active_mask,
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
        attention_residual_state: Optional[dict],
        gateskip_active_mask: Optional[torch.Tensor],
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
                    mhc_collapse=self.owner.mhc_collapse,
                    mtp_targets=mtp_targets,
                    gateskip_active_mask=gateskip_active_mask,
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
            mhc_collapse=self.owner.mhc_collapse,
            mtp_targets=mtp_targets,
            attention_residual_state=attention_residual_state,
            gateskip_active_mask=gateskip_active_mask,
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
        mhc_collapse: str = "first",
        moe_use_latent: bool = False,
        moe_latent_dim: Optional[int] = None,
        moe_latent_d_ff: Optional[int] = None,
        use_attention_matching_compaction: bool = False,
        attention_matching_keep_ratio: float = 0.25,
        attention_matching_trigger_len: int = 512,
        attention_matching_min_keep: int = 64,
        attention_matching_query_budget: int = 64,
        attention_matching_force_single_step: bool = False,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.use_gateskip = use_gateskip
        self.gate_budget = gate_budget
        self.gate_lambda = gate_lambda
        self.d_model = int(d_model)

        # mHC knobs (layer-level)
        self.use_mhc = bool(use_mhc)
        self.mhc_n_streams = int(mhc_n_streams)
        self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)
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
            moe_use_latent=moe_use_latent,
            moe_latent_dim=moe_latent_dim,
            moe_latent_d_ff=moe_latent_d_ff,
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

    def _new_mhc_connection(self) -> MHCHyperConnection:
        conn = MHCHyperConnection(
            d_model=self.d_model,
            n_streams=self.mhc_n_streams,
            sinkhorn_iters=self.mhc_sinkhorn_iters,
        )
        ref = next(self.parameters(), None)
        if ref is not None:
            conn = conn.to(device=ref.device, dtype=ref.dtype)
        return conn

    def _apply_runtime_mhc_overrides(
        self,
        *,
        use_mhc: Optional[bool],
        mhc_n_streams: Optional[int],
        mhc_sinkhorn_iters: Optional[int],
        mhc_collapse: Optional[str],
    ) -> None:
        """Apply per-call mHC overrides (for backward compatibility)."""
        if use_mhc is not None:
            self.use_mhc = bool(use_mhc)
        if mhc_n_streams is not None:
            self.mhc_n_streams = int(mhc_n_streams)
        if mhc_sinkhorn_iters is not None:
            self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)
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

    def _run_attnres_core(
        self,
        x: torch.Tensor,
        normw: NormWrapper,
        core_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[dict]]],
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        x_in = normw.norm(x) if normw.strategy in ("pre_norm", "sandwich_norm") else x
        out, updated = core_fn(x_in)
        out = normw.dropout(out)
        if normw.strategy in ("post_norm", "sandwich_norm"):
            out = normw.norm(out)
        return out, updated


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
            "gated_delta",
            "hybrid_gdn",
            "gdn_3to1",
        ] = "standard",
        # mHC toggles
        use_mhc: bool = False,
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_collapse: str = "first",  # "first" or "mean"
        # PatchTST-style patching (best default: patch encoder only)
        patch_encoder: bool = True,
        patch_decoder: bool = False,
        patch_len: int = 16,
        patch_stride: int = 8,
        patch_pad_end: bool = True,
        # Mixture-of-Depths
        use_mod: bool = False,
        mod_mode: Literal["token", "seq"] = "token",
        mod_lambda: float = 0.05,
        mod_budget_scheduler: Optional[MoDBudgetScheduler] = None,
        # Global scaling for MoE aux loss (FIX)
        moe_aux_lambda: float = 1.0,
        use_attention_residual: bool = False,
        attn_residual_type: str = "full",
        attention_residual_block_size: int = 8,
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
        if attention_mode == "standard" and att_type in {
            "linear",
            "sype",
            "kimi",
            "gated_delta",
        }:
            attention_mode = att_type
        self.attention_mode = attention_mode
        self.att_type = att_type
        self.budget_scheduler: Optional[BudgetScheduler] = None

        # MoE aux scaling (FIX)
        self.moe_aux_lambda = float(moe_aux_lambda)

        self.use_attention_residual = bool(use_attention_residual)
        self.attention_residual_mode = normalize_attention_residual_mode(
            attn_residual_type
        )
        self.attention_residual_block_size = int(attention_residual_block_size)
        if self.attention_residual_block_size <= 0:
            raise ValueError("attention_residual_block_size must be > 0")
        self.output_attention_residual = (
            AttentionResidual(self.d_model) if self.use_attention_residual else None
        )

        # mHC model-level
        self.use_mhc = bool(use_mhc)
        self.mhc_n_streams = int(mhc_n_streams)
        self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)
        self.mhc_collapse = str(mhc_collapse)

        # PatchTST-style patching knobs (subclasses decide whether to apply)
        self.patch_encoder = bool(patch_encoder)
        self.patch_decoder = bool(patch_decoder)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.patch_pad_end = bool(patch_pad_end)

        # Mixture-of-Depths knobs
        self.use_mod = bool(use_mod)
        self.mod_mode = str(mod_mode)
        self.mod_lambda = float(mod_lambda)
        self.mod_budget_scheduler = mod_budget_scheduler

        # Modules
        self.patcher = PatchTokenizer(
            self.d_model, self.patch_len, self.patch_stride, pad_end=self.patch_pad_end
        )
        self.unpatcher = PatchDetokenizer(
            self.d_model, self.patch_len, self.patch_stride
        )

        self.input_adapter = nn.Linear(input_size, self.d_model)
        if pos_encoder is not None:
            self.pos_encoder = pos_encoder
        elif self.attention_mode == "sype" or self.att_type == "sype":
            self.pos_encoder = LearnablePositionalEncoding(
                self.d_model,
                max_len=max_seq_len,
                dropout=dropout,
                scale_strategy="fixed",
                scale_value=pos_encoding_scale,
                use_layer_norm=False,
            )
        else:
            self.pos_encoder = PositionalEncoding(
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
                mhc_collapse=self.mhc_collapse,
                use_attention_residual=self.use_attention_residual,
                attn_residual_type=self.attention_residual_mode,
                attention_residual_block_size=self.attention_residual_block_size,
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
                    mhc_collapse=self.mhc_collapse,
                    use_attention_residual=self.use_attention_residual,
                    attn_residual_type=self.attention_residual_mode,
                    attention_residual_block_size=self.attention_residual_block_size,
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
        self.mod_routers = nn.ModuleList(
            [
                MoDRouter(
                    d_model=self.d_model,
                    mode=self.mod_mode,
                    hidden=0,
                    init_bias=2.0,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("mod_aux_loss", torch.tensor(0.0), persistent=False)
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
        if mode == "gated_delta":
            return "gated_delta"
        if mode in ("hybrid_gdn", "gdn_hybrid"):
            return "gated_delta" if layer_idx < (self.num_layers - 1) else "standard"
        if mode == "gdn_3to1":
            return "gated_delta" if (layer_idx % 4) < 3 else "standard"
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
        mhc_collapse: str,
        use_attention_residual: bool,
        attn_residual_type: str,
        attention_residual_block_size: int,
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
            "mhc_collapse": mhc_collapse,
            "use_attention_residual": use_attention_residual,
            "attn_residual_type": attn_residual_type,
            "attention_residual_block_size": attention_residual_block_size,
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
        collapse: Optional[str] = None,
    ) -> None:
        if n_streams is not None:
            self.mhc_n_streams = int(n_streams)
        if sinkhorn_iters is not None:
            self.mhc_sinkhorn_iters = int(sinkhorn_iters)
        if collapse is not None:
            self.mhc_collapse = str(collapse)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_mhc_attrs(layer)

    # ---- Mixture-of-Depths setters --------------------------------------------
    def set_use_mod(self, flag: bool) -> None:
        self.use_mod = bool(flag)

    def set_mod_budget_scheduler(self, scheduler: MoDBudgetScheduler) -> None:
        self.mod_budget_scheduler = scheduler

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
            f"attnres={self.use_attention_residual}/{self.attention_residual_mode}"
            f"/block={self.attention_residual_block_size} | "
            f"mhc={self.use_mhc} streams={self.mhc_n_streams} collapse={self.mhc_collapse} | "
            f"patch(enc={self.patch_encoder},dec={self.patch_decoder},len={self.patch_len},stride={self.patch_stride}) | "
            f"mod={self.use_mod}/{self.mod_mode} | "
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
        # Add MoD auxiliary after layer aggregation
        self.aux_loss = self.aux_loss + self.mod_aux_loss

    @staticmethod
    def _run_with_checkpoint(fn, *inputs: torch.Tensor, use_checkpoint: bool):
        if not use_checkpoint:
            return fn(*inputs)
        return torch.utils.checkpoint.checkpoint(fn, *inputs, use_reentrant=False)

    def _get_runtime_budget(self) -> Optional[float]:
        if self.training and (self.budget_scheduler is not None):
            return self.budget_scheduler.get_budget()
        return self.gate_budget

    def _validate_attention_residual_runtime(self) -> None:
        if not self.use_attention_residual:
            return
        if self.use_gateskip:
            raise RuntimeError(
                "Paper-style Attention Residuals replace the residual path and are "
                "not compatible with GateSkip in this implementation."
            )
        if self.use_mhc:
            raise RuntimeError(
                "Paper-style Attention Residuals are not wired for mHC stream mixing."
            )
        if self.use_mod:
            raise RuntimeError(
                "Paper-style Attention Residuals are not wired for Mixture-of-Depths."
            )

    def _init_attention_residual_state(self, x: torch.Tensor) -> Optional[dict]:
        if not self.use_attention_residual:
            return None
        return _init_attention_residual_state(
            x, self.attention_residual_mode, self.attention_residual_block_size
        )

    def _finalize_attention_residual_output(
        self, state: Optional[dict], fallback: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_attention_residual or state is None:
            return fallback
        if self.output_attention_residual is None:
            return fallback
        return self.output_attention_residual(_attention_residual_values(state))

    def _validate_mod_runtime(self) -> None:
        if not self.use_mod:
            return
        if self.mod_mode != "token":
            raise RuntimeError(
                "Mixture-of-Depths only supports token routing. Set mod_mode='token'."
            )
        if self.use_gateskip:
            raise RuntimeError(
                "Mixture-of-Depths is not wired together with GateSkip in this implementation."
            )
        if self.use_mhc:
            raise RuntimeError("Mixture-of-Depths is not wired for mHC stream mixing.")

    def _resolve_layer(self, layer_idx: int) -> nn.Module:
        layer = self._get_layer(layer_idx)
        if (self.shared_layer is not None) and hasattr(
            layer, "set_layer_attention_type"
        ):
            layer.set_layer_attention_type(self._get_layer_attention_type(layer_idx))
        return layer

    def _get_layer_keep_rate(self, layer_idx: int) -> float:
        if self.mod_budget_scheduler is None:
            return 1.0
        return float(self.mod_budget_scheduler.get_keep_rate(layer_idx))

    def _prepare_layer_routing(
        self,
        layer_idx: int,
        x: torch.Tensor,
        active_mask: Optional[torch.Tensor],
    ) -> Tuple[
        nn.Module,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        layer = self._resolve_layer(layer_idx)
        if not self.use_mod:
            return layer, None, None, None, None

        router_logits = self.mod_routers[layer_idx](x).squeeze(-1)
        keep_mask = mod_topk_mask(
            router_logits,
            self._get_layer_keep_rate(layer_idx),
            active_mask=active_mask,
        )
        if self.training and self.mod_lambda > 0:
            self.mod_aux_loss = self.mod_aux_loss + (
                self.mod_lambda
                * mod_router_aux_loss(
                    router_logits,
                    keep_mask,
                    active_mask=active_mask,
                )
            )

        capacity = mod_capacity(keep_mask)
        if capacity <= 0:
            return layer, router_logits, keep_mask, None, None
        indices, slot_mask = mod_routed_indices(keep_mask, capacity=capacity)
        return layer, router_logits, keep_mask, indices, slot_mask

    def _finalize_layer_stack(self, used_indices: List[int]) -> None:
        self._aggregate_aux_loss(used_indices)
        if self.training and self.budget_scheduler is not None:
            self.budget_scheduler.step()
        if self.training and self.mod_budget_scheduler is not None:
            self.mod_budget_scheduler.step()


__all__ = [
    "NormWrapper",
    "ResidualRunCfg",
    "ResidualBlockMixin",
    "MHCBlockMixin",
    "BaseTransformerLayer",
    "BaseTransformer",
]
