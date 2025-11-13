# -*- coding: utf-8 -*-
# Transformer with Toggleable GateSkip + Hybrid Attention (NO PatchTST)
# Adds: Kimi linear attention as a selectable option:
#   attention_mode ∈ {"standard","linear","hybrid","kimi","hybrid_kimi","kimi_3to1"}
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.node_spec import node

from .embeddings import InformerTimeEmbedding, PositionalEncoding
from .fusions import (  # <-- NEW: fused helpers
    fused_dropout_add,
    fused_dropout_add_norm,
    fused_dropout_gateskip_norm,
    get_dropout_p,
)
from .gateskip import *  # ResidualGate, gateskip_apply, apply_skip_to_kv, BudgetScheduler
from .kimi_att import KimiAttention  # <-- NEW
from .lin_att import LinearAttention
from .moe import *  # FeedForwardBlock (and optionally MoE)
from .multi_att import MultiAttention
from .norms import *  # create_norm_layer, RMSNorm, etc.


# ──────────────────────────────────────────────────────────────────────────────
# NormWrapper (holder to avoid accidental double residuals)
# ──────────────────────────────────────────────────────────────────────────────
class NormWrapper(nn.Module):
    """
    Holder for a norm layer + dropout; do NOT call forward.
    Access `.norm(x)` and `.dropout(y)` explicitly around your sublayer.
    """
    def __init__(
        self,
        d_model: int,
        norm_type: str = "rms",
        strategy: str = "pre_norm",
        dropout: float = 0.0,
        eps: float = 1e-5
    ):
        super().__init__()
        self.norm = create_norm_layer(norm_type, d_model, eps)
        self.strategy = strategy
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, *_, **__):
        raise RuntimeError(
            "NormWrapper.forward() disabled. Use `.norm(x)` then apply sublayer, "
            "then `.dropout(y)` and residual yourself."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Base layer
# ──────────────────────────────────────────────────────────────────────────────
class BaseTransformerLayer(nn.Module):
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
    ):
        super().__init__()
        self.use_moe = use_moe
        self.use_gateskip = use_gateskip
        self.gate_budget = gate_budget
        self.gate_lambda = gate_lambda
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

    def _reset_aux_loss(self):
        self.aux_loss.zero_()

    def _update_aux_loss(self, new_loss):
        if torch.is_tensor(new_loss):
            self.aux_loss += new_loss
        elif new_loss != 0:
            self.aux_loss += torch.tensor(new_loss, device=self.aux_loss.device)

    def _sync_gateskip(self, use_gateskip: bool, gate_budget: Optional[float], gate_lambda: float):
        self.use_gateskip = use_gateskip
        self.gate_budget = gate_budget
        self.gate_lambda = gate_lambda


# ──────────────────────────────────────────────────────────────────────────────
# Encoder layer
# ──────────────────────────────────────────────────────────────────────────────
class TransformerEncoderLayer(BaseTransformerLayer):
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
    ):
        super().__init__(
            d_model, dim_feedforward, dropout, activation,
            use_swiglu, use_moe, num_experts, top_k,
            use_gateskip=use_gateskip, gate_budget=gate_budget, gate_lambda=gate_lambda
        )
        # Choose attention type
        if layer_attention_type == "linear":
            self.self_attn = LinearAttention(
                d_model=d_model, n_heads=nhead, dropout=dropout,
            )
        elif layer_attention_type == "kimi":
            self.self_attn = KimiAttention(
                d_model=d_model, n_heads=nhead, dropout=dropout,
            )
        else:
            self.self_attn = MultiAttention(
                d_model=d_model, n_heads=nhead, dropout=dropout,
                attention_type=att_type, freq_modes=freq_modes,
            )
        self.attn_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)
        self.ff_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)
        self.gate_attn = ResidualGate(d_model)
        self.gate_ff = ResidualGate(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        gate_budget: Optional[float] = None,
        gate_lambda: Optional[float] = None,
        use_gateskip: Optional[bool] = None,
    ) -> torch.Tensor:
        self._reset_aux_loss()
        _use_gk = self.use_gateskip if use_gateskip is None else bool(use_gateskip)
        _budget = self.gate_budget if gate_budget is None else gate_budget
        _lambda = self.gate_lambda if gate_lambda is None else gate_lambda
        aux_l2_terms: list = []

        p_attn = get_dropout_p(getattr(self.attn_norm, "dropout", None))
        p_ff = get_dropout_p(getattr(self.ff_norm, "dropout", None))

        # Self-attention
        def attn_core(x):
            out, _, updated = self.self_attn(x, x, x, src_mask, src_key_padding_mask)
            return out, updated

        if self.attn_norm.strategy == "pre_norm":
            x_norm = self.attn_norm.norm(src)
            o, _ = attn_core(x_norm)
            # Pre-norm: no norm after residual; fuse dropout+residual if no GateSkip
            if _use_gk:
                if p_attn > 0.0 and self.training:
                    o = F.dropout(o, p=p_attn, training=True)
                src, _ = gateskip_apply(_use_gk, src, o, self.gate_attn, _budget, aux_l2_terms, _lambda)
            else:
                src = fused_dropout_add(src, o, p=p_attn, training=self.training)
        else:
            o, _ = attn_core(src)
            # Post-norm: fuse dropout → (gate or add) → norm
            if _use_gk:
                src, _ = fused_dropout_gateskip_norm(
                    residual=src,
                    update=o,
                    gate=self.gate_attn,
                    use_gateskip=_use_gk,
                    gate_budget=_budget,
                    aux_l2_terms=aux_l2_terms,
                    gate_lambda=_lambda,
                    norm_layer=self.attn_norm,
                    p=p_attn,
                    training=self.training,
                )
            else:
                src = fused_dropout_add_norm(
                    residual=src,
                    update=o,
                    norm_layer=self.attn_norm,
                    p=p_attn,
                    training=self.training,
                )

        # Feedforward
        def ff_core(x):
            if self.use_moe:
                out, aux = self.feed_forward(x, return_aux_loss=True)
                self._update_aux_loss(aux)
                return out
            return self.feed_forward(x)

        if self.ff_norm.strategy == "pre_norm":
            x_norm = self.ff_norm.norm(src)
            o = ff_core(x_norm)
            if _use_gk:
                if p_ff > 0.0 and self.training:
                    o = F.dropout(o, p=p_ff, training=True)
                src, _ = gateskip_apply(_use_gk, src, o, self.gate_ff, _budget, aux_l2_terms, _lambda)
            else:
                src = fused_dropout_add(src, o, p=p_ff, training=self.training)
        else:
            o = ff_core(src)
            if _use_gk:
                src, _ = fused_dropout_gateskip_norm(
                    residual=src,
                    update=o,
                    gate=self.gate_ff,
                    use_gateskip=_use_gk,
                    gate_budget=_budget,
                    aux_l2_terms=aux_l2_terms,
                    gate_lambda=_lambda,
                    norm_layer=self.ff_norm,
                    p=p_ff,
                    training=self.training,
                )
            else:
                src = fused_dropout_add_norm(
                    residual=src,
                    update=o,
                    norm_layer=self.ff_norm,
                    p=p_ff,
                    training=self.training,
                )

        if _use_gk and _lambda > 0 and aux_l2_terms:
            self._update_aux_loss(_lambda * torch.stack(aux_l2_terms).mean())
        return src


# ──────────────────────────────────────────────────────────────────────────────
# Decoder layer
# ──────────────────────────────────────────────────────────────────────────────
class TransformerDecoderLayer(BaseTransformerLayer):
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
    ):
        super().__init__(
            d_model, dim_feedforward, dropout, activation,
            use_swiglu, use_moe, num_experts, top_k,
            use_gateskip=use_gateskip, gate_budget=gate_budget, gate_lambda=gate_lambda
        )
        # Self-attention: can be linear, kimi, or standard
        if layer_attention_type == "linear":
            self.self_attn = LinearAttention(
                d_model=d_model, n_heads=nhead, dropout=dropout, cross_attention=False,
            )
        elif layer_attention_type == "kimi":
            self.self_attn = KimiAttention(
                d_model=d_model, n_heads=nhead, dropout=dropout, cross_attention=False,
            )
        else:
            self.self_attn = MultiAttention(
                d_model=d_model, n_heads=nhead, dropout=dropout,
                attention_type=att_type, freq_modes=freq_modes, cross_attention=False,
            )
        # Cross-attention: always standard
        self.cross_attn = MultiAttention(
            d_model=d_model, n_heads=nhead, dropout=dropout,
            attention_type=att_type, freq_modes=freq_modes, cross_attention=True,
        )
        self.is_causal = not informer_like
        self.self_attn_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)
        self.cross_attn_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)
        self.ff_norm = NormWrapper(d_model, custom_norm, norm_strategy, dropout, layer_norm_eps)
        self.gate_self = ResidualGate(d_model)
        self.gate_cross = ResidualGate(d_model)
        self.gate_ff = ResidualGate(d_model)

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
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        self._reset_aux_loss()
        _use_gk = self.use_gateskip if use_gateskip is None else bool(use_gateskip)
        _budget = self.gate_budget if gate_budget is None else gate_budget
        _lambda = self.gate_lambda if gate_lambda is None else gate_lambda

        # Standardized state keys
        state = {
            "self_attn": incremental_state.get("self_attn") if incremental_state else None,
            "cross_attn": incremental_state.get("cross_attn") if incremental_state else None,
        }
        aux_l2_terms: list = []

        p_self = get_dropout_p(getattr(self.self_attn_norm, "dropout", None))
        p_cross = get_dropout_p(getattr(self.cross_attn_norm, "dropout", None))
        p_ff = get_dropout_p(getattr(self.ff_norm, "dropout", None))

        # ---- Self-attention ----
        def self_core(x):
            out, _, updated = self.self_attn(
                x, x, x, tgt_mask, tgt_key_padding_mask,
                is_causal=self.is_causal, layer_state=state["self_attn"]
            )
            return out, updated

        if self.self_attn_norm.strategy == "pre_norm":
            x_norm = self.self_attn_norm.norm(tgt)
            o, updated_self = self_core(x_norm)
            if _use_gk:
                if p_self > 0.0 and self.training:
                    o = F.dropout(o, p=p_self, training=True)
                tgt, skip_mask = gateskip_apply(_use_gk, tgt, o, self.gate_self, _budget, aux_l2_terms, _lambda)
                if skip_mask is not None and updated_self is not None:
                    updated_self = apply_skip_to_kv(updated_self, skip_mask, prev_layer_state, 'self_attn')
                    state["self_attn"] = updated_self
            else:
                tgt = fused_dropout_add(tgt, o, p=p_self, training=self.training)
                if updated_self is not None:
                    state["self_attn"] = updated_self
        else:
            o, updated_self = self_core(tgt)
            if _use_gk:
                tgt, skip_mask = fused_dropout_gateskip_norm(
                    residual=tgt,
                    update=o,
                    gate=self.gate_self,
                    use_gateskip=_use_gk,
                    gate_budget=_budget,
                    aux_l2_terms=aux_l2_terms,
                    gate_lambda=_lambda,
                    norm_layer=self.self_attn_norm,
                    p=p_self,
                    training=self.training,
                )
                if skip_mask is not None and updated_self is not None:
                    updated_self = apply_skip_to_kv(updated_self, skip_mask, prev_layer_state, 'self_attn')
            else:
                tgt = fused_dropout_add_norm(
                    residual=tgt,
                    update=o,
                    norm_layer=self.self_attn_norm,
                    p=p_self,
                    training=self.training,
                )
            if updated_self is not None:
                state["self_attn"] = updated_self

        # ---- Cross-attention ----
        def cross_core(x):
            out, _, updated = self.cross_attn(
                x, memory, memory, memory_mask, memory_key_padding_mask,
                layer_state=state["cross_attn"]
            )
            return out, updated

        if self.cross_attn_norm.strategy == "pre_norm":
            x_norm = self.cross_attn_norm.norm(tgt)
            o, updated_cross = cross_core(x_norm)
            if _use_gk:
                if p_cross > 0.0 and self.training:
                    o = F.dropout(o, p=p_cross, training=True)
                tgt, skip_mask = gateskip_apply(_use_gk, tgt, o, self.gate_cross, _budget, aux_l2_terms, _lambda)
                if skip_mask is not None and updated_cross is not None:
                    updated_cross = apply_skip_to_kv(updated_cross, skip_mask, prev_layer_state, 'cross_attn')
                    state["cross_attn"] = updated_cross
            else:
                tgt = fused_dropout_add(tgt, o, p=p_cross, training=self.training)
                if updated_cross is not None:
                    state["cross_attn"] = updated_cross
        else:
            o, updated_cross = cross_core(tgt)
            if _use_gk:
                tgt, skip_mask = fused_dropout_gateskip_norm(
                    residual=tgt,
                    update=o,
                    gate=self.gate_cross,
                    use_gateskip=_use_gk,
                    gate_budget=_budget,
                    aux_l2_terms=aux_l2_terms,
                    gate_lambda=_lambda,
                    norm_layer=self.cross_attn_norm,
                    p=p_cross,
                    training=self.training,
                )
                if skip_mask is not None and updated_cross is not None:
                    updated_cross = apply_skip_to_kv(updated_cross, skip_mask, prev_layer_state, 'cross_attn')
            else:
                tgt = fused_dropout_add_norm(
                    residual=tgt,
                    update=o,
                    norm_layer=self.cross_attn_norm,
                    p=p_cross,
                    training=self.training,
                )
            if updated_cross is not None:
                state["cross_attn"] = updated_cross

        # ---- Feedforward ----
        def ff_core(x):
            if self.use_moe:
                out, aux = self.feed_forward(x, return_aux_loss=True)
                self._update_aux_loss(aux)
                return out
            return self.feed_forward(x)

        if self.ff_norm.strategy == "pre_norm":
            x_norm = self.ff_norm.norm(tgt)
            o = ff_core(x_norm)
            if _use_gk:
                if p_ff > 0.0 and self.training:
                    o = F.dropout(o, p=p_ff, training=True)
                tgt, _ = gateskip_apply(_use_gk, tgt, o, self.gate_ff, _budget, aux_l2_terms, _lambda)
            else:
                tgt = fused_dropout_add(tgt, o, p=p_ff, training=self.training)
        else:
            o = ff_core(tgt)
            if _use_gk:
                tgt, _ = fused_dropout_gateskip_norm(
                    residual=tgt,
                    update=o,
                    gate=self.gate_ff,
                    use_gateskip=_use_gk,
                    gate_budget=_budget,
                    aux_l2_terms=aux_l2_terms,
                    gate_lambda=_lambda,
                    norm_layer=self.ff_norm,
                    p=p_ff,
                    training=self.training,
                )
            else:
                tgt = fused_dropout_add_norm(
                    residual=tgt,
                    update=o,
                    norm_layer=self.ff_norm,
                    p=p_ff,
                    training=self.training,
                )

        if _use_gk and _lambda > 0 and aux_l2_terms:
            self._update_aux_loss(_lambda * torch.stack(aux_l2_terms).mean())

        ret_state = {"self_attn": state["self_attn"], "cross_attn": state["cross_attn"]}
        return tgt, ret_state


# ──────────────────────────────────────────────────────────────────────────────
# Base transformer (NO Patch/CI)
# ──────────────────────────────────────────────────────────────────────────────
class BaseTransformer(nn.Module, ABC):
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
            "standard", "linear", "hybrid", "kimi", "hybrid_kimi", "kimi_3to1"
        ] = "standard",
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
        self.attention_mode = attention_mode
        self.budget_scheduler: Optional[BudgetScheduler] = None

        self.input_adapter = nn.Linear(input_size, self.d_model)
        self.pos_encoder = pos_encoder or PositionalEncoding(
            self.d_model, max_len=max_seq_len, scale=pos_encoding_scale
        )

        # Build layers (per-layer attention type decided by mode)
        if share_layers:
            layer_attn_type = self._get_layer_attention_type(0)
            layer_kwargs = self._build_layer_kwargs(
                d_model, nhead, dim_feedforward, dropout, activation, att_type,
                norm_strategy, custom_norm, layer_norm_eps, use_swiglu, freq_modes,
                use_moe, num_experts, top_k, use_gateskip, gate_budget, gate_lambda,
                layer_attn_type, **kwargs
            )
            self.shared_layer = self._make_layer(**layer_kwargs)
            self.layers = None
        else:
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                layer_attn_type = self._get_layer_attention_type(i)
                layer_kwargs = self._build_layer_kwargs(
                    d_model, nhead, dim_feedforward, dropout, activation, att_type,
                    norm_strategy, custom_norm, layer_norm_eps, use_swiglu, freq_modes,
                    use_moe, num_experts, top_k, use_gateskip, gate_budget, gate_lambda,
                    layer_attn_type, **kwargs
                )
                self.layers.append(self._make_layer(**layer_kwargs))
            self.shared_layer = None

        self.final_norm = (
            create_norm_layer(custom_norm, d_model, layer_norm_eps)
            if use_final_norm else nn.Identity()
        )
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        self.apply(self._init_weights)

    def _get_layer_attention_type(self, layer_idx: int) -> str:
        mode = self.attention_mode
        if mode == "standard":
            return "standard"
        if mode == "linear":
            return "linear"
        if mode == "kimi":
            return "kimi"
        if mode in ("hybrid", "hybrid_linear"):
            # linear until last, then standard
            return "linear" if layer_idx < (self.num_layers - 1) else "standard"
        if mode in ("hybrid_kimi", "kimi_hybrid"):
            # kimi until last, then standard (Kimi Linear for early layers)
            return "kimi" if layer_idx < (self.num_layers - 1) else "standard"
        if mode == "kimi_3to1":
            # repeat (Kimi,Kimi,Kimi,Standard) pattern
            return "kimi" if (layer_idx % 4) < 3 else "standard"
        raise ValueError(f"Unknown attention_mode: {mode}")

    def _build_layer_kwargs(
        self, d_model, nhead, dim_feedforward, dropout, activation, att_type,
        norm_strategy, custom_norm, layer_norm_eps, use_swiglu, freq_modes,
        use_moe, num_experts, top_k, use_gateskip, gate_budget, gate_lambda,
        layer_attention_type, **kwargs
    ):
        return {
            "d_model": d_model, "nhead": nhead, "dim_feedforward": dim_feedforward,
            "dropout": dropout, "activation": activation, "att_type": att_type,
            "norm_strategy": norm_strategy, "custom_norm": custom_norm,
            "layer_norm_eps": layer_norm_eps, "use_swiglu": use_swiglu,
            "freq_modes": freq_modes, "use_moe": use_moe,
            "num_experts": num_experts, "top_k": top_k,
            "use_gateskip": use_gateskip, "gate_budget": gate_budget,
            "gate_lambda": gate_lambda,
            "layer_attention_type": layer_attention_type,
            **kwargs,
        }

    # ——— GateSkip runtime setters ———
    def set_use_gateskip(self, flag: bool):
        self.use_gateskip = bool(flag)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            if hasattr(layer, "_sync_gateskip"):
                layer._sync_gateskip(self.use_gateskip, self.gate_budget, self.gate_lambda)

    def set_gate_budget(self, budget: Optional[float]):
        self.gate_budget = budget
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            if hasattr(layer, "_sync_gateskip"):
                layer._sync_gateskip(self.use_gateskip, self.gate_budget, self.gate_lambda)

    def set_gate_lambda(self, lam: float):
        self.gate_lambda = float(lam)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            if hasattr(layer, "_sync_gateskip"):
                layer._sync_gateskip(self.use_gateskip, self.gate_budget, self.gate_lambda)

    def set_budget_scheduler(self, scheduler: BudgetScheduler):
        """Attach a BudgetScheduler for dynamic budgets during training."""
        self.budget_scheduler = scheduler

    @abstractmethod
    def _make_layer(self, **kwargs): ...

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _get_layer(self, idx: int) -> nn.Module:
        return self.shared_layer if self.layers is None else self.layers[idx]

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)

    def _aggregate_aux_loss(self, layers_used: int):
        total_aux = torch.tensor(0.0, device=self.aux_loss.device, dtype=self.aux_loss.dtype)
        for i in range(layers_used):
            layer = self._get_layer(i)
            if hasattr(layer, 'aux_loss'):
                total_aux = total_aux + layer.aux_loss
        self.aux_loss = total_aux / max(layers_used, 1)


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
    def __init__(self, input_size: int = 1, use_time_encoding: bool = False, **kwargs):
        self.use_time_encoding = use_time_encoding
        super().__init__(input_size, **kwargs)
        self.input_size = input_size
        if use_time_encoding:
            self.time_encoder = InformerTimeEmbedding(self.d_model)
        else:
            self.time_encoder = None

    def _make_layer(self, **kwargs):
        kwargs.pop("informer_like", None)
        return TransformerEncoderLayer(**kwargs)

    def forward(
        self,
        src: torch.Tensor,  # [B, T, C]
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,  # [B, T, F_tf]
    ) -> torch.Tensor:
        B, T, C = src.shape
        if C != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {C}")
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max {self.max_seq_len}")

        x = self.input_adapter(src)  # [B, T, D]
        x = self.pos_encoder(x)
        if self.time_encoder is not None and time_features is not None:
            time_emb = self.time_encoder(time_features)  # [B, T, D]
            if time_emb.shape[:2] != x.shape[:2]:
                raise ValueError(f"Time features shape {time_emb.shape} incompatible with input {x.shape}")
            x = x + time_emb
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)

        for i in range(self.num_layers):
            layer = self._get_layer(i)
            if self.training and self.budget_scheduler is not None:
                dynamic_budget = self.budget_scheduler.get_budget()
                if self.training and self.use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        lambda _x, l=layer, db=dynamic_budget: l(
                            _x, src_mask, src_key_padding_mask,
                            gate_budget=db,
                            gate_lambda=self.gate_lambda,
                            use_gateskip=self.use_gateskip
                        ),
                        x, use_reentrant=False
                    )
                else:
                    x = layer(
                        x, src_mask, src_key_padding_mask,
                        gate_budget=dynamic_budget,
                        gate_lambda=self.gate_lambda,
                        use_gateskip=self.use_gateskip
                    )
            else:
                if self.training and self.use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        lambda _x, l=layer: l(
                            _x, src_mask, src_key_padding_mask,
                            gate_budget=self.gate_budget,
                            gate_lambda=self.gate_lambda,
                            use_gateskip=self.use_gateskip
                        ),
                        x, use_reentrant=False
                    )
                else:
                    x = layer(
                        x, src_mask, src_key_padding_mask,
                        gate_budget=self.gate_budget,
                        gate_lambda=self.gate_lambda,
                        use_gateskip=self.use_gateskip
                    )

        self._aggregate_aux_loss(self.num_layers)
        if self.budget_scheduler is not None:
            self.budget_scheduler.step()  # Advance after forward
        x = self.final_norm(x)  # [B, T, D]
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
        informer_like: bool = False,
        use_time_encoding: bool = True,
        **kwargs,
    ):
        self.output_size = output_size
        self.label_len = label_len
        self.informer_like = informer_like
        self.use_time_encoding = use_time_encoding
        super().__init__(input_size, informer_like=informer_like, **kwargs)
        if use_time_encoding:
            self.time_encoder = InformerTimeEmbedding(self.d_model)
        else:
            self.time_encoder = None
        self.output_projection = (
            nn.Identity() if output_size == self.d_model
            else nn.Linear(self.d_model, output_size)
        )

    def _make_layer(self, **kwargs):
        return TransformerDecoderLayer(**kwargs)

    def _create_informer_padding_mask(self, B: int, T: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.informer_like or self.label_len >= T:
            return None
        label_len = max(0, min(self.label_len, T))
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        mask[:, label_len:] = True
        return mask

    def forward(
        self,
        tgt: torch.Tensor,                    # [B, T_tgt, C_tgt]
        memory: torch.Tensor,                 # [B, T_src, D]
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict] = None,
        return_incremental_state: bool = False,
        time_features: Optional[torch.Tensor] = None,
    ):
        B, T, _ = tgt.shape
        device = tgt.device
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(T, device)

        x = self.input_adapter(tgt)          # [B, T, D]
        x = self.pos_encoder(x)

        if self.time_encoder is not None and time_features is not None:
            time_emb = self.time_encoder(time_features)   # [B, T, D]
            if time_emb.shape[:2] == x.shape[:2]:
                x = x + time_emb

        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)

        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._create_informer_padding_mask(B, T, device)

        # Prepare per-layer states (list of dicts or None)
        layer_states = (incremental_state.get("layers", [None] * self.num_layers)
                        if incremental_state else [None] * self.num_layers)

        for i in range(self.num_layers):
            layer = self._get_layer(i)
            prev_state = layer_states[i - 1] if i > 0 else None  # enable KV copy from previous layer

            # Dynamic budget if scheduler is active
            if self.training and self.budget_scheduler is not None:
                dynamic_budget = self.budget_scheduler.get_budget()

                if self.training and self.use_gradient_checkpointing:
                    def checkpoint_fn(_t, l=layer, db=dynamic_budget, ps=prev_state, ls=layer_states[i]):
                        return l(
                            _t, memory, tgt_mask, memory_mask,
                            tgt_key_padding_mask, memory_key_padding_mask,
                            incremental_state=ls,
                            prev_layer_state=ps,
                            gate_budget=db,
                            gate_lambda=self.gate_lambda,
                            use_gateskip=self.use_gateskip,
                        )
                    res = torch.utils.checkpoint.checkpoint(checkpoint_fn, x, use_reentrant=False)
                    x = res[0]
                    if isinstance(res, tuple) and len(res) == 2:
                        layer_states[i] = res[1]
                else:
                    x, layer_states[i] = layer(
                        x, memory, tgt_mask, memory_mask,
                        tgt_key_padding_mask, memory_key_padding_mask,
                        incremental_state=layer_states[i],
                        prev_layer_state=prev_state,
                        gate_budget=dynamic_budget,
                        gate_lambda=self.gate_lambda,
                        use_gateskip=self.use_gateskip,
                    )
            else:
                if self.training and self.use_gradient_checkpointing:
                    def checkpoint_fn(_t, l=layer, ps=prev_state, ls=layer_states[i]):
                        return l(
                            _t, memory, tgt_mask, memory_mask,
                            tgt_key_padding_mask, memory_key_padding_mask,
                            incremental_state=ls,
                            prev_layer_state=ps,
                            gate_budget=self.gate_budget,
                            gate_lambda=self.gate_lambda,
                            use_gateskip=self.use_gateskip,
                        )
                    res = torch.utils.checkpoint.checkpoint(checkpoint_fn, x, use_reentrant=False)
                    x = res[0]
                    if isinstance(res, tuple) and len(res) == 2:
                        layer_states[i] = res[1]
                else:
                    x, layer_states[i] = layer(
                        x, memory, tgt_mask, memory_mask,
                        tgt_key_padding_mask, memory_key_padding_mask,
                        incremental_state=layer_states[i],
                        prev_layer_state=prev_state,
                        gate_budget=self.gate_budget,
                        gate_lambda=self.gate_lambda,
                        use_gateskip=self.use_gateskip,
                    )

        self._aggregate_aux_loss(self.num_layers)
        if self.budget_scheduler is not None:
            self.budget_scheduler.step()

        x = self.final_norm(x)
        out = self.output_projection(x)      # [B, T, output_size]

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
        return self.forward(
            tgt, memory,
            incremental_state=incremental_state or {},
            return_incremental_state=True,
            time_features=time_features,
        )
