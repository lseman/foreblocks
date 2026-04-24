from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ui.node_spec import node

from .attention.modules.gated_delta import GatedDeltaNet
from .attention.modules.kimi_att import KimiAttention
from .attention.modules.lin_att import LinearAttention
from .attention.multi_att import MultiAttention
from .attention.utils.residuals import (
    AttentionResidual,
    BlockAttentionResidual,
    normalize_attention_residual_mode,
)
from .embeddings import InformerTimeEmbedding
from .mhc import mhc_init_streams
from .patching import PatchInfo, patchify_padding_mask
from .skip.gateskip import ResidualGate
from .tf_base import (
    BaseTransformer,
    BaseTransformerLayer,
    MHCBlockMixin,
    NormWrapper,
    ResidualBlockMixin,
    _append_attention_residual_update,
    _attention_residual_input,
    _gateskip_active_mask_from_padding,
    _gather_padding_mask,
    _gather_query_mask,
    _gather_sequence_tokens,
    _gather_square_mask,
    _init_attention_residual_state,
    _ModelLayerInvokeStrategy,
    _patchify_gateskip_active_mask,
    _scatter_mixture_of_depths_output,
)


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
        gate_budget: float | None = None,
        gate_lambda: float = 0.1,
        layer_attention_type: str = "standard",
        use_mhc: bool = False,
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_collapse: str = "first",
        moe_use_latent: bool = False,
        moe_latent_dim: int | None = None,
        moe_latent_d_ff: int | None = None,
        use_attention_matching_compaction: bool = False,
        attention_matching_keep_ratio: float = 0.25,
        attention_matching_trigger_len: int = 512,
        attention_matching_min_keep: int = 64,
        attention_matching_query_budget: int = 64,
        attention_matching_force_single_step: bool = False,
        use_attention_residual: bool = True,
        attn_residual_type: str = "full",
        attention_residual_block_size: int = 8,
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
            mhc_collapse=mhc_collapse,
            moe_use_latent=moe_use_latent,
            moe_latent_dim=moe_latent_dim,
            moe_latent_d_ff=moe_latent_d_ff,
        )

        self.self_attn_std = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
            cross_attention=False,
            use_mla=not use_attention_matching_compaction,
            use_attention_matching_compaction=use_attention_matching_compaction,
            attention_matching_keep_ratio=attention_matching_keep_ratio,
            attention_matching_trigger_len=attention_matching_trigger_len,
            attention_matching_min_keep=attention_matching_min_keep,
            attention_matching_query_budget=attention_matching_query_budget,
            attention_matching_force_single_step=attention_matching_force_single_step,
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
            use_mla=not use_attention_matching_compaction,
            use_attention_matching_compaction=use_attention_matching_compaction,
            attention_matching_keep_ratio=attention_matching_keep_ratio,
            attention_matching_trigger_len=attention_matching_trigger_len,
            attention_matching_min_keep=attention_matching_min_keep,
            attention_matching_query_budget=attention_matching_query_budget,
            attention_matching_force_single_step=attention_matching_force_single_step,
        )
        self.self_attn_kimi = KimiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            cross_attention=False,
        )
        self.self_attn_gdn = GatedDeltaNet(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
        )
        self.layer_attention_type = str(layer_attention_type)

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

        self.mhc_conn_self: nn.Module | None = None
        self.mhc_conn_cross: nn.Module | None = None
        self.mhc_conn_ff: nn.Module | None = None
        if self.use_mhc:
            self._ensure_mhc_mixers()

        self.use_attention_residual = use_attention_residual
        self.attention_residual_mode = normalize_attention_residual_mode(
            attn_residual_type
        )
        self.attention_residual_block_size = int(attention_residual_block_size)
        residual_cls = (
            BlockAttentionResidual
            if self.attention_residual_mode == "block"
            else AttentionResidual
        )
        self.self_input_residual = residual_cls(d_model)
        self.cross_input_residual = residual_cls(d_model)
        self.ff_input_residual = residual_cls(d_model)

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
        if t == "gated_delta":
            return self.self_attn_gdn
        return self.self_attn_std

    def _ensure_mhc_mixers(self) -> None:
        if not self.use_mhc:
            self.mhc_conn_self = None
            self.mhc_conn_cross = None
            self.mhc_conn_ff = None
            return

        if (self.mhc_conn_self is None) or (
            getattr(self.mhc_conn_self, "n", None) != self.mhc_n_streams
        ):
            self.mhc_conn_self = self._new_mhc_connection()
        if (self.mhc_conn_cross is None) or (
            getattr(self.mhc_conn_cross, "n", None) != self.mhc_n_streams
        ):
            self.mhc_conn_cross = self._new_mhc_connection()
        if (self.mhc_conn_ff is None) or (
            getattr(self.mhc_conn_ff, "n", None) != self.mhc_n_streams
        ):
            self.mhc_conn_ff = self._new_mhc_connection()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        incremental_state: dict | None = None,
        prev_layer_state: dict | None = None,
        gate_budget: float | None = None,
        gate_lambda: float | None = None,
        use_gateskip: bool | None = None,
        streams: torch.Tensor | None = None,
        use_mhc: bool | None = None,
        mhc_n_streams: int | None = None,
        mhc_sinkhorn_iters: int | None = None,
        mhc_collapse: str | None = None,
        mtp_targets: torch.Tensor | None = None,
        attention_residual_state: dict | None = None,
        gateskip_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict | None, torch.Tensor | None]:
        self._reset_aux_loss()

        self._apply_runtime_mhc_overrides(
            use_mhc=use_mhc,
            mhc_n_streams=mhc_n_streams,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_collapse=mhc_collapse,
        )
        cfg = self._build_residual_cfg(
            use_gateskip=use_gateskip,
            gate_budget=gate_budget,
            gate_lambda=gate_lambda,
            training=self.training,
        )
        aux_l2_terms: list[torch.Tensor] = []

        if self.use_mhc and incremental_state is not None:
            raise RuntimeError(
                "mHC decoder does not support incremental_state/KV-cached decoding yet. "
                "Disable mHC for autoregressive decoding."
            )

        if memory_key_padding_mask is not None:
            if (
                memory_key_padding_mask.shape[0] != memory.shape[0]
                or memory_key_padding_mask.shape[1] != memory.shape[1]
            ):
                raise ValueError(
                    f"memory_key_padding_mask shape {tuple(memory_key_padding_mask.shape)} must match memory [B,Tm]=[{memory.shape[0]},{memory.shape[1]}]"
                )

        if incremental_state is not None:
            self_attn_state = incremental_state.get("self_attn")
            cross_attn_state = incremental_state.get("cross_attn")
            state = {
                "self_attn": (self_attn_state if self_attn_state is not None else {}),
                "cross_attn": (
                    cross_attn_state if cross_attn_state is not None else {}
                ),
            }
        else:
            state = {"self_attn": None, "cross_attn": None}
        gateskip_active_mask = (
            gateskip_active_mask.to(dtype=torch.bool)
            if gateskip_active_mask is not None
            else None
        )

        self_attn_mod = self._self_attn()
        self_attn_mask = tgt_mask
        if incremental_state is not None and tgt.size(1) == 1:
            self_attn_mask = None

        if self.use_attention_residual:
            if cfg.use_gateskip:
                raise RuntimeError(
                    "Attention Residuals are not compatible with GateSkip at the layer level."
                )
            if self.use_mhc:
                raise RuntimeError(
                    "Attention Residuals are not compatible with mHC at the layer level."
                )

            residual_state = attention_residual_state
            if residual_state is None:
                residual_state = _init_attention_residual_state(
                    tgt,
                    self.attention_residual_mode,
                    self.attention_residual_block_size,
                )

            def self_core_attnres(
                x_in: torch.Tensor,
            ) -> tuple[torch.Tensor, dict | None]:
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

            self_in = _attention_residual_input(
                residual_state["current"], residual_state, self.self_input_residual
            )
            self_out, updated_self = self._run_attnres_core(
                self_in, self.self_attn_norm, self_core_attnres
            )
            _append_attention_residual_update(residual_state, self_out)
            if updated_self is not None:
                state["self_attn"] = updated_self

            def cross_core_attnres(
                x_in: torch.Tensor,
            ) -> tuple[torch.Tensor, dict | None]:
                out, _, updated = self.cross_attn(
                    x_in,
                    memory,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    layer_state=state["cross_attn"],
                )
                return out, updated

            cross_in = _attention_residual_input(
                residual_state["current"], residual_state, self.cross_input_residual
            )
            cross_out, updated_cross = self._run_attnres_core(
                cross_in, self.cross_attn_norm, cross_core_attnres
            )
            _append_attention_residual_update(residual_state, cross_out)
            if updated_cross is not None:
                state["cross_attn"] = updated_cross

            def ff_core_attnres(
                x_in: torch.Tensor,
            ) -> tuple[torch.Tensor, dict | None]:
                return self._ff_forward_with_aux(x_in, mtp_targets=mtp_targets), None

            ff_in = _attention_residual_input(
                residual_state["current"], residual_state, self.ff_input_residual
            )
            ff_out, _ = self._run_attnres_core(ff_in, self.ff_norm, ff_core_attnres)
            _append_attention_residual_update(residual_state, ff_out)
            ret_state = {
                "self_attn": state["self_attn"],
                "cross_attn": state["cross_attn"],
            }
            return residual_state["current"], ret_state, None

        strategy = self._make_exec_strategy(x=tgt, streams=streams)
        if self.use_mhc:
            assert (
                self.mhc_conn_self is not None
                and self.mhc_conn_cross is not None
                and self.mhc_conn_ff is not None
            )

        def self_mhc_core(x_in: torch.Tensor) -> torch.Tensor:
            out, _, _ = self_attn_mod(
                x_in,
                x_in,
                x_in,
                self_attn_mask,
                tgt_key_padding_mask,
                is_causal=self.is_causal,
                layer_state=None,
            )
            return out

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

        updated_self, _ = strategy.run_block(
            normw=self.self_attn_norm,
            gate=self.gate_self,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=self_core,
            mhc_core=self_mhc_core,
            hyper_conn=self.mhc_conn_self,
            prev_layer_state=prev_layer_state,
            kv_key="self_attn",
            active_mask=gateskip_active_mask,
        )
        if updated_self is not None:
            state["self_attn"] = updated_self

        def cross_mhc_core(x_in: torch.Tensor) -> torch.Tensor:
            out, _, _ = self.cross_attn(
                x_in,
                memory,
                memory,
                memory_mask,
                memory_key_padding_mask,
                layer_state=None,
            )
            return out

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

        updated_cross, _ = strategy.run_block(
            normw=self.cross_attn_norm,
            gate=self.gate_cross,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=cross_core,
            mhc_core=cross_mhc_core,
            hyper_conn=self.mhc_conn_cross,
            prev_layer_state=prev_layer_state,
            kv_key="cross_attn",
            active_mask=gateskip_active_mask,
        )
        if updated_cross is not None:
            state["cross_attn"] = updated_cross

        def ff_core(x_in: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
            return self._ff_forward_with_aux(x_in, mtp_targets=mtp_targets), None

        def ff_mhc_core(x_in: torch.Tensor) -> torch.Tensor:
            return self._ff_forward_with_aux(x_in, mtp_targets=mtp_targets)

        strategy.run_block(
            normw=self.ff_norm,
            gate=self.gate_ff,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=ff_core,
            mhc_core=ff_mhc_core,
            hyper_conn=self.mhc_conn_ff,
            active_mask=gateskip_active_mask,
        )

        self._finalize_gateskip_aux(cfg, aux_l2_terms)

        out_tgt, out_streams = strategy.collapse(self.mhc_collapse)
        ret_state = {"self_attn": state["self_attn"], "cross_attn": state["cross_attn"]}
        return out_tgt, ret_state, out_streams


@node(
    type_id="transformer_decoder",
    name="Transformer Decoder",
    category="Decoder",
    outputs=["decoder"],
    color="bg-gradient-to-br from-purple-700 to-purple-800",
    config_sources=[BaseTransformerLayer, TransformerDecoderLayer],
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
    ) -> torch.Tensor | None:
        if not self.informer_like:
            return None
        label_len = int(self.label_len)
        if label_len <= 0 or label_len >= T:
            return None
        label_len = min(label_len, T)
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
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        incremental_state: dict | None = None,
        return_incremental_state: bool = False,
        time_features: torch.Tensor | None = None,
        mtp_targets: torch.Tensor | None = None,  # [B,T,F] or [B,T,H,D]
        gateskip_active_mask: torch.Tensor | None = None,  # [B,T] bool
        position_offset: torch.Tensor | int | None = None,
    ):
        B, T, _ = tgt.shape
        device = tgt.device
        user_tgt_key_padding_mask = tgt_key_padding_mask

        self.mod_aux_loss.zero_()

        if self.patch_decoder and incremental_state is not None:
            raise RuntimeError(
                "patch_decoder=True is not compatible with incremental_state/KV-cached decoding. "
                "Set patch_decoder=False (recommended) for autoregressive decoding."
            )
        if self.use_mod and incremental_state is not None:
            raise RuntimeError(
                "Mixture-of-Depths routing is currently implemented for full-sequence decoding only. "
                "Disable MoD for KV-cached autoregressive decoding."
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

        patch_info: PatchInfo | None = None
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
            gateskip_active_mask = _patchify_gateskip_active_mask(
                gateskip_active_mask,
                T=T,
                patch_len=self.patch_len,
                stride=self.patch_stride,
                pad_end=self.patch_pad_end,
            )

        if position_offset is None:
            x = self.pos_encoder(x)
        else:
            if isinstance(position_offset, int):
                pos = (
                    torch.arange(
                        position_offset,
                        position_offset + x.shape[1],
                        device=device,
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .expand(B, -1)
                )
            else:
                pos = position_offset.to(device=device, dtype=torch.long)
                if pos.dim() == 0:
                    pos = pos.view(1).expand(B)
                if pos.dim() != 1 or pos.shape[0] != B:
                    raise ValueError(
                        f"position_offset tensor must be scalar or [B], got {tuple(pos.shape)}"
                    )
                pos = pos.unsqueeze(1) + torch.arange(
                    x.shape[1], device=device, dtype=torch.long
                ).unsqueeze(0)
            x = self.pos_encoder(x, pos=pos)

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

        self._validate_attention_residual_runtime()
        self._validate_mod_runtime()
        attention_residual_state = self._init_attention_residual_state(x)

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

        if gateskip_active_mask is None:
            # Paper-like default for time series: all real positions are active.
            # We intentionally derive this only from an actual padding mask, not
            # from the auto-generated Informer masking over the forecast horizon.
            gateskip_active_mask = _gateskip_active_mask_from_padding(
                tgt_key_padding_mask
                if self.patch_decoder
                else user_tgt_key_padding_mask
            )

        if incremental_state is not None:
            raw_layer_states = incremental_state.get("layers")
            if raw_layer_states is None:
                layer_states = [
                    {"self_attn": {}, "cross_attn": {}} for _ in range(self.num_layers)
                ]
            else:
                layer_states = []
                for layer_state in raw_layer_states:
                    if layer_state is None:
                        layer_states.append({"self_attn": {}, "cross_attn": {}})
                    else:
                        layer_states.append(layer_state)
                if len(layer_states) != self.num_layers:
                    raise ValueError(
                        f"incremental_state['layers'] length {len(layer_states)} "
                        f"must equal num_layers={self.num_layers}"
                    )
        else:
            layer_states = [None] * self.num_layers

        streams: torch.Tensor | None = None
        if self.use_mhc:
            if incremental_state is not None:
                raise RuntimeError(
                    "Decoder mHC does not support incremental_state/KV-cached decoding. "
                    "Disable mHC for autoregressive decoding."
                )
            streams = mhc_init_streams(x, self.mhc_n_streams)

        use_ckpt = (
            self.training
            and self.use_gradient_checkpointing
            and (not self.use_mhc)
            and (not self.use_attention_residual)
        )
        invoke = _ModelLayerInvokeStrategy(owner=self, use_checkpoint=use_ckpt)
        runtime_budget = self._get_runtime_budget()

        used_indices: list[int] = []

        for i in range(self.num_layers):
            prev_state = layer_states[i - 1] if i > 0 else None

            if self.use_mod:
                (
                    layer,
                    router_logits,
                    _keep_mask,
                    routed_indices,
                    routed_slots,
                ) = self._prepare_layer_routing(i, x, gateskip_active_mask)
                if (
                    routed_indices is None
                    or routed_slots is None
                    or not bool(routed_slots.any())
                ):
                    continue

                x_routed = _gather_sequence_tokens(x, routed_indices)
                tgt_mask_routed = _gather_square_mask(tgt_mask, routed_indices)
                memory_mask_routed = _gather_query_mask(memory_mask, routed_indices)
                tgt_kpm_routed = _gather_padding_mask(
                    tgt_key_padding_mask, routed_indices, routed_slots
                )
                mtp_targets_routed = (
                    _gather_sequence_tokens(layer_mtp_targets, routed_indices)
                    if layer_mtp_targets is not None
                    else None
                )

                used_indices.append(i)
                x_routed_out, layer_states[i], streams = invoke.run_decoder_layer(
                    layer=layer,
                    x=x_routed,
                    memory=memory,
                    tgt_mask=tgt_mask_routed,
                    memory_mask=memory_mask_routed,
                    tgt_key_padding_mask=tgt_kpm_routed,
                    memory_key_padding_mask=memory_key_padding_mask,
                    layer_state=layer_states[i],
                    prev_state=prev_state,
                    budget=runtime_budget,
                    streams=streams,
                    mtp_targets=mtp_targets_routed,
                    attention_residual_state=attention_residual_state,
                    gateskip_active_mask=routed_slots,
                )
                x = _scatter_mixture_of_depths_output(
                    x,
                    x_routed,
                    x_routed_out,
                    routed_indices,
                    routed_slots,
                    router_logits,
                )
                continue

            layer = self._resolve_layer(i)
            used_indices.append(i)
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
                budget=runtime_budget,
                streams=streams,
                mtp_targets=layer_mtp_targets,
                attention_residual_state=attention_residual_state,
                gateskip_active_mask=gateskip_active_mask,
            )

        # FIX: aggregate only over executed layers
        self._finalize_layer_stack(used_indices)

        x = self._finalize_attention_residual_output(attention_residual_state, x)
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
        incremental_state: dict | None = None,
        time_features: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ):
        if self.patch_decoder:
            raise RuntimeError(
                "forward_one_step requires patch_decoder=False (recommended default)."
            )
        if self.use_mod:
            raise RuntimeError(
                "forward_one_step does not yet support Mixture-of-Depths with KV caching. "
                "Disable MoD for autoregressive decoding."
            )
        if tgt.dim() != 3 or tgt.size(1) <= 0:
            raise ValueError(
                f"forward_one_step expects tgt [B,T,C] with T>0, got {tuple(tgt.shape)}"
            )

        has_kv_cache = False
        if incremental_state is not None:
            layer_states = incremental_state.get("layers")
            if isinstance(layer_states, list):
                has_kv_cache = any(
                    layer_state is not None for layer_state in layer_states
                )

        step_tgt = tgt[:, -1:, :] if has_kv_cache else tgt
        step_time_features = (
            time_features[:, -1:, :]
            if (has_kv_cache and time_features is not None)
            else time_features
        )
        decoded_len = int((incremental_state or {}).get("_decoded_len", 0))
        out, next_state = self.forward(
            step_tgt,
            memory,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            incremental_state=incremental_state or {},
            return_incremental_state=True,
            time_features=step_time_features,
            position_offset=decoded_len,
        )
        next_state["_decoded_len"] = decoded_len + step_tgt.size(1)
        return out, next_state


__all__ = ["TransformerDecoderLayer", "TransformerDecoder"]
