from __future__ import annotations

import math
from typing import Literal

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
    _gather_sequence_tokens,
    _gather_square_mask,
    _init_attention_residual_state,
    _ModelLayerInvokeStrategy,
    _patchify_gateskip_active_mask,
    _scatter_mixture_of_depths_output,
)


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
            use_mla=not use_attention_matching_compaction,
            use_attention_matching_compaction=use_attention_matching_compaction,
            attention_matching_keep_ratio=attention_matching_keep_ratio,
            attention_matching_trigger_len=attention_matching_trigger_len,
            attention_matching_min_keep=attention_matching_min_keep,
            attention_matching_query_budget=attention_matching_query_budget,
            attention_matching_force_single_step=attention_matching_force_single_step,
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
            use_mla=not use_attention_matching_compaction,
            use_attention_matching_compaction=use_attention_matching_compaction,
            attention_matching_keep_ratio=attention_matching_keep_ratio,
            attention_matching_trigger_len=attention_matching_trigger_len,
            attention_matching_min_keep=attention_matching_min_keep,
            attention_matching_query_budget=attention_matching_query_budget,
            attention_matching_force_single_step=attention_matching_force_single_step,
        )
        self.self_attn_kimi = KimiAttention(
            d_model=d_model, n_heads=nhead, dropout=dropout
        )
        self.self_attn_gdn = GatedDeltaNet(
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

        self.mhc_conn_attn: nn.Module | None = None
        self.mhc_conn_ff: nn.Module | None = None
        if self.use_mhc:
            self._ensure_mhc_mixers()

        self.use_attention_residual = use_attention_residual
        self.attention_residual_mode = normalize_attention_residual_mode(
            attn_residual_type
        )
        self.attention_residual_block_size = int(attention_residual_block_size)
        self.attn_input_residual = (
            BlockAttentionResidual(d_model)
            if self.attention_residual_mode == "block"
            else AttentionResidual(d_model)
        )
        self.ff_input_residual = (
            BlockAttentionResidual(d_model)
            if self.attention_residual_mode == "block"
            else AttentionResidual(d_model)
        )

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
            self.mhc_conn_attn = None
            self.mhc_conn_ff = None
            return

        if (self.mhc_conn_attn is None) or (
            getattr(self.mhc_conn_attn, "n", None) != self.mhc_n_streams
        ):
            self.mhc_conn_attn = self._new_mhc_connection()
        if (self.mhc_conn_ff is None) or (
            getattr(self.mhc_conn_ff, "n", None) != self.mhc_n_streams
        ):
            self.mhc_conn_ff = self._new_mhc_connection()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._reset_aux_loss()
        gateskip_active_mask = (
            gateskip_active_mask.to(dtype=torch.bool)
            if gateskip_active_mask is not None
            else None
        )

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

        attn_mod = self._self_attn()

        if self.use_attention_residual:
            if cfg.use_gateskip:
                raise RuntimeError(
                    "Attention Residuals are not compatible with GateSkip at the layer level."
                )
            if self.use_mhc:
                raise RuntimeError(
                    "Attention Residuals are not compatible with mHC at the layer level."
                )

            state = attention_residual_state
            if state is None:
                state = _init_attention_residual_state(
                    src,
                    self.attention_residual_mode,
                    self.attention_residual_block_size,
                )

            attn_in = _attention_residual_input(
                state["current"], state, self.attn_input_residual
            )

            def attn_core_attnres(
                x_in: torch.Tensor,
            ) -> tuple[torch.Tensor, dict | None]:
                out, _, _ = attn_mod(x_in, x_in, x_in, src_mask, src_key_padding_mask)
                return out, None

            attn_out, _ = self._run_attnres_core(
                attn_in, self.attn_norm, attn_core_attnres
            )
            _append_attention_residual_update(state, attn_out)

            ff_in = _attention_residual_input(
                state["current"], state, self.ff_input_residual
            )

            def ff_core_attnres(
                x_in: torch.Tensor,
            ) -> tuple[torch.Tensor, dict | None]:
                return self._ff_forward_with_aux(x_in, mtp_targets=mtp_targets), None

            ff_out, _ = self._run_attnres_core(ff_in, self.ff_norm, ff_core_attnres)
            _append_attention_residual_update(state, ff_out)
            return state["current"], None

        strategy = self._make_exec_strategy(x=src, streams=streams)
        if self.use_mhc:
            assert self.mhc_conn_attn is not None and self.mhc_conn_ff is not None

        def attn_core(x_in):
            out, _, _ = attn_mod(x_in, x_in, x_in, src_mask, src_key_padding_mask)
            return out, None

        def attn_mhc_core(x_in: torch.Tensor) -> torch.Tensor:
            out, _, _ = attn_mod(x_in, x_in, x_in, src_mask, src_key_padding_mask)
            return out

        strategy.run_block(
            normw=self.attn_norm,
            gate=self.gate_attn,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=attn_core,
            mhc_core=attn_mhc_core,
            hyper_conn=self.mhc_conn_attn,
            active_mask=gateskip_active_mask,
        )

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

        out_src, out_streams = strategy.collapse(self.mhc_collapse)
        return out_src, out_streams


@node(
    type_id="transformer_encoder",
    name="Transformer Encoder",
    category="Encoder",
    outputs=["encoder"],
    color="bg-gradient-to-br from-green-700 to-green-800",
    config_sources=[BaseTransformerLayer, TransformerEncoderLayer],
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
        self.last_memory_key_padding_mask: torch.Tensor | None = None
        self.last_patch_info: PatchInfo | None = None

    @staticmethod
    def _compute_ct_patch_pad(T: int, P: int, S: int) -> int:
        if T <= 0:
            return 0
        if T < P:
            return P - T
        n_patches = math.ceil((T - P) / S) + 1
        T_pad = (n_patches - 1) * S + P
        return max(0, T_pad - T)

    def _ct_patchify(self, src: torch.Tensor) -> tuple[torch.Tensor, PatchInfo]:
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
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,  # [B,T] bool
        time_features: torch.Tensor | None = None,  # [B, T, F_tf]
        gateskip_active_mask: torch.Tensor | None = None,  # [B,T] bool
    ) -> torch.Tensor:
        B, T, C = src.shape
        if C != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {C}")
        if T > self.max_seq_len and (not self.patch_encoder) and (not self.ct_patchtst):
            raise ValueError(f"Sequence length {T} exceeds max {self.max_seq_len}")

        self.mod_aux_loss.zero_()

        patch_info: PatchInfo | None = None
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
            gateskip_active_mask = _patchify_gateskip_active_mask(
                gateskip_active_mask,
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
            gateskip_active_mask = _patchify_gateskip_active_mask(
                gateskip_active_mask,
                T=T,
                patch_len=self.patch_len,
                stride=self.patch_stride,
                pad_end=self.patch_pad_end,
            )

        if gateskip_active_mask is None:
            gateskip_active_mask = _gateskip_active_mask_from_padding(
                src_key_padding_mask
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

        self._validate_attention_residual_runtime()
        self._validate_mod_runtime()
        attention_residual_state = self._init_attention_residual_state(x)

        # mHC streams
        streams: torch.Tensor | None = (
            mhc_init_streams(x, self.mhc_n_streams) if self.use_mhc else None
        )

        # safest: checkpoint only when not using mHC (streams complicate checkpoint)
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
                src_mask_routed = _gather_square_mask(src_mask, routed_indices)
                src_kpm_routed = _gather_padding_mask(
                    src_key_padding_mask, routed_indices, routed_slots
                )

                used_indices.append(i)
                x_routed_out, streams = invoke.run_encoder_layer(
                    layer=layer,
                    x=x_routed,
                    src_mask=src_mask_routed,
                    src_key_padding_mask=src_kpm_routed,
                    budget=runtime_budget,
                    streams=streams,
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
            x, streams = invoke.run_encoder_layer(
                layer=layer,
                x=x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                budget=runtime_budget,
                streams=streams,
                attention_residual_state=attention_residual_state,
                gateskip_active_mask=gateskip_active_mask,
            )

        # FIX: aggregate only over executed layers
        self._finalize_layer_stack(used_indices)

        x = self._finalize_attention_residual_output(attention_residual_state, x)
        x = self.final_norm(x)
        # IMPORTANT: if patch_encoder=True, we DO NOT unpatch here.
        return x


__all__ = ["TransformerEncoderLayer", "TransformerEncoder"]
