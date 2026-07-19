"""foreblocks.models.transformer.core.encoder.

Transformer encoder with lazy multi-backend attention and mixture-of-depths routing.

Implements TransformerEncoderLayer with lazy attention module instantiation
(~22M fewer dead params per 12-layer model) supporting standard, linear, Kimi,
GLA, DeltaNet, GatedDeltaNet, and SyPE backends. Includes CT-PatchTST
alternative tokenization and Mixture-of-Depths layer routing.

Core API:
- TransformerEncoderLayer: encoder layer with lazy attention backends
- TransformerEncoder: full encoder with patching, time encoding, and MoD routing

"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.embeddings import InformerTimeEmbedding
from foreblocks.models.transformer.config import TransformerConfig
from foreblocks.models.transformer.features.mhc import mhc_init_streams
from foreblocks.models.transformer.runtime.outputs import TransformerEncoderOutput
from foreblocks.models.transformer.features.patching import PatchInfo, patchify_padding_mask
from foreblocks.models.transformer.core.base import (
    BaseTransformer,
    BaseTransformerLayer,
)
from foreblocks.models.transformer.runtime.execution import (
    LazyAttentionBackendMixin,
    MHCBlockMixin,
    NormWrapper,
    ResidualBlockMixin,
    _ModelLayerInvokeStrategy,
)
from foreblocks.models.transformer.runtime.residual_state import (
    _init_attention_residual_state,
)
from foreblocks.models.transformer.runtime.routing import (
    _gateskip_active_mask_from_padding,
    _gather_padding_mask,
    _gather_sequence_tokens,
    _gather_square_mask,
    _patchify_gateskip_active_mask,
    _run_mod_layer,
)
from foreblocks.modules.attention.utils.residuals import (
    AttentionResidual,
    BlockAttentionResidual,
    normalize_attention_residual_mode,
)
from foreblocks.modules.skip.gateskip import ResidualGate
from foreblocks.ui.node_spec import node


class TransformerEncoderLayer(
    ResidualBlockMixin, MHCBlockMixin, LazyAttentionBackendMixin, BaseTransformerLayer
):
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
        attn_implementation: str = "auto",
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
        moba_block_size: int | None = None,
        moba_topk: int = 4,
        use_attention_residual: bool = False,  # True enables depth attention residuals (incompatible with ckpt/mHC/mod/gateskip)
        attn_residual_type: str = "full",
        attention_residual_block_size: int = 8,
        pos_encoding_type: str = "rope",
        rope_base: float = 10000.0,
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
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

        # Init parameters for the lazy attention-backend registry (captured once, reused).
        # See core/attention_backends.py + runtime/execution.py's
        # LazyAttentionBackendMixin for the shared lazy-instantiation cache.
        self._attn_init_kwargs = {
            "d_model": d_model,
            "n_heads": nhead,
            "dropout": dropout,
        }
        self._attn_backend_cfg = dict(
            att_type=att_type,
            attn_implementation=attn_implementation,
            freq_modes=freq_modes,
            use_attention_matching_compaction=use_attention_matching_compaction,
            attention_matching_keep_ratio=attention_matching_keep_ratio,
            attention_matching_trigger_len=attention_matching_trigger_len,
            attention_matching_min_keep=attention_matching_min_keep,
            attention_matching_query_budget=attention_matching_query_budget,
            attention_matching_force_single_step=attention_matching_force_single_step,
            moba_block_size=moba_block_size,
            moba_topk=moba_topk,
            rope_base=rope_base,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
        )
        self.layer_attention_type = str(layer_attention_type)
        self._pos_encoding_type = str(pos_encoding_type)

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
        self.materialize_attention_type()

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

        self._validate_runtime_mhc_overrides(
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
            if attention_residual_state is None:
                attention_residual_state = _init_attention_residual_state(
                    src,
                    self.attention_residual_mode,
                    self.attention_residual_block_size,
                )

        strategy = self._make_exec_strategy(
            x=src, streams=streams, attention_residual_state=attention_residual_state
        )
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
            residual_module=self.attn_input_residual,
        )

        def ff_core(x_in: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
            return self._ff_forward_with_aux(
                x_in,
                mtp_targets=mtp_targets,
                padding_mask=src_key_padding_mask,
            ), None

        def ff_mhc_core(x_in: torch.Tensor) -> torch.Tensor:
            return self._ff_forward_with_aux(
                x_in,
                mtp_targets=mtp_targets,
                padding_mask=src_key_padding_mask,
            )

        strategy.run_block(
            normw=self.ff_norm,
            gate=self.gate_ff,
            cfg=cfg,
            aux_l2_terms=aux_l2_terms,
            core_fn=ff_core,
            mhc_core=ff_mhc_core,
            hyper_conn=self.mhc_conn_ff,
            active_mask=gateskip_active_mask,
            residual_module=self.ff_input_residual,
        )

        self._finalize_gateskip_aux(cfg, aux_l2_terms)

        if self.use_attention_residual:
            return strategy.x, None

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
        input_size: int | TransformerConfig = 1,
        use_time_encoding: bool = False,
        model_type: str = "transformer",
        ct_patchtst: bool = False,
        ct_patch_len: int = 16,
        ct_patch_stride: int = 8,
        ct_patch_pad_end: bool = True,
        ct_patch_fuse: Literal["mean", "linear"] = "linear",
        config: TransformerConfig | None = None,
        **kwargs,
    ):
        if isinstance(input_size, TransformerConfig):
            if config is not None:
                raise ValueError("pass TransformerConfig once, as input_size or config")
            config = input_size
        if config is not None:
            input_size = config.input_size
            model_type = config.model_type
            use_time_encoding = config.use_time_encoding
            ct_patchtst = config.ct_patchtst
            ct_patch_len = config.ct_patch_len
            ct_patch_stride = config.ct_patch_stride
            ct_patch_pad_end = config.ct_patch_pad_end
            ct_patch_fuse = config.ct_patch_fuse
            kwargs = {**config.model_kwargs(), **kwargs}
        else:
            config = TransformerConfig.from_kwargs(
                input_size=input_size,
                model_type=model_type,
                use_time_encoding=use_time_encoding,
                ct_patchtst=ct_patchtst,
                ct_patch_len=ct_patch_len,
                ct_patch_stride=ct_patch_stride,
                ct_patch_pad_end=ct_patch_pad_end,
                ct_patch_fuse=ct_patch_fuse,
                **kwargs,
            )
        config.validate_compatibility(role="encoder")
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
        self.config = config
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

        # These modules are created after BaseTransformer has applied the
        # model-wide initialization policy, so initialize them explicitly.
        # Otherwise they retain PyTorch defaults and have a seed-dependent
        # scale that is inconsistent with the rest of the transformer.
        for module in (self.time_encoder, self.ct_patch_embed, self.ct_channel_fuse):
            if module is not None:
                module.apply(self._init_weights)

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
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor | TransformerEncoderOutput:
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
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

        # Only apply input-level positional encoding for non-RoPE/ALiBi modes.
        # RoPE and ALiBi handle position encoding internally inside attention.
        if self.pos_encoding_type in ("sinusoidal", "learnable"):
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
        all_hidden_states: list[torch.Tensor] | None = (
            [x] if output_hidden_states else None
        )
        router_states: list[object] = []
        all_attentions: list[torch.Tensor] = []

        for i in range(self.num_layers):
            if self.use_mod:
                def gather_and_invoke(layer, routed_indices, routed_slots):
                    nonlocal streams
                    x_routed = _gather_sequence_tokens(x, routed_indices)
                    src_mask_routed = _gather_square_mask(src_mask, routed_indices)
                    src_kpm_routed = _gather_padding_mask(
                        src_key_padding_mask, routed_indices, routed_slots
                    )
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
                    return x_routed, x_routed_out

                x, was_used = _run_mod_layer(
                    self, i, x, gateskip_active_mask, all_hidden_states,
                    router_states, gather_and_invoke,
                )
                if was_used:
                    used_indices.append(i)
                continue

            layer = self._resolve_layer(i)
            attention_module = layer._self_attn()
            if hasattr(attention_module, "output_attentions"):
                attention_module.output_attentions = output_attentions
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
            if all_hidden_states is not None:
                all_hidden_states.append(x)
            ff_block = getattr(getattr(layer, "feed_forward", None), "block", None)
            router_state = getattr(ff_block, "last_routing_state", None)
            if router_state is not None:
                router_states.append(router_state)
            attention_weights = getattr(attention_module, "last_attn_weights", None)
            if attention_weights is not None:
                all_attentions.append(attention_weights)

        # FIX: aggregate only over executed layers
        self._finalize_layer_stack(used_indices)

        x = self._finalize_attention_residual_output(attention_residual_state, x)
        x = self.final_norm(x)
        if all_hidden_states is not None:
            all_hidden_states[-1] = x
        # IMPORTANT: if patch_encoder=True, we DO NOT unpatch here.
        if return_dict:
            return TransformerEncoderOutput(
                last_hidden_state=x,
                hidden_states=(
                    tuple(all_hidden_states) if all_hidden_states is not None else None
                ),
                aux_loss=self.aux_loss,
                padding_mask=src_key_padding_mask,
                router_states=tuple(router_states) if router_states else None,
                attentions=tuple(all_attentions) if all_attentions else None,
            )
        return x


__all__ = ["TransformerEncoderLayer", "TransformerEncoder"]
