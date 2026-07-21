"""foreblocks.models.transformer.core.decoder.

Transformer decoder with cross-attention, incremental KV caching, and MTP support.

Implements TransformerDecoderLayer with self-attention, cross-attention, and FFN
stages. Supports autoregressive decoding via incremental state, multi-step
ahead (MTP) targets for MoE FFNs, and Mixture-of-Depths routing.

Core API:
- TransformerDecoderLayer: decoder layer with self/cross attn and KV cache
- TransformerDecoder: full decoder with autoregressive forward_one_step

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.embeddings import InformerTimeEmbedding
from foreblocks.models.transformer.config import TransformerConfig
from foreblocks.models.transformer.core.base import (
    BaseTransformer,
    BaseTransformerLayer,
)
from foreblocks.models.transformer.features.mhc import mhc_init_streams
from foreblocks.models.transformer.generation import GenerationConfig
from foreblocks.models.transformer.runtime.decoder_forward import (
    build_decoder_output,
    coerce_decoder_state,
    prepare_layer_states,
    resolve_output_options,
)
from foreblocks.models.transformer.runtime.decoder_services import (
    DecoderCacheManager,
    GenerationEngine,
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
    _gather_query_mask,
    _gather_sequence_tokens,
    _gather_square_mask,
    _run_mod_layer,
)
from foreblocks.models.transformer.runtime.state import (
    AttentionCacheState,
    DecoderLayerState,
    DecoderState,
)
from foreblocks.modules.attention.cache.kv import StaticKVCache
from foreblocks.modules.attention.multi_att import MultiAttention
from foreblocks.modules.attention.utils.residuals import (
    AttentionResidual,
    BlockAttentionResidual,
    normalize_attention_residual_mode,
)
from foreblocks.modules.skip.gateskip import ResidualGate
from foreblocks.ui.node_spec import node


class TransformerDecoderLayer(
    ResidualBlockMixin, MHCBlockMixin, LazyAttentionBackendMixin, BaseTransformerLayer
):
    def __init__(
        self,
        d_model: int | TransformerConfig | None = None,
        nhead: int | None = None,
        *,
        config: TransformerConfig | None = None,
        layer_attention_type: str = "standard",
        dropout: float | None = None,
        informer_like: bool | None = None,
        **legacy_kwargs,
    ):
        legacy_construction = (
            not isinstance(d_model, TransformerConfig) and config is None
        )
        if isinstance(d_model, TransformerConfig):
            if config is not None:
                raise ValueError("pass TransformerConfig once")
            config = d_model
        elif config is None:
            if d_model is None or nhead is None:
                raise TypeError("d_model and nhead are required without config")
            legacy_kwargs.setdefault("dim_feedforward", 2048)
            config = TransformerConfig.from_kwargs(
                d_model=d_model, nhead=nhead, **legacy_kwargs
            )
            legacy_kwargs.clear()
        if legacy_kwargs:
            config = config.with_overrides(**legacy_kwargs)

        dropout = config.dropout if dropout is None else dropout
        if informer_like is None:
            informer_like = False if legacy_construction else config.informer_like
        super().__init__(config, dropout=dropout)

        d_model = config.d_model
        nhead = config.nhead

        # ── Lazy attention module placeholders (decoder uses 5 self-attn backends) ──
        # Init parameters for the lazy attention-backend registry (captured once, reused).
        # See core/attention_backends.py + runtime/execution.py's
        # LazyAttentionBackendMixin for the shared lazy-instantiation cache.
        self._attn_init_kwargs = {
            "d_model": d_model,
            "n_heads": nhead,
            "dropout": dropout,
            "cross_attention": False,
        }
        self._attn_backend_cfg = dict(
            att_type=config.att_type,
            attn_implementation=config.attn_implementation,
            freq_modes=config.freq_modes,
            use_attention_matching_compaction=config.use_attention_matching_compaction,
            attention_matching_keep_ratio=config.attention_matching_keep_ratio,
            attention_matching_trigger_len=config.attention_matching_trigger_len,
            attention_matching_min_keep=config.attention_matching_min_keep,
            attention_matching_query_budget=config.attention_matching_query_budget,
            attention_matching_force_single_step=config.attention_matching_force_single_step,
            moba_block_size=config.moba_block_size,
            moba_topk=config.moba_topk,
            rope_base=config.rope_base,
            rope_scaling_type=config.rope_scaling_type,
            rope_scaling_factor=config.rope_scaling_factor,
        )
        self.layer_attention_type = str(layer_attention_type)
        self._pos_encoding_type = str(config.pos_encoding_type)

        self.cross_attn = MultiAttention(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            attention_type=config.att_type,
            attn_implementation=config.attn_implementation,
            freq_modes=config.freq_modes,
            cross_attention=True,
            pos_encoding_type=self._pos_encoding_type,
            moba_block_size=config.moba_block_size,
            moba_topk=config.moba_topk,
        )

        self.is_causal = not informer_like

        self.self_attn_norm = NormWrapper.make(
            d_model,
            config.custom_norm,
            config.norm_strategy,
            dropout,
            config.layer_norm_eps,
        )
        self.cross_attn_norm = NormWrapper.make(
            d_model,
            config.custom_norm,
            config.norm_strategy,
            dropout,
            config.layer_norm_eps,
        )
        self.ff_norm = NormWrapper.make(
            d_model,
            config.custom_norm,
            config.norm_strategy,
            dropout,
            config.layer_norm_eps,
        )

        self.gate_self = ResidualGate(d_model)
        self.gate_cross = ResidualGate(d_model)
        self.gate_ff = ResidualGate(d_model)

        self.mhc_conn_self: nn.Module | None = None
        self.mhc_conn_cross: nn.Module | None = None
        self.mhc_conn_ff: nn.Module | None = None
        if self.use_mhc:
            self._ensure_mhc_mixers()

        self.use_attention_residual = config.use_attention_residual
        self.attention_residual_mode = normalize_attention_residual_mode(
            config.attn_residual_type
        )
        self.attention_residual_block_size = int(config.attention_residual_block_size)
        residual_cls = (
            BlockAttentionResidual
            if self.attention_residual_mode == "block"
            else AttentionResidual
        )
        self.self_input_residual = residual_cls(d_model)
        self.cross_input_residual = residual_cls(d_model)
        self.ff_input_residual = residual_cls(d_model)
        self.materialize_attention_type()

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
            if attention_residual_state is None:
                attention_residual_state = _init_attention_residual_state(
                    tgt,
                    self.attention_residual_mode,
                    self.attention_residual_block_size,
                )

        strategy = self._make_exec_strategy(
            x=tgt, streams=streams, attention_residual_state=attention_residual_state
        )
        self._record_aux_loss_device(tgt.device)
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
            residual_module=self.self_input_residual,
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
            residual_module=self.cross_input_residual,
        )
        if updated_cross is not None:
            state["cross_attn"] = updated_cross

        def ff_core(x_in: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
            return self._ff_forward_with_aux(
                x_in,
                mtp_targets=mtp_targets,
                padding_mask=tgt_key_padding_mask,
            ), None

        def ff_mhc_core(x_in: torch.Tensor) -> torch.Tensor:
            return self._ff_forward_with_aux(
                x_in,
                mtp_targets=mtp_targets,
                padding_mask=tgt_key_padding_mask,
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

        out_tgt, out_streams = strategy.collapse(self.mhc_collapse)
        ret_state = DecoderLayerState(
            self_attn=AttentionCacheState.from_legacy(state["self_attn"]),
            cross_attn=AttentionCacheState.from_legacy(state["cross_attn"]),
        )
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
        input_size: int | TransformerConfig = 1,
        output_size: int = 1,
        label_len: int = 0,
        informer_like: bool = True,
        use_time_encoding: bool = False,
        model_type: str = "transformer",
        cache_implementation: str = "auto",
        config: TransformerConfig | None = None,
        **kwargs,
    ):
        if isinstance(input_size, TransformerConfig):
            if config is not None:
                raise ValueError("pass TransformerConfig once, as input_size or config")
            config = input_size
        if config is not None:
            if kwargs:
                config = config.with_overrides(**kwargs)
            input_size = config.input_size
            output_size = config.output_size
            label_len = config.label_len
            informer_like = config.informer_like
            use_time_encoding = config.use_time_encoding
            model_type = config.model_type
            cache_implementation = config.cache_implementation
        else:
            config = TransformerConfig.from_kwargs(
                input_size=input_size,
                output_size=output_size,
                label_len=label_len,
                informer_like=informer_like,
                use_time_encoding=use_time_encoding,
                model_type=model_type,
                cache_implementation=cache_implementation,
                **kwargs,
            )
        kwargs.clear()
        config.validate_compatibility(role="decoder")
        self.model_type = model_type
        # Auto-configure for informer-like
        if model_type == "informer-like":
            config = config.with_overrides(informer_like=True, use_time_encoding=True)
            informer_like = config.informer_like
            use_time_encoding = config.use_time_encoding
            if label_len == 0:
                # Default label_len for Informer is often half of target_len (implicit)
                pass

        self.output_size = output_size
        self.label_len = label_len
        self.informer_like = informer_like
        self.use_time_encoding = use_time_encoding
        self.cache_implementation = str(cache_implementation).lower()
        if self.cache_implementation not in {"auto", "dynamic", "paged", "static"}:
            raise ValueError(
                "cache_implementation must be auto, dynamic, paged, or static"
            )
        pos_encoder = config.option("pos_encoder")
        mod_budget_scheduler = config.option("mod_budget_scheduler")
        layer_dropout_schedule = config.option("layer_dropout_schedule")
        super().__init__(
            input_size,
            config=config,
            informer_like=informer_like,
            pos_encoder=pos_encoder,
            mod_budget_scheduler=mod_budget_scheduler,
            layer_dropout_schedule=layer_dropout_schedule,
        )
        self.config = config

        self.time_encoder = (
            InformerTimeEmbedding(self.d_model) if use_time_encoding else None
        )

        self.output_projection = (
            nn.Identity()
            if output_size == self.d_model
            else nn.Linear(self.d_model, output_size)
        )

        # BaseTransformer initializes its module tree before decoder-specific
        # modules exist. Apply the same policy here so the forecast head does
        # not keep PyTorch's substantially wider default weights/random bias.
        if self.time_encoder is not None:
            self.time_encoder.apply(self._init_weights)
        self.output_projection.apply(self._init_weights)
        self._cache_manager = DecoderCacheManager(self)
        self._generation_engine = GenerationEngine(self, self._cache_manager)

    def init_static_cache(
        self,
        batch_size: int,
        *,
        max_cache_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict:
        reference = next(self.parameters())
        cache_device = reference.device if device is None else torch.device(device)
        cache_dtype = reference.dtype if dtype is None else dtype
        capacity = self.max_seq_len if max_cache_len is None else int(max_cache_len)
        layer_states: list[DecoderLayerState] = []
        for layer_idx in range(self.num_layers):
            layer = self._get_layer(layer_idx)
            attention = layer._self_attn()
            cache = StaticKVCache(
                batch_size=batch_size,
                num_heads=attention.n_kv_heads,
                max_cache_len=capacity,
                head_dim=attention.head_dim,
                device=cache_device,
                dtype=cache_dtype,
            )
            layer_states.append(
                DecoderLayerState(
                    self_attn=AttentionCacheState(static_cache=cache),
                    cross_attn=AttentionCacheState(),
                )
            )
        return DecoderState(layers=layer_states, cache_implementation="static")

    def _make_layer(
        self,
        config: TransformerConfig,
        layer_attention_type: str,
        dropout: float,
        informer_like: bool,
    ) -> nn.Module:
        return TransformerDecoderLayer(
            config,
            layer_attention_type=layer_attention_type,
            dropout=dropout,
            informer_like=informer_like,
        )

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
        cache_position: torch.Tensor | None = None,
        cache_update_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ):
        output_hidden_states, output_attentions, return_dict = resolve_output_options(
            self.config, output_hidden_states, output_attentions, return_dict
        )
        B, T, _ = tgt.shape
        device = tgt.device
        user_tgt_key_padding_mask = tgt_key_padding_mask

        incremental_state = coerce_decoder_state(
            incremental_state, num_layers=self.num_layers
        )

        if cache_position is None and incremental_state is not None:
            existing_layers = incremental_state.get("layers")
            if existing_layers:
                first_self_state = (existing_layers[0] or {}).get("self_attn", {})
                existing_cache = first_self_state.get("static_cache")
                if isinstance(existing_cache, StaticKVCache):
                    cache_position = existing_cache.lengths[:, None] + torch.arange(
                        T, device=device, dtype=torch.long
                    )

        if cache_position is not None:
            cache_position = cache_position.to(device=device, dtype=torch.long)
            if cache_position.ndim == 1:
                cache_position = cache_position.unsqueeze(0).expand(B, -1)
            if cache_position.shape != (B, T):
                raise ValueError(
                    f"cache_position must be [T] or [B,T], got {tuple(cache_position.shape)}"
                )
            if position_offset is None:
                position_offset = cache_position[:, 0]
        if cache_update_mask is not None:
            cache_update_mask = cache_update_mask.to(device=device, dtype=torch.bool)
            if cache_update_mask.shape != (B,):
                raise ValueError("cache_update_mask must have shape [B]")

        self.mod_aux_loss = 0.0

        compiler = getattr(torch, "compiler", None)
        is_compiling = bool(
            compiler is not None
            and hasattr(compiler, "is_compiling")
            and compiler.is_compiling()
        )
        wants_static_cache = self.cache_implementation == "static" or (
            self.cache_implementation == "auto" and is_compiling
        )
        if (
            incremental_state is None
            and return_incremental_state
            and wants_static_cache
        ):
            incremental_state = self.init_static_cache(
                batch_size=B,
                device=device,
                dtype=tgt.dtype,
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

        # Only apply input-level positional encoding for non-RoPE/ALiBi modes.
        # RoPE and ALiBi handle position encoding internally inside attention.
        if (
            self.pos_encoding_type in ("sinusoidal", "learnable")
            and self.pos_encoder is not None
        ):
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

        # Time encoding
        if (self.time_encoder is not None) and (time_features is not None):
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

        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._create_informer_padding_mask(B, T, device)

        if gateskip_active_mask is None:
            # Paper-like default for time series: all real positions are active.
            # We intentionally derive this only from an actual padding mask, not
            # from the auto-generated Informer masking over the forecast horizon.
            gateskip_active_mask = _gateskip_active_mask_from_padding(
                user_tgt_key_padding_mask
            )

        layer_states = prepare_layer_states(
            incremental_state, num_layers=self.num_layers
        )

        if cache_position is not None:
            for layer_state in layer_states:
                if layer_state is not None:
                    self_state = AttentionCacheState.from_legacy(
                        layer_state.setdefault("self_attn", {})
                    )
                    layer_state["self_attn"] = self_state
                    self_state.cache_position = cache_position
                    if cache_update_mask is not None:
                        self_state.cache_update_mask = cache_update_mask
                    else:
                        self_state.cache_update_mask = None

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
        all_hidden_states: list[torch.Tensor] | None = (
            [x] if output_hidden_states else None
        )
        router_states: list[object] = []
        all_attentions: list[torch.Tensor] = []
        all_cross_attentions: list[torch.Tensor] = []

        for i in range(self.num_layers):
            prev_state = layer_states[i - 1] if i > 0 else None

            if self.use_mod:

                def gather_and_invoke(layer, routed_indices, routed_slots):
                    nonlocal streams
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
                    return x_routed, x_routed_out

                x, was_used = _run_mod_layer(
                    self,
                    i,
                    x,
                    gateskip_active_mask,
                    all_hidden_states,
                    router_states,
                    gather_and_invoke,
                )
                if was_used:
                    used_indices.append(i)
                continue

            layer = self._resolve_layer(i)
            self_attention_module = layer._self_attn()
            cross_attention_module = layer.cross_attn
            if hasattr(self_attention_module, "output_attentions"):
                self_attention_module.output_attentions = output_attentions
            cross_attention_module.output_attentions = output_attentions
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
            if all_hidden_states is not None:
                all_hidden_states.append(x)
            ff_block = getattr(getattr(layer, "feed_forward", None), "block", None)
            router_state = getattr(ff_block, "last_routing_state", None)
            if router_state is not None:
                router_states.append(router_state)
            self_weights = getattr(self_attention_module, "last_attn_weights", None)
            cross_weights = getattr(cross_attention_module, "last_attn_weights", None)
            if self_weights is not None:
                all_attentions.append(self_weights)
            if cross_weights is not None:
                all_cross_attentions.append(cross_weights)

        # FIX: aggregate only over executed layers
        self._finalize_layer_stack(used_indices)

        x = self._finalize_attention_residual_output(attention_residual_state, x)
        x = self.final_norm(x)
        if all_hidden_states is not None:
            all_hidden_states[-1] = x

        out = self.output_projection(x)  # [B, T, output_size]

        if return_incremental_state:
            if incremental_state is None:
                incremental_state = DecoderState.from_legacy(
                    None, num_layers=self.num_layers
                )
            incremental_state["layers"] = layer_states
            if return_dict:
                return build_decoder_output(
                    out,
                    hidden_states=all_hidden_states,
                    state=incremental_state,
                    aux_loss=self.aux_loss,
                    router_states=router_states,
                    attentions=all_attentions,
                    cross_attentions=all_cross_attentions,
                )
            return out, incremental_state

        if return_dict:
            return build_decoder_output(
                out,
                hidden_states=all_hidden_states,
                state=incremental_state,
                aux_loss=self.aux_loss,
                router_states=router_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
        return out

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: dict | None = None,
        time_features: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        cache_update_mask: torch.Tensor | None = None,
    ):
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
        has_static_cache = False
        if incremental_state is not None:
            layers = incremental_state.get("layers") or []
            if layers:
                first_self_state = (layers[0] or {}).get("self_attn", {})
                has_static_cache = isinstance(
                    first_self_state.get("static_cache"), StaticKVCache
                )
        if cache_position is None and not has_static_cache:
            cache_position = torch.arange(
                decoded_len,
                decoded_len + step_tgt.size(1),
                device=step_tgt.device,
                dtype=torch.long,
            )
        call_state = incremental_state
        if call_state is None:
            compiler = getattr(torch, "compiler", None)
            compiling = bool(
                compiler is not None
                and hasattr(compiler, "is_compiling")
                and compiler.is_compiling()
            )
            if self.cache_implementation != "static" and not (
                self.cache_implementation == "auto" and compiling
            ):
                call_state = {}
        out, next_state = self.forward(
            step_tgt,
            memory,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            incremental_state=call_state,
            return_incremental_state=True,
            time_features=step_time_features,
            position_offset=decoded_len,
            cache_position=cache_position,
            cache_update_mask=cache_update_mask,
            return_dict=False,
        )
        next_state["_decoded_len"] = decoded_len + step_tgt.size(1)
        return out, next_state

    def prefill(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        **kwargs,
    ):
        kwargs.pop("incremental_state", None)
        kwargs.pop("return_incremental_state", None)
        return self.forward(
            tgt,
            memory,
            incremental_state=None,
            return_incremental_state=True,
            return_dict=False,
            **kwargs,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: dict,
        **kwargs,
    ):
        if incremental_state is None:
            raise ValueError("decode requires an initialized incremental_state")
        if tgt.size(1) > 1:
            return self.forward_multi_step(
                tgt, memory, incremental_state=incremental_state, **kwargs
            )
        return self.forward_one_step(
            tgt,
            memory,
            incremental_state=incremental_state,
            **kwargs,
        )

    def forward_multi_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: dict,
        **kwargs,
    ):
        if tgt.dim() != 3 or tgt.size(1) < 1:
            raise ValueError("forward_multi_step expects tgt [B,T,C] with T>0")
        output, state = self.forward(
            tgt,
            memory,
            incremental_state=incremental_state,
            return_incremental_state=True,
            return_dict=False,
            **kwargs,
        )
        state["_decoded_len"] = int(state.get("_decoded_len", 0)) + tgt.size(1)
        return output, state

    def reorder_incremental_state(
        self, incremental_state: dict, beam_idx: torch.LongTensor
    ) -> dict:
        return self._cache_manager.reorder(incremental_state, beam_idx)

    def cache_state_dict(self, incremental_state: dict) -> dict:
        return self._cache_manager.state_dict(incremental_state)

    def load_cache_state_dict(self, state: dict, *, device=None) -> dict:
        return self._cache_manager.load_state_dict(state, device=device)

    def offload_cache(self, incremental_state: dict) -> dict:
        return self._cache_manager.offload(incremental_state)

    def save_cache(self, incremental_state: dict, path) -> None:
        self._cache_manager.save(incremental_state, path)

    def load_cache(self, path, *, device=None) -> dict:
        return self._cache_manager.load(path, device=device)

    def speculative_decode(
        self,
        draft_tokens: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: dict,
        *,
        verifier_fn=None,
        **kwargs,
    ):
        return self._generation_engine.speculative_decode(
            draft_tokens,
            memory,
            incremental_state,
            verifier_fn=verifier_fn,
            **kwargs,
        )

    def compile_prefill(self, **compile_options):
        return self._generation_engine.compile_prefill(**compile_options)

    def compile_decode(self, **compile_options):
        return self._generation_engine.compile_decode(**compile_options)

    @torch.no_grad()
    def generate(
        self,
        initial_tgt: torch.Tensor,
        memory: torch.Tensor,
        max_new_tokens: int | None = None,
        *,
        generation_config: GenerationConfig | None = None,
        incremental_state: dict | None = None,
        feedback_fn=None,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ):
        return self._generation_engine.generate(
            initial_tgt,
            memory,
            max_new_tokens,
            generation_config=generation_config,
            incremental_state=incremental_state,
            feedback_fn=feedback_fn,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def beam_search(
        self,
        initial_tgt: torch.Tensor,
        memory: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        proposal_fn,
    ):
        return self._generation_engine.beam_search(
            initial_tgt, memory, max_new_tokens, num_beams, proposal_fn
        )


__all__ = ["TransformerDecoderLayer", "TransformerDecoder"]
