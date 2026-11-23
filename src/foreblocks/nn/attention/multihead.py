"""Configured attention construction and forward orchestration.

MultiAttention composes QKV preparation, position transforms, cache providers,
registered algorithm variants, and kernel dispatch. Private mixins keep feature
initialization, positional transforms, and cache coordination beside their
implementations while registering parameters on this module for checkpoint
compatibility. Concrete algorithms live under variants/ and algorithms/.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.nn.attention.cache.compaction import AttentionMatchingCompactor
from foreblocks.nn.attention.cache.decode_stream import (
    paged_stream_decode_standard,
)
from foreblocks.nn.attention.cache.kv import (
    KVProvider,
)
from foreblocks.nn.attention.cache.runtime import _AttentionCache
from foreblocks.nn.attention.cache.selection import AttentionCacheSelector
from foreblocks.nn.attention.config import AttentionConfig
from foreblocks.nn.attention.execution.backends import (
    ATTENTION_BACKENDS,
    _get_available_backends,
)
from foreblocks.nn.attention.execution.dispatch import AttentionKernelDispatcher
from foreblocks.nn.attention.features import _AttentionFeatures
from foreblocks.nn.attention.preparation.masking import AttentionMaskProcessor
from foreblocks.nn.attention.preparation.pipeline import QKVPipeline
from foreblocks.nn.attention.preparation.position import _AttentionPositions
from foreblocks.nn.attention.preparation.projections import QKVProjector
from foreblocks.nn.attention.variants.base import (
    AttentionImpl,
    AttentionOwner,
    MultiAttentionOwner,
)
from foreblocks.nn.attention.variants.registry import ATTENTION_VARIANTS
from foreblocks.nn.attention.variants.standard import StandardAttentionImpl
from foreblocks.nn.embeddings.symplectic import AdaptiveWarp, SyPERotator
from foreblocks.ops.attention import (
    triton_paged_decode,
)


# ─────────────────────────────────────────────────────────────────────────────
# MultiAttention (with paged cache support)
# ─────────────────────────────────────────────────────────────────────────────
class MultiAttention(
    _AttentionFeatures, _AttentionPositions, _AttentionCache, nn.Module
):
    # --------------------------------------------------------------------- #
    # Init
    # --------------------------------------------------------------------- #
    def __init__(self, config: AttentionConfig):
        super().__init__()

        shape = config.shape
        cache = config.cache
        position = config.position
        variant = config.variant
        features = config.features
        # Basic dims
        if shape.n_heads <= 0 or shape.d_model % shape.n_heads:
            raise ValueError("n_heads must be positive and divide d_model")
        self.d_model = shape.d_model
        self.n_heads = shape.n_heads
        self.head_dim = shape.d_model // shape.n_heads
        self.attention_type = variant.name
        self.output_attentions = False
        self.last_attn_weights: torch.Tensor | None = None
        self.attn_implementation = str(variant.backend).lower()
        ATTENTION_BACKENDS.validate(self.attn_implementation)
        self.cross_attention = shape.cross_attention

        # GQA/MQA
        self.n_kv_heads = shape.n_kv_heads or shape.n_heads
        if self.n_kv_heads <= 0 or shape.n_heads % self.n_kv_heads:
            raise ValueError("n_kv_heads must be positive and divide n_heads")
        self.n_rep = shape.n_heads // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.use_mla = bool(cache.use_mla)
        default_latent = max(32, self.kv_dim // 4)
        self.kv_latent_dim = (
            int(cache.kv_latent_dim)
            if cache.kv_latent_dim is not None
            else int(default_latent)
        )
        self.use_attention_matching_compaction = bool(cache.attention_matching)
        if self.use_attention_matching_compaction and self.use_mla:
            raise ValueError(
                "attention-matching KV compaction currently requires use_mla=False"
            )
        self.attention_matching_keep_ratio = float(cache.matching_keep_ratio)
        self.attention_matching_trigger_len = int(cache.matching_trigger_len)
        self.attention_matching_min_keep = int(cache.matching_min_keep)
        self.attention_matching_query_budget = int(cache.matching_query_budget)
        self.attention_matching_force_single_step = bool(
            cache.matching_force_single_step
        )
        if not (0.0 < self.attention_matching_keep_ratio <= 1.0):
            raise ValueError("attention_matching_keep_ratio must be in (0, 1]")
        if self.attention_matching_min_keep <= 0:
            raise ValueError("attention_matching_min_keep must be > 0")
        if self.attention_matching_query_budget <= 0:
            raise ValueError("attention_matching_query_budget must be > 0")
        self.attention_matching_compactor = (
            AttentionMatchingCompactor(
                keep_ratio=self.attention_matching_keep_ratio,
                trigger_len=self.attention_matching_trigger_len,
                min_keep=self.attention_matching_min_keep,
                query_budget=self.attention_matching_query_budget,
                force_single_step=self.attention_matching_force_single_step,
            )
            if self.use_attention_matching_compaction
            else None
        )

        # Attention hyper-parameters
        self.dropout_p = shape.dropout
        self.scale = self.head_dim**-0.5
        self.prob_sparse_factor = variant.probability_factor
        self.window_size = variant.window_size
        self.chunk_size = variant.chunk_size
        self.attention_dilation = int(variant.dilation)
        self.dilated_window_size = (
            int(variant.dilated_window_size)
            if variant.dilated_window_size is not None
            else int(variant.window_size) * max(1, self.attention_dilation)
        )
        self.use_flash_sliding = variant.use_flash_sliding

        self._init_attention_features(features)

        self.softpick_chunk_size = variant.softpick_chunk_size
        self.nsa_block_size = (
            int(variant.nsa_block_size)
            if variant.nsa_block_size is not None
            else int(cache.block_size)
        )
        self.nsa_topk_ratio = (
            float(variant.nsa_topk_ratio)
            if variant.nsa_topk_ratio is not None
            else float(variant.probability_factor)
        )
        self.moba_block_size = (
            int(variant.moba_block_size)
            if variant.moba_block_size is not None
            else int(variant.chunk_size)
        )
        self.moba_topk = int(variant.moba_topk)
        if self.moba_block_size <= 0:
            raise ValueError("moba_block_size must be > 0")
        if self.moba_topk <= 0:
            raise ValueError("moba_topk must be > 0")
        if self.attention_dilation <= 0:
            raise ValueError("attention_dilation must be > 0")
        if self.dilated_window_size <= 0:
            raise ValueError("dilated_window_size must be > 0")

        # Setup projections / type-specific modules
        self._setup_attention_modules(
            variant.name,
            variant.frequency_modes,
            variant.use_swiglu,
            position.encoding,
            position.rope_base,
            position.rope_scaling_type,
            position.rope_scaling_factor,
        )
        self.position_encoding_applier = self._build_position_encoding_applier()
        self._paged_stream_decode = paged_stream_decode_standard
        self._triton_paged_decode = triton_paged_decode
        self.context: AttentionOwner = MultiAttentionOwner(self)
        self._fallback_standard = StandardAttentionImpl(self.context)
        self.impl: AttentionImpl = self._create_impl()

        # Backends
        self.backends = (
            _get_available_backends()
            if variant.name
            in [
                "standard",
                "prob_sparse",
                "softpick",
                "sliding_window",
                "dilated_sliding_window",
                "dilated_window",
                "nsa",
            ]
            else {}
        )

        # Paged cache options
        self.use_paged_cache = cache.use_paged_cache
        self.cache_block_size = cache.block_size
        self.max_cache_blocks = cache.max_blocks
        self.cache_selector = AttentionCacheSelector(self)
        self.qkv_pipeline = QKVPipeline(self)
        self.qkv_projector = QKVProjector(self)
        self.mask_processor = AttentionMaskProcessor()
        self.kernel_dispatcher = AttentionKernelDispatcher(self)

        self._init_gated_attention(features)

        if features.verbose_init:
            gqa_info = (
                f"GQA({shape.n_heads}q/{self.n_kv_heads}kv)"
                if self.n_rep > 1
                else "MHA"
            )
            sota_features = []
            if self.logit_softcap is not None:
                sota_features.append(f"softcap={self.logit_softcap}")
            if self.use_learned_temp:
                sota_features.append("learned_temp")
            if self.use_subquery_norm:
                sota_features.append(f"subquery_norm({self.subquery_norm_mode})")
            if self.use_multiscale_mask:
                sota_features.append(
                    f"multiscale(w={self.multiscale_window_ratio:.2f})"
                )
            if self.use_normalized_attn_out:
                sota_features.append(f"norm_attn({self.norm_attn_type})")
            if self.use_head_importance:
                sota_features.append(
                    f"head_pruning(s={self.head_importance_sparsity:.2f})"
                )
            sota_str = ", ".join(sota_features) if sota_features else "none"
            print(
                f"[MultiAttention] {gqa_info}, type={variant.name}, "
                f"backends={self.backends}, rotary={self.use_rotary}, "
                f"paged_cache={self.use_paged_cache}, "
                f"mla={self.use_mla}(latent={self.kv_latent_dim}), "
                f"attn_match_compact={self.use_attention_matching_compaction}, "
                f"qk_norm={self.qk_norm}({self.qk_norm_type}), "
                f"sota=[{sota_str}]"
            )

    # --------------------------------------------------------------------- #
    # Setup helpers
    # --------------------------------------------------------------------- #
    def _setup_attention_modules(
        self,
        attention_type: str,
        freq_modes: int,
        use_swiglu: bool,
        pos_encoding_type: str = "rope",
        rope_base: float = 10000.0,
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
    ):
        if attention_type in [
            "standard",
            "sype",
            "prob_sparse",
            "softpick",
            "sliding_window",
            "dilated_sliding_window",
            "dilated_window",
            "nsa",
            "moba",
        ]:
            # QKV projections
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)

            # Output projection (optionally a SWiGLU MLP)
            if use_swiglu:
                self.out_proj = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 4, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.d_model * 4, self.d_model, bias=False),
                )
            else:
                self.out_proj = nn.Linear(self.d_model, self.d_model)

            self.dropout = nn.Dropout(self.dropout_p)

            if self.use_mla:
                self.kv_down_proj = nn.Linear(
                    self.d_model, self.kv_latent_dim, bias=False
                )
                self.k_up_proj = nn.Linear(self.kv_latent_dim, self.kv_dim, bias=False)
                self.v_up_proj = nn.Linear(self.kv_latent_dim, self.kv_dim, bias=False)
                self.k_proj = None
                self.v_proj = None
            else:
                self.kv_down_proj = None
                self.k_up_proj = None
                self.v_up_proj = None

            # Positional encoding type (stored for reference)
            self.pos_encoding_type = str(pos_encoding_type)
            # RoPE (self-attention only)
            self.use_rotary = (pos_encoding_type == "rope") and not self.cross_attention
            if self.use_rotary:
                from foreblocks.nn.embeddings.rotary import (
                    RotaryEmbedding,  # local import
                )

                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    base=rope_base,
                    scaling_type=rope_scaling_type,
                    scaling_factor=rope_scaling_factor,
                )
            else:
                self.rotary_emb = None

            # ALiBi (self-attention only; not applicable to cross-attention)
            self.use_alibi = (pos_encoding_type == "alibi") and not self.cross_attention
            if self.use_alibi:
                from foreblocks.nn.embeddings.alibi_bias import ALiBiPositionalBias

                self.alibi_bias = ALiBiPositionalBias(num_heads=self.n_heads)
            else:
                self.alibi_bias = None

            # SyPE (self-attention only)
            self.use_sype = (attention_type == "sype") and (not self.cross_attention)
            if self.use_sype:
                self.sype_warp = AdaptiveWarp(self.d_model)
                self.sype_rotator = SyPERotator(self.head_dim)
            else:
                self.sype_warp = None
                self.sype_rotator = None

            self.nsa_gate_proj = (
                nn.Linear(self.head_dim, 3, bias=True)
                if attention_type == "nsa"
                else None
            )

        else:
            # frequency / dwt / autocor
            self.pos_encoding_type = str(pos_encoding_type)
            self.use_rotary = False
            self.rotary_emb = None
            self.use_sype = False
            self.sype_warp = None
            self.sype_rotator = None
            self.kv_down_proj = None
            self.k_up_proj = None
            self.v_up_proj = None
            self.nsa_gate_proj = None

            if attention_type == "frequency":
                from foreblocks.nn.attention.algorithms.spectral.frequency import (
                    FrequencyAttention,
                )

                self.freq_attention = FrequencyAttention(
                    self.d_model, self.n_heads, self.dropout_p, modes=freq_modes
                )
            elif attention_type == "dwt":
                from foreblocks.nn.attention.algorithms.spectral.wavelet import (
                    DWTAttention,
                )

                self.dwt_attention = DWTAttention(
                    self.d_model, self.n_heads, self.dropout_p, modes=freq_modes
                )
            elif attention_type == "autocor":
                from foreblocks.nn.attention.algorithms.spectral.autocorrelation import (
                    AutoCorrelation,
                    AutoCorrelationLayer,
                )

                autocorr = AutoCorrelation(
                    mask_flag=True,
                    factor=1,
                    attention_dropout=0.1,
                    output_attention=False,
                )
                self.freq_attention = AutoCorrelationLayer(
                    correlation=autocorr,
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )

    def _create_impl(self) -> AttentionImpl:
        return ATTENTION_VARIANTS.create(self.attention_type, self.context)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = False,
        layer_state: dict[str, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
        key = key if key is not None else query
        value = value if value is not None else key

        result = self.impl.forward(
            query,
            key,
            value,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights or self.output_attentions,
            layer_state=layer_state,
            cu_seqlens=cu_seqlens,
        )
        self.last_attn_weights = result[1]
        return result

    def reset_cache(self):
        for attr_name in ["freq_attention", "dwt_attention"]:
            attn = getattr(self, attr_name, None)
            if attn is not None and hasattr(attn, "cache"):
                attn.cache.clear()

    # --------------------------------------------------------------------- #
    # Small utilities
    # --------------------------------------------------------------------- #
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1)

    def _project_qkv_heads(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        return self.qkv_projector.project(query, key, value)

    def _finalize_projected_output(
        self,
        out_bhtd: torch.Tensor,
        B: int,
        T_q: int,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_gated_attention and self.gated_attn_mode == "query_head":
            if query is None:
                raise ValueError("query_head gate requires query hidden states")
            gate = self.gate_proj(query).sigmoid().transpose(1, 2).unsqueeze(-1)
            out_bhtd = out_bhtd * gate
        # Apply head importance mask (before merging heads)
        out_bhtd = self._apply_head_importance_mask(out_bhtd)

        out = out_bhtd.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.dropout(out)
        # Apply normalization before output projection (LeLA-style)
        out = self._apply_normalized_attn_out(out)
        return self.out_proj(out)

    def _create_sliding_window_mask(
        self,
        T_q: int,
        T_k: int,
        device: torch.device,
        is_causal: bool = True,
        q_start_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mask_processor.sliding_window(
            T_q,
            T_k,
            window_size=self.window_size,
            device=device,
            is_causal=is_causal,
            q_start_pos=q_start_pos,
        )

    def _apply_masks(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.mask_processor.apply(scores, attn_mask, key_padding_mask)

    def _normalize_attn_mask(
        self,
        attn_mask: torch.Tensor,
        B: int,
        H: int,
        T_q: int,
        T_k: int,
    ) -> torch.Tensor:
        return self.mask_processor.normalize(
            attn_mask,
            batch_size=B,
            num_heads=H,
            query_length=T_q,
            key_length=T_k,
        )

    def _slice_attn_mask(
        self,
        attn_mask: torch.Tensor | None,
        B: int,
        H: int,
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        T_q_full: int,
        T_k_full: int,
    ) -> torch.Tensor | None:
        return self.mask_processor.slice(
            attn_mask,
            batch_size=B,
            num_heads=H,
            query_slice=slice(q_start, q_end),
            key_slice=slice(k_start, k_end),
            query_length=T_q_full,
            key_length=T_k_full,
        )

    def _dropout_weights(self, w: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout_p > 0:
            return F.dropout(w, p=self.dropout_p, training=True)
        return w

    # --------------------------------------------------------------------- #
    # QKV processing (non-paged + paged gather compatibility)
    # --------------------------------------------------------------------- #
    def _make_kv_provider(
        self,
        layer_state: dict | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        force_paged: bool | None = None,
    ) -> KVProvider:
        return self.cache_selector.select(
            layer_state,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            force_paged=force_paged,
        )

    def _prepare_qkv_with_provider(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: dict | None,
        *,
        force_paged: bool | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        KVProvider,
        torch.Tensor | None,
    ]:
        return self.qkv_pipeline.prepare(
            query, key, value, layer_state, force_paged=force_paged
        )

    def _process_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, kv_latent, provider, _ = self._prepare_qkv_with_provider(
            query,
            key,
            value,
            layer_state,
        )

        if not self.cross_attention:
            k, v = provider.get_kv(k, v, kv_latent=kv_latent)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        return q, k, v

    def _process_qkv_paged(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, kv_latent, provider, _ = self._prepare_qkv_with_provider(
            query,
            key,
            value,
            layer_state,
            force_paged=True,
        )

        if not self.cross_attention:
            k, v = provider.get_kv(k, v, kv_latent=kv_latent)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        return q, k, v

    def _prepare_qkv_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        q, k, v, kv_latent, provider, q_start_pos = self._prepare_qkv_with_provider(
            query,
            key,
            value,
            layer_state,
        )

        if not self.cross_attention:
            k, v = provider.get_kv(k, v, kv_latent=kv_latent)

        return q, k, v, q_start_pos

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        q_start_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.kernel_dispatcher.compute(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
            q_start_pos,
        )
