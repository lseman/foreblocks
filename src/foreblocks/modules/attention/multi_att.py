"""foreblocks.modules.attention.multi_att.

Multi-backend attention with SOTA enhancements (logit softcapping, learned temp, subquery norm).

Unified attention module supporting 8+ attention variants (standard, ProbSparse, sliding-window,
softpick, frequency/DWT/AutoCor, NSA, MoBA) across multiple backends (FlashAttention, xFormers,
SDPA). SOTA features: logit softcapping (Gemini-style), per-head learnable temperature,
subquery normalization (DeepSeek-style), multi-scale masking (hierarchical local+long-range),
GQA/MQA, RoPE, paged KV cache for autoregressive decode, QK normalization, gated attention,
and MLA (latent bottleneck).

Core API:
- MultiAttention: multi-backend SOTA attention with optional logit softcap, learned temp, subquery norm
- _get_available_backends: detect available attention backends

SOTA Features (all optional, disabled by default):
- logit_softcap: tanh-softcap pre-softmax scores to (-cap, cap) for stability (Gemma-style)
- use_learned_temp: per-head learnable temperature scaling
- use_subquery_norm: learnable per-QK-pair gating (DeepSeek-style)
- use_multiscale_mask: hierarchical attention (local window + top-K long-range)
- use_normalized_attn_out: layer norm on attention output for residual stability (LeLA)
- use_head_importance: learn which heads to prune (Lottery Ticket + sparsity regularization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.models.transformer.features.sype import AdaptiveWarp, SyPERotator
from foreblocks.modules.attention.backends import ATTENTION_BACKENDS
from foreblocks.modules.attention.cache.decode_stream import (
    paged_stream_decode_standard,
)
from foreblocks.modules.attention.cache.kv import (
    KVProvider,
)
from foreblocks.modules.attention.cache.paged import PagedKVCache
from foreblocks.modules.attention.cache.selection import AttentionCacheSelector
from foreblocks.modules.attention.config import AttentionConfig
from foreblocks.modules.attention.dispatch import AttentionKernelDispatcher
from foreblocks.modules.attention.masking import (
    normalize_blocked_mask,
)
from foreblocks.modules.attention.pipeline import QKVPipeline
from foreblocks.modules.attention.utils.compaction import (
    AttentionMatchingCompactor,
    AttentionMatchingConfig,
)
from foreblocks.modules.attention.utils.position import PositionEncodingApplier
from foreblocks.modules.attention.variants.base import (
    AttentionContext,
    AttentionImpl,
    MultiAttentionContext,
)
from foreblocks.modules.attention.variants.registry import ATTENTION_VARIANTS
from foreblocks.modules.attention.variants.standard import StandardAttentionImpl
from foreblocks.ops.attention import triton_apply_rope, triton_paged_decode


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_available_backends() -> dict[str, bool]:
    backends = {"flash": False, "xformers": False, "sdp": False, "softpick": False}

    # FlashAttention
    try:
        from flash_attn import flash_attn_func  # noqa: F401

        backends["flash"] = True
    except ImportError:
        pass

    # xFormers
    try:
        import xformers.ops  # noqa: F401

        backends["xformers"] = True
    except ImportError:
        pass

    # PyTorch scaled_dot_product_attention
    backends["sdp"] = hasattr(F, "scaled_dot_product_attention")

    # Optional third_party SoftPick
    try:
        from foreblocks.third_party.flash_softpick_attn import (  # noqa: F401
            parallel_softpick_attn,
        )

        backends["softpick"] = True
    except ImportError:
        pass

    return backends


# ─────────────────────────────────────────────────────────────────────────────
# MultiAttention (with paged cache support)
# ─────────────────────────────────────────────────────────────────────────────
class MultiAttention(nn.Module):
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
        d_model, n_heads = shape.d_model, shape.n_heads
        n_kv_heads, dropout = shape.n_kv_heads, shape.dropout
        max_seq_len, cross_attention = shape.max_seq_len, shape.cross_attention
        attention_type, attn_implementation = variant.name, variant.backend
        prob_sparse_factor, freq_modes = (
            variant.probability_factor,
            variant.frequency_modes,
        )
        softpick_chunk_size, window_size = (
            variant.softpick_chunk_size,
            variant.window_size,
        )
        global_attention_ratio, chunk_size = (
            variant.global_attention_ratio,
            variant.chunk_size,
        )
        use_flash_sliding, use_swiglu = variant.use_flash_sliding, variant.use_swiglu
        nsa_block_size, nsa_topk_ratio = variant.nsa_block_size, variant.nsa_topk_ratio
        moba_block_size, moba_topk = variant.moba_block_size, variant.moba_topk
        attention_dilation = variant.dilation
        dilated_window_size = variant.dilated_window_size
        pos_encoding_type, rope_base = position.encoding, position.rope_base
        rope_scaling_type, rope_scaling_factor = (
            position.rope_scaling_type,
            position.rope_scaling_factor,
        )
        use_paged_cache, cache_block_size = cache.use_paged_cache, cache.block_size
        max_cache_blocks, use_mla = cache.max_blocks, cache.use_mla
        kv_latent_dim = cache.kv_latent_dim
        use_attention_matching_compaction = cache.attention_matching
        attention_matching_keep_ratio = cache.matching_keep_ratio
        attention_matching_trigger_len = cache.matching_trigger_len
        attention_matching_min_keep = cache.matching_min_keep
        attention_matching_query_budget = cache.matching_query_budget
        attention_matching_force_single_step = cache.matching_force_single_step
        qk_norm, qk_norm_type = features.qk_norm, features.qk_norm_type
        logit_softcap = features.logit_softcap
        use_learned_temp, temp_init = (
            features.learned_temperature,
            features.temperature_init,
        )
        use_gated_attention = features.gated_attention
        gated_attn_mode, gated_attn_bias = (
            features.gated_attention_mode,
            features.gated_attention_bias,
        )
        use_subquery_norm, subquery_norm_mode = (
            features.subquery_norm,
            features.subquery_norm_mode,
        )
        use_multiscale_mask = features.multiscale_mask
        multiscale_window_ratio, multiscale_topk = (
            features.multiscale_window_ratio,
            features.multiscale_topk,
        )
        use_normalized_attn_out = features.normalized_output
        norm_attn_type = features.normalized_output_type
        use_head_importance = features.head_importance
        head_importance_sparsity = features.head_importance_sparsity
        verbose_init = features.verbose_init

        # Basic dims
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError("n_heads must be positive and divide d_model")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type
        self.output_attentions = False
        self.last_attn_weights: torch.Tensor | None = None
        self.attn_implementation = str(attn_implementation).lower()
        ATTENTION_BACKENDS.validate(self.attn_implementation)
        self.cross_attention = cross_attention

        # GQA/MQA
        self.n_kv_heads = n_kv_heads or n_heads
        if self.n_kv_heads <= 0 or n_heads % self.n_kv_heads:
            raise ValueError("n_kv_heads must be positive and divide n_heads")
        self.n_rep = n_heads // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.use_mla = bool(use_mla)
        default_latent = max(32, self.kv_dim // 4)
        self.kv_latent_dim = (
            int(kv_latent_dim) if kv_latent_dim is not None else int(default_latent)
        )
        self.use_attention_matching_compaction = bool(use_attention_matching_compaction)
        if self.use_attention_matching_compaction and self.use_mla:
            raise ValueError(
                "attention-matching KV compaction currently requires use_mla=False"
            )
        self.attention_matching_keep_ratio = float(attention_matching_keep_ratio)
        self.attention_matching_trigger_len = int(attention_matching_trigger_len)
        self.attention_matching_min_keep = int(attention_matching_min_keep)
        self.attention_matching_query_budget = int(attention_matching_query_budget)
        self.attention_matching_force_single_step = bool(
            attention_matching_force_single_step
        )
        if not (0.0 < self.attention_matching_keep_ratio <= 1.0):
            raise ValueError("attention_matching_keep_ratio must be in (0, 1]")
        if self.attention_matching_min_keep <= 0:
            raise ValueError("attention_matching_min_keep must be > 0")
        if self.attention_matching_query_budget <= 0:
            raise ValueError("attention_matching_query_budget must be > 0")
        self.attention_matching_compactor = (
            AttentionMatchingCompactor(
                AttentionMatchingConfig(
                    keep_ratio=self.attention_matching_keep_ratio,
                    trigger_len=self.attention_matching_trigger_len,
                    min_keep=self.attention_matching_min_keep,
                    query_budget=self.attention_matching_query_budget,
                    force_single_step=self.attention_matching_force_single_step,
                )
            )
            if self.use_attention_matching_compaction
            else None
        )

        # Attention hyper-parameters
        self.dropout_p = dropout
        self.scale = self.head_dim**-0.5
        self.prob_sparse_factor = prob_sparse_factor
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.attention_dilation = int(attention_dilation)
        self.dilated_window_size = (
            int(dilated_window_size)
            if dilated_window_size is not None
            else int(window_size) * max(1, self.attention_dilation)
        )
        self.use_flash_sliding = use_flash_sliding

        # ── QK Normalization ───────────────────────────────────────────────
        self.qk_norm = bool(qk_norm)
        if qk_norm_type not in {"rms", "l2"}:
            raise ValueError(f"qk_norm_type must be 'rms' or 'l2', got {qk_norm_type}")
        self.qk_norm_type = qk_norm_type
        if self.qk_norm and self.qk_norm_type == "rms":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        # ── Logit softcapping (Gemini-style) ───────────────────────────────
        self.logit_softcap = float(logit_softcap) if logit_softcap is not None else None
        self.use_learned_temp = bool(use_learned_temp)
        if self.use_learned_temp:
            # Per-head learnable temperature
            self.temperature = nn.Parameter(torch.ones(self.n_heads) * float(temp_init))
        else:
            self.temperature = None

        # ── Subquery normalization (DeepSeek-style) ────────────────────────
        self.use_subquery_norm = bool(use_subquery_norm)
        self.subquery_norm_mode = str(subquery_norm_mode)
        if self.use_subquery_norm:
            if self.subquery_norm_mode == "learned":
                # Per-head learnable gate (applied to attention output)
                self.subquery_gate = nn.Linear(self.head_dim, self.head_dim, bias=True)
            else:
                self.subquery_gate = None
        else:
            self.subquery_gate = None

        # ── Multi-scale masking (hierarchical attention) ───────────────────
        self.use_multiscale_mask = bool(use_multiscale_mask)
        self.multiscale_window_ratio = float(multiscale_window_ratio)
        self.multiscale_topk = int(multiscale_topk)
        if self.use_multiscale_mask:
            if not (0.0 < self.multiscale_window_ratio < 1.0):
                raise ValueError("multiscale_window_ratio must be in (0, 1)")
            if self.multiscale_topk <= 0:
                raise ValueError("multiscale_topk must be > 0")

        # ── Normalized attention output (LeLA-style) ──────────────────────
        self.use_normalized_attn_out = bool(use_normalized_attn_out)
        self.norm_attn_type = str(norm_attn_type)
        if self.use_normalized_attn_out:
            if norm_attn_type == "rms":
                self.attn_out_norm = nn.RMSNorm(self.d_model, eps=1e-5)
            elif norm_attn_type == "layer":
                self.attn_out_norm = nn.LayerNorm(self.d_model, eps=1e-5)
            else:
                raise ValueError(
                    f"norm_attn_type must be 'rms' or 'layer', got {norm_attn_type}"
                )
        else:
            self.attn_out_norm = None

        # ── Head importance pruning (Lottery Ticket style) ──────────────────
        self.use_head_importance = bool(use_head_importance)
        self.head_importance_sparsity = float(head_importance_sparsity)
        if self.use_head_importance:
            if not (0.0 <= self.head_importance_sparsity < 1.0):
                raise ValueError("head_importance_sparsity must be in [0, 1)")
            # Learnable head importance scores (per head)
            self.head_importance_scores = nn.Parameter(torch.ones(self.n_heads))
        else:
            self.head_importance_scores = None

        self.softpick_chunk_size = softpick_chunk_size
        self.nsa_block_size = (
            int(nsa_block_size) if nsa_block_size is not None else int(cache_block_size)
        )
        self.nsa_topk_ratio = (
            float(nsa_topk_ratio)
            if nsa_topk_ratio is not None
            else float(prob_sparse_factor)
        )
        self.moba_block_size = (
            int(moba_block_size) if moba_block_size is not None else int(chunk_size)
        )
        self.moba_topk = int(moba_topk)
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
            attention_type,
            freq_modes,
            use_swiglu,
            pos_encoding_type,
            rope_base,
            rope_scaling_type,
            rope_scaling_factor,
        )
        self.position_encoding_applier = self._build_position_encoding_applier()
        self._paged_stream_decode = paged_stream_decode_standard
        self._triton_paged_decode = triton_paged_decode
        self.context: AttentionContext = MultiAttentionContext(self)
        self._fallback_standard = StandardAttentionImpl(self.context)
        self.impl: AttentionImpl = self._create_impl()

        # Backends
        self.backends = (
            _get_available_backends()
            if attention_type
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
        self.use_paged_cache = use_paged_cache
        self.cache_block_size = cache_block_size
        self.max_cache_blocks = max_cache_blocks
        self.cache_selector = AttentionCacheSelector(self)
        self.qkv_pipeline = QKVPipeline(self)
        self.kernel_dispatcher = AttentionKernelDispatcher(self)

        # ── Gated Attention (G1) ─────────────────────────────────────────────
        self.use_gated_attention = use_gated_attention
        self.gated_attn_mode = gated_attn_mode

        if self.use_gated_attention:
            if gated_attn_mode not in {"per_head", "shared"}:
                raise ValueError(
                    f"gated_attn_mode must be 'per_head' or 'shared', got {gated_attn_mode}"
                )

            if gated_attn_mode == "per_head":
                # one gate per head and per channel in head_dim
                # out: [B,H,T,D] → linear on last dim
                self.gate_proj = nn.Linear(
                    self.head_dim, self.head_dim, bias=gated_attn_bias
                )
            else:
                # shared gate across heads (still per channel)
                self.gate_proj = nn.Linear(
                    self.head_dim, self.head_dim, bias=gated_attn_bias
                )
        else:
            self.gate_proj = None

        if verbose_init:
            gqa_info = (
                f"GQA({n_heads}q/{self.n_kv_heads}kv)" if self.n_rep > 1 else "MHA"
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
                f"[MultiAttention] {gqa_info}, type={attention_type}, "
                f"backends={self.backends}, rotary={self.use_rotary}, "
                f"paged_cache={self.use_paged_cache}, "
                f"mla={self.use_mla}(latent={self.kv_latent_dim}), "
                f"attn_match_compact={self.use_attention_matching_compaction}, "
                f"qk_norm={self.qk_norm}({self.qk_norm_type}), "
                f"sota=[{sota_str}]"
            )

    def _apply_logit_softcap(self, scores: torch.Tensor) -> torch.Tensor:
        if self.logit_softcap is None:
            return scores
        return torch.tanh(scores / self.logit_softcap) * self.logit_softcap

    def _apply_learned_temperature(self, scores: torch.Tensor) -> torch.Tensor:
        if self.temperature is None:
            return scores
        # scores: [B, H, T_q, T_k]
        # temperature: [H]
        return scores * self.temperature.view(1, -1, 1, 1)

    def _apply_subquery_norm(self, out_bhtd: torch.Tensor) -> torch.Tensor:
        if not self.use_subquery_norm:
            return out_bhtd
        if self.subquery_norm_mode == "learned" and self.subquery_gate is not None:
            # Gate learned from output itself (similar to gated attention)
            gate = torch.sigmoid(self.subquery_gate(out_bhtd))  # [B,H,T,D]
            return out_bhtd * gate
        elif self.subquery_norm_mode == "rms":
            # RMS normalize per token
            return F.normalize(out_bhtd, p=2.0, dim=-1)
        return out_bhtd

    def _create_multiscale_mask(
        self,
        T_q: int,
        T_k: int,
        device: torch.device,
        is_causal: bool = True,
    ) -> torch.Tensor:
        if not self.use_multiscale_mask:
            return None

        cache_key = (T_q, T_k, bool(is_causal), device)
        cached = getattr(self, "_multiscale_mask_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        window_size = max(1, int(T_k * self.multiscale_window_ratio))
        i = torch.arange(T_q, device=device).unsqueeze(1)  # [T_q, 1]
        j = torch.arange(T_k, device=device).unsqueeze(0)  # [1, T_k]

        # Local window
        if is_causal:
            local_mask = (j >= i - window_size) & (j <= i + window_size // 2)
        else:
            local_mask = (j >= i - window_size // 2) & (j <= i + window_size // 2)

        # Top-K long-range: every stride-th position in [0, i - window_size)
        if is_causal:
            limit = (i - window_size).clamp(min=0)  # [T_q, 1]
            stride = (limit // self.multiscale_topk).clamp(min=1)  # [T_q, 1]
            ks = torch.arange(self.multiscale_topk, device=device).unsqueeze(0)
            pos = ks * stride  # [T_q, K]
            valid = pos < limit  # implies i > window_size
            rows = i.expand_as(pos)
            local_mask[rows[valid], pos[valid]] = True

        self._multiscale_mask_cache = (cache_key, local_mask)
        return local_mask  # True = attend, False = mask

    def _apply_head_importance_mask(
        self,
        out_bhtd: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_head_importance or self.head_importance_scores is None:
            return out_bhtd

        # Compute soft mask: sigmoid of learned scores
        # At inference, can prune heads with low scores
        head_mask = torch.sigmoid(self.head_importance_scores)  # [H]
        return out_bhtd * head_mask.view(1, -1, 1, 1)

    def get_head_importance_loss(self) -> torch.Tensor | None:
        if not self.use_head_importance or self.head_importance_scores is None:
            return None

        # L1 sparsity on importance scores to encourage pruning
        importance = torch.sigmoid(self.head_importance_scores)
        # Target: sparsity% of heads should have low importance
        target = torch.tensor(
            self.head_importance_sparsity,
            device=importance.device,
            dtype=importance.dtype,
        )
        return torch.abs(importance.mean() - (1.0 - target)).mean()

    def _apply_normalized_attn_out(
        self,
        out_btd: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_normalized_attn_out or self.attn_out_norm is None:
            return out_btd
        return self.attn_out_norm(out_btd)

    def _apply_gated_attention(self, out_bhtd: torch.Tensor) -> torch.Tensor:
        if not self.use_gated_attention:
            return out_bhtd

        # gate computed from attention output itself (G1)
        # per_head: independent per-head gating (applied directly)
        if self.gated_attn_mode == "per_head":
            gate = torch.sigmoid(self.gate_proj(out_bhtd))  # [B,H,T,D]
            return out_bhtd * gate

        # shared mode: compute gate from head-mean then broadcast to heads
        # (head-shared is usually worse per paper, but useful to keep as option)
        out_mean = out_bhtd.mean(dim=1, keepdim=True)  # [B,1,T,D]
        gate = torch.sigmoid(self.gate_proj(out_mean))  # [B,1,T,D]
        return out_bhtd * gate

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
                from foreblocks.layers.embeddings.rotary import (
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
                from foreblocks.layers.embeddings.alibi_bias import ALiBiPositionalBias

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
                from foreblocks.modules.attention.implementations.frequency_att import (
                    FrequencyAttention,
                )

                self.freq_attention = FrequencyAttention(
                    self.d_model, self.n_heads, self.dropout_p, modes=freq_modes
                )
            elif attention_type == "dwt":
                from foreblocks.modules.attention.implementations.dwt_att import (
                    DWTAttention,
                )

                self.dwt_attention = DWTAttention(
                    self.d_model, self.n_heads, self.dropout_p, modes=freq_modes
                )
            elif attention_type == "autocor":
                from foreblocks.modules.attention.implementations.autocor_att import (
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

    def _build_position_encoding_applier(self) -> PositionEncodingApplier:
        applier = PositionEncodingApplier()
        applier.add_transform("rope", self._rope_position_transform)
        applier.add_transform("sype", self._sype_position_transform)
        # ALiBi is applied in _compute_attention, not here
        return applier

    def _create_impl(self) -> AttentionImpl:
        return ATTENTION_VARIANTS.create(self.attention_type, self.context)

    def register_position_transform(self, name: str, fn) -> None:
        self.position_encoding_applier.add_transform(name, fn)

    def _rope_position_transform(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        context: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (self.use_rotary and self.rotary_emb is not None):
            return q, k
        seqlen_offset = context.get("seqlen_offset", 0)
        return self._apply_rope(q, k, seqlen_offset=seqlen_offset)

    def _sype_position_transform(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        context: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_sype:
            return q, k
        query = context.get("query")
        key = context.get("key")
        layer_state = context.get("layer_state")
        if query is None or key is None:
            return q, k
        if query.shape[1] != key.shape[1]:
            return q, k
        tau = self.sype_warp(query)
        if layer_state is not None:
            prev_tau = layer_state.get("sype_tau")
            if prev_tau is not None:
                prev_tau = prev_tau.to(device=tau.device, dtype=tau.dtype)
                if prev_tau.dim() == 0:
                    prev_tau = prev_tau.view(1)
                if prev_tau.dim() != 1 or prev_tau.shape[0] != tau.shape[0]:
                    raise ValueError(
                        f"sype_tau state must be [B], got {tuple(prev_tau.shape)} for batch {tau.shape[0]}"
                    )
                tau = tau + prev_tau.unsqueeze(1)
            layer_state["sype_tau"] = tau[:, -1].detach()
        return self.sype_rotator.rotate_qk(q, k, tau)

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

    def reset_paged_cache(self, layer_state: dict[str, torch.Tensor] | None):
        if layer_state and "paged_cache" in layer_state:
            cache: PagedKVCache = layer_state["paged_cache"]
            for b in range(cache.B):
                cache.reset_seq(b)

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
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_mla:
            if (
                self.kv_down_proj is None
                or self.k_up_proj is None
                or self.v_up_proj is None
            ):
                raise RuntimeError("MLA projections are not initialized.")
            kv_latent = self.kv_down_proj(key)  # [B, T_k, L]
            k = (
                self.k_up_proj(kv_latent)
                .view(B, T_k, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_up_proj(kv_latent)
                .view(B, T_k, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            return q, k, v, kv_latent

        k = (
            self.k_proj(key)
            .view(B, T_k, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(B, T_k, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        return q, k, v, None

    def _finalize_projected_output(
        self,
        out_bhtd: torch.Tensor,
        B: int,
        T_q: int,
    ) -> torch.Tensor:
        # Apply head importance mask (before merging heads)
        out_bhtd = self._apply_head_importance_mask(out_bhtd)

        out = out_bhtd.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.dropout(out)
        # Apply normalization before output projection (LeLA-style)
        out = self._apply_normalized_attn_out(out)
        return self.out_proj(out)

    def _ensure_paged_cache(
        self,
        layer_state: dict,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PagedKVCache:
        return PagedKVCache.ensure(
            layer_state,
            batch_size=batch_size,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            latent_dim=(self.kv_latent_dim if self.use_mla else None),
            block_size=self.cache_block_size,
            device=device,
            dtype=dtype,
            max_blocks=self.max_cache_blocks,
        )

    def _can_apply_attention_matching_compaction(
        self,
        *,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        need_weights: bool,
        cache: PagedKVCache,
        t_new: int,
    ) -> bool:
        if not self.use_attention_matching_compaction:
            return False
        if self.attention_matching_compactor is None:
            return False
        if self.cross_attention or cache.use_latent_cache:
            return False
        if attn_mask is not None or key_padding_mask is not None or need_weights:
            return False
        if self.training or self.dropout_p != 0.0:
            return False
        for b in range(cache.B):
            if self.attention_matching_compactor.should_compact(cache, b, t_new=t_new):
                return True
        return False

    def _maybe_compact_paged_cache(
        self,
        *,
        cache: PagedKVCache,
        q: torch.Tensor,
        q_start_pos: torch.Tensor | None,
        t_new: int,
    ) -> None:
        if self.attention_matching_compactor is None:
            return
        if q_start_pos is None:
            raise ValueError(
                "q_start_pos is required for attention-matching compaction"
            )
        self.attention_matching_compactor.compact_batch(
            cache=cache,
            q_bhtd=q,
            q_start_pos=q_start_pos,
            kv_repeat=self.n_rep,
            scale=self.scale,
            t_new=t_new,
        )

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset=0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from foreblocks.layers.embeddings.rotary import (
            apply_rotary_emb,  # keep as-is for API compatibility
        )

        B, H, T, D = q.shape
        if isinstance(seqlen_offset, int):
            max_len = T + seqlen_offset
        else:
            max_len = T + int(seqlen_offset.max().item())

        # Update cached cos/sin up to max_len
        self.rotary_emb._update_cos_sin_cache(
            max_len,
            device=q.device,
            dtype=q.dtype,
        )

        # Switch to [B, T, H, D] for apply_rotary_emb
        q_bt_hd = q.transpose(1, 2).contiguous()
        k_bt_hd = k.transpose(1, 2).contiguous()

        cos_q, sin_q = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        if getattr(self.rotary_emb, "scale", None) is None:
            cos_k, sin_k = cos_q, sin_q
        else:
            cos_k, sin_k = self.rotary_emb._cos_k_cached, self.rotary_emb._sin_k_cached

        use_triton_rope = (
            q.is_cuda
            and k.is_cuda
            and isinstance(seqlen_offset, int)
            and (not bool(self.rotary_emb.interleaved))
            and cos_q.dim() == 2
            and sin_q.dim() == 2
            and cos_k.dim() == 2
            and sin_k.dim() == 2
        )
        if use_triton_rope:
            try:
                if cos_k is cos_q and sin_k is sin_q:
                    # Common path (no rotary scale): one fused launch does q and k
                    q_tr, k_tr = triton_apply_rope(
                        q,
                        k,
                        cos_q,
                        sin_q,
                        seqlen_offset=seqlen_offset,
                    )
                else:
                    q_tr, _ = triton_apply_rope(
                        q,
                        k,
                        cos_q,
                        sin_q,
                        seqlen_offset=seqlen_offset,
                    )
                    _, k_tr = triton_apply_rope(
                        q,
                        k,
                        cos_k,
                        sin_k,
                        seqlen_offset=seqlen_offset,
                    )
                return q_tr.contiguous(), k_tr.contiguous()
            except Exception:
                pass

        q_bt_hd = apply_rotary_emb(
            q_bt_hd,
            cos_q,
            sin_q,
            interleaved=self.rotary_emb.interleaved,
            seqlen_offsets=seqlen_offset,
            inplace=False,
        )
        k_bt_hd = apply_rotary_emb(
            k_bt_hd,
            cos_k,
            sin_k,
            interleaved=self.rotary_emb.interleaved,
            seqlen_offsets=seqlen_offset,
            inplace=False,
        )

        # Back to [B, H, T, D]
        return (
            q_bt_hd.transpose(1, 2).contiguous(),
            k_bt_hd.transpose(1, 2).contiguous(),
        )

    def _create_sliding_window_mask(
        self,
        T_q: int,
        T_k: int,
        device: torch.device,
        is_causal: bool = True,
    ) -> torch.Tensor:
        i = torch.arange(T_q, device=device).unsqueeze(1)  # [T_q, 1]
        j = torch.arange(T_k, device=device).unsqueeze(0)  # [1, T_k]

        if is_causal:
            # Only attend to keys within [i - window_size + 1, i] and j <= i
            return (j > i) | (j < (i - self.window_size + 1))
        else:
            half = self.window_size // 2
            return (j < (i - half)) | (j > (i + half))

    def _apply_masks(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, H, T_q, T_k = scores.shape

        if attn_mask is not None:
            mask = self._normalize_attn_mask(attn_mask, B, H, T_q, T_k)
            scores = scores.masked_fill(mask, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(B, 1, 1, T_k).bool(),
                float("-inf"),
            )

        return scores

    def _normalize_attn_mask(
        self,
        attn_mask: torch.Tensor,
        B: int,
        H: int,
        T_q: int,
        T_k: int,
    ) -> torch.Tensor:
        return normalize_blocked_mask(
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
        if attn_mask is None:
            return None
        full = self._normalize_attn_mask(attn_mask, B, H, T_q_full, T_k_full)
        return full[:, :, q_start:q_end, k_start:k_end]

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
