# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sype import AdaptiveWarp, SyPERotator
from .decode_stream import paged_stream_decode_standard
from .kernels import triton_apply_rope, triton_paged_decode
from .kv import DenseKVProvider, KVProvider, PagedKVProvider
from .paged import PagedKVCache
from .position import PositionEncodingApplier
from .variants.base import AttentionImpl
from .variants.standard import StandardAttentionImpl


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_available_backends() -> Dict[str, bool]:
    """
    Detect which attention backends are available in the current environment.

    Returns
    -------
    dict
        {
            "flash": bool,
            "xformers": bool,
            "sdp": bool,
            "softpick": bool,
        }
    """
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

    # Optional third-party SoftPick
    try:
        from ...third_party.flash_softpick_attn import (  # noqa: F401
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
    """
    Multi-backend attention with GQA/MQA support + optional paged KV cache.

    Features
    --------
    - GQA / MQA
    - RoPE for self-attention
    - Attention types:
        * standard
        * prob_sparse
        * softpick
        * sliding_window
        * frequency / dwt / autocor (Autoformer-style)
    - Backends:
        * PyTorch SDPA
        * FlashAttn
        * xFormers
        * SoftPick (optional third-party)
    - Boolean mask semantics (True = masked/disallowed)
    - Paged KV cache for causal self-attention decode (no growing torch.cat; no gather)
    """

    # --------------------------------------------------------------------- #
    # Init
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        attention_type: str = "standard",
        prob_sparse_factor: float = 0.4,
        freq_modes: int = 32,
        use_rotary: bool = True,
        max_seq_len: int = 4096,  # kept for API, not used here
        cross_attention: bool = False,
        softpick_chunk_size: int = 128,
        window_size: int = 64,
        global_attention_ratio: float = 0.1,  # kept for API
        chunk_size: int = 1024,
        use_flash_sliding: bool = True,
        use_swiglu: bool = True,
        verbose_init: bool = False,
        # NEW:
        use_paged_cache: bool = True,
        cache_block_size: int = 128,
        max_cache_blocks: int = 2048,
        # NEW: Gated Attention (G1) after SDPA output
        use_gated_attention: bool = False,
        gated_attn_mode: str = "per_head",  # {"per_head", "shared"}
        gated_attn_bias: bool = True,
        # NEW: Multi-head Latent Attention (MLA)
        use_mla: bool = True,
        kv_latent_dim: Optional[int] = None,
        # NEW: NSA-specific knobs
        nsa_block_size: Optional[int] = None,
        nsa_topk_ratio: Optional[float] = None,
    ):
        super().__init__()

        # Basic dims
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type
        self.cross_attention = cross_attention

        # GQA/MQA
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.use_mla = bool(use_mla)
        default_latent = max(32, self.kv_dim // 4)
        self.kv_latent_dim = (
            int(kv_latent_dim) if kv_latent_dim is not None else int(default_latent)
        )

        # Attention hyper-parameters
        self.dropout_p = dropout
        self.scale = self.head_dim**-0.5
        self.prob_sparse_factor = prob_sparse_factor
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.use_flash_sliding = use_flash_sliding
        self.softpick_chunk_size = softpick_chunk_size
        self.nsa_block_size = (
            int(nsa_block_size) if nsa_block_size is not None else int(cache_block_size)
        )
        self.nsa_topk_ratio = (
            float(nsa_topk_ratio)
            if nsa_topk_ratio is not None
            else float(prob_sparse_factor)
        )

        # Setup projections / type-specific modules
        self._setup_attention_modules(
            attention_type, freq_modes, use_swiglu, use_rotary
        )
        self.position_encoding_applier = self._build_position_encoding_applier()
        self._paged_stream_decode = paged_stream_decode_standard
        self._triton_paged_decode = triton_paged_decode
        self._fallback_standard = StandardAttentionImpl(self)
        self.impl: AttentionImpl = self._create_impl()

        # Backends
        self.backends = (
            _get_available_backends()
            if attention_type
            in ["standard", "prob_sparse", "softpick", "sliding_window", "nsa"]
            else {}
        )

        # Paged cache options
        self.use_paged_cache = use_paged_cache
        self.cache_block_size = cache_block_size
        self.max_cache_blocks = max_cache_blocks

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
            print(
                f"[MultiAttention] {gqa_info}, type={attention_type}, "
                f"backends={self.backends}, rotary={self.use_rotary}, "
                f"paged_cache={self.use_paged_cache}, "
                f"mla={self.use_mla}(latent={self.kv_latent_dim})"
            )

    def _apply_gated_attention(self, out_bhtd: torch.Tensor) -> torch.Tensor:
        """
        Gated Attention (G1): apply sigmoid gate AFTER attention output.

        out_bhtd: [B, H, T, D]
        returns:  [B, H, T, D]
        """
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
        use_rotary: bool,
    ):
        """Create QKV projections and attention-type-specific modules."""
        if attention_type in [
            "standard",
            "sype",
            "prob_sparse",
            "softpick",
            "sliding_window",
            "nsa",
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

            # RoPE (self-attention only)
            self.use_rotary = use_rotary and not self.cross_attention
            if self.use_rotary:
                from ..embeddings.rotary import RotaryEmbedding  # local import

                self.rotary_emb = RotaryEmbedding(self.head_dim)
            else:
                self.rotary_emb = None

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
                from .multi_att_helper import FrequencyAttention

                self.freq_attention = FrequencyAttention(
                    self.d_model, self.n_heads, self.dropout_p, modes=freq_modes
                )
            elif attention_type == "dwt":
                from .multi_att_helper import DWTAttention

                self.dwt_attention = DWTAttention(
                    self.d_model, self.n_heads, self.dropout_p, modes=freq_modes
                )
            elif attention_type == "autocor":
                from .multi_att_helper import AutoCorrelation, AutoCorrelationLayer

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
        return applier

    def _create_impl(self) -> AttentionImpl:
        from .variants import (
            NSAImpl,
            ProbSparseAttentionImpl,
            SlidingWindowAttentionImpl,
            SoftpickAttentionImpl,
            SpectralAttentionImpl,
            StandardAttentionImpl,
        )

        if self.attention_type in {"standard", "sype"}:
            return StandardAttentionImpl(self)
        if self.attention_type == "prob_sparse":
            return ProbSparseAttentionImpl(self)
        if self.attention_type == "nsa":
            return NSAImpl(self)
        if self.attention_type == "sliding_window":
            return SlidingWindowAttentionImpl(self)
        if self.attention_type == "softpick":
            return SoftpickAttentionImpl(self)
        if self.attention_type in {"frequency", "dwt", "autocor"}:
            return SpectralAttentionImpl(self)
        raise ValueError(f"Unknown attention_type: {self.attention_type}")

    def register_position_transform(self, name: str, fn) -> None:
        """Register additional positional transforms (e.g., ALiBi adapters)."""
        self.position_encoding_applier.add_transform(name, fn)

    def _rope_position_transform(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        context: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (self.use_rotary and self.rotary_emb is not None):
            return q, k
        seqlen_offset = context.get("seqlen_offset", 0)
        return self._apply_rope(q, k, seqlen_offset=seqlen_offset)

    def _sype_position_transform(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        context: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_sype:
            return q, k
        query = context.get("query")
        key = context.get("key")
        if query is None or key is None:
            return q, k
        if query.shape[1] != key.shape[1]:
            return q, k
        tau = self.sype_warp(query)
        return self.sype_rotator.rotate_qk(q, k, tau)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
        layer_state: Optional[Dict[str, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Parameters
        ----------
        query, key, value : torch.Tensor
            [B, T, C] tensors. If key/value are None, fall back to query (self-attn).
        attn_mask : torch.Tensor, optional
            Boolean or additive mask (True = masked) with broadcastable shape.
        key_padding_mask : torch.Tensor, optional
            [B, T_k] boolean mask (True = pad / ignore).
        is_causal : bool
            Whether to use causal masking.
        need_weights : bool
            Whether to return attention weights.
        layer_state : dict, optional
            Cache dict for KV, paged cache, etc.
        cu_seqlens : torch.LongTensor, optional
            For SoftPick backend with packed sequences.

        Returns
        -------
        output : torch.Tensor
            [B, T_q, C]
        attn_weights : torch.Tensor or None
        new_layer_state : dict or None
        """
        key = key if key is not None else query
        value = value if value is not None else key

        return self.impl.forward(
            query,
            key,
            value,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
            layer_state=layer_state,
            cu_seqlens=cu_seqlens,
        )

    def reset_cache(self):
        """Reset any non-paged cached patterns in frequency / DWT modules."""
        for attr_name in ["freq_attention", "dwt_attention"]:
            attn = getattr(self, attr_name, None)
            if attn is not None and hasattr(attn, "cache"):
                attn.cache.clear()

    def reset_paged_cache(self, layer_state: Optional[Dict[str, torch.Tensor]]):
        """Clear paged cache for all batch items (if present in layer_state)."""
        if layer_state and "paged_cache" in layer_state:
            cache: PagedKVCache = layer_state["paged_cache"]
            for b in range(cache.B):
                cache.reset_seq(b)

    # --------------------------------------------------------------------- #
    # Small utilities
    # --------------------------------------------------------------------- #
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA if needed."""
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1)

    def _project_qkv_heads(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Project [B,T,C] inputs into head-first tensors."""
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
        """Merge heads and apply output projection stack."""
        out = out_bhtd.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out))

    def _ensure_paged_cache(
        self,
        layer_state: Dict,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PagedKVCache:
        """Centralized wrapper around PagedKVCache.ensure."""
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

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset=0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to q and k, with an optional per-batch seqlen_offset.
        q, k: [B, H, T, D]
        """
        from foreblocks.tf.embeddings.rotary import (
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
        return q_bt_hd.transpose(1, 2).contiguous(), k_bt_hd.transpose(
            1, 2
        ).contiguous()

    def _create_sliding_window_mask(
        self,
        T_q: int,
        T_k: int,
        device: torch.device,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Boolean window mask: True means masked (disallowed).
        """
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
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply attn_mask and key_padding_mask with "True = masked" semantics.
        scores: [B, H, T_q, T_k]
        """
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
        """
        Normalize attn_mask to bool [B, H, T_q, T_k] (True = masked).
        Supports [T_q,T_k], [B,T_q,T_k], [1,H,T_q,T_k], [B,H,T_q,T_k].
        """
        mask = attn_mask.bool()
        if mask.dim() == 2:
            mask = mask.view(1, 1, T_q, T_k)
        elif mask.dim() == 3:
            if mask.shape[0] in (1, B):
                mask = mask.view(mask.shape[0], 1, T_q, T_k)
            else:
                raise ValueError(
                    f"Unsupported 3D attn_mask shape {tuple(mask.shape)}; expected [B,T_q,T_k]."
                )
        elif mask.dim() != 4:
            raise ValueError(
                f"Unsupported attn_mask rank {mask.dim()}; expected 2D/3D/4D."
            )

        if mask.shape[-2:] != (T_q, T_k):
            raise ValueError(
                f"attn_mask last dims {tuple(mask.shape[-2:])} != {(T_q, T_k)}."
            )
        if mask.shape[0] not in (1, B):
            raise ValueError(
                f"attn_mask batch dim {mask.shape[0]} incompatible with B={B}."
            )
        if mask.shape[1] not in (1, H):
            raise ValueError(
                f"attn_mask head dim {mask.shape[1]} incompatible with H={H}."
            )

        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1, -1)
        if mask.shape[1] == 1 and H > 1:
            mask = mask.expand(-1, H, -1, -1)
        return mask

    def _slice_attn_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        B: int,
        H: int,
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        T_q_full: int,
        T_k_full: int,
    ) -> Optional[torch.Tensor]:
        """Slice normalized attention mask for chunked attention regions."""
        if attn_mask is None:
            return None
        full = self._normalize_attn_mask(attn_mask, B, H, T_q_full, T_k_full)
        return full[:, :, q_start:q_end, k_start:k_end]

    def _dropout_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Single place for attention-weight dropout policy."""
        if self.training and self.dropout_p > 0:
            return F.dropout(w, p=self.dropout_p, training=True)
        return w

    # --------------------------------------------------------------------- #
    # QKV processing (non-paged + paged gather compatibility)
    # --------------------------------------------------------------------- #
    def _make_kv_provider(
        self,
        layer_state: Optional[Dict],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        force_paged: Optional[bool] = None,
    ) -> KVProvider:
        use_paged_now = (
            self.use_paged_cache
            and (layer_state is not None)
            and (not self.cross_attention)
        )
        if force_paged is not None:
            use_paged_now = bool(force_paged)

        if use_paged_now:
            if layer_state is None:
                raise ValueError("layer_state is required for paged KV provider")
            cache = self._ensure_paged_cache(
                layer_state,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            return PagedKVProvider(
                cache,
                use_mla=self.use_mla,
                k_up_proj=self.k_up_proj,
                v_up_proj=self.v_up_proj,
            )

        return DenseKVProvider(
            layer_state=layer_state,
            cross_attention=self.cross_attention,
            use_mla=self.use_mla,
            k_up_proj=self.k_up_proj,
            v_up_proj=self.v_up_proj,
        )

    def _prepare_qkv_with_provider(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: Optional[Dict],
        *,
        force_paged: Optional[bool] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        KVProvider,
        Optional[torch.Tensor],
    ]:
        B, _, _ = query.shape
        q, k, v, kv_latent = self._project_qkv_heads(query, key, value)

        provider = self._make_kv_provider(
            layer_state,
            batch_size=B,
            device=q.device,
            dtype=q.dtype,
            force_paged=force_paged,
        )

        q_start_pos: Optional[torch.Tensor] = None
        if not self.cross_attention:
            q_start_pos = provider.get_start_positions(B, q.device)

        q, k = self.position_encoding_applier.apply(
            q,
            k,
            query=query,
            key=key,
            seqlen_offset=(q_start_pos if q_start_pos is not None else 0),
        )

        return q, k, v, kv_latent, provider, q_start_pos

    def _process_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: Optional[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        layer_state: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        layer_state: Optional[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, kv_latent, provider, _ = self._prepare_qkv_with_provider(
            query,
            key,
            value,
            layer_state,
        )

        if not self.cross_attention:
            k, v = provider.get_kv(k, v, kv_latent=kv_latent)

        return q, self._repeat_kv(k), self._repeat_kv(v)

    # --------------------------------------------------------------------- #
    # Plain full attention compute (prefill / cross)
    # --------------------------------------------------------------------- #
    def _compute_attention(
        self,
        q: torch.Tensor,  # [B, H, T_q, D]
        k: torch.Tensor,  # [B, H, T_k, D]
        v: torch.Tensor,  # [B, H, T_k, D]
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
        q_start_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T_q,T_k]

        # Causal mask (self-attention only)
        if is_causal and not self.cross_attention:
            if q_start_pos is None:
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(
                    causal_mask.view(1, 1, T_q, T_k),
                    float("-inf"),
                )
            else:
                if q_start_pos.ndim != 1 or q_start_pos.shape[0] != B:
                    raise ValueError(
                        f"q_start_pos must be [B], got {tuple(q_start_pos.shape)}"
                    )
                q_pos = q_start_pos.to(device=q.device, dtype=torch.long).view(
                    B, 1, 1, 1
                ) + torch.arange(T_q, device=q.device, dtype=torch.long).view(
                    1, 1, T_q, 1
                )
                k_pos = torch.arange(T_k, device=q.device, dtype=torch.long).view(
                    1, 1, 1, T_k
                )
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        scores = self._apply_masks(scores, attn_mask, key_padding_mask)

        weights = F.softmax(scores, dim=-1)
        weights = self._dropout_weights(weights)

        out = torch.matmul(weights, v)  # [B,H,T_q,D]
        out = self._apply_gated_attention(out)
        return out, (weights if need_weights else None)
