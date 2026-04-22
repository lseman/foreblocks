"""Attention modules: SelfAttention, AttentionBridge, LearnedPoolingBridge."""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_positional import RotaryPositionalEncoding
from .bb_primitives import RMSNorm

__all__ = [
    "SelfAttention",
    "AttentionBridge",
    "LearnedPoolingBridge",
]


def _make_alibi_slopes(num_heads: int) -> torch.Tensor:
    num_heads = max(1, int(num_heads))
    base = torch.arange(1, num_heads + 1, dtype=torch.float32)
    return 1.0 / torch.pow(2.0, base / num_heads)


def _sinusoidal_features(
    seq_len: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> torch.Tensor:
    seq_len = int(max(1, seq_len))
    dim = int(max(1, dim))
    half = max(1, dim // 2)
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, half, device=device, dtype=dtype) * -(torch.log(torch.tensor(base, device=device, dtype=dtype)) / max(half, 1))
    )
    angles = position * div_term.unsqueeze(0)
    feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if feats.size(-1) < dim:
        feats = F.pad(feats, (0, dim - feats.size(-1)))
    return feats[:, :dim]


def _seasonal_relative_bias(
    query_len: int,
    key_len: int,
    num_heads: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    q_pos = torch.arange(query_len, device=device, dtype=dtype).unsqueeze(1)
    k_pos = torch.arange(key_len, device=device, dtype=dtype).unsqueeze(0)
    rel = q_pos - k_pos
    periods = torch.tensor([4.0, 8.0, 16.0, 24.0, 48.0], device=device, dtype=dtype)
    bias = torch.stack([torch.cos(2.0 * torch.pi * rel / p) for p in periods], dim=0).mean(dim=0)
    slopes = _make_alibi_slopes(num_heads).to(device=device, dtype=dtype).view(1, num_heads, 1, 1)
    return 0.1 * slopes * bias.view(1, 1, query_len, key_len)


class SelfAttention(nn.Module):
    """Self-attention block with selectable attention kernels.

    Supported modes: ``sdp``, ``linear``, ``probsparse``, ``cosine``,
    ``local``, ``auto``.
    """

    MODES: tuple[str, ...] = ("sdp", "linear", "probsparse", "cosine", "local")
    POSITION_MODES: tuple[str, ...] = ("rope", "alibi", "none", "seasonal")
    LOCAL_WINDOW_RATIO: float = 0.25
    PROBSPARSE_C: int = 5

    def __init__(
        self,
        dim,
        heads=4,
        dropout=0.0,
        causal=False,
        attention_type: str = "sdp",
        position_mode: str = "rope",
        rope_base: float = 500000.0,
        rope_max_seq_len: int = 1024,
    ):
        super().__init__()
        resolved_attention_type = str(attention_type).lower()
        resolved_position_mode = str(position_mode).lower()
        assert resolved_attention_type in (*self.MODES, "auto"), (
            "attention_type must be one of "
            f"{(*self.MODES, 'auto')}, got {resolved_attention_type!r}"
        )
        assert resolved_position_mode in (*self.POSITION_MODES, "auto"), (
            "position_mode must be one of "
            f"{(*self.POSITION_MODES, 'auto')}, got {resolved_position_mode!r}"
        )
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.causal = causal
        self.attention_type = resolved_attention_type
        self.searchable = resolved_attention_type == "auto"
        self.position_mode = resolved_position_mode
        self.position_searchable = resolved_position_mode == "auto"

        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout

        if self.searchable:
            self.register_parameter(
                "attn_alphas", nn.Parameter(0.01 * torch.randn(len(self.MODES)))
            )
        if self.position_searchable:
            self.register_parameter(
                "position_alphas",
                nn.Parameter(0.01 * torch.randn(len(self.POSITION_MODES))),
            )

        self.rotary_emb = RotaryPositionalEncoding(
            self.head_dim,
            max_seq_len=rope_max_seq_len,
            base=rope_base,
        )
        self.register_buffer(
            "alibi_slopes",
            _make_alibi_slopes(self.heads),
            persistent=False,
        )
        self.positional_scale = nn.Parameter(torch.tensor(1.0))

    def _apply_rotary_pair(
        self, q: torch.Tensor, k: torch.Tensor, q_len: int, k_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_q, sin_q = self.rotary_emb.get_embeddings_for_length(q_len, q.device)
        cos_k, sin_k = self.rotary_emb.get_embeddings_for_length(k_len, k.device)
        cos = cos_q.to(dtype=q.dtype)
        sin = sin_q.to(dtype=q.dtype)
        q = self.rotary_emb.apply_rotary_pos_emb(q, cos, sin)
        cos = cos_k.to(dtype=k.dtype)
        sin = sin_k.to(dtype=k.dtype)
        k = self.rotary_emb.apply_rotary_pos_emb(k, cos, sin)
        return q, k

    def _get_position_mode(self) -> str:
        if not self.position_searchable or not hasattr(self, "position_alphas"):
            return self.position_mode
        logits = self.position_alphas
        if self.training:
            weights = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=0)
            idx = int(torch.argmax(weights).item())
            return self.POSITION_MODES[idx]
        probs = F.softmax(logits.detach(), dim=0)
        idx = int(torch.argmax(probs).item())
        return self.POSITION_MODES[idx]

    def get_position_mode_probs(self) -> torch.Tensor:
        if self.position_searchable and hasattr(self, "position_alphas"):
            return F.softmax(self.position_alphas.detach(), dim=0)
        ref = next(self.parameters())
        probs = ref.new_zeros(len(self.POSITION_MODES))
        resolved = self.position_mode if self.position_mode in self.POSITION_MODES else "rope"
        probs[self.POSITION_MODES.index(resolved)] = 1.0
        return probs

    def resolve_position_mode(self) -> str:
        probs = self.get_position_mode_probs()
        idx = int(torch.argmax(probs).item())
        return self.POSITION_MODES[idx]

    def freeze_position_mode(self, position_mode: str) -> None:
        resolved = str(position_mode).lower()
        self.position_mode = resolved if resolved in self.POSITION_MODES else "rope"
        self.position_searchable = False
        if hasattr(self, "position_alphas"):
            self._parameters.pop("position_alphas", None)
            try:
                delattr(self, "position_alphas")
            except AttributeError:
                pass

    def _build_relative_bias(
        self, position_mode: str, query_len: int, key_len: int, device, dtype
    ) -> torch.Tensor | None:
        if position_mode == "alibi":
            q_pos = torch.arange(query_len, device=device, dtype=dtype).unsqueeze(1)
            k_pos = torch.arange(key_len, device=device, dtype=dtype).unsqueeze(0)
            rel = (q_pos - k_pos).abs()
            slopes = self.alibi_slopes.to(device=device, dtype=dtype).view(1, self.heads, 1, 1)
            return -slopes * rel.view(1, 1, query_len, key_len)
        if position_mode == "seasonal":
            return _seasonal_relative_bias(
                query_len,
                key_len,
                self.heads,
                device=device,
                dtype=dtype,
            )
        return None

    def _apply_position_mode(
        self, q: torch.Tensor, k: torch.Tensor, position_mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        query_len = q.size(-2)
        key_len = k.size(-2)
        if position_mode == "rope":
            q, k = self._apply_rotary_pair(q, k, query_len, key_len)
            return q, k, None
        if position_mode == "none":
            return q, k, None
        if position_mode == "seasonal":
            pos_q = _sinusoidal_features(
                query_len, self.head_dim, device=q.device, dtype=q.dtype
            ).view(1, 1, query_len, self.head_dim)
            pos_k = _sinusoidal_features(
                key_len, self.head_dim, device=k.device, dtype=k.dtype
            ).view(1, 1, key_len, self.head_dim)
            scale = self.positional_scale.to(dtype=q.dtype)
            q = q + scale * pos_q
            k = k + scale * pos_k
            bias = self._build_relative_bias(position_mode, query_len, key_len, q.device, q.dtype)
            return q, k, bias
        bias = self._build_relative_bias(position_mode, query_len, key_len, q.device, q.dtype)
        return q, k, bias

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads

        qkv = self.to_qkv(x).view(B, T, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q_raw, k_raw, v = qkv.unbind(0)
        position_mode = self._get_position_mode()
        q, k, position_bias = self._apply_position_mode(q_raw, k_raw, position_mode)

        def _apply(mode: str) -> torch.Tensor:
            dropout_p = self.dropout_p if self.training else 0.0

            if mode == "sdp":
                if position_bias is None:
                    return F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=dropout_p,
                        is_causal=self.causal,
                        scale=self.scale,
                    )
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + position_bias
                if self.causal:
                    mask = torch.triu(
                        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                    )
                    scores = scores.masked_fill(mask.view(1, 1, T, T), float("-inf"))
                attn = F.softmax(scores, dim=-1)
                attn = torch.nan_to_num(attn, nan=1.0 / float(max(T, 1)))
                if dropout_p > 0:
                    attn = F.dropout(attn, p=dropout_p)
                return torch.matmul(attn, v)

            if mode == "linear":
                if self.causal:
                    # Causal linear (ELU+1 Performer) requires a recurrent
                    # cumsum formulation for correctness; we fall back to
                    # FlashAttention-backed SDP which is still efficient and
                    # numerically exact.
                    return F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=dropout_p,
                        is_causal=True,
                        scale=self.scale,
                    )
                q_feat = F.elu(q * self.scale) + 1.0
                k_feat = F.elu(k) + 1.0
                kv = torch.einsum("bhtd,bhtv->bhdv", k_feat, v)
                k_sum = k_feat.sum(dim=2)
                denom = torch.einsum("bhtd,bhd->bht", q_feat, k_sum).clamp_min(1e-6)
                out_linear = torch.einsum("bhtd,bhdv->bhtv", q_feat, kv)
                return out_linear / denom.unsqueeze(-1)

            if mode == "cosine":
                qn = F.normalize(q, p=2, dim=-1)
                kn = F.normalize(k, p=2, dim=-1)
                scores = torch.matmul(qn, kn.transpose(-2, -1))
                if position_bias is not None:
                    scores = scores + position_bias
                if self.causal:
                    mask = torch.triu(
                        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                    )
                    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                attn = F.softmax(scores, dim=-1)
                if dropout_p > 0:
                    attn = F.dropout(attn, p=dropout_p)
                return torch.matmul(attn, v)

            if mode == "local":
                W = max(4, int(T * self.LOCAL_WINDOW_RATIO))
                pos = torch.arange(T, device=x.device)
                lo = (pos - (W // 2)).clamp(0, T - 1)
                hi = (pos + (W // 2)).clamp(0, T - 1)
                local_mask = (pos.unsqueeze(0) >= lo.unsqueeze(1)) & (
                    pos.unsqueeze(0) <= hi.unsqueeze(1)
                )
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                if position_bias is not None:
                    scores = scores + position_bias
                mask = ~local_mask
                if self.causal:
                    causal_mask = torch.triu(
                        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                    )
                    mask = mask | causal_mask
                scores = scores.masked_fill(
                    mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )
                attn = F.softmax(scores, dim=-1)
                attn = torch.nan_to_num(attn, nan=1.0 / float(T))
                if dropout_p > 0:
                    attn = F.dropout(attn, p=dropout_p)
                return torch.matmul(attn, v)

            import math

            c = self.PROBSPARSE_C
            n_top = min(T, max(1, int(c * math.log(T + 1))))
            n_sample = min(T, max(1, int(c * math.log(T + 1))))
            sample_idx = torch.randperm(T, device=x.device)[:n_sample]
            k_sample = k[:, :, sample_idx, :]
            q_scores = torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale
            M = q_scores.amax(dim=-1) - q_scores.mean(dim=-1)
            top_idx = M.topk(n_top, dim=-1).indices
            v_mean = v.mean(dim=2, keepdim=True).expand(B, H, T, self.head_dim)
            out_sparse = v_mean.clone()
            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            q_top = q.gather(2, idx_exp)
            scores_top = torch.matmul(q_top, k.transpose(-2, -1)) * self.scale
            if position_bias is not None:
                bias_expanded = position_bias.expand(B, -1, -1, -1)
                pos_bias_top = bias_expanded.gather(
                    2, top_idx.unsqueeze(-1).expand(-1, -1, -1, T)
                )
                scores_top = scores_top + pos_bias_top
            if self.causal:
                full_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, T)
                key_pos = torch.arange(T, device=x.device).view(1, 1, 1, T)
                causal_mask = key_pos > full_idx
                scores_top = scores_top.masked_fill(causal_mask, float("-inf"))
            attn_top = F.softmax(scores_top, dim=-1)
            attn_top = torch.nan_to_num(attn_top, nan=1.0 / float(T))
            if dropout_p > 0:
                attn_top = F.dropout(attn_top, p=dropout_p)
            out_top = torch.matmul(attn_top, v)
            out_sparse.scatter_(2, idx_exp, out_top)
            return out_sparse

        if self.searchable:
            tau = 1.0
            if self.training:
                weights = F.gumbel_softmax(self.attn_alphas, tau=tau, hard=False, dim=0)
            else:
                weights = F.softmax(self.attn_alphas, dim=0)
            out = sum(weights[i] * _apply(self.MODES[i]) for i in range(len(self.MODES)))
        else:
            out = _apply(self.attention_type)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class AttentionBridge(nn.Module):
    """Unified cross-attention bridge with searchable attention type.

    Attention modes
    ---------------
    "none"       – identity passthrough (no cross-attention)
    "sdp"        – scaled dot-product attention              O(T·S)
    "linear"     – ELU+1 Performer kernelised attention      O(T+S)
    "probsparse" – Informer ProbSparse: selects top-u queries
                   by sparsity score, fills rest with mean(V) O(L log L)
    "cosine"     – CosFormer: L2-norm Q/K + ReLU feature map O(T+S)
    "local"      – sliding-window cross-attention aligned
                   by seq-length ratio                       O(T·W), W≪S

    When ``attention_type="auto"`` (default), the module owns its own ``attn_alphas``
    over all modes and performs DARTS-style mixing during search.
    Pass a fixed string to pin a mode without searchable parameters.
    """

    MODES: tuple[str, ...] = ("none", "sdp", "linear", "probsparse", "cosine", "local")
    POSITION_MODES: tuple[str, ...] = ("rope", "alibi", "none", "seasonal")

    # Fraction of encoder length used as local window (clamped ≥ 4)
    LOCAL_WINDOW_RATIO: float = 0.25
    # Informer constant: n_top = min(L_Q, c * ln(L_Q + 1))
    PROBSPARSE_C: int = 5

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_type: str = "auto",
        position_mode: str = "rope",
    ):
        super().__init__()
        resolved_attention_type = str(attention_type).lower()
        resolved_position_mode = str(position_mode).lower()
        assert resolved_attention_type in (*self.MODES, "auto"), (
            "attention_type must be one of "
            f"{(*self.MODES, 'auto')}, got {resolved_attention_type!r}"
        )
        assert resolved_position_mode in (*self.POSITION_MODES, "auto"), (
            "position_mode must be one of "
            f"{(*self.POSITION_MODES, 'auto')}, got {resolved_position_mode!r}"
        )
        self.d_model = int(d_model)
        self.attention_type = resolved_attention_type
        self.searchable = resolved_attention_type == "auto"
        self.position_mode = resolved_position_mode
        self.position_searchable = resolved_position_mode == "auto"

        self.num_heads = min(max(1, int(num_heads)), self.d_model)
        while self.d_model % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout_p = float(max(dropout, 0.0))
        self.norm = RMSNorm(self.d_model)

        if self.searchable:
            self.register_parameter(
                "attn_alphas",
                nn.Parameter(0.01 * torch.randn(len(self.MODES))),
            )
        if self.position_searchable:
            self.register_parameter(
                "position_alphas",
                nn.Parameter(0.01 * torch.randn(len(self.POSITION_MODES))),
            )
        self.rotary_emb = RotaryPositionalEncoding(self.head_dim, max_seq_len=1024, base=500000.0)
        self.register_buffer(
            "alibi_slopes",
            _make_alibi_slopes(self.num_heads),
            persistent=False,
        )
        self.positional_scale = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------
    # Internal helpers — all return [B, L_dec, d_model]
    # ------------------------------------------------------------------

    def _reshape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """[B, L, D] → [B, H, L, hd]"""
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """[B, H, L, hd] → [B, L, D]"""
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def _apply_rotary_pair(
        self, q: torch.Tensor, k: torch.Tensor, q_len: int, k_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_q, sin_q = self.rotary_emb.get_embeddings_for_length(q_len, q.device)
        cos_k, sin_k = self.rotary_emb.get_embeddings_for_length(k_len, k.device)
        q = self.rotary_emb.apply_rotary_pos_emb(
            q, cos_q.to(dtype=q.dtype), sin_q.to(dtype=q.dtype)
        )
        k = self.rotary_emb.apply_rotary_pos_emb(
            k, cos_k.to(dtype=k.dtype), sin_k.to(dtype=k.dtype)
        )
        return q, k

    def _get_position_mode(self) -> str:
        if not self.position_searchable or not hasattr(self, "position_alphas"):
            return self.position_mode
        logits = self.position_alphas
        if self.training:
            weights = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=0)
            idx = int(torch.argmax(weights).item())
            return self.POSITION_MODES[idx]
        probs = F.softmax(logits.detach(), dim=0)
        idx = int(torch.argmax(probs).item())
        return self.POSITION_MODES[idx]

    def get_position_mode_probs(self) -> torch.Tensor:
        if self.position_searchable and hasattr(self, "position_alphas"):
            return F.softmax(self.position_alphas.detach(), dim=0)
        ref = next(self.parameters())
        probs = ref.new_zeros(len(self.POSITION_MODES))
        resolved = self.position_mode if self.position_mode in self.POSITION_MODES else "rope"
        probs[self.POSITION_MODES.index(resolved)] = 1.0
        return probs

    def resolve_position_mode(self) -> str:
        probs = self.get_position_mode_probs()
        idx = int(torch.argmax(probs).item())
        return self.POSITION_MODES[idx]

    def freeze_position_mode(self, position_mode: str) -> None:
        resolved = str(position_mode).lower()
        self.position_mode = resolved if resolved in self.POSITION_MODES else "rope"
        self.position_searchable = False
        if hasattr(self, "position_alphas"):
            self._parameters.pop("position_alphas", None)
            try:
                delattr(self, "position_alphas")
            except AttributeError:
                pass

    def _build_relative_bias(
        self, position_mode: str, query_len: int, key_len: int, device, dtype
    ) -> torch.Tensor | None:
        if position_mode == "alibi":
            q_pos = torch.arange(query_len, device=device, dtype=dtype).unsqueeze(1)
            k_pos = torch.arange(key_len, device=device, dtype=dtype).unsqueeze(0)
            rel = (q_pos - k_pos).abs()
            slopes = self.alibi_slopes.to(device=device, dtype=dtype).view(1, self.num_heads, 1, 1)
            return -slopes * rel.view(1, 1, query_len, key_len)
        if position_mode == "seasonal":
            return _seasonal_relative_bias(
                query_len,
                key_len,
                self.num_heads,
                device=device,
                dtype=dtype,
            )
        return None

    def _apply_position_mode(
        self, q: torch.Tensor, k: torch.Tensor, position_mode: str, q_len: int, k_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if position_mode == "rope":
            return (*self._apply_rotary_pair(q, k, q_len, k_len), None)
        if position_mode == "none":
            return q, k, None
        if position_mode == "seasonal":
            pos_q = _sinusoidal_features(
                q_len, self.head_dim, device=q.device, dtype=q.dtype
            ).view(1, 1, q_len, self.head_dim)
            pos_k = _sinusoidal_features(
                k_len, self.head_dim, device=k.device, dtype=k.dtype
            ).view(1, 1, k_len, self.head_dim)
            scale = self.positional_scale.to(dtype=q.dtype)
            q = q + scale * pos_q
            k = k + scale * pos_k
            return q, k, self._build_relative_bias(position_mode, q_len, k_len, q.device, q.dtype)
        return q, k, self._build_relative_bias(position_mode, q_len, k_len, q.device, q.dtype)

    # ---- vanilla SDP --------------------------------------------------

    def _sdp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        L_dec: int,
        L_enc: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._reshape(q, B, L_dec)
        k = self._reshape(k, B, L_enc)
        v = self._reshape(v, B, L_enc)
        dropout_p = self.dropout_p if self.training else 0.0
        if bias is not None:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + bias
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=1.0 / float(max(L_enc, 1)))
            if dropout_p > 0:
                attn = F.dropout(attn, p=dropout_p)
            return self._merge(torch.matmul(attn, v), B, L_dec)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.scale,
        )
        return self._merge(out, B, L_dec)

    # ---- Performer / ELU+1 linear ------------------------------------

    @staticmethod
    def _elu_feature(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def _linear(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        L_dec: int,
        L_enc: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._reshape(q, B, L_dec)
        k = self._reshape(k, B, L_enc)
        v = self._reshape(v, B, L_enc)
        q_f = self._elu_feature(q * self.scale)
        k_f = self._elu_feature(k)
        kv = torch.einsum("bhtd,bhtv->bhdv", k_f, v)
        k_sum = k_f.sum(dim=2)
        denom = torch.einsum("bhtd,bhd->bht", q_f, k_sum).clamp_min(1e-6)
        out = torch.einsum("bhtd,bhdv->bhtv", q_f, kv) / denom.unsqueeze(-1)
        if self.training and self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p)
        return self._merge(out, B, L_dec)

    # ---- Informer ProbSparse -----------------------------------------

    def _probsparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        L_dec: int,
        L_enc: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import math

        q = self._reshape(q, B, L_dec)  # [B, H, L_dec, hd]
        k = self._reshape(k, B, L_enc)  # [B, H, L_enc, hd]
        v = self._reshape(v, B, L_enc)  # [B, H, L_enc, hd]

        c = self.PROBSPARSE_C
        n_top = min(L_dec, max(1, int(c * math.log(L_dec + 1))))
        n_sample = min(L_enc, max(1, int(c * math.log(L_enc + 1))))

        # Sample a random subset of keys to estimate query sparsity
        sample_idx = torch.randperm(L_enc, device=q.device)[:n_sample]
        k_sample = k[:, :, sample_idx, :]  # [B, H, n_sample, hd]

        # Sparsity measure M  (max - mean over sampled keys)
        q_scores = (
            torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale
        )  # [B,H,L_dec,n_s]
        M = q_scores.amax(dim=-1) - q_scores.mean(dim=-1)  # [B, H, L_dec]

        # Select top-u query positions
        top_idx = M.topk(n_top, dim=-1).indices  # [B, H, n_top]

        # Lazy fill: initialise output as mean(V) broadcast to query shape
        v_mean = v.mean(dim=2, keepdim=True).expand(
            B, self.num_heads, L_dec, self.head_dim
        )
        out = v_mean.clone()

        # Full attention for top-u queries only
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        q_top = q.gather(2, idx_exp)  # [B, H, n_top, hd]
        scores_top = torch.matmul(q_top, k.transpose(-2, -1)) * self.scale
        if bias is not None:
            bias_expanded = bias.expand(B, -1, -1, -1)
            pos_bias_top = bias_expanded.gather(
                2, top_idx.unsqueeze(-1).expand(-1, -1, -1, L_enc)
            )
            scores_top = scores_top + pos_bias_top
        attn_top = F.softmax(scores_top, dim=-1)
        if self.training and self.dropout_p > 0:
            attn_top = F.dropout(attn_top, p=self.dropout_p)
        out_top = torch.matmul(attn_top, v)  # [B, H, n_top, hd]
        out.scatter_(2, idx_exp, out_top)

        return self._merge(out, B, L_dec)

    # ---- CosFormer (cosine / ReLU feature) ---------------------------

    def _cosine(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        L_dec: int,
        L_enc: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._reshape(q, B, L_dec)
        k = self._reshape(k, B, L_enc)
        v = self._reshape(v, B, L_enc)

        # L2-normalise, then use ReLU+ε as a non-negative feature map
        # → preserves cosine similarity structure, O(T+S) cost
        if bias is not None:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + bias
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=1.0 / float(max(L_enc, 1)))
            if self.training and self.dropout_p > 0:
                attn = F.dropout(attn, p=self.dropout_p)
            out = torch.matmul(attn, v)
            return self._merge(out, B, L_dec)

        q_f = F.relu(F.normalize(q, p=2, dim=-1)) + 1e-6
        k_f = F.relu(F.normalize(k, p=2, dim=-1)) + 1e-6

        kv = torch.einsum("bhtd,bhtv->bhdv", k_f, v)
        k_sum = k_f.sum(dim=2)
        denom = torch.einsum("bhtd,bhd->bht", q_f, k_sum).clamp_min(1e-6)
        out = torch.einsum("bhtd,bhdv->bhtv", q_f, kv) / denom.unsqueeze(-1)
        if self.training and self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p)
        return self._merge(out, B, L_dec)

    # ---- Sliding-window local cross-attention ------------------------

    def _local(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        L_dec: int,
        L_enc: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._reshape(q, B, L_dec)
        k = self._reshape(k, B, L_enc)
        v = self._reshape(v, B, L_enc)

        W = max(4, int(L_enc * self.LOCAL_WINDOW_RATIO))

        # For each decoder position i, find the aligned encoder centre
        dec_pos = torch.arange(L_dec, device=q.device, dtype=torch.float32)
        enc_center = (
            (dec_pos * (L_enc / max(L_dec, 1))).long().clamp(0, L_enc - 1)
        )  # [L_dec]

        enc_pos = torch.arange(L_enc, device=q.device)  # [L_enc]
        half = W // 2
        lo = (enc_center.unsqueeze(1) - half).clamp(0, L_enc - 1)  # [L_dec, 1]
        hi = (enc_center.unsqueeze(1) + half).clamp(0, L_enc - 1)  # [L_dec, 1]
        window_mask = (enc_pos.unsqueeze(0) >= lo) & (
            enc_pos.unsqueeze(0) <= hi
        )  # [L_dec, L_enc]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L_dec,L_enc]
        if bias is not None:
            scores = scores + bias
        scores = scores.masked_fill(
            ~window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn = F.softmax(scores, dim=-1)
        # Rows that are fully masked → NaN; replace with uniform attention
        attn = torch.nan_to_num(attn, nan=1.0 / L_enc)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.matmul(attn, v)
        return self._merge(out, B, L_dec)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        encoder_context: torch.Tensor | None = None,
        temperature: float = 1.0,
        single_path: bool = False,
    ) -> torch.Tensor:
        source = encoder_output if encoder_output is not None else encoder_context
        if source is None:
            return decoder_hidden
        if source.dim() == 2:
            source = source.unsqueeze(1)

        B, L_dec, _ = decoder_hidden.shape
        L_enc = source.size(1)

        q = self.q_proj(decoder_hidden)
        k = self.k_proj(source)
        v = self.v_proj(source)
        position_mode = self._get_position_mode()

        _kernels = {
            "sdp": self._sdp,
            "linear": self._linear,
            "probsparse": self._probsparse,
            "cosine": self._cosine,
            "local": self._local,
        }

        def _apply(mode: str) -> torch.Tensor:
            if mode == "none":
                return decoder_hidden
            q_heads = self._reshape(q, B, L_dec)
            k_heads = self._reshape(k, B, L_enc)
            q_pos, k_pos, position_bias = self._apply_position_mode(
                q_heads, k_heads, position_mode, L_dec, L_enc
            )
            raw = _kernels[mode](
                self._merge(q_pos, B, L_dec),
                self._merge(k_pos, B, L_enc),
                v,
                B,
                L_dec,
                L_enc,
                position_bias,
            )
            return self.norm(decoder_hidden + self.out_proj(raw))

        if not self.searchable:
            return _apply(self.attention_type)

        tau = max(float(temperature), 1e-3)
        if self.training and single_path:
            weights = F.gumbel_softmax(self.attn_alphas, tau=tau, hard=True, dim=0)
            idx = int(weights.argmax().item())
            return weights[idx] * _apply(self.MODES[idx])
        elif self.training:
            weights = F.gumbel_softmax(self.attn_alphas, tau=tau, hard=False, dim=0)
        else:
            weights = F.softmax(self.attn_alphas / tau, dim=0)

        return sum(weights[i] * _apply(self.MODES[i]) for i in range(len(self.MODES)))


class LearnedPoolingBridge(nn.Module):
    """Compress encoder sequences into a fixed-size decoder memory."""

    def __init__(
        self, dim: int, num_queries: int = 8, num_heads: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = max(1, int(num_queries))
        self.num_heads = min(max(1, int(num_heads)), dim)
        while dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.queries = nn.Parameter(torch.randn(1, self.num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.norm = RMSNorm(dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        if encoder_output is None:
            raise ValueError(
                "encoder_output must not be None for LearnedPoolingBridge."
            )
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)
        batch_size = encoder_output.size(0)
        queries = self.queries.expand(batch_size, -1, -1)
        memory, _ = self.attn(queries, encoder_output, encoder_output)
        return self.norm(memory)
