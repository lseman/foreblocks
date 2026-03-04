"""Attention modules: LinearSelfAttention, AttentionBridge, LearnedPoolingBridge."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_positional import RotaryPositionalEncoding
from .bb_primitives import RMSNorm

__all__ = ["LinearSelfAttention", "AttentionBridge", "LearnedPoolingBridge"]


class LinearSelfAttention(nn.Module):
    """Improved linear self-attention with consistent behavior"""

    def __init__(
        self,
        dim,
        heads=4,
        dropout=0.0,
        causal=False,
        rope_base: float = 500000.0,
        rope_max_seq_len: int = 1024,
    ):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.causal = causal

        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout

        self.rotary_emb = RotaryPositionalEncoding(
            self.head_dim,
            max_seq_len=rope_max_seq_len,
            base=rope_base,
        )

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.size(-2)
        cos, sin = self.rotary_emb.get_embeddings_for_length(seq_len, q.device)
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q = self.rotary_emb.apply_rotary_pos_emb(q, cos, sin)
        k = self.rotary_emb.apply_rotary_pos_emb(k, cos, sin)
        return q, k

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads

        qkv = self.to_qkv(x).view(B, T, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self._apply_rotary(q, k)

        if self.causal:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores.masked_fill_(mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout_p)
            out = torch.matmul(attn_weights, v)
        else:
            q_feat = self._feature_map(q * self.scale)
            k_feat = self._feature_map(k)
            kv = torch.einsum("bhtd,bhtv->bhdv", k_feat, v)
            k_sum = k_feat.sum(dim=2)
            denom = torch.einsum("bhtd,bhd->bht", q_feat, k_sum).clamp_min(1e-6)
            out = torch.einsum("bhtd,bhdv->bhtv", q_feat, kv) / denom.unsqueeze(-1)

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        if self.training and self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p)

        return out


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

    When ``attn_type="auto"`` (default), the module owns its own ``attn_alphas``
    over all modes and performs DARTS-style mixing during search.
    Pass a fixed string to pin a mode without searchable parameters.
    """

    MODES: Tuple[str, ...] = ("none", "sdp", "linear", "probsparse", "cosine", "local")

    # Fraction of encoder length used as local window (clamped ≥ 4)
    LOCAL_WINDOW_RATIO: float = 0.25
    # Informer constant: n_top = min(L_Q, c * ln(L_Q + 1))
    PROBSPARSE_C: int = 5

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_type: str = "auto",
    ):
        super().__init__()
        assert attn_type in (*self.MODES, "auto"), (
            f"attn_type must be one of {(*self.MODES, 'auto')}, got {attn_type!r}"
        )
        self.d_model = int(d_model)
        self.attn_type = attn_type
        self.searchable = attn_type == "auto"

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

    # ------------------------------------------------------------------
    # Internal helpers — all return [B, L_dec, d_model]
    # ------------------------------------------------------------------

    def _reshape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """[B, L, D] → [B, H, L, hd]"""
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """[B, H, L, hd] → [B, L, D]"""
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    # ---- vanilla SDP --------------------------------------------------

    def _sdp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        L_dec: int,
        L_enc: int,
    ) -> torch.Tensor:
        q = self._reshape(q, B, L_dec)
        k = self._reshape(k, B, L_enc)
        v = self._reshape(v, B, L_enc)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.matmul(attn, v)
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
    ) -> torch.Tensor:
        q = self._reshape(q, B, L_dec)
        k = self._reshape(k, B, L_enc)
        v = self._reshape(v, B, L_enc)

        # L2-normalise, then use ReLU+ε as a non-negative feature map
        # → preserves cosine similarity structure, O(T+S) cost
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
        encoder_output: Optional[torch.Tensor] = None,
        encoder_context: Optional[torch.Tensor] = None,
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
            raw = _kernels[mode](q, k, v, B, L_dec, L_enc)
            return self.norm(decoder_hidden + self.out_proj(raw))

        if not self.searchable:
            return _apply(self.attn_type)

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
