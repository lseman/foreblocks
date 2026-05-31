"""Attention modules: SelfAttention, AttentionBridge, LearnedPoolingBridge."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import RMSNorm

from .bb_positional import RotaryPositionalEncoding
from .bb_primitives import RMSNorm
from .utils import _causal_mask
from .utils import _make_alibi_slopes
from .utils import _seasonal_relative_bias
from .utils import _sinusoidal_features

__all__ = [
    "SelfAttention",
    "AttentionBridge",
    "LearnedPoolingBridge",
]


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
        attention_modes: Sequence[str] | None = None,
        temperature: float = 1.0,
        variant_gdas: bool = False,
    ):
        super().__init__()
        resolved_modes = tuple(
            str(mode).lower() for mode in (attention_modes or self.MODES)
        )
        if not resolved_modes:
            raise ValueError("attention_modes must contain at least one mode")
        self.MODES = resolved_modes
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
        self.temperature = max(float(temperature), 1e-3)
        self.variant_gdas = bool(variant_gdas)

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
        self.rotary_emb = RotaryPositionalEncoding(
            self.head_dim, max_seq_len=1024, base=500000.0
        )
        self.register_buffer(
            "alibi_slopes",
            _make_alibi_slopes(self.num_heads),
            persistent=False,
        )
        self.positional_scale = nn.Parameter(torch.tensor(1.0))
        # Learnable inverse-temperature for cosine kernel (see SelfAttention).
        self.cos_log_scale = nn.Parameter(torch.tensor(math.log(self.head_dim**0.5)))
        # Deterministic generator for ProbSparse sampling.
        self._probsparse_gen = torch.Generator().manual_seed(0xC0FFEE)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-3)

    def set_variant_gdas(self, enabled: bool) -> None:
        self.variant_gdas = bool(enabled)

    # ------------------------------------------------------------------
    # Internal helpers — all return [B, L_dec, d_model]
    # ------------------------------------------------------------------

    def _reshape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """[B, L, D] → [B, H, L, hd]"""
        return x.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """[B, H, L, hd] → [B, L, D]"""
        return x.transpose(1, 2).contiguous().reshape(B, L, self.d_model)

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

    def _position_mix_weights(self) -> torch.Tensor:
        """Soft (differentiable) weights over POSITION_MODES.

        When searchable + training, use Gumbel-Softmax (hard when
        ``variant_gdas`` is set, with straight-through gradient) so
        ``position_alphas`` receives gradient signal. When not searchable,
        returns a one-hot vector on ``self.position_mode``.
        """
        if not self.position_searchable or not hasattr(self, "position_alphas"):
            ref = next(self.parameters())
            w = ref.new_zeros(len(self.POSITION_MODES))
            resolved = (
                self.position_mode
                if self.position_mode in self.POSITION_MODES
                else "rope"
            )
            w[self.POSITION_MODES.index(resolved)] = 1.0
            return w
        tau = max(float(self.temperature), 1e-3)
        if self.training:
            return F.gumbel_softmax(
                self.position_alphas,
                tau=tau,
                hard=bool(self.variant_gdas),
                dim=0,
            )
        return F.softmax(self.position_alphas / tau, dim=0)

    def get_position_mode_probs(self) -> torch.Tensor:
        if self.position_searchable and hasattr(self, "position_alphas"):
            return F.softmax(self.position_alphas.detach(), dim=0)
        ref = next(self.parameters())
        probs = ref.new_zeros(len(self.POSITION_MODES))
        resolved = (
            self.position_mode if self.position_mode in self.POSITION_MODES else "rope"
        )
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
            slopes = self.alibi_slopes.to(device=device, dtype=dtype).reshape(
                1, self.num_heads, 1, 1
            )
            return -slopes * rel.reshape(1, 1, query_len, key_len)
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
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_mode: str,
        q_len: int,
        k_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if position_mode == "rope":
            return (*self._apply_rotary_pair(q, k, q_len, k_len), None)
        if position_mode == "none":
            return q, k, None
        if position_mode == "seasonal":
            pos_q = _sinusoidal_features(
                q_len, self.head_dim, device=q.device, dtype=q.dtype
            ).reshape(1, 1, q_len, self.head_dim)
            pos_k = _sinusoidal_features(
                k_len, self.head_dim, device=k.device, dtype=k.dtype
            ).reshape(1, 1, k_len, self.head_dim)
            scale = self.positional_scale.to(dtype=q.dtype)
            q = q + scale * pos_q
            k = k + scale * pos_k
            return (
                q,
                k,
                self._build_relative_bias(
                    position_mode, q_len, k_len, q.device, q.dtype
                ),
            )
        return (
            q,
            k,
            self._build_relative_bias(position_mode, q_len, k_len, q.device, q.dtype),
        )

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
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
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
        q = self._reshape(q, B, L_dec)  # [B, H, L_dec, hd]
        k = self._reshape(k, B, L_enc)  # [B, H, L_enc, hd]
        v = self._reshape(v, B, L_enc)  # [B, H, L_enc, hd]

        c = self.PROBSPARSE_C
        n_top = min(L_dec, max(1, int(c * math.log(L_dec + 1))))
        n_sample = min(L_enc, max(1, int(c * math.log(L_enc + 1))))

        # Deterministic sampling: reduces variance in alpha gradients during
        # DARTS search; Informer itself uses fresh randomness per step.
        cpu_idx = torch.randperm(L_enc, generator=self._probsparse_gen)[:n_sample]
        sample_idx = cpu_idx.to(q.device)
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
        # True cosine attention with learnable inverse-temperature. When a
        # position bias is present we fall back to the quadratic softmax form
        # (cannot add an arbitrary bias inside the linear-kernel denominator).
        q = self._reshape(q, B, L_dec)
        k = self._reshape(k, B, L_enc)
        v = self._reshape(v, B, L_enc)
        scale = torch.exp(self.cos_log_scale)
        qn = F.normalize(q, p=2, dim=-1)
        kn = F.normalize(k, p=2, dim=-1)

        if bias is not None:
            scores = torch.matmul(qn, kn.transpose(-2, -1)) * scale + bias
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=1.0 / float(max(L_enc, 1)))
            if self.training and self.dropout_p > 0:
                attn = F.dropout(attn, p=self.dropout_p)
            out = torch.matmul(attn, v)
            return self._merge(out, B, L_dec)

        # Linear-time path (CosFormer-style): ReLU on unit-norm features is a
        # non-negative kernel that preserves cosine similarity structure.
        q_f = F.relu(qn) + 1e-6
        k_f = F.relu(kn) + 1e-6
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

    def _attn_mix_weights(self, tau: float | None = None) -> torch.Tensor:
        """Soft weights over MODES; honours ``variant_gdas``.

        ``tau`` defaults to ``self.temperature``; pass an explicit value to
        override.
        """
        eff_tau = max(
            float(tau if tau is not None else self.temperature),
            1e-3,
        )
        if self.training:
            return F.gumbel_softmax(
                self.attn_alphas,
                tau=eff_tau,
                hard=bool(self.variant_gdas),
                dim=0,
            )
        return F.softmax(self.attn_alphas / eff_tau, dim=0)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        encoder_context: torch.Tensor | None = None,
        temperature: float | None = None,
        variant_gdas: bool | None = None,
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

        # Position-mode selection: argmax(weights) with straight-through
        # gradient via weights[idx]. Fixes the bug where position_alphas
        # received no gradient signal.
        pos_weights = self._position_mix_weights()
        pos_idx = int(torch.argmax(pos_weights.detach()).item())
        position_mode = self.POSITION_MODES[pos_idx]
        pos_scalar = pos_weights[pos_idx] if self.position_searchable else None

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
            if pos_scalar is not None:
                raw = pos_scalar * raw
            return self.norm(decoder_hidden + self.out_proj(raw))

        if not self.searchable:
            return _apply(self.attention_type)

        # Resolve effective temperature / single-path: explicit call arg wins
        # over module state (set via set_temperature / set_variant_gdas).
        eff_tau = (
            max(float(temperature), 1e-3)
            if temperature is not None
            else max(float(self.temperature), 1e-3)
        )
        eff_single = (
            bool(variant_gdas) if variant_gdas is not None else bool(self.variant_gdas)
        )

        if self.training and eff_single:
            # Single path (GDAS-style): only the argmax kernel runs, cutting
            # peak memory and FLOPs ~6x across all MODES. Gradient reaches the
            # alphas via the weights[idx] multiplier (straight-through).
            weights = F.gumbel_softmax(self.attn_alphas, tau=eff_tau, hard=True, dim=0)
            idx = int(weights.argmax().item())
            return weights[idx] * _apply(self.MODES[idx])

        weights = self._attn_mix_weights(eff_tau)
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
