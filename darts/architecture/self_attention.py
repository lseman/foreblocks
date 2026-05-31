"""Attention modules: SelfAttention, AttentionBridge, LearnedPoolingBridge."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import RMSNorm
from .utils import _causal_mask
from .utils import _make_alibi_slopes
from .utils import _seasonal_relative_bias
from .utils import _sinusoidal_features

from .bb_positional import RotaryPositionalEncoding
from .bb_primitives import RMSNorm

__all__ = [
    "SelfAttention",
    "AttentionBridge",
    "LearnedPoolingBridge",
]


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
        temperature: float = 1.0,
        variant_gdas: bool = False,
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
        self.temperature = max(float(temperature), 1e-3)
        self.variant_gdas = bool(variant_gdas)

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
        # Learnable cosine inverse-temperature: for unit-norm Q/K, scores live
        # in [-1,1]; without rescaling the softmax is near-uniform. Init at
        # log(head_dim ** 0.5) so exp(.) matches the SDP scale at start.
        self.cos_log_scale = nn.Parameter(torch.tensor(math.log(self.head_dim**0.5)))
        # Deterministic generator for ProbSparse sampling: reduces variance in
        # the architecture gradient since alphas are trying to estimate which
        # kernel is best.
        self._probsparse_gen = torch.Generator().manual_seed(0xC0FFEE)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-3)

    def set_variant_gdas(self, enabled: bool) -> None:
        self.variant_gdas = bool(enabled)

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
                1, self.heads, 1, 1
            )
            return -slopes * rel.reshape(1, 1, query_len, key_len)
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
            ).reshape(1, 1, query_len, self.head_dim)
            pos_k = _sinusoidal_features(
                key_len, self.head_dim, device=k.device, dtype=k.dtype
            ).reshape(1, 1, key_len, self.head_dim)
            scale = self.positional_scale.to(dtype=q.dtype)
            q = q + scale * pos_q
            k = k + scale * pos_k
            bias = self._build_relative_bias(
                position_mode, query_len, key_len, q.device, q.dtype
            )
            return q, k, bias
        bias = self._build_relative_bias(
            position_mode, query_len, key_len, q.device, q.dtype
        )
        return q, k, bias

    def _sdp_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_bias: torch.Tensor | None,
        T: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0
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
        attn_mask = position_bias
        if self.causal:
            mask = _causal_mask(T, q.device)
            attn_mask = position_bias.masked_fill(
                mask.reshape(1, 1, T, T), float("-inf")
            )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.scale,
        )

    def _linear_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0
        if self.causal:
            # Causal linear (ELU+1 Performer) requires a recurrent cumsum
            # formulation for correctness; fall back to FlashAttention-backed
            # SDP which is still efficient and numerically exact.
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

    def _cosine_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_bias: torch.Tensor | None,
        T: int,
    ) -> torch.Tensor:
        # True cosine attention: unit-norm Q/K, learnable inverse-temperature
        # (a la Swin-V2 / CosFormer). Without this rescaling, softmax on
        # cosine scores in [-1,1] is nearly uniform.
        dropout_p = self.dropout_p if self.training else 0.0
        qn = F.normalize(q, p=2, dim=-1)
        kn = F.normalize(k, p=2, dim=-1)
        scale = torch.exp(self.cos_log_scale)
        scores = torch.matmul(qn, kn.transpose(-2, -1)) * scale
        if position_bias is not None:
            scores = scores + position_bias
        if self.causal:
            mask = _causal_mask(T, q.device)
            scores = scores.masked_fill(mask.reshape(1, 1, T, T), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=1.0 / float(max(T, 1)))
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v)

    def _local_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_bias: torch.Tensor | None,
        T: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0
        W = max(4, int(T * self.LOCAL_WINDOW_RATIO))
        pos = torch.arange(T, device=q.device)
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
            mask = mask | _causal_mask(T, q.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=1.0 / float(T))
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v)

    def _probsparse_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_bias: torch.Tensor | None,
        B: int,
        H: int,
        T: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0
        c = self.PROBSPARSE_C
        n_top = min(T, max(1, int(c * math.log(T + 1))))
        n_sample = min(T, max(1, int(c * math.log(T + 1))))
        # Deterministic sampling: reduces variance in alpha gradients.
        cpu_idx = torch.randperm(T, generator=self._probsparse_gen)[:n_sample]
        sample_idx = cpu_idx.to(q.device)
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
            key_pos = torch.arange(T, device=q.device).reshape(1, 1, 1, T)
            scores_top = scores_top.masked_fill(key_pos > full_idx, float("-inf"))
        attn_top = F.softmax(scores_top, dim=-1)
        attn_top = torch.nan_to_num(attn_top, nan=1.0 / float(T))
        if dropout_p > 0:
            attn_top = F.dropout(attn_top, p=dropout_p)
        out_top = torch.matmul(attn_top, v)
        out_sparse.scatter_(2, idx_exp, out_top)
        return out_sparse

    def _apply_kernel(
        self,
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_bias: torch.Tensor | None,
        B: int,
        H: int,
        T: int,
    ) -> torch.Tensor:
        if mode == "sdp":
            return self._sdp_kernel(q, k, v, position_bias, T)
        if mode == "linear":
            return self._linear_kernel(q, k, v, T)
        if mode == "cosine":
            return self._cosine_kernel(q, k, v, position_bias, T)
        if mode == "local":
            return self._local_kernel(q, k, v, position_bias, T)
        return self._probsparse_kernel(q, k, v, position_bias, B, H, T)

    def _attn_mix_weights(self, tau: float | None = None) -> torch.Tensor:
        """Soft weights over MODES; honours ``variant_gdas``.

        ``tau`` defaults to ``self.temperature``; pass an explicit value to
        override (matches ``AttentionBridge._attn_mix_weights`` signature).
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

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads

        qkv = self.to_qkv(x).reshape(B, T, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q_raw, k_raw, v = qkv.unbind(0)

        # Position-mode selection: argmax(weights) with straight-through
        # gradient via ``weights[idx]`` multiplier below. Fixes the bug where
        # ``position_alphas`` received zero gradient signal.
        pos_weights = self._position_mix_weights()
        pos_idx = int(torch.argmax(pos_weights.detach()).item())
        position_mode = self.POSITION_MODES[pos_idx]
        q, k, position_bias = self._apply_position_mode(q_raw, k_raw, position_mode)
        pos_scalar = pos_weights[pos_idx] if self.position_searchable else None

        if self.searchable:
            weights = self._attn_mix_weights()
            if self.variant_gdas and self.training:
                # Single path: only the argmax kernel runs (~5x speedup). STE
                # gradient on alphas flows via the weights[idx] multiplier.
                idx = int(torch.argmax(weights.detach()).item())
                out = weights[idx] * self._apply_kernel(
                    self.MODES[idx], q, k, v, position_bias, B, H, T
                )
            else:
                out = sum(
                    weights[i]
                    * self._apply_kernel(self.MODES[i], q, k, v, position_bias, B, H, T)
                    for i in range(len(self.MODES))
                )
        else:
            out = self._apply_kernel(
                self.attention_type, q, k, v, position_bias, B, H, T
            )

        if pos_scalar is not None:
            out = pos_scalar * out

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


