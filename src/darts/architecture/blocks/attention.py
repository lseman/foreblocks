"""Multi-head self-attention with a searchable backend (SDP/linear/ProbSparse/...)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.attention_math import (
    _causal_mask,
    _make_alibi_slopes,
    _seasonal_relative_bias,
    _sinusoidal_features,
)
from .positional import RotaryPositionalEncoding

__all__ = [
    "SelfAttention",
]


class SelfAttention(nn.Module):
    """Self-attention block with selectable attention kernels.

    Modes: ``sdp``, ``linear``, ``probsparse``, ``cosine``, ``local``.
    Linear attention uses prefix sufficient statistics in causal mode;
    ProbSparse uses causal query sampling and prefix context.
    """

    MODES: tuple[str, ...] = ("sdp", "linear", "probsparse", "cosine", "local")
    CAUSAL_MODES: tuple[str, ...] = MODES
    POSITION_MODES: tuple[str, ...] = (
        "rope", "alibi", "none", "seasonal",
        "sinusoidal", "learned", "relative",
    )
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
        self.MODES = self.CAUSAL_MODES if causal else self.MODES
        resolved_attention_type = str(attention_type).lower()
        resolved_position_mode = str(position_mode).lower()
        if resolved_attention_type not in (*self.MODES, "auto"):
            raise ValueError(
                f"attention_type must be one of {(*self.MODES, 'auto')}, "
                f"got {resolved_attention_type!r}"
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

        # --- New position-encoding parameters (sinusoidal, learned, relative) ---
        # Learnable per-position embeddings (max_seq_len): for "learned" mode.
        self.learned_pos = nn.Parameter(
            torch.randn(rope_max_seq_len, self.head_dim) * 0.02
        )
        # Learnable relative-position bias table: (2*T-1) x heads
        # Scalar bias per relative displacement per head, added to scores.
        self.relative_pos_bias = nn.Parameter(
            torch.zeros(2 * rope_max_seq_len - 1, self.heads)
        )
        # ProbSparse sampling is seeded afresh per forward, so evaluating a
        # searched and fixed model uses the same sampled keys.

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        # Searches created while causal linear/ProbSparse were disabled have
        # three logits. Preserve those learned choices on load.
        key = prefix + "attn_alphas"
        saved = state_dict.get(key)
        if (
            self.causal and self.searchable and isinstance(saved, torch.Tensor)
            and saved.shape == (3,) and self.attn_alphas.shape == (5,)
        ):
            expanded = saved.new_full((5,), -30.0)
            expanded[[0, 3, 4]] = saved
            state_dict[key] = expanded
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

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

    def _apply_sinusoidal_pos(
        self, q: torch.Tensor, k: torch.Tensor, query_len: int, key_len: int
    ) -> tuple:
        """Sinusoidal positional encoding (Vaswani et al. 2017)."""
        pos_q = _sinusoidal_features(
            query_len, self.head_dim, device=q.device, dtype=q.dtype
        ).reshape(1, 1, query_len, self.head_dim)
        pos_k = _sinusoidal_features(
            key_len, self.head_dim, device=k.device, dtype=k.dtype
        ).reshape(1, 1, key_len, self.head_dim)
        scale = self.positional_scale.to(dtype=q.dtype)
        q = q + scale * pos_q
        k = k + scale * pos_k
        return q, k, None

    def _apply_learned_pos(
        self, q: torch.Tensor, k: torch.Tensor, query_len: int, key_len: int
    ) -> tuple:
        """Learned positional embeddings."""
        max_len = self.learned_pos.size(0)
        pos_q = self.learned_pos[:query_len].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
        pos_k = self.learned_pos[:key_len].unsqueeze(0).unsqueeze(0).to(dtype=k.dtype)
        q = q + pos_q
        k = k + pos_k
        return q, k, None

    def _apply_relative_pos(
        self, q: torch.Tensor, k: torch.Tensor,
        query_len: int, key_len: int, device, dtype
    ) -> tuple:
        """DeiT-style relative position bias (Touvron et al. 2021).

        Returns (q, k, bias) where bias has shape [1, heads, Q, K]
        for direct addition to attention scores.
        """
        # Relative displacement: rel[j] = j - i for k_pos=j, q_pos=i
        # Shape: [Q, K], range [-(K-1), Q-1]
        rel_idx = torch.arange(key_len, device=device).unsqueeze(0) - \
                  torch.arange(query_len, device=device).unsqueeze(1)
        offset = key_len - 1  # shift to [0, Q+K-2]
        rel_idx = (rel_idx + offset).clamp(0, query_len + key_len - 2)
        # Gather per-head bias: rel_idx [Q,K] x bias [max_rel,heads] -> [Q,K,heads]
        bias = self.relative_pos_bias[rel_idx]  # [Q, K, heads]
        # Transpose to [heads, Q, K] and add batch dim -> [1, heads, Q, K]
        bias = bias.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
        return q, k, bias

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
        self, q: torch.Tensor, k: torch.Tensor, position_mode: str,
        *, materialize_bias: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        query_len = q.size(-2)
        key_len = k.size(-2)
        if position_mode == "rope":
            q, k = self._apply_rotary_pair(q, k, query_len, key_len)
            return q, k, None
        if position_mode == "sinusoidal":
            q, k, _ = self._apply_sinusoidal_pos(q, k, query_len, key_len)
            return q, k, None
        if position_mode == "learned":
            q, k, _ = self._apply_learned_pos(q, k, query_len, key_len)
            return q, k, None
        if position_mode == "relative":
            if not materialize_bias:
                return q, k, None
            return self._apply_relative_pos(
                q, k, query_len, key_len, q.device, q.dtype
            )
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
            ) if materialize_bias else None
            return q, k, bias
        bias = self._build_relative_bias(
            position_mode, query_len, key_len, q.device, q.dtype
        ) if materialize_bias else None
        return q, k, bias

    def _linear_lag_weights(
        self, position_mode: str, length: int, dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Positive per-head lag weights for Toeplitz position biases."""
        lags = torch.arange(1 - length, length, device=device)
        if position_mode == "relative":
            bias = self.relative_pos_bias[length - 1 - lags].transpose(0, 1)
            bias = bias.to(dtype=dtype)
        elif position_mode == "alibi":
            slopes = self.alibi_slopes.to(device=device, dtype=dtype)
            bias = -slopes[:, None] * lags.abs().to(dtype)[None, :]
        elif position_mode == "seasonal":
            periods = torch.tensor(
                [4.0, 8.0, 16.0, 24.0, 48.0], device=device, dtype=dtype
            )
            wave = torch.cos(
                2.0 * torch.pi * lags.to(dtype)[None, :] / periods[:, None]
            ).mean(dim=0)
            slopes = self.alibi_slopes.to(device=device, dtype=dtype)
            bias = 0.1 * slopes[:, None] * wave[None, :]
        else:
            raise ValueError(f"No lag bias for position mode {position_mode!r}")
        if self.causal:
            valid = lags[None, :] >= 0
            maximum = bias.masked_fill(~valid, float("-inf")).amax(
                dim=-1, keepdim=True
            )
            weights = (bias - maximum).exp().masked_fill(~valid, 0.0)
        else:
            weights = (bias - bias.amax(dim=-1, keepdim=True)).exp()
        return weights

    @staticmethod
    def _lag_convolution(signal: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Sum a Toeplitz lag kernel over time with FFT convolution."""
        length = signal.size(2)
        fft_size = 1 << (3 * length - 3).bit_length()
        signal_fft = torch.fft.rfft(signal, n=fft_size, dim=2)
        weight_fft = torch.fft.rfft(weights, n=fft_size, dim=1)
        weight_fft = weight_fft.reshape(
            1, weights.size(0), weight_fft.size(1),
            *([1] * (signal.dim() - 3)),
        )
        result = torch.fft.irfft(
            signal_fft * weight_fft, n=fft_size, dim=2
        )
        return result[:, :, length - 1 : 2 * length - 1]

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
        position_bias: torch.Tensor | None,
        T: int,
        position_mode: str | None = None,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0
        accum_dtype = (
            torch.float32
            if q.dtype in (torch.float16, torch.bfloat16)
            else q.dtype
        )
        q_feat = F.elu(q.to(accum_dtype) * self.scale) + 1.0
        k_feat = F.elu(k.to(accum_dtype)) + 1.0
        values = v.to(accum_dtype)
        if position_mode in {"alibi", "seasonal", "relative"}:
            lag_weights = self._linear_lag_weights(
                position_mode, T, accum_dtype, q.device
            )
            key_sum = self._lag_convolution(k_feat, lag_weights)
            kv_sum = self._lag_convolution(
                k_feat.unsqueeze(-1) * values.unsqueeze(-2), lag_weights
            )
            denom = (q_feat * key_sum).sum(dim=-1).clamp_min(1e-6)
            output = (q_feat.unsqueeze(-1) * kv_sum).sum(dim=-2)
            output = (output / denom.unsqueeze(-1)).to(v.dtype)
            return F.dropout(output, p=dropout_p) if dropout_p else output
        if position_bias is not None:
            # Public kernel callers can still supply an arbitrary pairwise
            # bias, which has no lag representation and requires pairwise work.
            scores = torch.matmul(q_feat, k_feat.transpose(-2, -1))
            logits = scores.clamp_min(1e-12).log() + position_bias.to(accum_dtype)
            if self.causal:
                logits = logits.masked_fill(
                    _causal_mask(T, q.device).reshape(1, 1, T, T),
                    float("-inf"),
                )
            weights = F.softmax(logits, dim=-1)
            if dropout_p:
                weights = F.dropout(weights, p=dropout_p)
            return torch.matmul(weights, values).to(v.dtype)
        if self.causal:
            key_prefix = k_feat.cumsum(dim=2)
            kv_prefix = (k_feat.unsqueeze(-1) * values.unsqueeze(-2)).cumsum(dim=2)
            denom = (q_feat * key_prefix).sum(dim=-1).clamp_min(1e-6)
            out_linear = (q_feat.unsqueeze(-1) * kv_prefix).sum(dim=-2)
        else:
            kv = torch.einsum("bhtd,bhtv->bhdv", k_feat, values)
            key_sum = k_feat.sum(dim=2)
            denom = torch.einsum("bhtd,bhd->bht", q_feat, key_sum).clamp_min(1e-6)
            out_linear = torch.einsum("bhtd,bhdv->bhtv", q_feat, kv)
        output = out_linear / denom.unsqueeze(-1)
        output = output.to(v.dtype)
        return F.dropout(output, p=dropout_p) if dropout_p else output

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
        if self.causal:
            return self._causal_probsparse_kernel(q, k, v, position_bias, T)
        c = self.PROBSPARSE_C
        n_top = min(T, max(1, int(c * math.log(T + 1))))
        n_sample = min(T, max(1, int(c * math.log(T + 1))))
        # Deterministic sampling: reduces variance in alpha gradients.
        cpu_idx = torch.randperm(
            T, generator=torch.Generator().manual_seed(0xC0FFEE)
        )[:n_sample]
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

    def _causal_probsparse_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_bias: torch.Tensor | None,
        T: int,
    ) -> torch.Tensor:
        """Sparse exact queries over causal prefixes; lazy queries use prefix means.

        Each query samples only earlier keys and is selected using only earlier
        query sparsity scores. This makes outputs invariant to future tokens.
        """
        c = self.PROBSPARSE_C
        query_pos = torch.arange(T, device=q.device, dtype=torch.long)
        max_sample = min(T, max(1, int(c * math.log(T + 1))))
        sample_slots = torch.arange(max_sample, device=q.device, dtype=torch.long)
        sample_counts = torch.minimum(
            query_pos + 1,
            (c * torch.log(query_pos.to(torch.float64) + 2)).long().clamp_min(1),
        )
        valid_sample = sample_slots.unsqueeze(0) < sample_counts.unsqueeze(1)
        # A stable per-(query, slot) hash keeps earlier samples unchanged when
        # the sequence is extended. Repeated samples are permitted.
        sample_idx = (
            (query_pos.unsqueeze(1) * 1013904223
             + sample_slots.unsqueeze(0) * 2654435761) % (2**31)
        ) % (query_pos.unsqueeze(1) + 1)
        sampled_keys = k[:, :, sample_idx, :]
        sampled_scores = (q.unsqueeze(-2) * sampled_keys).sum(dim=-1) * self.scale
        largest = sampled_scores.masked_fill(
            ~valid_sample.reshape(1, 1, T, max_sample), float("-inf")
        ).amax(dim=-1)
        sample_mean = (
            sampled_scores
            * valid_sample.reshape(1, 1, T, max_sample)
        ).sum(dim=-1) / sample_counts.clamp_min(1)
        sparsity = largest - sample_mean

        # Online top-u: a query is exact if it enters the top-u at its own
        # position. Global top-k would let future queries change past outputs.
        leaders = sparsity.new_empty((*sparsity.shape[:2], 0))
        selected = []
        for t in range(T):
            candidates = torch.cat((leaders, sparsity[:, :, t : t + 1]), dim=-1)
            budget = min(t + 1, max(1, int(c * math.log(t + 2))))
            leaders, indices = candidates.topk(budget, dim=-1)
            selected.append((indices == candidates.size(-1) - 1).any(dim=-1))
        active = torch.stack(selected, dim=-1)

        prefix_mean = v.cumsum(dim=2) / (
            query_pos + 1
        ).to(v.dtype).reshape(1, 1, T, 1)
        output = prefix_mean.clone()
        key_pos = query_pos.unsqueeze(0)
        for batch in range(q.size(0)):
            for head in range(q.size(1)):
                chosen = active[batch, head].nonzero(as_tuple=True)[0]
                if chosen.numel() == 0:
                    continue
                scores = q[batch, head, chosen] @ k[batch, head].transpose(0, 1)
                scores = scores * self.scale
                if position_bias is not None:
                    scores = scores + position_bias[0, head, chosen]
                scores = scores.masked_fill(
                    key_pos > chosen.unsqueeze(1), float("-inf")
                )
                weights = F.softmax(scores, dim=-1)
                if self.training and self.dropout_p:
                    weights = F.dropout(weights, p=self.dropout_p)
                exact = weights @ v[batch, head]
                output[batch, head].index_copy_(0, chosen, exact)
        return output

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
        position_mode: str | None = None,
    ) -> torch.Tensor:
        if mode == "sdp":
            return self._sdp_kernel(q, k, v, position_bias, T)
        if mode == "linear":
            return self._linear_kernel(
                q, k, v, position_bias, T, position_mode=position_mode
            )
        if mode == "cosine":
            return self._cosine_kernel(q, k, v, position_bias, T)
        if mode == "local":
            return self._local_kernel(q, k, v, position_bias, T)
        if mode == "probsparse":
            return self._probsparse_kernel(q, k, v, position_bias, B, H, T)
        raise ValueError(f"Unsupported attention mode: {mode!r}")

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
        pos_scalar = pos_weights[pos_idx] if self.position_searchable else None
        weights = self._attn_mix_weights() if self.searchable else None
        single_path = self.searchable and self.variant_gdas and self.training
        selected_idx = (
            int(torch.argmax(weights.detach()).item())
            if single_path and weights is not None else None
        )
        active_mode = (
            self.MODES[selected_idx] if selected_idx is not None
            else self.attention_type if not self.searchable else None
        )
        q, k, position_bias = self._apply_position_mode(
            q_raw, k_raw, position_mode,
            materialize_bias=active_mode != "linear",
        )

        if self.searchable:
            if single_path:
                # Single path: only the argmax kernel runs (~5x speedup). STE
                # gradient on alphas flows via the weights[idx] multiplier.
                idx = selected_idx
                out = weights[idx] * self._apply_kernel(
                    self.MODES[idx], q, k, v, position_bias, B, H, T,
                    position_mode,
                )
            else:
                out = sum(
                    weights[i]
                    * self._apply_kernel(
                        self.MODES[i], q, k, v, position_bias, B, H, T,
                        position_mode,
                    )
                    for i in range(len(self.MODES))
                )
        else:
            out = self._apply_kernel(
                self.attention_type, q, k, v, position_bias, B, H, T,
                position_mode,
            )

        if pos_scalar is not None:
            out = pos_scalar * out

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
