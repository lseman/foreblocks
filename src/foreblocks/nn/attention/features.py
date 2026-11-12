"""Private score and output transforms for MultiAttention.

Initialization helpers register parameters directly on MultiAttention; the
mixin introduces no nested modules or checkpoint prefixes. It groups optional
temperature, softcap, multi-scale masking, gating, normalization, and
head-importance behavior used by kernel dispatch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.nn.attention.config import AttentionFeatureConfig


class _AttentionFeatures:
    """Feature transforms using the owning attention module's registered state."""

    def _init_attention_features(self, features: AttentionFeatureConfig) -> None:
        # ── QK Normalization ───────────────────────────────────────────────
        self.qk_norm = bool(features.qk_norm)
        if features.qk_norm_type not in {"rms", "l2"}:
            raise ValueError(
                f"qk_norm_type must be 'rms' or 'l2', got {features.qk_norm_type}"
            )
        self.qk_norm_type = features.qk_norm_type
        if self.qk_norm and self.qk_norm_type == "rms":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        # ── Logit softcapping (Gemini-style) ───────────────────────────────
        self.logit_softcap = (
            float(features.logit_softcap)
            if features.logit_softcap is not None
            else None
        )
        self.use_learned_temp = bool(features.learned_temperature)
        if self.use_learned_temp:
            # Per-head learnable temperature
            self.temperature = nn.Parameter(
                torch.ones(self.n_heads) * float(features.temperature_init)
            )
        else:
            self.temperature = None

        # ── Subquery normalization (DeepSeek-style) ────────────────────────
        self.use_subquery_norm = bool(features.subquery_norm)
        self.subquery_norm_mode = str(features.subquery_norm_mode)
        if self.use_subquery_norm:
            if self.subquery_norm_mode == "learned":
                # Per-head learnable gate (applied to attention output)
                self.subquery_gate = nn.Linear(self.head_dim, self.head_dim, bias=True)
            else:
                self.subquery_gate = None
        else:
            self.subquery_gate = None

        # ── Multi-scale masking (hierarchical attention) ───────────────────
        self.use_multiscale_mask = bool(features.multiscale_mask)
        self.multiscale_window_ratio = float(features.multiscale_window_ratio)
        self.multiscale_topk = int(features.multiscale_topk)
        if self.use_multiscale_mask:
            if not (0.0 < self.multiscale_window_ratio < 1.0):
                raise ValueError("multiscale_window_ratio must be in (0, 1)")
            if self.multiscale_topk <= 0:
                raise ValueError("multiscale_topk must be > 0")

        # ── Normalized attention output (LeLA-style) ──────────────────────
        self.use_normalized_attn_out = bool(features.normalized_output)
        self.norm_attn_type = str(features.normalized_output_type)
        if self.use_normalized_attn_out:
            if features.normalized_output_type == "rms":
                self.attn_out_norm = nn.RMSNorm(self.d_model, eps=1e-5)
            elif features.normalized_output_type == "layer":
                self.attn_out_norm = nn.LayerNorm(self.d_model, eps=1e-5)
            else:
                raise ValueError(
                    f"norm_attn_type must be 'rms' or 'layer', got {features.normalized_output_type}"
                )
        else:
            self.attn_out_norm = None

        # ── Head importance pruning (Lottery Ticket style) ──────────────────
        self.use_head_importance = bool(features.head_importance)
        self.head_importance_sparsity = float(features.head_importance_sparsity)
        if self.use_head_importance:
            if not (0.0 <= self.head_importance_sparsity < 1.0):
                raise ValueError("head_importance_sparsity must be in [0, 1)")
            # Learnable head importance scores (per head)
            self.head_importance_scores = nn.Parameter(torch.ones(self.n_heads))
        else:
            self.head_importance_scores = None

    def _init_gated_attention(self, features: AttentionFeatureConfig) -> None:
        # ── Gated Attention (G1) ─────────────────────────────────────────────
        self.use_gated_attention = features.gated_attention
        self.gated_attn_mode = features.gated_attention_mode

        if self.use_gated_attention:
            if features.gated_attention_mode not in {"per_head", "shared"}:
                raise ValueError(
                    f"gated_attn_mode must be 'per_head' or 'shared', got {features.gated_attention_mode}"
                )

            # Both modes project channels; shared mode averages heads at execution.
            self.gate_proj = nn.Linear(
                self.head_dim, self.head_dim, bias=features.gated_attention_bias
            )
        else:
            self.gate_proj = None

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
