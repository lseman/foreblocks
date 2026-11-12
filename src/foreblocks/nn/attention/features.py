"""Private score and output transforms for MultiAttention.

The mixin owns no parameters: initialization and parameter registration stay on
MultiAttention. It groups optional temperature, softcap, multi-scale masking,
gating, normalization, and head-importance behavior used by kernel dispatch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _AttentionFeatures:
    """Feature transforms using the owning attention module's registered state."""

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
