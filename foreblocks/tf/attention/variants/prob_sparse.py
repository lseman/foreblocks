import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class ProbSparseAttentionImpl:
    def __init__(self, parent):
        self.parent = parent

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
        layer_state=None,
        **_,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        B, T_q, _ = query.shape
        q, k, v = self.parent._prepare_qkv_attention(query, key, value, layer_state)
        out, weights = self._prob_sparse_attention(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
        )
        return self.parent._finalize_projected_output(out, B, T_q), weights, layer_state

    def _prob_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        u = max(1, min(T_q, int(5 * math.log(max(T_q, 2)))))
        sample_k = max(1, min(T_k, int(math.ceil(5 * math.log(max(T_k, 2))))))

        if u >= T_q or sample_k >= T_k:
            return self.parent._compute_attention(
                q,
                k,
                v,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
            )

        k_sample = k[:, :, :: max(1, T_k // sample_k), :][:, :, :sample_k, :]
        scores_sample = torch.matmul(q, k_sample.transpose(-2, -1)) * self.parent.scale

        sparsity = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)

        _, top_idx = torch.topk(sparsity, k=u, dim=-1)
        top_q = torch.gather(
            q,
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, D),
        )

        scores = torch.matmul(top_q, k.transpose(-2, -1)) * self.parent.scale

        if is_causal and not self.parent.cross_attention:
            q_pos = top_idx.unsqueeze(-1)
            k_pos = torch.arange(T_k, device=q.device).view(1, 1, 1, T_k)
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        if attn_mask is not None:
            mask = self.parent._normalize_attn_mask(attn_mask, B, H, T_q, T_k)
            mask_top = torch.gather(
                mask,
                2,
                top_idx.unsqueeze(-1).expand(-1, -1, -1, T_k),
            )
            scores = scores.masked_fill(mask_top, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(B, 1, 1, T_k),
                float("-inf"),
            )

        scores_max = scores.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(scores - scores_max, dim=-1)
        weights = self.parent._dropout_weights(weights)

        top_out = torch.matmul(weights, v)

        output = v.mean(dim=2, keepdim=True).expand(B, H, T_q, D).clone()
        output.scatter_(
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, D),
            top_out,
        )

        full_weights = None
        if need_weights:
            full_weights = torch.zeros(
                B,
                H,
                T_q,
                T_k,
                device=q.device,
                dtype=weights.dtype,
            )
            full_weights.scatter_(
                2,
                top_idx.unsqueeze(-1).expand(-1, -1, -1, T_k),
                weights,
            )

        return output, full_weights
