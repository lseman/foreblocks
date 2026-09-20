"""Transformer attention-residual feature policies.

Attention Residuals — learned softmax-weighted aggregation over layer outputs (arXiv:2603.15031).

https://arxiv.org/abs/2603.15031

Replaces fixed residual accumulation (h_prev + o) with a learnable depth-query
that attends over preceding layer outputs via softmax. Two variants: Full AttnRes
(attends over all layers) and Block AttnRes (attends over block-level
representations for reduced overhead).

Core API:
- AttentionResidual: full AttnRes — softmax attention over all preceding layer outputs
- BlockAttentionResidual: block AttnRes — softmax attention over block-level representations
- normalize_attention_residual_mode: normalize residual type strings

"""

import torch
import torch.nn.functional as F
from torch import nn

from foreblocks.layers.norms import RMSNorm


def normalize_attention_residual_mode(attn_residual_type: str) -> str:
    mode = str(attn_residual_type).strip().lower()
    if mode == "block":
        return "block"
    if mode == "full":
        return "full"
    raise ValueError(
        "attn_residual_type must be one of {'full', 'block'}; "
        f"got {attn_residual_type!r}"
    )


class AttentionResidual(nn.Module):
    def __init__(self, dim, scale=None):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))  # w_l
        self.norm = RMSNorm(dim)
        self.scale = dim**-0.5 if scale is None else scale

    def forward(self, history, return_weights=False):

        # [L, B, T, D]
        V = torch.stack(history, dim=0)

        # normalize keys
        K = self.norm(V)

        # compute scaled logits: scale * q^T k
        # [L, B, T]
        logits = torch.einsum("d, l b t d -> l b t", self.query, K) * self.scale

        # softmax over depth (L dimension)
        attn = F.softmax(logits, dim=0)

        # weighted sum
        h = torch.einsum("l b t, l b t d -> b t d", attn, V)

        if return_weights:
            return h, attn
        return h


class BlockAttentionResidual(nn.Module):
    def __init__(self, dim, scale=None):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim)
        self.scale = dim**-0.5 if scale is None else scale

    def forward(self, blocks, partial=None, return_weights=False):

        values = list(blocks)
        if partial is not None:
            values.append(partial)
        if not values:
            raise ValueError(
                "BlockAttentionResidual requires at least one value tensor."
            )

        V = torch.stack(values, dim=0)
        K = self.norm(V)

        logits = torch.einsum("d, l b t d -> l b t", self.query, K) * self.scale
        attn = F.softmax(logits, dim=0)

        h = torch.einsum("l b t, l b t d -> b t d", attn, V)

        if return_weights:
            return h, attn
        return h
