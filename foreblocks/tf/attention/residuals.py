import torch
import torch.nn.functional as F
from torch import nn

from foreblocks.tf.norms import RMSNorm


def normalize_attention_residual_mode(attn_residual_type: str) -> str:
    """
    Normalize public residual mode strings.
    """

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
    """
    Full Attention Residuals (AttnRes)

    Maintains history of previous layer outputs and performs
    softmax attention over depth.
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))  # w_l
        self.norm = RMSNorm(dim)

    def forward(self, history):
        """
        history: list of tensors [v0, v1, ..., v_{l-1}]
                 each of shape [B, T, D]
        """

        # [L, B, T, D]
        V = torch.stack(history, dim=0)

        # normalize keys
        K = self.norm(V)

        # compute logits: q^T k
        # [L, B, T]
        logits = torch.einsum("d, l b t d -> l b t", self.query, K)

        # softmax over depth (L dimension)
        attn = F.softmax(logits, dim=0)

        # weighted sum
        h = torch.einsum("l b t, l b t d -> b t d", attn, V)

        return h


class BlockAttentionResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim)

    def forward(self, blocks, partial=None):
        """
        blocks: [b0, b1, ..., b_{n-1}]
        partial: current block accumulation, or None for the first sub-layer
            in a block where only completed block representations are visible
        """

        values = list(blocks)
        if partial is not None:
            values.append(partial)
        if not values:
            raise ValueError("BlockAttentionResidual requires at least one value tensor.")

        V = torch.stack(values, dim=0)
        K = self.norm(V)

        logits = torch.einsum("d, l b t d -> l b t", self.query, K)
        attn = F.softmax(logits, dim=0)

        h = torch.einsum("l b t, l b t d -> b t d", attn, V)

        return h
