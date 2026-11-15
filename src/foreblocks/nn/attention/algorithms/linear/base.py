"""foreblocks.nn.attention.algorithms.linear.base.

Shared utilities and mixins for linear attention backends.

Provides RoPEMixin for per-backend rotary position embedding on Q/K tensors,
causal Conv1d helper, and FeatureMapRegistry for configurable linear attention
feature maps (ELU+1, ReLU, SiLU, RFF, etc.). RDA and GLA use RoPEMixin;
FeatureMapRegistry is used by the RDA backend.

Core API:
- RoPEMixin: per-backend RoPE on Q/K with lazy embedding initialization
- FeatureMapRegistry: factory for linear-attention feature maps
- _causal_conv1d: causal Conv1d + activation helper

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPEMixin:
    def _init_pos_encoding(self) -> None:
        self._rotary_emb: nn.Module | None = None

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int | torch.Tensor = 0,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "pos_encoding_type", "sinusoidal") != "rope":
            return q, k

        from foreblocks.nn.embeddings.rope_alibi_helpers import (
            create_rotary_embedding,
        )
        from foreblocks.nn.embeddings.rotary import apply_rotary_emb

        Lq = q.shape[2]
        Lk = k.shape[2]
        if isinstance(offset, torch.Tensor) or padding_mask is not None:
            starts = torch.as_tensor(offset, device=q.device, dtype=torch.long).expand(
                q.size(0)
            )
            if padding_mask is not None:
                positions = (~padding_mask).long().cumsum(-1).sub(1).clamp_min(
                    0
                ) + starts[:, None]
            else:
                positions = (
                    torch.arange(max(Lq, Lk), device=q.device)[None] + starts[:, None]
                )
            seqlen = int(positions.max().item()) + 1
        else:
            positions = None
            seqlen = offset + max(Lq, Lk)
        if self._rotary_emb is None:
            self._rotary_emb = create_rotary_embedding(
                head_dim=self.d_head, max_seq_len=seqlen
            )
        # RotaryEmbedding builds its cos/sin cache lazily and on-device.
        self._rotary_emb._update_cos_sin_cache(seqlen, device=q.device, dtype=q.dtype)
        cos = self._rotary_emb._cos_cached
        sin = self._rotary_emb._sin_cached
        if cos is None or sin is None:
            raise RuntimeError("RotaryEmbedding cache was not initialized")

        q_cos = (
            cos[offset : offset + Lq] if positions is None else cos[positions[:, :Lq]]
        )
        q_sin = (
            sin[offset : offset + Lq] if positions is None else sin[positions[:, :Lq]]
        )
        k_cos = (
            cos[offset : offset + Lk] if positions is None else cos[positions[:, :Lk]]
        )
        k_sin = (
            sin[offset : offset + Lk] if positions is None else sin[positions[:, :Lk]]
        )
        q_rot = apply_rotary_emb(q.transpose(1, 2), q_cos, q_sin).transpose(1, 2)
        k_rot = apply_rotary_emb(k.transpose(1, 2), k_cos, k_sin).transpose(1, 2)
        return q_rot, k_rot


def validate_key_padding_mask(
    mask: torch.Tensor | None, batch: int, length: int, device: torch.device
) -> torch.Tensor | None:
    """Padding is boolean [B,T], with True denoting a skipped token."""
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != (batch, length):
            raise ValueError("key_padding_mask must be boolean with shape [B,T]")
        if mask.device != device:
            raise ValueError("key_padding_mask must share the input device")
    return mask


def mask_padded_tokens(
    x: torch.Tensor, padding_mask: torch.Tensor | None
) -> torch.Tensor:
    return x if padding_mask is None else x.masked_fill(padding_mask.unsqueeze(-1), 0)


def validate_linear_attention_mask(
    mask: torch.Tensor | None, query: torch.Tensor, key: torch.Tensor, is_causal: bool
) -> None:
    """Accept redundant masks; arbitrary pairwise masks cannot use a scan."""
    if mask is None:
        return
    if mask.ndim not in (2, 3, 4) or mask.shape[-2:] != (query.size(1), key.size(1)):
        raise ValueError("linear attention mask must end in [query_length, key_length]")
    if mask.dtype == torch.bool:
        blocked = mask
    elif mask.is_floating_point() and bool(((mask == 0) | torch.isneginf(mask)).all()):
        blocked = torch.isneginf(mask)
    else:
        raise ValueError("linear attention supports boolean or zero/-inf masks only")
    if not bool(blocked.any()):
        return
    causal = torch.ones(
        query.size(1), key.size(1), device=mask.device, dtype=torch.bool
    ).triu(1)
    if is_causal and bool((blocked == causal).all()):
        return
    raise ValueError(
        "arbitrary attn_mask is unsupported by linear attention; use is_causal and key_padding_mask"
    )


def streaming_causal_conv1d(
    x: torch.Tensor,
    conv: nn.Conv1d,
    cache: dict | None = None,
    key: str = "history",
    *,
    activation: str | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convolve [B,T,D], retaining raw inputs needed by the next chunk."""
    from foreblocks.kernels.attention.short_conv import streaming_short_conv

    history = cache.get(key) if cache is not None else None
    if padding_mask is None:
        output, next_history = streaming_short_conv(
            x, conv.weight.squeeze(1), conv.bias, history, activation=activation
        )
    else:
        validate_key_padding_mask(padding_mask, x.size(0), x.size(1), x.device)
        clean = mask_padded_tokens(x, padding_mask)
        if history is None:
            history = x.new_zeros(x.size(0), x.size(2), conv.kernel_size[0] - 1)
        outputs, histories = [], []
        for b in range(x.size(0)):
            valid = (~padding_mask[b]).nonzero().flatten()
            # Preserve zero gradients for an entirely padded convolution.
            row = clean[b : b + 1] * 0 + conv.weight.sum() * 0
            if conv.bias is not None:
                row = row + conv.bias.sum() * 0
            next_row_history = history[b : b + 1]
            if valid.numel():
                values, next_row_history = streaming_short_conv(
                    clean[b : b + 1].index_select(1, valid),
                    conv.weight.squeeze(1),
                    conv.bias,
                    next_row_history,
                    activation=activation,
                )
                row = row.index_copy(1, valid, values)
            outputs.append(row)
            histories.append(next_row_history)
        output, next_history = torch.cat(outputs), torch.cat(histories)
    if cache is not None:
        cache[key] = next_history
    return output


class RandomFourierFeatures(nn.Module):
    def __init__(self, d_head: int, num_features: int) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        self.register_buffer(
            "omega", torch.randn(d_head, num_features) / num_features**0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x.square().sum(-1, keepdim=True)) * torch.cos(
            x @ self.omega
        )


def _causal_conv1d(
    x: torch.Tensor, conv: nn.Conv1d, activation: nn.Module, kernel_size: int
) -> torch.Tensor:
    T0 = x.size(1)
    x = x.transpose(1, 2).contiguous()  # [B, D, T]
    x = conv(x)[:, :, :T0].contiguous()  # crop causal padding
    return activation(x).transpose(1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Feature maps for RDA backend
# ─────────────────────────────────────────────────────────────────────────────


class FeatureMapRegistry:
    @staticmethod
    def build(name: str, d_head: int, num_features: int | None = None):
        if name == "elu":
            return lambda x: F.elu(x) + 1.0
        if name == "relu":
            return F.relu
        if name == "silu":
            return F.silu
        if name == "leaky_relu":
            return F.leaky_relu
        if name == "rff":
            return RandomFourierFeatures(
                d_head, d_head if num_features is None else num_features
            )
        if name == "tanh":
            return lambda x: torch.tanh(x) + 1.0  # keep non-negative
        if name == "cos_cos":
            # Cosine-Cosine: phi(x) = cos(x), but requires L2-normalised inputs
            return torch.cos
        raise ValueError(f"Unknown feature_map: {name}")
