"""foreblocks.sequence.block_stack.

Heterogeneous block stacking with Nemotron-style sequential composition.

Stacks a flat sequence of block types (e.g. mamba2, attn, moe) as pre-norm
residual sub-blocks. Unlike HybridMamba2Block which fuses SSM and attention
in parallel inside one block, this composes heterogeneous block types
sequentially — the pattern used by models like NVIDIA Nemotron.
Designed to be passed to ForecastingModel as the encoder backbone with
``model_type="hybrid_mamba"``. MoE auxiliary losses accumulate into
``self.aux_loss`` for automatic aggregation.

Core API:
- BlockStack: sequential stack of heterogeneous block types
- BLOCK_TYPES: allowed block type names

"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.modules.attention.config import AttentionConfig
from foreblocks.modules.attention.multi_att import MultiAttention
from foreblocks.modules.moe.ff import FeedForwardBlock
from foreblocks.sequence.mamba.mamba2 import Mamba2Block
from foreblocks.sequence.mamba.mamba3 import Mamba3Block

BLOCK_TYPES = ("mamba2", "mamba3", "attn", "moe")


class _Residual(nn.Module):
    def __init__(self, kind: str, sublayer: nn.Module, d_model: int, dropout: float):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None):
        h = self.norm(x)
        if self.kind in ("mamba2", "mamba3"):
            out = self.sublayer(h, attention_mask=attention_mask)
            aux = x.new_zeros(())
        elif self.kind == "attn":
            out, _, _ = self.sublayer(h, is_causal=True)
            aux = x.new_zeros(())
        else:  # moe
            padding_mask = (
                None
                if attention_mask is None
                else ~attention_mask.to(device=x.device, dtype=torch.bool)
            )
            out, aux = self.sublayer(h, return_aux_loss=True, padding_mask=padding_mask)
        return x + self.drop(out), aux


class BlockStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        spec: list[str],
        *,
        input_size: int | None = None,
        dropout: float = 0.0,
        mamba_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        moe_kwargs: dict | None = None,
    ):
        super().__init__()
        if not spec:
            raise ValueError("spec must be a non-empty list of block types")
        bad = [s for s in spec if s not in BLOCK_TYPES]
        if bad:
            raise ValueError(f"unknown block type(s) {bad}; valid: {BLOCK_TYPES}")

        self.d_model = d_model
        self.input_size = input_size or d_model
        self.output_size = d_model
        self.spec = list(spec)

        self.input_proj = (
            nn.Identity()
            if self.input_size == d_model
            else nn.Linear(self.input_size, d_model)
        )

        mamba_kwargs = mamba_kwargs or {}
        attn_kwargs = attn_kwargs or {}
        moe_kwargs = moe_kwargs or {}

        self.blocks = nn.ModuleList(
            _Residual(
                kind,
                self._make(kind, d_model, mamba_kwargs, attn_kwargs, moe_kwargs),
                d_model,
                dropout,
            )
            for kind in spec
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.aux_loss: torch.Tensor = torch.zeros(())

    @staticmethod
    def _make(kind, d_model, mamba_kwargs, attn_kwargs, moe_kwargs) -> nn.Module:
        # _Residual already pre-norms, so disable the blocks' own pre-norm.
        if kind == "mamba2":
            return Mamba2Block(
                d_model=d_model, **{"use_pre_norm": False, **mamba_kwargs}
            )
        if kind == "mamba3":
            return Mamba3Block(
                d_model=d_model, **{"use_pre_norm": False, **mamba_kwargs}
            )
        if kind == "attn":
            return MultiAttention(
                AttentionConfig.from_legacy_kwargs(
                    d_model=d_model, **{"n_heads": 8, **attn_kwargs}
                )
            )
        # moe: FeedForwardBlock needs dim_ff and use_moe=True to actually route.
        return FeedForwardBlock(
            **{
                "d_model": d_model,
                "dim_ff": 4 * d_model,
                "use_moe": True,
                **moe_kwargs,
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        aux = x.new_zeros(())
        for block in self.blocks:
            x, block_aux = block(x, attention_mask)
            aux = aux + block_aux
        self.aux_loss = aux
        return self.final_norm(x)
