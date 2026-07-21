"""foreblocks.layers.graph.conv.jump_knowledge.

Jump Knowledge layer implementation.

"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from foreblocks.layers.graph.common import Tensor, xavier_zero_bias


class JumpKnowledge(nn.Module):
    def __init__(
        self,
        mode: Literal["none", "last", "sum", "max", "concat", "lstm", "attn"] = "none",
        hidden_size: int | None = None,
        output_size: int | None = None,
        num_layers_hint: int | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.hidden = hidden_size
        self.output = output_size
        self.concat_proj: nn.Linear | None = None
        self.attn_score: nn.Linear | None = None
        self.input_proj: nn.Module | None = None
        self.out_proj: nn.Module = nn.Identity()

        if mode == "lstm":
            if hidden_size is None:
                raise ValueError("hidden_size required for LSTM JK")
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.out_proj = (
                nn.Identity()
                if output_size is None or output_size == hidden_size
                else nn.Linear(hidden_size, output_size)
            )
            if isinstance(self.out_proj, nn.Linear):
                xavier_zero_bias(self.out_proj)
        elif mode == "concat" and (num_layers_hint and hidden_size and output_size):
            self.concat_proj = nn.Linear(num_layers_hint * hidden_size, output_size)
            xavier_zero_bias(self.concat_proj)
        elif mode == "attn":
            self.attn_score = nn.LazyLinear(1)
            self.out_proj = (
                nn.Identity() if output_size is None else nn.LazyLinear(output_size)
            )

    def _lazy_init_attn_score(self, in_dim: int, like: torch.Tensor) -> None:
        if self.attn_score is None:
            self.attn_score = nn.Linear(in_dim, 1).to(like.device, like.dtype)
            xavier_zero_bias(self.attn_score)

    def _lazy_init_output_proj(self, in_dim: int, like: torch.Tensor) -> None:
        if self.output is not None and isinstance(self.out_proj, nn.Identity):
            self.out_proj = nn.Linear(in_dim, self.output).to(like.device, like.dtype)
            xavier_zero_bias(self.out_proj)

    def _lazy_init_concat(self, concat_dim: int, like: torch.Tensor) -> None:
        if self.concat_proj is None:
            if self.output is None:
                raise ValueError("output_size must be set for concat JK")
            self.concat_proj = nn.Linear(concat_dim, self.output).to(
                like.device, like.dtype
            )
            xavier_zero_bias(self.concat_proj)

    def _lazy_init_input_proj(self, in_dim: int, like: torch.Tensor) -> None:
        if self.input_proj is None:
            if self.hidden is None:
                raise ValueError("hidden_size must be set for LSTM JK")
            if in_dim != self.hidden:
                self.input_proj = nn.Linear(in_dim, self.hidden).to(
                    like.device, like.dtype
                )
                xavier_zero_bias(self.input_proj)
            else:
                self.input_proj = nn.Identity()

    def forward(self, xs: list[Tensor]) -> Tensor:
        if not xs:
            raise ValueError("JK received empty list")
        base = xs[0].shape[:-1]
        for idx, t in enumerate(xs[1:], 1):
            if t.shape[:-1] != base:
                raise ValueError(
                    f"JK shape mismatch at {idx}: {t.shape} vs {xs[0].shape}"
                )

        if self.mode in ("none", "last"):
            return xs[-1]
        if self.mode == "sum":
            y = xs[0].clone()
            for t in xs[1:]:
                y.add_(t)
            return y
        if self.mode == "max":
            y = xs[0]
            for t in xs[1:]:
                y = torch.maximum(y, t)
            return y
        if self.mode == "concat":
            y = torch.cat(xs, dim=-1)
            self._lazy_init_concat(y.size(-1), y)
            assert self.concat_proj is not None
            return self.concat_proj(y)
        if self.mode == "lstm":
            B, T, N, D = xs[0].shape
            L = len(xs)
            seq = torch.stack(xs, dim=2).reshape(B * T * N, L, D)
            self._lazy_init_input_proj(D, xs[0])
            assert self.input_proj is not None
            seq = self.input_proj(seq)
            out, _ = self.lstm(seq)
            y = out[:, -1, :].reshape(B, T, N, self.hidden)
            return self.out_proj(y)
        if self.mode == "attn":
            y = torch.stack(xs, dim=2)
            B, T, L, N, D = y.shape
            self._lazy_init_attn_score(D, y)
            assert self.attn_score is not None
            scores = self.attn_score(y).squeeze(-1)
            weights = torch.softmax(scores, dim=2)
            y = (y * weights.unsqueeze(-1)).sum(dim=2)
            self._lazy_init_output_proj(D, y)
            return self.out_proj(y)
        raise ValueError(f"Unknown JK mode: {self.mode}")
