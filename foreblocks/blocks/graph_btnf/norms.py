from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .common import ActivationType, Tensor

NormStrategy = Literal["pre_norm", "post_norm", "sandwich_norm", "none"]


class GraphNorm(nn.Module):
    """
    Normalize across the node dimension for tensors shaped [B, T, N, F].
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.mean_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-2, keepdim=True)
        var = x.var(dim=-2, keepdim=True, unbiased=False)
        x_norm = (x - mean * self.mean_scale) / (var + self.eps).sqrt()
        return x_norm * self.weight + self.bias


def make_activation(name: ActivationType) -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "none": nn.Identity(),
    }[name]


def make_feature_norm(
    dim: int,
    *,
    use_graph_norm: bool,
    eps: float = 1e-5,
) -> nn.Module:
    if use_graph_norm:
        return GraphNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)


def resolve_norm_strategy(
    norm_strategy: str | None = None,
    pre_norm: bool | None = None,
) -> NormStrategy:
    aliases = {
        "pre": "pre_norm",
        "post": "post_norm",
        "both": "sandwich_norm",
    }
    if norm_strategy is None:
        if pre_norm is None:
            return "pre_norm"
        return "pre_norm" if pre_norm else "post_norm"

    resolved = aliases.get(norm_strategy, norm_strategy)
    if resolved not in {"pre_norm", "post_norm", "sandwich_norm", "none"}:
        raise ValueError(
            "norm_strategy must be one of "
            "{'pre_norm','post_norm','sandwich_norm','none'}"
        )

    if pre_norm is not None:
        expected = "pre_norm" if pre_norm else "post_norm"
        if resolved != expected:
            raise ValueError(
                "pre_norm and norm_strategy disagree; pass only one or use matching values"
            )
    return resolved


def make_norm_pair(
    in_dim: int,
    out_dim: int,
    *,
    use_graph_norm: bool,
    norm_strategy: str | None = None,
    pre_norm: bool | None = None,
    eps: float = 1e-5,
) -> tuple[NormStrategy, nn.Module, nn.Module]:
    strategy = resolve_norm_strategy(norm_strategy=norm_strategy, pre_norm=pre_norm)
    pre = (
        make_feature_norm(in_dim, use_graph_norm=use_graph_norm, eps=eps)
        if strategy in ("pre_norm", "sandwich_norm")
        else nn.Identity()
    )
    post = (
        make_feature_norm(out_dim, use_graph_norm=use_graph_norm, eps=eps)
        if strategy in ("post_norm", "sandwich_norm")
        else nn.Identity()
    )
    return strategy, pre, post


__all__ = [
    "GraphNorm",
    "NormStrategy",
    "make_activation",
    "make_feature_norm",
    "make_norm_pair",
    "resolve_norm_strategy",
]
