from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


@dataclass(frozen=True)
class RouterConfig:
    hidden: int | None = None
    temperature: float = 1.0
    top_k: int | None = 2


def resolve_router_config(
    base: RouterConfig | None = None,
    *,
    hidden: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
) -> RouterConfig:
    cfg = base or RouterConfig()
    return RouterConfig(
        hidden=cfg.hidden if hidden is None else hidden,
        temperature=cfg.temperature if temperature is None else temperature,
        top_k=cfg.top_k if top_k is None else top_k,
    )


class TokenRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        *,
        router_config: RouterConfig | None = None,
        hidden: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ):
        super().__init__()
        cfg = resolve_router_config(
            router_config,
            hidden=hidden,
            temperature=temperature,
            top_k=top_k,
        )
        self.num_experts = num_experts
        self.temperature = float(cfg.temperature)
        self.top_k = cfg.top_k

        router_hidden = cfg.hidden or max(32, d_model // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )

    @torch.no_grad()
    def _topk_mask(self, logits: Tensor, k: int) -> Tensor:
        idx = torch.topk(logits, k=k, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return mask

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.net(x) / max(self.temperature, 1e-6)
        if self.top_k is not None and self.top_k < self.num_experts:
            logits = logits.masked_fill(
                ~self._topk_mask(logits, self.top_k), float("-inf")
            )
        probs = F.softmax(logits, dim=-1)
        return probs, logits


__all__ = ["RouterConfig", "TokenRouter", "resolve_router_config"]
