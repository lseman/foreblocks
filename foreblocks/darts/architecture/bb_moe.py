"""DARTS-local feed-forward / MoE blocks.

This module intentionally keeps the DARTS transformer path independent from the
generic ``foreblocks.tf`` MoE stack. The implementation here is small and
predictable: top-k token routing over a bank of SwiGLU experts, with optional
shared experts that are always evaluated.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_primitives import GeGLUFFN, ReluFFN, SwiGLUFFN

__all__ = ["DARTSFeedForward", "DARTSMoEFeedForward"]


class _SwiGLUExpert(nn.Module):
    """Single routed expert used by the simplified DARTS MoE block."""

    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = SwiGLUFFN(d_model, expand=expand)
        self.dropout = nn.Dropout(float(max(0.0, dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.ffn(x))


class DARTSMoEFeedForward(nn.Module):
    """Small top-k MoE FFN for DARTS.

    Design:
    - token-choice routing with a learned linear router
    - top-k softmax weights over routed experts
    - optional shared experts that are always active and averaged in
    - no auxiliary router loss, capacity scheduling, or distributed dispatch
    """

    def __init__(
        self,
        d_model: int,
        expand: int = 4,
        dropout: float = 0.0,
        num_experts: int = 8,
        top_k: int = 2,
        num_shared: int = 1,
        router_noise_std: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.expand = int(max(1, expand))
        self.dropout_p = float(max(0.0, dropout))
        self.num_experts = int(max(2, num_experts))
        self.num_shared = int(max(0, min(num_shared, self.num_experts - 1)))
        self.num_routed = int(max(1, self.num_experts - self.num_shared))
        self.top_k = int(max(1, min(top_k, self.num_routed)))
        self.router_noise_std = float(max(0.0, router_noise_std))

        self.router = nn.Linear(self.d_model, self.num_routed, bias=False)
        self.routed_experts = nn.ModuleList(
            [
                _SwiGLUExpert(
                    d_model=self.d_model,
                    expand=self.expand,
                    dropout=self.dropout_p,
                )
                for _ in range(self.num_routed)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                _SwiGLUExpert(
                    d_model=self.d_model,
                    expand=self.expand,
                    dropout=self.dropout_p,
                )
                for _ in range(self.num_shared)
            ]
        )
        self.out_dropout = nn.Dropout(self.dropout_p)
        self.last_routing_stats: dict[str, torch.Tensor | None] = {}

    def _route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x_flat)
        if self.training and self.router_noise_std > 0:
            logits = logits + self.router_noise_std * torch.randn_like(logits)
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_vals, dim=-1)
        return top_idx, top_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        top_idx, top_weights = self._route(x_flat)

        # Only run the selected top-k experts per token — O(top_k) not O(num_experts).
        N = x_flat.size(0)
        mixed = x_flat.new_zeros(N, self.d_model)
        for k in range(self.top_k):
            expert_ids = top_idx[:, k]  # [N]
            weight_k = top_weights[:, k]  # [N]
            for eid in range(self.num_routed):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                out_e = self.routed_experts[eid](x_flat[mask])
                mixed[mask] = mixed[mask] + weight_k[mask].unsqueeze(-1) * out_e

        if self.shared_experts:
            shared = torch.stack([expert(x_flat) for expert in self.shared_experts], dim=0)
            mixed = mixed + shared.mean(dim=0)

        mixed = self.out_dropout(mixed)
        out = mixed.reshape(*orig_shape)

        with torch.no_grad():
            usage = torch.zeros(self.num_routed, device=x.device, dtype=x.dtype)
            ones = torch.ones_like(top_idx, dtype=x.dtype)
            usage.scatter_add_(0, top_idx.reshape(-1), ones.reshape(-1))
            usage = usage / max(1, top_idx.numel())
            self.last_routing_stats = {
                "top_idx": top_idx.detach(),
                "top_weights": top_weights.detach(),
                "usage": usage.detach(),
            }

        return out

    def get_balance_loss(self) -> torch.Tensor:
        usage = self.last_routing_stats.get("usage")
        if usage is None:
            return self.router.weight.new_zeros(())
        target = torch.full_like(usage, 1.0 / max(1, self.num_routed))
        return F.mse_loss(usage, target)

    def get_router_stats(self) -> dict[str, torch.Tensor | None]:
        return dict(self.last_routing_stats)


class DARTSFeedForward(nn.Module):
    """DARTS-specific FFN wrapper with searchable dense/MoE mode.

    Searchable modes: ``swiglu`` (SwiGLU), ``geglu`` (GeGLU),
    ``relu`` (ReLU²), ``moe`` (SwiGLU mixture-of-experts).
    """

    MODE_NAMES = ("swiglu", "geglu", "relu", "moe")

    def __init__(
        self,
        d_model: int,
        expand: int = 4,
        dropout: float = 0.0,
        use_moe: bool = False,
        ffn_mode: str | None = None,
        temperature: float = 1.0,
        single_path_search: bool = True,
        num_experts: int = 8,
        top_k: int = 2,
        num_shared: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.expand = int(max(1, expand))
        self.dim_ff = int(self.d_model * self.expand)
        resolved_ffn_mode = (
            str(ffn_mode).lower() if ffn_mode is not None else ("moe" if use_moe else "swiglu")
        )
        if resolved_ffn_mode not in {*self.MODE_NAMES, "auto"}:
            resolved_ffn_mode = "swiglu"
        self.ffn_mode = resolved_ffn_mode
        self.temperature = max(float(temperature), 1e-3)
        self.single_path_search = bool(single_path_search)
        self.searchable = self.ffn_mode == "auto"
        self.supports_moe = self.ffn_mode in {"auto", "moe"}
        self.use_moe = self.ffn_mode == "moe"

        self.swiglu_block = SwiGLUFFN(self.d_model, expand=self.expand)
        self.geglu_block = GeGLUFFN(self.d_model, expand=self.expand)
        self.relu_block = ReluFFN(self.d_model, expand=self.expand)
        self.moe_block = DARTSMoEFeedForward(
            d_model=self.d_model,
            expand=self.expand,
            dropout=dropout,
            num_experts=int(max(2, num_experts)),
            top_k=int(max(1, top_k)),
            num_shared=int(max(0, num_shared)),
            **kwargs,
        )
        if self.searchable:
            self.register_parameter(
                "ffn_alphas", nn.Parameter(0.01 * torch.randn(len(self.MODE_NAMES)))
            )

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-3)

    def _get_mode_weights(self) -> torch.Tensor:
        if not self.searchable:
            ref = next(self.parameters())
            weights = ref.new_zeros(len(self.MODE_NAMES))
            mode = self.ffn_mode if self.ffn_mode in self.MODE_NAMES else "swiglu"
            weights[self.MODE_NAMES.index(mode)] = 1.0
            return weights

        tau = max(float(self.temperature), 1e-3)
        if self.training:
            return F.gumbel_softmax(
                self.ffn_alphas, tau=tau, hard=bool(self.single_path_search), dim=0
            )
        return F.softmax(self.ffn_alphas / tau, dim=0)

    def get_ffn_mode_probs(self) -> torch.Tensor:
        if self.searchable:
            return F.softmax(self.ffn_alphas.detach().float(), dim=0)
        weights = self._get_mode_weights().detach().float()
        return weights / weights.sum().clamp_min(1e-8)

    def resolve_ffn_mode(self) -> str:
        probs = self.get_ffn_mode_probs()
        return str(self.MODE_NAMES[int(torch.argmax(probs).item())])

    def freeze_ffn_mode(self, ffn_mode: str) -> None:
        resolved = str(ffn_mode).lower()
        if resolved not in self.MODE_NAMES:
            resolved = "swiglu"
        self.ffn_mode = resolved
        self.searchable = False
        self.use_moe = resolved == "moe"
        self.supports_moe = resolved == "moe"
        if hasattr(self, "ffn_alphas"):
            self._parameters.pop("ffn_alphas", None)
            try:
                delattr(self, "ffn_alphas")
            except AttributeError:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self._get_mode_weights()
        # Fast paths for fixed (non-searchable) modes.
        if not self.searchable:
            if self.ffn_mode == "swiglu":
                return self.swiglu_block(x)
            if self.ffn_mode == "geglu":
                return self.geglu_block(x)
            if self.ffn_mode == "relu":
                return self.relu_block(x)
            return self.moe_block(x)
        # Weighted mix — MODE_NAMES = ("swiglu", "geglu", "relu", "moe")
        return (
            weights[0] * self.swiglu_block(x)
            + weights[1] * self.geglu_block(x)
            + weights[2] * self.relu_block(x)
            + weights[3] * self.moe_block(x)
        )

    def get_balance_loss(self) -> torch.Tensor:
        if self.ffn_mode == "moe":
            return self.moe_block.get_balance_loss()
        if self.searchable:
            moe_prob = self.get_ffn_mode_probs()[self.MODE_NAMES.index("moe")]
            return moe_prob * self.moe_block.get_balance_loss()
        ref = next(self.parameters(), None)
        if ref is None:
            return torch.tensor(0.0)
        return ref.new_zeros(())

    def get_router_stats(self) -> dict[str, torch.Tensor | None]:
        stats: dict[str, torch.Tensor | None] = {}
        if self.supports_moe:
            stats.update(self.moe_block.get_router_stats())
        mode_probs = self.get_ffn_mode_probs()
        stats["ffn_mode_probs"] = mode_probs.detach()
        return stats
