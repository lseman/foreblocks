from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_types import ActiveHead, AlphaWeights, HeadSpec


class HeadStateManager(nn.Module):
    """
    Centralize mutable per-head state:
      - NAS alpha parameters
      - enable/disable masks
      - hardened architecture decisions
    """

    _EPS: float = 1e-6

    def __init__(
        self,
        active_heads: list[ActiveHead],
        *,
        enable_nas: bool,
        alpha_temperature: float,
        gumbel_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self._active_heads: list[ActiveHead] = list(active_heads)
        self._head_by_name: dict[str, ActiveHead] = {
            h.name: h for h in self._active_heads
        }
        self.enable_nas = bool(enable_nas)
        self.alpha_temperature = float(alpha_temperature)
        self.gumbel_temperature = (
            float(gumbel_temperature)
            if gumbel_temperature is not None
            else float(alpha_temperature)
        )

        self._alphas = nn.ParameterDict()
        if self.enable_nas:
            self._init_alpha_params(self._active_heads)

    @property
    def alphas(self) -> nn.ParameterDict:
        return self._alphas

    def _init_alpha_params(self, active_heads: list[ActiveHead]) -> None:
        for head in active_heads:
            spec = head.spec
            if spec.alpha_mode == "off":
                continue
            if spec.alpha_mode in ("soft", "gumbel"):
                p = nn.Parameter(
                    torch.tensor([spec.alpha_init, 0.0], dtype=torch.float32)
                )
            elif spec.alpha_mode == "gate":
                p = nn.Parameter(torch.tensor([spec.alpha_init], dtype=torch.float32))
            else:
                raise ValueError(
                    f"alpha_mode must be one of ('off','gate','soft','gumbel'), got '{spec.alpha_mode}'"
                )
            p.requires_grad_(bool(spec.alpha_trainable))
            self._alphas[spec.name] = p
            head.alpha_param = p

    def alpha_weights_for_head(self, head: ActiveHead) -> AlphaWeights:
        return self._alpha_weights_for_head(
            head, sample=head.spec.alpha_mode == "gumbel"
        )

    def deterministic_alpha_weights_for_head(self, head: ActiveHead) -> AlphaWeights:
        return self._alpha_weights_for_head(head, sample=False)

    def _alpha_weights_for_head(
        self,
        head: ActiveHead,
        *,
        sample: bool,
    ) -> AlphaWeights:
        spec = head.spec
        if not self.enable_nas or spec.alpha_mode == "off":
            return None, None

        a = (
            head.alpha_param
            if head.alpha_param is not None
            else self._alphas[spec.name]
        )
        if spec.alpha_mode == "gumbel":
            tau = max(self._EPS, float(self.gumbel_temperature))
        else:
            tau = max(self._EPS, float(self.alpha_temperature))

        if spec.alpha_mode == "soft":
            w = F.softmax(a / tau, dim=0)
            return w[0], w[1]

        if spec.alpha_mode == "gumbel":
            logits = a / tau
            if sample:
                u = torch.rand_like(a).clamp_(1e-8, 1.0 - 1e-8)
                gumbel_noise = -torch.log(-torch.log(u))
                logits = logits + gumbel_noise / tau
            w = F.softmax(logits, dim=0)
            return w[0], w[1]

        g = torch.sigmoid(a[0] / tau)
        return g, 1.0 - g

    @staticmethod
    def straight_through_hard_gate(g: torch.Tensor) -> torch.Tensor:
        g_hard = (g > 0.5).float()
        return g_hard.detach() - g.detach() + g

    def compute_effective_weight(
        self,
        spec: HeadSpec,
        w_head: torch.Tensor | None,
        w_skip: torch.Tensor | None,
        hardened: bool | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hardened is not None:
            h = float(hardened)
            return (
                torch.tensor(h, device=device, dtype=dtype),
                torch.tensor(1.0 - h, device=device, dtype=dtype),
            )

        if w_head is None:
            return (
                torch.tensor(1.0, device=device, dtype=dtype),
                torch.tensor(0.0, device=device, dtype=dtype),
            )

        if spec.alpha_mode == "soft":
            if w_skip is None:
                w_skip = 1.0 - w_head
            return w_head, w_skip

        g_used = self.straight_through_hard_gate(w_head)
        return g_used, 1.0 - g_used

    def check_head_exists(self, name: str) -> None:
        if name not in self._head_by_name:
            available = list(self._head_by_name.keys())
            raise KeyError(f"Unknown head name: '{name}'. Available: {available}")

    def get_head(self, name: str) -> ActiveHead:
        self.check_head_exists(name)
        return self._head_by_name[name]

    @torch.no_grad()
    def enable_head(self, name: str) -> None:
        head = self.get_head(name)
        head.enabled = True

    @torch.no_grad()
    def disable_head(self, name: str) -> None:
        head = self.get_head(name)
        head.enabled = False

    @torch.no_grad()
    def set_head_enabled(self, name: str, enabled: bool) -> None:
        head = self.get_head(name)
        head.enabled = bool(enabled)

    @torch.no_grad()
    def enable_all(self) -> None:
        for head in self._active_heads:
            head.enabled = True

    @torch.no_grad()
    def disable_all(self) -> None:
        for head in self._active_heads:
            head.enabled = False

    def enabled_report(self) -> dict[str, bool]:
        return {h.name: h.enabled for h in self._active_heads}

    @contextmanager
    def temporary_disable(self, *names: str) -> Iterator[None]:
        previous_states: dict[str, bool] = {}
        for name in names:
            head = self.get_head(name)
            previous_states[name] = head.enabled
            head.enabled = False
        try:
            yield
        finally:
            for name, prev in previous_states.items():
                self._head_by_name[name].enabled = prev

    @contextmanager
    def only_heads(self, *names: str) -> Iterator[None]:
        for name in names:
            self.check_head_exists(name)
        previous_states = {h.name: h.enabled for h in self._active_heads}
        self.disable_all()
        for name in names:
            self._head_by_name[name].enabled = True
        try:
            yield
        finally:
            for name, prev in previous_states.items():
                self._head_by_name[name].enabled = prev

    def arch_parameters(self) -> Iterator[nn.Parameter]:
        if not self.enable_nas:
            return iter([])
        return (p for p in self._alphas.values() if p.requires_grad)

    def alpha_report(self) -> dict[str, dict[str, Any]]:
        if not self.enable_nas:
            return {}

        rep: dict[str, dict[str, Any]] = {}
        for head in self._active_heads:
            spec = head.spec
            d: dict[str, Any] = {
                "mode": spec.alpha_mode,
                "mix_style": spec.alpha_mix_style,
                "enabled": head.enabled,
                "hardened": head.hardened,
            }
            if spec.alpha_mode == "off" or spec.name not in self._alphas:
                rep[spec.name] = d
                continue

            w_head, w_skip = self.deterministic_alpha_weights_for_head(head)
            if spec.alpha_mode in ("soft", "gumbel"):
                d["w_head"] = float(w_head.item())
                d["w_skip"] = float(w_skip.item())
                d["p_on"] = float(w_head.item())
            else:
                d["p_on"] = float(w_head.item())
            rep[spec.name] = d
        return rep

    @torch.no_grad()
    def discretize_(self, threshold: float = 0.5) -> dict[str, bool]:
        if not self.enable_nas:
            return {}

        decisions: dict[str, bool] = {}
        for head in self._active_heads:
            spec = head.spec
            if spec.alpha_mode == "off" or spec.name not in self._alphas:
                head.hardened = None
                continue
            w_head, _ = self.deterministic_alpha_weights_for_head(head)
            decision = bool(w_head >= threshold)
            head.hardened = decision
            decisions[spec.name] = decision
        return decisions

    @torch.no_grad()
    def clear_discretization_(self) -> None:
        for head in self._active_heads:
            head.hardened = None
