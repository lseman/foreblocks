"""Base wrapper for composable forecasting heads and auxiliary losses."""

from typing import Any

import torch
from torch import nn


class BaseHead(nn.Module):
    def __init__(self, module: nn.Module, name: str | None = None) -> None:
        super().__init__()
        self.module = module
        self.name: str = name or self.__class__.__name__
        self._fallback_device: torch.device | None = None

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # ---- aux loss handling ---------------------------------------------------
    def _infer_device(self) -> torch.device:
        for p in self.module.parameters():
            self._fallback_device = p.device
            return p.device
        for b in self.module.buffers():
            self._fallback_device = b.device
            return b.device

        self._fallback_device = self._fallback_device or torch.device("cpu")
        return self._fallback_device

    def get_aux_loss(self) -> torch.Tensor:
        device = self._infer_device()
        if hasattr(self.module, "aux_loss"):
            aux = self.module.aux_loss
            if torch.is_tensor(aux):
                return aux.to(device)
            return torch.as_tensor(aux, device=device)
        return torch.zeros((), device=device)

    # ---- attribute delegation ------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            module = self.__dict__.get("_modules", {}).get("module")
            if module is None:
                raise exc
            return getattr(module, name)

