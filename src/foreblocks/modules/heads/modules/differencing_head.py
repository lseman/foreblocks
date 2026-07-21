"""foreblocks.modules.heads.modules.differencing_head.

First-order time differencing with reversible inversion.

Computes delta(x) = x[t] - x[t-1] with length preservation (delta[:,0] = 0)
and stores the initial value for reconstruction. Use as a reversible
preprocessing head for non-stationary series — apply differencing before the
forecast and invert predictions to recover the original scale.

Core API:
- Differencing: differencing with invert()
- DifferencingHead: BaseHead wrapper

"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.core.model import BaseHead


class Differencing(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta = x.clone()
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        delta[:, :1, :] = 0.0
        ctx = {"x0": x[:, :1, :]}
        return delta, ctx

    def invert(self, y_hat: torch.Tensor, ctx: dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = ctx["x0"]
        rec = torch.cumsum(y_hat, dim=1)
        rec[:, :1, :] = 0.0
        return rec + x0


class DifferencingHead(BaseHead):
    def __init__(self):
        super().__init__(module=Differencing(), name="diff")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Learnable Fourier Seasonal
# ──────────────────────────────────────────────────────────────────────────────
