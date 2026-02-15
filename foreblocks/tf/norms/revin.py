from typing import Optional

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., NeurIPS 2021)."""

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "norm",
        stats: Optional[dict] = None,
    ):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x_norm = (x - self.mean) / self.std
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm

        if mode == "denorm":
            assert hasattr(self, "mean") and hasattr(
                self, "std"
            ), "Must call norm() before denorm()"
            mean = stats.get("mean", self.mean) if stats else self.mean
            std = stats.get("std", self.std) if stats else self.std
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            return x * std + mean

        raise ValueError(f"Unknown mode: {mode}")

    def reset_stats(self):
        if hasattr(self, "mean"):
            del self.mean
        if hasattr(self, "std"):
            del self.std


__all__ = ["RevIN"]
