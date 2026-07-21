"""Small tensor utilities shared across DARTS subsystems."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def as_probability_vector(
    alpha_like: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Interpret normalized inputs as probabilities, otherwise as logits."""
    if alpha_like.numel() == 0:
        return alpha_like

    with torch.no_grad():
        flat = alpha_like.detach().reshape(-1)
        checks = torch.stack(
            (
                torch.isfinite(flat).all(),
                flat.min() >= -1e-6,
                flat.max() <= 1.0 + 1e-6,
                (flat.sum() - 1.0).abs() <= 1e-4,
            )
        )
        looks_like_probs = bool(checks.all().item())

    temperature = max(float(temperature), 1e-6)
    if looks_like_probs:
        probs = alpha_like.clamp_min(1e-8)
        if abs(temperature - 1.0) > 1e-8:
            probs = probs.pow(1.0 / temperature)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return F.softmax(alpha_like / temperature, dim=-1)
