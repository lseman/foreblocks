"""Small tensor utilities shared across DARTS subsystems."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_one_hot(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return an argmax one-hot tensor without a device-to-host synchronization."""
    if probs.numel() == 0:
        return torch.zeros_like(probs)
    normalized_dim = dim % probs.ndim
    index = probs.argmax(dim=normalized_dim)
    encoded = F.one_hot(index, num_classes=probs.shape[normalized_dim])
    if normalized_dim != probs.ndim - 1:
        encoded = encoded.movedim(-1, normalized_dim)
    return encoded.to(
        device=probs.device, dtype=probs.dtype
    )


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
