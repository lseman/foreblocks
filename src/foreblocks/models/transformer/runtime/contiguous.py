"""Preparation and projection helpers for single-pass horizon decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PreparedContiguousForecast:
    """Context, placeholders, and coordinates for one masked horizon pass."""

    values: torch.Tensor
    value_mask: torch.Tensor
    attention_padding: torch.Tensor
    horizon: int
    padded_horizon: int
    num_targets: int
    num_context_patches: int
    num_horizon_patches: int


def _validate_series(
    name: str,
    values: torch.Tensor | None,
    *,
    batch_size: int,
    length: int,
) -> None:
    if values is not None and (
        values.ndim != 3 or values.shape[:2] != (batch_size, length)
    ):
        raise ValueError(
            f"{name} must have shape [B,{length},C], got {tuple(values.shape)}"
        )


def _coerce_mask(
    name: str,
    mask: torch.Tensor | None,
    values: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if values is None:
        if mask is not None:
            raise ValueError(f"{name} was provided without matching values")
        return None
    if mask is None:
        return torch.zeros_like(values, dtype=torch.bool, device=device)
    if mask.shape != values.shape:
        raise ValueError(f"{name} must match its values shape {tuple(values.shape)}")
    return mask.to(device=device, dtype=torch.bool)


def prepare_contiguous_forecast(
    target: torch.Tensor,
    horizon: int,
    *,
    input_size: int,
    patch_length: int,
    past_only_covariates: torch.Tensor | None = None,
    past_future_covariates: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    past_only_mask: torch.Tensor | None = None,
    past_future_mask: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
) -> PreparedContiguousForecast:
    """Append a fully masked target horizon while preserving known covariates."""
    if target.ndim != 3:
        raise ValueError(f"target must be [B,T,C], got {tuple(target.shape)}")
    if not target.is_floating_point():
        raise ValueError("target must be a floating-point tensor")
    batch_size, context_length, num_targets = target.shape
    if num_targets <= 0 or context_length <= 0:
        raise ValueError("target context and channel dimensions must be non-zero")

    if past_future_covariates is not None:
        if past_future_covariates.ndim != 3:
            raise ValueError("past_future_covariates must be [B,T+H,C]")
        inferred_horizon = past_future_covariates.shape[1] - context_length
        if horizon not in (0, inferred_horizon):
            raise ValueError("horizon does not match past_future_covariates length")
        horizon = inferred_horizon
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    _validate_series(
        "past_only_covariates",
        past_only_covariates,
        batch_size=batch_size,
        length=context_length,
    )
    _validate_series(
        "past_future_covariates",
        past_future_covariates,
        batch_size=batch_size,
        length=context_length + horizon,
    )
    if context_mask is not None and context_mask.shape != (
        batch_size,
        context_length,
    ):
        raise ValueError(
            f"context_mask must have shape {(batch_size, context_length)}, "
            f"got {tuple(context_mask.shape)}"
        )

    device, dtype = target.device, target.dtype
    past_only_covariates = (
        None
        if past_only_covariates is None
        else past_only_covariates.to(device=device, dtype=dtype)
    )
    past_future_covariates = (
        None
        if past_future_covariates is None
        else past_future_covariates.to(device=device, dtype=dtype)
    )
    target_mask = _coerce_mask("target_mask", target_mask, target, device=device)
    assert target_mask is not None
    past_only_mask = _coerce_mask(
        "past_only_mask", past_only_mask, past_only_covariates, device=device
    )
    past_future_mask = _coerce_mask(
        "past_future_mask",
        past_future_mask,
        past_future_covariates,
        device=device,
    )
    context_mask = (
        torch.zeros(batch_size, context_length, device=device, dtype=torch.bool)
        if context_mask is None
        else context_mask.to(device=device, dtype=torch.bool)
    )

    context_values = [target]
    context_masks = [target_mask]
    num_past_only = 0
    if past_only_covariates is not None:
        assert past_only_mask is not None
        num_past_only = past_only_covariates.shape[-1]
        context_values.append(past_only_covariates)
        context_masks.append(past_only_mask)
    num_known_future = 0
    if past_future_covariates is not None:
        assert past_future_mask is not None
        num_known_future = past_future_covariates.shape[-1]
        context_values.append(past_future_covariates[:, :context_length])
        context_masks.append(past_future_mask[:, :context_length])

    context_values_tensor = torch.cat(context_values, dim=-1)
    context_value_mask = torch.cat(context_masks, dim=-1)
    context_value_mask = context_value_mask | context_mask.unsqueeze(-1)
    num_variates = num_targets + num_past_only + num_known_future
    if num_variates != input_size:
        raise ValueError(
            f"target and covariates provide {num_variates} variates, "
            f"but input_size={input_size}"
        )

    context_padding = (-context_length) % patch_length
    horizon_padding = (-horizon) % patch_length
    padded_horizon = horizon + horizon_padding
    prefix_values = target.new_zeros(batch_size, context_padding, num_variates)
    prefix_mask = torch.ones(
        batch_size,
        context_padding,
        num_variates,
        device=device,
        dtype=torch.bool,
    )

    future_values = [
        target.new_zeros(batch_size, padded_horizon, num_targets),
        target.new_zeros(batch_size, padded_horizon, num_past_only),
    ]
    future_masks = [
        torch.ones(
            batch_size,
            padded_horizon,
            num_targets,
            device=device,
            dtype=torch.bool,
        ),
        torch.ones(
            batch_size,
            padded_horizon,
            num_past_only,
            device=device,
            dtype=torch.bool,
        ),
    ]
    if past_future_covariates is not None:
        known_future = past_future_covariates[
            :, context_length : context_length + horizon
        ]
        known_mask = past_future_mask[:, context_length : context_length + horizon]
        if horizon_padding:
            known_future = F.pad(known_future, (0, 0, 0, horizon_padding))
            known_mask = F.pad(known_mask, (0, 0, 0, horizon_padding), value=True)
        future_values.append(known_future)
        future_masks.append(known_mask)

    values = torch.cat(
        [prefix_values, context_values_tensor, torch.cat(future_values, dim=-1)],
        dim=1,
    )
    value_mask = torch.cat(
        [prefix_mask, context_value_mask, torch.cat(future_masks, dim=-1)],
        dim=1,
    )
    # Unknown horizon values are mask-token inputs, not padding: they must
    # remain visible to bidirectional sequence and variate attention.
    attention_padding = torch.zeros_like(value_mask)
    if context_padding:
        attention_padding[:, :context_padding] = True

    return PreparedContiguousForecast(
        values=values,
        value_mask=value_mask,
        attention_padding=attention_padding,
        horizon=horizon,
        padded_horizon=padded_horizon,
        num_targets=num_targets,
        num_context_patches=(context_length + context_padding) // patch_length,
        num_horizon_patches=padded_horizon // patch_length,
    )


def project_contiguous_quantiles(
    encoded: torch.Tensor,
    head: nn.Linear,
    prepared: PreparedContiguousForecast,
    *,
    patch_length: int,
    num_quantiles: int,
) -> torch.Tensor:
    """Project horizon mask-token states to ``[B,H,C_target,Q]``."""
    if encoded.ndim != 4:
        raise ValueError("encoded variate states must be [B,V,N,D]")
    horizon_states = encoded[
        :,
        : prepared.num_targets,
        prepared.num_context_patches : prepared.num_context_patches
        + prepared.num_horizon_patches,
    ]
    batch_size = encoded.shape[0]
    logits = head(horizon_states).reshape(
        batch_size,
        prepared.num_targets,
        prepared.num_horizon_patches,
        patch_length,
        num_quantiles,
    )
    forecasts = logits.reshape(
        batch_size,
        prepared.num_targets,
        prepared.padded_horizon,
        num_quantiles,
    )[:, :, : prepared.horizon]
    return forecasts.permute(0, 2, 1, 3).contiguous()


def contiguous_quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: tuple[float, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute pinball loss for ``[B,H,C,Q]`` forecasts."""
    if predictions.ndim != 4 or targets.shape != predictions.shape[:-1]:
        raise ValueError("predictions must be [B,H,C,Q] and targets [B,H,C]")
    levels = predictions.new_tensor(quantiles)
    errors = targets.unsqueeze(-1) - predictions
    losses = torch.maximum(levels * errors, (levels - 1.0) * errors)
    if mask is None:
        return losses.mean()
    if mask.shape != targets.shape:
        raise ValueError("quantile loss mask must match targets")
    valid = (~mask.to(device=losses.device, dtype=torch.bool)).unsqueeze(-1)
    return (losses * valid).sum() / valid.sum().clamp_min(1) / losses.shape[-1]


__all__ = [
    "PreparedContiguousForecast",
    "contiguous_quantile_loss",
    "prepare_contiguous_forecast",
    "project_contiguous_quantiles",
]
