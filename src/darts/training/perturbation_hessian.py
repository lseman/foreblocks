"""DARTS-PT perturbation and finite-difference Hessian utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.training import autocast_ctx


def _apply_darts_pt_perturbation(
    *,
    model: nn.Module,
    model_params: list[torch.Tensor],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_model_kwargs: dict[str, Any] | None,
    loss_fn,
    xi: float,
    device: str,
    use_amp: bool,
) -> list[torch.Tensor]:
    """Perturb model weights by ``-xi * unit(∇_w L_train)`` (DARTS-PT step).

    Returns the list of original parameter tensors so the caller can restore
    them via :func:`_restore_model_params` after the architecture backward pass.

    Moving weights toward the training loss minimum before evaluating the
    validation loss reduces the coupling bias that arises when shared weights
    are sub-optimally positioned for the current architecture distribution.
    """
    train_model_kwargs = dict(train_model_kwargs or {})

    with torch.no_grad():
        originals = [p.detach().clone() for p in model_params]

    with autocast_ctx(device, enabled=use_amp):
        train_loss = loss_fn(model(train_x, **train_model_kwargs), train_y)

    grads = torch.autograd.grad(
        train_loss,
        model_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    norm_sq = torch.tensor(0.0, device=device)
    for g in grads:
        if g is not None:
            norm_sq = norm_sq + g.pow(2).sum()
    norm = norm_sq.sqrt().clamp_min(1e-12)

    with torch.no_grad():
        for p, g in zip(model_params, grads):
            if g is not None:
                p.add_(g.detach() / norm, alpha=-float(xi))

    return originals


def _restore_model_params(
    model_params: list[torch.Tensor],
    originals: list[torch.Tensor],
) -> None:
    """Restore model parameters to their pre-perturbation values."""
    with torch.no_grad():
        for p, orig in zip(model_params, originals):
            p.copy_(orig)


def compute_implicit_arch_gradient_correction(
    *,
    model: nn.Module,
    loss_fn,
    arch_x: torch.Tensor,
    arch_y: torch.Tensor,
    arch_model_kwargs: dict[str, Any] | None,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_model_kwargs: dict[str, Any] | None,
    model_params: list[torch.Tensor],
    arch_params: list[torch.Tensor],
    xi: float,
    eps: float,
    device: str,
    use_amp: bool,
) -> list[torch.Tensor | None]:
    """Compute second-order implicit arch gradient correction.

    Implements the DARTS second-order approximation via finite differences in
    model-weight space.  Instead of adding a scalar curvature *penalty* to the
    arch loss, this function returns per-parameter gradient *corrections* that
    are applied directly to ``p.grad`` after the main backward pass
    (see ``BilevelOptimizer.step_architecture``).

    Correction:
        Δ∇_α = -xi / (2ε) * (∇_α L_val(w⁺) − ∇_α L_val(w⁻))

    where ``w± = w ± ε · unit(∇_w L_train)``.

    This removes the need for a tuned penalty weight (beyond ``xi``, which is
    already the model learning-rate scale) and avoids the instability of
    relu-clipped curvature scalars in noisy training regimes.
    """
    if not model_params or not arch_params or eps <= 0:
        return [None] * len(arch_params)

    arch_model_kwargs = dict(arch_model_kwargs or {})
    train_model_kwargs = dict(train_model_kwargs or {})

    # --- Training gradient direction ---
    with autocast_ctx(device, enabled=use_amp):
        train_loss = loss_fn(model(train_x, **train_model_kwargs), train_y)

    grads_w = torch.autograd.grad(
        train_loss,
        model_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    direction = [
        torch.zeros_like(p) if g is None else g.detach()
        for p, g in zip(model_params, grads_w)
    ]
    norm = torch.sqrt(sum(d.pow(2).sum() for d in direction)).clamp_min(1e-12)
    # step size in the unit-gradient direction
    scale = float(eps) / float(norm)

    with torch.no_grad():
        originals = [p.detach().clone() for p in model_params]

    corrections: list[torch.Tensor | None] = [None] * len(arch_params)
    try:
        # ∇_α L_val(w⁺)
        with torch.no_grad():
            for p, d in zip(model_params, direction):
                p.add_(d, alpha=scale)
        with autocast_ctx(device, enabled=use_amp):
            loss_plus = loss_fn(model(arch_x, **arch_model_kwargs), arch_y)
        grads_plus = torch.autograd.grad(
            loss_plus,
            arch_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # ∇_α L_val(w⁻)
        with torch.no_grad():
            for p, d in zip(model_params, direction):
                p.add_(d, alpha=-2.0 * scale)
        with autocast_ctx(device, enabled=use_amp):
            loss_minus = loss_fn(model(arch_x, **arch_model_kwargs), arch_y)
        grads_minus = torch.autograd.grad(
            loss_minus,
            arch_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Δ∇_α = -xi / (2ε) * (∇_α(w⁺) − ∇_α(w⁻))
        # The 2ε denominator corresponds to the actual step size (eps in unit-grad space).
        factor = -float(xi) / (2.0 * float(eps))
        for i, (gp, gm) in enumerate(zip(grads_plus, grads_minus)):
            if gp is not None and gm is not None:
                corrections[i] = (factor * (gp - gm)).detach()
            elif gp is not None:
                corrections[i] = (factor * gp).detach()
    finally:
        with torch.no_grad():
            for p, orig in zip(model_params, originals):
                p.copy_(orig)

    return corrections


def finite_difference_hessian_penalty(
    *,
    model: nn.Module,
    loss_fn,
    arch_loss: torch.Tensor,
    arch_x: torch.Tensor,
    arch_y: torch.Tensor,
    arch_model_kwargs: dict[str, Any] | None,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_model_kwargs: dict[str, Any] | None,
    model_params: list[torch.Tensor],
    device: str,
    eps: float = 1e-2,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Finite-difference curvature proxy used to penalise sharp architecture
    landscapes (from DARTS+ / SDARTS literature).
    """
    if eps <= 0 or not model_params:
        return torch.tensor(0.0, device=arch_x.device)

    arch_model_kwargs = dict(arch_model_kwargs or {})
    train_model_kwargs = dict(train_model_kwargs or {})

    with autocast_ctx(device, enabled=use_amp):
        train_loss = loss_fn(model(train_x, **train_model_kwargs), train_y)

    grads = torch.autograd.grad(
        train_loss,
        model_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    direction, norm_sq = [], torch.tensor(0.0, device=arch_x.device)
    for p, g in zip(model_params, grads):
        d = torch.zeros_like(p) if g is None else g.detach()
        direction.append(d)
        norm_sq = norm_sq + d.pow(2).sum()

    norm = torch.sqrt(norm_sq).clamp_min(1e-12)
    scale = float(eps) / norm

    with torch.no_grad():
        originals = [p.detach().clone() for p in model_params]

    try:
        with torch.no_grad():
            for p, d in zip(model_params, direction):
                p.add_(scale * d)
        with autocast_ctx(device, enabled=use_amp):
            loss_plus = loss_fn(model(arch_x, **arch_model_kwargs), arch_y)

        with torch.no_grad():
            for p, d in zip(model_params, direction):
                p.add_(-2.0 * scale * d)
        with autocast_ctx(device, enabled=use_amp):
            loss_minus = loss_fn(model(arch_x, **arch_model_kwargs), arch_y)
    finally:
        with torch.no_grad():
            for p, orig in zip(model_params, originals):
                p.copy_(orig)

    curvature = (loss_plus + loss_minus - 2.0 * arch_loss.detach()) / (eps**2)
    return F.relu(curvature)

