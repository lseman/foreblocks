"""Architecture-objective composition for DARTS bilevel training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darts_engine import compute_backward_loss, model_permutation_consistency_loss
from .edge_regularization import _add_edge_diversity_reg, _add_edge_sharpening


@dataclass(frozen=True)
class ArchitectureLossResult:
    """Loss and lightweight diagnostics produced by an architecture step."""

    loss: torch.Tensor
    edge_entropy: torch.Tensor
    edge_sharpen_weight: float
    edge_diversity_pairs: int


def compose_architecture_loss(
    *,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_kwargs: dict[str, Any],
    loss_fn: nn.Module,
    regularizer: Any,
    alpha_tracker: Any,
    arch_params: Sequence[torch.Tensor],
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    device: str,
    engine_variant: str | None,
    engine_cfg: Any,
    state_mix_ortho_reg_weight: float,
    beta_darts_weight: float,
    edge_diversity_weight: float,
    edge_usage_balance_weight: float,
    edge_identity_cap: float,
    edge_identity_cap_weight: float,
    moe_balance_weight: float,
    transformer_exploration_weight: float,
    edge_sharpening_max_weight: float,
    edge_sharpening_start_frac: float,
) -> ArchitectureLossResult:
    """Build the differentiable architecture objective and its diagnostics."""
    if engine_variant == "bi_darts":
        arch_loss = compute_backward_loss(
            model=model,
            x=x,
            y=y,
            loss_fn=loss_fn,
            backward_loss_weight=engine_cfg.bi_darts.backward_loss_weight,
            backward_passes=engine_cfg.bi_darts.backward_passes,
            model_kwargs=model_kwargs,
        )
    else:
        arch_loss = loss_fn(model(x, **model_kwargs), y)

    reg_losses = regularizer.compute_regularization(model, arch_params, epoch, epochs)
    total_loss = arch_loss + reg_losses["total"]

    if (
        engine_variant == "pc_darts"
        and engine_cfg.pc_darts.enable_permutation_consistency
    ):
        total_loss = total_loss + model_permutation_consistency_loss(
            model, engine_cfg.pc_darts.perm_l2_weight
        )

    if state_mix_ortho_reg_weight > 0.0 and hasattr(
        model, "get_orthogonal_regularization"
    ):
        total_loss = total_loss + float(
            state_mix_ortho_reg_weight
        ) * model.get_orthogonal_regularization()

    if beta_darts_weight > 0.0 and arch_params:
        total_loss = total_loss + beta_darts_weight * sum(
            parameter.pow(2).mean() for parameter in arch_params
        )

    total_loss, diversity_pairs = _add_edge_diversity_reg(
        model=model,
        total_arch_loss=total_loss,
        edge_diversity_weight=edge_diversity_weight,
        edge_usage_balance_weight=edge_usage_balance_weight,
        edge_identity_cap=edge_identity_cap,
        edge_identity_cap_weight=edge_identity_cap_weight,
        device=device,
    )

    if moe_balance_weight > 0.0 and hasattr(model, "get_moe_balance_loss"):
        total_loss = total_loss + float(
            moe_balance_weight
        ) * model.get_moe_balance_loss()

    if transformer_exploration_weight > 0.0:
        entropy_terms: list[torch.Tensor] = []
        for source in alpha_tracker.component_alpha_sources(model):
            name = str(source.get("name", ""))
            if not (
                name.startswith("encoder_")
                or name.startswith("decoder_")
                or name in {"decoder_style", "decoder_memory_queries"}
            ):
                continue
            alpha = source.get("alpha")
            if not isinstance(alpha, torch.Tensor) or alpha.numel() <= 1:
                continue
            probs = F.softmax(alpha, dim=0)
            entropy_terms.append(-(probs * torch.log(probs.clamp_min(1e-8))).sum())

        if entropy_terms:
            exploration_entropy = torch.stack(entropy_terms).mean()
            if epochs > warmup_epochs:
                progress = float(max(epoch - warmup_epochs, 0)) / float(
                    max(epochs - warmup_epochs, 1)
                )
            else:
                progress = float(epoch) / float(max(epochs, 1))
            total_loss = total_loss - (
                float(transformer_exploration_weight)
                * max(0.0, 1.0 - progress)
                * exploration_entropy
            )

    total_loss, edge_entropy, sharpen_weight = _add_edge_sharpening(
        model=model,
        total_arch_loss=total_loss,
        epoch=epoch,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        edge_sharpening_max_weight=edge_sharpening_max_weight,
        edge_sharpening_start_frac=edge_sharpening_start_frac,
        device=device,
    )
    return ArchitectureLossResult(
        loss=total_loss,
        edge_entropy=edge_entropy,
        edge_sharpen_weight=float(sharpen_weight),
        edge_diversity_pairs=diversity_pairs,
    )
