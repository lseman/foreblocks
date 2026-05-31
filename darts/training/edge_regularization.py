"""
DARTS bilevel training loop.

Extracted from ``DARTSTrainer`` so the logic lives in one focused module.
The public entry-point is :func:`train_darts_model`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from ..evaluation.metrics import compute_final_metrics
from ..training.helpers import (
    ArchitectureRegularizer,
    BilevelOptimizer,
    RegularizationType,
    TemperatureScheduler,
)
from ..utils.training import (
    autocast_ctx,
    build_arch_param_groups,
    capture_progressive_state,
    create_progress_bar,
    restore_progressive_state,
    split_arch_and_model_params,
    unpack_forecasting_batch,
)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def _add_edge_diversity_reg(
    *,
    model: nn.Module,
    total_arch_loss: torch.Tensor,
    edge_diversity_weight: float,
    edge_usage_balance_weight: float,
    edge_identity_cap: float,
    edge_identity_cap_weight: float,
    device: str,
) -> tuple:
    """Compute and add edge diversity + identity-cap regularisation terms."""
    edge_diversity_loss = torch.tensor(0.0, device=device)
    edge_diversity_pairs = 0
    edge_usage_balance_loss = torch.tensor(0.0, device=device)
    edge_usage_cells = 0
    edge_identity_cap_loss = torch.tensor(0.0, device=device)
    edge_identity_cells = 0

    for cell in getattr(model, "cells", []):
        if not hasattr(cell, "edges"):
            continue

        edge_probs_by_name = []
        union_op_names: list[str] = []

        for edge in cell.edges:
            probs = _extract_edge_probs(edge)
            if (
                probs is None
                or probs.numel() <= 1
                or not hasattr(edge, "available_ops")
                or len(edge.available_ops) != probs.numel()
            ):
                continue

            probs = probs.clamp_min(1e-8)
            probs = probs / probs.sum().clamp_min(1e-8)
            prob_map = {}
            for op_idx, op_name in enumerate(edge.available_ops):
                prob_map[op_name] = probs[op_idx]
                if op_name not in union_op_names:
                    union_op_names.append(op_name)
            edge_probs_by_name.append(prob_map)

        if len(edge_probs_by_name) < 2 or len(union_op_names) <= 1:
            continue

        base_zero = edge_probs_by_name[0][union_op_names[0]].new_tensor(0.0)
        aligned: list[torch.Tensor] = []
        for prob_map in edge_probs_by_name:
            vec = torch.stack(
                [prob_map.get(n, base_zero) for n in union_op_names], dim=0
            )
            vec = vec / vec.sum().clamp_min(1e-8)
            aligned.append(vec)

        for i in range(len(aligned)):
            vi = aligned[i] / aligned[i].norm(p=2).clamp_min(1e-8)
            for j in range(i + 1, len(aligned)):
                vj = aligned[j] / aligned[j].norm(p=2).clamp_min(1e-8)
                edge_diversity_loss = edge_diversity_loss + torch.dot(vi, vj)
                edge_diversity_pairs += 1

        mean_probs = torch.stack(aligned, dim=0).mean(dim=0)
        mean_probs = mean_probs / mean_probs.sum().clamp_min(1e-8)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
        norm_entropy = entropy / np.log(max(mean_probs.numel(), 2))
        edge_usage_balance_loss = edge_usage_balance_loss + (1.0 - norm_entropy)
        edge_usage_cells += 1

        if "Identity" in union_op_names and edge_identity_cap < 1.0:
            id_idx = union_op_names.index("Identity")
            id_prob = mean_probs[id_idx]
            edge_identity_cap_loss = edge_identity_cap_loss + F.relu(
                id_prob - float(edge_identity_cap)
            )
            edge_identity_cells += 1

    if edge_diversity_pairs > 0:
        total_arch_loss = total_arch_loss + edge_diversity_weight * (
            edge_diversity_loss / edge_diversity_pairs
        )
    if edge_usage_cells > 0:
        total_arch_loss = total_arch_loss + edge_usage_balance_weight * (
            edge_usage_balance_loss / edge_usage_cells
        )
    if edge_identity_cells > 0 and edge_identity_cap_weight > 0:
        total_arch_loss = total_arch_loss + edge_identity_cap_weight * (
            edge_identity_cap_loss / edge_identity_cells
        )

    return total_arch_loss, edge_diversity_pairs


def _add_edge_sharpening(
    *,
    model: nn.Module,
    total_arch_loss: torch.Tensor,
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    edge_sharpening_max_weight: float,
    edge_sharpening_start_frac: float,
    device: str,
):
    """Add late-phase entropy sharpening to encourage decisive operation choice."""
    edge_entropy = torch.tensor(0.0, device=device)
    edge_sharpen_weight = 0.0

    if edge_sharpening_max_weight <= 0 or epoch < warmup_epochs:
        return total_arch_loss, edge_entropy, edge_sharpen_weight

    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    if progress < edge_sharpening_start_frac:
        return total_arch_loss, edge_entropy, edge_sharpen_weight

    ramp = (progress - edge_sharpening_start_frac) / max(
        1e-8, 1.0 - edge_sharpening_start_frac
    )
    edge_sharpen_weight = edge_sharpening_max_weight * min(1.0, max(0.0, ramp))

    entropy_terms = []
    for cell in getattr(model, "cells", []):
        if not hasattr(cell, "edges"):
            continue
        for edge in cell.edges:
            if (
                hasattr(edge, "use_hierarchical")
                and edge.use_hierarchical
                and hasattr(edge, "group_alphas")
            ):
                gt = max(
                    float(
                        getattr(
                            edge, "group_temperature", getattr(edge, "temperature", 1.0)
                        )
                    ),
                    1e-6,
                )
                g_probs = F.softmax(edge.group_alphas / gt, dim=0)
                g_ent = -(g_probs * torch.log(g_probs + 1e-8)).sum() / np.log(
                    max(g_probs.numel(), 2)
                )
                entropy_terms.append(g_ent)
                if hasattr(edge, "op_alphas"):
                    ot = max(
                        float(
                            getattr(
                                edge,
                                "op_temperature",
                                getattr(edge, "temperature", 1.0),
                            )
                        ),
                        1e-6,
                    )
                    for alpha in edge.op_alphas.values():
                        o_probs = F.softmax(alpha / ot, dim=0)
                        o_ent = -(o_probs * torch.log(o_probs + 1e-8)).sum() / np.log(
                            max(o_probs.numel(), 2)
                        )
                        entropy_terms.append(o_ent)
            elif hasattr(edge, "_alphas"):
                ot = max(
                    float(
                        getattr(
                            edge, "op_temperature", getattr(edge, "temperature", 1.0)
                        )
                    ),
                    1e-6,
                )
                probs = F.softmax(edge._alphas / ot, dim=0)
                ent = -(probs * torch.log(probs + 1e-8)).sum() / np.log(
                    max(probs.numel(), 2)
                )
                entropy_terms.append(ent)

    if entropy_terms:
        edge_entropy = torch.stack(entropy_terms).mean()
        total_arch_loss = total_arch_loss + edge_sharpen_weight * edge_entropy

    return total_arch_loss, edge_entropy, edge_sharpen_weight


def _extract_edge_probs(edge) -> torch.Tensor | None:
    """Return per-operation probability vector for a single DARTS edge."""
    if (
        hasattr(edge, "use_hierarchical")
        and edge.use_hierarchical
        and hasattr(edge, "_get_weights")
        and hasattr(edge, "ops")
    ):
        try:
            routed = edge._get_weights(top_k=None)
            if routed:
                probs = torch.zeros(
                    len(edge.ops),
                    device=routed[0][1].device,
                    dtype=routed[0][1].dtype,
                )
                for op_idx, weight in routed:
                    probs[op_idx] = probs[op_idx] + weight
                return probs
        except Exception:
            pass
    if hasattr(edge, "_alphas"):
        temp = max(
            float(getattr(edge, "op_temperature", getattr(edge, "temperature", 1.0))),
            1e-6,
        )
        return F.softmax(edge._alphas / temp, dim=0)
    return None


