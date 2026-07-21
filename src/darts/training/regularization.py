"""Training-time helper utilities for DARTS search."""

from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from ..utils.tensors import as_probability_vector as default_as_probability_vector


class RegularizationType(Enum):
    """Types of regularization for architecture search."""

    ENTROPY = "entropy"
    KL_DIVERGENCE = "kl_divergence"
    L2_NORM = "l2_norm"
    DIVERSITY = "diversity"
    SPARSITY = "sparsity"
    EFFICIENCY = "efficiency"
    SCHEDULED = "scheduled"


class ArchitectureRegularizer:
    """Helper class for different types of architecture regularization."""

    def __init__(
        self,
        reg_types: list[RegularizationType],
        weights: list[float] | None = None,
        as_probability_vector_fn=default_as_probability_vector,
    ):
        self.reg_types = reg_types
        self.weights = weights or [1.0] * len(reg_types)
        self._as_probability_vector = as_probability_vector_fn
        if len(self.reg_types) != len(self.weights):
            raise ValueError(
                "Number of regularization weights must match regularization types"
            )
        self._dispatch = {
            RegularizationType.ENTROPY: lambda m, p, e, t: self._entropy_regularization(
                p
            ),
            RegularizationType.KL_DIVERGENCE: lambda m, p, e, t: (
                self._kl_divergence_regularization(p)
            ),
            RegularizationType.L2_NORM: lambda m, p, e, t: self._l2_norm_regularization(
                p
            ),
            RegularizationType.DIVERSITY: lambda m, p, e, t: (
                self._diversity_regularization(m)
            ),
            RegularizationType.SPARSITY: lambda m, p, e, t: (
                self._sparsity_regularization(p, e, t)
            ),
            RegularizationType.EFFICIENCY: lambda m, p, e, t: (
                self._efficiency_regularization(m)
            ),
            RegularizationType.SCHEDULED: lambda m, p, e, t: (
                self._scheduled_regularization(p, e, t)
            ),
        }

    @staticmethod
    def _zero_on_model_device(model: nn.Module) -> torch.Tensor:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    @staticmethod
    def _zero_on_arch_device(arch_params: list[torch.Tensor]) -> torch.Tensor:
        if arch_params:
            return torch.tensor(0.0, device=arch_params[0].device)
        return torch.tensor(0.0)

    def compute_regularization(
        self,
        model: nn.Module,
        arch_params: list[torch.Tensor],
        epoch: int = 0,
        total_epochs: int = 100,
    ) -> dict[str, torch.Tensor]:
        """Compute all specified regularization terms."""
        if not arch_params:
            zero = self._zero_on_model_device(model)
            reg_losses = {reg_type.value: zero.clone() for reg_type in self.reg_types}
            reg_losses["total"] = zero
            return reg_losses

        reg_losses: dict[str, torch.Tensor] = {}
        total_reg = self._zero_on_model_device(model)

        for reg_type, weight in zip(self.reg_types, self.weights):
            reg_loss = self._compute_single_regularization(
                model, arch_params, reg_type, epoch, total_epochs
            )
            reg_losses[reg_type.value] = reg_loss
            total_reg = total_reg + float(weight) * reg_loss

        reg_losses["total"] = total_reg
        return reg_losses

    def _compute_single_regularization(
        self,
        model: nn.Module,
        arch_params: list[torch.Tensor],
        reg_type: RegularizationType,
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        reg_fn = self._dispatch.get(reg_type)
        if reg_fn is None:
            return self._zero_on_model_device(model)
        return reg_fn(model, arch_params, epoch, total_epochs)

    def _iter_arch_probs(self, arch_params: list[torch.Tensor]):
        for param in arch_params:
            if param.dim() >= 1:
                yield F.softmax(param.view(-1, param.size(-1)), dim=-1)

    def _entropy_regularization(self, arch_params: list[torch.Tensor]) -> torch.Tensor:
        """Entropy regularization to encourage exploration."""
        total_entropy = self._zero_on_arch_device(arch_params)
        for probs in self._iter_arch_probs(arch_params):
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            total_entropy = total_entropy + 1.0 - entropy / np.log(probs.size(-1))
        return total_entropy

    def _kl_divergence_regularization(
        self, arch_params: list[torch.Tensor]
    ) -> torch.Tensor:
        """Encourage specialization by maximizing divergence from uniform."""
        total_kl = self._zero_on_arch_device(arch_params)
        for probs in self._iter_arch_probs(arch_params):
            uniform = torch.ones_like(probs) / probs.size(-1)
            kl_div = (
                (probs * torch.log(probs / (uniform + 1e-8) + 1e-8)).sum(dim=-1).mean()
            )
            total_kl = total_kl + kl_div

        # Return negative KL so minimizing total loss increases KL(probs || uniform).
        return -total_kl

    def _l2_norm_regularization(self, arch_params: list[torch.Tensor]) -> torch.Tensor:
        """L2 norm regularization on architecture parameters."""
        total_l2 = self._zero_on_arch_device(arch_params)
        for param in arch_params:
            total_l2 = total_l2 + torch.norm(param, p=2)
        return total_l2

    def _diversity_regularization(self, model: nn.Module) -> torch.Tensor:
        """Encourage diversity across different parts of the architecture."""
        diversity_loss = self._zero_on_model_device(model)

        all_weights = []
        for module in model.modules():
            if hasattr(module, "get_alphas"):
                try:
                    alphas = module.get_alphas()
                    if alphas.numel() > 0:
                        all_weights.append(self._as_probability_vector(alphas))
                except Exception:
                    continue

        if len(all_weights) >= 2:
            for i in range(len(all_weights)):
                for j in range(i + 1, len(all_weights)):
                    w1, w2 = all_weights[i], all_weights[j]
                    min_size = min(w1.size(0), w2.size(0))
                    w1_trunc = w1[:min_size]
                    w2_trunc = w2[:min_size]
                    cos_sim = F.cosine_similarity(w1_trunc, w2_trunc, dim=0)
                    diversity_loss = diversity_loss + cos_sim

        return diversity_loss

    def _sparsity_regularization(
        self, arch_params: list[torch.Tensor], epoch: int, total_epochs: int
    ) -> torch.Tensor:
        """Sparsity regularization that increases over time."""
        sparsity_loss = self._zero_on_arch_device(arch_params)
        sparsity_weight = min(1.0, epoch / max(total_epochs * 0.8, 1e-8))

        for probs in self._iter_arch_probs(arch_params):
            n = probs.size(-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            normalized_entropy = entropy / np.log(max(n, 2))
            sparsity_loss = sparsity_loss + sparsity_weight * normalized_entropy

        return sparsity_loss

    def _efficiency_regularization(self, model: nn.Module) -> torch.Tensor:
        """Efficiency regularization based on operation complexity."""
        efficiency_loss = self._zero_on_model_device(model)
        op_costs = {
            "Identity": 0.0,
            "ResidualMLP": 0.15,
            "TimeConv": 0.25,
            "TCN": 0.5,
            "ConvMixer": 0.45,
            "Fourier": 0.6,
            "Wavelet": 0.6,
            "GRN": 0.35,
            "MultiScaleConv": 0.7,
            "PyramidConv": 0.8,
            # Extended op pool — ops added after initial table
            "PatchEmbed": 0.35,
            "InvertedAttention": 0.65,
        }

        for module in model.modules():
            if hasattr(module, "get_alphas") and hasattr(module, "available_ops"):
                try:
                    alphas = module.get_alphas()
                    probs = self._as_probability_vector(alphas)
                    for i, op_name in enumerate(module.available_ops):
                        if i < len(probs):
                            cost = op_costs.get(op_name, 0.5)
                            efficiency_loss = efficiency_loss + probs[i] * cost
                except Exception:
                    continue

        return efficiency_loss

    def _scheduled_regularization(
        self,
        arch_params: list[torch.Tensor],
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        """Cosine-blend from entropy bonus (exploration) to sparsity penalty
        (exploitation) over the course of training.

        - Progress < 0.40: pure entropy *bonus* (negative contribution) —
          encourages diverse, high-entropy architecture distributions.
        - Progress > 0.70: pure sparsity *penalty* (positive contribution) —
          encourages decisive, low-entropy commitment to specific ops.
        - In between: smooth cosine interpolation from –1 to +1.
        """
        if not arch_params:
            return self._zero_on_arch_device(arch_params)

        progress = min(1.0, max(0.0, epoch / max(total_epochs, 1)))
        transition_start, transition_end = 0.40, 0.70
        if progress <= transition_start:
            blend = -1.0  # exploration bonus
        elif progress >= transition_end:
            blend = 1.0  # exploitation penalty
        else:
            t = (progress - transition_start) / (transition_end - transition_start)
            blend = float(-np.cos(np.pi * t))  # smoothly −1 → +1

        scheduled_loss = self._zero_on_arch_device(arch_params)
        for probs in self._iter_arch_probs(arch_params):
            n = probs.size(-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            normalized_entropy = entropy / np.log(max(n, 2))
            scheduled_loss = scheduled_loss + float(blend) * normalized_entropy

        return scheduled_loss
