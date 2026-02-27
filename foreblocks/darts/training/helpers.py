"""Training-time helper utilities for DARTS search."""

from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler


def default_as_probability_vector(
    alpha_like: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Convert logits/probabilities to a normalized probability vector safely."""
    if alpha_like.numel() == 0:
        return alpha_like

    with torch.no_grad():
        flat = alpha_like.detach().reshape(-1)
        finite_ok = torch.isfinite(flat).all().item()
        in_range = flat.min().item() >= -1e-6 and flat.max().item() <= 1.0 + 1e-6
        sum_close = abs(flat.sum().item() - 1.0) <= 1e-4
        looks_like_probs = finite_ok and in_range and sum_close

    temp = max(float(temperature), 1e-6)
    if looks_like_probs:
        probs = alpha_like.clamp_min(1e-8)
        if abs(temp - 1.0) > 1e-8:
            probs = probs.pow(1.0 / temp)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return F.softmax(alpha_like / temp, dim=-1)


class RegularizationType(Enum):
    """Types of regularization for architecture search."""

    ENTROPY = "entropy"
    KL_DIVERGENCE = "kl_divergence"
    L2_NORM = "l2_norm"
    DIVERSITY = "diversity"
    SPARSITY = "sparsity"
    EFFICIENCY = "efficiency"


class ArchitectureRegularizer:
    """Helper class for different types of architecture regularization."""

    def __init__(
        self,
        reg_types: List[RegularizationType],
        weights: Optional[List[float]] = None,
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
        }

    @staticmethod
    def _zero_on_model_device(model: nn.Module) -> torch.Tensor:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    @staticmethod
    def _zero_on_arch_device(arch_params: List[torch.Tensor]) -> torch.Tensor:
        if arch_params:
            return torch.tensor(0.0, device=arch_params[0].device)
        return torch.tensor(0.0)

    def compute_regularization(
        self,
        model: nn.Module,
        arch_params: List[torch.Tensor],
        epoch: int = 0,
        total_epochs: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Compute all specified regularization terms."""
        if not arch_params:
            zero = self._zero_on_model_device(model)
            reg_losses = {reg_type.value: zero.clone() for reg_type in self.reg_types}
            reg_losses["total"] = zero
            return reg_losses

        reg_losses: Dict[str, torch.Tensor] = {}
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
        arch_params: List[torch.Tensor],
        reg_type: RegularizationType,
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        reg_fn = self._dispatch.get(reg_type)
        if reg_fn is None:
            return self._zero_on_model_device(model)
        return reg_fn(model, arch_params, epoch, total_epochs)

    def _iter_arch_probs(self, arch_params: List[torch.Tensor]):
        for param in arch_params:
            if param.dim() >= 1:
                yield F.softmax(param.view(-1, param.size(-1)), dim=-1)

    def _entropy_regularization(self, arch_params: List[torch.Tensor]) -> torch.Tensor:
        """Entropy regularization to encourage exploration."""
        total_entropy = self._zero_on_arch_device(arch_params)
        for probs in self._iter_arch_probs(arch_params):
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            total_entropy = total_entropy + 1.0 - entropy / np.log(probs.size(-1))
        return total_entropy

    def _kl_divergence_regularization(
        self, arch_params: List[torch.Tensor]
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

    def _l2_norm_regularization(self, arch_params: List[torch.Tensor]) -> torch.Tensor:
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
        self, arch_params: List[torch.Tensor], epoch: int, total_epochs: int
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
            "ResidualMLP": 0.2,
            "TimeConv": 0.3,
            "TCN": 0.5,
            "ConvMixer": 0.4,
            "Fourier": 0.6,
            "Wavelet": 0.6,
            "GRN": 0.4,
            "MultiScaleConv": 0.7,
            "PyramidConv": 0.8,
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


class TemperatureScheduler:
    """Advanced temperature scheduling for architecture search."""

    def __init__(
        self,
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        schedule_type: str = "cosine",
        warmup_epochs: int = 5,
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs

    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """Get temperature for current epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_temp

        anneal_span = max(total_epochs - self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / anneal_span
        progress = min(max(progress, 0.0), 1.0)

        schedule_fns = {
            "cosine": lambda: (
                self.final_temp
                + (self.initial_temp - self.final_temp)
                * (1 + np.cos(np.pi * progress))
                / 2
            ),
            "exponential": lambda: (
                self.initial_temp
                * np.exp(
                    np.log(self.final_temp / self.initial_temp)
                    / anneal_span
                    * (epoch - self.warmup_epochs)
                )
            ),
            "linear": lambda: (
                self.initial_temp - (self.initial_temp - self.final_temp) * progress
            ),
            "step": lambda: (
                self.initial_temp
                if progress < 0.3
                else (self.initial_temp * 0.5 if progress < 0.7 else self.final_temp)
            ),
        }
        temp = schedule_fns.get(self.schedule_type, lambda: self.initial_temp)()
        return max(float(temp), float(self.final_temp))


class BilevelOptimizer:
    """Encapsulates bilevel architecture-update data flow and stepping."""

    def __init__(
        self,
        *,
        arch_optimizer,
        arch_scheduler,
        arch_params: List[torch.Tensor],
        edge_arch_params: List[torch.Tensor],
        component_arch_params: List[torch.Tensor],
        use_bilevel_optimization: bool,
        train_arch_loader,
        val_loader,
        train_model_loader,
    ):
        self.arch_optimizer = arch_optimizer
        self.arch_scheduler = arch_scheduler
        self.arch_params = arch_params
        self.edge_arch_params = edge_arch_params
        self.component_arch_params = component_arch_params

        self.use_bilevel_optimization = use_bilevel_optimization
        self.train_arch_loader = train_arch_loader
        self.val_loader = val_loader
        self.train_model_loader = train_model_loader

        self.train_arch_iter = (
            iter(train_arch_loader)
            if use_bilevel_optimization and train_arch_loader is not None
            else None
        )
        self.val_arch_iter = iter(val_loader)
        self.train_model_iter = iter(train_model_loader)

    @staticmethod
    def _next_batch(data_iter, loader):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        return batch, data_iter

    def next_arch_batch(self):
        if self.use_bilevel_optimization:
            batch, self.train_arch_iter = self._next_batch(
                self.train_arch_iter, self.train_arch_loader
            )
            return batch
        batch, self.val_arch_iter = self._next_batch(
            self.val_arch_iter, self.val_loader
        )
        return batch

    def next_hessian_batch(self):
        batch, self.train_model_iter = self._next_batch(
            self.train_model_iter, self.train_model_loader
        )
        return batch

    def zero_arch_grads(self):
        self.arch_optimizer.zero_grad()

    def step_architecture(
        self,
        total_arch_loss: torch.Tensor,
        scaler: GradScaler,
        *,
        already_backward: bool = False,
    ):
        if not already_backward:
            scaler.scale(total_arch_loss).backward()
        scaler.unscale_(self.arch_optimizer)

        if self.edge_arch_params:
            torch.nn.utils.clip_grad_norm_(self.edge_arch_params, max_norm=5.0)
        if self.component_arch_params:
            torch.nn.utils.clip_grad_norm_(self.component_arch_params, max_norm=3.0)
        if not self.edge_arch_params and not self.component_arch_params:
            torch.nn.utils.clip_grad_norm_(self.arch_params, max_norm=3.0)

        scaler.step(self.arch_optimizer)

    def step_scheduler(self):
        self.arch_scheduler.step()


class AlphaTracker:
    """Encapsulates architecture-alpha extraction, iteration, and logging."""

    def __init__(self, as_probability_vector_fn=default_as_probability_vector):
        self._as_probability_vector = as_probability_vector_fn

    def iter_edge_alphas(self, model):
        if hasattr(model, "cells"):
            for i, cell in enumerate(model.cells):
                if hasattr(cell, "edges"):
                    for j, edge in enumerate(cell.edges):
                        alphas = getattr(edge, "alphas", None)
                        if alphas is None and hasattr(edge, "get_alphas"):
                            try:
                                alphas = edge.get_alphas()
                            except Exception:
                                continue
                        if alphas is None:
                            continue
                        name = f"cell_{i}_edge_{j}"
                        available_ops = getattr(edge, "available_ops", None)
                        yield name, alphas, available_ops

    def component_alpha_sources(self, model):
        sources = []

        norm_alpha = getattr(model, "norm_alpha", None)
        if norm_alpha is not None:
            sources.append(
                {
                    "name": "norm",
                    "log_name": "norm",
                    "alpha": norm_alpha,
                    "choices": ["revin", "instance_norm", "identity"],
                }
            )

        forecast_encoder = getattr(model, "forecast_encoder", None)
        encoder_alpha = getattr(forecast_encoder, "alphas", None)
        if encoder_alpha is not None:
            encoder_choices = list(
                getattr(
                    forecast_encoder, "encoder_names", ["lstm", "gru", "transformer"]
                )
            )
            sources.append(
                {
                    "name": "encoder",
                    "log_name": "forecast_encoder",
                    "alpha": encoder_alpha,
                    "choices": encoder_choices,
                }
            )

        forecast_decoder = getattr(model, "forecast_decoder", None)
        decoder_alpha = getattr(forecast_decoder, "alphas", None)
        if decoder_alpha is not None:
            decoder_choices = list(
                getattr(
                    forecast_decoder, "decoder_names", ["lstm", "gru", "transformer"]
                )
            )
            sources.append(
                {
                    "name": "decoder",
                    "log_name": "forecast_decoder",
                    "alpha": decoder_alpha,
                    "choices": decoder_choices,
                }
            )

        attention_alpha = getattr(forecast_decoder, "attention_alphas", None)
        if attention_alpha is not None:
            if attention_alpha.numel() == 2:
                attention_choices = ["use_attention", "no_attention"]
            else:
                attention_choices = [
                    f"choice_{i}" for i in range(int(attention_alpha.numel()))
                ]
            sources.append(
                {
                    "name": "attention_bridge",
                    "log_name": "attention_bridge",
                    "alpha": attention_alpha,
                    "choices": attention_choices,
                }
            )

        return sources

    def iter_component_alphas(self, model):
        for source in self.component_alpha_sources(model):
            yield source["name"], source["alpha"]

    def iter_prob_vectors(self, model, temperature=1.0):
        for _, alphas, _ in self.iter_edge_alphas(model):
            yield self._as_probability_vector(alphas, temperature=temperature)
        for _, alphas in self.iter_component_alphas(model):
            yield self._as_probability_vector(alphas, temperature=temperature)

    def extract_alpha_values(self, model):
        current_alphas = []
        for name, alphas, _ in self.iter_edge_alphas(model):
            current_alphas.append(
                (name, self._as_probability_vector(alphas).detach().cpu().numpy())
            )
        for name, alphas in self.iter_component_alphas(model):
            current_alphas.append(
                (name, self._as_probability_vector(alphas).detach().cpu().numpy())
            )
        return current_alphas

    def log_component_arch_updates(
        self, model, prev_component_probs: Dict[str, torch.Tensor]
    ):
        for source in self.component_alpha_sources(model):
            comp_key = source["name"]
            comp_name = source["log_name"]
            alpha_tensor = source["alpha"]
            choice_names = source["choices"]

            probs = self._as_probability_vector(alpha_tensor.detach(), temperature=1.0)
            prev_probs = prev_component_probs.get(comp_key)
            delta = 0.0
            if prev_probs is not None and prev_probs.shape == probs.shape:
                delta = float((probs - prev_probs).abs().sum().item())
            if not np.isfinite(delta):
                delta = 0.0

            top_idx = int(torch.argmax(probs).item())
            top_weight = float(probs[top_idx].item())
            choice_name = (
                choice_names[top_idx]
                if top_idx < len(choice_names)
                else f"op_{top_idx}"
            )

            print(
                f"   [Arch Update] {comp_name}: top={top_idx}, "
                f"choice={choice_name}, weight={top_weight:.4f}, dL1={delta:.6f}"
            )
            prev_component_probs[comp_key] = probs.clone()

    def summarize_edge_updates(self, model, prev_edge_probs: Dict[str, torch.Tensor]):
        edge_deltas = []
        edge_confidences = []
        edge_samples = []

        for cell_idx, cell in enumerate(getattr(model, "cells", [])):
            if not hasattr(cell, "edges"):
                continue
            for edge_idx, edge in enumerate(cell.edges):
                if not hasattr(edge, "get_alphas"):
                    continue
                try:
                    probs = self._as_probability_vector(
                        edge.get_alphas().detach(), temperature=1.0
                    )
                except Exception:
                    continue
                if probs.numel() == 0:
                    continue

                edge_name = f"cell_{cell_idx}_edge_{edge_idx}"
                prev_probs = prev_edge_probs.get(edge_name)
                if prev_probs is not None and prev_probs.shape == probs.shape:
                    edge_deltas.append((probs - prev_probs).abs().sum().item())

                edge_confidences.append(float(probs.max().item()))

                if len(edge_samples) < 2:
                    top_idx = int(torch.argmax(probs).item())
                    op_name = (
                        edge.available_ops[top_idx]
                        if hasattr(edge, "available_ops")
                        and top_idx < len(edge.available_ops)
                        else f"op_{top_idx}"
                    )
                    edge_samples.append(
                        f"{edge_name}:{op_name}@{probs[top_idx].item():.3f}"
                    )

                prev_edge_probs[edge_name] = probs.clone()

        if not edge_confidences:
            return None

        mean_edge_conf = float(np.mean(edge_confidences))
        mean_edge_delta = float(np.mean(edge_deltas)) if edge_deltas else 0.0
        if not np.isfinite(mean_edge_delta):
            mean_edge_delta = 0.0
        sample_text = ", ".join(edge_samples)

        return mean_edge_conf, mean_edge_delta, sample_text

    def log_architecture_update_block(
        self,
        model,
        prev_component_probs: Dict[str, torch.Tensor],
        prev_edge_probs: Dict[str, torch.Tensor],
        *,
        last_edge_sharpen_weight: float,
        last_edge_entropy: float,
        hessian_penalty_weight: float,
        hessian_penalty: torch.Tensor,
    ):
        self.log_component_arch_updates(model, prev_component_probs)

        edge_summary = self.summarize_edge_updates(model, prev_edge_probs)
        if edge_summary is None:
            return

        mean_edge_conf, mean_edge_delta, sample_text = edge_summary
        print(
            f"   [Edge Update] mean_top={mean_edge_conf:.4f}, "
            f"mean_dL1={mean_edge_delta:.6f}, samples=[{sample_text}]"
        )

        if last_edge_sharpen_weight > 0:
            print(
                f"   [Edge Sharpen] weight={last_edge_sharpen_weight:.4f}, "
                f"entropy={last_edge_entropy:.4f}"
            )

        if hessian_penalty_weight > 0:
            hp = float(hessian_penalty.detach().item())
            print(
                f"   [Hessian Penalty] value={hp:.6f}, "
                f"weight={hessian_penalty_weight:.4f}"
            )
