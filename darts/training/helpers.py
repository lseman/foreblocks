"""Training-time helper utilities for DARTS search."""

from enum import Enum

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


class TemperatureScheduler:
    """Advanced temperature scheduling for architecture search."""

    def __init__(
        self,
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        schedule_type: str = "cosine",
        warmup_epochs: int = 5,
        initial_drnas_concentration: float = 10.0,
        final_drnas_concentration: float = 2.0,
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.initial_drnas_concentration = max(float(initial_drnas_concentration), 1e-3)
        self.final_drnas_concentration = max(float(final_drnas_concentration), 1e-3)

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

    def get_drnas_concentration(self, epoch: int, total_epochs: int) -> float:
        """Cosine-anneal DrNAS Dirichlet concentration from high (exploration) to low (exploitation).

        High concentration → samples cluster near the mean (softmax) → exploration.
        Low  concentration → samples spread toward simplex vertices       → exploitation.
        """
        if epoch < self.warmup_epochs:
            return self.initial_drnas_concentration
        anneal_span = max(total_epochs - self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / anneal_span
        progress = min(max(progress, 0.0), 1.0)
        return (
            self.final_drnas_concentration
            + (self.initial_drnas_concentration - self.final_drnas_concentration)
            * (1.0 + np.cos(np.pi * progress))
            / 2.0
        )


class BilevelOptimizer:
    """Encapsulates bilevel architecture-update data flow and stepping."""

    def __init__(
        self,
        *,
        arch_optimizer,
        arch_scheduler,
        arch_params: list[torch.Tensor],
        edge_arch_params: list[torch.Tensor],
        component_arch_params: list[torch.Tensor],
        use_bilevel_optimization: bool,
        train_arch_loader,
        val_loader,
        train_model_loader,
        arch_grad_ema_beta: float = 0.0,
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

        # EMA buffer for arch gradients.  beta > 0 smooths noisy bilevel
        # gradient estimates; beta = 0.0 (default) disables the feature.
        self.arch_grad_ema_beta = float(arch_grad_ema_beta)
        self._arch_grad_ema: dict[int, torch.Tensor] = {}

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
        implicit_corrections: list[torch.Tensor | None] | None = None,
    ):
        if not already_backward:
            scaler.scale(total_arch_loss).backward()
        scaler.unscale_(self.arch_optimizer)

        # Apply implicit arch gradient corrections (DARTS second-order term).
        # These are pre-computed via finite differences in model-weight space and
        # added directly to arch param gradients after unscaling.
        if implicit_corrections is not None:
            for p, corr in zip(self.arch_params, implicit_corrections):
                if p.grad is not None and corr is not None:
                    p.grad.add_(corr.to(p.grad.device, p.grad.dtype))

        # Optional EMA smoothing: blend raw arch grads with a running average
        # to reduce variance from the noisy bilevel validation estimate.
        if self.arch_grad_ema_beta > 0.0:
            beta = self.arch_grad_ema_beta
            for p in self.arch_params:
                if p.grad is None:
                    continue
                pid = id(p)
                g = p.grad.detach().clone()
                if pid not in self._arch_grad_ema:
                    self._arch_grad_ema[pid] = g
                else:
                    self._arch_grad_ema[pid].mul_(beta).add_(g, alpha=1.0 - beta)
                p.grad.copy_(self._arch_grad_ema[pid])

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

        def _stack_mean(tensors):
            if not tensors:
                return None
            if len(tensors) == 1:
                return tensors[0]
            return torch.stack(tensors, dim=0).mean(dim=0)

        def _layer_component(component, key):
            if component is None:
                return []
            submodule = getattr(component, "transformer", None)
            if submodule is None:
                submodule = getattr(component, "rnn", None)
            if submodule is None:
                return []
            layers = getattr(submodule, "layers", None)
            if not layers:
                return []
            out = []
            for layer in layers:
                if isinstance(layer, dict):
                    item = layer.get(key)
                elif hasattr(layer, "get"):
                    item = layer.get(key)
                elif hasattr(layer, "__contains__") and key in layer:
                    item = layer[key]
                else:
                    item = None
                if item is not None:
                    out.append(item)
            return out

        def _first_self_attn(component):
            items = _layer_component(component, "self_attn")
            return items[0] if items else None

        def _first_cross_attn(component):
            items = _layer_component(component, "cross_attn")
            return items[0] if items else None

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
        transformer = getattr(forecast_encoder, "transformer", None)
        patch_alpha = getattr(transformer, "patch_alpha_logits", None)
        if patch_alpha is not None:
            sources.append(
                {
                    "name": "encoder_tokenizer",
                    "log_name": "encoder_tokenizer_decision",
                    "alpha": patch_alpha,
                    "choices": list(
                        getattr(
                            transformer,
                            "patch_mode_names",
                            [
                                "direct",
                                "patch_8",
                                "patch_16",
                                "patch_32",
                                "multi_scale_patch",
                                "variate_tokens",
                            ],
                        )
                    ),
                }
            )
        enc_self_attn = _first_self_attn(forecast_encoder)
        enc_attn_alpha = _stack_mean(
            [
                alpha
                for alpha in (
                    getattr(item, "attn_alphas", None)
                    for item in _layer_component(forecast_encoder, "self_attn")
                )
                if isinstance(alpha, torch.Tensor)
            ]
        )
        if enc_attn_alpha is not None:
            sources.append(
                {
                    "name": "encoder_self_attention",
                    "log_name": "forecast_encoder_self_attention",
                    "alpha": enc_attn_alpha,
                    "choices": list(getattr(enc_self_attn, "MODES", [])),
                }
            )
        enc_pos_alpha = _stack_mean(
            [
                alpha
                for alpha in (
                    getattr(item, "position_alphas", None)
                    for item in _layer_component(forecast_encoder, "self_attn")
                )
                if isinstance(alpha, torch.Tensor)
            ]
        )
        if enc_pos_alpha is not None:
            sources.append(
                {
                    "name": "encoder_attention_position",
                    "log_name": "encoder_attention_position_decision",
                    "alpha": enc_pos_alpha,
                    "choices": list(getattr(enc_self_attn, "POSITION_MODES", [])),
                }
            )
        enc_ffn = None
        enc_ffn_items = _layer_component(forecast_encoder, "ffn")
        if enc_ffn_items:
            enc_ffn = enc_ffn_items[0]
        enc_ffn_alpha = _stack_mean(
            [
                alpha
                for alpha in (
                    getattr(item, "ffn_alphas", None) for item in enc_ffn_items
                )
                if isinstance(alpha, torch.Tensor)
            ]
        )
        if enc_ffn_alpha is not None:
            sources.append(
                {
                    "name": "encoder_ffn",
                    "log_name": "encoder_ffn_decision",
                    "alpha": enc_ffn_alpha,
                    "choices": list(getattr(enc_ffn, "MODE_NAMES", ("swiglu", "moe"))),
                }
            )

        forecast_decoder = getattr(model, "forecast_decoder", None)
        decoder_style_alpha = getattr(forecast_decoder, "decode_style_alphas", None)
        if decoder_style_alpha is not None:
            sources.append(
                {
                    "name": "decoder_style",
                    "log_name": "informer_decision",
                    "alpha": decoder_style_alpha,
                    "choices": list(
                        getattr(
                            forecast_decoder,
                            "decode_style_names",
                            ("autoregressive", "informer"),
                        )
                    ),
                }
            )
        decoder_query_alpha = getattr(model, "decoder_query_alphas", None)
        if decoder_query_alpha is not None:
            sources.append(
                {
                    "name": "decoder_query_generator",
                    "log_name": "decoder_query_generator_decision",
                    "alpha": decoder_query_alpha,
                    "choices": list(
                        getattr(
                            model,
                            "decoder_query_mode_names",
                            (
                                "repeat_last",
                                "zeros",
                                "learned_horizon_queries",
                                "shifted_target",
                                "future_covariate_queries",
                            ),
                        )
                    ),
                }
            )
        dec_self_attn = _first_self_attn(forecast_decoder)
        dec_attn_alpha = _stack_mean(
            [
                alpha
                for alpha in (
                    getattr(item, "attn_alphas", None)
                    for item in _layer_component(forecast_decoder, "self_attn")
                )
                if isinstance(alpha, torch.Tensor)
            ]
        )
        if dec_attn_alpha is not None:
            sources.append(
                {
                    "name": "decoder_self_attention",
                    "log_name": "forecast_decoder_self_attention",
                    "alpha": dec_attn_alpha,
                    "choices": list(getattr(dec_self_attn, "MODES", [])),
                }
            )
        dec_pos_alpha = _stack_mean(
            [
                alpha
                for alpha in (
                    getattr(item, "position_alphas", None)
                    for item in _layer_component(forecast_decoder, "self_attn")
                )
                if isinstance(alpha, torch.Tensor)
            ]
        )
        if dec_pos_alpha is not None:
            sources.append(
                {
                    "name": "decoder_attention_position",
                    "log_name": "decoder_attention_position_decision",
                    "alpha": dec_pos_alpha,
                    "choices": list(getattr(dec_self_attn, "POSITION_MODES", [])),
                }
            )
        dec_ffn = None
        dec_ffn_items = _layer_component(forecast_decoder, "ffn")
        if dec_ffn_items:
            dec_ffn = dec_ffn_items[0]
        dec_ffn_alpha = _stack_mean(
            [
                alpha
                for alpha in (
                    getattr(item, "ffn_alphas", None) for item in dec_ffn_items
                )
                if isinstance(alpha, torch.Tensor)
            ]
        )
        if dec_ffn_alpha is not None:
            sources.append(
                {
                    "name": "decoder_ffn",
                    "log_name": "decoder_ffn_decision",
                    "alpha": dec_ffn_alpha,
                    "choices": list(getattr(dec_ffn, "MODE_NAMES", ("swiglu", "moe"))),
                }
            )

        memory_alpha = getattr(forecast_decoder, "memory_query_alphas", None)
        if memory_alpha is not None:
            sources.append(
                {
                    "name": "decoder_memory_queries",
                    "log_name": "forecast_decoder_memory_queries",
                    "alpha": memory_alpha,
                    "choices": [
                        str(q)
                        for q in getattr(
                            forecast_decoder,
                            "memory_query_options",
                            range(memory_alpha.numel()),
                        )
                    ],
                }
            )

        cross_attn = _first_cross_attn(forecast_decoder)
        attention_alpha = _stack_mean(
            [
                alpha
                for alpha in (
                    getattr(item, "attn_alphas", None)
                    for item in _layer_component(forecast_decoder, "cross_attn")
                )
                if isinstance(alpha, torch.Tensor)
            ]
        )
        if attention_alpha is not None:
            sources.append(
                {
                    "name": "decoder_cross_attention",
                    "log_name": "decoder_cross_attention_decision",
                    "alpha": attention_alpha,
                    "choices": list(getattr(cross_attn, "MODES", [])),
                }
            )
        cross_pos_alpha = _stack_mean(
            [
                alpha
                for alpha in (
                    getattr(item, "position_alphas", None)
                    for item in _layer_component(forecast_decoder, "cross_attn")
                )
                if isinstance(alpha, torch.Tensor)
            ]
        )
        if cross_pos_alpha is not None:
            sources.append(
                {
                    "name": "decoder_cross_attention_position",
                    "log_name": "decoder_cross_attention_position_decision",
                    "alpha": cross_pos_alpha,
                    "choices": list(getattr(cross_attn, "POSITION_MODES", [])),
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
        self, model, prev_component_probs: dict[str, torch.Tensor]
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
            if comp_key == "decoder_style":
                named_probs = {
                    str(name): float(weight.item())
                    for name, weight in zip(choice_names, probs)
                }
                print(
                    "   [Arch Update] informer_probs: "
                    f"autoregressive={named_probs.get('autoregressive', 0.0):.4f}, "
                    f"informer={named_probs.get('informer', 0.0):.4f}, "
                    f"selected={choice_name}"
                )
            if comp_key == "encoder_tokenizer":
                named_probs = {
                    str(name): float(weight.item())
                    for name, weight in zip(choice_names, probs)
                }
                top_modes = sorted(
                    named_probs.items(), key=lambda item: item[1], reverse=True
                )[:3]
                print(
                    "   [Arch Update] encoder_tokenizer_probs: "
                    + ", ".join(f"{name}={weight:.4f}" for name, weight in top_modes)
                )
            if comp_key == "decoder_query_generator":
                named_probs = {
                    str(name): float(weight.item())
                    for name, weight in zip(choice_names, probs)
                }
                print(
                    "   [Arch Update] decoder_query_probs: "
                    + ", ".join(
                        f"{name}={weight:.4f}" for name, weight in named_probs.items()
                    )
                    + f", selected={choice_name}"
                )
            if comp_key in {"encoder_ffn", "decoder_ffn"}:
                named_probs = {
                    str(name): float(weight.item())
                    for name, weight in zip(choice_names, probs)
                }
                print(
                    f"   [Arch Update] {comp_key}_probs: "
                    + ", ".join(
                        f"{name}={weight:.4f}" for name, weight in named_probs.items()
                    )
                    + f", selected={choice_name}"
                )
            prev_component_probs[comp_key] = probs.clone()

    def summarize_edge_updates(self, model, prev_edge_probs: dict[str, torch.Tensor]):
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
        prev_component_probs: dict[str, torch.Tensor],
        prev_edge_probs: dict[str, torch.Tensor],
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
