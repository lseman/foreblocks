"""Training-time helper utilities for DARTS search."""

from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from .regularization import default_as_probability_vector


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
