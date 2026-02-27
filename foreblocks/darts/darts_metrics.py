import contextlib
import importlib
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .scoring import (
    normalize_metric_value as _normalize_metric_value_shared,
    score_from_metrics as _score_from_metrics_shared,
)

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Config:
    """Unified configuration"""

    max_samples: int = 32
    max_outputs: int = 10
    eps: float = 1e-8
    timeout: float = 30.0
    enable_mixed_precision: bool = False
    jacobian_probes: int = 2
    # SNIP is defined at initialization; keep this as the default behavior.
    snip_at_init: bool = True
    # Explicit mode: "init" (paper-consistent) or "current" (fast proxy).
    snip_mode: str = "init"
    heavy_metrics_batches: int = 1
    gradient_max_samples: int = 4
    conditioning_every_n_layers: int = 3
    conditioning_min_out_features: int = 0
    conditioning_power_iters: int = 6
    conditioning_exact_max_dim: int = 64
    conditioning_inverse_shift: float = 1e-6
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "synflow": 0.25,
            "grasp": 0.20,
            "fisher": 0.20,
            "jacobian": 0.15,
            "naswot": 0.15,
            "snip": 0.15,
            "params": -0.05,
            "conditioning": -0.10,
            "flops": -0.05,
            "sensitivity": 0.10,
            "activation_diversity": 0.10,
        }
    )


@dataclass
class Result:
    """Metric computation result"""

    value: float
    success: bool = True
    error: str = ""
    time: float = 0.0

    def __repr__(self):
        status = "✓" if self.success else "✗"
        return f"Result({status} {self.value:.4f}, {self.time:.3f}s)"


class CompatibilityHelper:
    """Handles model compatibility issues and fallback routines"""

    @staticmethod
    @contextlib.contextmanager
    def safe_mode(model):
        """Temporarily disables CuDNN and replaces SDPA with manual attention"""
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        original_sdpa = getattr(F, "scaled_dot_product_attention", None)
        if original_sdpa:
            F.scaled_dot_product_attention = CompatibilityHelper._manual_attention

        try:
            yield
        finally:
            torch.backends.cudnn.enabled = prev_cudnn
            if original_sdpa:
                F.scaled_dot_product_attention = original_sdpa

    @staticmethod
    def _manual_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        """Simplified SDPA fallback using matmul and masking"""
        _ = enable_gqa
        scale = scale or (query.size(-1) ** -0.5)
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn += attn_mask
        elif is_causal:
            L = query.size(-2)
            causal_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=query.device), diagonal=1
            )
            attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, value)

    @staticmethod
    def prepare_data(outputs, targets):
        """Align output and target shapes for metric computation"""
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Handle sequence-to-sequence alignment
        if outputs.ndim == 3 or targets.ndim == 3:
            outputs = outputs[:, -1] if outputs.ndim == 3 else outputs
            targets = targets[:, -1] if targets.ndim == 3 else targets

        # Classification: long targets
        if targets.dtype == torch.long:
            if outputs.ndim > 2:
                outputs = outputs.view(outputs.size(0), -1)
            if targets.ndim == 2 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            if outputs.size(-1) == 1 and targets.max() <= 1:
                outputs = outputs.squeeze(-1)
        else:
            # Regression: force shape match
            if outputs.shape != targets.shape:
                try:
                    outputs = outputs.view_as(targets)
                except RuntimeError:
                    if outputs.ndim == 1:
                        outputs = outputs.unsqueeze(1)
                    if targets.ndim == 1:
                        targets = targets.unsqueeze(1)

        return outputs, targets

    @staticmethod
    def get_loss_fn(targets):
        """Select appropriate loss function based on task type"""
        return nn.CrossEntropyLoss() if targets.dtype == torch.long else nn.MSELoss()


class MetricsComputer:
    """Optimized metrics computer with shared hooks and minimal forward passes"""

    def __init__(self, config: Config):
        self.config = config
        self.helper = CompatibilityHelper()

    @staticmethod
    def _unwrap_output(output: Any) -> torch.Tensor:
        """Extract a tensor prediction from common model output structures."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and len(output) > 0:
            for item in output:
                if torch.is_tensor(item):
                    return item
        if isinstance(output, dict):
            for key in ("pred", "preds", "prediction", "output", "logits"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            for value in output.values():
                if torch.is_tensor(value):
                    return value
        raise TypeError(f"Unsupported model output type for metrics: {type(output)}")

    def _finite_difference_sensitivity(self, model, inputs: torch.Tensor) -> float:
        """Finite-difference input sensitivity fallback (autograd-free)."""
        was_training = model.training
        model.eval()
        try:
            x = inputs[: min(inputs.size(0), self.config.max_samples)]
            eps = 1e-2
            noise = torch.randn_like(x)
            denom = noise.norm().item() + self.config.eps

            with torch.no_grad():
                y1 = self._unwrap_output(model(x))
                y2 = self._unwrap_output(model(x + eps * noise))

            num = (y2 - y1).norm().item()
            return float(num / (eps * denom + self.config.eps))
        finally:
            if was_training:
                model.train()

    def _finite_difference_jacobian(
        self, model, inputs: torch.Tensor, d_out: Optional[int] = None
    ) -> float:
        """Directional finite-difference proxy for Tr(JJ^T)/d_in."""
        was_training = model.training
        model.eval()
        try:
            bs = min(inputs.size(0), self.config.max_samples)
            x = inputs[:bs]

            with torch.no_grad():
                y0 = self._unwrap_output(model(x))

            if y0.dim() == 1:
                y0 = y0.unsqueeze(1)
            elif y0.dim() > 2:
                y0 = y0.flatten(1)

            total_out = int(y0.size(1))
            if total_out < 1:
                return 0.0

            d_eff = min(total_out, int(d_out or self.config.max_outputs))

            eps = 1e-2
            u = torch.randn_like(x)
            u_norm = u.view(bs, -1).norm(dim=1, keepdim=True).clamp_min(self.config.eps)
            u = u / u_norm.view([bs] + [1] * (x.dim() - 1))

            with torch.no_grad():
                yp = self._unwrap_output(model(x + eps * u))
                ym = self._unwrap_output(model(x - eps * u))

            if yp.dim() == 1:
                yp = yp.unsqueeze(1)
                ym = ym.unsqueeze(1)
            elif yp.dim() > 2:
                yp = yp.flatten(1)
                ym = ym.flatten(1)

            yp = yp[:, :d_eff]
            ym = ym[:, :d_eff]
            jvp = (yp - ym) / (2.0 * eps)

            # With unit-norm random input direction u:
            # E||J u||^2 = Tr(JJ^T) / d_in.
            trace_est = float((jvp.pow(2).sum(dim=1)).mean().item())
            normalized = trace_est
            return float(np.clip(np.log(normalized + self.config.eps), -12, 12))
        finally:
            if was_training:
                model.train()

    def compute_model_only_metrics(self, model: nn.Module) -> Dict[str, Result]:
        return {
            "params": self.params(model),
            "conditioning": self.conditioning(model),
        }

    def compute_all(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        include_heavy_metrics: bool = True,
        model_only_results: Optional[Dict[str, Result]] = None,
    ) -> Dict[str, Result]:
        """Compute all metrics with shared hooks and minimal forward passes"""
        results = {}

        # Model-only metrics (no forward pass needed)
        if model_only_results is None:
            results.update(self.compute_model_only_metrics(model))
        else:
            results.update(model_only_results)

        # Shared activation collection for multiple metrics
        activations = {}
        conv_linear_modules = []
        relu_modules = []
        flops_count = {}

        # Single hook setup for all metrics that need activations
        def activation_hook(name):
            def hook(module, inp, out):
                # Store for NASWOT and Zen-NAS
                act = out[0] if isinstance(out, tuple) else out
                activations[name] = act.detach()

                # FLOPS counting inline
                input_shape = inp[0].shape
                output_shape = out.shape if not isinstance(out, tuple) else out[0].shape

                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    kernel_ops = (
                        np.prod(module.kernel_size)
                        * module.in_channels
                        // module.groups
                    )
                    output_elements = np.prod(output_shape)
                    flops = output_elements * kernel_ops * 2
                elif isinstance(module, nn.Linear):
                    flops = (
                        input_shape[0] * module.in_features * module.out_features * 2
                    )
                else:
                    flops = 0

                flops_count[name] = flops

            return hook

        # Register hooks once for all metrics
        hooks = []

        def is_relu_like(module):
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                return True
            if isinstance(module, nn.LeakyReLU):
                return getattr(module, "negative_slope", 0.0) == 0.0
            return False

        for module_name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                conv_linear_modules.append((module_name, module))
                hooks.append(module.register_forward_hook(activation_hook(module_name)))
        relu_modules = [
            (name, module)
            for name, module in model.named_modules()
            if is_relu_like(module)
        ]

        try:
            # Single forward pass for multiple metrics
            was_training = model.training
            model.eval()

            shared_inputs = None
            if include_heavy_metrics:
                # Shared outputs may feed gradient-based metrics (GRASP/Fisher/SNIP,
                # Jacobian, sensitivity). For CuDNN RNNs, backward after an eval-mode
                # forward can fail; ensure this graph-producing pass runs in train mode.
                model.train()
                shared_inputs = inputs.detach().clone().requires_grad_(True)
                with self.helper.safe_mode(model):
                    shared_outputs = model(shared_inputs)
            else:
                with torch.no_grad():
                    with self.helper.safe_mode(model):
                        shared_outputs = model(inputs)

            # Process all metrics that only need activations
            results.update(
                self._compute_activation_metrics(
                    activations, conv_linear_modules, relu_modules, flops_count
                )
            )

            # Metrics requiring gradients (separate forward passes with minimal overhead)
            if targets is not None:
                results.update(
                    self._compute_gradient_metrics(
                        model,
                        inputs,
                        targets,
                        include_snip=include_heavy_metrics,
                        shared_inputs=shared_inputs,
                        shared_outputs=shared_outputs,
                    )
                )
                if "snip" not in results:
                    results["snip"] = Result(
                        0.0,
                        False,
                        "Skipped (include_heavy_metrics=False)",
                        0.0,
                    )

            # Jacobian must run before SynFlow when reusing shared graph tensors.
            # SynFlow mutates weights and calls model.zero_grad() in cleanup; even though
            # it is isolated, keeping Jacobian first makes graph-lifetime dependencies explicit.
            if include_heavy_metrics:
                results["jacobian"] = self._compute_jacobian(
                    model,
                    inputs,
                    shared_outputs=shared_outputs,
                    shared_inputs=shared_inputs,
                )
            else:
                results["jacobian"] = Result(
                    0.0,
                    False,
                    "Skipped (include_heavy_metrics=False)",
                    0.0,
                )

            # SynFlow (independent, runs after graph-dependent metrics)
            if include_heavy_metrics:
                results["synflow"] = self._compute_synflow(model, inputs)
            else:
                results["synflow"] = Result(
                    0.0,
                    False,
                    "Skipped (include_heavy_metrics=False)",
                    0.0,
                )

            # Sensitivity (prefer shared gradient pass when available)
            if "sensitivity" not in results:
                results["sensitivity"] = self.sensitivity(
                    model,
                    inputs,
                    shared_outputs=shared_outputs,
                    shared_inputs=shared_inputs,
                )

        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            if not was_training:
                model.eval()

        return results

    def _compute_activation_metrics(
        self, activations, conv_linear_modules, relu_modules, flops_count
    ):
        """Compute metrics that only need stored activations"""
        results = {}

        # ────────────────────────────────────────────────
        # NASWOT kernel from binary activation agreements:
        # K = B B^T + (1-B)(1-B)^T, then score = log|det(K)|.
        # ────────────────────────────────────────────────
        def _naswot():
            total_logdet = 0.0
            valid_layers = 0

            for name, _ in conv_linear_modules:
                if name not in activations:
                    continue
                act = activations[name]
                if act.size(0) < 2:  # need at least 2 samples for meaningful kernel
                    continue

                try:
                    flat = act.flatten(1)
                    binary = (flat > 0).to(dtype=torch.float64)
                    inv_binary = 1.0 - binary

                    # Agreement-count kernel used by NASWOT.
                    kernel = binary @ binary.t() + inv_binary @ inv_binary.t()
                    kernel = 0.5 * (kernel + kernel.t())

                    sign, logdet = torch.linalg.slogdet(kernel)
                    if sign.item() <= 0 or not torch.isfinite(logdet):
                        # Minimal jitter fallback for numerical stability only.
                        eye = torch.eye(
                            kernel.size(0), device=kernel.device, dtype=kernel.dtype
                        )
                        jitter = max(float(self.config.eps), 1e-12)
                        stable = False
                        for _ in range(6):
                            sign, logdet = torch.linalg.slogdet(kernel + jitter * eye)
                            if sign.item() > 0 and torch.isfinite(logdet):
                                stable = True
                                break
                            jitter *= 10.0
                        if not stable:
                            continue

                    total_logdet += float(logdet.item())
                    valid_layers += 1
                except RuntimeError:
                    continue

            if valid_layers == 0:
                return 0.0
            return total_logdet / valid_layers

        results["naswot"] = self._compute_safely(_naswot)

        # ────────────────────────────────────────────────
        # Activation diversity proxy (non-canonical Zen-NAS):
        # mean absolute pairwise cosine across layers.
        # ────────────────────────────────────────────────
        def _activation_diversity():
            total_score = 0.0
            valid_layers = 0

            for name, _ in relu_modules:
                if name not in activations:
                    continue
                act = activations[name]
                if act.size(0) < 2:
                    continue

                try:
                    flat = act.flatten(1)  # [B, features]
                    norm = flat.norm(dim=1, keepdim=True).clamp_min(self.config.eps)
                    normalized = flat / norm

                    cos = normalized @ normalized.t()  # [B,B]
                    eye = torch.eye(cos.size(0), device=cos.device, dtype=torch.bool)
                    pairwise = cos.masked_fill(eye, 0.0)

                    # Literature-style simple proxy: mean absolute pairwise cosine.
                    denom = max(cos.numel() - cos.size(0), 1)
                    score = pairwise.abs().sum() / denom

                    if torch.isfinite(score):
                        total_score += score.item()
                        valid_layers += 1
                except Exception:
                    continue

            if valid_layers == 0:
                return 0.0
            return total_score / valid_layers

        results["activation_diversity"] = self._compute_safely(_activation_diversity)

        # FLOPS remains unchanged (already reasonable)
        def _flops():
            total = sum(flops_count.values())
            return max(total, 1.0)

        results["flops"] = self._compute_safely(_flops)

        return results

    def _compute_gradient_metrics(
        self,
        model,
        inputs,
        targets,
        include_snip: bool = True,
        shared_inputs: Optional[torch.Tensor] = None,
        shared_outputs: Optional[torch.Tensor] = None,
    ):
        """Compute GRASP/Fisher on current weights and SNIP on init-time weights."""
        results = {}
        was_training = model.training
        model.train()

        try:
            grad_bs = int(getattr(self.config, "gradient_max_samples", 0) or 0)
            if grad_bs > 0:
                x = inputs[:grad_bs].clone().detach()
                y = targets[:grad_bs].clone().detach()
            else:
                x, y = inputs.clone().detach(), targets.clone().detach()
            loss_fn = self.helper.get_loss_fn(y)

            can_reuse_shared = (
                shared_inputs is not None
                and shared_outputs is not None
                and shared_inputs.requires_grad
                and shared_outputs.requires_grad
                and shared_inputs.size(0)
                >= (grad_bs if grad_bs > 0 else inputs.size(0))
                and shared_inputs.shape[1:] == inputs.shape[1:]
            )

            if can_reuse_shared:
                reuse_bs = grad_bs if grad_bs > 0 else inputs.size(0)
                x = cast(torch.Tensor, shared_inputs)[:reuse_bs]
                y = targets[: x.size(0)].clone().detach()
                outputs = cast(torch.Tensor, shared_outputs)[: x.size(0)]
            else:
                x.requires_grad = True
                # GRASP uses second-order derivatives; ensure the graph is built
                # with CuDNN-disabled kernels when shared graph reuse is unavailable.
                with self.helper.safe_mode(model):
                    outputs = model(x)
            outputs, y_prep = self.helper.prepare_data(outputs, y)
            loss = loss_fn(outputs, y_prep)

            if not torch.isfinite(loss):
                for name in ["grasp", "fisher", "snip"]:
                    results[name] = Result(0.0, False, "Non-finite loss", 0.0)
                return results

            weight_params = [
                (n, p) for n, p in model.named_parameters() if p.requires_grad
            ]
            weights = [p for _, p in weight_params]

            # Shared first-order gradients for Fisher/SNIP.
            grads_first_order = torch.autograd.grad(
                loss,
                weights,
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )

            snip_mode_raw = str(getattr(self.config, "snip_mode", "")).strip().lower()
            if snip_mode_raw in {"init", "current"}:
                snip_mode = snip_mode_raw
            else:
                snip_mode = (
                    "init" if bool(getattr(self.config, "snip_at_init", True)) else "current"
                )

            # ----- GRASP -----
            def _grasp():
                def _grasp_from_loss(
                    loss_local: torch.Tensor,
                    weights_local,
                    *,
                    retain_graph: bool,
                ) -> float:
                    grads_local = torch.autograd.grad(
                        loss_local,
                        weights_local,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )

                    hvp_seed = torch.tensor(0.0, device=loss_local.device)
                    valid_grads = 0
                    for g in grads_local:
                        if g is None or not torch.isfinite(g).all():
                            continue
                        # Seed for HVP: d/dw <g, stopgrad(g)> = H g
                        hvp_seed = hvp_seed + (g * g.detach()).sum()
                        valid_grads += 1

                    if valid_grads == 0:
                        return 0.0

                    hgs_local = torch.autograd.grad(
                        hvp_seed,
                        weights_local,
                        create_graph=False,
                        retain_graph=retain_graph,
                        allow_unused=True,
                    )

                    scores = []
                    for hg, g in zip(hgs_local, grads_local):
                        if hg is None or g is None:
                            continue
                        if not torch.isfinite(hg).all() or not torch.isfinite(g).all():
                            continue
                        scores.append((hg * g.detach()).sum().item())

                    if not scores:
                        return 0.0

                    # GRASP objective: - <H g, g>
                    return -sum(scores) / max(len(scores), 1)

                try:
                    return _grasp_from_loss(
                        loss,
                        weights,
                        retain_graph=True,
                    )
                except RuntimeError as e:
                    # CuDNN RNNs may fail on double-backward. Retry with a fresh
                    # CuDNN-disabled graph instead of failing the metric entirely.
                    if "_cudnn_rnn_backward" not in str(e):
                        raise

                    model.zero_grad()
                    x_retry = x.detach().clone().requires_grad_(True)
                    y_retry = y.detach().clone()
                    with self.helper.safe_mode(model):
                        out_retry = model(x_retry)
                    out_retry, y_retry_prep = self.helper.prepare_data(out_retry, y_retry)
                    loss_retry = loss_fn(out_retry, y_retry_prep)

                    if not torch.isfinite(loss_retry):
                        return 0.0

                    return _grasp_from_loss(
                        loss_retry,
                        weights,
                        retain_graph=False,
                    )

            # ----- Fisher -----
            def _fisher():
                fisher_per_sample = []
                x_f = x.detach()
                y_f = y.detach()

                for i in range(int(x_f.size(0))):
                    xi = x_f[i : i + 1]
                    yi = y_f[i : i + 1]
                    out_i = model(xi)
                    out_i, yi_prep = self.helper.prepare_data(out_i, yi)
                    loss_i = loss_fn(out_i, yi_prep)
                    if not torch.isfinite(loss_i):
                        continue

                    grads_i = torch.autograd.grad(
                        loss_i,
                        weights,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True,
                    )

                    vals_i = [
                        gi.pow(2).sum().item()
                        for gi in grads_i
                        if gi is not None and torch.isfinite(gi).all()
                    ]
                    if vals_i:
                        fisher_per_sample.append(sum(vals_i) / max(len(vals_i), 1))

                if not fisher_per_sample:
                    return 0.0
                return float(sum(fisher_per_sample) / len(fisher_per_sample))

            # ----- SNIP -----
            def _snip():
                if snip_mode == "current":
                    snip_value = 0.0
                    snip_count = 0
                    for (n, p), g in zip(weight_params, grads_first_order):
                        if "weight" not in n:
                            continue
                        if g is not None and torch.isfinite(g).all():
                            snip_value += (g.detach() * p.detach()).abs().sum().item()
                            snip_count += 1
                    if snip_count == 0:
                        return 0.0
                    return snip_value / snip_count

                state_backup = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                snip_value = 0.0
                snip_count = 0
                try:
                    for module in model.modules():
                        if hasattr(module, "reset_parameters"):
                            module.reset_parameters()

                    model.zero_grad()
                    x0 = x.clone().detach().requires_grad_(True)
                    y0 = y.clone().detach()

                    outputs0 = model(x0)
                    outputs0, y0_prep = self.helper.prepare_data(outputs0, y0)
                    loss0 = loss_fn(outputs0, y0_prep)

                    if not torch.isfinite(loss0):
                        return 0.0

                    init_weight_params = [
                        (n, p)
                        for n, p in model.named_parameters()
                        if p.requires_grad and "weight" in n
                    ]
                    init_weights = [p for _, p in init_weight_params]

                    init_grads = torch.autograd.grad(
                        loss0,
                        init_weights,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True,
                    )

                    for (_, p), g in zip(init_weight_params, init_grads):
                        if g is not None and torch.isfinite(g).all():
                            snip_value += (g * p).abs().sum().item()
                            snip_count += 1
                finally:
                    model.load_state_dict(state_backup, strict=False)
                    model.zero_grad()

                if snip_count == 0:
                    return 0.0
                return snip_value / snip_count

            results["grasp"] = self._compute_safely(_grasp)
            results["fisher"] = self._compute_safely(_fisher)
            if include_snip:
                results["snip"] = self._compute_safely(_snip)

            # Use the robust sensitivity implementation with finite-difference fallback.
            results["sensitivity"] = self.sensitivity(
                model,
                inputs,
                shared_outputs=shared_outputs,
                shared_inputs=shared_inputs,
            )

        finally:
            if not was_training:
                model.eval()
            # Jacobian uses torch.autograd.grad directly (not parameter .grad), so
            # clearing parameter grads here is safe; keep this ordering explicit.
            model.zero_grad()

        return results

    def _compute_synflow(self, model, inputs):
        """SynFlow score (original 2020 form): sum(|p * grad|)."""

        def _compute():
            was_training = model.training
            params = [p for p in model.parameters() if p.requires_grad]
            original_data = [p.detach().clone() for p in params]

            try:
                model.train()
                # Linearize model weights (abs) while preserving architecture-dependent magnitudes.
                with torch.no_grad():
                    for p in params:
                        p.abs_()

                x = torch.ones_like(inputs[: self.config.max_samples])
                model.zero_grad()
                out = self._unwrap_output(model(x))
                if out.dim() > 2:
                    out = out.flatten(1)
                out.sum().backward()

                score_sum = 0.0
                count = 0
                for p in model.parameters():
                    if p.grad is not None and p.requires_grad:
                        contrib = (p * p.grad).abs().sum().item()
                        if contrib > 0 and np.isfinite(contrib):
                            score_sum += contrib
                            count += 1

                if count == 0:
                    return 0.0

                # Return the raw SynFlow score; downstream normalization handles scaling.
                return float(score_sum)

            finally:
                # restore
                with torch.no_grad():
                    for p, p0 in zip(params, original_data):
                        p.copy_(p0)
                model.zero_grad()
                if not was_training:
                    model.eval()

        return self._compute_safely(_compute)

    def _compute_jacobian(
        self,
        model,
        inputs,
        shared_outputs: Optional[torch.Tensor] = None,
        shared_inputs: Optional[torch.Tensor] = None,
    ):
        """
        Jacobian trace approximation with multi-probe Hutchinson estimator.

        Improvements over one-probe baseline:
        - averages over multiple probes (lower variance)
        - random output-dimension sampling (avoids fixed first-dim bias)

        Returns log(Tr(JJ^T)/d_in).
        """

        def _compute():
            was_training = model.training
            model.train()
            bs = min(inputs.size(0), self.config.max_samples)
            try:
                can_reuse_shared = (
                    shared_inputs is not None
                    and shared_outputs is not None
                    and shared_inputs.requires_grad
                    and shared_outputs.requires_grad
                    and shared_inputs.size(0) >= bs
                )

                if can_reuse_shared:
                    x = cast(torch.Tensor, shared_inputs)[:bs]
                    out = cast(torch.Tensor, shared_outputs)[:bs]
                else:
                    x = inputs[:bs].detach().clone().requires_grad_(True)
                    with self.helper.safe_mode(model):
                        out = model(x)

                out = self._unwrap_output(out)

                if out is None or not out.requires_grad:
                    return self._finite_difference_jacobian(model, inputs)

                if out.dim() == 1:
                    out = out.unsqueeze(1)
                elif out.dim() > 2:
                    out = out.flatten(1)

                total_out = int(out.size(1))
                if total_out < 1:
                    return 0.0

                d_out = min(total_out, int(self.config.max_outputs))
                probes = max(1, int(getattr(self.config, "jacobian_probes", 2)))

                trace_vals: List[float] = []
                device = out.device

                for probe_idx in range(probes):
                    if d_out == total_out:
                        idx = torch.arange(total_out, device=device)
                    else:
                        idx = torch.randperm(total_out, device=device)[:d_out]

                    out_sel = out.index_select(1, idx)

                    # Rademacher probe tends to be stable and low-variance
                    v = torch.randint(
                        0, 2, out_sel.shape, device=device, dtype=torch.int8
                    )
                    v = (v.to(out_sel.dtype) * 2.0) - 1.0

                    (Jv,) = torch.autograd.grad(
                        out_sel,
                        x,
                        v,
                        retain_graph=(probe_idx < probes - 1),
                        create_graph=False,
                        allow_unused=True,
                    )

                    if Jv is None:
                        continue

                    trace_probe = (Jv.view(bs, -1) ** 2).sum(dim=1).mean().item()
                    if np.isfinite(trace_probe):
                        trace_vals.append(float(trace_probe))

                if not trace_vals:
                    return self._finite_difference_jacobian(model, inputs, d_out=d_out)

                trace_est = float(np.mean(trace_vals))
                d_in = max(int(x[0].numel()), 1)
                normalized = trace_est / (d_in + self.config.eps)
                return float(np.clip(np.log(normalized + self.config.eps), -12, 12))

            except Exception as e:
                # Robust fallback with fresh graph (avoids shared-graph lifetime issues).
                try:
                    x = inputs[:bs].detach().clone().requires_grad_(True)
                    with self.helper.safe_mode(model):
                        out = self._unwrap_output(model(x))

                    if out.dim() == 1:
                        out = out.unsqueeze(1)
                    elif out.dim() > 2:
                        out = out.flatten(1)

                    d_out = min(int(out.size(1)), int(self.config.max_outputs))
                    if d_out < 1:
                        return 0.0

                    out_sel = out[:, :d_out]
                    v = torch.randint(0, 2, out_sel.shape, device=out_sel.device)
                    v = (v.to(out_sel.dtype) * 2.0) - 1.0
                    (Jv,) = torch.autograd.grad(
                        out_sel,
                        x,
                        v,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    if Jv is None:
                        return 0.0
                    trace_est = (Jv.view(x.size(0), -1) ** 2).sum(dim=1).mean().item()
                    d_in = max(int(x[0].numel()), 1)
                    normalized = trace_est / (d_in + self.config.eps)
                    return float(np.clip(np.log(normalized + self.config.eps), -12, 12))
                except Exception:
                    print(f"Jacobian failed: {str(e)}")
                    return self._finite_difference_jacobian(model, inputs)

            finally:
                model.zero_grad()
                if not was_training:
                    model.eval()

        return self._compute_safely(_compute)

    def _compute_safely(self, compute_fn):
        try:
            start_time = time.time()
            value = compute_fn()
            elapsed = time.time() - start_time

            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    print(f"Metric failed due to nan/inf: {compute_fn.__name__}")
                    return Result(
                        0.0, False, "Numerical instability (nan/inf)", elapsed
                    )

                value = np.clip(value, -1e10, 1e10)

            return Result(float(value), True, "", elapsed)

        except Exception as e:
            print(f"Metric '{compute_fn.__name__}' failed with exception: {e}")
            return Result(0.0, False, str(e), 0.0)

    # Individual metric methods for compatibility
    def synflow(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """SynFlow metric"""
        return self._compute_synflow(model, inputs)

    def jacobian(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """Jacobian metric"""
        return self._compute_jacobian(model, inputs)

    def grasp(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Result:
        """GRASP metric"""
        return self._compute_gradient_metrics(model, inputs, targets)["grasp"]

    def fisher(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Result:
        """Fisher metric"""
        return self._compute_gradient_metrics(model, inputs, targets)["fisher"]

    def snip(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Result:
        """SNIP metric"""
        return self._compute_gradient_metrics(model, inputs, targets)["snip"]

    def params(self, model: nn.Module) -> Result:
        """Parameter count using fvcore if available"""

        def _compute():
            try:
                fvcore_nn = importlib.import_module("fvcore.nn")
                parameter_count = getattr(fvcore_nn, "parameter_count")
                count_dict = parameter_count(model)
                return sum(v for v in count_dict.values())
            except Exception:
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

        return self._compute_safely(_compute)

    def conditioning(self, model: nn.Module) -> Result:
        """Conditioning estimate using exact or iterative singular-value bounds."""

        def _compute():
            log_conditions = []
            every_n = max(
                1, int(getattr(self.config, "conditioning_every_n_layers", 1))
            )
            min_out = int(getattr(self.config, "conditioning_min_out_features", 0))
            iters = max(
                2,
                int(getattr(self.config, "conditioning_power_iters", 6)),
            )
            exact_max_dim = max(
                8, int(getattr(self.config, "conditioning_exact_max_dim", 256))
            )
            inverse_shift = max(
                float(getattr(self.config, "conditioning_inverse_shift", 1e-6)),
                self.config.eps,
            )
            eps = float(self.config.eps)
            layer_idx = 0
            for name, param in model.named_parameters():
                if "weight" in name and param.dim() >= 2 and param.requires_grad:
                    W = param.view(param.size(0), -1)
                    if layer_idx % every_n != 0:
                        layer_idx += 1
                        continue
                    if min_out > 0 and W.size(0) < min_out:
                        layer_idx += 1
                        continue
                    layer_idx += 1
                    if min(W.size()) > 1:
                        try:
                            # Use the smaller Gram matrix so we estimate non-zero singular values.
                            if W.size(0) <= W.size(1):
                                gram = W.matmul(W.t())
                            else:
                                gram = W.t().matmul(W)
                            gram = gram.to(dtype=torch.float64)
                            k = int(gram.size(0))
                            if k < 2:
                                continue

                            if k <= exact_max_dim:
                                eigvals = torch.linalg.eigvalsh(gram)
                                lambda_min = float(
                                    torch.clamp(eigvals[0], min=eps).item()
                                )
                                lambda_max = float(
                                    torch.clamp(eigvals[-1], min=eps).item()
                                )
                            else:
                                # Power iteration for largest eigenvalue.
                                v = torch.randn(k, device=gram.device, dtype=gram.dtype)
                                v = v / (v.norm() + eps)
                                for _ in range(iters):
                                    v = gram.matmul(v)
                                    v = v / (v.norm() + eps)
                                lambda_max = float(
                                    torch.clamp(torch.dot(v, gram.matmul(v)), min=eps).item()
                                )

                                # Shifted inverse iteration for smallest eigenvalue.
                                trace_mean = float(torch.trace(gram).item() / max(k, 1))
                                shift = max(inverse_shift * max(trace_mean, eps), eps)
                                eye = torch.eye(
                                    k,
                                    device=gram.device,
                                    dtype=gram.dtype,
                                )
                                shifted = gram + shift * eye

                                u = torch.randn(k, device=gram.device, dtype=gram.dtype)
                                u = u / (u.norm() + eps)
                                for _ in range(iters):
                                    try:
                                        u = torch.linalg.solve(shifted, u)
                                    except RuntimeError:
                                        u = torch.linalg.pinv(shifted).matmul(u)
                                    u = u / (u.norm() + eps)
                                lambda_min = float(
                                    torch.clamp(torch.dot(u, gram.matmul(u)), min=eps).item()
                                )

                            s_max = math.sqrt(lambda_max)
                            s_min = math.sqrt(lambda_min)
                            if math.isfinite(s_max) and math.isfinite(s_min):
                                cond = s_max / max(s_min, eps)
                                if cond > 0 and math.isfinite(cond):
                                    # log-scale for stability across very ill-conditioned layers
                                    log_cond = math.log(cond + eps)
                                    log_cond = float(np.clip(log_cond, 0.0, 30.0))
                                    log_conditions.append(log_cond)
                        except Exception:
                            continue
            return sum(log_conditions) / len(log_conditions) if log_conditions else 0.0

        return self._compute_safely(_compute)

    def flops(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """FLOP estimation using fvcore with fallback to manual hook-based count"""

        def _compute():
            try:
                fvcore_nn = importlib.import_module("fvcore.nn")
                FlopCountAnalysis = getattr(fvcore_nn, "FlopCountAnalysis")

                # Ensure input is in tuple format
                input_tuple = (inputs[:1].detach().clone(),)
                flops = FlopCountAnalysis(model, input_tuple)
                return flops.total()

            except Exception:
                # Fallback to original hook-based logic
                flops_count = {}

                def counting_hook(name):
                    def hook(module, inp, out):
                        input_shape = inp[0].shape
                        output_shape = (
                            out.shape if not isinstance(out, tuple) else out[0].shape
                        )

                        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                            kernel_ops = (
                                np.prod(module.kernel_size)
                                * module.in_channels
                                // module.groups
                            )
                            output_elements = np.prod(output_shape)
                            flops = output_elements * kernel_ops * 2
                        elif isinstance(module, nn.Linear):
                            flops = (
                                input_shape[0]
                                * module.in_features
                                * module.out_features
                                * 2
                            )
                        else:
                            flops = 0

                        flops_count[name] = flops

                    return hook

                hooks = []
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                        hooks.append(module.register_forward_hook(counting_hook(name)))

                try:
                    with torch.no_grad():
                        model(inputs[:1])  # only one sample needed
                    return sum(flops_count.values())
                finally:
                    for hook in hooks:
                        hook.remove()

        return self._compute_safely(_compute)

    def sensitivity(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        shared_outputs: Optional[torch.Tensor] = None,
        shared_inputs: Optional[torch.Tensor] = None,
    ) -> Result:
        """Input-gradient sensitivity (plain input influence signal)."""

        def _compute():
            was_training = model.training
            model.train()
            model.zero_grad()

            can_reuse_shared = (
                shared_inputs is not None
                and shared_outputs is not None
                and shared_inputs.requires_grad
                and shared_outputs.requires_grad
                and shared_inputs.shape == inputs.shape
            )

            if can_reuse_shared:
                x = cast(torch.Tensor, shared_inputs)
                output = cast(torch.Tensor, shared_outputs)
            else:
                x = inputs.clone().detach().requires_grad_(True)
                output = model(x)

            output = self._unwrap_output(output)

            try:
                # Gradient of output energy wrt input is a stable sensitivity proxy.
                if output.dim() == 1:
                    scalar = output.pow(2).mean()
                else:
                    scalar = output.flatten(1).pow(2).mean()
                scalar.backward()
            except RuntimeError as e:
                # Graph may have been consumed by another metric; fallback to local pass.
                warnings.warn(
                    f"Sensitivity shared-graph reuse failed; retrying with fresh pass: {e}",
                    RuntimeWarning,
                )
                model.zero_grad()
                x = inputs.clone().detach().requires_grad_(True)
                output = self._unwrap_output(model(x))
                scalar = (
                    output.flatten(1).pow(2).mean()
                    if output.dim() > 1
                    else output.pow(2).mean()
                )
                scalar.backward()

            # Input gradient norm only (standard, less custom than mixed proxy)
            input_grad_norm = (
                x.grad.norm(p=2, dim=tuple(range(1, x.grad.dim()))).mean().item()
                if x.grad is not None
                else 0.0
            )

            # Fallback: finite-difference sensitivity if gradient degenerates to ~0.
            if not np.isfinite(input_grad_norm) or input_grad_norm <= self.config.eps:
                input_grad_norm = self._finite_difference_sensitivity(model, inputs)

            model.zero_grad()
            if shared_inputs is None or x is not shared_inputs:
                x.requires_grad_(False)
            if not was_training:
                model.eval()

            return float(input_grad_norm)

        return self._compute_safely(_compute)


class ZeroCostNAS:
    """Main zero-cost NAS evaluation class"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.computer = MetricsComputer(self.config)

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a single model"""
        model = model.to(device)
        model.eval()

        batches = []
        for i, batch in enumerate(dataloader):
            # print(f"Loading batch {i + 1}...")
            if i >= num_batches:
                break

            inputs, targets = self._extract_inputs_targets(batch, model, device)
            inputs = inputs[: self.config.max_samples]
            targets = targets[: self.config.max_samples]
            batches.append((inputs, targets))

        # print(f"Computing metrics on {len(batches)} batches...")
        all_results = []
        model_only_results = self.computer.compute_model_only_metrics(model)
        heavy_batches = max(1, int(getattr(self.config, "heavy_metrics_batches", 1)))
        for i, (inputs, targets) in enumerate(batches):
            # if verbose:
            #    print(f"Processing batch {i+1}/{len(batches)}")
            batch_results = self.computer.compute_all(
                model,
                inputs,
                targets,
                include_heavy_metrics=(i < heavy_batches),
                model_only_results=model_only_results,
            )
            all_results.append(batch_results)
        # print("Aggregating results across batches...")
        final_results: Dict[str, Result] = self._aggregate_results(all_results)
        score = self._compute_score(final_results)

        return {
            "metrics": {k: r.value for k, r in final_results.items()},
            "success_rates": {k: r.success for k, r in final_results.items()},
            "error_messages": {
                k: r.error for k, r in final_results.items() if not r.success
            },
            "aggregate_score": score,
            "config": self.config,
        }

    def _extract_inputs_targets(self, batch, model, device):
        """Handles various batch formats and generates dummy targets if needed"""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0].to(device), batch[1].to(device)
        else:
            inputs = (
                batch[0].to(device)
                if isinstance(batch, (list, tuple))
                else batch.to(device)
            )
            with torch.no_grad():
                output = model(inputs[:1])
                if isinstance(output, tuple):
                    output = output[0]

                if output.dim() > 1 and output.size(1) > 1:
                    targets = torch.randint(
                        0, output.size(1), (inputs.size(0),), device=device
                    )
                else:
                    targets = (
                        torch.randint(0, 2, (inputs.size(0),), device=device)
                        if output.dim() > 1
                        else torch.randn(inputs.size(0), device=device)
                    )

        return inputs, targets

    def _aggregate_results(
        self, all_results: List[Dict[str, Result]]
    ) -> Dict[str, Result]:
        """Aggregate metric results across batches using sigma-clipped mean."""
        metrics = all_results[0].keys()
        aggregated = {}

        for metric in metrics:
            # print(f"Aggregating results for metric: {metric}")
            vals = [r[metric].value for r in all_results if r[metric].success]
            # print(f"Values for {metric}: {vals}")
            times = [r[metric].time for r in all_results]
            success = any(r[metric].success for r in all_results)
            avg_time = sum(times) / len(times) if times else 0.0

            agg_value = float("nan")
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                arr_kept = arr
                # Iterative sigma-clipping is more stable for tiny batch counts.
                for _ in range(5):
                    if arr_kept.size < 3:
                        break
                    mu = float(np.mean(arr_kept))
                    sd = float(np.std(arr_kept))
                    if not np.isfinite(sd) or sd <= 0:
                        break
                    keep = np.abs(arr_kept - mu) <= (2.5 * sd)
                    if np.all(keep):
                        break
                    new_arr = arr_kept[keep]
                    if new_arr.size == 0 or new_arr.size == arr_kept.size:
                        break
                    arr_kept = new_arr
                agg_value = float(np.mean(arr_kept)) if arr_kept.size else float("nan")

            is_nan = math.isnan(agg_value)

            aggregated[metric] = Result(
                value=0.0 if is_nan else agg_value,
                success=success and not is_nan,
                error=(
                    ""
                    if success and not is_nan
                    else f"{metric} resulted in NaN"
                    if is_nan
                    else "All batches failed"
                ),
                time=avg_time,
            )
        # print(f"Aggregated results: {aggregated}")

        return aggregated

    def _compute_score(self, results: Dict[str, Result]) -> float:
        """Compute weighted aggregate score"""
        total_score = 0.0
        total_weight = 0.0

        def _weight_for_metric(metric: str) -> Optional[float]:
            if metric in self.config.weights:
                return float(self.config.weights[metric])
            if metric == "activation_diversity" and "zennas" in self.config.weights:
                return float(self.config.weights["zennas"])
            if metric == "zennas" and "activation_diversity" in self.config.weights:
                return float(self.config.weights["activation_diversity"])
            return None

        for metric, result in results.items():
            if not result.success:
                continue

            weight = _weight_for_metric(metric)
            if weight is None:
                continue
            normalized = _normalize_metric_value_shared(metric, result.value)

            total_score += normalized * weight
            total_weight += abs(weight)

        return total_score / max(total_weight, 1.0)

    def evaluate_model_raw_metrics(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 3,
    ) -> Dict[str, Any]:
        """
        Compute raw metric values only (no weighting).
        Robust to individual metric failures, returning:
        - raw_metrics: aggregated raw values for metrics that succeeded at least once
        - success_rates: fraction of batches where each metric succeeded
        - errors: last error string seen for each metric (if any)
        """
        print("Evaluating raw metrics with robust error handling...")
        if isinstance(device, str):
            device = torch.device(device)

        model = model.to(device)
        model.eval()

        # ---- collect (inputs, targets) batches (same as evaluate_model)
        batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, batch in enumerate(dataloader):
            print(f"Extracting batch {i + 1} for raw metrics...")
            if i >= num_batches:
                break

            inputs, targets = self._extract_inputs_targets(batch, model, device)
            inputs = inputs[: self.config.max_samples]
            targets = targets[: self.config.max_samples]

            if inputs is None or targets is None:
                continue
            if inputs.numel() == 0 or targets.numel() == 0:
                continue

            batches.append((inputs, targets))

        if not batches:
            return {
                "raw_metrics": {},
                "success_rates": {},
                "errors": {"_global": "No valid batches after extraction/slicing."},
            }

        per_metric_values: Dict[str, List[float]] = {}
        per_metric_success: Dict[str, int] = {}
        per_metric_total: Dict[str, int] = {}
        per_metric_errors: Dict[str, str] = {}
        model_only_results = self.computer.compute_model_only_metrics(model)
        heavy_batches = max(1, int(getattr(self.config, "heavy_metrics_batches", 1)))

        # IMPORTANT: DO NOT use torch.no_grad() here (grad-based metrics need autograd)
        for batch_idx, (inputs, targets) in enumerate(batches):
            try:
                print(
                    f"Computing metrics for batch {batch_idx + 1}/{len(batches)} (heavy_metrics={batch_idx < heavy_batches})..."
                )
                results = self.computer.compute_all(
                    model,
                    inputs,
                    targets,
                    include_heavy_metrics=(batch_idx < heavy_batches),
                    model_only_results=model_only_results,
                )
            except Exception as e:
                per_metric_errors["_batch_compute_all"] = str(e)
                continue

            for name, res in results.items():
                per_metric_total[name] = per_metric_total.get(name, 0) + 1

                try:
                    if isinstance(res, Result):
                        val = float(res.value)
                    else:
                        val = float(res)

                    # Filter NaN/inf
                    if not torch.isfinite(torch.tensor(val)):
                        raise ValueError(f"Non-finite value: {val}")

                    per_metric_values.setdefault(name, []).append(val)
                    per_metric_success[name] = per_metric_success.get(name, 0) + 1

                except Exception as e:
                    per_metric_errors[name] = str(e)

        raw_metrics: Dict[str, float] = {
            name: float(sum(vals) / len(vals))
            for name, vals in per_metric_values.items()
            if len(vals) > 0
        }

        success_rates: Dict[str, float] = {
            name: float(per_metric_success.get(name, 0) / max(tot, 1))
            for name, tot in per_metric_total.items()
        }

        # Debug helper: if empty, print why (optional)
        if not raw_metrics:
            print("❌ evaluate_model_raw_metrics: all metrics failed.")
            for k, v in per_metric_errors.items():
                print(f"  {k}: {v}")

        return {
            "raw_metrics": raw_metrics,
            "success_rates": success_rates,
            "errors": per_metric_errors,
        }

    def score_from_metrics(
        self, metrics: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Compute weighted score from precomputed raw metrics."""
        return _score_from_metrics_shared(metrics, weights)

    @staticmethod
    def _normalize_metric_value(metric: str, value: float) -> float:
        """Normalize metric values on comparable scales while preserving signed signals."""
        return _normalize_metric_value_shared(metric, value)
