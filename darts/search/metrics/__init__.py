import contextlib
import importlib
import math
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..scoring import (
    normalize_metric_value as _normalize_metric_value_shared,
)
from ..scoring import (
    score_from_metrics as _score_from_metrics_shared,
)
from .activation_diversity import compute_activation_diversity
from .conditioning import compute_conditioning
from .fisher import compute_fisher
from .flops import compute_activation_flops, compute_flops
from .grasp import compute_grasp
from .jacobian import compute_jacobian
from .naswot import compute_naswot
from .params import compute_params
from .sensitivity import compute_sensitivity
from .snip import compute_snip
from .synflow import compute_synflow

warnings.filterwarnings("ignore", category=UserWarning)


# Phase-1 zero-cost evaluation runs many candidates concurrently in a thread
# pool (see search/multi_fidelity.py). Each candidate's metrics issue CUDA
# forward/backward work; running them in parallel oversubscribes a single GPU,
# so each candidate's kernels get starved by its siblings and a normally ~8s
# evaluation stretches past the timeout. Serialising GPU access with this lock
# lets each candidate run at full speed in turn — total throughput is the same
# (the GPU is the bottleneck) but no candidate stalls. The lock is a no-op cost
# on CPU-only runs.
_ZC_GPU_LOCK = threading.Lock()


# Env-gated stage tracer for diagnosing Phase-1 hangs. When FORE_ZC_TRACE=1,
# each stage of compute_all prints "[ZC] >> <stage>" on entry (so a hang leaves
# the stalling stage as the last line) and "[ZC] << <stage> (Ns)" on exit. The
# cuda.synchronize() ensures async kernel time is attributed to the right stage
# rather than leaking into the next one. Zero overhead when the env var is unset.
@contextlib.contextmanager
def _zc_trace(stage: str):
    if os.environ.get("FORE_ZC_TRACE") != "1":
        yield
        return
    print(f"[ZC] >> {stage}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[ZC] << {stage} ({time.perf_counter() - t0:.3f}s)", flush=True)


@dataclass
class Config:
    """Unified configuration"""

    max_samples: int = 32
    max_outputs: int = 10
    eps: float = 1e-8
    # NASWOT builds an [R, R] kernel from each layer's activation. For
    # transformer/MoE layers R = batch*seq (can be thousands); cap it so slogdet
    # stays cheap and bounded. Layers with R > 2*features are skipped as
    # rank-deficient (see metrics/naswot.py).
    naswot_max_rows: int = 256
    # Per-metric wall-clock timeout enforced via a daemon thread in
    # MetricsComputer._compute_safely. If a metric computation (e.g. GRASP
    # second-order backward) stalls in a CUDA kernel, the thread is abandoned
    # and the metric returns a failed Result so evaluation can proceed.
    timeout: float = 30.0
    enable_mixed_precision: bool = False
    # One probe is standard for proxy NAS and sufficient for candidate ranking;
    # more probes reduce Hutchinson estimator variance at the cost of extra backward passes.
    jacobian_probes: int = 1
    # SNIP is defined at initialization; keep this as the default behavior.
    snip_at_init: bool = True
    # Explicit mode: "current" is faster (no weight reset, no extra forward/backward
    # at init) and avoids hangs from reset_parameters on custom layers.
    # Use "init" only if paper-consistent SNIP scores are needed.
    snip_mode: str = "current"
    heavy_metrics_batches: int = 1
    gradient_max_samples: int = 4
    # False: uses the already-computed batch gradient squared (free, no extra
    # forward/backward passes). True: runs one forward+backward per sample
    # (more accurate diagonal Fisher, but O(N) passes extra).
    fisher_per_sample: bool = False
    enable_grasp: bool = True
    enable_jacobian: bool = True
    enable_synflow: bool = True
    conditioning_every_n_layers: int = 3
    conditioning_min_out_features: int = 0
    conditioning_power_iters: int = 6
    conditioning_exact_max_dim: int = 64
    conditioning_inverse_shift: float = 1e-6
    weights: dict[str, float] = field(
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




# ─── Lazy presets ───────────────────────────────────────────────────────
# Built once on first access to avoid recursion during class definition.

def _get_presets() -> dict[str, "Config"]:
    """Return the named preset dict (built lazily on first call)."""
    return {
        "full": Config(),
        "smart_fast": Config(
            max_samples=32,
            max_outputs=10,
            jacobian_probes=1,
            gradient_max_samples=4,
            fisher_per_sample=False,
            enable_grasp=False,
            enable_jacobian=False,
            enable_synflow=True,
            snip_mode="current",
            conditioning_every_n_layers=3,
            heavy_metrics_batches=1,
            weights={
                "synflow": 0.20,
                "grasp": 0.0,
                "fisher": 0.18,
                "jacobian": 0.0,
                "naswot": 0.15,
                "snip": 0.18,
                "params": -0.05,
                "flops": -0.05,
                "conditioning": -0.05,
                "sensitivity": 0.12,
                "activation_diversity": 0.07,
            },
        ),
        "ultra_fast": Config(
            max_samples=16,
            max_outputs=5,
            jacobian_probes=1,
            gradient_max_samples=2,
            fisher_per_sample=False,
            enable_grasp=False,
            enable_jacobian=False,
            enable_synflow=True,
            snip_mode="current",
            conditioning_every_n_layers=10,
            heavy_metrics_batches=1,
            weights={
                "synflow": 0.30,
                "grasp": 0.0,
                "fisher": 0.0,
                "jacobian": 0.0,
                "naswot": 0.0,
                "snip": 0.25,
                "params": -0.10,
                "flops": 0.0,
                "conditioning": 0.0,
                "sensitivity": 0.0,
                "activation_diversity": 0.0,
            },
        ),
    }


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
                outputs = outputs.reshape(outputs.size(0), -1)
            if targets.ndim == 2 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            if outputs.size(-1) == 1 and targets.max() <= 1:
                outputs = outputs.squeeze(-1)
        else:
            # Regression: force shape match
            if outputs.shape != targets.shape:
                try:
                    outputs = outputs.reshape_as(targets)
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
    def _is_backend_double_backward_error(err: RuntimeError) -> bool:
        """Detect backend limitations that break second-order gradients."""
        msg = str(err).lower()
        return (
            "_cudnn_rnn_backward" in msg
            or "double backwards is not supported for cudnn rnns" in msg
            or "scaled_dot_product" in msg
            or "flash_attention" in msg
            or "efficient_attention" in msg
            or "sdp" in msg
            and "derivative" in msg
            or "derivative for" in msg
            and "not implemented" in msg
        )

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

    @staticmethod
    def _has_cudnn_rnn_modules(model: nn.Module) -> bool:
        for module in model.modules():
            if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                return True
        return False

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
        self, model, inputs: torch.Tensor, d_out: int | None = None
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
            u_norm = (
                u.reshape(bs, -1).norm(dim=1, keepdim=True).clamp_min(self.config.eps)
            )
            u = u / u_norm.reshape([bs] + [1] * (x.dim() - 1))

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

    def compute_model_only_metrics(self, model: nn.Module) -> dict[str, Result]:
        return {
            "params": self.params(model),
            "conditioning": self.conditioning(model),
        }

    def compute_all(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        include_heavy_metrics: bool = True,
        model_only_results: dict[str, Result] | None = None,
    ) -> dict[str, Result]:
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
            elif is_relu_like(module):
                relu_modules.append((module_name, module))
                hooks.append(module.register_forward_hook(activation_hook(module_name)))

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
                # Keep CuDNN enabled for speed by default.
                # GRASP handles CuDNN double-backward via retry path in _grasp.
                with _zc_trace("shared_forward(train)"):
                    shared_outputs = model(shared_inputs)
            else:
                with torch.no_grad(), _zc_trace("shared_forward(eval)"):
                    shared_outputs = model(inputs)

            # Process all metrics that only need activations
            with _zc_trace("activation_metrics"):
                results.update(
                    self._compute_activation_metrics(
                        activations, conv_linear_modules, relu_modules, flops_count
                    )
                )

            # Metrics requiring gradients (separate forward passes with minimal overhead)
            if targets is not None:
                with _zc_trace("gradient_metrics"):
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
            if include_heavy_metrics and bool(
                getattr(self.config, "enable_jacobian", True)
            ):
                with _zc_trace("jacobian"):
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
                    "Skipped (include_heavy_metrics=False or enable_jacobian=False)",
                    0.0,
                )

            # SynFlow (independent, runs after graph-dependent metrics)
            if include_heavy_metrics and bool(
                getattr(self.config, "enable_synflow", True)
            ):
                with _zc_trace("synflow"):
                    results["synflow"] = self._compute_synflow(model, inputs)
            else:
                results["synflow"] = Result(
                    0.0,
                    False,
                    "Skipped (include_heavy_metrics=False or enable_synflow=False)",
                    0.0,
                )

            # Sensitivity (prefer shared gradient pass when available)
            if "sensitivity" not in results:
                with _zc_trace("sensitivity"):
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
        """Compute metrics that only need stored activations."""
        results = {}
        naswot_modules = relu_modules if relu_modules else conv_linear_modules
        results["naswot"] = compute_naswot(self, activations, naswot_modules)
        results["activation_diversity"] = compute_activation_diversity(
            self, activations, relu_modules
        )
        results["flops"] = compute_activation_flops(self, flops_count)
        return results

    def _compute_gradient_metrics(
        self,
        model,
        inputs,
        targets,
        include_snip: bool = True,
        shared_inputs: torch.Tensor | None = None,
        shared_outputs: torch.Tensor | None = None,
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
            with _zc_trace("grad_metrics:first_order_backward"):
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
                    "init"
                    if bool(getattr(self.config, "snip_at_init", True))
                    else "current"
                )

            if bool(getattr(self.config, "enable_grasp", True)):
                results["grasp"] = self._compute_safely(
                    lambda: compute_grasp(self, model, x, y, loss, loss_fn, weights)
                )
            else:
                results["grasp"] = Result(
                    0.0, False, "Skipped (enable_grasp=False)", 0.0
                )
            results["fisher"] = self._compute_safely(
                lambda: compute_fisher(
                    self, model, x, y, loss_fn, weights, grads_first_order
                )
            )
            # Reuse the live shared graph before SNIP optionally resets weights.
            results["sensitivity"] = self.sensitivity(
                model,
                inputs,
                shared_outputs=shared_outputs,
                shared_inputs=shared_inputs,
            )
            if include_snip:
                results["snip"] = self._compute_safely(
                    lambda: compute_snip(
                        self,
                        model,
                        x,
                        y,
                        loss_fn,
                        weight_params,
                        grads_first_order,
                        snip_mode,
                    )
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
        return compute_synflow(self, model, inputs)

    def _compute_jacobian(
        self,
        model,
        inputs,
        shared_outputs: torch.Tensor | None = None,
        shared_inputs: torch.Tensor | None = None,
    ):
        """Jacobian trace approximation with multi-probe Hutchinson estimator."""
        return compute_jacobian(
            self,
            model,
            inputs,
            shared_outputs=shared_outputs,
            shared_inputs=shared_inputs,
        )

    def _compute_safely(self, compute_fn):
        """Run ``compute_fn()`` with a hard wall-clock timeout.

        Each metric is executed in a daemon thread.  If it does not finish
        within ``self.config.timeout`` seconds (e.g. GRASP second-order
        backward stalls in a CUDA kernel) the method returns a failed Result
        immediately, allowing the rest of the evaluation to continue and
        releasing ``_ZC_GPU_LOCK`` so the next candidate can proceed.
        The abandoned thread eventually terminates on its own.
        """
        name = getattr(compute_fn, "__name__", "metric")
        timeout = float(getattr(self.config, "timeout", 0.0) or 0.0)
        start_time = time.time()

        if timeout > 0:
            _result: list[Any] = [None]
            _exc: list[BaseException | None] = [None]

            def _run() -> None:
                try:
                    _result[0] = compute_fn()
                except Exception as exc:
                    _exc[0] = exc

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=timeout)
            elapsed = time.time() - start_time

            if t.is_alive():
                warnings.warn(
                    f"Zero-cost metric '{name}' timed out after {timeout:.0f}s; "
                    "skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return Result(0.0, False, f"timeout after {timeout:.0f}s", elapsed)

            if _exc[0] is not None:
                return Result(0.0, False, str(_exc[0]), elapsed)

            value = _result[0]

        else:
            try:
                value = compute_fn()
            except Exception as e:
                return Result(0.0, False, str(e), 0.0)
            elapsed = time.time() - start_time

        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return Result(0.0, False, "Numerical instability (nan/inf)", elapsed)
            value = np.clip(value, -1e10, 1e10)

        return Result(float(value), True, "", elapsed)

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
        """Parameter count using fvcore if available."""
        return compute_params(self, model)

    def conditioning(self, model: nn.Module) -> Result:
        """Conditioning estimate using exact or iterative singular-value bounds."""
        return compute_conditioning(self, model)

    def flops(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """FLOP estimation using fvcore with fallback to manual hook count."""
        return compute_flops(self, model, inputs)

    def sensitivity(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        shared_outputs: torch.Tensor | None = None,
        shared_inputs: torch.Tensor | None = None,
    ) -> Result:
        """Input-gradient sensitivity (plain input influence signal)."""
        return compute_sensitivity(
            self,
            model,
            inputs,
            shared_outputs=shared_outputs,
            shared_inputs=shared_inputs,
        )


class ZeroCostNAS:
    """Main zero-cost NAS evaluation class"""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.computer = MetricsComputer(self.config)

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 3,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a single model"""
        # Serialise GPU access so concurrent candidate threads don't
        # oversubscribe the device (see _ZC_GPU_LOCK).
        with _ZC_GPU_LOCK:
            model = model.to(device)
            model.eval()

            batches = []
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                inputs, targets = self._extract_inputs_targets(batch, model, device)
                inputs = inputs[: self.config.max_samples]
                targets = targets[: self.config.max_samples]
                batches.append((inputs, targets))

            if not batches:
                return {
                    "metrics": {},
                    "success_rates": {},
                    "error_messages": {
                        "_global": "No valid batches after extraction/slicing."
                    },
                    "aggregate_score": float("-inf"),
                    "config": self.config,
                }

            all_results = []
            model_only_results = self.computer.compute_model_only_metrics(model)
            heavy_batches = max(
                1, int(getattr(self.config, "heavy_metrics_batches", 1))
            )
            for i, (inputs, targets) in enumerate(batches):
                batch_results = self.computer.compute_all(
                    model,
                    inputs,
                    targets,
                    include_heavy_metrics=(i < heavy_batches),
                    model_only_results=model_only_results,
                )
                all_results.append(batch_results)
            final_results: dict[str, Result] = self._aggregate_results(all_results)
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
        self, all_results: list[dict[str, Result]]
    ) -> dict[str, Result]:
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

    def _compute_score(self, results: dict[str, Result]) -> float:
        """Compute weighted aggregate score"""
        total_score = 0.0
        total_weight = 0.0

        def _weight_for_metric(metric: str) -> float | None:
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
    ) -> dict[str, Any]:
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

        per_metric_values: dict[str, list[float]] = {}
        per_metric_success: dict[str, int] = {}
        per_metric_total: dict[str, int] = {}
        per_metric_errors: dict[str, str] = {}

        # Serialise GPU access across concurrent candidate threads (see
        # _ZC_GPU_LOCK). Held across batch extraction + metric computation; the
        # pure-CPU aggregation below runs outside the lock.
        with _ZC_GPU_LOCK:
            model = model.to(device)
            model.eval()

            # ---- collect (inputs, targets) batches (same as evaluate_model)
            batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            for i, batch in enumerate(dataloader):
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

            model_only_results = self.computer.compute_model_only_metrics(model)
            heavy_batches = max(
                1, int(getattr(self.config, "heavy_metrics_batches", 1))
            )

            # IMPORTANT: DO NOT use torch.no_grad() here (grad-based metrics
            # need autograd)
            for batch_idx, (inputs, targets) in enumerate(batches):
                try:
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

                self._accumulate_raw_batch(
                    results,
                    per_metric_values,
                    per_metric_success,
                    per_metric_total,
                    per_metric_errors,
                )

        return self._finalize_raw_metrics(
            per_metric_values, per_metric_success, per_metric_total, per_metric_errors
        )

    @staticmethod
    def _accumulate_raw_batch(
        results,
        per_metric_values,
        per_metric_success,
        per_metric_total,
        per_metric_errors,
    ) -> None:
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

    @staticmethod
    def _finalize_raw_metrics(
        per_metric_values, per_metric_success, per_metric_total, per_metric_errors
    ) -> dict[str, Any]:
        raw_metrics: dict[str, float] = {
            name: float(sum(vals) / len(vals))
            for name, vals in per_metric_values.items()
            if len(vals) > 0
        }

        success_rates: dict[str, float] = {
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
        self, metrics: dict[str, float], weights: dict[str, float]
    ) -> float:
        """Compute weighted score from precomputed raw metrics."""
        return _score_from_metrics_shared(metrics, weights)

    @staticmethod
    def _normalize_metric_value(metric: str, value: float) -> float:
        """Normalize metric values on comparable scales while preserving signed signals."""
        return _normalize_metric_value_shared(metric, value)
