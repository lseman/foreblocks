import contextlib
import os
import threading
import time
import warnings
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from .activation_diversity import compute_activation_diversity
from .compatibility import CompatibilityHelper
from .conditioning import compute_conditioning
from .config import Config, Result
from .fisher import compute_fisher
from .flops import compute_activation_flops, module_flops
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
            or ("sdp" in msg
            and "derivative" in msg)
            or ("derivative for" in msg
            and "not implemented" in msg)
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
                output = out[0] if isinstance(out, tuple) else out
                flops_count[name] = flops_count.get(name, 0) + module_flops(module, output)

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
                        activations, conv_linear_modules, relu_modules, flops_count,
                        batch_size=inputs.size(0),
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
        self, activations, conv_linear_modules, relu_modules, flops_count,
        batch_size=None,
    ):
        """Compute metrics that only need stored activations."""
        results = {}
        naswot_modules = relu_modules if relu_modules else conv_linear_modules
        results["naswot"] = compute_naswot(
            self, activations, naswot_modules, batch_size=batch_size
        )
        results["activation_diversity"] = compute_activation_diversity(
            self, activations, relu_modules
        )
        if self.config.enable_flops:
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

    def params(self, model: nn.Module) -> Result:
        return compute_params(self, model)

    def conditioning(self, model: nn.Module) -> Result:
        return compute_conditioning(self, model)

    def sensitivity(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        shared_outputs: torch.Tensor | None = None,
        shared_inputs: torch.Tensor | None = None,
    ) -> Result:
        return compute_sensitivity(
            self,
            model,
            inputs,
            shared_outputs=shared_outputs,
            shared_inputs=shared_inputs,
        )
