import contextlib
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Config:
    """Unified configuration"""

    max_samples: int = 32
    max_outputs: int = 10
    eps: float = 1e-8
    timeout: float = 30.0
    enable_mixed_precision: bool = False
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
            "zennas": 0.10,
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
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        """Simplified SDPA fallback using matmul and masking"""
        scale = scale or (q.size(-1) ** -0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn += attn_mask
        elif is_causal:
            L = q.size(-2)
            causal_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=q.device), diagonal=1
            )
            attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v)

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

    def __init__(self, config):
        self.config = config
        self.helper = CompatibilityHelper()

    def compute_all(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor = None
    ) -> Dict[str, Result]:
        """Compute all metrics with shared hooks and minimal forward passes"""
        results = {}

        # Model-only metrics (no forward pass needed)
        results["params"] = self.params(model)
        results["conditioning"] = self.conditioning(model)

        # Shared activation collection for multiple metrics
        activations = {}
        conv_linear_modules = []
        relu_modules = []
        flops_count = {}

        # Single hook setup for all metrics that need activations
        def activation_hook(name):
            def hook(module, inp, out):
                # Store for NASWOT and Zen-NAS
                activations[name] = out[0] if isinstance(out, tuple) else out

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

            # with torch.no_grad():
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
                results.update(self._compute_gradient_metrics(model, inputs, targets))

            # SynFlow (needs special handling)
            results["synflow"] = self._compute_synflow(model, inputs)

            # Jacobian (needs special handling)
            results["jacobian"] = self._compute_jacobian(model, inputs)

            # Sensitivity
            results["sensitivity"] = self.sensitivity(model, inputs, shared_outputs)

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
        # NASWOT ─ corrected version (logdet of the NTK-like kernel)
        # Most common in recent zero-cost NAS literature
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
                    flat = act.flatten(1)  # [B, C*H*W]
                    binary = (flat > 0).to(flat.dtype)  # binarized ReLU
                    K = binary @ binary.t()  # [B,B]
                    K += self.config.eps * torch.eye(K.size(0), device=K.device)

                    sign, logdet = torch.slogdet(K)
                    if sign.item() > 0 and torch.isfinite(logdet):
                        total_logdet += logdet.item()
                        valid_layers += 1
                except RuntimeError:  # singular / numerical issues
                    continue

            if valid_layers == 0:
                return 0.0
            return total_logdet / valid_layers

        results["naswot"] = self._compute_safely(_naswot)

        # ────────────────────────────────────────────────
        # Zen-NAS ─ corrected version (cosine similarity based)
        # Closer to the original Zen-NAS score intention
        # ────────────────────────────────────────────────
        def _zennas():
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
                    eye = torch.eye(cos.size(0), device=cos.device)
                    off_diag = cos.masked_fill(eye.bool(), 0.0)

                    # Average positive cosine similarity (common proxy)
                    pos_cos = off_diag.clamp_min(0.0)
                    score = pos_cos.sum() / (cos.numel() - cos.size(0))

                    if torch.isfinite(score):
                        total_score += score.item()
                        valid_layers += 1
                except:
                    continue

            if valid_layers == 0:
                return 0.0
            return total_score / valid_layers

        results["zennas"] = self._compute_safely(_zennas)

        # FLOPS remains unchanged (already reasonable)
        def _flops():
            total = sum(flops_count.values())
            return max(total, 1.0)

        results["flops"] = self._compute_safely(_flops)

        return results

    def _compute_gradient_metrics(self, model, inputs, targets):
        """Compute gradient-based metrics (GRASP, Fisher, SNIP) with a shared forward+backward"""
        results = {}
        was_training = model.training
        model.train()

        try:
            x, y = inputs.clone().detach(), targets.clone().detach()
            loss_fn = self.helper.get_loss_fn(y)

            x.requires_grad = True
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

            grads = torch.autograd.grad(
                loss,
                weights,
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )

            # ----- GRASP -----
            def _grasp():
                scores = [
                    (g * p).pow(2).sum().item()
                    for g, p in zip(grads, weights)
                    if g is not None and torch.isfinite(g).all()
                ]
                return sum(scores) / max(len(scores), 1)

            # ----- Fisher -----
            def _fisher():
                scores = [
                    g.pow(2).sum().item()
                    for g in grads
                    if g is not None and torch.isfinite(g).all()
                ]
                return sum(scores) / max(len(scores), 1)

            # ----- SNIP -----
            def _snip():
                snip_scores = []
                for (name, p), g in zip(weight_params, grads):
                    if "weight" in name and g is not None and torch.isfinite(g).all():
                        snip_scores.append((g * p).abs().sum().item())
                return sum(snip_scores) / max(len(snip_scores), 1)

            results["grasp"] = self._compute_safely(_grasp)
            results["fisher"] = self._compute_safely(_fisher)
            results["snip"] = self._compute_safely(_snip)

        finally:
            if not was_training:
                model.eval()
            model.zero_grad()

        return results

    def _compute_synflow(self, model, inputs):
        """SynFlow – changed to sum of log |param · grad| (more common)"""

        def _compute():
            was_training = model.training
            original_data = {
                n: p.data.clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }

            try:
                model.train()
                # Set all weights to 1
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.fill_(1.0)

                x = torch.ones_like(inputs[: self.config.max_samples])
                model.zero_grad()
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                out.sum().backward()

                log_sum = 0.0
                count = 0
                for p in model.parameters():
                    if p.grad is not None and p.requires_grad:
                        contrib = (p * p.grad).abs().sum().item()
                        if contrib > 0 and np.isfinite(contrib):
                            log_sum += np.log(contrib + self.config.eps)
                            count += 1

                if count == 0:
                    return 0.0

                # Most papers use sum(log) rather than mean(log)
                # → equivalent to log of product of absolutes
                return np.clip(log_sum, -100.0, 100.0)

            finally:
                # restore
                for n, p in model.named_parameters():
                    if n in original_data:
                        p.data.copy_(original_data[n])
                model.zero_grad()
                if not was_training:
                    model.eval()

        return self._compute_safely(_compute)

    def _compute_jacobian(self, model, inputs, shared_outputs=None):
        """
        Improved Jacobian trace approximation (Hutchinson + better normalization)
        Returns log(Tr(JJᵀ)/d_out) style quantity — more comparable across models
        """

        def _compute():
            was_training = model.training
            model.train()
            try:
                bs = min(inputs.size(0), self.config.max_samples)
                x = inputs[:bs].detach().clone().requires_grad_(True)

                with self.helper.safe_mode(model):
                    out = shared_outputs if shared_outputs is not None else model(x)

                if out is None or not out.requires_grad:
                    return 0.0

                if out.dim() == 1:
                    out = out.unsqueeze(1)

                d_out = min(out.size(1), self.config.max_outputs)
                v = torch.randn_like(out[:, :d_out])

                # Jv = ∇_x (out · v)
                (Jv,) = torch.autograd.grad(out[:, :d_out], x, v, retain_graph=False)

                if Jv is None:
                    return 0.0

                # Estimate Tr(JJᵀ) ≈ ||Jv||²  (Hutchinson — one vector here for speed)
                trace_est = (Jv.view(bs, -1) ** 2).sum(dim=1).mean().item()

                # Normalize by output dimension + log form
                normalized = trace_est / (d_out + self.config.eps)
                return float(np.clip(np.log(normalized + self.config.eps), -12, 12))

            except Exception as e:
                print(f"Jacobian failed: {str(e)}")
                return 0.0

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
                from fvcore.nn import parameter_count

                count_dict = parameter_count(model)
                return sum(v for v in count_dict.values())
            except Exception:
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

        return self._compute_safely(_compute)

    def conditioning(self, model: nn.Module) -> Result:
        """Weight conditioning with numerical stability"""

        def _compute():
            conditions = []
            for name, param in model.named_parameters():
                if "weight" in name and param.dim() >= 2 and param.requires_grad:
                    W = param.view(param.size(0), -1)
                    if min(W.size()) > 1:
                        try:
                            _, S, _ = torch.linalg.svd(W, full_matrices=False)
                            s_max = S[0].item()
                            s_min = S[-1].item()
                            if math.isfinite(s_max) and math.isfinite(s_min):
                                cond = s_max / (s_min + self.config.eps)
                                cond = min(
                                    cond, 1e6
                                )  # Optional clamp to avoid outliers
                                conditions.append(cond)
                        except Exception:
                            continue
            return sum(conditions) / len(conditions) if conditions else 1.0

        return self._compute_safely(_compute)

    def flops(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """FLOP estimation using fvcore with fallback to manual hook-based count"""

        def _compute():
            try:
                from fvcore.nn import FlopCountAnalysis

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
    ) -> Result:
        """Input sensitivity"""

        def _compute():
            model.train()
            model.zero_grad()

            x = inputs.clone().detach().requires_grad_(True)
            output = model(x)
            target = torch.randn_like(output)
            loss = F.mse_loss(output, target)
            loss.backward()

            # Input gradient norm
            input_grad_norm = (
                x.grad.norm(p=2, dim=tuple(range(1, x.grad.dim()))).mean().item()
                if x.grad is not None
                else 0.0
            )

            # Parameter gradient norm
            param_grad_sum = sum(
                p.grad.abs().sum().item()
                for p in model.parameters()
                if p.grad is not None and p.requires_grad
            )
            param_count = sum(
                p.numel()
                for p in model.parameters()
                if p.grad is not None and p.requires_grad
            )
            param_sensitivity = param_grad_sum / max(param_count, 1)

            model.zero_grad()
            x.requires_grad_(False)

            return param_sensitivity + input_grad_norm

        return self._compute_safely(_compute)


class ZeroCostNAS:
    """Main zero-cost NAS evaluation class"""

    def __init__(self, config: Config = None):
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
            if i >= num_batches:
                break

            inputs, targets = self._extract_inputs_targets(batch, model, device)
            inputs = inputs[: self.config.max_samples]
            targets = targets[: self.config.max_samples]
            batches.append((inputs, targets))

        all_results = []
        for i, (inputs, targets) in enumerate(batches):
            # if verbose:
            #    print(f"Processing batch {i+1}/{len(batches)}")
            batch_results = self.computer.compute_all(model, inputs, targets)
            all_results.append(batch_results)

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
        """Aggregate metric results across batches"""
        metrics = all_results[0].keys()
        aggregated = {}

        for metric in metrics:
            # print(f"Aggregating results for metric: {metric}")
            vals = [r[metric].value for r in all_results if r[metric].success]
            # print(f"Values for {metric}: {vals}")
            times = [r[metric].time for r in all_results]
            success = any(r[metric].success for r in all_results)
            avg_time = sum(times) / len(times) if times else 0.0

            median_value = float(np.median(vals)) if vals else float("nan")
            is_nan = math.isnan(median_value)

            aggregated[metric] = Result(
                value=0.0 if is_nan else median_value,
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

        for metric, result in results.items():
            if not result.success or metric not in self.config.weights:
                continue

            weight = self.config.weights[metric]

            # Normalize values
            if metric in {"synflow", "params", "flops"}:
                normalized = np.log1p(result.value)
                # check for nan
                if np.isnan(normalized):
                    normalized = 0.0
            elif metric == "conditioning":
                normalized = min(result.value, 100.0) / 100.0
            else:
                normalized = result.value

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
        if isinstance(device, str):
            device = torch.device(device)

        model = model.to(device)
        model.eval()

        # ---- collect (inputs, targets) batches (same as evaluate_model)
        batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
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

        per_metric_values: Dict[str, List[float]] = {}
        per_metric_success: Dict[str, int] = {}
        per_metric_total: Dict[str, int] = {}
        per_metric_errors: Dict[str, str] = {}

        # IMPORTANT: DO NOT use torch.no_grad() here (grad-based metrics need autograd)
        for inputs, targets in batches:
            try:
                results = self.computer.compute_all(model, inputs, targets)
            except Exception as e:
                per_metric_errors["_batch_compute_all"] = str(e)
                continue

            for name, res in results.items():
                per_metric_total[name] = per_metric_total.get(name, 0) + 1

                try:
                    val = float(res.value) if hasattr(res, "value") else float(res)

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
        total_score = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            if metric not in weights:
                continue
            w = weights[metric]

            # keep your existing normalization rules
            if metric in {"synflow", "params", "flops"}:
                norm = np.log1p(value)
                if np.isnan(norm):
                    norm = 0.0
            elif metric == "conditioning":
                norm = min(value, 100.0) / 100.0
            else:
                norm = value

            total_score += norm * w
            total_weight += abs(w)

        return total_score / max(total_weight, 1.0)
