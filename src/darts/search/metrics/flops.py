import numpy as np
import torch
import torch.nn as nn


def module_flops(module: nn.Module, output: torch.Tensor) -> int:
    """Count multiply and add operations per example for linear and conv layers."""
    output_elements = int(output.numel() // max(output.shape[0], 1))
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        kernel_ops = int(np.prod(module.kernel_size)) * module.in_channels // module.groups
        return 2 * output_elements * kernel_ops
    if isinstance(module, nn.Linear):
        return 2 * output_elements * module.in_features
    return 0


def compute_activation_flops(computer, flops_count):
    """Return the hook-collected FLOP estimate used during shared forwards."""

    def _compute():
        total = sum(flops_count.values())
        return float(total)

    return computer._compute_safely(_compute)


def compute_flops(computer, model: nn.Module, inputs: torch.Tensor):
    """Count supported layer operations per example, matching shared hooks."""

    def _compute():
        flops_count = {}

        def counting_hook(name):
            def hook(module, inp, out):
                output = out[0] if isinstance(out, tuple) else out
                flops_count[name] = flops_count.get(name, 0) + module_flops(module, output)

            return hook

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                hooks.append(module.register_forward_hook(counting_hook(name)))

        try:
            with torch.no_grad():
                model(inputs[:1])
            return sum(flops_count.values())
        finally:
            for hook in hooks:
                hook.remove()

    return computer._compute_safely(_compute)
