import importlib

import numpy as np
import torch
import torch.nn as nn


def compute_activation_flops(computer, flops_count):
    """Return the hook-collected FLOP estimate used during shared forwards."""

    def _compute():
        total = sum(flops_count.values())
        return max(total, 1.0)

    return computer._compute_safely(_compute)


def compute_flops(computer, model: nn.Module, inputs: torch.Tensor):
    """FLOP estimation using fvcore with fallback to manual hook-based count."""

    def _compute():
        try:
            fvcore_nn = importlib.import_module("fvcore.nn")
            FlopCountAnalysis = getattr(fvcore_nn, "FlopCountAnalysis")
            input_tuple = (inputs[:1].detach().clone(),)
            flops = FlopCountAnalysis(model, input_tuple)
            return flops.total()
        except Exception:
            flops_count = {}

            def counting_hook(name):
                def hook(module, inp, out):
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
                    model(inputs[:1])
                return sum(flops_count.values())
            finally:
                for hook in hooks:
                    hook.remove()

    return computer._compute_safely(_compute)
