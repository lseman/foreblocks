import warnings
from typing import cast

import numpy as np
import torch


def compute_sensitivity(
    computer,
    model,
    inputs: torch.Tensor,
    shared_outputs: torch.Tensor | None = None,
    shared_inputs: torch.Tensor | None = None,
):
    """Input-gradient sensitivity with finite-difference fallback."""

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

        output = computer._unwrap_output(output)

        try:
            scalar = (
                output.flatten(1).pow(2).mean()
                if output.dim() > 1
                else output.pow(2).mean()
            )
            scalar.backward()
        except RuntimeError as e:
            warnings.warn(
                f"Sensitivity shared-graph reuse failed; retrying with fresh pass: {e}",
                RuntimeWarning,
            )
            model.zero_grad()
            x = inputs.clone().detach().requires_grad_(True)
            output = computer._unwrap_output(model(x))
            scalar = (
                output.flatten(1).pow(2).mean()
                if output.dim() > 1
                else output.pow(2).mean()
            )
            scalar.backward()

        input_grad_norm = (
            x.grad.norm(p=2, dim=tuple(range(1, x.grad.dim()))).mean().item()
            if x.grad is not None
            else 0.0
        )

        if not np.isfinite(input_grad_norm) or input_grad_norm <= computer.config.eps:
            input_grad_norm = computer._finite_difference_sensitivity(model, inputs)

        model.zero_grad()
        if shared_inputs is None or x is not shared_inputs:
            x.requires_grad_(False)
        if not was_training:
            model.eval()

        return float(input_grad_norm)

    return computer._compute_safely(_compute)
