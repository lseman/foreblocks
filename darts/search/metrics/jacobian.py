from typing import cast

import numpy as np
import torch


def compute_jacobian(
    computer,
    model,
    inputs,
    shared_outputs: torch.Tensor | None = None,
    shared_inputs: torch.Tensor | None = None,
):
    """Jacobian trace approximation with multi-probe Hutchinson estimator."""

    def _compute():
        was_training = model.training
        model.train()
        bs = min(inputs.size(0), computer.config.max_samples)
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
                with computer.helper.safe_mode(model):
                    out = model(x)

            out = computer._unwrap_output(out)

            if out is None or not out.requires_grad:
                return computer._finite_difference_jacobian(model, inputs)

            if out.dim() == 1:
                out = out.unsqueeze(1)
            elif out.dim() > 2:
                out = out.flatten(1)

            total_out = int(out.size(1))
            if total_out < 1:
                return 0.0

            d_out = min(total_out, int(computer.config.max_outputs))
            probes = max(1, int(getattr(computer.config, "jacobian_probes", 2)))
            trace_vals: list[float] = []
            device = out.device

            for probe_idx in range(probes):
                if d_out == total_out:
                    idx = torch.arange(total_out, device=device)
                else:
                    idx = torch.randperm(total_out, device=device)[:d_out]

                out_sel = out.index_select(1, idx)
                v = torch.randint(0, 2, out_sel.shape, device=device, dtype=torch.int8)
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

                trace_probe = (Jv.reshape(bs, -1) ** 2).sum(dim=1).mean().item()
                if np.isfinite(trace_probe):
                    trace_vals.append(float(trace_probe))

            if not trace_vals:
                return computer._finite_difference_jacobian(model, inputs, d_out=d_out)

            trace_est = float(np.mean(trace_vals))
            d_in = max(int(x[0].numel()), 1)
            normalized = trace_est / (d_in + computer.config.eps)
            return float(np.clip(np.log(normalized + computer.config.eps), -12, 12))

        except Exception as e:
            try:
                x = inputs[:bs].detach().clone().requires_grad_(True)
                with computer.helper.safe_mode(model):
                    out = computer._unwrap_output(model(x))

                if out.dim() == 1:
                    out = out.unsqueeze(1)
                elif out.dim() > 2:
                    out = out.flatten(1)

                d_out = min(int(out.size(1)), int(computer.config.max_outputs))
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
                trace_est = (
                    (Jv.reshape(x.size(0), -1) ** 2).sum(dim=1).mean().item()
                )
                d_in = max(int(x[0].numel()), 1)
                normalized = trace_est / (d_in + computer.config.eps)
                return float(np.clip(np.log(normalized + computer.config.eps), -12, 12))
            except Exception:
                print(f"Jacobian failed: {str(e)}")
                return computer._finite_difference_jacobian(model, inputs)

        finally:
            model.zero_grad()
            if not was_training:
                model.eval()

    return computer._compute_safely(_compute)
