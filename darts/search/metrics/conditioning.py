import math

import numpy as np
import torch


def compute_conditioning(computer, model):
    """Conditioning estimate using exact or iterative singular-value bounds."""

    def _compute():
        log_conditions = []
        config = computer.config
        every_n = max(1, int(getattr(config, "conditioning_every_n_layers", 1)))
        min_out = int(getattr(config, "conditioning_min_out_features", 0))
        iters = max(2, int(getattr(config, "conditioning_power_iters", 6)))
        exact_max_dim = max(8, int(getattr(config, "conditioning_exact_max_dim", 256)))
        inverse_shift = max(
            float(getattr(config, "conditioning_inverse_shift", 1e-6)),
            config.eps,
        )
        eps = float(config.eps)
        layer_idx = 0

        for name, param in model.named_parameters():
            if "weight" not in name or param.dim() < 2 or not param.requires_grad:
                continue

            W = param.reshape(param.size(0), -1)
            if layer_idx % every_n != 0:
                layer_idx += 1
                continue
            if min_out > 0 and W.size(0) < min_out:
                layer_idx += 1
                continue
            layer_idx += 1
            if min(W.size()) <= 1:
                continue

            try:
                gram = W.matmul(W.t()) if W.size(0) <= W.size(1) else W.t().matmul(W)
                gram = gram.to(dtype=torch.float64)
                k = int(gram.size(0))
                if k < 2:
                    continue

                if k <= exact_max_dim:
                    eigvals = torch.linalg.eigvalsh(gram)
                    lambda_min = float(torch.clamp(eigvals[0], min=eps).item())
                    lambda_max = float(torch.clamp(eigvals[-1], min=eps).item())
                else:
                    v = torch.randn(k, device=gram.device, dtype=gram.dtype)
                    v = v / (v.norm() + eps)
                    for _ in range(iters):
                        v = gram.matmul(v)
                        v = v / (v.norm() + eps)
                    lambda_max = float(
                        torch.clamp(torch.dot(v, gram.matmul(v)), min=eps).item()
                    )

                    trace_mean = float(torch.trace(gram).item() / max(k, 1))
                    shift = max(inverse_shift * max(trace_mean, eps), eps)
                    eye = torch.eye(k, device=gram.device, dtype=gram.dtype)
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
                        log_conditions.append(
                            float(np.clip(math.log(cond + eps), 0.0, 30.0))
                        )
            except Exception:
                continue

        return sum(log_conditions) / len(log_conditions) if log_conditions else 0.0

    return computer._compute_safely(_compute)
