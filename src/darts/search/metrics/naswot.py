import numpy as np
import torch


def compute_naswot(computer, activations, conv_linear_modules, batch_size=None):
    """Log determinant of the sum of layerwise activation agreement kernels."""

    def _compute():
        # Each row represents one example, including all its sequence positions.
        max_rows = max(8, int(getattr(computer.config, "naswot_max_rows", 256)))
        kernel = None

        for name, _ in conv_linear_modules:
            if name not in activations:
                continue
            act = activations[name]
            if act.size(0) < 2:
                continue

            try:
                if batch_size and act.size(0) != batch_size and act.numel() % batch_size == 0:
                    flat = act.reshape(batch_size, -1)
                else:
                    flat = act.flatten(1)
                if flat.size(0) > max_rows:
                    flat = flat[:max_rows]
                binary = (flat > 0).to(dtype=torch.float64)
                inv_binary = 1.0 - binary
                contribution = binary @ binary.t() + inv_binary @ inv_binary.t()
                if kernel is None:
                    kernel = contribution
                elif kernel.shape == contribution.shape:
                    kernel = kernel + contribution
            except RuntimeError:
                continue

        if kernel is None:
            return 0.0
        kernel = 0.5 * (kernel + kernel.t())
        sign, logdet = torch.linalg.slogdet(kernel)
        if sign.item() <= 0 or not torch.isfinite(logdet):
            eye = torch.eye(kernel.size(0), device=kernel.device, dtype=kernel.dtype)
            jitter = max(float(computer.config.eps), 1e-12)
            for _ in range(6):
                sign, logdet = torch.linalg.slogdet(kernel + jitter * eye)
                if sign.item() > 0 and torch.isfinite(logdet):
                    break
                jitter *= 10.0
            else:
                return 0.0
        value = float(logdet.item())
        return float(value if np.isfinite(value) else 0.0)

    return computer._compute_safely(_compute)
