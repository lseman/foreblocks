import numpy as np
import torch


def compute_naswot(computer, activations, conv_linear_modules):
    """NASWOT binary activation agreement kernel score."""

    def _compute():
        total_logdet = 0.0
        valid_layers = 0

        for name, _ in conv_linear_modules:
            if name not in activations:
                continue
            act = activations[name]
            if act.size(0) < 2:
                continue

            try:
                flat = act.flatten(1)
                binary = (flat > 0).to(dtype=torch.float64)
                inv_binary = 1.0 - binary
                kernel = binary @ binary.t() + inv_binary @ inv_binary.t()
                kernel = 0.5 * (kernel + kernel.t())

                sign, logdet = torch.linalg.slogdet(kernel)
                if sign.item() <= 0 or not torch.isfinite(logdet):
                    eye = torch.eye(
                        kernel.size(0), device=kernel.device, dtype=kernel.dtype
                    )
                    jitter = max(float(computer.config.eps), 1e-12)
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
        value = total_logdet / valid_layers
        return float(value if np.isfinite(value) else 0.0)

    return computer._compute_safely(_compute)
