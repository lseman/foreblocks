import numpy as np
import torch


def compute_naswot(computer, activations, conv_linear_modules):
    """NASWOT binary activation agreement kernel score."""

    def _compute():
        total_logdet = 0.0
        valid_layers = 0
        # NASWOT builds an [R, R] binary-agreement kernel where R is the
        # activation's leading dim. For transformer/MoE layers the leading dim is
        # batch*seq (e.g. 2048), not batch, so without a cap the kernel becomes
        # huge. Worse, with F features each term has rank <= F, so when R > 2*F the
        # kernel is guaranteed singular and slogdet falls into the (CUDA-stalling)
        # jitter loop on a large matrix. Cap rows and skip rank-deficient layers.
        max_rows = max(8, int(getattr(computer.config, "naswot_max_rows", 256)))

        for name, _ in conv_linear_modules:
            if name not in activations:
                continue
            act = activations[name]
            if act.size(0) < 2:
                continue

            try:
                flat = act.flatten(1)
                if flat.size(0) > max_rows:
                    flat = flat[:max_rows]
                # Skip layers whose kernel cannot be full-rank: rank(kernel) <= 2F,
                # so R > 2F means a singular kernel and a meaningless logdet.
                if flat.size(0) > 2 * flat.size(1):
                    continue
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
