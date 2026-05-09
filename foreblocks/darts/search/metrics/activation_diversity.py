import torch


def compute_activation_diversity(computer, activations, relu_modules):
    """Mean absolute pairwise activation cosine across ReLU-like layers."""

    def _compute():
        total_score = 0.0
        valid_layers = 0

        for name, _ in relu_modules:
            if name not in activations:
                continue
            act = activations[name]
            if act.size(0) < 2:
                continue

            try:
                flat = act.flatten(1)
                norm = flat.norm(dim=1, keepdim=True).clamp_min(computer.config.eps)
                normalized = flat / norm
                cos = normalized @ normalized.t()
                eye = torch.eye(cos.size(0), device=cos.device, dtype=torch.bool)
                pairwise = cos.masked_fill(eye, 0.0)
                denom = max(cos.numel() - cos.size(0), 1)
                score = pairwise.abs().sum() / denom

                if torch.isfinite(score):
                    total_score += score.item()
                    valid_layers += 1
            except Exception:
                continue

        if valid_layers == 0:
            return 0.0
        return total_score / valid_layers

    return computer._compute_safely(_compute)
