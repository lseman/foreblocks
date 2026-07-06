import torch


def compute_activation_diversity(computer, activations, relu_modules):
    """Mean pairwise activation *diversity* across ReLU-like layers.

    For each layer we take ``1 - mean(|cosine|)`` over off-diagonal sample
    pairs: 0 means all samples produce collinear activations (no diversity),
    1 means mutually orthogonal activations (maximally diverse). The scoring
    config gives this a positive weight, so the search rewards architectures
    whose activations spread across the representation space.
    """

    def _compute():
        # Accumulate per-layer diversity on-device and sync once at the end to
        # avoid a CUDA stall per layer.
        layer_scores: list[torch.Tensor] = []

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
                # mean |cosine| in [0, 1]: 1 = collinear (no diversity).
                # Diversity = 1 - correlation so higher = more spread.
                correlation = pairwise.abs().sum() / denom
                layer_scores.append(1.0 - correlation)
            except Exception:
                continue

        if not layer_scores:
            return 0.0
        stacked = torch.stack(layer_scores)
        finite = stacked[torch.isfinite(stacked)]
        if finite.numel() == 0:
            return 0.0
        return float(finite.mean().item())

    return computer._compute_safely(_compute)
