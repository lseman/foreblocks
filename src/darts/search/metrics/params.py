def compute_params(computer, model):
    """Count unique trainable parameters in the candidate model."""

    def _compute():
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return computer._compute_safely(_compute)
