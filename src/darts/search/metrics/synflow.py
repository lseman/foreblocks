import torch


def compute_synflow(computer, model, inputs):
    """SynFlow score: sum(|p * grad|) after weight linearization."""

    def _compute():
        was_training = model.training
        params = [p for p in model.parameters() if p.requires_grad]
        original_data = [p.detach().clone() for p in params]

        try:
            model.train()
            with torch.no_grad():
                for p in params:
                    p.abs_()

            # SynFlow is architecture-dependent, not data-dependent.
            # A single all-ones input suffices; more samples just scale the
            # gradient by N without changing the candidate ranking.
            x = torch.ones_like(inputs[:1])
            model.zero_grad()
            out = computer._unwrap_output(model(x))
            if out.dim() > 2:
                out = out.flatten(1)
            out.sum().backward()

            # Accumulate per-parameter contributions on-device and sync once.
            contribs = [
                (p * p.grad).abs().sum()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if not contribs:
                return 0.0
            stacked = torch.stack(contribs)
            # Original semantics: keep only positive, finite contributions.
            stacked = stacked[torch.isfinite(stacked) & (stacked > 0)]
            if stacked.numel() == 0:
                return 0.0
            return float(stacked.sum().item())

        finally:
            with torch.no_grad():
                for p, p0 in zip(params, original_data):
                    p.copy_(p0)
            model.zero_grad()
            if not was_training:
                model.eval()

    return computer._compute_safely(_compute)
