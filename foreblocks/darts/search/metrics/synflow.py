import numpy as np
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

            x = torch.ones_like(inputs[: computer.config.max_samples])
            model.zero_grad()
            out = computer._unwrap_output(model(x))
            if out.dim() > 2:
                out = out.flatten(1)
            out.sum().backward()

            score_sum = 0.0
            count = 0
            for p in model.parameters():
                if p.grad is not None and p.requires_grad:
                    contrib = (p * p.grad).abs().sum().item()
                    if contrib > 0 and np.isfinite(contrib):
                        score_sum += contrib
                        count += 1

            if count == 0:
                return 0.0
            return float(score_sum)

        finally:
            with torch.no_grad():
                for p, p0 in zip(params, original_data):
                    p.copy_(p0)
            model.zero_grad()
            if not was_training:
                model.eval()

    return computer._compute_safely(_compute)
