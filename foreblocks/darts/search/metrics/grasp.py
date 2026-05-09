import numpy as np
import torch


def compute_grasp(computer, model, x, y, loss, loss_fn, weights):
    """GRASP objective: negative Hessian-gradient alignment."""

    def _grasp_from_loss(
        loss_local: torch.Tensor,
        weights_local,
        *,
        retain_graph: bool,
    ) -> float:
        grads_local = torch.autograd.grad(
            loss_local,
            weights_local,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        hvp_seed = torch.tensor(0.0, device=loss_local.device)
        valid_grads = 0
        for g in grads_local:
            if g is None or not torch.isfinite(g).all():
                continue
            hvp_seed = hvp_seed + (g * g.detach()).sum()
            valid_grads += 1

        if valid_grads == 0:
            return 0.0

        hgs_local = torch.autograd.grad(
            hvp_seed,
            weights_local,
            create_graph=False,
            retain_graph=retain_graph,
            allow_unused=True,
        )

        scores = []
        for hg, g in zip(hgs_local, grads_local):
            if hg is None or g is None:
                continue
            if not torch.isfinite(hg).all() or not torch.isfinite(g).all():
                continue
            score_item = (hg * g.detach()).sum().item()
            if np.isfinite(score_item):
                scores.append(float(score_item))

        if not scores:
            return 0.0

        value = -sum(scores) / max(len(scores), 1)
        if not np.isfinite(value):
            return 0.0
        return float(value)

    def _retry_with_safe_mode() -> float:
        model.zero_grad()
        x_retry = x.detach().clone().requires_grad_(True)
        y_retry = y.detach().clone()
        with computer.helper.safe_mode(model):
            out_retry = model(x_retry)
        out_retry, y_retry_prep = computer.helper.prepare_data(out_retry, y_retry)
        loss_retry = loss_fn(out_retry, y_retry_prep)

        if not torch.isfinite(loss_retry):
            return 0.0

        return _grasp_from_loss(loss_retry, weights, retain_graph=False)

    try:
        value = _grasp_from_loss(loss, weights, retain_graph=True)
        if np.isfinite(value):
            return float(value)
        return _retry_with_safe_mode()
    except RuntimeError as e:
        if not computer._is_backend_double_backward_error(e):
            raise

        try:
            return _retry_with_safe_mode()
        except RuntimeError as retry_error:
            if computer._is_backend_double_backward_error(retry_error):
                return 0.0
            raise
        except Exception:
            return 0.0
