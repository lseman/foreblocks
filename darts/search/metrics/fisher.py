import torch


def compute_fisher(computer, model, x, y, loss_fn, weights, grads_first_order):
    """Fisher score from squared parameter gradients."""
    if not bool(getattr(computer.config, "fisher_per_sample", True)):
        # Mean over parameter tensors of ||grad||^2; accumulate on-device.
        per_tensor = [
            g.pow(2).sum()
            for g in grads_first_order
            if g is not None and torch.isfinite(g).all()
        ]
        if not per_tensor:
            return 0.0
        return float(torch.stack(per_tensor).mean().item())

    # True Fisher needs per-sample gradients, which a single batched backward
    # cannot recover (it sums grads across the batch). We keep the per-sample
    # loop but accumulate each sample's mean ||grad||^2 on-device, syncing only
    # once at the end instead of once per (sample, parameter) pair.
    x_f = x.detach()
    y_f = y.detach()
    per_sample: list[torch.Tensor] = []

    for i in range(int(x_f.size(0))):
        xi = x_f[i : i + 1]
        yi = y_f[i : i + 1]
        out_i = model(xi)
        out_i, yi_prep = computer.helper.prepare_data(out_i, yi)
        loss_i = loss_fn(out_i, yi_prep)
        if not torch.isfinite(loss_i):
            continue

        grads_i = torch.autograd.grad(
            loss_i,
            weights,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )

        sq = [gi.pow(2).sum() for gi in grads_i if gi is not None]
        if not sq:
            continue
        stacked = torch.stack(sq)
        stacked = stacked[torch.isfinite(stacked)]
        if stacked.numel() > 0:
            per_sample.append(stacked.mean())

    if not per_sample:
        return 0.0
    return float(torch.stack(per_sample).mean().item())
