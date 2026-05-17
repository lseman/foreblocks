import torch


def compute_fisher(computer, model, x, y, loss_fn, weights, grads_first_order):
    """Fisher score from squared parameter gradients."""
    if not bool(getattr(computer.config, "fisher_per_sample", True)):
        vals = [
            g.pow(2).sum().item()
            for g in grads_first_order
            if g is not None and torch.isfinite(g).all()
        ]
        if not vals:
            return 0.0
        return float(sum(vals) / max(len(vals), 1))

    fisher_per_sample = []
    x_f = x.detach()
    y_f = y.detach()

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

        vals_i = [
            gi.pow(2).sum().item()
            for gi in grads_i
            if gi is not None and torch.isfinite(gi).all()
        ]
        if vals_i:
            fisher_per_sample.append(sum(vals_i) / max(len(vals_i), 1))

    if not fisher_per_sample:
        return 0.0
    return float(sum(fisher_per_sample) / len(fisher_per_sample))
