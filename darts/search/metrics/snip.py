import torch


def compute_snip(
    computer,
    model,
    x,
    y,
    loss_fn,
    weight_params,
    grads_first_order,
    snip_mode: str,
):
    """SNIP saliency at initialization or current weights."""
    if snip_mode == "current":
        # Accumulate per-tensor saliencies on-device; sync once at the end.
        per_tensor = [
            (g.detach() * p.detach()).abs().sum()
            for (n, p), g in zip(weight_params, grads_first_order)
            if "weight" in n and g is not None and torch.isfinite(g).all()
        ]
        if not per_tensor:
            return 0.0
        return float(torch.stack(per_tensor).mean().item())

    state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    snip_value = 0.0
    snip_count = 0
    try:
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        model.zero_grad()
        x0 = x.clone().detach().requires_grad_(True)
        y0 = y.clone().detach()

        outputs0 = model(x0)
        outputs0, y0_prep = computer.helper.prepare_data(outputs0, y0)
        loss0 = loss_fn(outputs0, y0_prep)

        if not torch.isfinite(loss0):
            return 0.0

        init_weight_params = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and "weight" in n
        ]
        init_weights = [p for _, p in init_weight_params]

        init_grads = torch.autograd.grad(
            loss0,
            init_weights,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )

        for (_, p), g in zip(init_weight_params, init_grads):
            if g is not None and torch.isfinite(g).all():
                snip_value += (g * p).abs().sum().item()
                snip_count += 1
    finally:
        model.load_state_dict(state_backup, strict=False)
        model.zero_grad()

    if snip_count == 0:
        return 0.0
    return snip_value / snip_count
