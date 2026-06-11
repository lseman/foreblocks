import torch
import torch.nn as nn
import pytest

from darts.search import zero_cost
from darts.search.metrics import (
    Config,
    MetricsComputer,
    _default_enable_flops,
    _torch_version_tuple,
)
from darts.search.metrics.grasp import compute_grasp


class TinyReluNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_relu_activation_metrics_are_hooked() -> None:
    torch.manual_seed(0)
    model = TinyReluNet()
    computer = MetricsComputer(Config(timeout=0.0, max_samples=4))
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)

    results = computer.compute_all(model, x, y, include_heavy_metrics=False)

    assert results["activation_diversity"].success
    assert results["activation_diversity"].value > 0.0
    assert results["naswot"].success
    assert results["naswot"].value != 0.0


def test_grasp_uses_weight_hessian_gradient_alignment() -> None:
    torch.manual_seed(1)
    model = nn.Linear(2, 1, bias=False)
    computer = MetricsComputer(Config(timeout=0.0))
    loss_fn = nn.MSELoss()
    x = torch.tensor([[0.5, -1.0], [1.5, 2.0]])
    y = torch.tensor([[0.25], [-0.75]])

    out = model(x)
    loss = loss_fn(out, y)
    weights = [model.weight]
    grads = torch.autograd.grad(
        loss,
        weights,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    hvp_seed = sum((g * g.detach()).sum() for g in grads if g is not None)
    hgs = torch.autograd.grad(
        hvp_seed,
        weights,
        create_graph=False,
        retain_graph=True,
        allow_unused=True,
    )
    expected = -sum((hg * w.detach()).sum().item() for hg, w in zip(hgs, weights))

    actual = compute_grasp(computer, model, x, y, loss, loss_fn, weights)

    assert actual == pytest.approx(float(expected), rel=1e-6, abs=1e-8)


def test_flops_version_gate_handles_local_torch_suffix(monkeypatch) -> None:
    assert _torch_version_tuple("2.11.0+cu126") == (2, 11)
    assert _torch_version_tuple("2.12.0.dev20260601") == (2, 12)

    monkeypatch.setattr(torch, "__version__", "2.11.0+cu126")
    assert not _default_enable_flops()

    monkeypatch.setattr(torch, "__version__", "2.12.0")
    assert _default_enable_flops()


def test_zero_cost_config_drops_flops_when_version_gate_disables(monkeypatch) -> None:
    monkeypatch.setattr(torch, "__version__", "2.11.0")

    cfg = zero_cost._make_config(max_samples=4, fast_mode=True)

    assert not cfg.enable_flops
    assert "flops" not in cfg.weights
