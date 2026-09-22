import pytest
import torch
import torch.nn as nn

from darts.search import zero_cost
from darts.search.metrics import (
    Config,
    MetricsComputer,
    _default_enable_flops,
    _torch_version_tuple,
)
from darts.search.metrics.grasp import compute_grasp
from darts.search.metrics.flops import compute_flops
from darts.search.metrics.compatibility import CompatibilityHelper
from darts.search.metrics.config import Result
from darts.search.metrics.synflow import compute_synflow
from darts.search.metrics.naswot import compute_naswot
from darts.search.metrics.snip import compute_snip
from darts.search.metrics.zero_cost_nas import ZeroCostNAS
from darts.trainer import DARTSTrainer
from darts.architecture.blocks.attention import SelfAttention
from darts.architecture.darts.time_series_darts import TimeSeriesDARTS


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


def test_flops_default_does_not_depend_on_torch_tracer_version(monkeypatch) -> None:
    assert _torch_version_tuple("2.11.0+cu126") == (2, 11)
    assert _torch_version_tuple("2.12.0.dev20260601") == (2, 12)

    monkeypatch.setattr(torch, "__version__", "2.11.0+cu126")
    assert _default_enable_flops()

    monkeypatch.setattr(torch, "__version__", "2.12.0")
    assert _default_enable_flops()


def test_zero_cost_config_keeps_hook_based_flops(monkeypatch) -> None:
    monkeypatch.setattr(torch, "__version__", "2.11.0")

    cfg = zero_cost._make_config(max_samples=4, fast_mode=True)

    assert cfg.enable_flops
    assert "flops" in cfg.weights


class ReusedSequenceLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.layer(x)
        return self.layer(x)


def test_flops_count_sequence_positions_and_reused_modules() -> None:
    model = ReusedSequenceLinear()
    computer = MetricsComputer(Config(timeout=0.0, enable_flops=True))
    x = torch.randn(4, 7, 3)

    shared = computer.compute_all(model, x, include_heavy_metrics=False)["flops"]
    standalone = compute_flops(computer, model, x)

    expected = 2 * 7 * 3 * 5 * 2
    assert shared.success and shared.value == expected
    assert standalone.success and standalone.value == expected


def test_bilevel_split_keeps_time_order() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(10))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    arch_loader, weight_loader = DARTSTrainer._create_bilevel_loaders(
        object(), loader
    )

    assert list(weight_loader.dataset.indices) == list(range(7))
    assert list(arch_loader.dataset.indices) == list(range(7, 10))


def test_parameter_count_counts_shared_weights_once() -> None:
    shared = nn.Linear(3, 2)
    model = nn.Sequential(shared, shared)
    result = MetricsComputer(Config(timeout=0.0)).params(model)
    assert result.success and result.value == sum(p.numel() for p in shared.parameters())


def test_raw_aggregation_excludes_failed_metrics() -> None:
    values, successes, totals, errors = {}, {}, {}, {}
    ZeroCostNAS._accumulate_raw_batch(
        {"snip": Result(0.0, False, "timed out")},
        values, successes, totals, errors,
    )
    assert "snip" not in values
    assert successes == {}
    assert totals == {"snip": 1}
    assert errors == {"snip": "timed out"}


def test_synflow_preserves_batch_norm_state_and_mode() -> None:
    model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    model.train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    computer = MetricsComputer(Config(timeout=0.0))
    result = compute_synflow(computer, model, torch.ones(4, 3))
    assert result.success
    assert model.training
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_safe_attention_boolean_mask_matches_sdpa() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    mask = torch.tensor([[True, False, True], [True, True, False], [False, True, True]])
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    actual = CompatibilityHelper._manual_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_naswot_sums_layer_kernels_before_logdet() -> None:
    activations = {
        "a": torch.tensor([[1., -1.], [-1., 1.]]),
        "b": torch.tensor([[1., 1.], [1., -1.]]),
    }
    computer = MetricsComputer(Config(timeout=0.0))
    result = compute_naswot(computer, activations, [("a", None), ("b", None)])
    # K_a = [[2, 0], [0, 2]], K_b = [[2, 1], [1, 2]], det(K_a+K_b)=15.
    assert result.success and result.value == pytest.approx(torch.log(torch.tensor(15.)).item())


def test_snip_sums_saliency_across_parameter_tensors() -> None:
    model = nn.Sequential(nn.Linear(2, 1), nn.Linear(1, 1))
    params = list(model.named_parameters())
    grads = [torch.ones_like(param) for _, param in params]
    computer = MetricsComputer(Config(timeout=0.0))
    expected = sum(
        param.abs().sum().item()
        for name, param in params if "weight" in name
    )
    actual = compute_snip(computer, model, None, None, None, params, grads, "current")
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["linear", "probsparse"])
def test_causal_attention_supports_distinct_kernels(mode: str) -> None:
    attention = SelfAttention(dim=8, heads=2, causal=True,
                              attention_type=mode, position_mode="none")
    assert attention.attention_type == mode


def test_decoder_style_selection_receives_architecture_gradient() -> None:
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decode_style_alphas = nn.Parameter(torch.tensor([2.0, 0.0, -1.0]))

        def get_decode_style_weights(self):
            return torch.softmax(self.decode_style_alphas, dim=0)

    class Stub(TimeSeriesDARTS):
        def __init__(self):
            nn.Module.__init__(self)
            self.forecast_decoder = Decoder()

        def _decode_autoregressive_path(self, *args):
            return torch.ones(1, 2, 1)

    model = Stub()
    result = model._decode_with_style(
        x_seq=torch.ones(1, 2, 1), x_future=None, decoder_targets=None,
        teacher_forcing_ratio=0.0, memory=torch.ones(1, 2, 1),
        encoder_output=torch.ones(1, 2, 1), decoder_hidden=None,
    )
    result.sum().backward()
    assert model.forecast_decoder.decode_style_alphas.grad is not None
    assert model.forecast_decoder.decode_style_alphas.grad.abs().sum() > 0


def test_decoder_style_fallback_passes_autoregressive_arguments() -> None:
    class Stub(TimeSeriesDARTS):
        def __init__(self):
            nn.Module.__init__(self)
            self.forecast_decoder = nn.Identity()

        def _decode_autoregressive_path(self, **kwargs):
            return kwargs

    model = Stub()
    targets = torch.ones(1, 2, 1)
    result = model._decode_with_style(
        x_seq=targets, x_future=None, decoder_targets=targets,
        teacher_forcing_ratio=0.25, memory=targets,
        encoder_output=targets, decoder_hidden=None,
    )
    assert result["decoder_targets"] is targets
    assert result["teacher_forcing_ratio"] == 0.25
    assert result["memory"] is targets
