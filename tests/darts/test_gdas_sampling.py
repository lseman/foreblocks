import pytest
import torch
from torch import nn

from darts.architecture.darts.mixed_op import MixedOp


def test_single_path_preserves_output_and_weight_gradient(monkeypatch):
    edge = MixedOp(4, 4, 3, ["Identity", "ResidualMLP"], use_hierarchical=False,
                   adaptive_sampling=False)
    linear = nn.Linear(4, 4)
    edge.ops[0] = linear
    monkeypatch.setattr(torch.nn.functional, "gumbel_softmax",
                        lambda logits, **kwargs: logits.softmax(dim=0))
    with torch.no_grad():
        edge._alphas.copy_(torch.tensor([1., 0.]))
    x = torch.randn(2, 3, 4)
    actual, expected = edge(x), linear(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), linear.weight, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), linear.weight)[0])
    assert torch.autograd.grad(actual.sum(), edge._alphas)[0].abs().sum() > 0


@pytest.mark.parametrize("fair", [True, False])
def test_gdas_sample_frequencies_match_ranked_probabilities(fair):
    torch.manual_seed(13)
    edge = MixedOp(4, 4, 3, ["Identity", "ResidualMLP", "TimeConv"],
                   adaptive_sampling=True, use_fair_darts_hierarchical=fair)
    edge.ops = nn.ModuleList(nn.Identity() for _ in edge.available_ops)
    with torch.no_grad():
        edge.group_alphas.zero_()
        for logits in edge.op_alphas.values():
            logits.zero_()
        expected = edge.get_alphas()
        x = torch.ones(1, 3, 4)
        for _ in range(1600):
            edge(x)
        torch.testing.assert_close(edge.usage_counter / 1600, expected, atol=.04, rtol=0)
