import sys
from pathlib import Path

import torch
import torch.nn as nn

from darts.training.edge_regularization import _add_edge_diversity_reg


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



class _FakeEdge:
    def __init__(self, ops, probs):
        self.available_ops = list(ops)
        self._probs = torch.as_tensor(probs, dtype=torch.float32)


class _FakeCell:
    def __init__(self, edges):
        self.edges = list(edges)


class _FakeModel(nn.Module):
    def __init__(self, cells):
        super().__init__()
        self.cells = list(cells)


def test_edge_diversity_reg_uses_finite_elementwise_similarity(monkeypatch) -> None:
    def fake_extract(edge):
        return edge._probs

    monkeypatch.setattr(
        "darts.training.edge_regularization._extract_edge_probs",
        fake_extract,
    )
    model = _FakeModel(
        [
            _FakeCell(
                [
                    _FakeEdge(["Identity", "ResidualMLP", "TimeConv"], [0.2, 0.3, 0.5]),
                    _FakeEdge(["Identity", "ResidualMLP", "TimeConv"], [0.5, 0.4, 0.1]),
                    _FakeEdge(["Identity", "ResidualMLP", "TimeConv"], [float("nan"), 1.0, float("inf")]),
                ]
            )
        ]
    )
    loss = torch.tensor(1.0, requires_grad=True)

    total, pairs = _add_edge_diversity_reg(
        model=model,
        total_arch_loss=loss,
        edge_diversity_weight=0.03,
        edge_usage_balance_weight=0.04,
        edge_identity_cap=0.45,
        edge_identity_cap_weight=0.02,
        device="cpu",
    )

    assert pairs == 3
    assert torch.isfinite(total)
    total.backward()
    assert loss.grad is not None
