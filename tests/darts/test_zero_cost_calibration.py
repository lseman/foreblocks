"""Calibration smoke test for the zero-cost proxy suite (search/metrics/).

The proxy suite (NASWOT, SynFlow, SNIP, GRASP, Jacobian, Fisher, params,
FLOPs) is comprehensive but, before this test, nothing checked that a proxy
score actually tracks anything about the candidate — a metric could start
silently returning a constant or garbage value and every existing test
would keep passing, since none of them look at *ranking quality*.

This guards against exactly that: build several DARTS-search-space
operations at increasing capacity (latent_dim) and assert a proxy's score
ranks them consistently with that capacity ordering.

Design note: an earlier version of this test tried to correlate proxy
scores with *post-training* validation loss on a toy regression task, per
the original design sketch. Empirically (see PR discussion) that was flaky
— with only a handful of gradient steps on a tiny synthetic task, trained
loss did not reliably decrease with capacity across random seeds, so the
Spearman correlation swung from ~0.8 to negative between runs. SynFlow
score vs. capacity, by contrast, was ``1.0`` across every seed tested
because SynFlow is a deterministic, data-free function of the model's
initialized weights — that determinism is exactly what makes it suitable as
a *fast* proxy in the first place, and it is what this test exploits to
stay non-flaky.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from darts.architecture.op_registry import build_op
from darts.search.metrics import Config, MetricsComputer
from darts.search.weight_schemes import spearman_from_scores

CAPACITIES = [4, 8, 16, 32, 64, 96]


class _TinyCandidate(nn.Module):
    """A minimal single-op DARTS "architecture" at a given capacity."""

    def __init__(self, op_name: str, latent_dim: int, seq_length: int = 16):
        super().__init__()
        self.op = build_op(op_name, 1, latent_dim, seq_length)
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.op(x))


def _compute_metrics(op_name: str, seed: int) -> dict[str, list[float]]:
    torch.manual_seed(seed)
    x = torch.randn(8, 16, 1)
    y = torch.randn(8, 16, 1)

    scores: dict[str, list[float]] = {"params": [], "flops": [], "synflow": []}
    for capacity in CAPACITIES:
        torch.manual_seed(seed * 100 + capacity)
        model = _TinyCandidate(op_name, capacity)
        computer = MetricsComputer(Config(timeout=0.0, max_samples=8))
        results = computer.compute_all(model, x, y, include_heavy_metrics=True)

        for name in scores:
            result = results[name]
            if not result.success:
                raise AssertionError(f"{name} metric failed for capacity={capacity}: {result.error}")
            scores[name].append(float(result.value))
    return scores


class TestZeroCostProxyTracksCapacity(unittest.TestCase):
    def test_params_and_flops_are_monotonic_in_capacity(self):
        for op_name in ("ResidualMLP", "TimeConv"):
            scores = _compute_metrics(op_name, seed=0)
            self.assertEqual(scores["params"], sorted(scores["params"]), msg=op_name)
            self.assertEqual(scores["flops"], sorted(scores["flops"]), msg=op_name)

    def test_synflow_rank_correlates_with_capacity(self):
        for op_name in ("ResidualMLP", "TimeConv"):
            for seed in range(3):
                scores = _compute_metrics(op_name, seed=seed)
                corr = spearman_from_scores(
                    np.array(scores["synflow"]), np.array(CAPACITIES, dtype=float)
                )
                self.assertGreater(
                    corr,
                    0.5,
                    msg=f"op={op_name} seed={seed} synflow/capacity correlation too low: {corr}",
                )


if __name__ == "__main__":
    unittest.main()
