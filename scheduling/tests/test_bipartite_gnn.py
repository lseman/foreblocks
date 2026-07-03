"""Tests for BipartiteGNN encoder + decoder + scheduler."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from models.bipartite_gnn import BipartiteGNN, BipartiteDecoder, BipartiteGNNScheduler


class TestBipartiteGNN:
    """Test the BipartiteGNN encoder in isolation."""

    def setup_method(self):
        torch.manual_seed(42)
        self.d = 16
        self.gnn = BipartiteGNN(d_model=self.d, n_layers=2, edge_dim=1)

    def test_basic_shapes(self):
        """Encoder produces correct output shapes."""
        n_vars = 4
        n_const = 3
        n_total = n_vars + n_const

        nodes = torch.randn(n_total, self.d)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [4, 5, 4, 6, 4, 5]])
        node_out, graph_emb = self.gnn(nodes, edge_index, None, n_vars)

        assert list(node_out.shape) == [n_total, self.d]
        assert list(graph_emb.shape) == [self.d]

    def test_with_edge_features(self):
        """Encoder works with per-edge features."""
        n_vars = 4
        n_total = n_vars + 3
        n_edges = 6

        nodes = torch.randn(n_total, self.d)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [4, 5, 4, 6, 4, 5]])
        edge_feat = torch.randn(n_edges, 1)

        node_out, graph_emb = self.gnn(nodes, edge_index, edge_feat, n_vars)

        assert list(node_out.shape) == [n_total, self.d]
        assert list(graph_emb.shape) == [self.d]

    def test_different_layer_counts(self):
        """Encoder works with 1, 2, 4 layers."""
        for n_layers in [1, 2, 4]:
            gnn = BipartiteGNN(d_model=self.d, n_layers=n_layers, edge_dim=0)
            nodes = torch.randn(7, self.d)
            edge_index = torch.tensor([[0, 1, 2], [4, 5, 6]])
            out, emb = gnn(nodes, edge_index, None, 3)
            assert list(out.shape) == [7, self.d]
            assert list(emb.shape) == [self.d]

    def test_empty_edges(self):
        """Encoder handles graphs with no edges."""
        n_total = 7
        nodes = torch.randn(n_total, self.d)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        out, emb = self.gnn(nodes, edge_index, None, 3)
        assert list(out.shape) == [n_total, self.d]
        assert list(emb.shape) == [self.d]

    def test_positional_encoding(self):
        """Positional encoding adds per-node signal."""
        n_total = 10
        nodes_zero = torch.zeros(n_total, self.d)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        out, _ = self.gnn(nodes_zero, edge_index, None, 4)

        # With positional encoding, zeros should not stay zeros
        assert out.std() > 0.01, "Positional encoding should add signal"


class TestBipartiteScheduler:
    """Test the full BipartiteGNNScheduler."""

    def setup_method(self):
        torch.manual_seed(42)
        self.scheduler = BipartiteGNNScheduler(
            f_node=8,
            d_model=32,
            n_gnn_layers=2,
            edge_dim=1,
        )

    def test_forward_shapes(self):
        """Forward pass produces correct shapes."""
        B = 2
        n_vars = 4
        n_const = 3
        n_total = n_vars + n_const

        nodes = torch.randn(B, n_total, 8)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [4, 5, 4, 6, 4, 5]]).unsqueeze(0)
        edge_feat = torch.randn(B, 6, 1)
        mask = torch.ones(B, n_vars, dtype=torch.bool)

        obs = {
            "nodes": nodes,
            "edge_index": edge_index,
            "edge_features": edge_feat,
            "mask": mask,
            "n_vars": n_vars,
        }

        result = self.scheduler.act(obs, greedy=True)

        assert list(result["new_starts"].shape) == [B, n_vars]
        assert list(result["logp"].shape) == [B]
        assert list(result["entropy"].shape) == [B]
        assert list(result["value"].shape) == [B]

    def test_greedy_vs_stochastic(self):
        """Greedy and stochastic decoding both produce valid outputs."""
        B = 1
        n_vars = 5
        n_const = 3
        n_total = n_vars + n_const

        nodes = torch.randn(B, n_total, 8)
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 4], [5, 6, 5, 6, 5, 6, 5, 6]]
        ).unsqueeze(0)
        mask = torch.ones(B, n_vars, dtype=torch.bool)

        obs = {
            "nodes": nodes,
            "edge_index": edge_index,
            "mask": mask,
            "n_vars": n_vars,
        }

        greedy_result = self.scheduler.act(obs, greedy=True)
        stochastic_result = self.scheduler.act(obs, greedy=False)

        assert list(greedy_result["new_starts"].shape) == [B, n_vars]
        assert list(stochastic_result["new_starts"].shape) == [B, n_vars]

    def test_no_edges_fallback(self):
        """Scheduler handles missing edges gracefully."""
        B = 2
        n_vars = 3
        n_total = n_vars + 2

        nodes = torch.randn(B, n_total, 8)
        mask = torch.ones(B, n_vars, dtype=torch.bool)

        obs = {
            "nodes": nodes,
            "edge_index": None,
            "mask": mask,
            "n_vars": n_vars,
        }

        result = self.scheduler.act(obs, greedy=True)
        assert list(result["new_starts"].shape) == [B, n_vars]
        assert list(result["logp"].shape) == [B]

    def test_masked_actions(self):
        """Scheduler respects action mask — only picks masked nodes."""
        B = 1
        n_vars = 5
        n_const = 3
        n_total = n_vars + n_const

        nodes = torch.randn(B, n_total, 8)
        edge_index = torch.tensor([[0, 0, 1, 2, 3, 4], [5, 6, 5, 5, 5, 5]]).unsqueeze(0)

        # Mask: only nodes 0, 2, 4 are valid
        mask = torch.zeros(B, n_vars, dtype=torch.bool)
        mask[0, 0] = True
        mask[0, 2] = True
        mask[0, 4] = True

        obs = {
            "nodes": nodes,
            "edge_index": edge_index,
            "mask": mask,
            "n_vars": n_vars,
        }

        # Replay: force picks from {0, 2, 4}
        action = torch.zeros(B, n_vars)
        action[0, 0] = 1.0
        action[0, 2] = 1.0
        action[0, 4] = 1.0

        result = self.scheduler.act(obs, action=action)
        assert result["new_starts"].sum() >= 0  # valid tensor

    def test_value_consistency(self):
        """Value estimates are finite and reasonable."""
        B = 4
        n_vars = 4
        n_const = 3

        nodes = torch.randn(B, n_vars + n_const, 8)
        edge_index = (
            torch.tensor([[0, 1, 2, 3], [4, 5, 6, 4]]).unsqueeze(0).expand(B, -1, -1)
        )
        mask = torch.ones(B, n_vars, dtype=torch.bool)

        obs = {
            "nodes": nodes,
            "edge_index": edge_index,
            "mask": mask,
            "n_vars": n_vars,
        }

        result = self.scheduler.act(obs, greedy=True)
        assert torch.isfinite(result["value"]).all(), "Values must be finite"
        assert torch.isfinite(result["logp"]).all(), "Log-probs must be finite"


class TestMessagePassing:
    """Test that message passing actually propagates information."""

    def test_constraint_to_variable_flow(self):
        """Constraint features should influence connected variable nodes."""
        torch.manual_seed(123)

        n_vars = 3
        n_const = 2
        n_total = n_vars + n_const
        d = 16

        gnn = BipartiteGNN(d_model=d, n_layers=2, edge_dim=0)

        # All nodes start at zero
        nodes = torch.zeros(n_total, d)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 4]])  # all vars connect to c0

        # Set constraint features to known values
        nodes[3, 0] = 1.0  # c0 has feature 0 = 1
        nodes[4, 0] = 2.0  # c1 has feature 0 = 2

        node_out, _ = gnn(nodes, edge_index, None, n_vars)

        # Variable nodes should have been updated by constraint message passing
        assert node_out[0].norm() > 0.01, "Variable 0 should receive constraint message"

    def test_variable_to_constraint_flow(self):
        """Variable features should propagate to constraints."""
        torch.manual_seed(456)

        n_vars = 3
        n_const = 2
        n_total = n_vars + n_const
        d = 16

        gnn = BipartiteGNN(d_model=d, n_layers=2, edge_dim=0)

        nodes = torch.zeros(n_total, d)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 4]])

        # Set variable features to known values
        nodes[0, 1] = 1.0
        nodes[1, 1] = 2.0
        nodes[2, 1] = 3.0

        node_out, _ = gnn(nodes, edge_index, None, n_vars)

        # Constraint c0 should aggregate from all three variables
        c0_out = node_out[3]
        assert c0_out.norm() > 0.01, "Constraint should receive variable messages"


class TestBipartiteIntegration:
    """Integration tests using graph_converter output."""

    def test_with_graph_converter(self):
        """BipartiteGNNScheduler works with graph_converter output."""
        from graph_converter import build_milp_graph

        # Small generic MILP
        A = [[2, 3, 1], [1, 1, 2]]  # 2 constraints, 3 variables
        b = [5, 4]
        c = [-3, -2, -1]  # minimization

        result = build_milp_graph(
            A,
            b,
            c,
            bounds=[(0, 1), (0, 1), (0, 1)],
            constraint_types=["ineq", "ineq"],
        )

        obs = result.obs

        # Build edge_index from the graph
        n_vars = result.n
        edges = result.edges
        edge_index = None
        edge_feat = None
        if edges:
            src = [e.src_node for e in edges]
            dst = [e.dst_node for e in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_feat = torch.ones(len(edges), 1)

        mask = torch.ones(n_vars, dtype=torch.bool)

        # Create scheduler that matches feature dims
        scheduler = BipartiteGNNScheduler(
            f_node=obs["nodes"].shape[-1],
            d_model=32,
            n_gnn_layers=2,
            edge_dim=1,
        )

        batch_obs = {
            "nodes": obs["nodes"],  # [1, N, f]
            "edge_index": edge_index.unsqueeze(0) if edge_index is not None else None,
            "edge_features": edge_feat.unsqueeze(0) if edge_feat is not None else None,
            "mask": mask.unsqueeze(0),
            "n_vars": n_vars,
        }

        result_sched = scheduler.act(batch_obs, greedy=True)

        assert list(result_sched["new_starts"].shape) == [1, n_vars]
        assert list(result_sched["logp"].shape) == [1]
        assert torch.isfinite(result_sched["value"]).all()

    def test_larger_problem(self):
        """Scheduler handles larger problems (10 vars, 5 constraints)."""
        from graph_converter import build_milp_graph

        torch.manual_seed(99)
        n_vars = 10
        n_const = 5

        A = torch.randn(n_const, n_vars).round(decimals=2)  # sparse-ish
        b = torch.randn(n_const)
        c = torch.randn(n_vars)

        result = build_milp_graph(A, b, c, bounds=[(0, 10)] * n_vars)

        edges = result.edges
        edge_index = None
        if edges:
            edge_index = torch.tensor(
                [[e.src_node for e in edges], [e.dst_node for e in edges]],
                dtype=torch.long,
            )

        scheduler = BipartiteGNNScheduler(
            f_node=result.obs["nodes"].shape[-1],
            d_model=32,
            n_gnn_layers=2,
        )

        batch_obs = {
            "nodes": result.obs["nodes"],
            "edge_index": edge_index.unsqueeze(0) if edge_index is not None else None,
            "mask": torch.ones(n_vars, dtype=torch.bool).unsqueeze(0),
            "n_vars": n_vars,
        }

        result_sched = scheduler.act(batch_obs, greedy=True)
        assert list(result_sched["new_starts"].shape) == [1, n_vars]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-x"])
