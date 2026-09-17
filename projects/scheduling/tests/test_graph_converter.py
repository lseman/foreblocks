"""Tests for graph_converter.py."""

from __future__ import annotations

import sys
import os
import pytest
import torch

# Ensure the scheduling directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_converter import (
    MILPGraphBuilder,
    MILPSolution,
    GraphTopology,
    MessageProtocol,
    GraphConfig,
    build_milp_graph,
)


class TestMILPGraphBuilder:
    """Tests for MILP graph builder."""

    def test_basic_bipartite(self):
        """Test basic bipartite graph construction."""
        A = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 1.0, 3.0]]
        )  # 2 constraints, 3 variables
        b = torch.tensor([8.0, 10.0])
        c = torch.tensor([4.0, 3.0, 1.0])

        builder = MILPGraphBuilder(A, b, c)
        solution = builder.build()

        assert isinstance(solution, MILPSolution)
        assert solution.n == 3  # 3 variables
        assert solution.config.topology == GraphTopology.BIPARTITE_VAR_TO_CONST
        # Edges: 2 constraints × 3 vars = 6 edges (all non-zero)
        assert len(solution.edges) == 6

    def test_sparse_matrix(self):
        """Test with sparse constraint matrix."""
        A = torch.zeros(3, 4)
        A[0, 0] = 1.0
        A[0, 2] = 2.0
        A[1, 1] = 1.0
        A[1, 3] = 3.0
        A[2, 0] = 1.0
        A[2, 3] = 1.0

        b = torch.tensor([5.0, 7.0, 6.0])
        c = torch.tensor([2.0, 1.0, 4.0, 3.0])

        builder = MILPGraphBuilder(A, b, c)
        solution = builder.build()

        # 6 non-zero entries = 6 edges
        assert len(solution.edges) == 6

    def test_full_bipartite(self):
        """Test full bipartite topology (both directions)."""
        A = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        b = torch.tensor([3.0, 4.0])
        c = torch.tensor([1.0, 1.0])

        config = GraphConfig(topology=GraphTopology.BIPARTITE_FULL)
        builder = MILPGraphBuilder(A, b, c, config=config)
        solution = builder.build()

        # 2 non-zeros × 2 directions = 4 edges
        assert len(solution.edges) == 4

    def test_variable_only(self):
        """Test variable-only topology (no constraints as nodes)."""
        A = torch.tensor([[1.0, 2.0], [1.0, 1.0]])
        b = torch.tensor([5.0, 4.0])
        c = torch.tensor([3.0, 2.0])

        config = GraphConfig(topology=GraphTopology.VARIABLE_ONLY)
        builder = MILPGraphBuilder(A, b, c, config=config)
        solution = builder.build()

        assert len(solution.edges) == 0  # No edges in variable-only mode

    def test_observation_format(self):
        """Test that observation dict has required keys."""
        A = torch.tensor([[1.0, 2.0, 1.0]])
        b = torch.tensor([5.0])
        c = torch.tensor([1.0, 1.0, 1.0])

        builder = MILPGraphBuilder(A, b, c)
        solution = builder.build()

        obs = solution.obs
        assert "nodes" in obs
        assert "mask" in obs
        assert "context" in obs
        assert "task_static" in obs
        assert "glob" in obs
        assert "candidate" in obs
        assert "feas_start" in obs
        assert "task_draw" in obs
        assert "budget" in obs

    def test_pointer_mask(self):
        """Test that pointer mask only targets variables."""
        A = torch.tensor([[1.0, 2.0, 1.0]])
        b = torch.tensor([5.0])
        c = torch.tensor([1.0, 1.0, 1.0])

        builder = MILPGraphBuilder(A, b, c)
        solution = builder.build()

        mask = solution.obs["mask"]
        # 3 variables + 1 constraint = 4 nodes
        # First 3 should be True (variables), last 1 False (constraint)
        assert mask.shape[0] == 1
        assert mask.shape[1] == 4
        assert mask[0, :3].all()  # All variables are pointer targets
        assert not mask[0, 3]  # Constraint is not a pointer target

    def test_custom_bounds(self):
        """Test with custom variable bounds."""
        A = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([5.0])
        c = torch.tensor([1.0, 1.0])
        bounds = [(0.0, 1.0), (0.5, 2.0)]

        builder = MILPGraphBuilder(A, b, c, bounds=bounds)
        # Bounds are set correctly before normalization (normalization happens in build())
        assert builder.bounds.shape[0] == 2
        assert builder.bounds.shape[1] == 2
        assert torch.allclose(builder.bounds[0], torch.tensor([0.0, 1.0]))
        assert torch.allclose(builder.bounds[1], torch.tensor([0.5, 2.0]))

    def test_constraint_types(self):
        """Test with mixed constraint types."""
        A = torch.tensor([[1.0, 2.0], [1.0, 1.0], [2.0, 1.0]])
        b = torch.tensor([5.0, 3.0, 4.0])
        c = torch.tensor([1.0, 1.0])
        constraint_types = ["ineq", "eq", "geq"]

        builder = MILPGraphBuilder(A, b, c, constraint_types=constraint_types)
        solution = builder.build()

        assert len(solution.edges) == 6  # All entries non-zero

    def test_message_passing(self):
        """Test message passing enriches node features."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([5.0, 6.0])
        c = torch.tensor([1.0, 1.0])

        config = GraphConfig(
            message_protocol=MessageProtocol.EDGE_WEIGHTED,
            num_message_steps=3,
        )
        builder = MILPGraphBuilder(A, b, c, config=config)
        solution = builder.build()

        # Node features should be enriched after message passing
        nodes = solution.obs["nodes"]
        assert nodes.shape[0] == 1
        assert nodes.shape[1] == 4
        assert nodes.shape[2] == config.node_feature_dim  # 2 vars + 2 constraints

    def test_normalize(self):
        """Test normalization of features."""
        A = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
        b = torch.tensor([500.0, 600.0])
        c = torch.tensor([1000.0, 1000.0])

        config = GraphConfig(auto_normalize=True)
        builder = MILPGraphBuilder(A, b, c, config=config)
        solution = builder.build()

        # Features should be normalized
        # (exact values depend on normalization, but should be reasonable)
        nodes = solution.obs["nodes"]
        assert not torch.isnan(nodes).any()
        assert not torch.isinf(nodes).any()

    def test_get_graph_structure(self):
        """Test graph structure inspection."""
        A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        b = torch.tensor([5.0, 6.0])
        c = torch.tensor([1.0, 1.0, 1.0])

        builder = MILPGraphBuilder(A, b, c)
        builder._extract_edges()  # Extract edges before inspection
        structure = builder.get_graph_structure()

        assert structure["num_variables"] == 3
        assert structure["num_constraints"] == 2
        assert structure["num_edges"] == 3  # 3 non-zero entries
        assert structure["topology"] == "bipartite_var_to_const"


class TestBuildMILPGraph:
    """Tests for convenience function."""

    def test_convenience_function(self):
        """Test build_milp_graph one-shot function."""
        A = [[1.0, 2.0], [1.0, 1.0]]
        b = [5.0, 4.0]
        c = [1.0, 1.0]

        solution = build_milp_graph(A, b, c)

        assert isinstance(solution, MILPSolution)
        assert len(solution.edges) == 4

    def test_convenience_with_config(self):
        """Test with custom config."""
        A = [[1.0, 2.0], [1.0, 1.0]]
        b = [5.0, 4.0]
        c = [1.0, 1.0]

        solution = build_milp_graph(
            A,
            b,
            c,
            topology=GraphTopology.BIPARTITE_FULL,
            message_protocol=MessageProtocol.MEAN,
            node_feature_dim=32,
        )

        assert solution.config.topology == GraphTopology.BIPARTITE_FULL
        assert solution.config.message_protocol == MessageProtocol.MEAN
        assert solution.config.node_feature_dim == 32


class TestMessageProtocols:
    """Tests for message passing protocols."""

    def test_mean_protocol(self):
        """Test mean message passing."""
        A = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        b = torch.tensor([1.0, 1.0])
        c = torch.tensor([1.0, 1.0])

        config = GraphConfig(message_protocol=MessageProtocol.MEAN, num_message_steps=2)
        builder = MILPGraphBuilder(A, b, c, config=config)
        solution = builder.build()

        assert not torch.isnan(solution.obs["nodes"]).any()

    def test_edge_weighted_protocol(self):
        """Test edge-weighted message passing."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([1.0, 1.0])
        c = torch.tensor([1.0, 1.0])

        config = GraphConfig(
            message_protocol=MessageProtocol.EDGE_WEIGHTED, num_message_steps=2
        )
        builder = MILPGraphBuilder(A, b, c, config=config)
        solution = builder.build()

        assert not torch.isnan(solution.obs["nodes"]).any()

    def test_max_protocol(self):
        """Test max message passing."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([1.0, 1.0])
        c = torch.tensor([1.0, 1.0])

        config = GraphConfig(message_protocol=MessageProtocol.MAX, num_message_steps=2)
        builder = MILPGraphBuilder(A, b, c, config=config)
        solution = builder.build()

        assert not torch.isnan(solution.obs["nodes"]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
