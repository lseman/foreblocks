"""Tests for constraint-aware graph representation."""
import torch
import pytest

from projects.scheduling.constraint_aware_graph import (
    ConstraintAwareGraphBuilder,
    ConstraintInfo,
    ConstraintType,
    build_constraint_aware_graph,
)


class TestConstraintInfo:
    """Test ConstraintInfo dataclass."""

    def test_create_hard_constraint(self):
        info = ConstraintInfo(
            index=0,
            constraint_type=ConstraintType.HARD,
            relaxable=False,
            relaxation_cost=float('inf'),
            rhs_value=1.0,
            tightness=0.5,
        )
        assert info.index == 0
        assert info.constraint_type == ConstraintType.HARD
        assert info.relaxable is False
        assert info.relaxation_cost == float('inf')

    def test_create_soft_constraint(self):
        info = ConstraintInfo(
            index=1,
            constraint_type=ConstraintType.SOFT,
            relaxable=True,
            relaxation_cost=0.1,
            rhs_value=2.0,
            tightness=0.8,
        )
        assert info.constraint_type == ConstraintType.SOFT
        assert info.relaxable is True
        assert info.relaxation_cost == 0.1


class TestConstraintAwareGraphBuilder:
    """Test ConstraintAwareGraphBuilder."""

    @pytest.fixture
    def simple_milp(self):
        """Create a simple MILP for testing."""
        A = torch.tensor([
            [1.0, 0.5, 0.3],
            [0.2, 1.0, 0.4],
            [0.1, 0.3, 1.0],
        ])
        b = torch.tensor([1.0, 1.5, 2.0])
        c = torch.tensor([0.5, 0.3, 0.2])
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        return A, b, c, bounds

    @pytest.fixture
    def constraint_info(self):
        """Create constraint info for testing."""
        return [
            ConstraintInfo(
                index=0,
                constraint_type=ConstraintType.HARD,
                relaxable=False,
                relaxation_cost=float('inf'),
                rhs_value=1.0,
                tightness=0.5,
            ),
            ConstraintInfo(
                index=1,
                constraint_type=ConstraintType.SOFT,
                relaxable=True,
                relaxation_cost=0.1,
                rhs_value=1.5,
                tightness=0.8,
            ),
            ConstraintInfo(
                index=2,
                constraint_type=ConstraintType.PREFERENCE,
                relaxable=True,
                relaxation_cost=0.01,
                rhs_value=2.0,
                tightness=0.3,
            ),
        ]

    def test_build_constraint_aware_graph(self, simple_milp, constraint_info):
        """Test building a constraint-aware graph."""
        A, b, c, bounds = simple_milp
        
        builder = ConstraintAwareGraphBuilder(
            A, b, c,
            bounds=bounds,
            constraint_info=constraint_info,
        )
        
        solution = builder.build()
        
        # Check observation structure
        obs = solution.obs
        assert 'nodes' in obs
        assert 'mask' in obs
        assert 'glob' in obs
        
        # Check node features
        nodes = obs['nodes']  # [1, total_nodes, f_node]
        assert nodes.shape[0] == 1
        assert nodes.shape[1] == 6  # 3 variables + 3 constraints
        assert nodes.shape[2] == 22  # 16 base + 6 extra features
        
    def test_constraint_relaxability_features(self, simple_milp, constraint_info):
        """Test that constraint relaxability features are correctly set."""
        A, b, c, bounds = simple_milp
        
        builder = ConstraintAwareGraphBuilder(
            A, b, c,
            bounds=bounds,
            constraint_info=constraint_info,
        )
        
        v_raw, c_raw = builder._build_node_features()
        
        # Check constraint features
        assert c_raw.shape[0] == 3  # 3 constraints
        assert c_raw.shape[1] == 22  # 16 base + 6 extra
        
        # Check relaxable flag (offset 0)
        assert c_raw[0, 16] == 0.0  # HARD constraint not relaxable
        assert c_raw[1, 16] == 1.0  # SOFT constraint relaxable
        assert c_raw[2, 16] == 1.0  # PREFERENCE constraint relaxable
        
        # Check relaxation cost (offset 1)
        assert c_raw[0, 17] > 0  # HARD has high cost
        assert c_raw[1, 17] == pytest.approx(0.1 / max(1.0, b.abs().max().item()))
        assert c_raw[2, 17] == pytest.approx(0.01 / max(1.0, b.abs().max().item()))
        
    def test_constraint_type_onehot(self, simple_milp, constraint_info):
        """Test that constraint type one-hot encoding is correct."""
        A, b, c, bounds = simple_milp
        
        builder = ConstraintAwareGraphBuilder(
            A, b, c,
            bounds=bounds,
            constraint_info=constraint_info,
        )
        
        v_raw, c_raw = builder._build_node_features()
        
        # Check type one-hot (offsets 3-5)
        # HARD: [1, 0, 0]
        assert c_raw[0, 19] == pytest.approx(1.0)
        assert c_raw[0, 20] == pytest.approx(0.0)
        assert c_raw[0, 21] == pytest.approx(0.0)
        
        # SOFT: [0, 1, 0]
        assert c_raw[1, 19] == pytest.approx(0.0)
        assert c_raw[1, 20] == pytest.approx(1.0)
        assert c_raw[1, 21] == pytest.approx(0.0)
        
        # PREFERENCE: [0, 0, 1]
        assert c_raw[2, 19] == pytest.approx(0.0)
        assert c_raw[2, 20] == pytest.approx(0.0)
        assert c_raw[2, 21] == pytest.approx(1.0)

    def test_observation_has_relaxability_metadata(self, simple_milp, constraint_info):
        """Test that observation includes relaxability metadata."""
        A, b, c, bounds = simple_milp
        
        builder = ConstraintAwareGraphBuilder(
            A, b, c,
            bounds=bounds,
            constraint_info=constraint_info,
        )
        
        solution = builder.build()
        obs = solution.obs
        
        assert 'constraint_relaxable' in obs
        assert 'constraint_relaxation_cost' in obs
        
        relaxable = obs['constraint_relaxable'].squeeze(0)  # [m]
        cost = obs['constraint_relaxation_cost'].squeeze(0)  # [m]
        
        assert relaxable.shape[0] == 3
        assert cost.shape[0] == 3
        
        assert relaxable[0].item() == False
        assert relaxable[1].item() == True
        assert relaxable[2].item() == True


class TestBuildConstraintAwareGraph:
    """Test convenience factory function."""

    def test_convenience_function(self):
        """Test that build_constraint_aware_graph works correctly."""
        A = torch.tensor([[1.0, 0.5], [0.2, 1.0]])
        b = torch.tensor([1.0, 1.5])
        c = torch.tensor([0.5, 0.3])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        constraint_info = [
            ConstraintInfo(
                index=0,
                constraint_type=ConstraintType.HARD,
                relaxable=False,
                relaxation_cost=float('inf'),
                rhs_value=1.0,
                tightness=0.5,
            ),
            ConstraintInfo(
                index=1,
                constraint_type=ConstraintType.SOFT,
                relaxable=True,
                relaxation_cost=0.1,
                rhs_value=1.5,
                tightness=0.8,
            ),
        ]
        
        solution = build_constraint_aware_graph(
            A, b, c, bounds=bounds, constraint_info=constraint_info
        )
        
        assert solution is not None
        assert 'nodes' in solution.obs
        assert 'constraint_relaxable' in solution.obs
