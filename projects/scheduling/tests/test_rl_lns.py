import itertools
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix

from projects.scheduling.milp import ONTSMilp
from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_lns import RLLNSActorCritic, lns_observation


def toy_env():
    problem = SimpleNamespace(J=3, T=1, n_x=3,
        A=coo_matrix(np.array([[1., -2., 0., 1., 0., 0., 0.],
                              [0., 1., 3., 0., 1., 0., -1.]])),
        c=np.array([-1., -2., -3., 0., 0., 0., 0.]),
        lower=np.zeros(7), upper=np.ones(7), integrality=np.array([1]*6+[0]),
        row_lower=np.array([-np.inf, 0.]), row_upper=np.array([1., np.inf]))
    problem.assignment = lambda current: np.r_[current.numpy().ravel(), np.ones(3), .8]
    return SimpleNamespace(problem=problem, current=torch.tensor([[1.], [0.], [1.]]),
                           search_context=lambda: torch.zeros(10),
                           subset_action_mask=lambda: torch.tensor([False]))


def test_true_coefficient_graph_dynamic_activity_and_frozen_snapshots():
    env = toy_env()
    obs, mask = lns_observation(env)
    assert obs.variables.shape == (7, 9) and obs.constraints.shape == (2, 8)
    assert obs.edge_index.shape[1] == env.problem.A.nnz
    np.testing.assert_allclose(obs.coefficients[:, 0].numpy(), [1/4, -2/4, 1/4, 1/6, 3/6, 1/6, -1/6])
    assignment = env.problem.assignment(env.current)
    np.testing.assert_allclose(obs.constraints[:, 0].numpy(), (env.problem.A @ assignment)/[4., 6.])
    cache = env._rl_lns_topology
    env.current[0, 0] = 0.
    changed, _ = lns_observation(env)
    assert env._rl_lns_topology is cache
    assert obs.variables[0, 3] == 1 and changed.variables[0, 3] == 0
    assert not torch.equal(obs.constraints, changed.constraints)
    changed.coefficients.zero_()
    assert obs.coefficients.abs().sum() > 0
    assert not mask[0]


def test_coefficients_state_gradients_normalization_and_safe_checkpoint(tmp_path):
    torch.manual_seed(624)
    torch.set_num_threads(1)
    model = RLLNSActorCritic(hidden=12, layers=2)
    env = toy_env()
    obs, mask = model.observation(env)
    distribution, value = model(obs, mask)
    reversed_coeff = replace(obs, coefficients=-obs.coefficients)
    assert not torch.allclose(model(reversed_coeff, mask)[0].logits, distribution.logits, atol=1e-8, rtol=1e-8)
    env.current[1, 0] = 1
    updated, _ = model.observation(env)
    assert not torch.allclose(model(updated, mask)[0].logits, distribution.logits, atol=1e-8, rtol=1e-8)
    subsets = [torch.tensor(bits) for bits in itertools.product((0, 1), repeat=3) if any(bits)]
    mass = torch.stack([distribution.log_prob(bits).exp() for bits in subsets]).sum()
    assert float(mass.detach()) == pytest.approx(1., abs=1e-6)
    loss = -distribution.log_prob(torch.tensor([1, 0, 1])) + value.square()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0
    path = tmp_path/'model.pt'
    model.save(path, {'adaptation': 'task-block', 'episodes': 2})
    loaded = RLLNSActorCritic.load(path)
    torch.testing.assert_close(loaded(obs, mask)[0].logits, distribution.logits)
    stopped, _ = loaded(obs, torch.tensor([True]))
    assert not stopped.sample().any() and stopped.log_prob(torch.zeros(3)) == 0


def test_actual_milp_assignment_graph_and_restricted_repair():
    instance, witness = witness_instance(seed=82411, regime='balanced')
    problem = ONTSMilp(instance)
    env = SimpleNamespace(problem=problem, current=witness.clone(),
        search_context=lambda: torch.zeros(10), subset_action_mask=lambda: torch.tensor([False]))
    model = RLLNSActorCritic(hidden=8)
    obs, mask = model.observation(env)
    bits = model(obs, mask)[0].sample()
    free = bits.bool()[:, None].expand_as(witness).clone()
    result = problem.solve(time_limit=.05, incumbent=witness, free_mask=free, solver_threads=1)
    assert result.schedule is not None and result.validation.feasible
    assert torch.equal(result.schedule[~free], witness[~free].bool())
    assert result.objective >= problem.config.validate(instance, witness).objective
    assert obs.variables.shape[0] == problem.n_vars
    assert obs.constraints.shape[0] == problem.A.shape[0]
    with pytest.raises(ValueError):
        RLLNSActorCritic(layers=0)
