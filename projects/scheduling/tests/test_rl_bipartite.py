import numpy as np
import pytest
import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_bipartite import (
    BipartiteActorCritic, BipartiteObservation, TaskNeighborhoodEnv,
    bipartite_observation, listwise_update,
)
from projects.scheduling.rl_learning import ppo_batch_update
from projects.scheduling.rl_portfolio import portfolio_rollout


def environment(**kwargs):
    instance, witness = witness_instance(seed=514788, regime='balanced')
    return TaskNeighborhoodEnv(instance, witness, budget=2, **kwargs)


def test_concrete_action_coverage_and_failure_memory():
    env = environment()
    groups = {a.jobs for a in env.actions if a.family == 4}
    assert len(groups) == env.problem.J*(env.problem.J+1)//2
    assert all((j,) in groups for j in range(env.problem.J))
    target = next(i for i, a in enumerate(env.actions, 1) if a.jobs == (0,))
    env.failed_subsets.add((0,))
    assert not env.action_mask()[target] and env.action_mask()[target+1]
    assert not env.action_mask()[0]


def test_edges_expose_schedule_pairing_and_are_frozen():
    env = environment()
    observation, _ = bipartite_observation(env)
    torch.testing.assert_close(observation.edges[..., 0], env.incumbent.float())
    assert not observation.edges[~observation.eligible].any()
    assert observation.eligible.shape == env.incumbent.shape
    copied = observation.edges.clone()
    env.state.last_mask[:] = True
    torch.testing.assert_close(copied, observation.edges)


def test_bipartite_task_and_action_permutation_equivariance():
    torch.manual_seed(17)
    observation, mask = bipartite_observation(environment())
    policy = BipartiteActorCritic(hidden=16, layers=1)
    dist, value = policy(observation, mask)
    tasks = torch.arange(len(observation.tasks)-1, -1, -1)
    actions = torch.tensor([0, *range(len(mask)-1, 0, -1)])
    permuted = BipartiteObservation(
        observation.rows[actions], observation.tasks[tasks], observation.times,
        observation.released[actions][:, tasks], observation.edges[tasks], observation.eligible[tasks])
    other, v2 = policy(permuted, mask[actions])
    torch.testing.assert_close(other.probs, dist.probs[actions], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value, v2, atol=1e-6, rtol=1e-5)


def test_graph_uses_edges_and_off_control_ignores_them():
    torch.manual_seed(91)
    observation, mask = bipartite_observation(environment())
    changed = BipartiteObservation(observation.rows, observation.tasks, observation.times,
                                  observation.released, observation.edges+2, observation.eligible)
    for graph in (True, False):
        policy = BipartiteActorCritic(hidden=16, layers=1, graph=graph)
        original, _ = policy(observation, mask)
        altered, _ = policy(changed, mask)
        if graph:
            assert not torch.equal(original.logits[mask], altered.logits[mask])
        else:
            torch.testing.assert_close(original.probs, altered.probs)


def test_isolated_nodes_stop_only_and_safe_checkpoint(tmp_path):
    observation, mask = bipartite_observation(environment())
    observation.eligible.zero_()
    mask.zero_()
    mask[0] = True
    model = BipartiteActorCritic(hidden=16)
    dist, value = model(observation, mask)
    assert dist.probs[0] == 1 and torch.isfinite(value)
    path = tmp_path/'graph.pt'
    model.save(path, metadata=dict(seed=4))
    loaded = BipartiteActorCritic.load(path)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key])


def test_listwise_preserves_ties_and_learns_high_quality_actions():
    torch.manual_seed(812)
    observation, mask = bipartite_observation(environment())
    policy = BipartiteActorCritic(hidden=16, layers=1)
    gain = torch.zeros(len(mask))
    gain[9:11] = 1
    before = float(policy(observation, mask)[0].probs[9:11].sum().detach())
    result = listwise_update(policy, torch.optim.Adam(policy.parameters(), lr=.01),
                            [(observation, mask, gain)], epochs=12)
    after = float(policy(observation, mask)[0].probs[9:11].sum().detach())
    assert after > before and result['updates'] == 12
    result = listwise_update(policy, torch.optim.Adam(policy.parameters()),
                            [(observation, mask, torch.zeros(len(mask)))], epochs=1)
    assert result['updates'] == 0


def test_real_rollout_and_ppo_preserve_feasibility():
    env = environment()
    env.deadline = env.clock()+.2
    policy = BipartiteActorCritic(hidden=16, layers=1)
    trajectory = portfolio_rollout(env, policy)
    assert env.problem.config.validate(env.problem.instance, env.incumbent).feasible
    assert sum(t[-1] for t in trajectory) == pytest.approx((env.objective-env.initial_objective)/env.scale)
    metrics = ppo_batch_update(policy, torch.optim.Adam(policy.parameters()), [trajectory],
                               trace_lambda=1, minibatch_size=4, epochs=1)
    assert np.isfinite(metrics['loss'])
