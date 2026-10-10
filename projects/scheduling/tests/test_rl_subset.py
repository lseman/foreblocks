import itertools

import pytest
import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_bipartite import BipartiteObservation, bipartite_observation
from projects.scheduling.rl_learning import ppo_batch_update
from projects.scheduling.rl_subset import (
    NonemptySubset, SubsetActorCritic, SubsetEnv, subset_imitation, subset_rollout,
)


def test_conditioned_subset_exact_normalization_entropy_and_mode():
    for logits in (torch.tensor([-1., .5, 1.2], dtype=torch.float64),
                   torch.full((3,), -20., dtype=torch.float64)):
        distribution = NonemptySubset(logits)
        subsets = [torch.tensor(bits) for bits in itertools.product((0, 1), repeat=3) if any(bits)]
        probabilities = torch.stack([distribution.log_prob(bits).exp() for bits in subsets])
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1., dtype=torch.float64))
        torch.testing.assert_close(distribution.entropy(), -(probabilities*probabilities.log()).sum(), atol=1e-6, rtol=1e-6)
        assert distribution.log_prob(distribution.mode()).exp() >= probabilities.max()-1e-10
        assert torch.isneginf(distribution.log_prob(torch.zeros(3)))


def test_conditioned_sampling_matches_exhaustive_distribution():
    torch.manual_seed(132)
    dist = NonemptySubset(torch.tensor([-1., .4, -.3]))
    counts = torch.zeros(8)
    for _ in range(4000):
        bits = dist.sample()
        counts[int(bits @ torch.tensor([1, 2, 4]))] += 1
    assert counts[0] == 0
    for i in range(1, 8):
        bits = torch.tensor([(i >> j) & 1 for j in range(3)])
        assert abs(float(counts[i]/4000-dist.log_prob(bits).exp())) < .03


def test_subset_permutation_and_joint_ppo_ratio():
    instance, witness = witness_instance(seed=9752, regime='balanced')
    env = SubsetEnv(instance, witness, budget=2)
    obs, mask = bipartite_observation(env)
    torch.manual_seed(721)
    policy = SubsetActorCritic()
    dist, value = policy(obs, mask)
    perm = torch.arange(instance.J-1, -1, -1)
    reordered = BipartiteObservation(obs.rows, obs.tasks[perm], obs.times,
        obs.released[:, perm], obs.edges[perm], obs.eligible[perm])
    other, other_value = policy(reordered, mask)
    torch.testing.assert_close(other.logits, dist.logits[perm])
    torch.testing.assert_close(other_value, value)
    action = dist.sample()
    torch.testing.assert_close(other.log_prob(action[perm]), dist.log_prob(action))
    old = dist.log_prob(action).detach()
    assert float((policy(obs, mask)[0].log_prob(action)-old).exp().detach()) == pytest.approx(1.)
    changed = NonemptySubset(dist.logits+1.)
    independent_ratio = torch.exp(torch.distributions.Bernoulli(logits=dist.logits+1.).log_prob(action.float()).sum()
                                  -torch.distributions.Bernoulli(logits=dist.logits).log_prob(action.float()).sum())
    assert not torch.isclose((changed.log_prob(action)-old).exp(), independent_ratio)


def test_subset_repair_training_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    instance, witness = witness_instance(seed=5172, regime='balanced')
    model = SubsetActorCritic(hidden=16)
    env = SubsetEnv(instance, witness, budget=.3)
    trajectory = subset_rollout(env, model)
    assert trajectory and env.problem.config.validate(instance, env.incumbent).feasible
    assert env.objective >= env.initial_objective
    assert all(sum(t[2]) >= 1 for t in trajectory)
    metrics = ppo_batch_update(model, torch.optim.Adam(model.parameters()), [trajectory], epochs=1)
    assert metrics['updates'] > 0
    obs, mask, bits, *_ = trajectory[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = float(model(obs, mask)[0].log_prob(torch.tensor(bits)).detach())
    subset_imitation(model, optimizer, [(obs, mask, bits, 1.)], epochs=8)
    assert float(model(obs, mask)[0].log_prob(torch.tensor(bits)).detach()) > before
    path = tmp_path/'subset.pt'
    model.save(path)
    loaded = SubsetActorCritic.load(path)
    torch.testing.assert_close(model(obs, mask)[0].logits, loaded(obs, mask)[0].logits)
    with pytest.raises(ValueError):
        env.step_subset(torch.zeros(instance.J))


def test_stop_distribution_and_rare_nonempty_sampling_are_finite():
    logits = torch.full((5,), -30.)
    dist = NonemptySubset(logits)
    for _ in range(10):
        sample = dist.sample()
        assert sample.sum() >= 1 and torch.isfinite(dist.log_prob(sample))
    assert torch.isfinite(dist.entropy())
    stopped = NonemptySubset(logits, stopped=True)
    assert not stopped.sample().any()
    assert stopped.log_prob(stopped.sample()) == 0 and stopped.entropy() == 0


def test_compact_graph_edge_control_and_expired_stop():
    torch.set_num_threads(1)
    instance, witness = witness_instance(seed=61391, regime='balanced')
    env = SubsetEnv(instance, witness, budget=1)
    obs, mask = bipartite_observation(env)
    changed = BipartiteObservation(obs.rows, obs.tasks, obs.times, obs.released,
                                  obs.edges+1, obs.eligible)
    for graph in (True, False):
        model = SubsetActorCritic(graph=graph)
        original = model(obs, mask)[0].logits
        altered = model(changed, mask)[0].logits
        if graph:
            assert not torch.equal(original, altered)
        else:
            torch.testing.assert_close(original, altered)
    env.deadline = env.clock()-1
    assert subset_rollout(env, model) == []
    assert env.done and env.log[-1]['family'] == 'stop'
    assert env.problem.config.validate(instance, env.incumbent).feasible
