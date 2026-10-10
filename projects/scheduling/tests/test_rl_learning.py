import numpy as np
import pytest
import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_learning import (
    GraphActorCritic, GraphObservation, generalized_advantages,
    initialize_value_head, ppo_batch_update, preference_update,
)
from projects.scheduling.rl_search import ActorCritic, SearchEnv


@pytest.fixture
def graph_sample():
    instance, witness = witness_instance(seed=87631, regime='balanced')
    env = SearchEnv(instance, witness, budget=30, stop_mode='exhaustion')
    policy = GraphActorCritic(hidden=8)
    return env, policy, policy.observation(env)


def test_graph_log_probability_replay_and_candidate_permutation(graph_sample):
    env, policy, (obs, mask) = graph_sample
    first, value = policy(obs, mask)
    assert torch.isfinite(value)
    assert first.probs[0] == 0
    # Replacing live history must not mutate the frozen observation.
    env.state.last_mask[:] = True
    env.operator_gains[:] = 0.5
    replay, _ = policy(obs, mask)
    torch.testing.assert_close(first.logits[1:], replay.logits[1:])
    order = [2, 0, 3, 1]
    permuted = GraphObservation(obs.state, [obs.candidates[i] for i in order],
                               obs.rows, obs.operator_history)
    result, _ = policy(permuted, mask)
    torch.testing.assert_close(result.probs[1:], first.probs[1:][order])


def test_preference_loss_learns_winner_and_ignores_ties(graph_sample):
    _, policy, (obs, mask) = graph_sample
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.005)
    initial = {k: v.clone() for k, v in policy.state_dict().items()}
    assert preference_update(policy, optimizer, [(obs, mask, [0]*5)])['updates'] == 0
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, initial[name])
    before, _ = policy(obs, mask)
    metrics = preference_update(policy, optimizer, [(obs, mask, [0, 1, 0, 0, 0])], epochs=8)
    after, _ = policy(obs, mask)
    assert metrics['updates'] == 8
    assert after.probs[1] > before.probs[1]


def test_gae_terminal_bootstrap_and_episode_boundaries():
    trajectory = [(None, None, 1, 0., 0.2, 0.1), (None, None, 2, 0., 0.3, 0.4)]
    advantage, targets = generalized_advantages(trajectory, trace_lambda=1)
    assert targets == pytest.approx([0.5, 0.4])
    assert advantage == pytest.approx([0.3, 0.1])
    assert generalized_advantages([], trace_lambda=1) == ([], [])
    with pytest.raises(ValueError):
        generalized_advantages(trajectory, trace_lambda=2)


def test_batched_ppo_update_and_kl_guard():
    torch.manual_seed(412)
    policy = ActorCritic()
    initialize_value_head(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
    rows = torch.randn(5, 18)
    mask = torch.tensor([False, True, True, True, True])
    with torch.no_grad():
        dist, value = policy(rows, mask)
        first = (rows, mask, 1, float(dist.log_prob(torch.tensor(1))), float(value), 0.4)
        second = (rows, mask, 2, float(dist.log_prob(torch.tensor(2))), float(value), 0.0)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    metrics = ppo_batch_update(policy, optimizer, [[first], [second]], minibatch_size=2)
    assert metrics['steps'] == 2 and metrics['updates'] > 0
    assert np.isfinite(metrics['loss'])
    assert any(not torch.equal(v, before[k]) for k, v in policy.state_dict().items())
    deliberately_stale = (rows, mask, 1, -100., 0., 1.)
    metrics = ppo_batch_update(policy, optimizer, [[deliberately_stale]], target_kl=0.001)
    assert metrics['early_stop'] and metrics['updates'] == 0


def test_safe_checkpoint_roundtrip(tmp_path):
    from projects.scheduling.rl_learning import load_rl_policy
    policy = GraphActorCritic(hidden=8, memory=False)
    path = tmp_path/'model.pt'
    torch.save(dict(format='onts-rl-upgrade-v1', method='graph_ppo', architecture='graph',
                    model_config=dict(hidden=8, layers=1, semantics=True, memory=False),
                    state_dict=policy.state_dict(), plan=dict(seed=123)), path)
    loaded = load_rl_policy(path)
    assert not loaded.training and not loaded.memory
    for k, v in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)
