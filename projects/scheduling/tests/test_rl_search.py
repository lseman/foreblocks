import numpy as np
import pytest
import torch

from projects.scheduling.milp import SolveResult
from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_search import ActorCritic, SearchEnv, ppo_update, rollout


def environment(**kwargs):
    instance, initial = witness_instance(seed=987001, regime='balanced')
    return SearchEnv(instance, initial, budget=10, **kwargs)


def test_stop_and_masked_policy():
    env = environment()
    rows, mask = env.observe()
    assert rows.shape == (5, 18)
    policy = ActorCritic()
    mask[1:] = False
    dist, _ = policy(rows, mask)
    assert int(dist.sample()) == 0
    reward, done = env.step(0)
    assert done and reward == 0
    with pytest.raises(RuntimeError):
        env.step(0)


def test_late_result_and_timeout_preserve_incumbent(monkeypatch):
    now = [0.0]
    env = environment(clock=lambda: now[0])
    original = env.incumbent.clone()

    def late(**kwargs):
        now[0] = 11
        return SolveResult('limit', '', original, env.objective, None, None, 11, None)

    monkeypatch.setattr(env.problem, 'solve', late)
    reward, done = env.step(1)
    assert reward == 0 and done
    assert not env.log[-1]['accepted']
    assert torch.equal(env.incumbent, original)


def test_rewards_telescope_and_ppo_updates():
    env = environment()
    # Restrict episode length independently of variable solver runtime.
    env.deadline = env.clock() + 0.5
    policy = ActorCritic()
    trajectory = rollout(env, policy)
    assert sum(t[-1] for t in trajectory) == pytest.approx(
        (env.objective-env.initial_objective)/env.scale)
    assert env.problem.config.validate(env.problem.instance, env.incumbent).feasible
    metrics = ppo_update(policy, torch.optim.Adam(policy.parameters()), trajectory)
    assert np.isfinite(metrics['loss'])


def test_invalid_initial_schedule():
    instance, initial = witness_instance(seed=987001, regime='periodic')
    with pytest.raises(ValueError, match='feasible'):
        SearchEnv(instance, torch.zeros_like(initial))


def test_exhaustion_stop_mask_tracks_deadline():
    now = [0.0]
    env = environment(clock=lambda: now[0], stop_mode='exhaustion')
    _, mask = env.observe()
    assert mask.tolist() == [False, True, True, True, True]
    with pytest.raises(ValueError, match='masked'):
        env.step(0)
    now[0] = 10
    _, mask = env.observe()
    assert mask.tolist() == [True, False, False, False, False]
    assert env.step(0) == (0.0, True)


def test_infeasible_and_missing_results_rejected(monkeypatch):
    env = environment()
    original = env.incumbent.clone()
    for candidate in (None, torch.zeros_like(original)):
        monkeypatch.setattr(env.problem, 'solve', lambda **kwargs: SolveResult(
            'limit', '', candidate, None, None, None, 0, None))
        reward, _ = env.step(1)
        assert reward == 0
        assert torch.equal(env.incumbent, original)
        assert not env.log[-1]['accepted']
