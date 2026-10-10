import numpy as np
import pytest
import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_learning import ppo_batch_update
from projects.scheduling.rl_portfolio import (
    PortfolioActorCritic, PortfolioEnv, PortfolioObservation, portfolio_rollout,
)


def environment(**kwargs):
    instance, witness = witness_instance(seed=786177, regime='balanced')
    return PortfolioEnv(instance, witness, budget=2, **kwargs)


def test_portfolio_masks_and_expired_repair_leave_profile_available():
    now = [0.]
    env = environment(clock=lambda: now[0])
    assert len(env.actions) == 17
    assert not env.action_mask()[0]
    now[0] = env.deadline-.01
    mask = env.action_mask()
    assert not mask[1:9].any() and mask[9:].all()
    before = env.incumbent.clone()
    reward, done = env.step(2)
    assert reward == 0 and not done
    assert not env.log[-1]['executed']
    torch.testing.assert_close(before, env.incumbent)
    with pytest.raises(ValueError, match='STOP'):
        env.step(0)
    now[0] = env.deadline
    assert env.step(0) == (0., True)


def test_real_profile_move_validated_and_rewards_telescope():
    env = environment()
    initial = env.objective
    reward, _ = env.step(9)
    report = env.problem.config.validate(env.problem.instance, env.incumbent)
    assert report.feasible and report.objective >= initial
    assert reward == pytest.approx((report.objective-initial)/env.scale)
    assert env.log[-1]['family'] == 'profile_pressure'
    assert len(env.log[-1]['jobs']) == 1


def test_late_profile_result_rejected(monkeypatch):
    now = [0.]
    env = environment(clock=lambda: now[0])
    original = env.incumbent.clone()

    def late(*args, **kwargs):
        now[0] = 3
        return original, env.objective, dict(evaluations=1)

    monkeypatch.setattr('projects.scheduling.rl_portfolio.reprofile', late)
    reward, done = env.step(9)
    assert reward == 0 and done and not env.log[-1]['accepted']
    torch.testing.assert_close(env.incumbent, original)


def test_attention_action_permutation_and_frozen_observation():
    torch.manual_seed(217)
    env = environment()
    policy = PortfolioActorCritic(hidden=16)
    obs, mask = policy.observation(env)
    assert obs.rows.shape == (18, 32)
    first, value = policy(obs, mask)
    order = torch.tensor([0, *range(17, 0, -1)])
    permuted = PortfolioObservation(obs.rows[order], obs.tasks, obs.times, obs.released[order])
    result, v2 = policy(permuted, mask[order])
    torch.testing.assert_close(result.probs, first.probs[order], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value, v2, atol=1e-6, rtol=1e-5)
    env.state.last_mask[:] = True
    replay, _ = policy(obs, mask)
    torch.testing.assert_close(replay.probs, first.probs)
    exhausted = torch.zeros_like(mask)
    exhausted[0] = True
    dist, v = policy(obs, exhausted)
    assert int(dist.sample()) == 0 and torch.isfinite(v)


def test_portfolio_ppo_and_safe_checkpoint(tmp_path):
    env = environment()
    env.deadline = env.clock()+.2
    policy = PortfolioActorCritic(hidden=16)
    trajectory = portfolio_rollout(env, policy)
    assert sum(t[-1] for t in trajectory) == pytest.approx((env.objective-env.initial_objective)/env.scale)
    result = ppo_batch_update(policy, torch.optim.Adam(policy.parameters()),
                              [trajectory], trace_lambda=1, minibatch_size=4)
    assert np.isfinite(result['loss'])
    path = tmp_path/'policy.pt'
    policy.save(path, metadata=dict(seed=2))
    loaded = PortfolioActorCritic.load(path)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])


def test_proposal_randomness_is_separate_from_selector():
    a, b = environment(seed=456), environment(seed=456)
    a.selection_rng.random(1000)
    a.refresh()
    b.refresh()
    for x, y in zip(a.actions, b.actions):
        assert x.family == y.family and x.jobs == y.jobs
        torch.testing.assert_close(x.mask, y.mask)
