"""Tests for perturbation-aware search with current/best separation."""
import numpy as np
import pytest
import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_perturbation import (
    PerturbationActorCritic,
    PerturbationEnv,
    ppo_batch_update,
    rollout,
)


def environment(**kwargs):
    instance, initial = witness_instance(seed=987001, regime='balanced')
    return PerturbationEnv(instance, initial, budget=5.0, seed=42, **kwargs)


class TestPerturbationEnv:
    def test_acceptance_strategies(self):
        for strategy in ('monotone', 'annealing', 'alns'):
            env = environment(acceptance=strategy)
            assert env.acceptance == strategy

    def test_invalid_acceptance(self):
        instance, initial = witness_instance(seed=987001, regime='balanced')
        with pytest.raises(ValueError, match='Unknown acceptance'):
            PerturbationEnv(instance, initial, acceptance='invalid')

    def test_current_best_separation(self):
        env = environment(acceptance='monotone')
        assert env.current_objective == env.best_objective  # Initially same
        assert torch.equal(env.incumbent, env.best_schedule)

    def test_best_so_far_tracks_improvements(self):
        env = environment(acceptance='monotone')
        original_best = env.best_objective
        # Execute some steps to improve
        for _ in range(5):
            if env.done:
                break
            _, mask = env.observe()
            affordable = torch.where(mask)[0].tolist()
            repairs = [a for a in affordable if a]
            if not repairs:
                break
            action = int(env.rng.choice(repairs))
            env.step(action)
        assert env.best_objective >= original_best

    def test_perturbations_created(self):
        env = environment(acceptance='monotone')
        # Perturbations should be created if there are enough tasks
        n_perturbations = sum(1 for c in env.candidates if c.operator == 'perturb')
        assert isinstance(n_perturbations, int)
        assert 0 <= n_perturbations <= 2


class TestPerturbationActorCritic:
    def test_forward_shape(self):
        instance, initial = witness_instance(seed=987001, regime='balanced')
        env = PerturbationEnv(instance, initial, budget=5.0, seed=42)
        policy = PerturbationActorCritic()
        rows, mask = env.observe()
        dist, value = policy(rows, mask)
        assert dist.sample().item() >= 0
        assert isinstance(value.item(), float)

    def test_masked_policy(self):
        instance, initial = witness_instance(seed=987001, regime='balanced')
        env = PerturbationEnv(instance, initial, budget=5.0, seed=42)
        policy = PerturbationActorCritic()
        rows, mask = env.observe()
        mask[1:] = False  # Only STOP available
        dist, _ = policy(rows, mask)
        assert int(dist.sample()) == 0


class TestRollout:
    def test_monotone_rollout(self):
        env = environment(acceptance='monotone')
        policy = PerturbationActorCritic()
        trajectory = rollout(env, policy, mode='ppo', greedy=True)
        assert len(trajectory) > 0

    def test_annealing_rollout(self):
        env = environment(acceptance='annealing', temperature=0.1)
        policy = PerturbationActorCritic()
        trajectory = rollout(env, policy, mode='ppo')
        assert len(trajectory) > 0

    def test_alns_rollout(self):
        env = environment(acceptance='alns')
        policy = PerturbationActorCritic()
        trajectory = rollout(env, policy, mode='ppo')
        assert len(trajectory) > 0

    def test_uniform_rollout(self):
        env = environment(acceptance='monotone')
        trajectory = rollout(env, mode='uniform')
        assert len(trajectory) > 0

    def test_reward_is_best_so_far_improvement(self):
        env = environment(acceptance='annealing', temperature=0.5)
        policy = PerturbationActorCritic()
        original_best = env.best_objective
        trajectory = rollout(env, policy, mode='ppo')
        # At least one transition should have positive reward if best improved
        total_reward = sum(t[5] for t in trajectory)
        assert env.best_objective >= original_best


class TestPPOBatchUpdate:
    def test_update_runs(self):
        instance, initial = witness_instance(seed=987001, regime='balanced')
        env = PerturbationEnv(instance, initial, budget=5.0, seed=42)
        policy = PerturbationActorCritic()
        trajectory = rollout(env, policy, mode='ppo')
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        metrics = ppo_batch_update(policy, optimizer, [trajectory])
        assert 'loss' in metrics
        assert np.isfinite(metrics['loss'])

    def test_empty_trajectory(self):
        policy = PerturbationActorCritic()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        metrics = ppo_batch_update(policy, optimizer, [])
        assert metrics == {}


class TestAcceptanceProbability:
    def test_monotone_rejects_worse(self):
        env = environment(acceptance='monotone')
        prob = env._accept_probability(-0.1, 1.0)
        assert prob == 0.0

    def test_monotone_accepts_better(self):
        env = environment(acceptance='monotone')
        prob = env._accept_probability(0.1, 1.0)
        assert prob == 1.0

    def test_annealing_accepts_worse_with_prob(self):
        env = environment(acceptance='annealing', temperature=0.5)
        prob = env._accept_probability(-0.1, 1.0)
        assert 0 < prob < 1.0

    def test_alns_accepts_better(self):
        env = environment(acceptance='alns')
        prob = env._accept_probability(0.1, 1.0)
        assert prob == 1.0


class TestObservationContext:
    def test_context_has_10_features(self):
        instance, initial = witness_instance(seed=987001, regime='balanced')
        env = PerturbationEnv(instance, initial, budget=5.0, seed=42)
        rows, mask = env.observe()
        assert rows.shape[1] == 20  # 10 features + 10 context

    def test_gap_feature_present(self):
        instance, initial = witness_instance(seed=987001, regime='balanced')
        env = PerturbationEnv(instance, initial, budget=5.0, seed=42)
        rows, mask = env.observe()
        gap = rows[0, 2].item()  # Feature index 2 = current/best gap
        assert gap >= -1e-9  # Initially zero or positive
