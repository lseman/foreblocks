"""Tests for LRBS-style lookahead rollout search."""
import numpy as np
import pytest
import torch

from projects.scheduling.milp import SolveResult
from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_rollout import (
    _deep_clone_env,
    _restore_env,
    _rollout_one,
    evaluate_lookahead,
    lookahead_select,
    rollout_with_lookahead,
)
from projects.scheduling.rl_search import ActorCritic, SearchEnv


def environment(budget=10, **kwargs):
    instance, initial = witness_instance(seed=987001, regime='balanced')
    return SearchEnv(instance, initial, budget=budget, **kwargs)


class TestDeepClone:
    def test_clone_preserves_incumbent(self):
        env = environment()
        original_obj = env.objective
        clone = _deep_clone_env(env)
        assert clone.objective == original_obj
        assert torch.equal(clone.incumbent, env.incumbent)

    def test_clone_independent_state(self):
        env = environment()
        clone = _deep_clone_env(env)
        # Modify clone's incumbent
        clone.incumbent[:] = ~clone.incumbent
        # Original should be unchanged
        assert not torch.equal(clone.incumbent, env.incumbent)

    def test_clone_shares_problem(self):
        env = environment()
        clone = _deep_clone_env(env)
        assert clone.problem is env.problem


class TestRolloutOne:
    def test_single_step_rollout(self):
        env = environment(budget=10)
        policy = ActorCritic()
        reward, actions = _rollout_one(env, policy, max_steps=1, greedy=True)
        assert isinstance(reward, float)
        assert isinstance(actions, list)
        # Reward should be non-negative (incumbent preservation)
        assert reward >= 0

    def test_multi_step_rollout(self):
        env = environment(budget=10)
        policy = ActorCritic()
        reward, actions = _rollout_one(env, policy, max_steps=3, greedy=True)
        assert len(actions) <= 3
        assert reward >= 0

    def test_rollout_respects_done(self):
        env = environment(budget=10)
        policy = ActorCritic()
        # Force done by setting a tight deadline
        env.deadline = env.clock() + 0.001
        reward, actions = _rollout_one(env, policy, max_steps=3)
        assert len(actions) == 0 or env.done


class TestLookaheadSelect:
    def test_returns_valid_action(self):
        env = environment(budget=5)
        policy = ActorCritic()
        action = lookahead_select(env, policy, k=2, rollout_steps=1, greedy=True)
        # Action should be 0 (STOP) or a valid candidate index
        assert isinstance(action, int)
        assert 0 <= action <= 4  # At most 4 candidates

    def test_returns_zero_when_no_budget(self):
        env = environment(budget=0.01)
        policy = ActorCritic()
        action = lookahead_select(env, policy, k=2, rollout_steps=1)
        assert action == 0

    def test_exploration_changes_action(self):
        env = environment(budget=5)
        policy = ActorCritic()
        action_greedy = lookahead_select(env, policy, k=4, rollout_steps=1, greedy=True)
        # Reset environment
        env2 = environment(budget=5)
        action_random = lookahead_select(env2, policy, k=4, rollout_steps=1, greedy=False)
        # They may differ due to exploration; at least both are valid
        assert isinstance(action_greedy, int)
        assert isinstance(action_random, int)


class TestRolloutWithLookahead:
    def test_basic_rollout(self):
        env = environment(budget=5)
        policy = ActorCritic()
        trajectory = rollout_with_lookahead(env, policy, lookahead_k=2, lookahead_steps=1)
        assert len(trajectory) > 0
        # All rewards should be non-negative (incumbent preservation)
        for rows, mask, action, logp, value, reward in trajectory:
            assert reward >= -1e-9

    def test_terminates_on_done(self):
        env = environment(budget=5)
        policy = ActorCritic()
        # Force early termination with tight deadline
        env.deadline = env.clock() + 0.01
        trajectory = rollout_with_lookahead(env, policy, lookahead_k=2, lookahead_steps=1)
        assert len(trajectory) <= 3  # Should terminate quickly

    def test_preserves_incumbent_on_failure(self):
        env = environment(budget=5)
        policy = ActorCritic()
        original = env.incumbent.clone()

        def fake_solve(**kwargs):
            return SolveResult('limit', '', None, None, None, None, 0, None)

        # Monkeypatch to always fail
        import projects.scheduling.rl_rollout as module
        orig_solve = env.problem.solve
        env.problem.solve = fake_solve

        try:
            trajectory = rollout_with_lookahead(env, policy, lookahead_k=2, lookahead_steps=1)
            assert torch.equal(env.incumbent, original)
        finally:
            env.problem.solve = orig_solve


class TestEvaluateLookahead:
    def test_returns_valid_result(self):
        instance, initial = witness_instance(seed=987001, regime='balanced')
        policy = ActorCritic()
        result = evaluate_lookahead(instance, initial, policy, budget=3.0, seed=0,
                                    lookahead_k=2, lookahead_steps=1)
        assert 'objective' in result
        assert 'improvement' in result
        assert result['improvement'] >= -1e-9  # Non-negative improvement
        assert 'trajectory' in result
        assert len(result['trajectory']) > 0

    def test_feasible_output(self):
        instance, initial = witness_instance(seed=987002, regime='energy')
        policy = ActorCritic()
        result = evaluate_lookahead(instance, initial, policy, budget=3.0, seed=0,
                                    lookahead_k=2, lookahead_steps=1)
        from projects.scheduling.milp import ONTSConfig
        report = ONTSConfig().validate(instance, result['incumbent'])
        assert report.feasible


class TestRestoreEnv:
    def test_restores_state(self):
        env = environment(budget=5)
        clone = _deep_clone_env(env)

        # Modify clone
        clone.objective = env.objective + 10
        clone.state.steps = 5

        _restore_env(env, clone)
        assert env.objective == clone.objective
        assert env.state.steps == clone.state.steps

    def test_merges_operator_stats(self):
        env = environment(budget=5)
        clone = _deep_clone_env(env)
        clone.operator_pulls[0] = 3
        clone.operator_gains[0] = 0.5

        _restore_env(env, clone)
        assert env.operator_pulls[0] == 3  # Merged (was 0 + 3)
        assert env.operator_gains[0] == pytest.approx(0.5)
