"""Behavioral checks for structured proposals, outcome ranking and repair safety."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from projects.scheduling.environments.onts_env import ONTSInstance
from projects.scheduling.milp import ONTSMilp
from projects.scheduling.ranked_search import solve_ranked
from projects.scheduling.ranking import (
    NeighborhoodRanker,
    RankingGraphBuilder,
    SearchState,
    make_candidates,
)
from projects.scheduling.ranking_learning import outcome_loss, train_neighborhood_ranker
from projects.scheduling.research.datasets import witness_instance


@pytest.fixture
def problem():
    instance, witness = witness_instance(seed=915731, regime="balanced")
    return ONTSMilp(instance), witness


def test_candidates_are_bounded_reproducible_and_include_all_operators(problem):
    p, x = problem
    state = SearchState.empty(p.J, p.T)
    a = make_candidates(p, x, state, np.random.default_rng(9), fraction=0.2)
    b = make_candidates(p, x, state, np.random.default_rng(9), fraction=0.2)
    assert {c.operator for c in a} == {"random", "utility", "run", "energy"}
    assert all(torch.equal(c.mask, d.mask) for c, d in zip(a, b))
    for c in a:
        assert c.mask.dtype == torch.bool
        assert c.mask.shape == (p.J, p.T)
        assert 0 < c.mask.sum() <= np.ceil(p.n_x * 0.2)
        assert torch.isfinite(c.features).all()
        assert not c.mask.numpy()[p.upper[: p.n_x].reshape(p.J, p.T) == 0].any()


def test_run_candidates_never_split_an_active_run(problem):
    p, x = problem
    candidates = make_candidates(
        p, x, SearchState.empty(p.J, p.T), np.random.default_rng(3)
    )
    for candidate in candidates:
        if candidate.operator == "run":
            for j, row in enumerate(x.bool().numpy()):
                endpoints = np.flatnonzero(np.diff(np.r_[False, row, False]))
                for start, end in zip(endpoints[::2], endpoints[1::2]):
                    released = candidate.mask[j, start:end]
                    assert not released.any() or released.all()


def test_ranker_is_invariant_to_job_relabeling(problem):
    p, x = problem
    permuted = ONTSInstance(
        **{
            k: v if k == "recurso_p" else v.flip(0)
            for k, v in asdict(p.instance).items()
        }
    )
    other = ONTSMilp(permuted)
    state = SearchState.empty(p.J, p.T)
    candidates = make_candidates(p, x, state, np.random.default_rng(1))
    from projects.scheduling.ranking import Candidate

    reversed_candidates = [
        Candidate(c.mask.flip(0), c.operator, c.features.clone()) for c in candidates
    ]
    ranker = NeighborhoodRanker(hidden=8, layers=1)
    first = ranker(RankingGraphBuilder(p).build_state(x, state), candidates)
    second = ranker(
        RankingGraphBuilder(other).build_state(x.flip(0), state), reversed_candidates
    )
    assert torch.allclose(first[0], second[0], atol=1e-5)
    assert torch.allclose(first[1], second[1], atol=1e-5)


def test_ranking_preferences_favor_gain_over_zero_outcomes():
    gains = torch.tensor([0.0, 0.5, 0.1])
    costs = torch.ones(3)
    right = outcome_loss(torch.tensor([-0.5, 0.8, 0.2]), costs, gains, costs)
    wrong = outcome_loss(torch.tensor([0.8, -0.5, 0.2]), costs, gains, costs)
    assert right < wrong
    tied_scores = torch.zeros(3, requires_grad=True)
    loss = outcome_loss(tied_scores, costs, torch.zeros(3), costs)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(tied_scores.grad).all()


def test_sparse_ranker_backprop_and_checkpoint_roundtrip(problem, tmp_path):
    p, x = problem
    state = SearchState.empty(p.J, p.T)
    graph = RankingGraphBuilder(p).build_state(x, state)
    candidates = make_candidates(p, x, state, np.random.default_rng(1))
    ranker = NeighborhoodRanker(hidden=8, layers=1)
    scores, costs = ranker(graph, candidates)
    outcome_loss(
        scores, costs, torch.linspace(0, 0.5, len(scores)), torch.ones_like(costs)
    ).backward()
    assert all(
        torch.isfinite(v.grad).all() for v in ranker.parameters() if v.grad is not None
    )
    path = tmp_path / "ranker.pt"
    ranker.save(path, metadata={"trained": True, "config": asdict(p.config)})
    loaded = NeighborhoodRanker.load(path)
    actual = loaded(graph, candidates)
    assert torch.allclose(scores, actual[0]) and torch.allclose(costs, actual[1])


def test_candidate_scoring_is_invariant_to_candidate_order(problem):
    p, x = problem
    state = SearchState.empty(p.J, p.T)
    graph = RankingGraphBuilder(p).build_state(x, state)
    candidates = make_candidates(p, x, state, np.random.default_rng(1))
    ranker = NeighborhoodRanker(hidden=8, layers=1)
    scores, _ = ranker(graph, candidates)
    reversed_scores, _ = ranker(graph, list(reversed(candidates)))
    assert torch.allclose(scores, reversed_scores.flip(0), atol=1e-6)


@pytest.mark.parametrize("mode", ["ranked", "uniform", "bandit", "adaptive"])
def test_all_modes_preserve_feasibility_and_incumbent(problem, mode):
    p, x = problem
    initial = p.config.validate(p.instance, x).objective
    policy = NeighborhoodRanker(hidden=8, layers=1)
    result = solve_ranked(
        p.instance,
        policy=policy,
        mode=mode,
        initial_schedule=x,
        time_limit=0.3,
        repair_time=0.05,
        max_iterations=2,
    )
    assert result.objective >= initial
    assert p.config.validate(p.instance, result.schedule).feasible
    objectives = [
        h["objective"] for h in result.history if h.get("objective") is not None
    ]
    assert objectives == sorted(objectives)
    assert result.upper_bound is None  # Restricted bounds are never global.


def test_training_keeps_zero_gain_outcomes(problem):
    p, _ = problem
    optimum = p.solve(time_limit=2).schedule
    policy, metadata = train_neighborhood_ranker(
        [p.instance],
        initial_schedules=[optimum],
        rounds=1,
        epochs=1,
        candidate_count=4,
        repair_time=0.03,
        hidden=8,
        layers=1,
    )
    assert metadata["repair_outcomes"] == 4
    assert metadata["zero_gain_outcomes"] == 4
    assert not metadata["trained"]
    assert not policy.training


def test_invalid_parameters_and_initial_incumbent(problem):
    p, x = problem
    with pytest.raises(ValueError, match="requires a ranker"):
        solve_ranked(p.instance)
    with pytest.raises(ValueError, match="infeasible"):
        solve_ranked(p.instance, mode="uniform", initial_schedule=torch.zeros_like(x))
    with pytest.raises(ValueError):
        make_candidates(
            p, x, SearchState.empty(p.J, p.T), np.random.default_rng(1), count=1
        )
    with pytest.raises(ValueError):
        solve_ranked(p.instance, mode="uniform", time_limit=float("nan"))


def test_history_ablation_does_not_mutate_graph_or_candidate_features(problem):
    p, x = problem
    state = SearchState.empty(p.J, p.T)
    state.last_mask[:] = True
    state.stagnation = 3
    graph = RankingGraphBuilder(p).build_state(x, state)
    candidates = make_candidates(p, x, state, np.random.default_rng(1))
    original = graph[1].clone()
    features = candidates[0].features.clone()
    ranker = NeighborhoodRanker(hidden=8, layers=1, memory=False, semantics=False)
    assert torch.isfinite(ranker(graph, candidates)[0]).all()
    assert torch.equal(graph[1], original)
    assert torch.equal(candidates[0].features, features)
