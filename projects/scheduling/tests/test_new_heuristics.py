"""Job-profile DP, weighted local search, adaptive LB and predict-and-search."""

import itertools
import math

import numpy as np
import pytest
import torch

from projects.scheduling.classical import adapt_radius, solve_classical
from projects.scheduling.environments.onts_env import ONTSInstance
from projects.scheduling.job_profile import (
    JobRules,
    battery_trace,
    best_profile,
    lp_energy_prices,
    reprofile,
    solve_profile,
)
from projects.scheduling.local_search import solve_local_search
from projects.scheduling.milp import ONTSMilp
from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.trust_region import (
    SolutionPredictor,
    contrastive_loss,
    solve_predict_search,
    static_graph,
    trust_region_row,
)
from projects.scheduling.validation import validate_schedule


@pytest.fixture
def problem():
    instance, witness = witness_instance(seed=815771, regime="energy")
    return ONTSMilp(instance), witness


def single_job(rules, T):
    values = (1.0, 0.0, *(getattr(rules, f) for f in JobRules.__dataclass_fields__))
    return ONTSInstance(
        *(torch.tensor([v]) for v in values),
        recurso_p=torch.full((T,), 100.0, dtype=torch.float64),
    )


@pytest.mark.parametrize("truncated", [True, False])
def test_profile_dp_matches_brute_force(truncated):
    rng = np.random.default_rng(1 + truncated)
    for _ in range(60):
        T = int(rng.integers(5, 11))
        smin, rmin, gmin = (
            int(rng.integers(a, b)) for a, b in ((0, 3), (1, 4), (1, 6))
        )
        wmin = int(rng.integers(0, 3))
        rules = JobRules(
            smin,
            smin + int(rng.integers(0, 3)),
            rmin,
            rmin + int(rng.integers(0, 4)),
            gmin,
            gmin + int(rng.integers(0, T)) if rng.random() < 0.6 else T + 1,
            wmin,
            int(rng.integers(wmin + 1, T + 3)),
        )
        reward = rng.normal(size=T)
        reward[rng.random(T) < 0.1] = -math.inf
        instance = single_job(rules, T)
        best = -math.inf
        for bits in itertools.product([0, 1], repeat=T):
            row = np.array(bits, dtype=bool)
            if (row & ~np.isfinite(reward)).any():
                continue
            report = validate_schedule(
                instance,
                torch.tensor(row[None]).int(),
                allow_truncated_final_run=truncated,
            )
            if report.feasible:
                best = max(best, float(reward[row].sum()))
        row, value = best_profile(reward, rules, truncated_final_run=truncated)
        if row is None:
            assert best == -math.inf
            continue
        assert value == pytest.approx(best)
        assert value == pytest.approx(float(reward[row].sum()))
        assert validate_schedule(
            instance, torch.tensor(row[None]).int(), allow_truncated_final_run=truncated
        ).feasible


def test_battery_closed_form_matches_validator(problem):
    p, x = problem
    usage = p.instance.uso_p.numpy() @ x.numpy()
    expected = np.array(p.config.validate(p.instance, x).soc)
    assert np.allclose(battery_trace(p, usage), expected)


def test_reprofile_preserves_feasibility_and_never_decreases(problem):
    p, x = problem
    start = p.config.validate(p.instance, x).objective
    rng = np.random.default_rng(0)
    for jobs in ([0], [1, 2], list(range(p.J))):
        schedule, value, info = reprofile(p, x, jobs, rng)
        report = p.config.validate(p.instance, schedule)
        assert report.feasible and value == pytest.approx(report.objective)
        assert value >= start - 1e-9 and info["gain"] == pytest.approx(value - start)
        # Only released jobs may change.
        others = [j for j in range(p.J) if j not in jobs]
        assert torch.equal(schedule[others], x.bool()[others])


def test_lp_prices_are_nonnegative(problem):
    prices = lp_energy_prices(problem[0], time_limit=1.0)
    assert prices is not None and prices.shape == (problem[0].T,)
    assert (prices >= 0).all()


@pytest.mark.parametrize("kwargs", [{}, {"prices": "lp"}, {"fallback_time": 0.05}])
def test_profile_search_preserves_incumbent(problem, kwargs):
    p, x = problem
    result = solve_profile(
        p.instance, initial_schedule=x, time_limit=0.6, max_iterations=6, **kwargs
    )
    assert p.config.validate(p.instance, result.schedule).feasible
    assert result.objective >= p.config.validate(p.instance, x).objective
    assert result.upper_bound is None


def test_local_search_cold_start_and_polish(problem):
    p, x = problem
    cold = solve_local_search(p.instance, time_limit=0.6, seed=0)
    assert cold.schedule is not None
    assert p.config.validate(p.instance, cold.schedule).feasible
    assert cold.upper_bound is None and cold.time_to_first_feasible is not None
    warm = solve_local_search(p.instance, time_limit=0.4, initial_schedule=x)
    assert warm.objective >= p.config.validate(p.instance, x).objective


def test_adaptive_radius_rules():
    assert adapt_radius(10, gain=1.0, status="limit", n_x=100) == 10
    assert adapt_radius(10, gain=0.0, status="optimal", n_x=100) == 15
    assert adapt_radius(80, gain=0.0, status="optimal", n_x=100) == 100
    assert adapt_radius(10, gain=0.0, status="limit", n_x=100) == 6
    assert adapt_radius(2, gain=0.0, status="limit", n_x=100) == 2


@pytest.mark.parametrize("mode", ["adaptive_local_branch", "portfolio_plus"])
def test_new_classical_modes_preserve_incumbent(problem, mode):
    p, x = problem
    result = solve_classical(
        p.instance,
        mode=mode,
        initial_schedule=x,
        time_limit=0.8,
        repair_time=0.08,
        max_iterations=6,
    )
    assert p.config.validate(p.instance, result.schedule).feasible
    assert result.objective >= p.config.validate(p.instance, x).objective
    assert result.upper_bound is None
    radii = [h["radius"] for h in result.history if h.get("radius") is not None]
    assert all(r >= 2 for r in radii)


def test_trust_region_row_contains_predicted_schedule(problem):
    p, x = problem
    probabilities = x.numpy().ravel().astype(float)
    row, ones, zeros = trust_region_row(p, probabilities, k1=5, k0=20, delta=0)
    assert set(ones.tolist()).isdisjoint(zeros.tolist())
    values = np.zeros(p.n_vars)
    values[: p.n_x] = probabilities
    assert (row.A @ values)[0] <= row.ub[0] + 1e-9


def test_contrastive_loss_prefers_positive():
    legal = torch.ones(6)
    positive = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
    negative = ~positive
    aligned = torch.where(positive, 3.0, -3.0)
    assert contrastive_loss(aligned, legal, [positive], [negative]) < contrastive_loss(
        -aligned, legal, [positive], [negative]
    )


def test_predictor_round_trip_and_search(problem, tmp_path):
    p, _ = problem
    model = SolutionPredictor(8, 1)
    model.save(tmp_path / "m.pt", metadata={"trained": True})
    loaded = SolutionPredictor.load(tmp_path / "m.pt")
    graph = static_graph(p)
    assert torch.allclose(model(graph), loaded(graph))
    result = solve_predict_search(p.instance, loaded, time_limit=1.0)
    if result.schedule is not None:
        assert p.config.validate(p.instance, result.schedule).feasible
    assert result.upper_bound is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"controller": "stagnation", "job_selection": "rules", "patience": 2},
        {"controller": "stagnation", "job_selection": "adaptive", "patience": 2},
        {"controller": "ucb", "job_selection": "rules"},
    ],
)
def test_profile_portfolio_preserves_incumbent_and_uses_both_operators(problem, kwargs):
    from projects.scheduling.profile_portfolio import solve_profile_portfolio

    p, x = problem
    result = solve_profile_portfolio(
        p.instance,
        initial_schedule=x,
        time_limit=1.5,
        repair_time=0.05,
        max_iterations=40,
        **kwargs,
    )
    assert p.config.validate(p.instance, result.schedule).feasible
    assert result.objective >= p.config.validate(p.instance, x).objective
    assert result.upper_bound is None
    operators = {h["operator"] for h in result.history if h["event"] == "repair"}
    assert operators == {"profile", "repair"}
    objectives = [h["objective"] for h in result.history]
    assert objectives == sorted(objectives)


def test_profile_portfolio_local_search_cold_start(problem):
    from projects.scheduling.profile_portfolio import solve_profile_portfolio

    p, _ = problem
    result = solve_profile_portfolio(
        p.instance, init="local_search", time_limit=1.0, repair_time=0.05
    )
    assert result.schedule is not None
    assert p.config.validate(p.instance, result.schedule).feasible
    with pytest.raises(ValueError):
        solve_profile_portfolio(p.instance, controller="nope")


def test_reprofile_options_keep_objective_and_raise_reserve(problem):
    p, x = problem
    start = p.config.validate(p.instance, x).objective
    reserve = float(battery_trace(p, p.instance.uso_p.numpy() @ x.numpy()).sum())
    rng = np.random.default_rng(3)
    for jobs in ([0], [1, 2]):
        schedule, value, _ = reprofile(
            p, x, jobs, rng, accept_equal=True, surplus_weight=1e-6
        )
        assert p.config.validate(p.instance, schedule).feasible
        assert value >= start - 1e-9
        new_reserve = float(
            battery_trace(p, p.instance.uso_p.numpy() @ schedule.numpy()).sum()
        )
        if value <= start + 1e-9 and not torch.equal(schedule, x.bool()):
            assert new_reserve > reserve


def test_local_search_stops_soon_after_first_feasible(problem):
    from time import perf_counter

    from projects.scheduling.local_search import weighted_local_search

    p, _ = problem
    started = perf_counter()
    schedule, _, _ = weighted_local_search(
        p,
        None,
        np.random.default_rng(0),
        deadline=started + 5.0,
        polish_after_feasible=0.1,
    )
    assert schedule is not None and perf_counter() - started < 2.5


def test_portfolio_v2_options_preserve_incumbent(problem):
    from projects.scheduling.profile_portfolio import solve_profile_portfolio

    p, x = problem
    result = solve_profile_portfolio(
        p.instance,
        initial_schedule=x,
        time_limit=1.0,
        repair_time=0.05,
        patience=2,
        adaptive_fraction=True,
        accept_equal=True,
        surplus_weight=0.05,
    )
    assert p.config.validate(p.instance, result.schedule).feasible
    assert result.objective >= p.config.validate(p.instance, x).objective
    fractions = [h["fraction"] for h in result.history if "fraction" in h]
    assert all(0.02 <= f <= 0.6 for f in fractions)


def test_vectorized_dp_matches_scalar_reference():
    from reference_profile_dp import reference_best_profile

    rng = np.random.default_rng(5)
    for _ in range(150):
        T = int(rng.integers(5, 90))
        smin = int(rng.integers(0, 4))
        rmin = int(rng.integers(1, 6))
        gmin = int(rng.integers(1, 15))
        wmin = int(rng.integers(0, 5))
        rules = JobRules(
            smin,
            smin + int(rng.integers(0, 6)),
            rmin,
            rmin + int(rng.integers(0, 10)),
            gmin,
            gmin + int(rng.integers(0, T)) if rng.random() < 0.6 else T + 1,
            wmin,
            int(rng.integers(wmin + 1, T + 5)),
        )
        reward = rng.normal(size=T) + 0.3
        reward[rng.random(T) < 0.05] = -math.inf
        if rng.random() < 0.3:
            reward = np.round(reward)
        truncated = bool(rng.random() < 0.7)
        a, va = reference_best_profile(reward, rules, truncated_final_run=truncated)
        b, vb = best_profile(reward, rules, truncated_final_run=truncated)
        assert (a is None) == (b is None)
        if a is not None:
            assert va == pytest.approx(vb) and np.array_equal(a, b)
