"""Callback availability, independent replay and fixed-cell inference."""
import itertools
from time import perf_counter

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csc_matrix

from projects.scheduling.constellation import (
    best_profile_multi, generate_constellation, solve_constellation, validate_constellation,
)
from projects.scheduling.job_profile import JobRules
from projects.scheduling.observed_highs import milp_with_incumbents
from projects.scheduling.research.constellation_oracle import data_hash, from_data, instance_data, replay
from projects.scheduling.research.constellation_replication import analyze, blocked_infer
from projects.scheduling.tests.test_constellation import single_constellation_task


def test_independent_replay_matches_all_tiny_assignments():
    rules = JobRules(1, 2, 1, 2, 2, 4, 0, 4)
    instance = single_constellation_task(rules, 2, 4)
    data = instance_data(instance)
    assert data_hash(instance_data(from_data(data))) == data_hash(data)
    for assignment in itertools.product(range(3), repeat=4):
        X = np.zeros((2, 1, 4), dtype=bool)
        for s in range(2):
            X[s, 0] = np.asarray(assignment) == s + 1
        objective, bad = replay(data, X)
        expected, violations = validate_constellation(instance, X)
        assert bool(bad) == bool(violations)
        assert objective == pytest.approx(expected)
    assert replay(data, np.full((2, 1, 4), .5))[1]


def test_replay_checks_recursive_battery_and_power():
    rules = JobRules(0, 5, 1, 4, 1, 5, 0, 4)
    instance = single_constellation_task(rules, 1, 4)
    instance.solar[:] = 0
    instance.soc0[:] = .01
    instance.con["q"][:] = 20
    X = np.ones((1, 1, 4), bool)
    _, bad = replay(instance_data(instance), X)
    assert any("power" in v for v in bad) and any("battery" in v for v in bad)


@pytest.mark.parametrize("mode", ["portfolio", "exact"])
def test_validated_callbacks_preserve_credited_schedule(mode):
    instance = generate_constellation(S=2, Js=1, Jc=1, T=24, seed=73100017)
    observed = []
    result = solve_constellation(instance, mode=mode, time_limit=2., observe_incumbents=True,
        on_incumbent=lambda X, value, time, source: observed.append((X, value, time, source)))
    assert observed
    assert all(not replay(instance_data(instance), X)[1] for X, _, _, _ in observed)
    assert all(value == pytest.approx(validate_constellation(instance, X)[0])
               for X, value, _, _ in observed)
    assert [value for _, value, _, _ in observed] == sorted(value for _, value, _, _ in observed)
    assert [(time, value) for _, value, time, _ in observed] == result["history"]
    if mode == "exact":
        assert any(source == "highs_callback" for _, _, _, source in observed)


def test_model_loading_deadline_can_expire_without_solve():
    result = milp_with_incumbents(c=np.array([-1.]), integrality=np.array([1]),
        bounds=Bounds([0.], [1.]), constraints=LinearConstraint(csc_matrix([[1.]]), [0.], [1.]),
        options={"threads": 1, "time_limit": 1.}, on_incumbent=lambda X: pytest.fail("Late solve"),
        deadline=perf_counter()-1.)
    assert result.status == 1 and result.x is None


def test_blocked_variance_uses_within_cell_replication():
    values, cells = [0., 2., 4., 10., 12., 14.], [0, 0, 0, 1, 1, 1]
    result = blocked_infer(values, cells)
    assert result["mean_pp"] == pytest.approx(7.)
    assert result["standard_error"] == pytest.approx(np.sqrt(2/3))
    assert result["degrees_of_freedom"] == pytest.approx(4.)
    assert blocked_infer([0., 0.], [0, 0])["two_sided_p"] == 1.
    with pytest.raises(ValueError, match="two"):
        blocked_infer([0., 1.], [0, 1])


def test_analysis_retains_unsolved_and_averages_seeds_before_inference():
    plan = dict(entries=[dict(instance=i, cell=i//2) for i in range(4)],
                seeds=[1301, 1409], primary=["feasibility_pp", "score_pp"], budget=1.)
    rows = []
    for i in range(4):
        for method, seeds in (("portfolio", plan["seeds"]), ("exact", [0])):
            for seed in seeds:
                value = 100. if i < 3 and method == "portfolio" else None
                rows.append(dict(instance=i, method=method, seed=seed, objective=value,
                    feasible=value is not None, history=[dict(time=.2, objective=value)] if value else [],
                    runtime=.9, error=None, validation_passed=True))
    result = analyze(rows, plan)
    assert result["all_unsolved"] == 1
    assert result["comparisons"]["score_pp"]["mean_pp"] == 75.
    assert len(result["comparisons"]["score_pp"]["instance_deltas"]) == 4
    with pytest.raises(ValueError, match="Complete unique"):
        analyze(rows[:-1], plan)
    rows[0]["error"] = "broken"
    with pytest.raises(ValueError, match="errors block"):
        analyze(rows, plan)


def test_adjacent_cross_satellite_runs_can_escape_idle_separated_neighborhood():
    rules = JobRules(2, 2, 1, 1, 1, 5, 0, 4)
    instance = single_constellation_task(rules, 2, 4)
    rewards = np.zeros((2, 4))
    rewards[0, 1], rewards[1, 2] = 10., 10.
    adjacent = np.zeros((2, 1, 4), bool)
    adjacent[0, 0, 1] = adjacent[1, 0, 2] = True
    assert not replay(instance_data(instance), adjacent)[1]
    _, _, restricted = best_profile_multi(rewards, rules)
    assert restricted == 10.
    assert float((adjacent[:, 0] * rewards).sum()) == 20.
