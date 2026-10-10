"""Selective policies must fall back to DP without violating their gates."""
from types import SimpleNamespace

import pytest

from projects.scheduling.milp import ONTSMilp
from projects.scheduling.profile_portfolio import solve_profile_portfolio
from projects.scheduling.research.datasets import witness_instance


def run_clocked(monkeypatch, *, setup_seconds=0., **options):
    import projects.scheduling.profile_portfolio as module
    instance, witness = witness_instance(seed=815771, regime="balanced")
    value = ONTSMilp(instance).config.validate(instance, witness).objective
    clock = [0.]
    calls = []
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])
    original_candidates = module.make_candidates

    def candidates(*args, **kwargs):
        clock[0] += setup_seconds
        return original_candidates(*args, **kwargs)

    monkeypatch.setattr(module, "make_candidates", candidates)

    def profile(problem, incumbent, jobs, rng, **kwargs):
        clock[0] += .25
        return incumbent, value, dict(jobs=jobs)

    def solve(self, *, incumbent, time_limit, **kwargs):
        calls.append((clock[0], time_limit))
        clock[0] += time_limit
        return SimpleNamespace(schedule=incumbent, objective=value, status="optimal")

    monkeypatch.setattr(module, "reprofile", profile)
    monkeypatch.setattr(ONTSMilp, "solve", solve)
    result = solve_profile_portfolio(
        instance, initial_schedule=witness, time_limit=4., patience=1,
        repair_time=.5, job_selection="rules", max_iterations=40, **options)
    assert result.objective == value
    assert ONTSMilp(instance).config.validate(instance, result.schedule).feasible
    return calls, result


def test_delayed_repairs_wait_and_dp_keeps_running(monkeypatch):
    calls, result = run_clocked(monkeypatch, repair_start_fraction=.5)
    assert calls and min(start for start, _ in calls) >= 2.
    assert sum(e.get("operator") == "profile" for e in result.history) >= 8


def test_quota_caps_calls_and_preserves_tail_for_profiles(monkeypatch):
    calls, result = run_clocked(monkeypatch, repair_budget_fraction=.2,
                                repair_min_remaining=1.)
    assert calls and sum(cap for _, cap in calls) == pytest.approx(.8)
    assert all(start < 2.8 for start, _ in calls)
    assert result.history[-1]["operator"] == "profile"


def test_zero_quota_never_calls_milp(monkeypatch):
    calls, result = run_clocked(monkeypatch, repair_budget_fraction=0.)
    assert calls == []
    assert all(e.get("operator") != "repair" for e in result.history)


def test_mask_construction_can_exhaust_quota_without_solver_call(monkeypatch):
    calls, result = run_clocked(monkeypatch, setup_seconds=.5,
                                repair_budget_fraction=.1)
    assert calls == []
    skipped = [e for e in result.history if e["event"] == "repair_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["solve_runtime"] == pytest.approx(.5)
    assert result.history[-1]["operator"] == "profile"


@pytest.mark.parametrize("settings", [
    {"repair_start_fraction": -1}, {"repair_start_fraction": float("nan")},
    {"repair_budget_fraction": 1.1}, {"repair_min_remaining": -1},
])
def test_invalid_scheduling_settings(settings):
    with pytest.raises(ValueError, match="repair scheduling"):
        solve_profile_portfolio(None, **settings)


def test_selective_milp_only_rejected():
    with pytest.raises(ValueError, match="fallback"):
        solve_profile_portfolio(None, enable_profiles=False, repair_start_fraction=.5)
