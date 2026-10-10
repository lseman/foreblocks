"""Constellation ONTS: DP exactness, validator/MILP agreement, search validity."""

import itertools
import math

import numpy as np
import pytest

from projects.scheduling.constellation import (
    ConstellationInstance,
    ConstellationMilp,
    best_profile_multi,
    generate_constellation,
    solve_constellation,
    validate_constellation,
)
from projects.scheduling.job_profile import JobRules


def single_constellation_task(rules, S, T):
    con = dict(
        u=np.array([1.0]),
        q=np.array([0.0]),
        f_min=np.array([rules.min_run]),
        f_max=np.array([rules.max_run]),
        p_min=np.array([rules.min_gap]),
        p_max=np.array([rules.max_gap]),
        w_min=np.array([rules.win_min]),
        w_max=np.array([rules.win_max + 1]),
        g_min=np.array([rules.min_starts]),
        g_max=np.array([rules.max_starts]),
    )
    empty = {
        k: np.zeros((S, 0))
        for k in (
            "u",
            "q",
            "f_min",
            "f_max",
            "p_min",
            "p_max",
            "w_min",
            "w_max",
            "y_min",
            "y_max",
        )
    }
    return ConstellationInstance(
        solar=np.full((S, T), 100.0), soc0=np.full(S, 0.7), sat=empty, con=con
    )


def test_multi_satellite_dp_matches_brute_force_on_idle_separated_profiles():
    rng = np.random.default_rng(0)
    S = 2
    for _ in range(40):
        T = int(rng.integers(4, 8))
        smin, rmin, gmin = (
            int(rng.integers(0, 3)),
            int(rng.integers(1, 3)),
            int(rng.integers(1, 4)),
        )
        rules = JobRules(
            smin,
            smin + int(rng.integers(0, 3)),
            rmin,
            rmin + int(rng.integers(0, 3)),
            gmin,
            gmin + int(rng.integers(0, T)) if rng.random() < 0.6 else T + 1,
            0,
            T,
        )
        rewards = rng.normal(size=(S, T))
        rewards[rng.random((S, T)) < 0.1] = -math.inf
        inst = single_constellation_task(rules, S, T)
        best = -math.inf
        for assign in itertools.product(range(S + 1), repeat=T):
            a = np.array(assign)
            # Idle-separated subset: no switch between adjacent active slots.
            if any(a[t] and a[t + 1] and a[t] != a[t + 1] for t in range(T - 1)):
                continue
            X = np.zeros((S, 1, T), bool)
            for s in range(S):
                X[s, 0] = a == s + 1
            if not np.isfinite(np.where(X[:, 0], rewards, 0)).all():
                continue
            if np.any(X[:, 0] & ~np.isfinite(rewards)):
                continue
            _, bad = validate_constellation(inst, X)
            if not bad:
                best = max(best, float(np.where(X[:, 0], rewards, 0).sum()))
        row, owner, value = best_profile_multi(rewards, rules)
        if row is None:
            assert best == -math.inf
            continue
        assert value == pytest.approx(best)
        X = np.zeros((S, 1, T), bool)
        for s in range(S):
            X[s, 0] = row & (owner == s)
        assert not validate_constellation(inst, X)[1]


@pytest.fixture(scope="module")
def small():
    return generate_constellation(S=3, Js=3, Jc=2, T=48, seed=7)


def test_validator_agrees_with_milp_feasibility(small):
    inst = small
    result = solve_constellation(inst, time_limit=4, patience=5, seed=0)
    assert result["schedule"] is not None
    base = result["schedule"]
    model = ConstellationMilp(inst)
    rng = np.random.default_rng(1)
    agree = 0
    for trial in range(12):
        X = base.copy()
        if trial:
            flips = rng.integers(0, X.size, size=int(rng.integers(1, 4)))
            X.ravel()[flips] ^= True
        feasible_validator = not validate_constellation(inst, X)[1]
        solved, status, _ = model.solve(
            time_limit=5, incumbent=X, free=np.zeros_like(X)
        )
        feasible_milp = solved is not None
        assert feasible_validator == feasible_milp, (trial, status)
        agree += 1
    assert agree == 12


def test_search_and_exact_return_valid_schedules(small):
    inst = small
    port = solve_constellation(inst, time_limit=3, patience=3, seed=1)
    assert not validate_constellation(inst, port["schedule"])[1]
    objectives = [v for _, v in port["history"]]
    assert objectives == sorted(objectives)
    exact = solve_constellation(inst, time_limit=5, mode="exact")
    if exact["schedule"] is not None:
        assert not validate_constellation(inst, exact["schedule"])[1]
        if exact["bound"] is not None:
            assert port["objective"] <= exact["bound"] + 1e-6
