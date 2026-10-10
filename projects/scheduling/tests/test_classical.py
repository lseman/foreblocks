"""Classical search restrictions must preserve feasibility and bound scope."""

import numpy as np
import pytest
import torch

from projects.scheduling.classical import (
    energy_exchange_mask,
    local_branch,
    lp_disagreement_mask,
    lp_relaxation,
    solve_classical,
)
from projects.scheduling.milp import ONTSMilp
from projects.scheduling.research.datasets import witness_instance


@pytest.fixture
def problem():
    instance, witness = witness_instance(seed=815771, regime="balanced")
    return ONTSMilp(instance), witness


def test_zero_radius_is_identity_and_does_not_mutate_model(problem):
    p, x = problem
    original = p.constraints
    r = local_branch(p, x, radius=0, time_limit=0.3)
    assert torch.equal(r.schedule, x.bool())
    assert r.restricted and r.upper_bound is None and r.relative_gap is None
    assert p.constraints is original


def test_local_branch_respects_distance_and_feasibility(problem):
    p, x = problem
    r = local_branch(p, x, radius=8, time_limit=0.3)
    assert (r.schedule != x.bool()).sum() <= 8
    assert p.config.validate(p.instance, r.schedule).feasible
    assert r.objective >= p.config.validate(p.instance, x).objective
    with pytest.raises(ValueError):
        local_branch(p, x, radius=-1, time_limit=0.3)


def test_lp_disagreement_and_agreement_rule(problem):
    p, x = problem
    relaxation = x.numpy().astype(float)
    j, t = next(
        (j, t) for j in range(p.J) for t in range(p.T) if p.upper[j * p.T + t] > 0
    )
    relaxation[j, t] = 1 - relaxation[j, t]
    rins = lp_disagreement_mask(p, x, relaxation, rins=True)
    assert rins.sum() == 1 and rins[j, t]
    top = lp_disagreement_mask(p, x, relaxation, fraction=0.1)
    assert top[j, t] and top.sum() <= np.ceil(p.n_x * 0.1)
    with pytest.raises(ValueError):
        lp_disagreement_mask(p, x, np.array([np.nan]))


def test_lp_solution_is_a_relaxation(problem):
    p, x = problem
    values = lp_relaxation(p, time_limit=0.4)
    assert values is not None
    lp_obj = float((values * p.instance.priority.numpy()[:, None]).sum())
    assert lp_obj >= p.config.validate(p.instance, x).objective - 1e-5


def test_exchange_is_reproducible_bounded_and_closes_runs(problem):
    p, x = problem
    a = energy_exchange_mask(p, x, np.random.default_rng(3), fraction=0.2)
    b = energy_exchange_mask(p, x, np.random.default_rng(3), fraction=0.2)
    assert torch.equal(a, b)
    assert a.sum() <= np.ceil(p.n_x * 0.2)
    for j, row in enumerate(x.bool().numpy()):
        ends = np.flatnonzero(np.diff(np.r_[False, row, False]))
        for lo, hi in zip(ends[::2], ends[1::2]):
            assert not a[j, lo:hi].any() or a[j, lo:hi].all()


@pytest.mark.parametrize("mode", ["lp_guided", "rins", "local_branch", "portfolio"])
def test_modes_preserve_incumbent_without_restricted_global_bounds(problem, mode):
    p, x = problem
    r = solve_classical(
        p.instance,
        mode=mode,
        initial_schedule=x,
        time_limit=0.45,
        repair_time=0.08,
        max_iterations=4,
    )
    assert p.config.validate(p.instance, r.schedule).feasible
    assert r.objective >= p.config.validate(p.instance, x).objective
    assert r.upper_bound is None
    assert all(h.get("restricted") for h in r.history if h["event"] == "repair")


def test_bad_parameters(problem):
    p, x = problem
    with pytest.raises(ValueError):
        solve_classical(p.instance, mode="not_a_method")
    with pytest.raises(ValueError):
        solve_classical(p.instance, time_limit=float("nan"))
