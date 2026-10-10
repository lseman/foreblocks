"""Deadline observations, real repair removal, and paired component inference."""
from __future__ import annotations

import json

import numpy as np
import pytest

from projects.scheduling.milp import ONTSMilp
from projects.scheduling.profile_portfolio import solve_profile_portfolio
from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.research.portfolio_ablation import VARIANTS, analyze, make_plan
from projects.scheduling.research.protocol import objectives_at_budget


@pytest.fixture
def case():
    return witness_instance(seed=815771, regime="balanced")


def test_first_feasible_precedes_polish_completion(case, monkeypatch):
    import projects.scheduling.profile_portfolio as portfolio
    instance, witness = case
    clock = [10.]
    monkeypatch.setattr(portfolio, 'perf_counter', lambda: clock[0])

    def initialized(problem, start, rng, *, on_improvement, **kwargs):
        clock[0] += .2
        value = problem.config.validate(instance,witness).objective
        on_improvement(witness,value)
        clock[0] += .6
        return witness,value,1

    monkeypatch.setattr(portfolio, 'weighted_local_search', initialized)
    result=portfolio.solve_profile_portfolio(instance,time_limit=5,max_iterations=0)
    assert result.time_to_first_feasible == pytest.approx(.2)
    assert result.initialization_complete_at == pytest.approx(.8)
    assert result.first_feasible_source == 'local_search_validation'
    # The observation is credited even when initialization finishes after this cutoff.
    assert objectives_at_budget(result.to_dict(),.3) == result.objective
    assert objectives_at_budget(result.to_dict(),.1) is None


def test_no_repairs_never_invokes_milp(case, monkeypatch):
    instance,witness=case
    def forbidden(*args, **kwargs):
        pytest.fail('Disabled MILP repairs were invoked')
    monkeypatch.setattr(ONTSMilp,'solve',forbidden)
    result=solve_profile_portfolio(instance,initial_schedule=witness,time_limit=.5,
                                   patience=1,max_iterations=12,enable_repairs=False)
    assert result.time_to_first_feasible == 0.
    assert all(e.get('operator') != 'repair' for e in result.history)
    assert ONTSMilp(instance).config.validate(instance,result.schedule).feasible


def test_exact_callback_observes_valid_incumbent(case):
    instance,_=case
    result=ONTSMilp(instance).solve(time_limit=.4,solver_threads=1,observe_incumbents=True)
    assert result.schedule is not None
    assert result.first_feasible_source == 'highs_incumbent_callback'
    assert 0 <= result.time_to_first_feasible <= result.runtime
    assert result.incumbent_history
    assert result.validation.feasible
    assert [e['objective'] for e in result.incumbent_history] == sorted(e['objective'] for e in result.incumbent_history)


def test_counterfactuals_change_one_setting():
    assert VARIANTS['improvement/no_repairs'] == {'enable_repairs':False}
    assert VARIANTS['improvement/no_surplus'] == {'surplus_weight':0.}
    assert VARIANTS['cold/no_early_stop'] == {'polish_after_feasible':None}


def synthetic_rows():
    rows=[]
    for i in range(2):
        for seed in [101,211]:
            for key in VARIANTS:
                track,method=key.split('/')
                # One all-unsolved instance must remain in the all-instance score.
                value=100. if i == 0 else None
                if method == 'no_surplus' and value is not None: value=80.
                history=[] if value is None else [dict(time=.1,objective=value)]
                rows.append(dict(instance=i,seed=seed,track=track,method=method,
                    objective=value,feasible=value is not None,history=history,
                    time_to_first_feasible=.1 if value is not None else None,
                    initialization_complete_at=.5 if value is not None else None,
                    shared_start_hash=f'{i}-{seed}',validation_passed=True,error=None))
    return rows


def test_paired_analysis_retains_failures_and_adjusts_pvalues():
    plan=dict(count=2,seeds=[101,211],budget=1.)
    rows=synthetic_rows();result=analyze(rows,plan)
    assert result['methods']['improvement/full']['mean_score_percent'] == 50.
    assert result['methods']['improvement/no_surplus']['mean_score_percent'] == 40.
    assert result['all_unsolved'] == {'cold':1,'improvement':1}
    assert result['comparisons'][1]['score_pp']['mean_percent'] == pytest.approx(10.)
    assert all('holm_p' in c['score_pp'] for c in result['comparisons'])
    json.dumps(result,allow_nan=False)
    with pytest.raises(ValueError,match='Complete unique'):
        analyze(rows[:-1],plan)
    rows[0]['shared_start_hash']='different'
    with pytest.raises(AssertionError):analyze(rows,plan)
