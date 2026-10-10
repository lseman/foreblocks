"""Energy-balance and complete-row neighborhood invariants."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from projects.scheduling.job_profile import battery_trace, curtailed_power
from projects.scheduling.milp import ONTSMilp
from projects.scheduling.profile_portfolio import solve_profile_portfolio
from projects.scheduling.research.datasets import witness_instance


def test_curtailment_matches_recursive_energy_balance():
    # k=0.1 SoC/W; saturation with no excess must not count as free energy.
    config = SimpleNamespace(ef=6., q=1., v_bat=1., soc_inicial=.8)
    solar = np.array([2., 1., 5., 0., 4.])
    usage = np.array([0., 0., 2., 3., 1.])
    problem = SimpleNamespace(config=config, instance=SimpleNamespace(recurso_p=torch.tensor(solar)))
    expected_soc, expected_spill = [], []
    level = config.soc_inicial
    for supply, demand in zip(solar, usage):
        uncapped = level + .1 * (supply-demand)
        expected_spill.append(max(0., uncapped-1.) / .1)
        level = min(1., uncapped)
        expected_soc.append(level)
    np.testing.assert_allclose(battery_trace(problem, usage), expected_soc)
    np.testing.assert_allclose(curtailed_power(problem, usage), expected_spill, atol=1e-12)
    assert expected_soc[0] == pytest.approx(1.)
    assert expected_spill[0] == pytest.approx(0.)
    # A three-watt job at the second slot has only one watt of free power.
    assert max(0., 3.-curtailed_power(problem, usage)[1]) == pytest.approx(2.)


@pytest.mark.parametrize('strategy',['whole_jobs','time_window'])
def test_structured_repair_mask(monkeypatch,strategy):
    import projects.scheduling.profile_portfolio as module
    instance, witness = witness_instance(seed=815771, regime='balanced')
    model = ONTSMilp(instance)
    initial = model.config.validate(instance, witness).objective
    seen = []
    monkeypatch.setattr(module, 'reprofile', lambda p,x,jobs,rng,**kw: (x,initial,dict(jobs=jobs,gain=0.)))
    def solve(self, *, incumbent, free_mask, **kwargs):
        seen.append(free_mask.numpy().copy())
        return SimpleNamespace(schedule=incumbent, objective=initial, status='optimal')
    monkeypatch.setattr(ONTSMilp, 'solve', solve)
    result = solve_profile_portfolio(instance,initial_schedule=witness,patience=1,
                                    repair_strategy=strategy,time_limit=1.,max_iterations=2)
    assert len(seen) == 1
    legal = (model.upper[:model.n_x] > 0).reshape(model.J,model.T)
    mask = seen[0]
    assert mask.any()
    if strategy=='whole_jobs':
        for row, allowed in zip(mask, legal):
            assert not row.any() or np.array_equal(row,allowed)
    else:
        columns=np.flatnonzero(mask.any(axis=0))
        assert np.array_equal(columns,np.arange(columns[0],columns[-1]+1))
        np.testing.assert_array_equal(mask[:,columns],legal[:,columns])
    assert result.objective == initial
    assert model.config.validate(instance,result.schedule).feasible


def test_refined_modes_keep_valid_nondecreasing_incumbents():
    instance,witness = witness_instance(seed=815771,regime='balanced')
    initial = ONTSMilp(instance).config.validate(instance,witness).objective
    result=solve_profile_portfolio(instance,initial_schedule=witness,time_limit=.5,
        patience=2,surplus_weight=.05,surplus_mode='curtailment',repair_strategy='whole_jobs')
    assert result.objective >= initial
    assert ONTSMilp(instance).config.validate(instance,result.schedule).feasible
    values=[e['objective'] for e in result.history if e.get('objective') is not None]
    assert values == sorted(values)


def test_refinement_inference_preserves_failed_instances_and_pairing():
    from projects.scheduling.research.component_refinements import VARIANTS, analyze, report
    plan=dict(count=2,seeds=[1,2],budget=1.,phase='test')
    rows=[]
    for i in range(2):
        for seed in plan['seeds']:
            for method in VARIANTS:
                value=(100. if method=='time_window' else 90.) if i==0 else None
                rows.append(dict(instance=i,seed=seed,method=method,objective=value,
                    history=[] if value is None else [dict(time=0.,objective=value)],
                    shared_start_hash=f'{i}-{seed}',initial_objective=value,feasible=value is not None,error=None))
    result=analyze(rows,plan)
    assert result['all_unsolved']==1
    assert result['methods']['time_window']['score_pp']==50.
    assert result['comparisons'][0]['score_pp']['mean_percent']==pytest.approx(5.)
    assert 'Simultaneous 95%' in report(result,plan)
    with pytest.raises(ValueError,match='Complete unique'):analyze(rows[:-1],plan)
    rows[0]['shared_start_hash']='wrong'
    with pytest.raises(AssertionError):analyze(rows,plan)
