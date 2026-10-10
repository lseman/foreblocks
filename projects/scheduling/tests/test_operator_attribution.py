"""Operator exclusion, paired populations and total-budget inference."""
from types import SimpleNamespace

import pytest

from projects.scheduling.milp import ONTSMilp
from projects.scheduling.profile_portfolio import solve_profile_portfolio
from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.research.operator_attribution import COMPARISONS, VARIANTS, analyze


def test_milp_only_never_calls_profile_and_keeps_valid_start(monkeypatch):
    import projects.scheduling.profile_portfolio as module
    instance, witness = witness_instance(seed=815771, regime="balanced")
    objective = ONTSMilp(instance).config.validate(instance, witness).objective
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled DP operator was invoked")

    def restricted(self, *, incumbent, free_mask, **kwargs):
        calls.append(free_mask)
        return SimpleNamespace(schedule=incumbent, objective=objective, status="optimal")

    monkeypatch.setattr(module, "reprofile", forbidden)
    monkeypatch.setattr(ONTSMilp, "solve", restricted)
    result = solve_profile_portfolio(instance, initial_schedule=witness,
                                     enable_profiles=False, time_limit=2., max_iterations=3)
    assert len(calls) == 3
    assert all(e.get("operator") != "profile" for e in result.history)
    assert ONTSMilp(instance).config.validate(instance, result.schedule).feasible
    assert result.objective == objective


def test_cannot_disable_both_operators():
    with pytest.raises(ValueError, match="At least one"):
        solve_profile_portfolio(None, enable_profiles=False, enable_repairs=False)


def synthetic_rows():
    rows = []
    for i in range(2):
        for seed in (601, 701):
            for m in VARIANTS:
                value = (100. if m == "reference" else 80.) if i == 0 else None
                history = [] if value is None else [
                    dict(time=.2, objective=60., stage="initialization"),
                    dict(time=.5, objective=60., stage="handoff"),
                    dict(time=.8, objective=value, stage="improvement"),
                    # A late observation must not improve credited score or AUC.
                    dict(time=1.1, objective=120., stage="improvement")]
                rows.append(dict(instance=i, seed=seed, method=m, objective=value,
                                 feasible=value is not None, history=history,
                                 shared_start_hash=f"{i}-{seed}", initialization_runtime=.5,
                                 improvement_budget=.5, initial_objective=60. if i == 0 else None,
                                 error=None, validation_passed=True))
    return rows


def test_inference_retains_failed_starts_and_initialization_timeline():
    plan = dict(count=2, seeds=[601, 701], budget=1., variants=VARIANTS,
                comparisons=COMPARISONS, practical_threshold_pp=1.)
    rows = synthetic_rows()
    result = analyze(rows, plan)
    assert result["all_unsolved"] == 1
    assert result["methods"]["reference"]["score_pp"] == 50.
    assert result["methods"]["reference"]["conditional_score_pp"] == 100.
    assert result["methods"]["reference"]["auc_pp"] == pytest.approx(28.)
    assert result["comparisons"][0]["score_pp"]["mean_percent"] == 10.
    with pytest.raises(ValueError, match="Complete unique"):
        analyze(rows[:-1], plan)
    rows[0]["improvement_budget"] = .6
    with pytest.raises(ValueError, match="Paired starts"):
        analyze(rows, plan)


def test_execution_errors_block_primary_inference():
    rows = synthetic_rows()
    rows[0]["error"] = "solver failed"
    plan = dict(count=2, seeds=[601, 701], variants=VARIANTS)
    with pytest.raises(ValueError, match="errors block"):
        analyze(rows, plan)


def test_all_five_methods_run_from_one_saved_schedule(tmp_path, monkeypatch):
    import projects.scheduling.research.operator_attribution as study
    from projects.scheduling.research.datasets import instance_hash, source_hashes
    from projects.scheduling.research.portfolio_ablation import COMMON, save
    instance, witness = witness_instance(seed=815771, regime="balanced")
    initial = ONTSMilp(instance).config.validate(instance, witness).objective
    monkeypatch.setattr(study, "realistic_instance", lambda seed: instance)
    (tmp_path / "initializers").mkdir()
    save(tmp_path / "initializers" / "0-607.json",
         dict(schedule=witness.int().tolist(), objective=initial, usable=True,
              runtime=.05, instance_hash=instance_hash(instance),
              history=[dict(time=.02, objective=initial)]))
    plan = dict(base=66_000_000, instance_hashes=[instance_hash(instance)],
                source_hashes=source_hashes(), bundled_input_hashes=study.input_hashes(),
                variants=VARIANTS, order_seed=971017, budget=.55,
                common=COMMON, repair_time=.05)
    rows = study.run_pair((0, 607, plan, str(tmp_path)))
    assert len(rows) == 5
    assert all(r["error"] is None and r["feasible"] for r in rows)
    assert all(r["objective"] >= initial for r in rows)
    assert len({r["shared_start_hash"] for r in rows}) == 1
    assert len({r["improvement_budget"] for r in rows}) == 1
    assert all(r["history"][0]["stage"] == "initialization" for r in rows)
