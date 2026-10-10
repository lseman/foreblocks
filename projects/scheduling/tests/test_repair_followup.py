"""Margin direction, multiplicity and failed-population behavior."""
import numpy as np
import pytest

from projects.scheduling.research.repair_followup import COMPARISONS, VARIANTS, analyze, noninferiority


def test_noninferiority_uses_lower_bound_and_one_sided_null():
    good = noninferiority([-.1, -.2, -.15, -.25] * 5, margin=.5)
    assert good["simultaneous_ci95"][0] > -.5
    assert good["one_sided_p"] < .001
    bad = noninferiority([-.7, -.8, -.75, -.85] * 5, margin=.5)
    assert bad["simultaneous_ci95"][1] < -.5
    assert bad["one_sided_p"] > .99
    noisy = noninferiority([-3., 3.] * 10, margin=.5)
    assert noisy["mean_percent"] == 0.
    assert noisy["simultaneous_ci95"][0] < -.5
    assert noisy["one_sided_p"] > .05
    with pytest.raises(ValueError, match="margin"):
        noninferiority([0., 0.], margin=-.1)


def rows_and_plan(*, loss=.2, failed=False):
    plan = dict(count=20, seeds=[809, 907], budget=1., variants=VARIANTS,
                comparisons=COMPARISONS, practical_threshold_pp=1., noninferiority_margin_pp=.5,
                checkpoints=[1., 3., 10., 30.])
    rows = []
    for i in range(plan["count"]):
        for seed in plan["seeds"]:
            for m in VARIANTS:
                value = None if failed or i == 19 else {"reference":100., "dp_only":100.-loss, "time_window":100.1}[m]
                history = [] if value is None else [dict(time=.2, objective=60., stage="initialization"),
                    dict(time=.5, objective=60., stage="handoff"),
                    dict(time=.8, objective=value, stage="improvement"),
                    dict(time=1.1, objective=120., stage="improvement")]
                rows.append(dict(instance=i, seed=seed, method=m, objective=value,
                    feasible=value is not None, history=history, shared_start_hash=f"{i}-{seed}",
                    initialization_runtime=.5, improvement_budget=.5, initial_objective=60. if value else None,
                    error=None, validation_passed=True, runtime=1.1))
    return rows, plan


def test_paired_followup_retains_failures_and_tests_three_hypotheses():
    rows, plan = rows_and_plan()
    result = analyze(rows, plan)
    assert result["primary_family_size"] == 3
    assert result["all_unsolved"] == 1
    assert result["noninferiority"]["noninferiority_established"]
    assert result["noninferiority"]["mean_percent"] < 0
    assert result["methods"]["reference"]["score_pp"] < 100.
    assert result["methods"]["reference"]["checkpoint_scores_pp"]["30.0"] == result["methods"]["reference"]["score_pp"]
    expected = result["comparisons"][0]["score_pp"]
    np.testing.assert_allclose(result["noninferiority"]["simultaneous_ci95"],
                               -np.asarray(expected["simultaneous_ci95"])[::-1])
    with pytest.raises(ValueError, match="Complete unique"):
        analyze(rows[:-1], plan)


def test_noninferiority_rejects_excess_loss_and_uninformative_population():
    rows, plan = rows_and_plan(loss=1.)
    assert not analyze(rows, plan)["noninferiority"]["noninferiority_established"]
    rows, plan = rows_and_plan(failed=True)
    assert not analyze(rows, plan)["noninferiority"]["noninferiority_established"]
