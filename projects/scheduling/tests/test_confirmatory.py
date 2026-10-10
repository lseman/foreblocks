"""Inference cannot manufacture independent samples or omit failed pairs."""

import pytest

from projects.scheduling.research.confirmatory import (
    METHODS,
    analyze,
    infer,
    power_requirements,
)


def test_constant_effect_and_no_effect():
    positive = infer([2, 2, 2])
    assert positive["simultaneous_ci95"] == [2, 2]
    assert positive["two_sided_p"] == 0
    assert infer([0, 0, 0])["two_sided_p"] == 1
    assert infer([-2, -2, -2])["mean_percent"] < 0


def test_seed_means_are_not_independent_instances():
    entries = [dict(id=f"i{i}", split="test" if i < 2 else "ood") for i in range(4)]
    rows = [
        dict(
            instance_id=e["id"],
            search_seed=s,
            method=m,
            score=102 if m == "ranked" else 100,
            validation_passed=True,
            feasible=True,
        )
        for e in entries
        for s in (1, 2, 3)
        for m in METHODS
    ]
    result = analyze(rows, entries, seeds=(1, 2, 3))
    assert result["claim_supported"]
    assert all(c["instances"] == 4 for c in result["comparisons"])
    rows[0].update(error="failure", score=0, validation_passed=False, feasible=False)
    assert not analyze(rows, entries, seeds=(1, 2, 3))["claim_supported"]
    with pytest.raises(ValueError, match="Complete unique"):
        analyze(rows[:-1], entries, seeds=(1, 2, 3))
    with pytest.raises(ValueError, match="Complete unique"):
        analyze([*rows, rows[0]], entries, seeds=(1, 2, 3))


def test_mean_advantage_is_not_enough():
    result = infer([-10, -10, 30])
    assert result["mean_percent"] > 0
    assert result["simultaneous_ci95"][0] < 0
    assert result["two_sided_p"] > 0.05
    assert power_requirements(10)["1.0"] > power_requirements(10)["3.0"]


def test_significance_does_not_imply_practical_superiority():
    entries = [dict(id=f"i{i}", split="test" if i < 2 else "ood") for i in range(4)]
    rows = [
        dict(
            instance_id=e["id"],
            search_seed=s,
            method=m,
            score=100.5 if m == "ranked" else 100,
            validation_passed=True,
            feasible=True,
        )
        for e in entries
        for s in (1, 2, 3)
        for m in METHODS
    ]
    result = analyze(rows, entries, seeds=(1, 2, 3))
    assert all(c["statistical_superiority"] for c in result["comparisons"])
    assert not any(c["practical_superiority_1_percent"] for c in result["comparisons"])
    assert not result["claim_supported"]
