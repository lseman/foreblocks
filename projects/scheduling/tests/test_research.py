"""Verification of independent oracle, split integrity, statistics and protocol."""

import json
from dataclasses import asdict

import pytest
import torch

from projects.scheduling.environments.onts_env import ONTSInstance, load_onts_instance
from projects.scheduling.milp import ONTSConfig, ONTSMilp
from projects.scheduling.neighborhood import GraphBuilder, NeighborhoodPolicy
from projects.scheduling.research.datasets import (
    build_suite,
    instance_hash,
    witness_instance,
)
from projects.scheduling.research.oracle import audit_instance, random_tiny_instance
from projects.scheduling.research.protocol import (
    AblatedPolicy,
    append_record,
    objectives_at_budget,
    read_jsonl,
)
from projects.scheduling.research.statistics import (
    bootstrap_mean,
    holm_adjust,
    normalized_auc,
    paired_comparison,
)


def test_report_handles_instances_without_feasible_references(tmp_path, monkeypatch):
    from projects.scheduling.research import report

    monkeypatch.setattr(report, "make_figures", lambda *args: None)
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    documents = {
        "plan.json": {"budgets": [1.0]},
        "audit.json": {
            "schedules": 1,
            "cases": 1,
            "disagreement_count": 0,
            "optimum_mismatches": 0,
        },
        "data/manifest.json": {"entries": [{"split": "bundled"}]},
        "completion.json": {"expected_runs": 1, "completed_runs": 1},
    }
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value))
    append_record(
        tmp_path / "references.jsonl",
        {
            "instance_id": "unknown",
            "reference_objective": None,
            "initial_objective": None,
            "setup_runtime": 0,
        },
    )
    append_record(
        tmp_path / "runs.jsonl",
        {
            "instance_id": "unknown",
            "method": "exact",
            "split": "bundled",
            "track": "cold",
            "budget": 1.0,
            "feasible": False,
            "objective": None,
            "history": [],
            "runtime": 1.0,
            "search_seed": 0,
        },
    )
    report.generate_report(tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text())
    summary = analysis["summaries"][0]
    assert summary["feasible"] == 0
    assert summary["runs"] == 1
    assert summary["auc_instances"] == 0
    assert summary["mean_auc_to_best_known"] is None


@pytest.mark.parametrize("seed", range(12))
def test_independent_oracle_agrees_on_every_schedule(seed):
    report = audit_instance(*random_tiny_instance(seed + 8000))
    assert report["disagreement_count"] == 0
    assert report["optimum_matches"]


@pytest.mark.parametrize("regime", ["balanced", "energy", "periodic"])
@pytest.mark.parametrize("large", [False, True])
def test_generated_witness_is_feasible_without_filtering(regime, large):
    instance, witness = witness_instance(seed=1234, regime=regime, large=large)
    assert ONTSConfig().validate(instance, witness).feasible


def test_splits_are_content_disjoint_and_hash_survives_loading(tmp_path):
    manifest = build_suite(
        tmp_path,
        train_per_regime=1,
        validation_per_regime=1,
        test_per_regime=1,
        ood_per_regime=1,
    )
    hashes = [entry["hash"] for entry in manifest["entries"]]
    assert len(hashes) == len(set(hashes)) == 12
    for entry in manifest["entries"]:
        assert instance_hash(load_onts_instance(entry["path"])) == entry["hash"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_hash_ignores_name_but_detects_parameter_changes():
    instance, _ = witness_instance(seed=1, regime="energy")
    before = instance_hash(instance)
    instance.priority[0] += 1
    assert instance_hash(instance) != before


def test_bootstrap_constant_and_holm_known_values():
    result = bootstrap_mean([3, 3, 3], resamples=100)
    assert result["ci95"] == [3, 3]
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_statistics_average_seeds_within_instance():
    rows = []
    for instance, delta in [("a", 10), ("b", -10)]:
        for seed in range(3):
            for method, value in [("exact", 100), ("neural", 100 + delta)]:
                rows.append(
                    dict(
                        instance_id=instance,
                        search_seed=seed,
                        track="cold",
                        budget=3,
                        split="test",
                        method=method,
                        feasible=True,
                        objective=value,
                    )
                )
    comparison = paired_comparison(
        rows, method="neural", baseline="exact", track="cold", budget=3
    )
    assert comparison["relative_objective_change_percent"]["instances"] == 2
    assert comparison["relative_objective_change_percent"]["mean"] == 0
    assert comparison["wins"] == comparison["losses"] == 1


def test_infeasible_pairs_are_not_fabricated_objectives():
    rows = [
        dict(
            instance_id="a",
            search_seed=0,
            track="cold",
            budget=1,
            method="exact",
            feasible=False,
            objective=None,
        ),
        dict(
            instance_id="a",
            search_seed=0,
            track="cold",
            budget=1,
            method="neural",
            feasible=True,
            objective=10,
        ),
    ]
    result = paired_comparison(
        rows, method="neural", baseline="exact", track="cold", budget=1
    )
    assert result["relative_objective_change_percent"] is None
    assert result["paired_feasibility"]["method_only_feasible"] == 1


def test_auc_and_deadline_censoring():
    history = [
        {"time": 1, "objective": 5},
        {"time": 3, "objective": 10},
        {"time": 6, "objective": 20},
    ]
    assert normalized_auc(history, budget=5, reference=10) == pytest.approx(0.6)
    assert objectives_at_budget({"history": history}, 5) == 10
    assert objectives_at_budget({"history": history}, 0.5) is None


def test_interrupted_result_record_is_repaired(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_record(path, {"id": 1})
    with path.open("a") as f:
        f.write('{"id":')
    assert read_jsonl(path) == [{"id": 1}]
    append_record(path, {"id": 2})
    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]


def tiny():
    data = dict(
        priority=[3.0, 1.0],
        uso_p=[10.0, 5.0],
        min_statup=[0, 0],
        max_statup=[2, 2],
        min_cpu_time=[1, 1],
        max_cpu_time=[2, 2],
        min_periodo_job=[2, 2],
        max_periodo_job=[5, 5],
        win_min=[0, 0],
        win_max=[4, 4],
        recurso_p=[0.0, 2.0, 0.0, 3.0],
    )
    return ONTSInstance(**{key: torch.tensor(value) for key, value in data.items()})


def test_job_permutation_preserves_optimum_and_policy_logits():
    instance = tiny()
    p = ONTSMilp(instance)
    result = p.solve(time_limit=2)
    fields = asdict(instance)
    permuted = ONTSInstance(
        **{
            key: value.flip(0) if key != "recurso_p" else value
            for key, value in fields.items()
        }
    )
    other = ONTSMilp(permuted)
    assert other.solve(time_limit=2).objective == result.objective
    policy = NeighborhoodPolicy(hidden=16, layers=1)
    first = policy(GraphBuilder(p).build(result.schedule)).reshape(2, 4)
    second = policy(GraphBuilder(other).build(result.schedule.flip(0))).reshape(2, 4)
    assert torch.allclose(first.flip(0), second, atol=1e-5)


def test_ablation_does_not_mutate_shared_graph():
    p = ONTSMilp(tiny())
    result = p.solve(time_limit=2)
    graph = GraphBuilder(p).build(result.schedule)
    variables = graph.variables.clone()
    edges = graph.coefficients.clone()
    policy = NeighborhoodPolicy(hidden=16, layers=1)
    for wrapper in (
        AblatedPolicy(policy, no_future=True),
        AblatedPolicy(policy, no_edges=True),
    ):
        assert wrapper(graph).shape == (p.n_x,)
    assert torch.equal(graph.variables, variables)
    assert torch.equal(graph.coefficients, edges)
