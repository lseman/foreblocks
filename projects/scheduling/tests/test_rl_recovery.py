from dataclasses import replace
from itertools import product

import numpy as np
import torch

from projects.scheduling.milp import ONTSConfig
from projects.scheduling.rl_construct import ConstructionEnv
from projects.scheduling.rl_recovery import RecoveryPolicy, completion_probe, recover_construct
from projects.scheduling.environments.onts_env import onts_instance_to_json_data
from projects.scheduling.research.oracle import oracle_check
from test_rl_construct import tiny


def bottleneck():
    return replace(tiny(), recurso_p=torch.ones(6), win_max=torch.tensor([6, 2]),
                   max_cpu_time=torch.tensor([2, 2]))


def test_completion_rejects_locally_safe_blocking_action():
    env = ConstructionEnv(bottleneck(), ONTSConfig(bat_usage=0))
    blocked = next(c for c in env.candidates() if c.task == 0 and c.row[0])
    result = completion_probe(env, blocked)
    assert result["status"] == "reject" and result["task"] == 1
    safe = next(c for c in env.candidates() if c.task == 1)
    assert completion_probe(env, safe)["status"] == "not_ruled_out"


def test_backtracking_finds_feasible_completion_without_lookahead():
    row, _, _ = recover_construct(bottleneck(), mode="priority", lookahead=False,
                                  config=ONTSConfig(bat_usage=0), budget=.5)
    assert row["success"] and row["backtracks"] > 0
    assert ONTSConfig(bat_usage=0).validate(bottleneck(), torch.tensor(row["schedule"])).feasible


def test_probe_does_not_reject_any_exhaustively_feasible_completion():
    instance = bottleneck(); config = ONTSConfig(bat_usage=0)
    env = ConstructionEnv(instance, config)
    for choice in env.candidates():
        trial = np.zeros((2, 6), dtype=int); trial[choice.task] = choice.row
        other = 1-choice.task
        exists = False
        for bits in product((0, 1), repeat=6):
            trial[other] = bits
            if config.validate(instance, torch.tensor(trial)).feasible:
                exists = True; break
        if exists:
            assert completion_probe(env, choice)["status"] != "reject"


def test_energy_bound_can_reject_jointly_impossible_remaining_load():
    instance = replace(tiny(), recurso_p=torch.zeros(6), min_cpu_time=torch.ones(2, dtype=torch.int),
                       max_cpu_time=torch.ones(2, dtype=torch.int))
    # Each task individually fits, but both minimum one-slot runs do not.
    config = ONTSConfig(q=.009, soc_inicial=.7, limite_inferior=0., bat_usage=10)
    env = ConstructionEnv(instance, config)
    choice = env.candidates()[0]
    assert completion_probe(env, choice)["reason"] == "minimum_energy_exceeds_capacity"


def test_unknown_probe_and_timeout_are_not_infeasibility_certificates():
    env = ConstructionEnv(tiny())
    choice = env.candidates()[0]
    assert completion_probe(env, choice, deadline=0)["status"] == "unknown"
    row, _, _ = recover_construct(tiny(), mode="uniform", budget=1e-12)
    assert not row["success"] and not row["exhausted"]


def test_differentiable_search_and_incumbent_preservation():
    torch.manual_seed(2)
    model = RecoveryPolicy()
    row, logs, entropies = recover_construct(tiny(), model, seed=3, budget=.3, training=True, max_decisions=12)
    assert row["success"] and logs
    (-torch.stack(logs).sum()-.001*torch.stack(entropies).mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    credited = [e["objective"] for e in row["events"] if e["kind"] == "complete" and e["credited"]]
    assert row["objective"] == max(credited)
    assert row["time_to_first_feasible"] <= row["cutoff"]


def test_late_completion_cannot_escape_through_partial_fallback(monkeypatch):
    import projects.scheduling.rl_recovery as recovery
    real_clock = recovery.perf_counter
    offset = [0.]
    real_validate = ONTSConfig.validate
    def delayed_validate(self, instance, schedule):
        report = real_validate(self, instance, schedule)
        if report.feasible:
            offset[0] = .475  # After acceptance cutoff, before final runtime limit.
        return report
    monkeypatch.setattr(recovery, "perf_counter", lambda: real_clock()+offset[0])
    monkeypatch.setattr(ONTSConfig, "validate", delayed_validate)
    row, _, _ = recover_construct(tiny(), mode="uniform", budget=.5, max_decisions=2)
    assert not row["success"] and not row["feasible"]
    assert row["completed"] < tiny().J
    assert any(e["kind"] == "complete" and not e["credited"] for e in row["events"])


def test_memoization_skips_task_order_duplicates_and_preserves_quality():
    a, _, _ = recover_construct(tiny(), mode="uniform", budget=.5, max_decisions=256, memoize=True)
    b, _, _ = recover_construct(tiny(), mode="uniform", budget=.5, max_decisions=256, memoize=False)
    assert a["success"] and b["success"] and a["objective"] == b["objective"]
    assert a["memoized"] > 0 and a["completions"] < b["completions"]


def test_completion_masks_against_independent_exhaustive_boundary_cases():
    for truncated, period, minimum in product((False, True), (3, 5), (0, 1)):
        instance = replace(tiny(), recurso_p=torch.ones(4), win_max=torch.tensor([4, 3]),
            min_statup=torch.full((2,),minimum), max_statup=torch.tensor([2, 2]),
            min_cpu_time=torch.tensor([2, 1]), max_cpu_time=torch.tensor([3, 2]),
            max_periodo_job=torch.full((2,),period))
        config = ONTSConfig(allow_truncated_final_run=truncated, bat_usage=0, q=.02)
        data = onts_instance_to_json_data(instance)
        feasible = []
        for bits in product((0, 1), repeat=8):
            schedule = np.array(bits).reshape(2, 4)
            if oracle_check(data, schedule.tolist(), config)[0]:
                feasible.append(schedule)
        env = ConstructionEnv(instance, config)
        for choice in env.candidates():
            if any(np.array_equal(x[choice.task], choice.row) for x in feasible):
                assert completion_probe(env, choice)["status"] != "reject"
