from dataclasses import replace

import numpy as np
import pytest
import torch

from projects.scheduling.environments.onts_env import ONTSInstance, onts_instance_to_json_data
from projects.scheduling.milp import ONTSConfig
from projects.scheduling.research.oracle import oracle_check
from projects.scheduling.rl_construct import ConstructionEnv, ConstructPolicy, construct


def tiny():
    return ONTSInstance(
        priority=torch.tensor([2., 1.]), uso_p=torch.tensor([1., 1.]),
        min_statup=torch.tensor([1, 1]), max_statup=torch.tensor([1, 1]),
        min_cpu_time=torch.tensor([2, 2]), max_cpu_time=torch.tensor([3, 3]),
        min_periodo_job=torch.tensor([2, 2]), max_periodo_job=torch.tensor([7, 7]),
        win_min=torch.tensor([0, 0]), win_max=torch.tensor([6, 6]),
        recurso_p=torch.ones(6) * 3,
    )


def test_empty_schedule_is_infeasible_but_cold_constructor_solves():
    instance = tiny()
    assert not ONTSConfig().validate(instance, torch.zeros(2, 6)).feasible
    row, _, _ = construct(instance, None, mode="priority", budget=2.)
    assert row["success"] and row["completed"] == 2
    valid, objective, _ = oracle_check(onts_instance_to_json_data(instance), row["schedule"], ONTSConfig())
    assert valid and objective == row["objective"]


def test_resource_dead_end_is_reported_as_failure():
    instance = replace(tiny(), uso_p=torch.ones(2) * 1000, recurso_p=torch.zeros(6))
    row, _, _ = construct(instance, None, mode="uniform")
    assert not row["success"] and not row["feasible"]
    assert row["objective"] is None and row["completed"] == 0


def test_periodic_profiles_and_battery_are_independently_validated():
    instance = replace(tiny(), max_periodo_job=torch.tensor([4, 4]), max_statup=torch.tensor([3, 3]))
    row, _, _ = construct(instance, None, mode="uniform", seed=12)
    valid, _, _ = oracle_check(onts_instance_to_json_data(instance), row["schedule"], ONTSConfig())
    assert valid == row["feasible"]
    assert row["success"]


def test_learning_has_finite_gradients_and_reproducible_actions():
    torch.manual_seed(1)
    model = ConstructPolicy()
    first, logs, entropies = construct(tiny(), model, seed=23, training=True)
    second, _, _ = construct(tiny(), model, seed=23)
    assert first["schedule"] == second["schedule"]
    loss = -torch.stack(logs).sum() - .001 * torch.stack(entropies).sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_candidates_deduplicated_and_locally_resource_safe():
    env = ConstructionEnv(tiny())
    choices = env.candidates()
    keys = [(c.task, c.row.tobytes()) for c in choices]
    assert len(keys) == len(set(keys))
    for choice in choices:
        trial = env.schedule.copy(); trial[choice.task] = choice.row
        assert env.resource_safe(trial)
    env.place(choices[0])
    with pytest.raises(ValueError):
        env.place(choices[0])


def test_timeout_never_credits_partial_schedule():
    row, _, _ = construct(tiny(), None, mode="uniform", budget=1e-12)
    assert not row["success"] and row["completed"] == 0
    with pytest.raises(ValueError):
        construct(tiny(), None, mode="uniform", budget=np.nan)


def test_locally_safe_action_can_block_a_known_feasible_completion():
    instance = replace(tiny(), recurso_p=torch.ones(6),
                       win_max=torch.tensor([6, 2]), max_cpu_time=torch.tensor([2, 2]))
    config = ONTSConfig(bat_usage=0)
    witness = torch.tensor([[0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]])
    assert config.validate(instance, witness).feasible
    env = ConstructionEnv(instance, config)
    early = next(c for c in env.candidates() if c.task == 0 and c.row[0])
    env.place(early)
    assert not env.candidates()  # Local safety is not completion feasibility.


def test_battery_deficit_rejects_profiles_despite_instantaneous_power():
    instance = replace(tiny(), recurso_p=torch.zeros(6), uso_p=torch.ones(2) * 5)
    config = ONTSConfig(q=.001, soc_inicial=.1, bat_usage=10)
    env = ConstructionEnv(instance, config)
    assert not env.candidates()
