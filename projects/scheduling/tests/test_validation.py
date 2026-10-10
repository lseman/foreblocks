import pytest
import torch

from projects.scheduling.environments.onts_env import ONTSInstance
from projects.scheduling.validation import validate_schedule


def instance(**changes):
    data = dict(
        priority=[2],
        uso_p=[1],
        min_statup=[1],
        max_statup=[2],
        min_cpu_time=[2],
        max_cpu_time=[3],
        min_periodo_job=[3],
        max_periodo_job=[5],
        win_min=[0],
        win_max=[5],
        recurso_p=[1] * 5,
    )
    data.update(changes)
    return ONTSInstance(**{k: torch.tensor(v) for k, v in data.items()})


def test_valid_objective_and_startups():
    result = validate_schedule(instance(), torch.tensor([[1, 1, 0, 1, 1]]))
    assert result.feasible
    assert result.objective == 8
    assert result.soc[-1] >= 0.7


def test_maximum_period_includes_boundary_windows():
    result = validate_schedule(
        instance(max_periodo_job=[3]), torch.tensor([[0, 0, 0, 1, 1]])
    )
    assert "maximum_period: job=0" in result.violations


def test_horizon_semantics_are_explicit():
    x = torch.tensor([[0, 0, 0, 0, 1]])
    assert validate_schedule(instance(), x).feasible
    assert (
        "minimum_run: job=0, time=4"
        in validate_schedule(instance(), x, allow_truncated_final_run=False).violations
    )


def test_window_and_run_violations():
    result = validate_schedule(instance(win_max=[3]), torch.ones(1, 5))
    assert "execution_window: job=0" in result.violations
    assert "maximum_run: job=0, time=0" in result.violations


def test_curtailment_and_energy_deficit():
    assert validate_schedule(
        instance(recurso_p=[100] * 5), torch.tensor([[1, 1, 0, 0, 0]])
    ).feasible
    result = validate_schedule(
        instance(uso_p=[18], recurso_p=[0] * 5),
        torch.tensor([[1, 1, 0, 0, 0]]),
        soc_inicial=0.001,
    )
    assert "battery_soc: time=0" in result.violations
    assert result.soc[0] < 0  # Never conceal deficits by clamping at zero.


def test_power_limit():
    result = validate_schedule(
        instance(uso_p=[20], recurso_p=[0] * 5), torch.tensor([[1, 1, 0, 0, 0]])
    )
    assert "power: time=0" in result.violations


@pytest.mark.parametrize(
    "x", [torch.zeros(2, 5), torch.full((1, 5), 0.5), torch.full((1, 5), float("nan"))]
)
def test_reject_malformed_schedule(x):
    with pytest.raises(ValueError):
        validate_schedule(instance(), x)
