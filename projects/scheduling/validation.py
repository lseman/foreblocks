"""Independent complete-schedule validation for the base ONTS formulation.

Auxiliary valid inequalities from original.py are deliberately excluded: they
need a separate validity audit. Solar power is already in watts (no rescaling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import torch

from .environments.onts_env import ONTSInstance


@dataclass(frozen=True)
class ScheduleValidation:
    objective: float
    violations: tuple[str, ...]
    soc: tuple[float, ...]

    @property
    def feasible(self) -> bool:
        return not self.violations


def validate_schedule(
    instance: ONTSInstance,
    schedule: torch.Tensor,
    *,
    soc_inicial: float = 0.7,
    limite_inferior: float = 0.0,
    ef: float = 0.9,
    v_bat: float = 3.6,
    q: float = 5.0,
    bat_usage: float = 5.0,
    allow_truncated_final_run: bool = True,
    tolerance: float = 1e-6,
) -> ScheduleValidation:
    """Check one [jobs, horizon] binary schedule without trusting env masks.

    Maximum periods use every full sliding window, including the first and
    last. The reference permits a final run shorter than min_cpu_time.
    Battery feasibility uses the maximum attainable SoC with solar curtailment;
    by monotonicity a lower trajectory cannot rescue an energy deficit.
    Malformed inputs raise ValueError; scheduling violations are returned.
    """
    params = (soc_inicial, limite_inferior, ef, v_bat, q, bat_usage, tolerance)
    if not all(math.isfinite(p) for p in params):
        raise ValueError("Battery parameters and tolerance must be finite")
    if not 0 <= limite_inferior <= soc_inicial <= 1:
        raise ValueError("Expected 0 <= limite_inferior <= soc_inicial <= 1")
    if ef <= 0 or v_bat <= 0 or q <= 0 or bat_usage < 0 or tolerance < 0:
        raise ValueError("Invalid battery parameters or tolerance")
    J, T = instance.J, instance.T
    if J <= 0 or T <= 0:
        raise ValueError("Instance must have jobs and time slots")
    fields = (
        "priority",
        "uso_p",
        "min_statup",
        "max_statup",
        "min_cpu_time",
        "max_cpu_time",
        "min_periodo_job",
        "max_periodo_job",
        "win_min",
        "win_max",
        "recurso_p",
    )
    data = {}
    for name in fields:
        values = getattr(instance, name).detach().cpu().double()
        if (
            values.shape != (T if name == "recurso_p" else J,)
            or not torch.isfinite(values).all()
        ):
            raise ValueError(f"Invalid instance field {name}")
        if name not in ("priority", "uso_p", "recurso_p") and not torch.equal(
            values, values.round()
        ):
            raise ValueError(f"Expected integer values for {name}")
        data[name] = values.tolist()
    if any(v < 0 for name in ("uso_p", "recurso_p") for v in data[name]):
        raise ValueError("Power must be nonnegative")
    for j in range(J):
        for lo, hi, minimum in (
            ("min_statup", "max_statup", 0),
            ("min_cpu_time", "max_cpu_time", 1),
            ("min_periodo_job", "max_periodo_job", 1),
        ):
            if not minimum <= data[lo][j] <= data[hi][j]:
                raise ValueError(f"Invalid {lo}/{hi} for job {j}")
        # Bundled instances use windows ending beyond the finite horizon.
        # Membership is evaluated only on existing slots, without rescaling.
        if not 0 <= data["win_min"][j] <= data["win_max"][j]:
            raise ValueError(f"Invalid execution window for job {j}")
    x = torch.as_tensor(schedule).detach().cpu().double()
    if x.shape != (J, T) or not torch.isfinite(x).all():
        raise ValueError(f"Expected finite schedule with shape {(J, T)}")
    if not ((x == 0) | (x == 1)).all():
        raise ValueError("Schedule must be binary; do not threshold silently")
    violations = []
    for j in range(J):
        row = x[j].tolist()
        starts = [t for t in range(T) if row[t] and (t == 0 or not row[t - 1])]
        if not data["min_statup"][j] <= len(starts) <= data["max_statup"][j]:
            violations.append(f"startup_count: job={j}")
        if any(
            row[t] and not data["win_min"][j] <= t < data["win_max"][j]
            for t in range(T)
        ):
            violations.append(f"execution_window: job={j}")
        min_gap = int(data["min_periodo_job"][j])
        if any(b - a < min_gap for a, b in pairwise(starts)):
            violations.append(f"minimum_period: job={j}")
        period = int(data["max_periodo_job"][j])
        phi = [int(t in starts) for t in range(T)]
        if any(sum(phi[t : t + period]) == 0 for t in range(T - period + 1)):
            violations.append(f"maximum_period: job={j}")
        for start in starts:
            end = start
            while end < T and row[end]:
                end += 1
            minimum = int(data["min_cpu_time"][j])
            if allow_truncated_final_run:
                minimum = min(minimum, T - start)
            if end - start < minimum:
                violations.append(f"minimum_run: job={j}, time={start}")
            if end - start > data["max_cpu_time"][j]:
                violations.append(f"maximum_run: job={j}, time={start}")
    usage = (
        (x * torch.tensor(data["uso_p"], dtype=torch.float64)[:, None]).sum(0).tolist()
    )
    soc, trace = soc_inicial, []
    for t, power in enumerate(usage):
        solar = data["recurso_p"][t]
        if power > solar + bat_usage * v_bat + tolerance:
            violations.append(f"power: time={t}")
        soc = min(1.0, soc + ef * (solar - power) / (q * v_bat * 60))
        trace.append(soc)
        if soc < limite_inferior - tolerance:
            violations.append(f"battery_soc: time={t}")
    objective = float(
        (x * torch.tensor(data["priority"], dtype=torch.float64)[:, None]).sum()
    )
    return ScheduleValidation(objective, tuple(violations), tuple(trace))
