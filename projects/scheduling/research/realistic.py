"""Bundled-like ONTS instances resampled from the bundled benchmark.

Each instance takes a horizon T from the bundled set, the real solar profile of
one bundled instance with that horizon, and J job rows (all job parameters of a
row kept together) sampled without replacement from bundled instances with the
same horizon. Priorities are a random permutation of 1..J, as in most bundled
instances; power draws get a +/-10% jitter. There is no feasibility filter, so
some instances may have no feasible schedule; those score zero for every method.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import torch

from ..environments.onts_env import ONTSInstance

BUNDLED = Path(__file__).resolve().parents[1] / "instances" / "examples"
JOB_FIELDS = (
    "uso_p",
    "min_statup",
    "max_statup",
    "min_cpu_time",
    "max_cpu_time",
    "min_periodo_job",
    "max_periodo_job",
    "win_min",
    "win_max",
)
JOB_COUNTS = (9, 13, 18, 20, 22, 24)


@cache
def _pools():
    pools: dict[int, dict] = {}
    for path in sorted(BUNDLED.glob("*.json")):
        data = json.loads(path.read_text())
        T = int(data["tamanho"])
        pool = pools.setdefault(T, {"solar": [], "rows": []})
        pool["solar"].append(np.asarray(data["recurso_p"], dtype=float)[:T])
        for j in range(int(data["jobs"])):
            pool["rows"].append({f: data[f][j] for f in JOB_FIELDS})
    return pools


def realistic_instance(seed: int, *, T: int | None = None, J: int | None = None):
    rng = np.random.default_rng(seed)
    pools = _pools()
    T = int(rng.choice(sorted(pools))) if T is None else T
    pool = pools[T]
    J = int(rng.choice(JOB_COUNTS)) if J is None else J
    J = min(J, len(pool["rows"]))
    solar = pool["solar"][int(rng.integers(len(pool["solar"])))]
    rows = [pool["rows"][i] for i in rng.choice(len(pool["rows"]), J, replace=False)]

    def column(field, dtype=torch.int64):
        return torch.tensor([row[field] for row in rows], dtype=dtype)

    draw = column("uso_p", torch.float64) * torch.from_numpy(rng.uniform(0.9, 1.1, J))
    return ONTSInstance(
        priority=torch.from_numpy(rng.permutation(J) + 1.0),
        uso_p=draw,
        min_statup=column("min_statup"),
        max_statup=column("max_statup"),
        min_cpu_time=column("min_cpu_time"),
        max_cpu_time=column("max_cpu_time"),
        min_periodo_job=column("min_periodo_job"),
        max_periodo_job=column("max_periodo_job"),
        win_min=column("win_min"),
        win_max=torch.clamp(column("win_max"), max=T),
        recurso_p=torch.from_numpy(solar.copy()),
    )
