"""MILP-backed ONTS environment.

This environment follows the optimization model in ``original.py``:
maximize priority-weighted task execution under startup, execution-window,
periodicity, run-length, resource, and battery SoC constraints.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ONTSInstance:
    """Parameters for one ONTS scheduling instance."""

    priority: torch.Tensor
    uso_p: torch.Tensor
    min_statup: torch.Tensor
    max_statup: torch.Tensor
    min_cpu_time: torch.Tensor
    max_cpu_time: torch.Tensor
    min_periodo_job: torch.Tensor
    max_periodo_job: torch.Tensor
    win_min: torch.Tensor
    win_max: torch.Tensor
    recurso_p: torch.Tensor

    @property
    def J(self) -> int:
        return int(self.priority.numel())

    @property
    def T(self) -> int:
        return int(self.recurso_p.numel())


def _as_1d_tensor(values: Any, *, dtype: torch.dtype, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.dim() != 1:
        tensor = tensor.flatten()
    return tensor


def _extract_jl_assignment(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*", text)
    if match is None:
        raise ValueError(f"Missing ONTS field {key!r}")
    start = match.end()
    while start < len(text) and text[start].isspace() and text[start] != "\n":
        start += 1
    if start < len(text) and text[start] == "[":
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "[":
                depth += 1
            elif text[idx] == "]":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        raise ValueError(f"Unclosed bracketed assignment for {key!r}")
    end = text.find("\n", start)
    if end < 0:
        end = len(text)
    return text[start:end]


INSTANCE_FORMAT = "onts-instance-v1"
INSTANCE_FIELDS = (
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
INSTANCE_INT_FIELDS = {
    "min_statup",
    "max_statup",
    "min_cpu_time",
    "max_cpu_time",
    "min_periodo_job",
    "max_periodo_job",
    "win_min",
    "win_max",
}


def parse_onts_jl(path: str | Path) -> dict[str, Any]:
    """Parse a Julia-style ONTS instance file into plain Python data."""
    text = Path(path).read_text()
    int_keys = {
        "jobs",
        "tamanho",
        "priority",
        "min_statup",
        "max_statup",
        "min_cpu_time",
        "max_cpu_time",
        "min_periodo_job",
        "max_periodo_job",
        "win_min",
        "win_max",
    }
    float_keys = {"recurso_p", "uso_p"}
    parsed: dict[str, list[float] | list[int]] = {}

    def numbers(raw: str, *, as_float: bool) -> list[float] | list[int]:
        found = re.findall(r"\d+(?:\.\d+)?", raw)
        if as_float:
            return [float(x) for x in found]
        return [int(float(x)) for x in found]

    def parse_key(key: str, *, as_float: bool) -> list[float] | list[int]:
        try:
            raw = _extract_jl_assignment(text, key)
        except ValueError as exc:
            raise ValueError(f"{exc} in {path}") from exc
        ones = re.fullmatch(
            r"\s*(?:(\d+(?:\.\d+)?)\s*\*\s*)?ones\((\w+|\d+)\)\s*",
            raw,
        )
        if ones is not None:
            factor = float(ones.group(1) or 1.0)
            count_ref = ones.group(2)
            count = int(parsed[count_ref][0]) if count_ref in parsed else int(count_ref)
            values = [factor] * count
            if as_float:
                return values
            return [int(v) for v in values]
        return numbers(raw, as_float=as_float)

    parsed["jobs"] = parse_key("jobs", as_float=False)
    parsed["tamanho"] = parse_key("tamanho", as_float=False)

    for key in sorted((int_keys | float_keys) - {"jobs", "tamanho"}):
        parsed[key] = parse_key(key, as_float=key in float_keys)

    J = int(parsed["jobs"][0])  # type: ignore[index]
    T = int(parsed["tamanho"][0])  # type: ignore[index]
    recurso_p = parsed["recurso_p"]  # type: ignore[assignment]
    normalizations = []
    if len(recurso_p) > T:
        normalizations.append(f"trimmed recurso_p from {len(recurso_p)} to {T}")
        recurso_p = recurso_p[:T]
    elif len(recurso_p) < T:
        normalizations.append(
            f"set tamanho from {T} to recurso_p length {len(recurso_p)}"
        )
        T = len(recurso_p)

    for name in INSTANCE_FIELDS:
        if name == "recurso_p":
            continue
        vals = parsed[name]
        if len(vals) > J:
            normalizations.append(f"trimmed {name} from {len(vals)} to {J}")
            parsed[name] = vals[:J]
        elif len(vals) != J:
            raise ValueError(f"Expected {J} values for {name}, got {len(vals)}")

    data = {
        "format": INSTANCE_FORMAT,
        "name": Path(path).stem,
        "source": Path(path).name,
        "jobs": J,
        "tamanho": T,
        **{
            name: (recurso_p if name == "recurso_p" else parsed[name])
            for name in INSTANCE_FIELDS
        },
    }
    if normalizations:
        data["normalizations"] = normalizations
    return data


def onts_instance_to_json_data(
    instance: ONTSInstance, *, name: str = ""
) -> dict[str, Any]:
    """Convert an in-memory instance to JSON-serializable data."""
    data = {
        "format": INSTANCE_FORMAT,
        "name": name,
        "jobs": instance.J,
        "tamanho": instance.T,
    }
    for field in INSTANCE_FIELDS:
        values = getattr(instance, field).detach().cpu().tolist()
        data[field] = values
    return data


def onts_json_data_to_instance(
    data: dict[str, Any], *, device: str = "cpu"
) -> ONTSInstance:
    """Build an ``ONTSInstance`` from JSON-decoded data."""
    if data.get("format", INSTANCE_FORMAT) != INSTANCE_FORMAT:
        raise ValueError(f"Unsupported ONTS instance format: {data.get('format')!r}")
    missing = [field for field in INSTANCE_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Missing ONTS instance fields: {', '.join(missing)}")

    J = int(data.get("jobs", len(data["priority"])))
    T = int(data.get("tamanho", len(data["recurso_p"])))
    if len(data["recurso_p"]) != T:
        raise ValueError(f"Expected {T} solar values, got {len(data['recurso_p'])}")
    for field in INSTANCE_FIELDS:
        if field == "recurso_p":
            continue
        if len(data[field]) != J:
            raise ValueError(f"Expected {J} values for {field}, got {len(data[field])}")

    def tensor(name: str) -> torch.Tensor:
        dtype = torch.long if name in INSTANCE_INT_FIELDS else torch.float32
        return _as_1d_tensor(data[name], dtype=dtype, device=device)

    return ONTSInstance(
        priority=tensor("priority"),
        uso_p=tensor("uso_p"),
        min_statup=tensor("min_statup"),
        max_statup=tensor("max_statup"),
        min_cpu_time=tensor("min_cpu_time"),
        max_cpu_time=tensor("max_cpu_time"),
        min_periodo_job=tensor("min_periodo_job"),
        max_periodo_job=tensor("max_periodo_job"),
        win_min=tensor("win_min"),
        win_max=tensor("win_max"),
        recurso_p=tensor("recurso_p"),
    )


def load_onts_instance(path: str | Path, *, device: str = "cpu") -> ONTSInstance:
    """Load an ONTS instance from JSON or a Julia-style ``.jl`` file."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return onts_json_data_to_instance(json.loads(path.read_text()), device=device)
    data = parse_onts_jl(path)
    return onts_json_data_to_instance(data, device=device)


class ONTSEnv:
    """Vectorized ONTS environment compatible with the NCO trainer."""

    def __init__(
        self,
        instance: ONTSInstance | None = None,
        *,
        B: int = 64,
        J: int = 9,
        T: int = 100,
        seed: int = 0,
        device: str = "cpu",
        soc_inicial: float = 0.7,
        limite_inferior: float = 0.0,
        ef: float = 0.9,
        v_bat: float = 3.6,
        q: float = 5.0,
        bat_usage: float = 5.0,
        graph_observation: bool = False,
    ) -> None:
        self.B, self.dev = B, device
        self.graph_observation = graph_observation
        self.soc_inicial = soc_inicial
        self.limite_inferior = limite_inferior
        self.ef = ef
        self.v_bat = v_bat
        self.q = q
        self.bat_usage = bat_usage
        self.max_battery_power = bat_usage * v_bat

        if instance is None:
            instance = self.random_instance(J=J, T=T, seed=seed, device=device)
        self._set_instance(instance)
        self.reset()

    def _set_instance(self, instance: ONTSInstance) -> None:
        self.instance = instance
        self.J = self.N = instance.J
        self.T = instance.T

        self.priority = instance.priority.to(self.dev).float()
        self.uso_p = instance.uso_p.to(self.dev).float()
        self.min_statup = instance.min_statup.to(self.dev).long()
        self.max_statup = instance.max_statup.to(self.dev).long()
        self.min_cpu_time = instance.min_cpu_time.to(self.dev).long()
        self.max_cpu_time = instance.max_cpu_time.to(self.dev).long()
        self.min_periodo_job = instance.min_periodo_job.to(self.dev).long()
        self.max_periodo_job = instance.max_periodo_job.to(self.dev).long()
        self.win_min = instance.win_min.to(self.dev).long()
        self.win_max = instance.win_max.to(self.dev).long()
        self.recurso_p = instance.recurso_p.to(self.dev).float()

    @classmethod
    def from_jl(cls, path: str | Path, **kwargs: Any) -> "ONTSEnv":
        device = kwargs.get("device", "cpu")
        return cls(load_onts_instance(path, device=device), **kwargs)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "ONTSEnv":
        device = kwargs.get("device", "cpu")
        return cls(load_onts_instance(path, device=device), **kwargs)

    @staticmethod
    def random_instance(
        *, J: int = 9, T: int = 100, seed: int = 0, device: str = "cpu"
    ) -> ONTSInstance:
        g = torch.Generator("cpu").manual_seed(seed)

        def _r(op):
            return op(device="cpu").to(device=device, non_blocking=True)

        min_cpu = _r(lambda **kw: torch.randint(2, 6, (J,), **kw))
        max_cpu = min_cpu + _r(lambda **kw: torch.randint(2, 8, (J,), **kw))
        win_min = torch.zeros(J, device=device)
        win_min[1:] = _r(lambda **kw: torch.randint(0, 2, (J - 1,), **kw))  # most tasks startable at t=0 or t=1
        win_max = T - _r(lambda **kw: torch.randint(0, max(1, T // 5), (J,), **kw))
        win_max = torch.maximum(win_max, win_min + min_cpu)
        phase = torch.linspace(0, torch.pi, T, device=device)
        solar = 8.0 + 6.0 * torch.sin(phase).clamp(min=0)
        solar = solar + _r(lambda **kw: torch.rand(T, **kw))
        return ONTSInstance(
            priority=_r(lambda **kw: torch.randint(1, 10, (J,), **kw)).float(),
            uso_p=(0.5 + _r(lambda **kw: torch.rand(J, **kw)) * 2.5),
            min_statup=_r(lambda **kw: torch.randint(1, 3, (J,), **kw)),
            max_statup=_r(lambda **kw: torch.randint(3, 6, (J,), **kw)),
            min_cpu_time=min_cpu,
            max_cpu_time=max_cpu,
            min_periodo_job=_r(lambda **kw: torch.randint(3, 10, (J,), **kw)),
            max_periodo_job=_r(lambda **kw: torch.randint(12, 30, (J,), **kw)),
            win_min=win_min,
            win_max=win_max,
            recurso_p=solar,
        )

    def reset(self) -> dict[str, torch.Tensor]:
        B, J, T = self.B, self.J, self.T
        self.t = 0
        self.soc = torch.full((B,), self.soc_inicial, device=self.dev)
        self.active = torch.zeros(B, J, dtype=torch.bool, device=self.dev)
        self.run_len = torch.zeros(B, J, dtype=torch.long, device=self.dev)
        self.starts = torch.zeros(B, J, dtype=torch.long, device=self.dev)
        self.last_start = torch.full((B, J), -10_000, dtype=torch.long, device=self.dev)
        self.x = torch.zeros(B, J, T, dtype=torch.bool, device=self.dev)
        self.phi = torch.zeros(B, J, T, dtype=torch.bool, device=self.dev)
        self.power = torch.zeros(B, T, device=self.dev)
        self.soc_trace = torch.zeros(B, T, device=self.dev)
        return self._obs()

    def _deadline_start(self) -> torch.Tensor:
        t = self.t
        no_start = self.starts == 0
        first_deadline = (self.max_periodo_job - 1).unsqueeze(0)
        next_deadline = self.last_start + self.max_periodo_job.unsqueeze(0)
        deadline = torch.where(no_start, first_deadline, next_deadline)
        return t >= deadline

    def _required_for_min_startups(self) -> torch.Tensor:
        remaining = max(self.T - self.t - 1, 0)
        missing = (self.min_statup.unsqueeze(0) - self.starts).clamp(min=0)
        return missing > remaining

    def _battery_next(
        self, usage: torch.Tensor, soc: torch.Tensor | None = None
    ) -> torch.Tensor:
        if soc is None:
            soc = self.soc
        battery_power = self.recurso_p[self.t] - usage
        current = battery_power / self.v_bat
        unconstrained = soc + (self.ef / self.q) * (current / 60.0)
        return unconstrained.clamp(max=1.0)

    def _energy_ok_for_usage(
        self, usage: torch.Tensor, soc: torch.Tensor | None = None
    ) -> torch.Tensor:
        power_ok = usage <= self.recurso_p[self.t] + self.max_battery_power + 1e-6
        soc_ok = self._battery_next(usage, soc=soc) >= self.limite_inferior - 1e-6
        return power_ok & soc_ok

    def _masks(self) -> tuple[torch.Tensor, ...]:
        B, J, t = self.B, self.J, self.t
        if t >= self.T:
            empty = torch.zeros(B, J, dtype=torch.bool, device=self.dev)
            return empty, empty, empty, empty, empty, empty

        forced_on = self.active & (self.run_len < self.min_cpu_time.unsqueeze(0))
        must_stop = self.active & (self.run_len >= self.max_cpu_time.unsqueeze(0))
        optional_cont = self.active & (~forced_on) & (~must_stop)

        within = (t >= self.win_min) & (t < self.win_max)
        min_gap_ok = (t - self.last_start) >= self.min_periodo_job.unsqueeze(0)
        startup_budget = self.starts < self.max_statup.unsqueeze(0)
        can_finish_min_run = t + self.min_cpu_time <= self.T
        base_start = (
            (~self.active)
            & within.unsqueeze(0)
            & min_gap_ok
            & startup_budget
            & can_finish_min_run.unsqueeze(0)
        )

        base_usage = (forced_on.float() * self.uso_p.unsqueeze(0)).sum(1)
        extra_usage = base_usage.unsqueeze(1) + self.uso_p.unsqueeze(0)
        energy_start = self._energy_ok_for_usage(
            extra_usage.reshape(-1),
            soc=self.soc.repeat_interleave(J),
        ).reshape(B, J)
        feas_start = base_start & energy_start

        forced_start = self._deadline_start() | self._required_for_min_startups()
        forced_start = forced_start & feas_start
        candidate = optional_cont | feas_start | forced_start
        return forced_on, must_stop, optional_cont, feas_start, forced_start, candidate

    def _obs(self) -> dict[str, torch.Tensor]:
        B, J, T, t = self.B, self.J, self.T, self.t
        forced_on, must_stop, optional_cont, feas_start, forced_start, candidate = (
            self._masks()
        )
        denom_priority = self.priority.max().clamp(min=1.0)
        denom_power = self.recurso_p.max().clamp(min=1.0)
        stat = (
            torch.stack(
                [
                    self.priority / denom_priority,
                    self.uso_p / denom_power,
                    self.min_statup.float() / T,
                    self.max_statup.float() / T,
                    self.min_cpu_time.float() / T,
                    self.max_cpu_time.float() / T,
                    self.min_periodo_job.float() / T,
                    self.max_periodo_job.float() / T,
                    self.win_min.float() / T,
                    self.win_max.float() / T,
                ],
                -1,
            )
            .unsqueeze(0)
            .expand(B, J, -1)
        )
        last_gap = ((t - self.last_start).float() / T).clamp(min=0.0, max=1.0)
        dyn = torch.stack(
            [
                self.active.float(),
                self.run_len.float() / T,
                self.starts.float() / self.max_statup.clamp(min=1).unsqueeze(0),
                last_gap,
                forced_on.float(),
                must_stop.float(),
                optional_cont.float(),
                feas_start.float(),
                forced_start.float(),
                (
                    self.run_len.float() / self.max_cpu_time.clamp(min=1).unsqueeze(0)
                ).clamp(max=1.0),
            ],
            -1,
        )
        solar_now = self.recurso_p[t] if t < T else torch.tensor(0.0, device=self.dev)
        solar_future = (
            self.recurso_p[min(t + 1, T - 1) : min(t + 6, T - 1) + 1].mean()
            if t + 1 < T
            else solar_now
        )
        context = torch.stack(
            [
                self.soc,
                solar_now.expand(B) / denom_power,
                torch.full((B,), t / T, device=self.dev),
                solar_future.expand(B) / denom_power,
            ],
            -1,
        )
        nodes = torch.cat([stat, dyn], dim=-1)
        base_usage = (forced_on.float() * self.uso_p.unsqueeze(0)).sum(1)
        budget = (self.recurso_p[t] + self.max_battery_power - base_usage).clamp(min=0)
        obs = {
            "nodes": nodes,
            "task_nodes": nodes,
            "mask": candidate,
            "context": context,
            "task_static": stat,
            "task_dynamic": dyn,
            "glob": context,
            "candidate": candidate,
            "feas_start": feas_start,
            "forced_on": forced_on,
            "forced_start": forced_start,
            "task_draw": self.uso_p.unsqueeze(0).expand(B, J),
            "budget": budget,
            "n_vars": J,
        }
        if self.graph_observation:
            obs.update(self._valid_inequality_graph(nodes, candidate))
        return obs

    def _valid_inequality_graph(
        self, task_nodes: torch.Tensor, candidate: torch.Tensor
    ) -> dict[str, torch.Tensor | int]:
        B, J, T, t = self.B, self.J, self.T, self.t
        dev = task_nodes.device
        f_node = task_nodes.shape[-1]
        family_count = 8
        denom_power = self.recurso_p.max().clamp(min=1.0)
        denom_priority = self.priority.max().clamp(min=1.0)

        usage = self.active.float() * self.uso_p.unsqueeze(0)
        starts_f = self.starts.float()
        run_f = self.run_len.float()
        remaining = max(T - t, 1)
        rows: list[torch.Tensor] = []

        def add_family(
            family_id: int,
            rhs: torch.Tensor,
            activity: torch.Tensor,
            window: torch.Tensor,
            applicable: torch.Tensor,
        ) -> None:
            feat = torch.zeros(B, J, f_node, device=dev)
            feat[:, :, 0] = family_id / max(family_count - 1, 1)
            feat[:, :, 1] = rhs
            feat[:, :, 2] = activity
            feat[:, :, 3] = (rhs - activity).clamp(min=-1.0, max=1.0)
            feat[:, :, 4] = window
            feat[:, :, 5] = applicable.float()
            feat[:, :, 6] = candidate.float()
            feat[:, :, 7] = self.priority.unsqueeze(0) / denom_priority
            rows.append(feat)

        add_family(
            0,
            self.max_statup.float().unsqueeze(0).expand(B, J) / T,
            starts_f / T,
            torch.full((B, J), remaining / T, device=dev),
            self.starts < self.max_statup.unsqueeze(0),
        )
        add_family(
            1,
            torch.ones(B, J, device=dev),
            usage / denom_power,
            torch.full((B, J), 1.0, device=dev),
            candidate,
        )
        add_family(
            2,
            torch.ones(B, J, device=dev),
            (run_f / self.min_cpu_time.clamp(min=1).unsqueeze(0)).clamp(max=1.0),
            self.min_cpu_time.float().unsqueeze(0).expand(B, J) / T,
            self.run_len <= self.min_cpu_time.unsqueeze(0),
        )
        total_cpu_cap = (
            self.max_cpu_time.float() * self.max_statup.float().clamp(min=1)
        ).unsqueeze(0)
        elapsed_work = (
            self.x[:, :, :t].float().sum(2) if t > 0 else torch.zeros(B, J, device=dev)
        )
        add_family(
            3,
            (total_cpu_cap / T).clamp(max=1.0),
            (elapsed_work / T).clamp(max=1.0),
            self.max_cpu_time.float().unsqueeze(0).expand(B, J) / T,
            torch.ones(B, J, dtype=torch.bool, device=dev),
        )
        add_family(
            4,
            torch.ones(B, J, device=dev),
            (run_f / self.max_cpu_time.clamp(min=1).unsqueeze(0)).clamp(max=1.0),
            self.max_cpu_time.float().unsqueeze(0).expand(B, J) / T,
            self.run_len < self.max_cpu_time.unsqueeze(0),
        )
        gap = (t - self.last_start).float().clamp(min=0)
        add_family(
            5,
            torch.ones(B, J, device=dev),
            (gap / self.min_periodo_job.clamp(min=1).unsqueeze(0)).clamp(max=1.0),
            self.min_periodo_job.float().unsqueeze(0).expand(B, J) / T,
            gap >= self.min_periodo_job.unsqueeze(0),
        )
        add_family(
            6,
            torch.ones(B, J, device=dev),
            (run_f / self.max_cpu_time.clamp(min=1).unsqueeze(0)).clamp(max=1.0),
            self.max_periodo_job.float().unsqueeze(0).expand(B, J) / T,
            self.max_cpu_time.unsqueeze(0)
            < (self.max_periodo_job - self.min_cpu_time).unsqueeze(0),
        )

        power_feat = torch.zeros(B, 1, f_node, device=dev)
        power_usage = (self.active.float() * self.uso_p.unsqueeze(0)).sum(1)
        power_rhs = self.recurso_p[t] + self.max_battery_power
        power_feat[:, 0, 0] = 7 / max(family_count - 1, 1)
        power_feat[:, 0, 1] = power_rhs / denom_power
        power_feat[:, 0, 2] = power_usage / denom_power
        power_feat[:, 0, 3] = ((power_rhs - power_usage) / denom_power).clamp(
            min=-1.0, max=1.0
        )
        power_feat[:, 0, 4] = self.soc
        power_feat[:, 0, 5] = 1.0
        power_feat[:, 0, 6] = candidate.any(1).float()
        power_feat[:, 0, 7] = 1.0

        constraint_nodes = torch.cat([torch.cat(rows, dim=1), power_feat], dim=1)
        graph_nodes = torch.cat([task_nodes, constraint_nodes], dim=1)

        src: list[int] = []
        dst: list[int] = []
        edge_feat: list[list[float]] = []
        for family_id in range(7):
            for j in range(J):
                constraint_idx = J + family_id * J + j
                coeff = 1.0
                family = family_id / 7.0
                src.extend([j, constraint_idx])
                dst.extend([constraint_idx, j])
                edge_feat.extend(
                    [
                        [coeff, abs(coeff), 1.0, family, 1.0, 0.0],
                        [coeff, abs(coeff), 1.0, family, 0.0, 1.0],
                    ]
                )

        power_idx = J + 7 * J
        for j in range(J):
            coeff = float((self.uso_p[j] / denom_power).item())
            src.extend([j, power_idx])
            dst.extend([power_idx, j])
            edge_feat.extend(
                [
                    [coeff, abs(coeff), 1.0, 1.0, 1.0, 0.0],
                    [coeff, abs(coeff), 1.0, 1.0, 0.0, 1.0],
                ]
            )

        edge_index = torch.tensor([src, dst], dtype=torch.long, device=dev)
        edge_features = torch.tensor(edge_feat, dtype=torch.float32, device=dev)
        graph_mask = torch.zeros(B, graph_nodes.shape[1], dtype=torch.bool, device=dev)
        graph_mask[:, :J] = candidate
        return {
            "nodes": graph_nodes,
            "mask": graph_mask,
            "candidate": graph_mask,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "n_vars": J,
            "constraint_family_count": family_count,
        }

    def _select_feasible(self, requested: torch.Tensor) -> torch.Tensor:
        forced_on, must_stop, optional_cont, feas_start, forced_start, candidate = (
            self._masks()
        )
        requested = requested.bool() & candidate
        active_now = forced_on | forced_start | requested
        active_now = active_now & (~must_stop)

        for b in range(self.B):
            usage = (active_now[b].float() * self.uso_p).sum()
            if self._energy_ok_for_usage(usage.unsqueeze(0))[0]:
                continue
            optional = torch.where(
                active_now[b] & (~forced_on[b]) & (~forced_start[b])
            )[0]
            if optional.numel() == 0:
                continue
            order = optional[torch.argsort(self.priority[optional])]
            for j in order:
                active_now[b, j] = False
                usage = (active_now[b].float() * self.uso_p).sum()
                if self._energy_ok_for_usage(usage.unsqueeze(0))[0]:
                    break
        return active_now

    def step(
        self, picks: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor, bool]:
        if picks.dim() == 1:
            picks = picks.unsqueeze(0)
        picks = picks.to(self.dev)
        if picks.shape[0] != self.B:
            if picks.shape[0] == 1:
                picks = picks.expand(self.B, -1)
            else:
                raise ValueError(f"Expected batch {self.B}, got {picks.shape[0]}")
        if picks.shape[1] > self.J:
            picks = picks[:, : self.J]
        active_now = self._select_feasible(picks > 0.5)
        new_starts = active_now & (~self.active)

        usage = (active_now.float() * self.uso_p.unsqueeze(0)).sum(1)
        next_soc = self._battery_next(usage)
        violation = next_soc < self.limite_inferior - 1e-6
        next_soc = next_soc.clamp(min=self.limite_inferior, max=1.0)

        self.x[:, :, self.t] = active_now
        self.phi[:, :, self.t] = new_starts
        self.power[:, self.t] = usage
        self.soc_trace[:, self.t] = next_soc

        reward = (active_now.float() * self.priority.unsqueeze(0)).sum(1)
        reward = reward - violation.float() * self.priority.sum()

        self.starts = self.starts + new_starts.long()
        self.last_start = torch.where(
            new_starts,
            torch.full_like(self.last_start, self.t),
            self.last_start,
        )
        self.run_len = torch.where(
            active_now, self.run_len + 1, torch.zeros_like(self.run_len)
        )
        self.active = active_now
        self.soc = next_soc
        self.t += 1

        done = self.t >= self.T
        if done:
            unmet = (self.min_statup.unsqueeze(0) - self.starts).clamp(min=0).float()
            over = (self.starts - self.max_statup.unsqueeze(0)).clamp(min=0).float()
            penalty = 3.0 * ((unmet + over) * self.priority.unsqueeze(0)).sum(1)
            reward = reward - penalty
            return None, reward, True
        return self._obs(), reward, False


class ONTSInstancePoolEnv(ONTSEnv):
    """ONTSEnv variant that samples a JSON instance on each episode reset."""

    def __init__(
        self,
        instance_dir: str | Path,
        *,
        pattern: str = "*.json",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.instance_paths = sorted(Path(instance_dir).glob(pattern))
        if not self.instance_paths:
            raise ValueError(
                f"No ONTS instances found in {instance_dir} with {pattern}"
            )
        self._pool_rng = torch.Generator(device="cpu").manual_seed(seed)
        super().__init__(
            load_onts_instance(
                self.instance_paths[0], device=kwargs.get("device", "cpu")
            ),
            seed=seed,
            **kwargs,
        )

    def reset(self) -> dict[str, torch.Tensor]:
        idx = int(
            torch.randint(
                len(self.instance_paths), (1,), generator=self._pool_rng
            ).item()
        )
        self._set_instance(
            load_onts_instance(self.instance_paths[idx], device=self.dev)
        )
        return super().reset()
