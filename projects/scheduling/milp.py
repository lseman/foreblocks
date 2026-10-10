"""Reusable sparse ONTS MILP, exact solving, and neighborhood repair."""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .environments.onts_env import ONTSInstance
from .validation import ScheduleValidation, validate_schedule


@dataclass(frozen=True)
class ONTSConfig:
    soc_inicial: float = 0.7
    limite_inferior: float = 0.0
    ef: float = 0.9
    v_bat: float = 3.6
    q: float = 5.0
    bat_usage: float = 5.0
    allow_truncated_final_run: bool = True

    def validate(
        self, instance: ONTSInstance, schedule: torch.Tensor
    ) -> ScheduleValidation:
        return validate_schedule(instance, schedule, **asdict(self))


@dataclass
class SolveResult:
    status: str
    message: str
    schedule: torch.Tensor | None
    objective: float | None
    upper_bound: float | None
    relative_gap: float | None
    runtime: float
    validation: ScheduleValidation | None
    restricted: bool = False
    time_to_first_feasible: float | None = None
    first_feasible_source: str | None = None
    incumbent_history: list[dict] = field(default_factory=list)


class ONTSMilp:
    """Base ONTS constraints with curtailment and exact startup indicators.

    x and phi are binary; soc is continuous. Discarding surplus power is
    represented by soc[t]-soc[t-1] <= k*(solar[t]-usage[t]). Together with
    SoC bounds this is equivalent to the current inequality in original.py.
    Optional reference cuts are excluded pending mathematical validation.
    """

    def __init__(self, instance: ONTSInstance, config: ONTSConfig | None = None):
        self.instance = instance
        self.config = config or ONTSConfig()
        self.J, self.T = instance.J, instance.T
        # Reuse public input checks; an empty schedule need not be feasible.
        self.config.validate(instance, torch.zeros(self.J, self.T))
        J, T = self.J, self.T
        self.n_x = J * T
        self.n_vars = 2 * self.n_x + T
        self.c = np.zeros(self.n_vars)
        self.c[: self.n_x] = -np.repeat(instance.priority.cpu().double().numpy(), T)
        self.lower = np.zeros(self.n_vars)
        self.upper = np.ones(self.n_vars)
        self.lower[2 * self.n_x :] = self.config.limite_inferior
        self.integrality = np.zeros(self.n_vars, dtype=np.int32)
        self.integrality[: 2 * self.n_x] = 1
        self.row_names: list[str] = []
        rr, cc, vv, lower, upper = [], [], [], [], []

        def add(name, coeffs, lo=-np.inf, hi=np.inf):
            row = len(lower)
            for col, value in coeffs.items():
                if value:
                    rr.append(row)
                    cc.append(col)
                    vv.append(float(value))
            lower.append(float(lo))
            upper.append(float(hi))
            self.row_names.append(name)

        def x(j, t):
            return j * T + t

        def phi(j, t):
            return self.n_x + x(j, t)

        for j in range(J):

            def get(name):
                return int(getattr(instance, name)[j])

            for t in range(T):
                if not get("win_min") <= t < get("win_max"):
                    self.upper[x(j, t)] = 0
                coeffs = {phi(j, t): 1, x(j, t): -1}
                if t:
                    coeffs[x(j, t - 1)] = 1
                add(f"startup_lower:{j}:{t}", coeffs, lo=0)
                add(f"startup_on:{j}:{t}", {phi(j, t): 1, x(j, t): -1}, hi=0)
                if t:
                    add(
                        f"startup_previous:{j}:{t}",
                        {phi(j, t): 1, x(j, t - 1): 1},
                        hi=1,
                    )
            add(
                f"startup_count:{j}",
                {phi(j, t): 1 for t in range(T)},
                lo=get("min_statup"),
                hi=get("max_statup"),
            )
            for field, rhs, bound in (
                ("min_periodo_job", 1, "hi"),
                ("max_periodo_job", 1, "lo"),
            ):
                period = get(field)
                if field == "min_periodo_job" and period > T:
                    add(
                        f"{field}:{j}:horizon", {phi(j, tt): 1 for tt in range(T)}, hi=1
                    )
                for t in range(T - period + 1):
                    add(
                        f"{field}:{j}:{t}",
                        {phi(j, tt): 1 for tt in range(t, t + period)},
                        **{bound: rhs},
                    )
            minimum, maximum = get("min_cpu_time"), get("max_cpu_time")
            for t in range(T):
                length = min(minimum, T - t)
                if length < minimum and not self.config.allow_truncated_final_run:
                    self.upper[phi(j, t)] = 0
                else:
                    coeffs = {x(j, tt): 1 for tt in range(t, t + length)}
                    coeffs[phi(j, t)] = -length
                    add(f"minimum_run:{j}:{t}", coeffs, lo=0)
            for t in range(T - maximum):
                add(
                    f"maximum_run:{j}:{t}",
                    {x(j, tt): 1 for tt in range(t, t + maximum + 1)},
                    hi=maximum,
                )

        k = self.config.ef / (self.config.q * self.config.v_bat * 60)
        for t in range(T):
            usage = {x(j, t): float(instance.uso_p[j]) for j in range(J)}
            solar = float(instance.recurso_p[t])
            add(
                f"power:{t}",
                usage,
                hi=solar + self.config.bat_usage * self.config.v_bat,
            )
            energy = {col: k * value for col, value in usage.items()}
            energy[2 * self.n_x + t] = 1
            rhs = k * solar
            if t:
                energy[2 * self.n_x + t - 1] = -1
            else:
                rhs += self.config.soc_inicial
            add(f"energy:{t}", energy, hi=rhs)
        self.A = coo_matrix((vv, (rr, cc)), shape=(len(lower), self.n_vars)).tocsc()
        self.row_lower, self.row_upper = np.array(lower), np.array(upper)
        self.constraints = LinearConstraint(self.A, self.row_lower, self.row_upper)

    def assignment(self, schedule: torch.Tensor) -> np.ndarray:
        report = self.config.validate(self.instance, schedule)
        if not report.feasible:
            raise ValueError(f"Incumbent is infeasible: {report.violations}")
        x = schedule.detach().cpu().double().numpy()
        prev = np.concatenate([np.zeros((self.J, 1)), x[:, :-1]], axis=1)
        return np.concatenate([x.ravel(), (x > prev).astype(float).ravel(), report.soc])

    def solve(
        self,
        *,
        time_limit: float = 10.0,
        incumbent: torch.Tensor | None = None,
        free_mask: torch.Tensor | None = None,
        proposal: torch.Tensor | None = None,
        solver_threads: int | None = None,
        observe_incumbents: bool = False,
    ) -> SolveResult:
        """Maximize objective or repair a proposal by minimum Hamming distance.

        A restricted repair fixes only x outside free_mask. Startup and SoC
        variables remain free. Its bound applies ONLY to that neighborhood.
        A valid incumbent is retained even when the solver times out.
        """
        started = perf_counter()
        if solver_threads is None and os.environ.get("ONTS_SOLVER_THREADS"):
            solver_threads = int(os.environ["ONTS_SOLVER_THREADS"])
        if solver_threads is not None and (
            not isinstance(solver_threads, int) or solver_threads < 1
        ):
            raise ValueError("solver_threads must be a positive integer")
        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("time_limit must be positive and finite")
        if free_mask is not None and incumbent is None:
            raise ValueError("free_mask requires a feasible incumbent")
        if proposal is not None and incumbent is not None:
            raise ValueError(
                "Proposal repair and incumbent improvement are separate operations"
            )
        lo, hi = self.lower.copy(), self.upper.copy()
        restricted = free_mask is not None
        baseline = None
        if incumbent is not None:
            self.assignment(incumbent)
            baseline = self.config.validate(self.instance, incumbent)
            if free_mask is not None:
                mask = torch.as_tensor(free_mask).detach().cpu()
                if mask.shape != (self.J, self.T) or mask.dtype != torch.bool:
                    raise ValueError("free_mask must be boolean with shape [J,T]")
                fixed = ~mask.numpy().ravel()
                values = incumbent.detach().cpu().double().numpy().ravel()
                lo[: self.n_x][fixed] = values[fixed]
                hi[: self.n_x][fixed] = values[fixed]
        objective = self.c
        if proposal is not None:
            # Validation checks structure and binary values; proposal may be infeasible.
            self.config.validate(self.instance, proposal)
            objective = np.zeros(self.n_vars)
            objective[: self.n_x] = (
                1 - 2 * proposal.detach().cpu().double().numpy().ravel()
            )
        incumbent_history = []

        def observed(raw, source="highs_incumbent_callback"):
            if raw is None or not np.isfinite(raw).all():
                return
            cells = np.asarray(raw[:self.n_x])
            if np.max(np.abs(cells - cells.round())) > 1e-5:
                return
            candidate = torch.from_numpy(cells.round().reshape(self.J, self.T)).bool()
            checked = self.config.validate(self.instance, candidate)
            if not checked.feasible:
                return
            if incumbent_history and checked.objective <= incumbent_history[-1]["objective"]:
                return
            incumbent_history.append(dict(time=perf_counter() - started,
                                          objective=checked.objective, source=source))

        if baseline is not None:
            incumbent_history.append(dict(time=0.0, objective=baseline.objective,
                                          source="provided_schedule"))
        solve_milp = milp
        if observe_incumbents:
            from .observed_highs import milp_with_incumbents
            def solve_milp(**kwargs):
                return milp_with_incumbents(**kwargs, on_incumbent=observed)

        # SciPy deliberately forwards additional options to HiGHS. Fix the
        # solver thread count for reproducible serial benchmark execution.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected:.*",
                category=RuntimeWarning,
            )
            result = solve_milp(
                c=objective,
                integrality=self.integrality,
                bounds=Bounds(lo, hi),
                constraints=self.constraints,
                options={
                    "time_limit": time_limit,
                    "mip_rel_gap": 0.0,
                    **(
                        {"threads": solver_threads}
                        if solver_threads is not None
                        else {}
                    ),
                    "random_seed": 0,
                },
            )
        if observe_incumbents and result.x is not None:
            observed(result.x, source="solver_return_validation")
        status = {
            0: "optimal",
            1: "limit",
            2: "infeasible",
            3: "unbounded",
            4: "error",
        }.get(result.status, "error")
        schedule, report = None, None
        if result.x is not None and np.isfinite(result.x).all():
            raw = result.x[: self.n_x]
            if np.max(np.abs(raw - raw.round())) <= 1e-5:
                candidate = torch.from_numpy(raw.round().reshape(self.J, self.T)).bool()
                candidate_report = self.config.validate(self.instance, candidate)
                if candidate_report.feasible:
                    schedule, report = candidate, candidate_report
        if baseline is not None and (
            report is None or baseline.objective >= report.objective
        ):
            schedule, report = incumbent.detach().cpu().bool().clone(), baseline
        bound = getattr(result, "mip_dual_bound", None)
        upper_bound = (
            -float(bound)
            if proposal is None and bound is not None and math.isfinite(bound)
            else None
        )
        if upper_bound is not None and report is not None:
            upper_bound = max(upper_bound, report.objective)
        gap = None
        if upper_bound is not None and report is not None:
            gap = max(0.0, upper_bound - report.objective) / max(
                abs(report.objective), 1e-9
            )
        if result.status == 0 and report is None:
            status = "validation_failed"
        return SolveResult(
            status,
            str(result.message),
            schedule,
            report.objective if report else None,
            upper_bound,
            gap,
            perf_counter() - started,
            report,
            restricted,
            time_to_first_feasible=incumbent_history[0]["time"] if incumbent_history else None,
            first_feasible_source=incumbent_history[0]["source"] if incumbent_history else None,
            incumbent_history=incumbent_history,
        )
