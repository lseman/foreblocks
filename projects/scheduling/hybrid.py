"""Anytime ONTS large-neighborhood search with exact, validated repair."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch

from .environments.onts_env import ONTSInstance
from .milp import ONTSConfig, ONTSMilp
from .neighborhood import GraphBuilder, NeighborhoodPolicy


@dataclass
class HybridResult:
    status: str
    schedule: torch.Tensor | None
    objective: float | None
    upper_bound: float | None
    relative_gap: float | None
    runtime: float
    time_to_first_feasible: float | None
    iterations: int
    history: list[dict] = field(default_factory=list)
    initialization_complete_at: float | None = None
    first_feasible_source: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "feasible": self.schedule is not None,
            "schedule": self.schedule.int().tolist()
            if self.schedule is not None
            else None,
            "objective": self.objective,
            "upper_bound": self.upper_bound,
            "relative_gap": self.relative_gap,
            "runtime": self.runtime,
            "time_to_first_feasible": self.time_to_first_feasible,
            "initialization_complete_at": self.initialization_complete_at,
            "first_feasible_source": self.first_feasible_source,
            "iterations": self.iterations,
            "history": self.history,
        }


def select_neighborhood(
    scores: np.ndarray, *, J: int, T: int, fraction: float, block_size: int = 4
) -> torch.Tensor:
    """Release ranked contiguous job/time blocks up to an exact cell budget."""
    if not 0 < fraction <= 1 or block_size < 1:
        raise ValueError("Expected fraction in (0,1] and positive block_size")
    if scores.shape != (J * T,) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite score per execution variable")
    count = max(1, math.ceil(J * T * fraction))
    mask = np.zeros((J, T), dtype=bool)
    for index in np.argsort(-scores, kind="stable"):
        j, t = divmod(int(index), T)
        start = max(0, min(t - block_size // 2, T - block_size))
        for tt in range(start, min(T, start + block_size)):
            if not mask[j, tt]:
                mask[j, tt] = True
                count -= 1
                if count == 0:
                    return torch.from_numpy(mask)
    return torch.from_numpy(mask)


def solve_hybrid(
    instance: ONTSInstance,
    *,
    time_limit: float = 30,
    repair_time: float = 2,
    init_fraction: float = 0.3,
    neighborhood_fraction: float = 0.25,
    block_size: int = 4,
    selector: str = "heuristic",
    policy: NeighborhoodPolicy | None = None,
    seed: int = 0,
    config: ONTSConfig | None = None,
    initial_schedule: torch.Tensor | None = None,
    proposal_fn: Callable[[ONTSInstance], torch.Tensor] | None = None,
    exploration: float = 0.2,
    max_iterations: int = 1000,
    initial_exact: bool = True,
) -> HybridResult:
    """Initialize, then repeatedly release x cells and solve the restricted MILP.

    Time includes construction, neural inference, solver calls and validation.
    Solver limits are cooperative: native solver/setup overhead may overrun the
    wall budget slightly. Only an unrestricted objective solve supplies a global
    bound. Proposal repair optimizes Hamming distance and supplies no such bound.
    """
    started = perf_counter()
    if not all(math.isfinite(v) and v > 0 for v in (time_limit, repair_time)):
        raise ValueError("Time budgets must be positive and finite")
    if not 0 < init_fraction <= 1 or not 0 < neighborhood_fraction <= 1:
        raise ValueError("Fractions must be in (0,1]")
    if not 0 <= exploration <= 1 or block_size < 1 or max_iterations < 0:
        raise ValueError("Invalid exploration, block size or iteration limit")
    if selector not in ("exact", "random", "heuristic", "neural"):
        raise ValueError("Unknown selector")
    if selector == "neural" and policy is None:
        raise ValueError("Neural selection requires a trained policy")
    if initial_schedule is not None and proposal_fn is not None:
        raise ValueError("Choose initial_schedule or proposal_fn")
    deadline = started + time_limit
    p = ONTSMilp(instance, config)
    rng = np.random.default_rng(seed)
    incumbent, objective, bound, first = None, None, None, None
    history = []
    iterations = 0
    status = "unknown"

    # Leave validation/serialization headroom so a solver completion is
    # normally observable before the advertised wall-clock deadline.
    reserve = min(0.25, 0.05 * time_limit)

    def remaining():
        return deadline - perf_counter() - reserve

    def record(event, result):
        history.append(
            {
                "time": perf_counter() - started,
                "event": event,
                "status": result.status,
                "objective": result.objective,
                "restricted": result.restricted,
                "solve_runtime": result.runtime,
            }
        )

    if initial_schedule is not None:
        report = p.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError(
                f"Initial schedule violates constraints: {report.violations}"
            )
        incumbent, objective = (
            initial_schedule.detach().cpu().bool().clone(),
            report.objective,
        )
        first = perf_counter() - started
        history.append(
            {"time": first, "event": "initial_schedule", "objective": objective}
        )
    if proposal_fn is not None and remaining() > 0:
        proposal = proposal_fn(instance)
        report = p.config.validate(instance, proposal)
        if report.feasible:
            incumbent, objective = (
                proposal.detach().cpu().bool().clone(),
                report.objective,
            )
            first = perf_counter() - started
            history.append(
                {"time": first, "event": "neural_proposal", "objective": objective}
            )
        elif remaining() > 0:
            result = p.solve(
                proposal=proposal,
                time_limit=min(remaining(), time_limit * init_fraction),
            )
            record("proposal_repair", result)
            incumbent, objective = result.schedule, result.objective
            if incumbent is not None:
                first = perf_counter() - started
    # Always try a full objective solve: gives a legitimate global bound and
    # a fallback when the policy or nearest-proposal repair finds no incumbent.
    if remaining() > 0 and (initial_exact or selector == "exact" or incumbent is None):
        budget = (
            remaining()
            if selector == "exact"
            else min(remaining(), time_limit * init_fraction)
        )
        result = p.solve(time_limit=budget, incumbent=incumbent)
        record("initial_exact", result)
        incumbent, objective, bound = (
            result.schedule,
            result.objective,
            result.upper_bound,
        )
        status = result.status
        if incumbent is not None and first is None:
            first = perf_counter() - started
    builder = None
    if selector == "neural" and incumbent is not None and remaining() > 0:
        builder = GraphBuilder(p)
    while (
        selector != "exact"
        and remaining() > 0
        and iterations < max_iterations
        and status != "infeasible"
    ):
        if incumbent is not None and bound is not None and bound - objective <= 1e-6:
            status = "optimal"
            break
        if incumbent is None:
            # No incumbent: unrestricted retries; never fix an invalid proposal.
            # A short initialization may exhaust its budget before producing
            # an incumbent. Repeated tiny cold starts can never make progress.
            result = p.solve(time_limit=remaining())
            record("initial_retry", result)
            if result.upper_bound is not None:
                bound = (
                    min(bound, result.upper_bound)
                    if bound is not None
                    else result.upper_bound
                )
            if result.status == "infeasible":
                status = "infeasible"
                break
            incumbent, objective = result.schedule, result.objective
            if incumbent is not None:
                first = perf_counter() - started
            iterations += 1
            continue
        if selector == "neural" and rng.random() >= exploration:
            if builder is None:
                builder = GraphBuilder(p)
            device = next(policy.parameters()).device
            was_training = policy.training
            policy.eval()
            try:
                with torch.no_grad():
                    scores = policy(builder.build(incumbent).to(device)).cpu().numpy()
            finally:
                policy.train(was_training)
            # Perturb scores to explore alternatives with the same learned ranking.
            scores = scores + 0.1 * rng.gumbel(size=p.n_x)
        elif selector == "heuristic" and rng.random() >= exploration:
            priority = instance.priority.cpu().numpy()
            draw = instance.uso_p.cpu().numpy()
            utility = np.repeat(priority / np.maximum(draw, 0.01), p.T)
            unused = 1 - incumbent.numpy().ravel()
            scores = utility * unused + rng.gumbel(size=p.n_x)
        else:
            scores = rng.random(p.n_x)
        free = select_neighborhood(
            scores, J=p.J, T=p.T, fraction=neighborhood_fraction, block_size=block_size
        )
        if remaining() <= 0:
            break
        result = p.solve(
            time_limit=min(remaining(), repair_time),
            incumbent=incumbent,
            free_mask=free,
        )
        record("repair", result)
        incumbent, objective = result.schedule, result.objective
        # Restricted bounds cannot bound the original problem.
        if not result.restricted and result.upper_bound is not None:
            bound = (
                min(bound, result.upper_bound)
                if bound is not None
                else result.upper_bound
            )
        iterations += 1
    if incumbent is not None:
        status = (
            "optimal" if bound is not None and bound - objective <= 1e-6 else "feasible"
        )
        if not p.config.validate(instance, incumbent).feasible:
            raise RuntimeError("Search returned an invalid incumbent")
    elif status not in ("infeasible", "error", "validation_failed"):
        status = "unknown"
    gap = (
        max(0, bound - objective) / max(abs(objective), 1e-9)
        if bound is not None and objective is not None
        else None
    )
    return HybridResult(
        status,
        incumbent,
        objective,
        bound,
        gap,
        perf_counter() - started,
        first,
        iterations,
        history,
    )


def rollout_proposal(
    model,
    *,
    graph_observation: bool = False,
    device: str = "cpu",
    config: ONTSConfig | None = None,
):
    """Adapt an existing NCO/PPO model into solve_hybrid(proposal_fn=...)."""
    from dataclasses import asdict

    from .environments.onts_env import ONTSEnv

    def propose(instance):
        kwargs = asdict(config or ONTSConfig())
        kwargs.pop("allow_truncated_final_run")
        env = ONTSEnv(
            instance, B=1, device=device, graph_observation=graph_observation, **kwargs
        )
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                obs = env.reset()
                while obs is not None:
                    action = model.act(obs, temperature=0.0)
                    obs, _, _ = env.step(action["new_starts"])
        finally:
            model.train(was_training)
        return env.x[0].cpu()

    return propose
