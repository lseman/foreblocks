"""Cold-start RL construction using constraint-valid whole-task profiles.

No incumbent or witness is accepted. Local resource safety does not imply
completion feasibility; a dead end is a failed construction, never a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .environments.onts_env import ONTSInstance
from .job_profile import battery_trace, best_profile, job_rules
from .milp import ONTSConfig, ONTSMilp


@dataclass
class Placement:
    task: int
    row: np.ndarray
    features: list[float]


class ConstructPolicy(nn.Module):
    """Shared candidate scorer; supports variable task/horizon counts."""

    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(16, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scorer(features).squeeze(-1)

    def score_choices(self, env, choices, features):
        return self(features)


class ConstructionEnv:
    def __init__(self, instance: ONTSInstance, config: ONTSConfig | None = None):
        self.problem = ONTSMilp(instance, config)
        self.schedule = np.zeros((instance.J, instance.T), dtype=bool)
        self.remaining = set(range(instance.J))

    def resource_safe(self, schedule: np.ndarray) -> bool:
        p = self.problem
        usage = p.instance.uso_p.cpu().double().numpy() @ schedule
        solar = p.instance.recurso_p.cpu().double().numpy()
        return bool(
            np.all(usage <= solar + p.config.bat_usage * p.config.v_bat + 1e-6)
            and np.all(battery_trace(p, usage) >= p.config.limite_inferior - 1e-6)
        )

    def candidates(self, deadline: float = float("inf")) -> list[Placement]:
        p, i = self.problem, self.problem.instance
        solar = i.recurso_p.cpu().double().numpy()
        powers = i.uso_p.cpu().double().numpy()
        usage = powers @ self.schedule
        residual = solar + p.config.bat_usage * p.config.v_bat - usage
        charge = battery_trace(p, usage)
        candidates = []
        for j in sorted(self.remaining):
            if perf_counter() >= deadline:
                break
            rules = job_rules(p, j)
            # Minimal execution, solar-favored minimal execution, and quality.
            # Negative rewards reserve resources; positive rewards seek quality.
            objectives = [
                -np.ones(p.T),
                -1 + 0.4 * (solar - usage) / max(1., float(np.max(np.abs(solar - usage)))),
                np.ones(p.T),
            ]
            seen = set()
            for mode, reward in enumerate(objectives):
                if perf_counter() >= deadline:
                    break
                reward = reward.copy()
                reward[residual < powers[j] - 1e-6] = -np.inf
                row, _ = best_profile(reward, rules, truncated_final_run=p.config.allow_truncated_final_run)
                if row is None or row.tobytes() in seen:
                    continue
                seen.add(row.tobytes())
                trial = self.schedule.copy()
                trial[j] = row
                if not self.resource_safe(trial):
                    continue
                trial_charge = battery_trace(p, powers @ trial)
                future = self.remaining - {j}
                mandatory = sum(int(i.min_statup[k]) * int(i.min_cpu_time[k]) * powers[k] for k in future)
                features = [
                    float(i.priority[j]) / max(1., float(i.priority.max())),
                    powers[j] / max(1., float(powers.max())),
                    rules.min_starts / p.T, rules.max_starts / p.T,
                    rules.min_run / p.T, rules.max_run / p.T,
                    rules.min_gap / p.T, min(rules.max_gap / p.T, 2.),
                    (rules.win_max - rules.win_min) / p.T,
                    float(row.mean()), mode / 2., len(self.remaining) / p.J,
                    float(trial_charge.min()), float(charge.mean()),
                    mandatory / max(1., float(solar.sum()) + p.config.soc_inicial * p.config.q * p.config.v_bat * 60 / p.config.ef),
                    float(np.mean(residual)) / max(1., float(np.max(solar)) + p.config.bat_usage * p.config.v_bat),
                ]
                candidates.append(Placement(j, row, features))
        return candidates

    def place(self, candidate: Placement) -> None:
        if candidate.task not in self.remaining:
            raise ValueError("Task already placed")
        self.schedule[candidate.task] = candidate.row
        self.remaining.remove(candidate.task)


def construct(instance: ONTSInstance, policy: ConstructPolicy | None, *,
              mode: str = "sample", seed: int = 0, budget: float = 2.,
              config: ONTSConfig | None = None, training: bool = False,
              record_decisions: bool = False, record_graph: bool = False):
    """Return complete validation plus differentiable sampled log probabilities.

    A cooperative deadline includes setup, candidate generation and validation.
    Late feasible outputs are recorded, but are not credited as on-time successes.
    """
    if mode not in {"sample", "greedy", "uniform", "priority"}:
        raise ValueError("Unknown construction mode")
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be positive and finite")
    if mode in {"sample", "greedy"} and policy is None:
        raise ValueError("A policy is required")
    started = perf_counter()
    deadline = started + budget
    env = ConstructionEnv(instance, config)
    generator = torch.Generator().manual_seed(seed)
    logs, entropies, actions, decisions = [], [], [], []
    while env.remaining and perf_counter() < deadline:
        choices = env.candidates(deadline)
        if not choices or perf_counter() >= deadline:
            break
        features = torch.tensor([c.features for c in choices], dtype=torch.float32)
        with torch.set_grad_enabled(training):
            logits = policy.score_choices(env, choices, features) if policy is not None else torch.zeros(len(choices))
            if mode == "priority":
                logits = features[:, 0] * features[:, 9]
            probabilities = logits.softmax(0)
            index = int(logits.argmax()) if mode in {"greedy", "priority"} else int(torch.multinomial(probabilities, 1, generator=generator))
            if perf_counter() >= deadline:
                break
            if training:
                logs.append(probabilities[index].log())
                entropies.append(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        if record_decisions:
            decision = {"features": features.tolist(), "index": index}
            if record_graph:
                from .rl_construct_graph import graph_observation
                decision["graph"] = {k: v.tolist() for k, v in graph_observation(env, choices).items()}
            decisions.append(decision)
        env.place(choices[index])
        actions.append({"task": choices[index].task, "profile": choices[index].row.astype(int).tolist()})
    schedule = torch.from_numpy(env.schedule.copy())
    report = env.problem.config.validate(instance, schedule)
    runtime = perf_counter() - started
    success = report.feasible and runtime <= budget
    # Feasibility dominates quality. Partial completion only separates failures.
    quality = report.objective / max(1., float(instance.priority.sum()) * instance.T)
    reward = 1. + quality if success else 0.1 * (instance.J - len(env.remaining)) / instance.J
    return dict(schedule=schedule.int().tolist(), feasible=report.feasible,
                success=success, objective=report.objective if report.feasible else None,
                runtime=runtime, time_to_first_feasible=runtime if report.feasible else None,
                overrun=max(0., runtime - budget), completed=instance.J-len(env.remaining),
                reward=reward, actions=actions, decisions=decisions), logs, entropies
