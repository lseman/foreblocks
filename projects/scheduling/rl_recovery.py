"""Necessary completion checks and anytime backtracking over ONTS profiles."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .job_profile import best_profile, job_rules
from .rl_construct import ConstructionEnv, Placement


class RecoveryPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, features):
        return self.scorer(features).squeeze(-1)


@dataclass
class SearchNode:
    schedule: np.ndarray
    remaining: set[int]
    actions: list[dict]
    decisions: list[dict]
    choices: list[Placement] | None = None
    initial_choices: int = 0
    minimum_energy: dict[int, float] = field(default_factory=dict)


def minimum_profile(env, task, schedule):
    """Exact minimum execution under a relaxed single-task power constraint."""
    p, i = env.problem, env.problem.instance
    power = i.uso_p.cpu().double().numpy()
    residual = i.recurso_p.cpu().double().numpy() + p.config.bat_usage * p.config.v_bat - power @ schedule
    reward = -np.ones(p.T)
    reward[residual < power[task] - 1e-6] = -np.inf
    row, _ = best_profile(reward, job_rules(p, task), truncated_final_run=p.config.allow_truncated_final_run)
    return row


def energy_capacity(env):
    p, c = env.problem, env.problem.config
    return float(p.instance.recurso_p.cpu().double().numpy().sum()) + (c.soc_inicial-c.limite_inferior) * 60 * c.q * c.v_bat / c.ef


def completion_probe(env, candidate, deadline=float("inf")):
    """Reject only necessary-condition violations, never an unsafe DP witness.

    A relaxed DP has all other unplaced tasks absent. If its feasible set is
    empty, adding other nonnegative loads cannot restore that task's feasibility.
    Summed minimum execution energy is a lower bound; energy curtailment is
    ignored, giving an upper bound on available energy. Timeout stays unknown.
    """
    trial = env.schedule.copy()
    trial[candidate.task] = candidate.row
    power = env.problem.instance.uso_p.cpu().double().numpy()
    lower = float((power @ trial).sum())
    minima = {}
    for task in sorted(env.remaining - {candidate.task}):
        if perf_counter() >= deadline:
            return dict(status="unknown", reason="deadline", minima=minima)
        row = minimum_profile(env, task, trial)
        if perf_counter() >= deadline:
            return dict(status="unknown", reason="deadline", minima=minima)
        if row is None:
            return dict(status="reject", reason="remaining_task_has_no_profile", task=task, minima=minima)
        minima[task] = int(row.sum())
        lower += power[task] * int(row.sum())
    capacity = energy_capacity(env)
    if lower > capacity + 1e-5:
        return dict(status="reject", reason="minimum_energy_exceeds_capacity", lower=lower, capacity=capacity, minima=minima)
    return dict(status="not_ruled_out", lower=lower, capacity=capacity, minima=minima)


def recovery_features(env, node, choices, remaining_time, budget):
    """Cheap completion context; expensive conditional probes run after sampling."""
    i, p = env.problem.instance, env.problem
    power = i.uso_p.cpu().double().numpy()
    solar = i.recurso_p.cpu().double().numpy()
    usage = power @ env.schedule
    slots = np.arange(p.T)
    rows = []
    for choice in choices:
        future = env.remaining - {choice.task}
        energy = float(usage.sum()) + float(power[choice.task] * choice.row.sum()) + sum(node.minimum_energy.get(j, 0.) for j in future)
        residual = solar + p.config.bat_usage * p.config.v_bat - usage - power[choice.task] * choice.row
        tightness = 1.
        for j in future:
            eligible = (slots >= int(i.win_min[j])) & (slots < int(i.win_max[j]))
            usable = eligible & (residual >= power[j] - 1e-6)
            tightness = min(tightness, float(usable.sum()) / max(1, int(eligible.sum())))
        rows.append(choice.features + [
            (energy_capacity(env)-energy) / max(1., energy_capacity(env)), tightness,
            max(0., remaining_time) / budget, len(choices) / max(1, node.initial_choices),
        ])
    return torch.tensor(rows, dtype=torch.float32)


def recover_construct(instance, policy=None, *, mode="sample", seed=0, budget=2.,
                      config=None, lookahead=True, training=False,
                      record_decisions=False, max_decisions=4096, memoize=True, record_observation=False, decision_callback=None):
    """Weighted depth-first search with cached alternatives and best preservation.

    Restarting after exhausting the deterministic menu would revisit the same
    tree. Backtracking explores untried branches instead. The search is complete
    only for its bounded deterministic profile tree, not the ONTS feasible set.
    """
    if mode not in {"sample", "uniform", "priority", "greedy"}:
        raise ValueError("Unknown search mode")
    if mode in {"sample", "greedy"} and policy is None:
        raise ValueError("A policy is required")
    if not np.isfinite(budget) or budget <= 0 or max_decisions < 1:
        raise ValueError("Invalid search budget or decision cap")
    started = perf_counter()
    deadline = started + budget
    cutoff = deadline - min(.05, .05 * budget)
    env = ConstructionEnv(instance, config)
    root = SearchNode(env.schedule.copy(), env.remaining.copy(), [], [])
    stack = [root]
    generator = torch.Generator().manual_seed(seed)
    best, best_objective, best_actions, best_decisions = None, -float("inf"), [], []
    deepest, deepest_actions = env.schedule.copy(), []
    first_feasible = None
    events, logs, entropies = [], [], []
    backtracks = rejected = completions = decisions = memoized = 0
    seen = {(root.schedule.tobytes(), tuple(sorted(root.remaining)))}
    initial = env.problem.config.validate(instance, torch.from_numpy(env.schedule))
    if initial.feasible and perf_counter() <= cutoff:
        best = env.schedule.copy(); best_objective = initial.objective
        first_feasible = perf_counter()-started
    while stack and decisions < max_decisions and perf_counter() < cutoff:
        node = stack[-1]
        env.schedule = node.schedule.copy(); env.remaining = node.remaining.copy()
        if node.choices is None:
            node.choices = env.candidates(cutoff)
            node.initial_choices = len(node.choices)
            for j in sorted(env.remaining):
                if perf_counter() >= cutoff:
                    break
                row = minimum_profile(env, j, env.schedule)
                node.minimum_energy[j] = float(instance.uso_p[j]) * int(row.sum()) if row is not None else 0.
        if perf_counter() >= cutoff:
            break
        if not node.choices:
            stack.pop(); backtracks += 1
            events.append(dict(kind="backtrack", depth=len(node.actions), elapsed=perf_counter()-started))
            continue
        features = recovery_features(env, node, node.choices, cutoff-perf_counter(), budget)
        with torch.set_grad_enabled(training):
            logits = (policy.score_choices(env, node.choices, features) if hasattr(policy, "score_choices")
                      else policy(features)) if policy is not None else torch.zeros(len(node.choices))
            if mode == "priority":
                logits = features[:, 0] * features[:, 9]
            distribution = torch.distributions.Categorical(logits=logits)
            index = int(logits.argmax()) if mode in {"priority", "greedy"} else int(torch.multinomial(distribution.probs, 1, generator=generator))
            if perf_counter() >= cutoff:
                break
            if training:
                logs.append(distribution.log_prob(torch.tensor(index)))
                entropies.append(distribution.entropy())
        if decision_callback is not None:
            decision_callback(env, node.choices, features, index, logits.detach())
        snapshot = None
        if record_decisions and record_observation:
            from .rl_architectures import observation
            snapshot = {k: v.tolist() for k, v in observation(env, node.choices, features).items()}
        candidate = node.choices.pop(index)
        decisions += 1
        trial = env.schedule.copy(); trial[candidate.task] = candidate.row
        child_key = (trial.tobytes(), tuple(sorted(env.remaining-{candidate.task})))
        duplicate = memoize and child_key in seen
        probe = dict(status="duplicate_state") if duplicate else completion_probe(env, candidate, cutoff) if lookahead else dict(status="not_checked")
        events.append(dict(kind="proposal", depth=len(node.actions), task=candidate.task,
                           profile=candidate.row.astype(int).tolist(), probe=probe, elapsed=perf_counter()-started,
                           prefix=[a["task"] for a in node.actions]))
        if probe["status"] == "unknown" or perf_counter() >= cutoff:
            break
        if duplicate:
            memoized += 1
            continue
        if probe["status"] == "reject":
            rejected += 1
            continue
        decision = dict(features=features.tolist(), index=index)
        if snapshot is not None:
            decision["observation"] = snapshot
        env.place(candidate)
        seen.add(child_key)
        events[-1]["placed"] = True
        actions = node.actions + [dict(task=candidate.task, profile=candidate.row.astype(int).tolist())]
        path_decisions = node.decisions + [decision] if record_decisions else []
        # Keep only partial schedules as the failure fallback. A completion
        # validated after the cutoff must never bypass incumbent acceptance.
        if env.remaining and len(actions) > len(deepest_actions):
            deepest, deepest_actions = env.schedule.copy(), actions
        if env.remaining:
            stack.append(SearchNode(env.schedule.copy(), env.remaining.copy(), actions, path_decisions))
            continue
        report = env.problem.config.validate(instance, torch.from_numpy(env.schedule))
        elapsed = perf_counter()-started
        timely = perf_counter() <= cutoff
        events.append(dict(kind="complete", actions=actions, feasible=report.feasible,
                           objective=report.objective, elapsed=elapsed, credited=timely and report.feasible))
        completions += 1
        if report.feasible and timely:
            if first_feasible is None:
                first_feasible = elapsed
            if report.objective > best_objective:
                best, best_objective = env.schedule.copy(), report.objective
                best_actions, best_decisions = actions, path_decisions
    returned = best if best is not None else deepest
    report = env.problem.config.validate(instance, torch.from_numpy(returned))
    runtime = perf_counter()-started
    success = report.feasible and runtime <= budget
    quality = report.objective / max(1., float(instance.priority.sum()) * instance.T)
    reward = 1+quality if success else .1 * len(deepest_actions) / instance.J
    return dict(schedule=returned.astype(int).tolist(), feasible=report.feasible,
                objective=report.objective if report.feasible else None, success=success,
                runtime=runtime, overrun=max(0., runtime-budget), cutoff=cutoff-started,
                time_to_first_feasible=first_feasible, reward=reward,
                actions=best_actions if best is not None else deepest_actions,
                decisions=best_decisions, events=events, backtracks=backtracks,
                rejected=rejected, completions=completions, sampled_decisions=decisions,
                completed=len(best_actions if best is not None else deepest_actions),
                deepest_completed=len(deepest_actions), exhausted=not stack,
                memoized=memoized, memoize=memoize), logs, entropies
