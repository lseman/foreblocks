"""Solver-imitation training for ONTS neighborhood selection.

Initialization repairs an all-off proposal to obtain a low-activity feasible
schedule, leaving room for improvement even on small instances.
Labels are changed execution cells in strictly improving random-neighborhood
repairs. Instances without such repairs produce no fabricated positive labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .environments.onts_env import ONTSInstance
from .hybrid import select_neighborhood
from .milp import ONTSConfig, ONTSMilp
from .neighborhood import GraphBuilder, NeighborhoodPolicy


@dataclass
class ImitationExample:
    instance_index: int
    incumbent: torch.Tensor
    target: torch.Tensor
    improvement: float


def train_neighborhood_policy(
    instances: list[ONTSInstance],
    *,
    hidden: int = 64,
    layers: int = 2,
    rounds: int = 8,
    epochs: int = 10,
    init_time: float = 0.5,
    repair_time: float = 1,
    fraction: float = 0.5,
    block_size: int = 4,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    config: ONTSConfig | None = None,
    initial_schedules: list[torch.Tensor] | None = None,
):
    """Collect improving repairs, then fit a variable-release classifier.

    Caller owns the train/test split. Returns a usable policy and diagnostics,
    even if no improvements are found; diagnostics explicitly say untrained.
    """
    if not instances or rounds < 1 or epochs < 1:
        raise ValueError("Provide training instances and positive rounds/epochs")
    if (
        init_time <= 0
        or repair_time <= 0
        or lr <= 0
        or not 0 < fraction <= 1
        or block_size < 1
    ):
        raise ValueError("Invalid training budgets or hyperparameters")
    if initial_schedules is not None and len(initial_schedules) != len(instances):
        raise ValueError("Provide one initial schedule per training instance")
    started = perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy = NeighborhoodPolicy(hidden, layers).to(device)
    problems = [ONTSMilp(instance, config) for instance in instances]
    builders = [GraphBuilder(p) for p in problems]
    examples: list[ImitationExample] = []
    feasible_instances = 0
    for index, problem in enumerate(problems):
        if initial_schedules is None:
            result = problem.solve(
                time_limit=init_time, proposal=torch.zeros(problem.J, problem.T)
            )
            incumbent, objective = result.schedule, result.objective
        else:
            incumbent = initial_schedules[index].detach().cpu().clone()
            report = problem.config.validate(problem.instance, incumbent)
            if not report.feasible:
                raise ValueError(f"Invalid training incumbent: {report.violations}")
            objective = report.objective
        if incumbent is None:
            continue
        feasible_instances += 1
        for _ in range(rounds):
            free = select_neighborhood(
                rng.random(problem.n_x),
                J=problem.J,
                T=problem.T,
                fraction=fraction,
                block_size=block_size,
            )
            repaired = problem.solve(
                time_limit=repair_time, incumbent=incumbent, free_mask=free
            )
            if repaired.objective > objective + 1e-6:
                target = (repaired.schedule != incumbent).float().ravel()
                examples.append(
                    ImitationExample(
                        index,
                        incumbent.clone(),
                        target,
                        repaired.objective - objective,
                    )
                )
                incumbent, objective = repaired.schedule, repaired.objective
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    losses = []
    policy.train()
    for _ in range(epochs if examples else 0):
        total = 0.0
        for index in rng.permutation(len(examples)):
            example = examples[int(index)]
            graph = builders[example.instance_index].build(example.incumbent).to(device)
            target = example.target.to(device)
            positive_weight = (
                (target.numel() - target.sum()) / target.sum().clamp(min=1)
            ).clamp(min=1, max=50)
            logits = policy(graph)
            loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        losses.append(total / len(examples))
    policy.eval()
    return policy, {
        "trained": bool(examples),
        "training_instances": len(instances),
        "feasible_initializations": feasible_instances,
        "improving_repairs": len(examples),
        "loss": losses,
        "label_and_training_runtime": perf_counter() - started,
        "seed": seed,
        "initialization": "provided-feasible"
        if initial_schedules is not None
        else "minimal-activity",
    }
