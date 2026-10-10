"""Train complete-neighborhood rankings from matched, budgeted repair outcomes."""

from __future__ import annotations

import math
from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .environments.onts_env import ONTSInstance
from .milp import ONTSConfig, ONTSMilp
from .ranking import (
    NeighborhoodRanker,
    RankingGraphBuilder,
    SearchState,
    make_candidates,
)
from .research.datasets import instance_hash, source_hashes


def outcome_loss(scores, costs, gains, observed_costs):
    """Pairwise realized-gain preferences plus gain/cost calibration.

    Zero gains remain observations. A timeout label means no observed gain
    within its cap, not mathematical neighborhood infeasibility.
    """
    differences = gains[:, None] - gains[None, :]
    pairs = differences > 1e-7
    if pairs.any():
        margins = scores[:, None] - scores[None, :]
        weights = differences[pairs] / differences[pairs].max().clamp(min=1e-7)
        ranking = (F.softplus(-margins[pairs]) * weights).sum() / weights.sum()
    else:
        ranking = scores.sum() * 0
    return (
        ranking
        + 0.2 * F.smooth_l1_loss(scores, gains)
        + 0.1 * F.smooth_l1_loss(costs, observed_costs)
    )


def train_neighborhood_ranker(
    instances: list[ONTSInstance],
    *,
    initial_schedules=None,
    config=None,
    hidden=32,
    layers=2,
    rounds=4,
    epochs=12,
    repair_time=0.25,
    init_time=0.5,
    fraction=0.35,
    candidate_count=8,
    block_size=4,
    seed=0,
    lr=1e-3,
    device="cpu",
    semantics=True,
    memory=True,
):
    if not instances or rounds < 1 or epochs < 1:
        raise ValueError("Provide instances and positive rounds/epochs")
    if not all(
        math.isfinite(value) and value > 0 for value in (repair_time, init_time, lr)
    ):
        raise ValueError(
            "Training time limits and learning rate must be positive and finite"
        )
    if not 0 < fraction <= 1 or candidate_count < 4 or block_size < 1:
        raise ValueError("Invalid candidate parameters")
    if initial_schedules is not None and len(initial_schedules) != len(instances):
        raise ValueError("Provide one initial schedule per instance")
    config = config or ONTSConfig()
    started = perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy = NeighborhoodRanker(hidden, layers, semantics=semantics, memory=memory).to(
        device
    )
    groups, outcomes = [], []
    initializations = 0
    for index, instance in enumerate(instances):
        problem = ONTSMilp(instance, config)
        builder = RankingGraphBuilder(problem)
        if initial_schedules is None:
            incumbent = problem.solve(
                time_limit=init_time, proposal=torch.zeros(problem.J, problem.T)
            ).schedule
        else:
            incumbent = initial_schedules[index].detach().cpu().clone()
            if not config.validate(instance, incumbent).feasible:
                raise ValueError("Training initial schedule is infeasible")
        if incumbent is None:
            continue
        initializations += 1
        state = SearchState.empty(problem.J, problem.T)
        scale = max(float(instance.priority.sum()) * problem.T, 1.0)
        for step in range(rounds):
            original = incumbent.clone()
            objective = config.validate(instance, original).objective
            candidates = make_candidates(
                problem,
                original,
                state,
                rng,
                fraction=fraction,
                count=candidate_count,
                block_size=block_size,
            )
            graph = builder.build_state(
                original,
                state,
                remaining_fraction=1 - step / rounds,
                repair_time=repair_time,
            )
            results = []
            # Randomize execution order to reduce systematic timing effects.
            results = [None] * len(candidates)
            for candidate_index in rng.permutation(len(candidates)):
                candidate_index = int(candidate_index)
                candidate = candidates[candidate_index]
                result = problem.solve(
                    time_limit=repair_time, incumbent=original, free_mask=candidate.mask
                )
                results[candidate_index] = result
                outcomes.append(
                    {
                        "instance_index": index,
                        "step": step,
                        "candidate_index": candidate_index,
                        "operator": candidate.operator,
                        "released_cells": int(candidate.mask.sum()),
                        "gain": max(0.0, result.objective - objective),
                        "runtime": result.runtime,
                        "repair_cap": repair_time,
                        "status": result.status,
                    }
                )
            gains = torch.tensor(
                [max(0.0, r.objective - objective) / scale for r in results],
                dtype=torch.float32,
            )
            costs = torch.tensor(
                [math.log1p(r.runtime / repair_time) for r in results],
                dtype=torch.float32,
            )
            groups.append((graph, candidates, gains, costs))
            winner = int(torch.argmax(gains))
            incumbent = results[winner].schedule
            state.advance(
                candidates[winner].mask,
                original,
                incumbent,
                results[winner].objective - objective,
                results[winner].runtime,
            )
    informative = sum(
        bool((gains.max() - gains.min()) > 1e-7) for _, _, gains, _ in groups
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    losses = []
    policy.train()
    for _ in range(epochs if informative else 0):
        total = 0.0
        for index in rng.permutation(len(groups)):
            graph, candidates, gains, costs = groups[int(index)]
            scores, predicted_costs = policy(graph, candidates)
            loss = outcome_loss(
                scores, predicted_costs, gains.to(device), costs.to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        losses.append(total / len(groups))
    policy.eval()
    metadata = {
        "format": "onts-ranker-training-v1",
        "trained": informative > 0,
        "training_instances": len(instances),
        "feasible_initializations": initializations,
        "groups": len(groups),
        "informative_groups": informative,
        "repair_outcomes": len(outcomes),
        "zero_gain_outcomes": sum(o["gain"] <= 1e-6 for o in outcomes),
        "improving_repairs": sum(o["gain"] > 1e-6 for o in outcomes),
        "loss": losses,
        "outcomes": outcomes,
        "seed": seed,
        "config": asdict(config),
        "training_hashes": [instance_hash(i) for i in instances],
        "source_hashes": source_hashes(),
        "fraction": fraction,
        "candidate_count": candidate_count,
        "block_size": block_size,
        "repair_time": repair_time,
        "semantics": semantics,
        "memory": memory,
        "runtime": perf_counter() - started,
        "objective": "same-incumbent realized-gain pairwise ranking; gain and log-cost calibration",
    }
    policy.training_metadata = metadata
    return policy, metadata
