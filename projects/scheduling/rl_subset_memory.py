"""Evaluation-only MAP diversification: skip failed groups at one incumbent.

This is a deterministic search intervention, not the stochastic policy used by
PPO. It deliberately exposes no training trajectory or on-policy likelihood.
"""
from __future__ import annotations

import heapq
from time import perf_counter

import torch

from .rl_bipartite import bipartite_observation


def best_untried_subset(logits, failed, *, deadline=None, clock=perf_counter):
    """Exact highest-density nonempty Bernoulli subset outside a finite taboo set.

    Start from the unconstrained MAP bits. Flipping bit j reduces log density by
    abs(logit_j). A best-first enumeration of flip combinations returns the first
    admissible subset. The nonempty normalizer is common to every candidate.
    Cost grows with the number of excluded subsets; this is not a scalable
    replacement for a learned correlated decoder.
    """
    logits = torch.as_tensor(logits).detach().cpu()
    if logits.ndim != 1 or not len(logits) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite task logits')
    width = len(logits)
    if len(failed) >= (1 << width)-1:
        return None
    base = tuple((logits >= 0).long().tolist())
    penalties = logits.abs().tolist()
    queue = [(0., ())]
    while queue:
        if deadline is not None and clock() >= deadline:
            return None
        _, flips = heapq.heappop(queue)
        bits = list(base)
        for j in flips:
            bits[j] = 1-bits[j]
        candidate = tuple(bits)
        if any(candidate) and candidate not in failed:
            return torch.tensor(bits, dtype=torch.long)
        # Increasing flip indices generate each combination exactly once.
        start = flips[-1]+1 if flips else 0
        for j in range(start, width):
            group = (*flips, j)
            heapq.heappush(queue, (sum(penalties[k] for k in group), group))
    return None


def memory_map_rollout(env, policy):
    failed = set()
    while not env.done:
        started = env.clock()
        observation, mask = bipartite_observation(env)
        if mask[0]:
            env.step(0)
            env.log[-1]['selection_runtime'] = env.clock()-started
            break
        with torch.no_grad():
            dist, _ = policy(observation, mask)
            bits = best_untried_subset(dist.logits, failed, deadline=env.deadline, clock=env.clock)
        if bits is None:
            # Every task group failed with this repair at this incumbent. This
            # is exhaustion of the tried groups, not a local-optimality proof:
            # randomized reprofile can produce different outcomes on a retry.
            env.done = True
            break
        selection_time = env.clock()-started
        previous = env.incumbent.clone()
        env.step_subset(bits)
        env.log[-1]['selection_runtime'] = selection_time
        env.log[-1]['excluded_groups'] = len(failed)
        if not torch.equal(previous, env.incumbent):
            failed.clear()
        elif env.log[-1]['executed']:
            failed.add(tuple(bits.tolist()))
    return env.incumbent
