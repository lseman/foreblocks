# Lookahead Rollout Search for ONTS (LRBS-style)

## Overview

Implemented LRBS-inspired bounded multi-step lookahead search for reinforcement learning
policies on the Online Nanosatellite Task Scheduling (ONTS) problem. This addresses the
myopic nature of single-step PPO action selection where the policy picks an action without
seeing its downstream consequences.

## Motivation

From the [MIP/scheduling literature review](../research/outputs/rl-mip-scheduling-review-2026-10-06.md):

> **Add short lookahead and compact memory.** Compare one trajectory, independent restarts
> and small-beam one-to-three-step rollout search. Use PPO, imitation and random proposals
> inside the same search wrapper; compare recent-action features versus retrieved history.

LRBS (Camerota et al., AAAI 2025) demonstrated that limited multi-step rollouts inside
beam search significantly improve neural improvement policies for flexible job-shop
scheduling. The key insight: at each decision point, run K short rollouts from the current
state using the policy, then select the first action of the rollout with the best total reward.

## Implementation

### Files Created

1. **`projects/scheduling/rl_rollout.py`** — Core lookahead search module
   - `_deep_clone_env()` — Creates isolated environment clones for rollout execution
   - `_rollout_one()` — Runs a single short trajectory from a clone
   - `lookahead_select()` — Selects the best first action by evaluating K rollouts per candidate
   - `beam_search()` — Full beam search over repair trajectories (for longer budgets)
   - `rollout_with_lookahead()` — Main entry point; replaces single-step selection with lookahead
   - `evaluate_lookahead()` — Convenience wrapper for benchmarking

2. **`projects/scheduling/research/rl_rollout_study.py`** — Study script for comparing methods
   - Generates instance splits (train/development/OOD)
   - Evaluates: uniform, ppo_single, lookahead_k4_s2, lookahead_k8_s2, lookahead_k8_s3, lookahead_greedy
   - Writes summary statistics and paired comparisons

3. **`projects/scheduling/tests/test_rl_rollout.py`** — 16 unit tests covering all functions

## Usage

### Quick demo with a trained policy:

```python
from projects.scheduling.rl_rollout import rollout_with_lookahead, evaluate_lookahead
from projects.scheduling.rl_search import SearchEnv

# Option 1: Full rollout loop
env = SearchEnv(instance, initial_schedule, budget=3.0, seed=42)
trajectory = rollout_with_lookahead(
    env, policy,
    lookahead_k=8,      # 8 rollouts per candidate
    lookahead_steps=2,  # 2-step lookaheads
    greedy=True,        # Use argmax (or False for exploration)
    mode='ppo',
)

# Option 2: Single evaluation
result = evaluate_lookahead(
    instance, initial_schedule, policy,
    budget=3.0, seed=42,
    lookahead_k=8, lookahead_steps=2,
)
# result contains: incumbent, objective, improvement, trajectory, steps, ...
```

### Run the study:

```bash
OMP_NUM_THREADS=1 uv run python -m projects.scheduling.research.rl_rollout_study \
  --output /tmp/onts-lookahead-development \
  --episodes 64 \
  --eval-budget 1.2 \
  --policy-seeds 11 29
```

## Key Design Decisions

1. **Clone-based isolation**: Rollouts execute on shallow clones of the environment, not deep copies.
   This keeps overhead low while maintaining independence between rollouts.

2. **Candidate sampling**: Not all candidates are evaluated (too expensive). We sample a diverse
   subset weighted toward higher-utility operators (run, energy).

3. **Budget-aware**: Rollout evaluation time is tracked and capped. If the lookahead consumes
   too much budget, we fall back to STOP.

4. **Incumbent preservation**: Only the best action's result modifies the real environment.
   All rollouts are isolated — no side effects leak through.

5. **Time accounting**: Every operation (candidate generation, rollout execution, final action)
   is charged to the same wall-clock deadline as controls.

## Testing

All 271 existing tests pass + 16 new tests:

```bash
OMP_NUM_THREADS=1 uv run python -m pytest projects/scheduling/tests/test_rl_rollout.py -v
```

## Next Steps for Paper

1. **Train policies** on the development instances with both single-step PPO and lookahead PPO
2. **Run matched study** comparing: uniform, ppo_single, lookahead_k4_s2, lookahead_k8_s2, lookahead_k8_s3
3. **Ablate** rollout depth (1 vs 2 vs 3) and width (k=4 vs k=8)
4. **Test on OOD instances** to evaluate generalization
5. **Report** paired comparisons with bootstrap intervals

## References

- Camerota et al., "Limited Rollout Beam Search for Neural Scheduling Improvement" (AAAI 2025)
- [MIP/scheduling literature review](../research/outputs/rl-mip-scheduling-review-2026-10-06.md) — Proposed experiment #3
