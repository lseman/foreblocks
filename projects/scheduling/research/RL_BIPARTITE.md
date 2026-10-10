# Task--time learned neighborhoods

Read the [literature review](outputs/rl-bipartite-literature-2026-10-06.md),
[results review](results/rl-bipartite-20261006/RESULTS_REVIEW.md) and
[second paper](../paper2/main.pdf). The completed experiment improves the prior
controller on larger cases; graph messages and PPO-specific gains are not
established by its matched controls.

`rl_bipartite.py` provides `TaskNeighborhoodEnv` and `BipartiteActorCritic`.
The environment offers every task singleton/pair, three triple proposals and
eight MILP proposals. Feasible incumbents, validation, monotonic acceptance and
deadline checks retain their existing meaning. Enumeration and observation cost
must be measured before using task counts beyond the tested maximum of twelve.

From repository-root Python, given an `ONTSInstance` and a validated initial
schedule, replay a declared greedy hybrid checkpoint:

```python
import torch
from projects.scheduling.rl_bipartite import BipartiteActorCritic, TaskNeighborhoodEnv
from projects.scheduling.rl_portfolio import portfolio_rollout

torch.set_num_threads(1)
policy = BipartiteActorCritic.load(
    "projects/scheduling/research/results/rl-bipartite-20261006/graph_greedy-11.pt"
)
env = TaskNeighborhoodEnv(instance, initial_schedule, budget=1.2, seed=42)
portfolio_rollout(env, policy, greedy=True)
schedule = env.incumbent
```

Checkpoint loading uses `weights_only=True`. The replay-only diagnostic models
are stored separately in `continuation-diagnostic`; their observed means must
not be treated as a checkpoint-selection result on an untouched test set.

The [paper README](../paper2/README.md) supplies training, reproduction and
independent-audit commands. Results preserve exact instance tensors, all teacher
outcomes, source hashes/archive, training transitions and returned schedules.
