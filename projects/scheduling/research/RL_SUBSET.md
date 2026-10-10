# Compact learned task subsets

`rl_subset.py` implements a factorized task-set policy inspired by
[Wu et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/file/fc9e62695def29ccdb9eb3fed5b4c8c8-Paper.pdf).
It selects any nonempty task subset, rather than scoring an enumerated portfolio.
The exact likelihood conditions independent Bernoulli decisions on nonemptiness;
PPO includes that normalization. Sampling chooses the first included task and
then independent subsequent bits, avoiding rejection loops. Greedy deployment
uses the conditional MAP subset.

The compact encoder mean-pools eligible task–time edge messages. Its observation
storage is O(JT); the previous candidate-specific release tensor was O(PJT).
This variant uses profile repair only, with the same complete validator,
nondegrading acceptance and deadline checks. It has no failed-subset masking or
MILP actions. This makes comparisons to the previous portfolio full-recipe
comparisons, rather than a representation-only ablation.

Supervision combines validated one-step repairs and rerun-verified hindsight
subsets, adapting a direction from
[SPL-LNS, 2025 preprint](https://arxiv.org/abs/2508.16171).
Changed tasks in a future trajectory are candidate labels; a fresh repair from
the earlier incumbent must actually produce a feasible positive gain. This is
not a reproduction of SPL-LNS's annealed, nonmonotone search.

The frozen experiment is `results/rl-subset-20261006`. It declares final
checkpoints, sample and MAP deployment, one-step imitation, hindsight imitation,
replay-only continuation, PPO continuation, a graph-off control, uniform-
cardinality random subsets, the preceding replay graph checkpoint and the
profile heuristic. Training uses twelve instances at three incumbent stages;
evaluation uses eight new development and six new larger instances, two seeds,
and a 1.2-second wall-clock budget including policy work. These are synthetic
development instances, not a published ONTS benchmark replication.

Reproduce from the repository root into a fresh directory:

```bash
ONTS_SOLVER_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m projects.scheduling.research.rl_subset_study \
  --output /tmp/onts-subset-development --episodes 64
```

Replay and audit saved evidence:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  projects/scheduling/research/results/rl-subset-20261006/check_integrity.py
```

The result directory preserves source hashes/archive, exact instances, candidate
and hindsight schedules, supervision, training logs and final evaluation
schedules. Paper 2 includes the method and protocol in `subset.tex`; its generated
numerical table comes from the completed experiment's `summary.tex`.

The primary run validates all 308 final schedules, 1341 candidate teacher repairs,
288 hindsight-trajectory repairs and 270 labels. On the six larger cases,
sampled hindsight imitation reaches 0.204439, replay-only sampling 0.204052 and
PPO sampling 0.202998. Random subsets reach 0.203754 and the preceding graph
checkpoint 0.199919. These means are normalized incumbent improvement, not
optimality gaps. PPO does not improve its teacher/replay controls.

MAP deployment repeatedly selects an unsuccessful group at an unchanged
incumbent (1687 of 1721 larger-instance PPO moves). Its mean is 0.103409.
The separate `memory-diagnostic` evaluates an exact best-untried MAP decoder from
`rl_subset_memory.py`, with unchanged checkpoints. This is post-hoc evaluation
on the examined instances; it is not independent confirmation or a PPO update.
No training trajectory is returned because deterministic taboo-set decoding
changes action selection from the stochastic policy used in training.

Mean larger-instance selection work is 0.0761 s for PPO sampling versus 0.2315 s
for the preceding graph controller; their repair counts differ. The compact
model has 4154 parameters. Both implementations run on CPU in this study.

The post-hoc memory control restores larger-case PPO MAP quality to 0.195455,
but remains below stochastic PPO (0.202998). See the
[full results review](results/rl-subset-20261006/RESULTS_REVIEW.md).
All 308 primary and 112 supplementary final schedules pass independent replay.
The scheduling suite passes 195 tests.

Replay the declared stochastic PPO checkpoint on a validated incumbent:

```python
import torch
from projects.scheduling.rl_subset import SubsetActorCritic, SubsetEnv, subset_rollout

torch.set_num_threads(1)
policy = SubsetActorCritic.load(
    'projects/scheduling/research/results/rl-subset-20261006/ppo-11.pt'
)
env = SubsetEnv(instance, initial_schedule, budget=1.2, seed=42)
subset_rollout(env, policy, greedy=False)
schedule = env.incumbent
```

This is a replay example, not a recommendation chosen from an untouched test
set. The teacher-only models outperform PPO in the larger-case development study.
