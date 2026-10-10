# Ordered task groups and repair choice

`rl_sequence.py` samples a repair operator, cardinality and a sequence of distinct
tasks. Each pointer decision reads the selected-task mean embedding. PPO retains
the sampled order and evaluates the exact product of operator, cardinality and
conditional task probabilities. Repair consumes the unordered set, but training
never mistakes one sequence probability for the sum over all its permutations.

Profile repair and two whole-task MILP allowances (0.025/0.1 seconds) share the
existing full validator and nondegrading deadline acceptance. Unaffordable
operators are masked. The decoder receives task–time graph features with a
mean message pass; it does not enumerate candidate groups. A prefix-off control
removes the selected-task mean while retaining cardinality, masks and parameter
shapes. Without-replacement sampling still creates dependencies in that control.

The experiment compares teacher-only initialization, matched label replay, PPO
and leave-one-out REINFORCE. Two groups of four rollouts per update share their
instance and initial incumbent. Each leave-one-out baseline uses the other
three final normalized improvements in its own group. Updates use fresh
trajectories once. No learned critic supplies this baseline. PPO uses the same
grouped instance schedule, with its own clipped policy/value loss. Both replay
the same supervised label indices and disable entropy regularization. The
compatibility wrapper supplies zero **regularization**, not an estimated joint
sequence entropy; PPO's entropy metric is explicitly `None`.

The adaptation draws on
[Kool et al., ICLR 2019](https://arxiv.org/abs/1803.08475) for partial-solution
autoregressive decoding and
[POMO, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html)
for related-rollout return baselines. It does not reproduce their routing state,
forced diverse first choices, greedy rollout baseline or augmented inference.

The frozen study is `results/rl-sequence-20261006`. Twelve training instances
supply three incumbent stages. Eight fresh development and six fresh larger
cases use seeds 11/29 and a two-second evaluation allowance that charges setup,
observations, neural selection, repair and validation. All twelve methods,
final checkpoints and **sampled-only** deployment are declared before evaluation.
Mixed/profile-only random groups share the respective repair interfaces.
The preceding factorized imitation checkpoint and full heuristic are full-recipe
controls. The mixed PPO checkpoint also runs with profile-only availability as
a declared deployment intervention, without retraining.

Reproduce from the repository root into a fresh directory:

```bash
ONTS_SOLVER_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m projects.scheduling.research.rl_sequence_study \
  --output /tmp/onts-ordered-development --episodes 64
```

Independent replay of completed evidence:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  projects/scheduling/research/results/rl-sequence-20261006/check_integrity.py
```

The result archive preserves exact instances, candidate teacher schedules,
matched supervision, all online training final schedules and transitions, source
hashes, models and evaluation schedules. `paper2/sequence.tex` describes the
method and protocol; completed numbers come from its saved `summary.tex`.
This is a synthetic development study, not a published ONTS benchmark replication.

Completed ordered-group study: 336 primary and 168 inference-only evaluations,
all independently replayed as feasible and nondegrading, with no overruns.
Primary larger-case PPO/replay means are 0.197033/0.195594 (paired interval
includes zero). Removing likelihood recomputation without retraining gives
0.197664/0.195657, exploratory paired difference 0.002007 with interval
[0.000139, 0.004052]. This uses examined instances, remains below the heuristic
mean 0.197848, and requires independent confirmation. Prefix conditioning and
MILP access have no established benefit. The full scheduling suite passes
202 tests. Evidence and audits: `research/results/rl-sequence-20261006/`.

Deployment avoids training likelihood and trajectory work:

```python
from projects.scheduling.rl_sequence import SequenceEnv, SequenceActorCritic
from projects.scheduling.rl_sequence_inference import sequence_inference

policy = SequenceActorCritic.load(checkpoint_path)
env = SequenceEnv(instance, initial_schedule, budget=2.0, mixed=True)
schedule = sequence_inference(env, policy)
```
