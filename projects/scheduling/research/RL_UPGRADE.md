# RL upgrade: methods, reproducibility and evidence limits

Implemented 6 October 2026. This is an experimental ONTS learning system,
not a demonstrated state-of-the-art ONTS solver.

## What changed

The first smoke run stopped voluntarily in 9 of 12 training episodes and 9 of
12 PPO evaluation runs. In the fixed-budget, quality-only improvement objective,
validated repairs cannot reduce the incumbent objective. STOP therefore has no
quality advantage while a repair is affordable. The new exhaustion mode masks
STOP until the deadline is too close for a repair. Optional STOP remains
available for the legacy comparison and future objectives charging compute.
Every method in the controlled MILP selection comparison uses the same four
candidate families: random, utility, run and energy. They release binary
execution decisions, and the MILP repair enforces the complete model.

The stabilized trainer collects eight complete episodes before updating. It
uses gamma=1 generalized advantages, lambda=.95, batch advantage normalization,
clipped policy and value losses, entropy regularization, gradient clipping and
an approximate-KL stop. Zero initial critic outputs prevent an arbitrary initial
value offset from dominating the small normalized gains. Advantages never cross
episode boundaries. Frozen observations and masks are replayed for likelihood
ratios; no new candidate generation occurs inside a PPO update.

The graph policy encodes all execution, startup and battery variables and actual
MILP constraints. Coefficient-aware sparse attention passes messages in both
directions. Constraint-family embeddings distinguish startup, duration,
periodicity, power and battery equations. Readout pools each candidate's released
variables, alongside a global graph representation. Search history includes
last released/changed cells, stagnation, previous outcomes, and per-operator gain,
time, frequency and recency. Candidate order does not change which physical
neighborhood receives a score. This compact history remains an approximation to
complete search state, rather than a claim of an exact Markov representation.

For the warm start, independent offline repairs evaluate every candidate from
the same incumbent. Two states per training instance include initial and
partially improved schedules. A best-anchored pairwise loss learns preferences
between complete candidate actions, weighting strict preferences by normalized
objective differences. Ties generate no preference gradient. Teacher time and
pretraining time are separate from online search; no development, validation or
OOD instance supplies labels. The graph-preference checkpoint is retained before
PPO for a direct comparison with its RL continuation.

## Research basis

[MIStar, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/77629f2df5ef97ad967157c69cd8af7b-Abstract-Conference.html)
uses heterogeneous graphs and historical trajectories for learned improvement
in flexible job-shop scheduling. It motivates the graph/history direction here;
ONTS uses its own MILP variable/constraint graph and repair actions.

[BOPO, ICML 2025](https://proceedings.mlr.press/v267/liao25a.html)
uses best-anchored, objective-guided solution preferences for neural combinatorial
optimization, including scheduling. Our complete-neighborhood preference loss
is an adaptation, not BOPO's original sequence objective or a replication.

[ReSched, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/467e109270678b3e2b3c2e7fa6b3d575-Abstract-Conference.html)
provides a recent scheduling Transformer comparator. Its FJSP state reduction
does not establish sufficiency for recurring starts and battery dynamics. The
current implementation uses sparse graph attention; it does not implement
ReSched's decoder or Transformer blocks.

The PPO stabilization components are established algorithmic practice, not a new
contribution. The contribution sought is whether resource-aware learned search
improves ONTS quality under equal runtime and transfers beyond its training sizes.

## Run and inspect

From the repository root:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m projects.scheduling.research.rl_upgrade \
  --output /tmp/onts-rl-upgrade --episodes 32 \
  --train-budget 0.9 --eval-budget 1.2 --policy-seeds 11 29
```

The output directory must be new. The runner saves a disjoint manifest, exact
instance tensors, source hashes and a source archive, teacher outcomes,
training and evaluation JSONL, checkpoints, offline costs, paired analysis and
a report. Training and online evaluation each use one CPU thread. The final
checkpoint is fixed in advance; validation does not select a favorable model.
PPO uses stochastic evaluation, with matched search seeds and randomized method
order. These two seeds pair training and evaluation variation; they do not
separately estimate their variances.

Load a graph or compact checkpoint without unpickling custom Python objects:

```python
from projects.scheduling.rl_learning import load_rl_policy
from projects.scheduling.rl_search import SearchEnv, rollout

policy = load_rl_policy("/tmp/onts-rl-upgrade/graph_preference_ppo-11.pt")
# instance and incumbent must follow the model's single-satellite conventions.
env = SearchEnv(instance, incumbent, budget=3, stop_mode="exhaustion")
rollout(env, policy)
schedule = env.incumbent
```

Initialization, graph construction, candidate generation, inference, repair and
validation consume the online allowance. Native solver limits are cooperative:
log actual runtime and overruns, and reject candidates validated after the search
deadline. Reward sums telescope to final normalized improvement. Returned
incumbents remain independently validated and nondegrading even after solver
failure, timeout or a late result.

## What the comparison can establish

Legacy versus stable PPO compares the entire stabilization package, including
STOP masking, rather than isolating each component. Graph PPO versus stable PPO
compares representation and history under the same stabilized trainer. Preference
plus PPO versus graph PPO tests the warm start. Preference plus PPO versus the
preference checkpoint tests whether RL adds value. Uniform, utility and
contextual-UCB controls test whether learned selection is useful. The existing
profile heuristic is a stronger end-to-end comparator using a different action
portfolio.

These are small synthetic planted splits: 12 training, 3 validation, 6 development
and 3 larger OOD instances. OOD changes size within the same generator family.
Two policy seeds are averaged within each instance for exploratory paired
bootstrap intervals. There is no confirmatory holdout, external benchmark,
cold-start or constellation evaluation, proof of optimality, or correction for
multiple exploratory comparisons. Architecture, memory and constraint-semantics
claims need independent retrained ablations. A Transformer or a larger GNN
should be tested only if its added inference cost is justified by results.

## Completed development evidence

The [paired experiment review](results/rl-upgrade-20261006/RESULTS_REVIEW.md)
records the two-seed run and independent replay of 252 main/intervention
schedules. Stabilized execution improves over original PPO, primarily through
STOP handling. Graph and preference training have not yet demonstrated additional
learning benefit; the full profile heuristic remains stronger. This result
motivates a learned profile/MILP action portfolio rather than an ONTS SOTA claim.
