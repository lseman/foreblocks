# Search-mechanism diagnosis after longer training

The completed continuation does not establish a PPO benefit over teacher replay
or the short checkpoints. This diagnostic investigates repair headroom, training
reward density and MIStar's deployment mechanisms before another training change.

Training records are analyzed only on the preceding training split, with bins
for decisions 1–4, 5–8 and 9–16. A positive reward means normalized best-objective
increment above 1e-9; this suppresses numerical ties. Repeated proposals mean the
same full schedule, task group and operator appeared in a previous transition.
Acceptance temperature is ignored in this repeat definition. Within-batch
repeats are reported separately. A repeat is not necessarily wasted work:
profile repair can randomize reinsertion order, MILP may find another solution,
and perturbation is stochastic. Nothing is masked based on these statistics.

Headroom probes use twelve deterministically selected training states from
MIStar PPO seed 11: the first zero-reward transition at step eight or later
with working quality equal to best quality, for the first six small and first
six larger training-instance indices. This is a transient-stall sample, not a
claim of local optimality or a representative held-out performance estimate.
Every probe repairs the same frozen base schedule. It tests all current
candidate task groups with profile repair (up to 2 s), MILP at 0.05 and 0.25 s,
and all-task MILP at 2 s. Selection is offline and uncharged to an online search
competitor. Full candidate schedules, objectives, groups, solver status and
elapsed time are saved for independent replay.

A separate fresh-instance inference study uses unchanged continuation weights:

- Full MIStar PPO: memory embeddings, repetition bias and four proposals.
- Memory off: removes both memory embeddings and repetition bias.
- One proposal: preserves memory and bias, changes proposal count to one.
- Memory off plus one proposal: the remaining cell of the two-by-two interface.
- Memory embedding off: zeros the retrieved embedding while retaining its bias.
- Repetition bias off: retains memory embeddings, sets penalty coefficient zero.
- Original MIStar imitation, continuation DR-ALNS PPO, and companion heuristic.

Six fresh small and six fresh larger instances, two seeds and nine methods yield
216 evaluations at 1.2 s. All hashes are disjoint from previous RL manifests.
Larger instances occurred in the preceding training set, so this is held-out
performance rather than transfer to an unseen size. The protocol, sources and
input checkpoint/trajectory hashes are frozen before probing or evaluation.
Interventions receive the same online wall-clock budget. Proposal count changes
inference frequency, number of state updates and the history's decision depth;
these are deployment-interface effects. The intervention does not retrain the
policy, and distribution shifts can affect results. It does not establish the
benefit of a separately trained architecture. Marginal paired intervals use
instance means over two seeds and remain exploratory.

Run into a new result directory:

```bash
ONTS_SOLVER_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m projects.scheduling.research.rl_search_diagnostic \
  --output projects/scheduling/research/results/rl-search-diagnostic-20261007
```

Audit a completed run:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m projects.scheduling.research.audit_rl_search_diagnostic \
  --output projects/scheduling/research/results/rl-search-diagnostic-20261007
```

The auditor checks all returned, working and proposal schedules, fixed tasks,
reward accounting, exact logged cutoff eligibility, acceptance timestamps,
headroom states against original training transitions, source/checkpoint/record
hashes and saved aggregate statistics. Best-update timestamps are unavailable;
best changes must match eligible proposals. The companion heuristic retains its
conservative history-credit audit limitation. Actual overruns are counted.
