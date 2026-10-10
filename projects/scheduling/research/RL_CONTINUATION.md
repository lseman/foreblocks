# Longer training of MIStar-inspired and RL-controlled ALNS policies

This continuation starts from the completed 32-episode PPO checkpoints in
`results/rl-literature-20261006`. It preserves the original implementations,
checkpoints and evidence. It targets the two strongest short-run recipes;
this selection uses the preceding development study and is therefore disclosed.

The new protocol is fixed before evaluation:

- Two policy seeds, 11 and 29; two models, MIStar-inspired and DR-ALNS-inspired.
- Two continuation arms per model/seed: PPO and teacher-actor replay.
- Both arms start from exactly the same previous PPO weights, with new Adam
  optimizers at learning rate 0.0001. Original optimizer states were unavailable;
  this is weight continuation with an optimizer reset.
- 256 additional episodes per arm, an allowance of 0.8 seconds per episode and
  a cap of 16 decisions. Earlier training used 32 episodes, 0.35 seconds and
  four decisions. There are 2,048 continuation training outputs in total.
- 36 fresh synthetic training instances: 24 small and 12 larger instances.
  Teacher labels come only from these training instances. They are verified
  positive-gain repairs, with the existing top-three candidate teacher.
- PPO updates after four episodes: three epochs, minibatch 32, entropy
  coefficient 0.001, clip 0.2, undiscounted returns, trace lambda 1, KL target
  0.03 and gradient norm cap 0.5.
- Teacher replay uses teacher actor likelihood and return regression for the
  critic on its own collected trajectories. Its optimizer-step count equals
  the nominal PPO schedule before PPO's KL stopping. Actual steps, transitions
  and training time are reported; this is not exact compute matching.
- MIStar's teacher labels a single categorical proposal. Deployment and PPO
  retain the full likelihood of four independently sampled proposals.
- Snapshots at 64-episode intervals are retained for recovery. Only the final
  continuation checkpoint is evaluated; snapshots are not ranked or selected.
- Fresh held-out evaluation uses six small and twelve larger instances, two
  seeds, twelve methods and a common 1.2-second allowance: 432 evaluations.
  Controls include the original imitation and short-PPO checkpoints, replay
  continuation, classical ALNS, uniform selection, random four-proposal search
  and the companion heuristic. Method order is randomized.
- All new instance hashes must be disjoint from all earlier RL manifests.
  Larger instances also appear in training: the larger held-out split is no
  longer a size-out-of-distribution test. All data still use the disclosed
  planted synthetic generator.

Because training duration, horizon and data coverage change together, this
compares complete continuation recipes. It does not isolate training duration
or establish a causal benefit of memory, multiple proposals or annealing.
PPO-versus-replay and PPO-versus-short contrasts distinguish additional training
from an RL-specific improvement as far as these controls allow. Two policy
seeds and marginal instance-bootstrap intervals remain exploratory.

Shared-engine evaluation logs the exact setup-adjusted acceptance cutoff,
proposal validation times and working-state acceptance times. Returned quality,
full feasibility, reward accounting and unchanged tasks can be replayed.
Best-update timestamps are not logged; best changes must match eligible
proposals. The companion heuristic retains its previous credit convention and
its audit limitation: credited quality is checked against a feasible returned
schedule and a conservative history cutoff.

Run into a new output directory:

```bash
ONTS_SOLVER_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m projects.scheduling.research.rl_literature_continuation \
  --output projects/scheduling/research/results/rl-literature-continuation-20261006
```

The run freezes the plan, input checkpoint hashes, all scheduling Python source,
instance manifest, environment and source archive before training. Final model
hashes are written before evaluation; optimizer states and RNG states are also
saved. Large trajectory records use gzip JSON lines. The auditor checks saved
schedules and accounting rather than regenerating runtime-dependent trajectories:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m projects.scheduling.research.audit_rl_continuation \
  --output projects/scheduling/research/results/rl-literature-continuation-20261006
```

For paper2, copy the completed `summary.tex` to
`paper2/tables/literature_continuation_summary.tex` and use the exact saved
paired contrasts. Failed or inconclusive comparisons must remain in the paper.

## Completed results

The full continuation collects 32,487 decisions and performs 3,072 optimizer
steps (384 per arm), taking 747.42 seconds plus approximately 26 seconds for
teacher generation. All final checkpoints were fixed before fresh evaluation.

On twelve larger held-out instances, MIStar longer PPO reaches 0.190834, versus
0.190385 for short PPO, 0.191289 for replay and 0.191398 for original imitation.
DR-ALNS longer PPO reaches 0.190844, versus 0.190083 for short PPO and 0.191277
for replay. Longer-minus-short and PPO-minus-replay paired intervals include
zero for both models. Smaller-case PPO means do not improve over short PPO.
Additional PPO continuation therefore has no established benefit here.

The heuristic larger-case mean is 0.187593. Longer PPO's exploratory paired
differences against it are 0.003241 [0.000278, 0.007276] for MIStar and
0.003251 [0.000268, 0.007375] for DR-ALNS. These favor the complete recipes,
without attributing the advantage to additional PPO or claiming published SOTA.

Independent replay passes 432 evaluation outputs, 2,048 training outputs,
864 teacher repairs, 16 checkpoint loads and eight optimizer states. It checks
56,648 transitions and 120,809 proposals; 295 late feasible proposals are
excluded. One original DR-ALNS imitation evaluation takes 1.349091 seconds,
an overrun of 0.149091 seconds; every continuation PPO/replay evaluation stays
within 1.2 seconds. The overrun's final unlogged work is not separately timed,
so its cause is not identified. The full scheduling suite passes 220 tests.

The archived auditor's 1e-12 nominal budget-bound tolerance was too tight for
subtracting large monotonic clock values. The result-local `audit_corrected.py`
uses 1e-9 for that bound only. Recorded cutoff eligibility and acceptance
comparisons remain exact. `audit-correction.json` preserves the reason and
hashes, and the experiment source archive is unchanged. Deliberate reward
and cutoff corruptions fail replay. Recheck this completed frozen study with:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python \
  projects/scheduling/research/results/rl-literature-continuation-20261006/audit_corrected.py \
  --output projects/scheduling/research/results/rl-literature-continuation-20261006
```

Training curves use 32-episode blocks, averaging two seeds; shading is their
range, not a confidence interval. Different sampled-instance composition
prevents reading the slope as fixed-instance convergence. These curves were
created after evaluation and did not select checkpoints. Artifacts:
`summary.json`, `summary.tex`, `independent-audit.json`,
`cutoff-relation-audit.json`, `audit-negative-controls.json`,
`training-curve.csv`, `training-curve.pdf` and their provenance records.
