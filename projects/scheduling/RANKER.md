# Semantic neighborhood ranking

This extension learns which complete repair neighborhood to execute. It keeps
the audited ONTS MILP and complete-schedule validator. The original variable
imitation policy remains available for comparisons and existing checkpoints.

## What changes

- **Comparative outcome training.** Several candidates are repaired from the
  same incumbent under the same time cap. Zero-gain and time-limited outcomes
  remain in the training records. Pairwise realized-gain preferences are combined
  with gain and log-runtime calibration. No label claims that a time-limited
  neighborhood cannot improve.
- **Sparse semantic attention.** The graph includes every actual MILP coefficient
  and numerical slack, plus ten row categories for startup, count, period, run,
  power and battery constraints. Attention runs along sparse coefficient edges;
  it does not create dense attention over every variable/constraint pair.
- **Structured candidate moves.** Random blocks, priority/power utility blocks,
  complete-run relocation regions and energy-pressure time regions compete.
  Run proposals close over every active run they touch and respect a maximum
  released-cell count. When no complete run fits, their fallback releases only
  inactive cells. Windows remain enforced by variable bounds.
- **Search memory.** Recent release masks, changed cells, startup counts, run
  age, stagnation, last gain/runtime and remaining budget condition selection.
- **Controls and adaptation.** `uniform` samples the same candidate pool without
  a neural score. `bandit` learns operator selection online without a GNN.
  `ranked` chooses candidates with the learned score; `adaptive` combines bandit
  operator selection with learned within-operator ranking. Exploration is
  explicit and seeded. None guarantees matching the best constituent operator.

One state encoding scores all candidate masks, and only the chosen candidate
receives a repair solve during inference. Teacher generation executes more
candidate repairs during training; its cost is recorded separately. The cost
head is an auxiliary prediction, not an unverified gain-per-second objective.

Every repair retains the previous validated schedule unless a valid improvement
is found. Only x outside the mask is fixed; startup and battery variables remain
free. Restricted bounds are never reported as global bounds. Constraint category
predictions do not turn inequalities into equalities or relax the physical model.

## Run

From the repository root, using the existing virtual environment:

```bash
.venv/bin/python -m projects.scheduling.examples.train_ranker \
  --instance-dir path/to/training-only-json \
  --output runs/onts/ranker.pt --rounds 4 --epochs 12

.venv/bin/python -m projects.scheduling.examples.run_hybrid \
  --instance-dir path/to/heldout-json \
  --methods exact random heuristic uniform bandit ranked adaptive \
  --ranker-checkpoint runs/onts/ranker.pt \
  --time-limit 30 --repair-time 1 --fraction 0.35 \
  --output runs/onts/ranker-benchmark.json
```

To include the original learned selector, pass its separate `--checkpoint` and
add `neural` to the methods. Ranker evaluation rejects content overlap with its
recorded training identities and mismatched model configurations. Checkpoints
include architecture, constraint categories, training outcomes and source hashes.
Training saves diagnostics without claiming a trained model when all candidate
gains are tied. Evaluation defaults to one Torch/HiGHS thread; `--threads` changes
both. Start a fresh process for a different native solver thread count.

CLI summaries credit only objectives recorded by the requested deadline.
`returned_objective`, `returned_feasible` and `overrun` preserve eventual outputs.
The result histories are internal instrumentation, not a streaming API.

## Fresh pilot

```bash
.venv/bin/python -m projects.scheduling.research.ranker_study \
  --output projects/scheduling/research/results/ranker-pilot
```

The pilot freezes source/data, trains on nine fresh identities, validates on
three, and compares eleven methods on six IID and three larger OOD identities
using one training/search seed and three-second common-incumbent budgets. The
original learned policy is retrained on the same training identities, with its
lower teacher-call count recorded. Both learners share initialization schedules.
Candidate-pool methods share the validation-selected fraction. Legacy fractions
remain fixed from the prior study. The semantic/memory interventions mask a
trained model at inference; they are not retrained architecture comparisons.

These 99 comparisons are an exploratory engineering check, not a new confirmatory
study or a state-of-the-art claim. The earlier 1,314-run study and its source
archive are retained. Resuming an incomplete pilot rejects changed code or data.
A new implementation requires a new output directory. The older study's live
source checker may reject newly added modules; reproduce that version from its
saved source archive rather than rewriting its frozen hashes.

## Evidence and research context

The [literature review](research/outputs/neural-scheduling-review-2026-10-05.md)
connects this design to contrastive LNS, objective-guided preferences, semantic
MILP representation and memory-enhanced improvement. The implementation is an
ONTS adaptation, not a faithful reproduction of TLNS, BOPO, MIStar or ReSched.

Tests cover candidate bounds and reproducibility, complete-run preservation,
preference direction including tied/zero outcomes, sparse-attention gradients,
checkpoint round trips, candidate-order and job-relabeling invariance, nonmutating
feature interventions and incumbent preservation across every search mode.
All 99 pilot outputs passed independent validation; the suite passes 123 tests.
The ranker averaged +3.62% versus old imitation on OOD instances but trailed
heuristic and uniform selection. Superiority is not established, and the heuristic
default is retained. See the [pilot review](research/results/ranker-pilot/RESULTS_REVIEW.md)
and [complete comparisons](research/results/ranker-pilot/REPORT.md).

## Confirmatory evidence

The subsequent frozen study completed 720 comparisons on 60 fresh instances
with three paired training/search seeds and matched teacher-call budgets.
It supports a small mean gain over uniform selection (+0.284%, simultaneous
95% CI [0.041%, 0.526%], Holm p=0.0166), but not superiority over imitation
or heuristic selection. The joint >1% practical-improvement criterion failed.
All outputs passed validation; all 127 tests pass. See the
[protocol](research/CONFIRMATORY.md) and
[results and supported paper wording](research/results/ranker-confirmatory/RESULTS_REVIEW.md).
