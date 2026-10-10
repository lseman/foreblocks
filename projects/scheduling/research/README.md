# ONTS research protocol

The separate [RL manuscript](../paper2/README.md) is compiled as
[paper2/main.pdf](../paper2/main.pdf). Its profile/MILP portfolio experiment is
saved in `results/rl-portfolio-20261006`; the independent audit verifies all 192
evaluation schedules. These are exploratory development results, separate from
the frozen local study below.

The [task--time graph literature review](outputs/rl-bipartite-literature-2026-10-06.md)
documents the subsequent concrete-neighborhood implementation and its matched
controls. Its frozen experiment is in `results/rl-bipartite-20261006`.
See [RL_BIPARTITE.md](RL_BIPARTITE.md) for checkpoint replay and
[`RESULTS_REVIEW.md`](results/rl-bipartite-20261006/RESULTS_REVIEW.md) for the
420 primary and 126 supplementary runs and their interpretation.

The [compact task-set extension](RL_SUBSET.md) replaces candidate enumeration
with exact nonempty-subset sampling and adds repair-verified hindsight labels.
Its frozen study is `results/rl-subset-20261006`, with matched imitation,
replay-only and PPO continuation controls.

The [ordered-group extension](RL_SEQUENCE.md) tests conditional task selection,
MILP/profile repair choice and grouped leave-one-out REINFORCE against PPO and
teacher-replay controls. Its frozen study is `results/rl-sequence-20261006`.

This package tests correctness, incumbent improvement and comparative
performance as separate claims. The local study is bounded by the user's
one-hour compute budget; its short budgets and instance families define the
scope of the resulting evidence.

The [2026 literature review](outputs/neural-scheduling-review-2026-10-05.md)
compares recent neural MILP and scheduling methods, verified code, transfer
limitations and a proposed ONTS experiment sequence. Its recommendations are
hypotheses; they do not change the completed study's results.

## Reproduce or resume

From the repository root:

```bash
bash projects/scheduling/research/autoresearch.sh
```

The frozen local study is saved under `results/local-study`. Individual runs
are appended durably to `runs.jsonl`; repeating the command skips completed
runs. It repairs an interrupted last JSONL record. Resume rejects changed source
hashes, changed manifest contents, changed instance semantics and duplicate
instance content. A new implementation requires a new output directory.

For a completed study, the shell command regenerates reports without rerunning
optimization. This local study required a reporting-only correction for unknown
reference objectives. `results/local-study/reporting-fix.json` records the old
and corrected reporting hashes; `frozen-source.tar.gz` preserves the original
experiment source. Every optimization source hash remains unchanged.

Source hashes cover the scheduling implementation, environment loader and
research Python modules. Checkpoint metadata records training instance hashes,
model parameters and source hashes. Evaluation rejects content overlap with
training. Instance identities use the loader's float32 parameter representation.

A new pilot:

```bash
MPLCONFIGDIR=/tmp/onts-matplotlib .venv/bin/python -m \
  projects.scheduling.research.study \
  --output /tmp/onts-new-pilot --pilot --no-bundled
```

A larger experiment uses a new output directory and JSON plan overrides:

```bash
.venv/bin/python -m projects.scheduling.research.study \
  --output runs/onts-longer-study --plan-overrides longer-plan.json
```

Example overrides (estimate compute cost before increasing all dimensions):

```json
{
  "budgets": [10.0, 30.0, 60.0],
  "bundled_budgets": [10.0, 30.0],
  "train_per_regime": 20,
  "test_per_regime": 20,
  "ood_per_regime": 10,
  "reference_time": 120.0
}
```

Phases can be run separately with `--phase prepare`, `audit`, `train`,
`references`, `validate`, `benchmark` or `report`. The report phase regenerates
CSV, Markdown and PNG/PDF figures from saved data. Native solver calls run with
one HiGHS thread; PyTorch uses one thread. The study invokes SciPy's documented
option forwarding to specify HiGHS threads. Start a fresh Python process to
avoid conflicting with an already-initialized HiGHS global thread scheduler.

For the completed `local-study`, use the shell wrapper to regenerate reports:
its frozen plan predates the documented reporting correction, so source-checked
study phases deliberately reject the corrected report hash. Optimization must
not be resumed under changed source hashes. Independent evidence replay is:

```bash
PYTHONPATH=. .venv/bin/python \
  projects/scheduling/research/results/local-study/check_integrity.py
```

## Frozen local study

- 12 planted-witness training instances and 6 validation instances.
- 12 held-out IID test instances and 6 larger OOD instances.
- All 42 bundled JSON instances, used only for external cold evaluation.
- Three paired training/search seeds; seed means are paired within instance.
- Four methods: exact, random LNS, heuristic LNS and learned LNS.
- Three common-incumbent interventions: untrained weights, masked future solar
  feature, and removed message-passing edges.
- Cold synthetic budgets: 1 and 3 seconds; bundled budget: 1 second.
- Common-incumbent budget: 3 seconds on synthetic test/OOD data.
- 1,314 planned benchmark runs, plus independent training, validation,
  reference and correctness computations.

The common-incumbent track gives each method the same validated initial
schedule. Setup cost is reported separately. This schedule is retained as a
fallback; SciPy does not inject it as a native MIP start. The exact comparator
is a single unrestricted solve, while LNS repeatedly solves restricted models.

Search fractions are selected on validation data for every selector separately.
All other choices are frozen in `plan.json`. The full dataset seed differs
from the exploratory pilot seed. The exploratory pilot does not supply
confirmatory test evidence.

## What is measured

Returned feasibility and objective are independently checked. Fixed-budget
scores credit only objectives observable by the requested deadline, using
recorded history timestamps. Outputs arriving later remain in raw records;
the report does not silently credit their additional compute. A small reserve
in the search loop leaves headroom for validation. Actual runtime and overrun
are recorded.

Feasible rate uses all runs, including unknown outcomes. Objective comparisons
require both methods to be feasible; these conditional pairs and unmatched
feasibility outcomes are reported separately. Repeated seeds are averaged
within instance before bootstrap intervals and paired sign tests. Holm
adjustment covers all reported sign tests; plotted 95% intervals are marginal,
not simultaneous. No optimality assumption is made for a solver incumbent.

AUC integrates the internally recorded incumbent objective divided by the common
best-known feasible objective. Before any feasible schedule becomes available
its score is zero. The common reference combines independently validated
reference and benchmark incumbents. This is a best-known quality metric, not
an optimum-based primal integral. Cases without a positive reference are
ineligible for that normalization.

LNS logs each repaired incumbent, while SciPy exposes an unrestricted MILP
incumbent only when the native call returns. AUC therefore has asymmetric
observability and cannot establish superiority over the solver's native
incumbent trajectory. The API does not stream these events to a caller. Use
deadline objective and feasibility comparisons for the primary claim.

The inference interventions do not retrain the architecture. They test
sensitivity and computation cost; they cannot establish that an architecture
without the removed feature/edges would train better or worse.

## Correctness evidence and limitations

[FORMULATION.md](FORMULATION.md) defines the normalized base model and proves
startup/run equivalence, battery-state feasibility and incumbent preservation
under its stated assumptions. The independent oracle uses change-point runs
and sentinel gaps rather than the production validator's run scans and full
sliding windows. Exhaustive enumeration compares the oracle, validator,
sparse-row feasibility and optimal objective on tiny instances.

The local random audit checks 120 instances and 33,280 schedules. Supplemental
systematic cases exercise many feasible schedules, coupled resources and both
horizon conventions. Complete cases and numerical outcomes are preserved in
`results/exhaustive-audit.json` and `results/supplemental-audit.json`.
`results/reference-cut-counterexample.json` records a feasible schedule
excluded by the sixth reference cut; those unaudited cuts are not part of the
benchmark model.

Synthetic instances contain known feasible planted schedules. This avoids
solver-based rejection filtering but introduces a disclosed construction bias.
Bundled instances may share unknown generator families. Three paired seeds do
not separately identify training variance and search variance. The baseline
is SciPy/HiGHS; no tuned Gurobi, SCIP or published ONTS neural comparator has
been evaluated. Positive results cannot support claims beyond the tested
budgets and distributions.

## Files

- `oracle.py`: independent tiny-instance oracle and exhaustive comparison.
- `datasets.py`: witness generation, disjoint manifests and content hashes.
- `study.py`: frozen phases and durable, randomized-order benchmark execution.
- `protocol.py`: deadline scoring, interruption recovery and interventions.
- `statistics.py`: instance-level bootstrap, paired tests, Holm adjustment, AUC.
- `report.py`: tables, figures and claim status derived from recorded runs.
- `autoresearch.md` / `.jsonl`: hypotheses, decisions and recorded outcomes.

The scheduling test suite includes regression tests for split integrity,
statistics, interrupted records, deadline censoring, graph intervention
isolation and job-permutation equivariance:

```bash
.venv/bin/python -m pytest projects/scheduling/tests -q
```

Expanded [MIP/scheduling literature review](outputs/rl-mip-scheduling-review-2026-10-06.md) distinguishes RL, imitation, solver-internal control and improvement search. It records proposed experiments and primary-source provenance; these mechanisms are not claimed as evaluated additions.

[Executable ReSched/MIStar/RL-LNS/adaptive-LNS alternatives](RL_LITERATURE_IMPLEMENTATIONS.md)
provide a common feasible improvement engine, training driver and independent
artifact audit. Original-paper differences and current/best search semantics are
explicit; these are ONTS adaptations, not published benchmark reproductions.

Longer MIStar/DR-ALNS weight continuation, PPO/replay controls and fresh held-out
evaluation are documented in [RL_CONTINUATION.md](RL_CONTINUATION.md).

Post-continuation repair headroom and inference mechanism diagnostics:
[RL_SEARCH_DIAGNOSTIC.md](RL_SEARCH_DIAGNOSTIC.md).

## DP/MILP operator attribution

[OPERATOR_ATTRIBUTION.md](OPERATOR_ATTRIBUTION.md) freezes a five-pipeline
comparison on 40 fresh bundled-like cases and two paired seeds. Every pipeline
shares a saved initializer, charges its actual runtime to the 30-second allowance,
and retains failed starts. DP-only and MILP-only variants use explicit operator
switches; local search restarts from the common schedule. This separates repair
quality from construction without claiming transfer to new mission populations.

Run from the repository root in the activated virtual environment:

```bash
python -m projects.scheduling.research.operator_attribution \
  --output projects/scheduling/research/results/operator-attribution-fresh-20261007
python projects/scheduling/paper/scripts/audit_operator_attribution.py \
  projects/scheduling/research/results/operator-attribution-fresh-20261007
python projects/scheduling/paper/scripts/operator_attribution_tables.py
```

The output retains the frozen plan, source snapshots, input hashes, shared starts,
complete histories, analysis and independent audit. Resume requires identical
source files, benchmark inputs and settings. Separate smoke outputs are excluded
from inference. `paper/scripts/evidence_sensitivity.py` reproduces the historical
all-instance sensitivity; `paper/scripts/failure_classification.py` performs
post-hoc infeasibility checks without changing either scored population.

## Focused DP-only and repair-mask follow-up

[REPAIR_FOLLOWUP.md](REPAIR_FOLLOWUP.md) fixes a separate 80-instance,
two-seed comparison of reference, DP-only improvement and the existing
time-window repair masks. Its endpoint family includes two superiority tests
and a DP-only noninferiority test with a 0.5 percentage-point margin on mean
normalized score. Initialization, deadline credit and failed-start treatment
match the operator-attribution experiment. The earlier and new populations
are analyzed separately.

```bash
python -m projects.scheduling.research.repair_followup \
  --output projects/scheduling/research/results/repair-followup-fresh-20261007
python projects/scheduling/paper/scripts/audit_operator_attribution.py \
  projects/scheduling/research/results/repair-followup-fresh-20261007
python projects/scheduling/paper/scripts/repair_followup_tables.py
```
