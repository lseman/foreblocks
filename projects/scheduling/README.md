# ONTS neural-guided scheduling

[New heuristics](HEURISTICS.md) add whole-job DP profile moves, weighted local
search, adaptive local branching and predict-and-search, with development results.

The [semantic neighborhood ranker](RANKER.md) adds outcome-based candidate
ranking, typed sparse attention, complete-run moves and search history. It
preserves the original imitation selector as a baseline and includes uniform
and adaptive operator controls. See its fresh pilot report for measured results.

Solve the Offline Nanosatellite Task Scheduling problem derived from
`original.py`: maximize priority-weighted execution under startup, window,
run-length, periodicity, power and battery constraints.

The hybrid pipeline uses a coefficient-aware bipartite GNN to select job/time
neighborhoods and SciPy/HiGHS to repair them. Every returned schedule is checked
independently. Random and heuristic neighborhood selectors and a solver-only
baseline are included. This implementation does not establish a performance or
novelty claim; see [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the paper protocol.

All commands below run from the **repository root**. Dependencies are PyTorch,
NumPy, SciPy >= 1.10 and pytest for tests. Matplotlib is needed only for the
existing PPO evaluation plots.

## Solve and benchmark

Run immediately without training or a commercial solver:

```bash
.venv/bin/python -m projects.scheduling.examples.run_hybrid \
  --instance projects/scheduling/instances/examples/125_9.json \
  --methods exact random heuristic \
  --time-limit 30 --repair-time 2 \
  --output runs/onts/baselines.json
```

`--instance-dir` evaluates all JSON instances in a directory. `--seeds 0 1 2`
runs repeated searches. Each method gets its own wall-clock budget, including
model construction, inference, solving and validation. Native solver calls and
individual construction/inference operations can overrun the requested budget;
actual elapsed runtime is reported. Loading JSON/checkpoints happens before the
per-run timer. Time to first feasible is when the first validated incumbent is
available to the caller, rather than an internal solver discovery timestamp.

Output JSON contains complete schedules, objective, feasibility, status,
global upper bound and relative gap when available, runtime, time to first
feasible, and objective history. `unknown` means no feasible solution was found
within the budget; it is distinct from proven `infeasible`. A restricted repair's
bound is never reported as a global bound. Short budgets can fail to initialize
an incumbent; increase the budget or provide one through the Python API.

## Train and evaluate learned neighborhoods

Generate separate train and test directories with different seeds:

```bash
.venv/bin/python projects/scheduling/instances/generate_onts_instances.py \
  --count 100 --jobs 9 --horizon 97 --seed 0 --out instances/onts-train
.venv/bin/python projects/scheduling/instances/generate_onts_instances.py \
  --count 20 --jobs 9 --horizon 97 --seed 100 --out instances/onts-test

.venv/bin/python -m projects.scheduling.examples.train_neighborhood \
  --instance-dir instances/onts-train \
  --rounds 8 --epochs 10 --init-time 1 --repair-time 2 \
  --output runs/onts/neighborhood.pt

.venv/bin/python -m projects.scheduling.examples.run_hybrid \
  --instance-dir instances/onts-test \
  --methods exact random heuristic neural \
  --checkpoint runs/onts/neighborhood.pt \
  --time-limit 30 --repair-time 2 --seeds 0 1 2 \
  --output runs/onts/comparison.json
```

Initialization repairs an all-off proposal to produce a low-activity feasible
schedule. Random neighborhoods are solved exactly under a per-call budget.
Strictly improving repairs supply labels identifying the execution variables
that changed. The GNN learns release logits with weighted binary cross-entropy.
This is solver imitation, not an implementation of a named published training
algorithm. The trainer records label counts, losses, train instance paths,
model parameters and total label/training cost. If no improving repairs are
found it exits with a diagnostic and does not save an untrained checkpoint.

The graph contains every execution/startup/SoC variable and every actual base
MILP constraint. Edges carry normalized signed coefficients; node features
include incumbent activity/slack, objective coefficients, bounds, variable
roles, solar availability, future average solar and battery slack. Search uses
ranked contiguous job/time blocks, with stochastic exploration. Only execution
variables outside the neighborhood are fixed; startups and battery states stay
free. Validated improvements replace the incumbent.

`--fraction`, `--block-size`, `--init-fraction` and `--repair-time` control search.
Policy training and evaluation accept `--config path.json` with `ONTSConfig`
fields. Evaluation rejects a checkpoint trained with different model parameters.
Use held-out distributions and genuinely energy-constrained instances for paper
experiments; the default generator's small task draws often make instantaneous
power constraints inactive.

## Existing PPO integration

The existing PPO entrypoint can now repair its neural proposal after training:

```bash
.venv/bin/python projects/scheduling/examples/train_onts.py \
  --instance projects/scheduling/instances/examples/125_9.json \
  --encoder transformer --hybrid-time-limit 30 --outdir runs/onts/ppo
```

Raw policy evaluation reports independently checked violations. The hybrid
result is saved to `hybrid.json`. Both transformer and bipartite PPO encoders
can provide proposals. A proposal that violates constraints is repaired by
minimizing Hamming distance before objective-improving search.

Python API:

```python
from projects.scheduling.environments.onts_env import load_onts_instance
from projects.scheduling.hybrid import rollout_proposal, solve_hybrid

instance = load_onts_instance("projects/scheduling/instances/examples/125_9.json")
result = solve_hybrid(instance, time_limit=30, selector="heuristic")
# With an existing trained NCO model:
# result = solve_hybrid(instance, proposal_fn=rollout_proposal(model), time_limit=30)
print(result.status, result.objective, result.relative_gap)
```

## Model conventions

JSON solar values are used directly as watts; the lossy `int(value) * 10`
conversion from `original.py` is not applied. Surplus solar can be curtailed.
Maximum periodicity requires a startup in every full sliding window. Minimum
startup separation applies even when its period exceeds the horizon. Execution
windows may extend beyond the horizon; only existing slots are evaluated.
Final minimum runs may be truncated at the horizon by default, as in the
reference. `--strict-final-run` requires complete runs instead. Startup indicators
are derived exactly from off-to-on transitions.

Unaudited reference valid inequalities are excluded. The instance generator
uses the shared base model by default; its explicit `--valid-inequalities`
option retains the legacy cuts and battery-equality model for reproduction.
Instances generated in that legacy mode are labeled in their metadata.
The independent [validator](validation.py) checks base constraints, not those
legacy cuts. Invalid schedules are never accepted on reward alone.

## Files and tests

- `milp.py`: shared sparse formulation, unrestricted solve, Hamming repair and
  neighborhood repair; no Gurobi dependency.
- `neighborhood.py`: full MILP graph, GNN and checkpoint serialization.
- `hybrid.py`: anytime search, baseline selectors and NCO proposal adapter.
- `learning.py`: solver-imitation label collection and training.
- `validation.py`: independent complete-schedule validation.
- `examples/`: PPO training, neighborhood training and benchmark commands.

```bash
.venv/bin/python -m pytest projects/scheduling/tests -q
```

Tests include exhaustive small-instance feasibility checks, coupled energy
optimization, horizon boundaries, fixed-cell repair, incumbent preservation on
timeouts, global versus restricted bounds, GNN gradients, training, checkpoints
and the existing neural decoder interfaces.

## Reproducible empirical study

The [research protocol](research/README.md) provides exhaustive oracle checks,
disjoint train/validation/test/OOD splits, matched-budget comparisons,
common-incumbent diagnostics, inference interventions, paired statistical
analysis and regenerable figures. The [formulation](research/FORMULATION.md)
states the correctness arguments and model assumptions.

```bash
bash projects/scheduling/research/autoresearch.sh
```

Completed evidence is written to `research/results/local-study/REPORT.md`.
The [completed results review](research/results/RESULTS_REVIEW.md) combines
107,008 exhaustive schedule checks, 1,314 benchmark runs and independent replay
of every returned schedule. The local study does not establish learned-selector
superiority over random or heuristic LNS.
The protocol preserves negative comparisons and timeouts; it does not assume
that learned neighborhoods improve over the non-neural controls.
