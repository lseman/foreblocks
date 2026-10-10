# New ONTS heuristics

Four additions motivated by the [literature search](research/outputs/new-heuristics-search-2026-10-05.md).
Every method returns only schedules accepted by `validate_schedule`, keeps the
incumbent when a move fails, and reports no global bound from restricted solves.

| Method | Module | Idea | Solver calls |
|---|---|---|---|
| `profile` | `job_profile.py` | Remove 1–3 whole jobs; re-insert each by an exact per-job DP over activation profiles (startups, windows, min/max period, run length), with energy priced per slot; replay the battery and raise prices on deficit windows | None (optional MILP fallback: `profile_fallback`) |
| `local_search` | `local_search.py` | Feasibility Jump–style weighted violation search on run moves; after a feasible solution, an objective target acts as a weighted constraint (Local-MIP breakthrough analogue) | None |
| `adaptive_local_branch` | `classical.py` | Local branching whose Hamming radius grows when a ball is proven exhausted and shrinks after a timed-out repair without gain (Fischetti–Lodi rule) | One per step |
| `classical_portfolio_plus` | `classical.py` | Round-robin over LP disagreement, energy exchange, run relocation, adaptive local branching, profile and local search | Mixed |
| `profile_portfolio` | `profile_portfolio.py` | Cheap-first: profile moves until 10 consecutive failures, then one uniform-candidate MILP repair; local-search initialization when no schedule is given | One per stagnation episode |
| `predict_search[_polish]` | `trust_region.py` | ConPaS-style predict-and-search: GNN marginals from instance-only features, contrastively trained; MILP restricted by `sum_{S1}(1-x)+sum_{S0}x <= delta`; `_polish` hands leftover time to `profile` | One per region |

The DP is exact for one job's own constraints; this was checked against brute-force
enumeration through the validator (see `tests/test_new_heuristics.py`). The battery
is coupled only through prices and validation, so a profile move is a heuristic.
`prices="zero"` is the default. LP-dual prices (`prices="lp"`) were worse in
development runs; they appear to over-price energy.

## Run

```bash
# Predictor for predict-and-search (training-only instances)
.venv/bin/python -m projects.scheduling.examples.train_predictor \
  --instance-dir path/to/train-json --output runs/onts/predictor.pt

.venv/bin/python -m projects.scheduling.examples.run_hybrid \
  --instance-dir path/to/heldout-json \
  --methods heuristic uniform profile local_search adaptive_local_branch \
            classical_portfolio_plus predict_search_polish \
  --predictor-checkpoint runs/onts/predictor.pt \
  --time-limit 3 --repair-time 0.3 --fraction 0.2 --output runs/onts/new.json

# Development comparison used below
.venv/bin/python -m projects.scheduling.research.heuristics_dev --seeds 0 1 2
```

## Development results (exploratory, not confirmatory)

Instances are 15 fresh planted instances: 9 IID and 6 larger OOD, with seeds
from 31,000,000 and hashes checked against all earlier study manifests. Each
instance was run with 3 search seeds, a 3 s budget, a 0.3 s repair cap and a
common witness start. Predict-and-search starts cold, by design. The table shows
mean relative change versus `heuristic`; full table in
[SUMMARY.md](research/results/heuristics-dev/SUMMARY.md).

| Method | IID | OOD |
|---|---:|---:|
| uniform (existing) | −0.29% | +7.33% |
| **profile** | −0.43% (4W/4L) | **+7.87%** (4W/2L) |
| classical_portfolio_plus | −1.77% | +6.18% |
| local_search (no MILP) | −4.99% | +1.40% |
| predict_search_polish | −0.50% | −10.73% |
| adaptive_local_branch | −25.3% | −55.4% |
| local_branch (fixed radius) | −55.1% | −78.3% |

Reading:

- **Profile moves** match MILP-repair LNS on IID instances and are the best
  method on OOD instances, without calling a solver. This is the strongest
  candidate for the paper's algorithmic contribution. Adding an exact MILP
  fallback (`profile_fallback`) hurts, because it spends the short budget on repairs.
- **Local search** is clearly weaker as an improver. Its use is a solver-free
  cold start: it finds a first feasible schedule in about 5–30 ms. A 3 s exact
  HiGHS solve was beaten on large instances during smoke tests.
- **Local branching** is limited by its cost per step: each step solves the
  full model inside a Hamming ball, giving about 11 steps in 3 s from the weak
  planted start. The adaptive radius more than halves the loss of fixed-radius
  LB, but both remain far behind. Expect LB to be competitive only with longer budgets.
- **Predict-and-search** was trained on only 12 small instances (36 s of
  labeling). It is weak cold and fails to transfer to larger instances.
  It is the SatGNN-family baseline; more training data is needed before
  drawing conclusions about it.

None of this supports a superiority claim. Fifteen instances were used for
development and selection; claims require a new frozen protocol on fresh instances
(see `research/CONFIRMATORY.md` for the template). Obvious next steps,
selected on development data only:

- a portfolio of `profile` and uniform MILP repair, dropping LB at short budgets;
- ranker or bandit selection of *which jobs* to re-profile;
- local-search cold start in place of exact initialization when no witness exists.

## Profile portfolio: selection and confirmation

**Selection.** Settings were chosen on the development instances only
([profile-selection](research/results/profile-selection/selection.json)). The
criterion, fixed in advance, was the mean of the IID and OOD changes versus
`heuristic`. Results:

- The `stagnation` controller beat `ucb`.
- Patience 10 beat 30 and 100. This is at the edge of the grid.
- Rule-based job selection beat adaptive (per-job UCB) selection slightly in
  every pairing. Online job learning did not help.

**Confirmatory study** ([protocol](research/PROFILE_CONFIRMATORY.md),
[results review](research/results/profile-confirmatory/RESULTS_REVIEW.md)).
Setup: 720 runs on 60 fresh instances, three paired seeds, 3 s budgets, witness
start. All outputs were independently valid, with no overruns.

| Comparator | Mean | Simultaneous 95% CI | Holm p | Verdict |
|---|---:|---|---:|---|
| uniform | +0.86% | [+0.52, +1.20] | 6e-7 | superiority established; >1% practical criterion fails |
| heuristic | +4.01% | [−0.32, +8.33] | 0.037 | not established (CI crosses 0); +7.85% OOD, +0.17% IID |

## Bundled instances, cold start

`examples/bundled_coldstart.py`: all 42 bundled instances, 30 s, one thread, no
witness ([summary](research/results/bundled-coldstart/SUMMARY.md)). This is
descriptive: a fixed benchmark, six instances of which were inspected during development.

| Method | Feasible by 30 s | Mean gap to best found | Best found |
|---|---:|---:|---:|
| exact (HiGHS) | 19/42 | 59.9% | 3 |
| heuristic | 18/42 | 59.4% | 5 |
| uniform | 18/42 | 58.6% | 2 |
| local_search | 39/42 | 12.4% | 7 |
| **profile_portfolio** | **38/42** | **9.6%** | **31** |

On the 18 instances where both the portfolio and HiGHS produced a schedule, the
portfolio's median gap to HiGHS's proven upper bound is 7.0% (max 29.7%).

- **Initialization is the bottleneck on these benchmarks.** Single-thread
  HiGHS starts fail on most instances with 18 or more jobs. Weighted local search
  finds a first schedule in a median of 0.35 s.
- **Reported time to first feasible is late.** The portfolio reports about 9 s
  because its local-search phase runs its full initialization share (30% of the
  budget) before handing over. Stopping that phase soon after the first
  feasible schedule is a clear next improvement. It was not changed here, to keep
  the confirmed algorithm fixed.
- **Missing comparators.** These are HiGHS results. Published ONTS methods
  (branch-cut-and-price, SatGNN with Gurobi/SCIP) must be compared on the same
  instances before claiming anything relative to the literature.
