# Confirming portfolio v2 on bundled-like instances

Written before any test run. Settings were selected on development instances only
([selection](results/portfolio-v2-selection/selection.json)).

## Frozen algorithm

v2 = `solve_profile_portfolio(controller="stagnation", patience=10,
job_selection="rules", fraction=0.2, polish_after_feasible=0.5,
surplus_weight=0.05)`, with local-search initialization (no witness). Source hashes
and a source archive are stored with the plan.

## Protocol

- **Instances:** 40 fresh bundled-like instances, `realistic_instance(61000000 + i)`,
  i = 0..39 ([generator](realistic.py)). There is no feasibility filter.
- **Methods:**
  - `v2`;
  - `v1` (the earlier confirmed configuration, vectorized DP, otherwise unchanged);
  - `heuristic` (hybrid, exact initialization);
  - `uniform` (ranked-uniform);
  - `exact` (unrestricted HiGHS).
- **Runs:** 30 s budget, 1.0 s repair cap, one thread per run, seeds 101 and 211.
  This gives 400 runs, executed in a seeded random order on 12 worker processes.
  Outputs are checked by the independent oracle.
- **Score:** objective divided by the best objective found by any run on that
  instance. Infeasible-by-deadline runs score 0. Instances where no run finds a
  feasible schedule are excluded symmetrically, and their count is reported.
  Seeds are averaged per instance.

## Hypotheses and decision

The primary family has three comparisons: `v2` versus `heuristic`, `uniform` and
`exact`. Each uses the paired difference in normalized score, in percentage points,
with two-sided t-tests, Holm adjustment and Bonferroni simultaneous 95% CIs.
Superiority for a comparison requires Holm p < 0.05 and a lower limit > 0.

The following are descriptive only:
- `v2` versus `v1`;
- feasible-by-deadline counts;
- time to first feasible.

The study has a fixed size, with no interim checks. Scope: bundled-like
resampled instances at 30 s on one CPU thread. No claim is made relative to
published ONTS methods.
