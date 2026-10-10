# Selective MILP repair pilot

Fixed exploratory study, declared before execution on 8 October 2026. Motivation:
the prior 80-instance follow-up found a 0.371-point endpoint benefit from MILP
repairs, using 76.1% of logged operator time. Prior runs are not pooled here.

Use 24 fresh bundled-like instances (generator seeds 69,000,000–69,000,023),
paired search seeds 1009 and 1103, four pipelines and eight workers. Total budget
is 10 seconds, with a common initializer allowed at most 3 seconds and its actual
runtime deducted from every pipeline. Failed starts remain paired zeros. This
short-budget pilot tests the policies; it does not confirm the earlier 30-second
effect, establish transfer to other missions, or justify a production default.

Pipelines retain the reference's patience 10, rule-based selection, 20% repair
masks, 1-second repair cap and surplus weight 0.05:

- Reference: unchanged repair scheduling.
- DP only: no MILP improvement calls; the common initializer can use MILP.
- Delayed: repairs become eligible one-third of the improvement allowance after
  improvement starts, including improvement model construction in that clock.
- Budget aware: logged repair time (mask construction, solving and production
  validation) has a target quota of 25% of the improvement allowance. Solver caps
  are reduced to the remaining quota, and calls reserve two repair caps' worth of
  remaining search time. Noninterruptible work can overshoot the target quota;
  record this rather than claim a hard resource guarantee.

Both selective policies retain the stagnation trigger and use DP whenever a
repair is ineligible. No parameter tuning, intermediate analysis, sample-size
extension or default promotion during this study. Smoke runs use another seed
range and output directory. Freeze sources, protocol, inputs, instance hashes,
method order and environment before execution. Resume only identical plans.

Normalize by the best credited objective across all four methods and both seeds
per instance; retain all instances. Credit only validated observations at or
before the total deadline with no tolerance. Average paired seeds within each
instance. Compare delayed minus reference and budget-aware minus reference as
one endpoint family and a separate secondary AUC family. Use simultaneous
Bonferroni 95% intervals and Holm-adjusted two-sided tests, requiring both a
positive lower bound and adjusted p < 0.05 for superiority. These remain
exploratory pilot results. DP-only results, repair costs, calls, acceptance and
deadline returns are descriptive. No absence of significance proves equivalence.

Independently replay saved schedules and starts, verify source/input/protocol
hashes, deadlines and inference, and inspect actual call times against the two
policy gates. Errors block interpreting a complete study. A later 30-second
confirmation should freeze one selected rule on separate development evidence,
use fresh instances and its own fixed precision target; do not pool this pilot.

Run from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ONTS_SOLVER_THREADS=1 \
python3 -m projects.scheduling.research.selective_repairs \
  --output projects/scheduling/research/results/selective-repairs-pilot-20261008
```
