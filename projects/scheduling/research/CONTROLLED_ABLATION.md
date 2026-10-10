# Controlled portfolio component study

This study measures component effects on the 40 bundled-like instances previously
used during development and confirmation. It is exploratory: these instances have
already been inspected. The executable plan freezes source hashes, copies of the
sources, instance hashes, seed pairs, budgets, worker count and run order before
execution. Resume requires an identical plan. Existing ablation rows cannot be
imported into this format.

## Paired tracks

**Improvement:** once per instance/seed pair, run the full initialization policy
with a dedicated 9-second allowance (30% of the default 30-second improvement
budget). Its local search may use the entire initialization allowance and uses
the early-stop rule. Validate and save the returned schedule. Give this same
schedule to three independently seeded 30-second improvement runs:

- Full portfolio: stagnation controller, patience 10, rule-based task selection,
  20% repair masks, 1-second repair cap, surplus weight 0.05.
- No repairs: identical settings, with `enable_repairs=False`. No MILP repair
  calls occur; initialization is shared rather than disabled.
- No surplus: identical settings, with surplus weight zero.

Initialization is outside the improvement budget. All three runs share the saved
schedule and search seed. If initialization fails, retain the paired records as
failures and count them. Report all-instance scores and conditional scores
separately. Results concern improvement from this construction policy, not
end-to-end performance at a 30-second budget.

**Cold start:** run the full policy and a variant with only
`polish_after_feasible=None`. Both keep surplus weight 0.05 and MILP repairs.
The 30-second budget includes model construction and initialization. A third
cold-start run solves the full MILP with HiGHS; it is a descriptive comparator,
not an ablation. There is no 0.5-second deadline tolerance.

All methods have seeds 101 and 211. Each worker uses one solver thread and one
PyTorch thread. Instance/seed pairs and method order are deterministically
shuffled. Pair workers run in fresh processes to isolate solver state. Timed
solver calls use the HiGHS engine bundled with SciPy. Its callback API is private
and isolated in `observed_highs.py`; unavailable APIs fail rather than silently
substituting inaccurate timestamps.

## Observations and endpoint credit

Local search emits each validated incumbent while initialization is still running.
The portfolio records its first such observation and separately records
`initialization_complete_at`. Exact initialization and the exact comparator use
HiGHS improving-solution callbacks. Each callback solution is checked by the ONTS
validator before its timestamp and objective enter the history. If HiGHS returns
a solution without an earlier callback, label that observation
`solver_return_validation`; do not present it as an internal discovery time.
A supplied schedule is available at time zero in the improvement track.

The recorded timestamp is a wall-clock observation after validation, not the
solver's internal discovery instant. Deadline credit includes only validated
history events with time at most the budget. Failed runs remain at score zero.
Returned schedules are also checked by the independent oracle. Persist schedules,
histories, timing sources and shared-start hashes for review.

## Analysis

Normalize objectives by the best observed objective per instance and track.
An instance with no feasible schedule contributes zero for every method; it is
counted and retained in the primary all-instance analysis. Also report the
conditional mean on instances solved by some run. Normalization is not an
optimality certificate. Integrate the observed incumbent curve over the budget
for normalized AUC. First-feasible medians use successful runs; also report the
mean capped time, assigning the budget to failures.

Average paired seeds within each instance before inference. The three primary
comparisons are full minus no repairs, full minus no surplus, and cold full minus
no early stop. Use two-sided paired t tests, Holm adjustment over these three
comparisons, and simultaneous Bonferroni 95% confidence intervals for score
differences. AUC and capped-time inference are secondary families, explicitly
labeled. Do not infer equivalence from a nonsignificant difference or attribute
initialization speed to first-feasible discovery time. Error counts accompany
results; execution errors prevent treating the run as a complete component
comparison.

## Commands

```sh
python -m projects.scheduling.research.portfolio_ablation \
  --output projects/scheduling/research/results/controlled-components-20261007 \
  --workers 12
```

The defaults are 40 instances, two seeds and a 30-second budget. `--plan-only`
freezes without executing. `--count`, `--seeds` and `--budget` support integration
checks in separate output directories; those results are not the paper study.
