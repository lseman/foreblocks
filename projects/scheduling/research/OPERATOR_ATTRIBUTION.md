# Frozen operator attribution study

Question: does whole-task DP repair improve schedules beyond MILP-only repair
and local search from the same constructed incumbent within a 30-second total
allowance? This study also checks the previously fixed time-window repair against
the reference portfolio. No new parameters are selected from these results.

Population: 40 fresh bundled-like instances, generator seeds 65,000,000 through
65,000,039; paired search seeds 601 and 701. Instances must be hash-disjoint from
all existing result plans/manifests before execution. This is replication within
the existing generator, not transfer to independently sourced missions.

For each instance/seed pair, construct one common start with the reference
initializer (9-second maximum, early stopping 0.5 seconds, no improvement moves).
Save the actual initialization runtime, validated observation history and final
schedule. A start is usable only if returned within the initialization allowance
and independently feasible. If construction fails, retain all five paired failures
at zero; no method receives a private salvage initialization. Thus these are
shared-initialization pipelines, not unrestricted cold-start policies. A usable
start is supplied to every variant. Each gets 30 seconds minus actual initializer
runtime, including construction of its improvement model and validation. Local
search is restarted from that same schedule; tabu lists and violation weights
are reset. It is not a continuation of the initializer's internal search state.

Variants (all fixed before execution):

- Reference: patience 10, rule-based task selection, maximum three tasks, uniform
  candidate MILP masks, fraction 0.2, repair cap 1 second, surplus weight 0.05.
- DP only: reference with MILP repairs disabled by a real switch.
- MILP only: reference with DP moves disabled by a real switch. Same candidate
  mask generator, fraction, cap and uniform selection as reference repairs.
- Restarted local search: weighted constraint search from the common start for
  the entire remaining allowance; no DP or MILP improvement calls.
- Time-window: reference with the previously fixed contiguous interval mask.

Four primary comparisons: reference minus MILP only; reference minus restarted
local search; reference minus DP only; time-window minus reference. Endpoint
scores use objective/best observed credited objective on that instance across
all methods/seeds. Initialization observations are shifted into each pipeline's
timeline. Validated observations at or before 30 seconds count, without tolerance.
Failed starts and all-unsolved instances remain in the all-40 mean. Also report
conditional means, raw objective improvements from shared starts and feasibility.

Instances are inference units; average paired seeds within instances. Two-sided
paired t tests with Holm p correction and simultaneous Bonferroni 95% intervals
cover the four primary endpoint contrasts. Superiority requires adjusted p < 0.05
and a positive simultaneous lower bound. Normalized observed AUC uses a separate
secondary family. Bootstrap intervals and sign tests are descriptive robustness
checks and do not replace the primary decision rule. One percentage point is a
prespecified practical reference: report whether the lower endpoint interval
exceeds it, separately from statistical superiority.

Run eight workers, one solver and one Torch/BLAS thread per worker, deterministic
random pair and method order. Workers are fresh for each pair. Record sources,
protocol, environment, bundled input hashes, instance hashes, shared-start hashes,
all final returned schedules, per-method histories and operator calls/gains/time.
Diagnostics are descriptive because search paths differ. Record every execution
error; any error blocks presenting the study as a completed primary comparison.
Do not stop early, tune settings, discard inconvenient cases or replace a failed
run silently. Resume requires identical sources, protocol, inputs and plan.

Independent audit checks all schedules, complete paired records, fingerprints,
budget credit, operator exclusion, source/input/protocol hashes, normalized scores,
AUC and simultaneous inference. Optimization timers cover production validation;
the independent offline oracle check is outside timed search, equally for all arms.

The study does not compare against published branch-and-price or establish
constellation effects. Parallel-worker wall times are specific to this setup.
Smoke runs use separate generator seeds and output directories, and are excluded
from the paper study.
