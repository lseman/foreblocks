# Frozen constellation replication protocol — 8 October 2026

Constellation scheduling is the paper's central application. The historical
36-instance comparison remains descriptive. This new fixed-size study replicates
each design cell and observes validated incumbents consistently. The protocol is
frozen before execution; completion requires all records and independent audit.

Use the existing 36 factorial cells: horizons 96/144/192, satellites 5/10/15,
private tasks per satellite 8/12 and constellation tasks 4/8. Generate three
independent instances in each cell (108 instances), with new fixed generator
seeds 73,000,000–73,000,107 and archived serialized instances/hashes. Keep the generator's per-task
feasibility acceptance test and report redraw counts; do not filter by the
full method's construction success. Solar profiles still originate from the
bundled pool, so replication is within this generator, not new mission transfer.

Evaluate the frozen reference portfolio with two paired search seeds per instance
1301 and 1409, and full HiGHS once per instance, each with a 60-second total allowance including
model construction and production validation. The current exact constellation
solver does not apply its `seed` argument to HiGHS: identical deterministic
exact runs must not be counted as independent search-seed replications.
There are 324 planned solves (216 portfolio, 108 exact), approximately 27 minutes
of allowance time with twelve workers, before generation and process overhead.
The Ryzen 9950X has 16 physical cores. Each worker uses one solver, BLAS and
tensor thread. Resource contention remains a limitation of wall-clock results.
Freeze protocol, sources, input files, order, software environment, thread
settings and instance identities before execution. Do not change the method
based on intermediate results.

The historical runner uses a 0.5-second tolerance and return-only exact logging.
Do not reuse it unchanged for this comparison. Use common strict
deadline credit and save the schedule associated with each credited observation.
Observe HiGHS improving-solution callbacks and portfolio accepted improvements,
with validation and a schedule snapshot before each timestamp. Include model
translation in the exact solver's remaining allowance. Credit observations at
or before 60 seconds, even if a solve subsequently returns late; no tolerance.
Archive both the best credited and final returned schedules in compressed files.
If callbacks fail, mark an execution error; do not silently change observation rules.
Late-return frequency and solver bound availability must be reported separately.

Primary outcomes: difference in feasible-run probability and difference in
all-instance normalized endpoint score. Average portfolio seeds within each
instance; pair each with that instance's exact result. Normalize using a fixed
per-instance pool consisting of both portfolio seeds and exact solving, assigning
zero when no method is credited. Retain all instances and disclose unsolved
counts. Treat the two outcomes as one multiplicity family, with instance-level
paired inference and an explicitly prespecified treatment of the factorial cells.
Use equal weights for the 36 design cells; report cell summaries descriptively.
Bootstrap sensitivity must resample independent instances within cells, never
individual search seeds as independent units. Predeclare the analysis and a
precision justification before the run; three draws per cell provide replication
but limited precision for cell-specific effects and interactions.

Independently replay every credited and returned schedule, check cross-satellite
startup/periodicity/exclusivity constraints and battery semantics, and verify
source/input hashes and statistical reproduction. Small exhaustive instances
should separately quantify the idle-separated-neighborhood restriction.

Even a successful study compares this implementation with full HiGHS under the
curtailing model. It does not establish superiority over published branch-and-price
methods or the original non-curtailing constellation model. Those claims require
additional matched-semantic comparators.

Run from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ONTS_SOLVER_THREADS=1 \
python3 -m projects.scheduling.research.constellation_replication \
  --output projects/scheduling/research/results/constellation-replication-20261008
```

Smoke runs use `--smoke`, base 73,100,000, two draws in one small cell, a shorter
allowance and a separate output. They are excluded from the primary study.
For inference, define one paired difference per independently generated instance:
the mean across its two portfolio seeds minus its single exact result. For each
metric, estimate the equal-weighted mean of the 36 cell means. Its variance is
the sum of within-cell sample variances divided by the number of draws per cell
and by the squared number of cells. Use a Satterthwaite degrees-of-freedom
approximation for two-sided t tests and Bonferroni simultaneous 95% intervals.
Holm-adjust the two primary p values as one family; superiority requires both
adjusted p < 0.05 and a positive simultaneous lower bound. Degenerate zero-variance
data receive zero-width intervals and p=1 for a zero effect or p=0 otherwise;
report that degenerate case explicitly. This is inference over generator draws
within the fixed design, not a random sample of mission types or factorial cells.

Sample size is fixed at 108 before execution. For planning only, a paired
standard deviation of 40 percentage points gives an approximately 9-point
normal-approximation simultaneous half-width at n=108; a 100-point deviation
gives approximately 22 points. These are approximate precision scenarios, not
guaranteed widths, achieved power or a justification based on the new results.
Cell-specific effects have only three draws and remain descriptive. No interim
testing, optional stopping, sample-size extension, tuning or pooling with the
historical study. Randomize the 324 run order with seed 981109.

Normalized incumbent AUC, first observation times, dual-bound gaps, late returns,
design-factor summaries and 10,000-resample within-cell bootstrap sensitivity
(seed 981009) are descriptive. Keep missing bounds distinct from zero gaps, and
never interpret best-observed normalized scores as optimality certificates.
