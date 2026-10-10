# Focused DP-only and repair-mask follow-up

This is a new fixed-size measurement, motivated by the completed 40-case
operator-attribution study. That earlier study is used for planning only;
its runs are not pooled into the primary analysis. No new algorithm settings
are selected in this follow-up.

Population: 80 bundled-like instances with generator seeds 67,000,000 through
67,000,079, paired search seeds 809 and 907, three pipelines, 480 records. Require
hash-disjointness from all earlier result manifests before execution. This is
replication within the empirical generator, not a new mission population.

Pipelines: the frozen reference portfolio; DP only, with MILP improvement
repairs genuinely disabled; time-window repair, changing only the reference's
repair-mask strategy to the previously fixed contiguous-window rule. Patience
10, rule-based task selection, at most three tasks per profile move, repair
fraction 0.2, repair cap 1 second and surplus weight 0.05 are unchanged.

Each instance/seed pair uses one common initializer with a 9-second maximum
allowance and the existing early-stop rule. Save its schedule and observations.
Only a validated schedule returned within that allowance is usable. Charge
actual initialization runtime to each 30-second total allowance; the remaining
time includes construction of the improvement model and production validation.
Failed common starts produce paired zeros, with no private salvage. DP only
means no MILP improvement calls; the common initializer can use its existing
MILP fallback. The three pipelines must receive identical starts and allowances.

Credit only validated observations at or before 30 seconds, without tolerance.
Score is objective divided by the best credited objective per instance across
the three pipelines and both seeds. All 80 cases, including all-unsolved cases,
remain in primary means. Report feasibility, conditional means, per-instance
differences, raw improvements and the normalized observed incumbent AUC.

Primary family of three hypotheses:

1. Reference minus DP only: two-sided superiority comparison. This measures
   the incremental value of reference MILP repairs.
2. Time-window minus reference: two-sided superiority comparison. This measures
   replacing the repair mask, holding all other settings fixed.
3. DP only is noninferior to reference for mean normalized endpoint score,
   using a predeclared margin of 0.5 percentage points: the null is that
   DP-only minus reference is at most -0.5 points. This margin is half the
   previous study's one-point practical reference and expresses an average
   quality tradeoff for removing MILP improvement calls. It does not guarantee
   a loss below 0.5 points on every instance.

Use instances as the inference units, averaging paired seeds within each case.
Holm-adjust the two two-sided p values and the noninferiority one-sided p value
as one three-hypothesis family. Use simultaneous Bonferroni two-sided 95%
intervals over that family; these are conservative for the directional test.
Superiority requires adjusted p < 0.05 and a positive simultaneous lower bound.
Noninferiority requires adjusted p < 0.05 and the simultaneous lower bound for
DP-only minus reference above -0.5. An entirely unsolved population is labeled
uninformative rather than supporting a simplification decision. Nonsignificance
alone does not establish noninferiority or equivalence.

AUC has a separate secondary family of the two superiority contrasts. Bootstrap
intervals and sign tests are descriptive. Operator times, improvement rates,
deadline overruns and quality-versus-time checkpoints at 1, 3, 10 and 30 seconds
are descriptive mechanism measurements. No claim of faster search is inferred
from avoiding solver calls under a fixed wall-clock allowance.

Sample-size choice: the prior instance standard deviations were about 0.535
points for reference minus DP only and 0.401 for time-window minus reference.
At 80 cases the planned simultaneous half-widths are approximately 0.146 and
0.110 points, assuming similar dispersion. This is a precision target, not a
promise of significance or a retrospective power claim. The sample size is
fixed before execution, with no interim testing, extension, optional stopping,
parameter changes or pooling to obtain significance.

Run eight fresh pair workers, with one solver, Torch and BLAS thread each,
and a frozen deterministic randomized pair/method order. Archive sources,
protocol, environment, instance and benchmark input hashes before execution.
Preserve all failures and execution errors. Any execution/validation error
blocks treating the comparison as a completed primary study. Resume requires
the same sources, protocol, inputs and settings. Smoke runs use different
generator seeds and separate outputs and are excluded from inference.

Independently audit all returned schedules, common starts, operator exclusion,
strict deadline credit, source/protocol/input hashes, endpoint/AUC inference and
the margin-based one-sided test. Optimization timers include production
validation; independent offline oracle checks are outside timed search for
every pipeline. Do not promote defaults based only on exploratory call counts.
