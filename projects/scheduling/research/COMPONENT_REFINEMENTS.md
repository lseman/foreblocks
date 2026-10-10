# Repair and surplus refinements

This study follows the controlled component audit. It evaluates two fixed changes:
contiguous time-window MILP masks (cycling battery pressure, actual spill, and uniform window selection) and a
finite-curtailment surplus discount. Existing defaults remain unchanged.

The finite discount uses watts actually curtailed in the partial schedule, rather
than treating any saturated slot as enough free energy for an entire job draw.
It is a local insertion surrogate: partial consumption can change later battery
levels. Full-schedule validation remains the acceptance criterion.

Six common-start variants: legacy full, no repairs, reserve-only (zero discount with equal-objective reserve acceptance),
time-window repair only, finite-curtailment discount with equal-objective reserve acceptance, and both refinements.
All use identical saved feasible starts and paired random seeds; initialization
is outside the independent improvement budget. Other settings equal the previous
controlled study: patience 10, rules job selection, fraction .2, repair cap 1 s,
surplus weight .05, early stopping .5 s. Time-window fraction releases ceil(.2 T) consecutive slots across all jobs.
Window weights sum pressure or spill, plus a 5% exploration floor.

Primary comparisons: time-window minus legacy, surplus-and-reserve minus legacy,
combined minus legacy, time-window minus no repairs, surplus-and-reserve minus reserve-only. Instance means over paired seeds are the inference unit. Simultaneous
Bonferroni t intervals and Holm p values cover this five-comparison family.
AUC is secondary in its own family. Normalize by best observed per instance;
failures and all-unsolved instances remain zero. No optimum claim.

Pilot: base 64,000,000, 12 instances, seed 307, 10 s improvement budget.
Fresh evaluation: base 63,000,000, 40 instances, seeds 401 and 503,
30 s improvement budget. Both use a separately budgeted 9 s common initializer.
These generator seeds differ from the previously inspected component study.
The distribution is unchanged. This evaluates improvement from a common start,
not cold-start feasibility or cross-distribution generalization.

Freeze source and instance hashes before each phase, preserve source snapshots,
randomize method order per pair, enforce one solver and Torch thread per worker.
Record all moves, deadlines, raw objectives, validated final schedules, and shared
start fingerprints. Never delete failures. Independently revalidate returned
schedules. Report component call counts, gains and elapsed time descriptively;
they are path-dependent and do not by themselves demonstrate causal benefit.
No parameter selection from fresh evaluation; do not replace manuscript claims
without completing and auditing all prespecified comparisons.

A prior pilot (base 62,000,000) tested complete-job masks. Only 1/67 repairs
improved the incumbent and the final-score point estimate was -1.299 pp versus
legacy. That candidate was rejected before fresh evaluation; its sources and
records remain archived in component-refinements-pilot-20261007. No fresh results
were inspected to select the time-window candidate.

The time-window pilot (base 64,000,000; archived in window-pilot) had a
+0.816 pp score estimate versus legacy; its interval included zero. Finite
curtailment alone was nearly tied (+0.007 pp). A final development pilot adds
lexicographic acceptance: equal objective is accepted only with strictly larger
sum of battery charge. A reserve-only control holds acceptance fixed while
removing the discount. Fresh evaluation starts only after this candidate is fixed.
