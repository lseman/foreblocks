# Confirming the profile portfolio

Written before any test-set run. The algorithm and its settings were selected on
development instances only ([heuristics-dev](results/heuristics-dev/SUMMARY.md),
[profile-selection](results/profile-selection/selection.json)). Nothing here is tuned
on the confirmatory instances.

## Frozen algorithm

`solve_profile_portfolio(controller="stagnation", patience=10,
job_selection="rules", max_jobs=3, fraction=0.2, candidate_count=8)`:
whole-job DP profile moves, with one uniform-candidate MILP repair after 10
consecutive failed profile moves. Source hashes are stored in `plan.json`. A study
whose source changes cannot be resumed.

## Fixed protocol

- **Instances:** 60 fresh planted instances from `build_suite(seed=47000000)`:
  30 IID and 30 larger OOD, split equally between balanced, energy and periodic
  regimes. They are rejected if any hash overlaps an earlier study manifest.
- **Methods:** `portfolio` (frozen above) and three comparators:
  - `heuristic` (hybrid, fraction 0.5, as in the earlier confirmatory study);
  - `uniform` (ranked-uniform, fraction 0.2);
  - `profile` (profile moves only; secondary).
- **Runs:** seeds 101, 211, 307; 720 runs in total. Every run gets the common
  witness, 3 s of CPU wall time, a 0.3 s repair cap and one solver/Torch thread.
  Runs execute in a seeded random order across 8 worker processes.
- **Scoring:** every output is checked by the independent oracle, and deadline
  scoring truncates late events. Failures score zero and stay in the data.
- **Unit of analysis:** the 60 instances. Seed results are averaged per
  instance before inference.

## Hypotheses and decision rule

The primary family has two comparisons: `portfolio` versus `heuristic`, and
`portfolio` versus `uniform`. For each, the statistic is the mean relative
objective change over the equal IID/OOD mixture, tested with two-sided paired
t-tests, Holm-adjusted, with Bonferroni simultaneous 95% CIs.

- **Statistical superiority:** Holm p < 0.05 and a simultaneous lower limit > 0.
- **Practical superiority:** statistical superiority with a lower limit > 1%,
  for both comparisons, and every output valid by the deadline.

The comparison with `profile` and the IID/OOD split means are descriptive only.
Development data suggest the gain is concentrated on OOD instances, with about
+0.3% on IID. A supported mixture effect will therefore not be presented as an
IID effect.

## Stopping and scope

The study has a fixed size of 720 runs, with no interim significance checks and
no optional stopping. Inference is conditional on the planted generator, the
witness warm starts, a 3 s budget and these seeds. No cross-problem or
state-of-the-art claim follows.
