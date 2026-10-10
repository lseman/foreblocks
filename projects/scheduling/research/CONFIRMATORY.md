# Confirming the semantic ranker

This study tests a frozen algorithm, rather than changing the algorithm until a
p-value is small. A larger sample can establish superiority only if the
population effect supports it. The preceding pilot lost to strong controls.

## Fixed protocol

The executable [confirmatory.py](confirmatory.py) writes its plan, source hashes,
source archive and disjoint dataset before training or evaluation. The study
uses 24 fresh training instances and 60 fresh test instances: 30 IID and 30
larger OOD, equally divided between balanced, energy and periodic regimes.
Nine additional validation identities are reserved and unused: parameters are
fixed from the earlier pilot. Architecture and fractions are not tuned here.

Three independent training/search seeds (101, 211, 307) produce 720 runs across
ranker, old imitation, heuristic and uniform candidate selection. Both learners
use the same training identities and common initialization schedules, 768
teacher repair calls per seed, equal repair caps, hidden size and epochs.
Wall time, successful labels and optimization update counts can differ and are
reported. Matching calls does not match training computation exactly.

Every search receives the same known feasible witness, three seconds of CPU
wall time and the same repair cap. Ranker and uniform share the same candidate
pool design and fraction. Legacy heuristic/imitation fractions are fixed from
the earlier study. Training repair-call order and evaluation order are seeded.
Each returned schedule is independently checked; deadline scoring truncates
late objective events. Failures receive score zero and remain in the dataset.
The practical-superiority decision requires every output to be valid by deadline.

## Primary estimand and decision

For each instance and method, average deadline objectives over the three paired
training/search seeds. Compute the ranker's relative change from each baseline
using these means. Instances are the 60 independent experimental units; seeds
are repeated measurements, not 180 independent examples. The primary target is
an equally weighted mixture of the six fixed distribution/size strata.

Use two-sided paired t tests on the instance differences, Holm adjustment for
three primary comparisons, and Bonferroni simultaneous 95% t intervals. A
statistical-superiority claim requires adjusted p < .05 and a positive
simultaneous lower confidence bound. The predefined stronger practical claim
requires every simultaneous lower bound to exceed 1%. All three baselines must
meet that criterion. Report means, dispersion, win/tie/loss counts and
unadjusted sign tests as sensitivity descriptions. IID/OOD breakdowns are
secondary descriptions, not separate confirmatory discoveries.

The t intervals rely on an approximately normal sampling distribution for the
instance mean; they are not distribution-free certificates. Extremely skewed
outcomes or failures must be disclosed and can limit interpretation. Inference
is conditional on these training datasets, three seeds, planted-instance
generator, warm incumbents and short CPU budgets. This study cannot establish
universal superiority or a state-of-the-art claim across scheduling problems.

The paired-observation formulation follows [NIST's paired comparison guidance](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm).
Planning uses the effect/standard-deviation relation in [NIST's sample-size guidance](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm).

## Power and stopping

The fixed sample of 60 fits the authorized roughly one-hour local compute
budget. The nine-instance, one-seed pilot estimates a 3.43 percentage-point
standard deviation against old imitation. A normal approximation with 80%
power and three-comparison correction suggests approximately 124 instances to
detect a one-percentage-point effect above a null, or 31 for two points. These
small-pilot calculations are uncertain; this study is not claimed to have 80%
power for every contrast. For a confidence lower bound above 1%, the effect in
that calculation is the distance beyond 1%, not the total improvement.

The heuristic and uniform pilot means are negative. Their relatively small
variance does not make positive superiority plausible by itself. More data may
instead support a negative conclusion. Do not expand the sample, retune the
algorithm or choose a favorable subgroup after viewing significance.

No interim significance calculation or early efficacy stopping is performed.
Complete the fixed 720 runs. If interrupted, resume identical code and data;
incomplete records support no confirmatory claim. Further algorithm development
needs fresh held-out identities and a separate frozen protocol.

```bash
.venv/bin/python -m projects.scheduling.research.confirmatory \
  --output projects/scheduling/research/results/ranker-confirmatory
```

See the frozen plan and final report in that directory. Adding this study module
changes the live source map: prior studies remain reproducible from their saved
source archives, and their frozen hashes must not be rewritten.

## Completed result

All 720 runs completed. A small mean gain over uniform selection is supported
(+0.284%, simultaneous 95% CI [0.041%, 0.526%], Holm p=0.0166).
Superiority over imitation or heuristic selection and the joint >1% criterion
are not established. See [the complete interpretation](results/ranker-confirmatory/RESULTS_REVIEW.md).
