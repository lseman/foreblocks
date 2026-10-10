# ONTS experiment log

## Hypotheses

H1: the sparse model and complete-schedule checker describe the same feasible
set as an independent run/gap/energy oracle.
H2: solver-guided neighborhood search improves feasible objective over a
short-budget unrestricted solver on held-out ONTS instances.
H3: learned neighborhood selection improves over random and heuristic selection
under a matched wall-clock budget.

H1 is tested by exhaustive tiny-instance enumeration and a formulation proof.
H2 and H3 require measurements; improving an initial schedule alone does not
establish either comparison.

## Local budget and protocol

The user authorized up to about one hour of local compute. Run serially with
one HiGHS thread and one PyTorch thread. No external compute is provisioned.
Freeze source hashes and dataset identities before the confirmatory run.
Preserve checkpoints and raw result records. Only validation data selects
search fractions. The exploratory pilot seed and confirmatory dataset seed
differ. Credit only solutions available by the advertised deadline. Report
later outputs and runtime overruns separately.

The local protocol includes three training/search seed pairs, IID test data,
larger-size OOD data and all bundled JSON instances. All methods receive the
same physical model. The common-incumbent track isolates search; it excludes
preprocessing from the search timer and reports that cost separately. Supplied
incumbents are retained, not native HiGHS MIP starts.

## Exploratory pilot

The first pilot used six training, three validation, three IID test and three
OOD instances with one seed. It completed 66 search runs without runtime errors.
The neural method improved some incumbent schedules and exceeded the
short-budget exact solver on some cases; it also lost to random and heuristic
neighborhood selection. This pilot does not establish learned superiority.
The full study uses new data identities rather than reusing the observed test
instances. No architecture is selected by pilot test performance.

## Decision rules

Do not claim state-of-the-art results. Separate correctness, incumbent
improvement, comparison with exact solving, and comparison with non-neural LNS.
Use paired instance means, marginal bootstrap intervals and Holm-adjusted sign
tests. Keep infeasible/unknown/error outcomes in feasibility denominators.
A conditional objective comparison is not evidence of better feasibility.
Record a negative H3 result if the learned selector does not beat controls.

## Completed confirmatory study

All 1,314 comparisons completed without runtime exceptions. Every one of the
849 returned schedules passed independent replay; 837 runs had a feasible
incumbent recorded by deadline. Together, the two exhaustive audits checked
107,008 schedules across 144 cases without disagreements or optimum mismatch.
The scheduling suite passes 110 tests.

The primary IID cold 3-second comparison estimated +3.39% neural objective
change versus MILP, with marginal 95% bootstrap interval [0.98%,6.08%] and
Holm-adjusted sign-test p=0.1714. Its predefined evidence threshold is not met.
Neural selection averaged slightly below random and heuristic LNS. H3 is not
supported, and this study does not substantiate state-of-the-art claims.

Keep the verified model, repair framework, non-neural selectors and reproducible
evaluation infrastructure. Retain the learned selector as experimental.
Future changes need fresh validation and held-out identities. Do not tune on
these completed test outcomes.

Report generation required a post-completion correction for null reference
objectives; hashes and the original source archive document that change.
The AUC report now explicitly discloses that LNS logs intermediate repairs
while SciPy supplies unrestricted incumbents only at return, so its AUC cannot
establish native solver trajectory superiority. Optimization records are
unchanged. See [RESULTS_REVIEW.md](results/RESULTS_REVIEW.md) for complete limits.

## Semantic candidate ranker pilot

Implemented outcome preference training, sparse semantic attention, history and
complete-run candidate moves. The fresh 99-run exploratory pilot returned 99
independently valid schedules. All 123 tests pass. OOD improvement versus old
imitation was +3.62%, but adjusted comparisons did not establish superiority
and strong non-neural controls remained better. Keep the heuristic default;
retain the new selector as experimental. Do not retune on this pilot. See
[the decision and limitations](results/ranker-pilot/RESULTS_REVIEW.md).

## Fixed semantic ranker confirmation

Completed 720 runs across 60 fresh held-out identities and three paired seeds.
Matched teacher calls and froze all choices before evaluation. All schedules
passed independent and deadline checks. Mean superiority over uniform selection
is supported (+0.284%, simultaneous CI [0.041%,0.526%], Holm p=0.0166);
imitation and heuristic superiority are not established. The joint >1%
criterion failed. Keep the heuristic default. No optional stopping or test
retuning. See [full interpretation](results/ranker-confirmatory/RESULTS_REVIEW.md).
