# Feasibility-first RL construction

The target is a frozen policy that receives only an unseen ONTS instance,
constructs a feasible schedule, and then approaches a certified optimum.
Existing repair-controller studies begin with feasible schedules and cannot
establish that capability. Their evidence and frozen records remain intact.

## First implemented experiment

`../rl_construct.py` starts from all-zero execution with no incumbent input.
RL selects a task and complete activation profile. The existing per-task DP
enforces startup counts, duration, windows and periodicity, with three pricing
templates: minimum execution, solar-favored minimum execution, and maximum
execution. Identical profiles are deduplicated within each task. Candidate
placements must satisfy instantaneous power and the maximal battery trajectory.
These local checks do not imply that remaining tasks admit completion.

The shared candidate scorer has 16 features and one hidden layer of width 48.
It includes timing parameters, profile execution, remaining-task count, charge,
residual power, and a rough remaining mandatory-energy summary. This last feature
is not a bound or a feasibility test. The implementation is a hybrid neural
constructor over DP-generated actions, not a solver-free decoder. Its restricted
menu and one-pass placement can miss feasible completions; it has no backtracking,
repair fallback, or inference-time exact solver.

Training teachers are the best on-time feasible construction among one priority
rollout and three uniform rollouts per training instance. Labels include the
visited states' actual candidate sets. All policy seeds receive the same labels.
Imitation warm-starts the candidate scorer. Grouped leave-one-out REINFORCE uses
four rollouts of the same training instance per update. A replay arm starts from
the identical warm checkpoint and receives one supervised update per RL update.
Both continuation arms reset Adam; warm-start-only imitation is also retained.
The terminal reward is `1 + objective / (T * sum(priority))` for on-time feasible
outputs, otherwise `0.1 * placed_tasks / J`. Thus each successful episode outranks
any failure. The reward trades feasibility probability against feasible quality;
it is not an exact lexicographic population objective. Entropy regularization is
0.001. Training costs and rollout counts differ between RL and replay.

## Frozen pilot protocol

Default data: nine training cases, six held-out same-size cases and three larger
cases, balanced across balanced/energy/periodic regimes. Training has horizons
48/72 and 6/9 tasks; larger evaluation has horizon 96 and 12 tasks. The planted
generator verifies a witness privately, then discards it. Instance JSONs and
policy calls contain no witness. Split hashes must be unique and are checked
against hashes in existing saved manifests. The policy is selected only by the
fixed final training step; held-out data is never used for teacher labels or
checkpoint selection. This tests instance and size transfer within one synthetic
generator family, not external or generator-family generalization.

Two policy seeds, 64 updates per seed, 20 warm-start epochs, 2-second inference
budget. Timing includes MILP model setup (used for rule/config access), proposal
generation, neural selection and final validation. Native DP work is cooperative;
overruns are recorded and late outputs do not count as successes. One construction
attempt is evaluated per policy seed and instance. This is a small exploratory
pilot, not a confirmatory statistical test.

Comparators: uniform construction, priority-times-execution construction, frozen
imitation, matched-update teacher replay, and RL continuation. Full-instance MILP
references run separately after checkpoints are frozen, with a 5-second allowance.
Only an optimal solver status supplies a certified optimum. A timeout is unknown,
not proof of infeasibility. Reference computation is an offline measurement cost,
not part of the policy's inference budget.

Record feasibility over every attempt, on-time success, time to first feasible
output, runtime overruns, objective and certified optimum gaps. Failed attempts
have no objective gap and must remain in the feasibility denominator. Report
certified-optimal counts separately from conditional mean gaps. All final policy
schedules are cross-checked with the independent run/gap/battery oracle.

Run from the foreblocks root into a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ONTS_SOLVER_THREADS=1 \
  .venv/bin/python -m projects.scheduling.research.rl_construct_study \
  --output projects/scheduling/research/results/rl-construct-20261011 \
  --episodes 64 --policy-seeds 11 29 --budget 2 --reference-budget 5
```

The first milestone is reliable cold-start feasibility, with RL's contribution
measured against the matched replay arm. High success on this planted family
alone does not meet the eventual external-generalization target. Next decisions
should follow observed failures: enlarge the profile menu or add learned
backtracking when completion fails, and introduce untouched generator families
and real instances before scaling training. A proof of arbitrary-instance
optimality requires a complete exact search backend; this policy provides none.

## Completed pilot, 11 October 2026

`results/rl-construct-20261011/REPORT.md` records 90 evaluations: RL succeeds
on 12/12 same-size and 5/6 larger attempts; uniform and imitation succeed on
18/18 each. Teacher replay succeeds on 16/18. No evaluation exceeds budget.
There is no established RL-specific feasibility advantage. Only one of nine
instances is certified optimal within the reference allowance; RL's mean gap
on its two attempts is 0.2817. These data do not meet the eventual goal.

Independent replay validates 638 constructions (90 evaluations, 512 training
rollouts, 36 teacher rollouts), nine reference schedules and six checkpoints:

```bash
.venv/bin/python -m projects.scheduling.research.audit_rl_construct \
  --output projects/scheduling/research/results/rl-construct-20261011
```

The source archive was copied after execution; each byte hash matches the
pre-training protocol's source hash. The audit can verify recorded timing credit
and schedule correctness, but cannot independently recreate elapsed execution
or certify the solver's optimal status. All larger-reference solves timed out.
