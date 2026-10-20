# Completion-aware decisions and dead-end recovery

The constructor now checks necessary conditions for a completion and can undo
placements to explore alternatives. It receives only the instance, with no
witness, feasible incumbent or exact solver at inference. The implementation is
`../rl_recovery.py`.

## Completion checks

For a proposed placement, each remaining task receives a relaxed single-task
DP: all other unplaced tasks are absent, but the proposed partial load remains.
Slots exceeding residual instantaneous power are forbidden. The DP minimizes
execution while enforcing startup counts, windows, duration and recurrence.
If any relaxed problem has no profile, a complete schedule cannot exist under
that irrevocable partial schedule: adding nonnegative loads cannot restore an
unavailable slot. A timeout is unknown, and is not cached as an infeasibility
claim.

Let `m_j` be each relaxed task's minimum number of execution slots. The total
minimum energy demand is the partial load plus `sum(p_j * m_j)`. A necessary
energy condition compares it against

```
sum(solar) + (initial_charge - minimum_charge) * 60 * Q * V / eta.
```

This follows by summing the battery update inequalities and requiring final
charge at least its minimum. Ignoring curtailment gives an optimistic resource
bound. Passing these checks is only **not ruled out**; simultaneous tasks and
time-dependent battery constraints may still prevent completion. A particular
minimum-execution profile failing its battery check is not used as proof that
all completions fail.

## Search and learned decisions

Search caches each state's untried choices and backtracks after dead ends or
complete schedules. The best independently validated, timely feasible schedule
is retained while alternatives are explored. No feasible result is supplied at
initialization. Restarting this deterministic menu would revisit the same tree;
backtracking explores different branches instead.

The current version also memoizes `(execution bitmap, remaining-task set)`.
Different task orders leading to the same partial state have the same future
DP menu. Skipping repeats preserves discovery within this restricted menu; it
does not establish completeness for ONTS. Defaults permit 4096 sampled decisions
within the wall-clock budget. Deadline expiry is unknown, not tree exhaustion.

The recovery policy is a 1409-parameter `20 → 64 → 1` candidate MLP. It extends
the sixteen candidate summaries with optimistic remaining-energy slack,
remaining-task slot scarcity, remaining-runtime fraction, and fraction of
untried choices at this node. This is a recovery-aware constructor, not a new
graph model. Imposing the same completion/recovery backend on uniform, imitation,
replay and RL distinguishes algorithmic recovery from learned control.

Training uses a shared imitation start and grouped leave-one-out REINFORCE with
four search trajectories per instance. The likelihood includes sampled rejected
and backtracked choices, not only the final solution path. Final-search feasible
utility supplies the return; failed searches receive a small deepest-partial
completion reward. Teacher labels come from the best validated search path at
visited training states. Teacher replay matches one optimizer update per RL
update but does not match training transition counts or runtime.

## New benchmark family and first pilot

The tighter planted family privately verifies a feasible witness, then discards
it. Solar follows witness load plus 0.15--1.2 W; windows narrow around the two
witness runs. Each task needs two startups. Battery capacity is Q=0.2, minimum
charge 0.1 and maximum discharge current 0.3. This deliberately creates stronger
resource competition than the preceding broad family. It remains synthetic and
planted, and changing the benchmark is not an algorithmic improvement claim.

`results/rl-recovery-20261011` freezes nine training, six untouched same-size
and three larger instances; three seeds, 32 four-rollout RL updates, 20 teacher
epochs. Teacher/training/evaluation allowances are 0.7/0.35/2 seconds, with
64 training and 256 evaluation decisions. Larger sizes occur only in evaluation.
The one-pass uniform, backtracking uniform, completion uniform, imitation,
teacher-replay and RL arms yield 162 evaluations. Method order is randomized
within instance/seed. Checkpoints freeze before held-out evaluation. Full-instance
references run separately with five seconds; all nine have certified optima.

The declared primary comparison is completion RL minus completion replay on
pooled on-time utility: one plus normalized objective for success, zero for
failure. Seeds are averaged within instance before 4000 bootstrap resamples.
Intervals are exploratory and marginal; nine instances cannot establish broad
generalization or SOTA.

The initial pilot finds no feasible one-pass uniform schedule (0/27); backtracking
alone succeeds 2/27, completion-uniform 5/27, replay 9/27 and RL 7/27. RL's pooled
utility difference from replay is -0.077675 with interval [-0.193416, 0]. No RL
advantage is established. Every recovery evaluation hits the 256-decision cap,
with completion methods averaging roughly 0.23 seconds despite a two-second
allowance. This diagnoses a search-budget bottleneck as well as a restricted
profile-menu problem.

Strict replay validates 573 final construction records, 1383 complete candidates,
17349 placed prefixes, all 48699 rejection events (24524 distinct checks), and
nine checkpoints. A post-run auditor correction checks exact schedule identity
and timely validation after discovering a late-completion fallback edge case.
No recorded output exploited that edge case. Current code excludes complete
schedules from the partial failure fallback; a regression test injects a delayed
completion. Original experiment source and records remain frozen, with the
auditor correction and its hash stored separately.

## Fresh memoization/cap follow-up

`results/rl-recovery-memo-20261011` freezes another six same-size and three
larger instances. It retains the exact final checkpoint bytes, increases the
decision cap to 4096, enables state memoization, and uses the corrected acceptance
fallback. This is an inference intervention, not retraining. The same six
comparators receive two seconds, and the declared primary contrast remains
RL versus replay. Instance hashes are checked against previous manifests and
protocols. Both positive and negative results must remain reportable.

The completed follow-up finds 19/27 feasible completion-uniform outputs versus
7/27 for backtracking alone and 0/27 for one-pass uniform. The paired completion
minus backtracking utility interval is [0.235982, 0.665108]. This is favorable
exploratory evidence for the recovery recipe. RL and replay each succeed 16/27;
their pooled difference is -0.000129 with interval [-0.115741, 0.115352]. There
is no established RL-specific advantage. All nine references are certified
optimal. Uniform/replay/RL reach those optima on 12/11/9 attempts respectively;
conditional mean gaps are 0.0276/0.0192/0.0288. Failures remain in the success
and optimality denominators. These supplementary gaps are stored separately in
`optimality-supplement.json`, derived from existing outputs without new solves.

Two deadline overruns (0.290 and 0.389 seconds) count as on-time failures. There
are no decision-cap hits. The follow-up audit checks all 162 final records,
1140 complete candidates, 61158 placements and memoization traces, plus nine
checkpoints. It replays 6144 rejection events representing the first 512 distinct
states; 177561 rejection replays are skipped. This is bounded pruning replay,
not a claim that every necessary-condition result was independently recomputed.
The core pruning tests also preserve all independently enumerated feasible
completions on tiny boundary cases. Fifty-three relevant tests pass using the
selected NumPy backend.

Replay the strict auditor revision against the frozen runtime sources:

```bash
/home/seman/dev/foreblocks/.venv/bin/python research/audit_rl_construct_frozen.py \
  --module audit_rl_recovery \
  --output research/results/rl-recovery-memo-20261011 \
  --auditor-source research/results/rl-recovery-memo-20261011/audit-correction/audit_rl_recovery.py
```

All arms use the hash-verified NumPy profile backend from the preceding graph
snapshot, isolated from concurrent Numba-backend edits. New source archives
contain the exact runtime package, not the unrelated live backend. Run a new
training study into a new directory from the scheduling directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ONTS_SOLVER_THREADS=1 \
  /home/seman/dev/foreblocks/.venv/bin/python research/run_recovery.py \
  --profile-snapshot research/results/rl-construct-graph-20261011-fresh \
  --output /tmp/onts-new-recovery-study
```

Run an inference follow-up with frozen checkpoints:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ONTS_SOLVER_THREADS=1 \
  /home/seman/dev/foreblocks/.venv/bin/python research/run_recovery.py \
  --profile-snapshot research/results/rl-construct-graph-20261011-fresh \
  --study-module rl_recovery_followup \
  --checkpoints research/results/rl-recovery-20261011 \
  --output /tmp/onts-new-recovery-followup --data-seed 2026201400
```

Every comparison uses its own frozen study's shared budget and backend. A
recovery gain over one-pass construction does not establish an RL gain. Failure
after exhausting the available DP proposals does not prove ONTS infeasibility.
