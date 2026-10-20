# Completion-supervised, imitation-preserving actor-critic

This intervention keeps the pooled MLP actor from the architecture comparison
and changes its training and decision guidance. It adds a 20→64→1 completion
classifier and a 40→64→1 state-value network to the 12,545-parameter actor.
The complete model has 16,643 parameters. Implementation: `rl_completion.py`.

## What each head learns

The completion head scores each task-profile candidate from the twenty recovery
features. Positive examples require a credited, validated descendant whose
**entire task-profile prefix** matches the proposed action. Negative examples
require an exact necessary-condition rejection or exhaustion of that candidate's
restricted-menu subtree. Exhaustion refers to our profile menu, not the entire
ONTS feasible set. A branch interrupted by a deadline is unknown; a duplicate
state alone is also unknown. Neither is automatically a negative example.

Completion supervision is balanced between observed positive and negative
examples. Scores therefore do not constitute calibrated probabilities of true
ONTS feasibility. Inference adds `0.5 * log(sigmoid(completion_logit))` to actor
logits. This is soft ranking only. Learned scores never prune actions, replace
exact completion probes, relax constraints, or decide whether an output is valid.

The state-value head receives candidate-feature means and maxima and predicts
observed final search reward, bounded between zero and two. Its target is the
final budget-limited return, including failed searches. Failure under a budget is
an observed outcome, not a claim that the partial state is infeasible. The value
head predicts search return; it is distinct from the branch-completion classifier.
The critic omits some search history/incumbent context, so it is an approximate
baseline rather than a complete Markov-state representation.

## Conservative PPO and imitation protection

Four searches per training instance supply on-policy candidate observations,
selected actions, old log probabilities and old state values. Rejected and
backtracked choices are included. The advantage is the undiscounted terminal
return minus the learned value. Each group has two PPO epochs, using a 0.2
clipped likelihood ratio. The policy surrogate sums decisions within searches;
the critic reduces variance, while completion evidence supplies local supervision.
This version does not add dense environment rewards or use GAE.

The actor retains supervised teacher replay with coefficient 0.1 and a KL
penalty against its frozen initial imitation policy. Each attempted update
measures exact mean `KL(initial imitation || updated actor)` over the collected
states. If it exceeds 0.02 or produces nonfinite parameters, **all model and
optimizer state changes are rolled back**. This bounds drift on sampled training
states only, not on every possible state or unseen instance. The completion
score is excluded from the actor-anchor KL but included in the sampled action
likelihood used by PPO. PPO does not backpropagate policy loss into the completion
classifier; that head learns from completion evidence.

The method uses the clipped surrogate from
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347),
with project-specific imitation anchoring, evidence labels and recovery actions.
It is an adaptation, not a reproduction or SOTA claim.

## Controlled pilot

The protocol freezes nine new training cases, six new same-size cases and three
new larger cases from the tighter planted family. Their hashes are disjoint
from all previous recorded instances. The private generator witness is discarded.
Training sizes are 6 tasks/48 slots and 9/72; the larger evaluation size is 12/96.
Three seeds, ten initial imitation epochs, fifty classifier warmup updates,
32 four-search groups, and two PPO epochs per group are declared before training.
Teacher/training/evaluation search allowances are 0.7/0.35/2 seconds, with
128 training decisions and 4096 evaluation decisions. Source snapshots use the
same verified NumPy profile backend as preceding construction experiments.

The six recovery arms are:

- Uniform completion-aware recovery.
- Initial imitation, with no subsequent updates.
- Additional teacher replay, with 64 attempted optimizer updates.
- Previous grouped REINFORCE, with 32 optimizer updates.
- Completion-guided anchored teacher replay.
- Completion-guided anchored actor-critic PPO.

The two completion-guided arms start from identical actor/classifier/value
weights. They receive the same teacher minibatches and balanced classifier
examples from teacher evidence and PPO's visited training states. Both attempt
64 updates, use the same classifier loss, actor anchor and rollback rule, and
receive the same inference guidance. The replay control uses supervised actor
updates and does not train a search-value baseline. PPO uses policy gradients
and value supervision. Their primary paired difference addresses RL training
beyond supervised completion guidance and imitation protection. Accepted update
counts may differ because of rollbacks; training time and data volume are not
matched. REINFORCE's smaller optimizer-step count makes it a diagnostic reference,
not a single-variable critic ablation.

All policies use identical backtracking, memoization, profiles, exact probes and
validated-best retention. Full-instance MILP is separately evaluated with the
same total two-second allowance, including assembly and validation. Five-second
MILP references provide optima where certified. Failed and late outputs remain
in the feasibility and optimum denominators and have zero evaluation utility.

The primary contrast is pooled completion-PPO minus completion-replay on-time
utility. Seeds are averaged within each instance before 4000 bootstrap resamples.
Secondary contrasts use imitation, plain replay and REINFORCE. Intervals are
exploratory and marginal, without multiple-comparison correction. Nine instances
and this synthetic family cannot establish robust arbitrary-instance feasibility
or a broad RL advantage.

## Reproduction and audit

Run into a fresh directory:

```bash
OMP_NUM_THREADS=1 /home/seman/dev/foreblocks/.venv/bin/python research/run_recovery.py \
  --profile-snapshot research/results/rl-construct-graph-20261011-fresh \
  --study-module rl_completion_study --output /tmp/onts-completion-critic \
  --data-seed 2026201600
```

Audit the exact frozen sources:

```bash
/home/seman/dev/foreblocks/.venv/bin/python research/audit_rl_construct_frozen.py \
  --module audit_rl_recovery --output /tmp/onts-completion-critic
```

The run archives the protocol, source hashes, instances, teacher searches and
completion evidence, training searches, compact learning traces, update/KL/
rollback diagnostics, fifteen final checkpoints, evaluations, MILP outputs and
paired contrasts. Temporary full on-policy observations are consumed by updates
and not archived in every training transition. The auditor checks all recorded
schedules and prefixes, teacher snapshots, hashes, deadlines, memoization,
completion labels reconstructed from events, and recorded accepted-update KL.
Rejection replay remains bounded to 512 distinct states. Label reconstruction
uses the frozen labeling implementation; it does not independently enumerate
all exhausted subtrees or numerically reproduce every optimizer update.

Regression tests cover profile-sensitive labels, unknown/censored branches,
finite actor/critic gradients, atomic optimizer/model rollback, the sampled-state
KL bound, and preservation of feasible search despite pessimistic learned scores.
The relevant construction, recovery, architecture, validation and heuristic
suite passes 68 tests using the selected profile backend.

The teacher data contain 432 positive and 16,124 negative branch labels, with
751 unknown branches. Demonstrations cover four of nine training cases.
All 192 PPO update attempts satisfy the sampled-state KL cap and are accepted;
108 of 192 replay-control attempts are accepted. Maximum accepted KL is
0.012205 for PPO and 0.019906 for replay. Thus attempted update counts match,
but accepted counts differ materially.

Pilot output: `results/rl-completion-critic-20261011`.

## Completed pilot

On fresh held-out instances, completion PPO and completion replay each produce
13/27 on-time feasible outputs and eight certified optima. Their per-instance
mean utilities are identical. Plain teacher replay reaches 18/27 feasible outputs
and ten optima; PPO minus plain replay has pooled utility difference -0.193703
and exploratory instance-bootstrap interval [-0.348827, -0.039095]. MILP solves
all nine instances optimally in mean total runtime 0.050 seconds. This experiment
does not establish an RL advantage.

The value head learns: prediction MSE falls from 0.566 to 0.288 between the first
and last eight training groups. However, 382/384 PPO searches hit the training
cap. The frozen audit passes for 957 construction records, 3,425 completions,
56,250 prefixes, 63 teacher snapshots, fifteen checkpoints, nine solver outputs
and 63,169 completion labels, subject to the bounded-replay limitations above.
Two late held-out outputs count as failures.

A separate training-only diagnostic uses the first policy seed and all nine
training instances, with the same 0.7-second budget across variants. The trained
PPO policy succeeds on 1/9 with cap 128 and 5/9 with cap 512. At cap 4096 it
succeeds on 5/9 with completion guidance and 6/9 with guidance disabled; initial
imitation also reaches 6/9. All 54 returned schedules are independently checked.
The cap limits discovery, while the learned guidance provides no benefit in this
check. These are descriptive training results from one seed, not held-out
evidence or a selected improved model. Sources, input hashes, protocol and traces
are saved in `results/rl-completion-critic-20261011/training-diagnostic`.

The next training hypothesis is to increase search exposure and improve the
completion labels before repeating a predeclared comparison on new held-out
instances. The present results cannot support a superiority or SOTA claim.
