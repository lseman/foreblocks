# Research direction: neural-guided ONTS optimization

## Target and hypothesis

Solve the priority-weighted offline nanosatellite scheduling problem defined
by the base constraints of `original.py`. Test whether a policy that selects
job/time neighborhoods, followed by exact MILP repair, improves feasible
objective value under a fixed wall-clock budget. This is a research hypothesis,
not an established state-of-the-art result.

## Correctness gate

Use `validation.validate_schedule(instance, x)` on every returned schedule.
It checks derived startups, counts, execution windows, full sliding-window
periodicity, run lengths, power and battery energy independently of the policy.
Report feasibility separately from reward; never rank invalid schedules as
solutions. This checker targets base constraints and excludes optional valid
inequalities pending an audit. It accepts one complete [J,T] schedule; check each
batch member individually. Example from the repository root:

```python
from projects.scheduling.validation import validate_schedule

report = validate_schedule(env.instance, env.x[0])
print(report.feasible, report.objective, report.violations)
```

Before building a baseline, reconcile these model differences:

- `original.py` uses `int(value) * 10` for solar power, losing fractional
  precision. JSON/environment values are used directly. Select and document
  one physical unit conversion for all methods.
- Battery current in the reference is bounded above by available net power,
  allowing curtailment. The generator's cumulative energy equality does not
  allow curtailment, while the environment saturates at full charge. Use one
  common battery model. The checker uses curtailment and reference defaults.
- Minimum startup separation is enforced even for periods longer than the
  horizon; the reference's full-window loop is vacuous in that edge case.
- The reference permits horizon-truncated minimum runs. The environment
  disallows those starts. The checker exposes both conventions explicitly.
- One reference inequality indexes negative startup times and can raise
  KeyError. Its final inequality applies only to the last job because it is
  outside the job loop. Audit each proposed cut for validity, not just indexing.
- Local environment masks do not prove complete-schedule feasibility. Forced
  jobs can collectively exceed power; run continuation can cross a window;
  missing periodic starts are not all captured by terminal startup penalties.
- Generator defaults (nine small tasks drawing at most 2.5 W each, with
  10–15 W solar plus 18 W battery power) make instantaneous power constraints
  inactive. Include genuinely energy-constrained instances in evaluation.

## Implemented foundation and remaining research

The shared sparse MILP, independent validation, random/heuristic/neural LNS,
solver-imitation training, checkpoint loading and benchmark commands are now
implemented. The generator defaults to the shared curtailment model; legacy
cuts/battery equality require an explicit option. The following describes the
research sequence, including work still needed for a paper.

1. Build one reusable sparse MILP constraint representation, shared by exact
   solving and repair. Maximize priority-weighted execution. Return incumbent,
   bound, status, runtime and proven gap; a timeout with no incumbent is unknown,
   not a proof of infeasibility. Validate incumbents independently.
2. Establish solver-only, priority-per-power greedy, current PPO, and random
   neighborhood LNS baselines. Use the same model and objective for every method.
3. Produce an initial feasible incumbent with the exact solver or validate a
   neural proposal and repair it. Keep that incumbent throughout the search.
4. Encode job/time decision variables and actual constraint coefficients in
   a bipartite graph. Existing family-level summary nodes are useful features
   but are not the full MILP graph. Add battery slack, upcoming solar trajectory,
   startup deadline slack and incumbent values.
5. Predict which job/time blocks to release. Fix other execution variables to
   the incumbent; leave startup and battery variables free to preserve their
   dependencies. Solve the restricted MILP under a per-neighborhood budget.
   Accept only independently validated objective improvements. Reserve budget
   for initialization, graph construction and all solver calls.
6. Train neighborhood selection by imitation of successful solver repairs,
   then evaluate policy improvement. Include random and heuristic selection
   ablations before attributing gains to learning.

Exact repair guarantees constraints for returned validated incumbents; it does
not guarantee that every proposed neighborhood has a solution or that a
budgeted run finds an initial incumbent.

## Experimental protocol

Split by instance identity before solver labeling; keep generation seeds and
train/validation/test manifests. Include bundled instances and generated sets
varying jobs, horizon, solar scarcity, initial charge, windows and periodicity.
Test larger unseen sizes and unseen energy profiles separately. Feasibility
filtering changes the sampled distribution: publish acceptance rates and avoid
selecting test instances according to a particular method's success.

Use matched wall-clock budgets on the same hardware, multiple training seeds
and paired evaluation on the same test instances. Include graph construction,
repair and validation time; report training and label-generation cost separately.
Report feasible solution rate, time to first feasible solution, validated
objective, solver upper bound and relative primal gap when available, and
objective-versus-time curves. An unproven solver incumbent is not an optimum.
Report infeasible and unknown instances separately. Bootstrap paired per-instance
objective differences and show variation across training seeds.

Ablate encoder, future-energy features, neighborhood selector, neighborhood
size, repair budget and valid inequalities. The paper claim should be limited
to the tested ONTS distributions and baselines until broader evidence exists.

## Primary literature

- Sonnerat et al., *Learning a Large Neighborhood Search Algorithm for Mixed
  Integer Programs*: https://arxiv.org/abs/2107.10201
  Neural initialization and learned neighborhood selection with exact repair.
- Huang et al., *Searching Large Neighborhoods for Integer Linear Programs
  with Contrastive Learning* (ICML 2023):
  https://proceedings.mlr.press/v202/huang23g/huang23g.pdf
  Contrastive neighborhood learning and anytime evaluation.
- Liao et al., *BOPO: Neural Combinatorial Optimization via Best-anchored and
  Objective-guided Preference Optimization* (ICML 2025):
  https://proceedings.mlr.press/v267/liao25a.html
  Preference learning baseline relevant to scheduling. Do not identify an
  arbitrary best/worst logistic loss as a faithful reproduction without checking
  the algorithm, reference policy, trajectory probabilities and batching.

These references motivate candidates; they do not establish that our proposed
ONTS method is novel or superior. A focused literature review remains necessary.
