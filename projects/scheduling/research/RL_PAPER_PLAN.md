# Reinforcement learning for ONTS: separate-paper research plan

Date: 6 October 2026. Status: proposed experiments; no RL performance claim.

## Research question and contribution

Can a policy trained on sequences of ONTS search decisions improve feasible
priority-weighted execution within a fixed online runtime, and generalize to
unseen battery/solar regimes, horizon lengths and constellation sizes?

The existing paper contributes a profile-based primal heuristic. This project
will study learning search control for the same problem. A credible contribution
requires evidence that temporal/resource information and sequential training
help beyond its strongest heuristic, a contextual bandit, and supervised
neighborhood selection. Merely combining a GNN with PPO is insufficient.

Retain the original objective, battery equations, horizon conventions and
constellation constraints. Deterministic offline scheduling is the initial
scope. Uncertain solar forecasts and online arrivals are later extensions with
separate information assumptions. Training randomness does not make the
scheduling problem stochastic.

## First approach: RL improvement policy

Start from a validated feasible schedule. An action chooses a structured repair:
an operator family, tasks (initially up to three), and a repair-budget bucket.
Initial families are profile reinsertion, energy-bottleneck repair, and broader
MILP repair. Use a bounded list of candidate actions for the first pilot; then
test autoregressive task selection without replacement. Include STOP and
exclude actions whose maximum allowance exceeds the remaining deadline.
All methods receive the same candidate-generation machinery in controlled
selector comparisons; generating and encoding candidates consumes runtime.

The policy observes the instance, incumbent, current battery trajectories,
startup/run/deadline information, remaining runtime, and search history.
Explicitly record recent action outcomes, overlap and stagnation. A graph or
typed-attention encoder should represent tasks, time slots, satellites and
task-time-satellite eligibility, with battery transitions across time. Preserve
task/satellite permutation equivariance and avoid dense attention across all
binary variables as the default.

For an exact Markov formulation, include all persistent search state that
affects future transitions, including adaptive energy prices/operator state.
A compressed observation is an approximation; use recurrence and document
omitted state rather than claiming that a handful of features is sufficient.
An action consumes variable wall time, making this naturally a semi-Markov
control problem. Freeze solver settings and log durations and timeout statuses.

Repair produces a candidate. Independently validate it and retain the incumbent
unless the candidate is feasible and nondegrading. Solver failure or timeout
without a candidate leaves the incumbent unchanged; it is not an infeasibility
certificate. This establishes returned-incumbent preservation, not optimality.

Use a fixed wall-clock episode budget. The primary reward is

    r_k = (f(x_{k+1}) - f(x_k)) / C(instance)

where C is a positive, fixed scale independent of learned choices (for
nonnegative priorities, the unconstrained sum of available weighted execution
is an option; handle zero separately). With undiscounted finite-episode return,
these rewards telescope to normalized final improvement. Runtime affects how
many actions fit in the episode. Test a compute-cost penalty only as a separate
objective ablation; improvement/time ratios can change the desired objective.
Reserve time for validation and credit only outcomes available by the deadline.

Begin with categorical PPO and an actor-critic encoder. PPO is an established
scheduling baseline, not a claim about the newest or best optimizer. Use an
optional imitation warm start from strong heuristic trajectories, then compare
PPO from scratch, imitation only, and imitation plus PPO. Train on multiple
initial incumbents and budgets. For hierarchical actions, calculate the joint
log probability with the actual conditional masks; do not apply PPO to labels
that do not define a policy distribution.

## Second approach: direct construction

After the search-control pilot, test an autoregressive policy selecting
task/run/start/satellite decisions, with explicit defer/stop decisions. Avoid a
flat action enumerating every simultaneous task subset (exponential in tasks).

Local action masks can enforce eligibility, startup separation and instantaneous
power. They cannot by themselves ensure that remaining mandatory starts,
minimum runs, future periodicity windows and battery needs admit a completion.
Report dead-end rate. Compare necessary lookahead masks, a completion/repair
oracle and a purely constructive decoder, charging oracle cost to inference.
Never describe local masking as a full feasibility guarantee.

Keep the offline solar profile available to every offline method. A policy
with only causal observations solves a different information problem. For
constellations, enforce global startup counts/spacing and nonoverlap as well
as each satellite's energy constraints.

## Relevant methods and their roles

These are transfer references, not verified ONTS state-of-the-art solvers.

| Reference | Role in this project |
| --- | --- |
| [Learning to Dispatch, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html) | Foundational graph-based RL dispatching; PPO construction baseline. |
| [Learning Large Neighborhood Search Policy, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/fc9e62695def29ccdb9eb3fed5b4c8c8-Abstract.html) | RL selection of variables for integer-programming improvement. |
| [MIStar, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/77629f2df5ef97ad967157c69cd8af7b-Paper-Conference.pdf) | Memory-enhanced learning of FJSP improvement moves; motivates search-history experiments. |
| [ReSched, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/467e109270678b3e2b3c2e7fa6b3d575-Abstract-Conference.html) | Contemporary Transformer scheduling comparator; its FJSP state simplification needs ONTS-specific adaptation. |
| [Balans, IJCAI 2025](https://www.ijcai.org/proceedings/2025/286) | Adaptive bandit LNS comparator that tests whether sequential neural RL is necessary. |

Job-shop machine/precedence representations do not directly encode recurring
ONTS starts or battery dynamics. Reimplement only clearly identified transferable
components and label them as adaptations. Consult the existing
[neural scheduling review](outputs/neural-scheduling-review-2026-10-05.md)
for additional ranking, semantic-constraint and preference-learning references.

## Evaluation contract

Use new manifests, checkpoints and result directories. Leave the completed
heuristic studies intact. Previous development/test instances are known data;
create fresh confirmatory holdouts and reserve literature instances for clearly
labeled external evaluation. Split by instance content and generator family;
fit normalization and select checkpoints only on training/validation data.

Baselines: the full current heuristic; uniform and utility selectors over the
same actions; a contextual bandit; supervised candidate ranker; PPO with a small
non-graph encoder; and a competitive exact solver. Include a contemporary
graph/Transformer RL adaptation when the basic experiment works. Use the same
repair backend in selector ablations. Compare commercial solvers if available,
otherwise state the backend limitation explicitly.

Run both common-incumbent and cold-start tracks. The first isolates control;
the second includes initialization. Evaluate several deadlines (a proposed
pilot grid is 3, 10, 30 and 60 seconds). Charge initialization, candidate
generation, inference, repair and validation. Report hardware/threads and
parallel sampling; separate offline training and teacher-label costs.

Measure feasibility over all instances, deadline objective, paired objective
improvement, and runtime overruns. Report conditional feasible-pair quality
alongside unmatched feasibility outcomes. Use optimum gaps only with certified
optima/bounds; otherwise label references best-known. Treat anytime comparisons
carefully when backends expose incumbents at different times.

Separate training-seed and evaluation-seed variation. Bootstrap at instance
level; preregister primary comparisons and adjust multiple comparisons. Test
larger horizons/task sets and unseen energy/periodicity regimes; test larger
constellations only after single-satellite correctness and learning are stable.

Retrain ablations: temporal/battery features, history, graph edges, RL versus
imitation, candidate selection versus task selection, and repair-budget control.
Compare long episodes with one-decision episodes to test the value of sequential
planning beyond a contextual bandit. Failed or neutral outcomes remain reportable.

## First executable milestone

1. Implement a common-incumbent environment over a small fixed action portfolio,
   with complete transition logs and independently validated returned schedules.
2. Verify reward telescoping, deadline accounting, STOP, timeout handling,
   mask/log-probability consistency, and train/test separation on tiny instances.
3. Compare uniform, utility, contextual-bandit and small-encoder PPO policies
   on a fresh development split before investing in a large graph model.
4. Add typed graph/attention and history only after the environment and learning
   curves are credible. Freeze a new confirmatory protocol after the pilot.

Implementation and training have not started under this plan. Compute budgets
must be set for this project; an earlier study's allowance does not transfer.
The paper's working title is “Learning Resource-Aware Search Policies for
On-Board Nanosatellite Task Scheduling.” Its results and claims will be written
from the new experiments, not assumed in advance.

## Implementation status (6 October 2026)

The first executable development scaffold is now in `../rl_search.py` and
`rl_pilot.py`: compact actor-critic PPO, four existing MILP neighborhood actions,
STOP, uniform/utility/contextual-UCB controls, transition logs, and fresh
content-disjoint planted splits. Profile reinsertion, variable repair budgets,
graph encoders, cold-start tracks and the full heuristic comparator remain
future experiments. This scaffold is intentionally narrower than the full plan.

Run a fresh output directory with:

```bash
.venv/bin/python -m projects.scheduling.research.rl_pilot \
  --output /tmp/onts-rl-development --episodes 24 --budget 1
```

The next implementation adds exhaustion-only STOP, batched PPO, a typed sparse
constraint-graph actor-critic, operator history and outcome-preference warm
starts. See [RL_UPGRADE.md](RL_UPGRADE.md) for the implementation, literature
basis, runnable paired experiment and explicit evidence limits.
