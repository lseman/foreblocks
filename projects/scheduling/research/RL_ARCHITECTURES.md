# Neural architecture comparison for completion-aware construction

The experiment compares four ONTS policy recipes on identical raw observations,
training instances, teacher snapshots, recovery actions and inference budgets.
MILP is a full-instance solver baseline rather than a neural architecture.

| Recipe | Encoder | Decision mechanism |
|---|---|---|
| MLP | Shared independent task/time MLPs | Shared candidate scorer with task, profile-slot and global pooling |
| Recurrent pointer | Bidirectional GRUs over tasks and time | Additive attention pointing to available task-profile candidates |
| Bipartite GNN | Two alternating task/time edge-attention rounds; temporal GRU | Shared candidate scorer |
| Transformer | Two four-head self-attention layers over typed task/time tokens | Shared candidate scorer |
| MILP | Exact ONTS formulation, HiGHS | Cold-start full-instance optimization |
| Uniform recovery | No learned model | Uniform sampling of the same recovery actions |

`rl_architectures.py` implements the policies. Every neural policy receives the
same 12 task features, eight time features, three edge features, task-window
incidence, remaining-task mask, candidate profiles, and 20 recovery features.
Candidate inputs include the selected task, pooled executed time slots, pooled
profile-edge attributes, and remaining-task/time summaries. MLP/pointer/
Transformer use profile-edge summaries; the GNN additionally propagates edges.
Time features explicitly encode ordered time. Task labels have no learned index
embeddings. The recurrent pointer intentionally encodes the supplied task order;
the other three models pass task-permutation-equivariance tests.

These are adapted architecture families. The pointer has a recurrent encoder
and additive pointer head, but no persistent autoregressive decoder state: the
current recovery state is re-encoded after each decision. The Transformer uses
dense self-attention with time features rather than a routing-specific decoder.
Our bipartite graph connects **tasks to time slots**, unlike the **MILP variables
to constraints** graph used for learned branch-and-bound. The latter is a
separate solver integration, not an alternative name for the current model.

Relevant primary sources: [Pointer Networks](https://arxiv.org/abs/1506.03134),
[Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475),
and [Exact Combinatorial Optimization with Graph Convolutional Neural
Networks](https://arxiv.org/abs/1906.01629). This experiment does not reproduce
those published methods or support a state-of-the-art claim.

## Two different questions

For every neural recipe and policy seed:

1. Fit imitation to the same successful teacher decisions and save that model.
2. Clone the model into RL and additional imitation-replay branches.
3. Run grouped leave-one-out REINFORCE and match its optimizer update count
   with additional teacher replay, saving final checkpoints before evaluation.

The primary evidence for **RL advantage** is RL versus additional replay within
the same architecture. Comparing RL architectures addresses which complete
recipe works better. Beating uniform recovery alone cannot establish that RL
training, rather than the teacher initialization, caused the improvement.

A shared width of 32 and shared candidate embedding/decoder dimensions control
some capacity differences. Parameter counts and training runtimes are recorded;
they are **not matched**. The same training wall-clock allowance gives expensive
encoders fewer decisions. This measures practical recipes under budgets, not a
pure topology effect. Training is deliberately a bounded exploratory pilot.

## Protocol and interpretation

The pilot freezes nine training instances, six fresh same-size instances and
three fresh larger instances from the tighter private-witness generator. Instance
hashes are disjoint from previous manifests. The witness is discarded before any
policy or baseline call. Three seeds, ten teacher epochs and sixteen four-rollout
RL updates are used. Teacher, training and evaluation allowances are 0.7, 0.35,
and 2 seconds. Training/evaluation decision caps are 64/4096.

All neural and uniform methods share completion checks, backtracking,
memoization, candidate profiles and validated-best retention. MILP receives no
witness or incumbent and optimizes its full feasible set. Its two-second timing
includes formulation construction, solving and independent output validation.
Independent five-second solves provide optimum references where certified.
The deterministic MILP baseline runs once per instance; neural arms run once
per policy seed. There are 13 recovery arms, plus MILP.

Failures and overruns have zero held-out utility and remain in every success
and optimality denominator. Utility on success is one plus normalized objective.
Optimum hits require a timely feasible output matching a certified reference.
Paired utility contrasts average policy seeds within each instance, then
bootstrap instances. These marginal exploratory intervals have no multiple
comparison correction. Nine held-out instances and short training cannot
establish general architecture superiority or robust unseen-instance feasibility.

The shared teacher succeeds on six of nine training cases (13 of 27 searches),
supplying 45 selected path decisions. Parameter counts are 12,545 for the MLP,
25,216 for the recurrent pointer, 47,825 for the bipartite GNN, and 29,697 for
the Transformer. Missing teacher coverage on three training cases is shared by
all architectures; RL still samples those cases. Mean combined imitation/RL/replay
training time per seed is 6.943 seconds (MLP), 58.182 (pointer), 58.952
(bipartite GNN), and 11.550 (Transformer). These CPU timings include optimizer
work outside the search-trajectory allowance and are not matched budgets.
Grouped rewards vary in 41/48 MLP groups, 41/48 pointer groups, 44/48 GNN
groups and 40/48 Transformer groups. Training succeeds on 99/192, 82/192,
74/192 and 89/192 searches respectively. Thus missing held-out RL gains cannot
be attributed simply to absent reward variation. Training caps remain a
limitation: the MLP reaches 64 decisions on every training search.

The run archives source hashes, data, protocol, teachers, all construction traces,
training costs, checkpoints, evaluations, solver evaluations, optimum references,
and contrasts. Auditing checks all final/complete schedules and placed prefixes,
deadline accounting, checkpoint/input hashes, memoization, teacher snapshots
against their reconstructed pre-placement state, and bounded rejection replay.
Solver status and timing are recorded evidence rather than independent proofs.

Run a new comparison into a new directory:

```bash
OMP_NUM_THREADS=1 /home/seman/dev/foreblocks/.venv/bin/python research/run_recovery.py \
  --profile-snapshot research/results/rl-construct-graph-20261011-fresh \
  --study-module rl_architecture_study --output /tmp/onts-architecture-study \
  --data-seed 2026201500
```

The launcher uses the hash-verified NumPy profile backend, isolating the study
from concurrent edits to the workspace's Numba backend. Audit frozen sources:

```bash
/home/seman/dev/foreblocks/.venv/bin/python research/audit_rl_construct_frozen.py \
  --module audit_rl_recovery --output /tmp/onts-architecture-study
```

See the completed pilot report in `results/rl-architectures-20261011/REPORT.md`.

## Completed pilot and decision

MILP finds the certified optimum on **9/9** instances with mean total runtime
**0.067 seconds**, well within the shared two-second allowance. Same-size and
larger means are 0.036 and 0.130 seconds. The neural policies have 27 attempts
per arm (nine instances, three seeds), so their denominators differ from MILP.

| Network | Initial imitation feasible | RL feasible | RL optimum hits | Larger RL feasible |
|---|---:|---:|---:|---:|
| Pooled MLP | 22/27 | 20/27 | 15/27 | 3/9 |
| Recurrent pointer | 19/27 | 14/27 | 11/27 | 1/9 |
| Task-time bipartite GNN | 19/27 | 18/27 | 15/27 | 0/9 |
| Transformer | 19/27 | 17/27 | 13/27 | 1/9 |

Uniform recovery succeeds 12/27 and hits nine optima. MLP imitation is the best
observed neural recipe: 22 feasible outputs and 17 optimum hits. Additional
replay reduces MLP feasibility to 19/27; RL reaches 20/27. Its pooled RL-minus-
replay utility difference is 0.038298, with marginal interval
[-0.078704, 0.155016]. Pointer, GNN and Transformer differences are negative,
and their pooled intervals also include zero. There is **no established
RL-specific advantage for any tested architecture**. The GNN's zero larger
successes make it an unsupported replacement for the MLP under this protocol.

Retain MILP as the exact-solver baseline and the pooled MLP as the economical
neural baseline. These results support those practical choices on this pilot;
they do not establish a universal best network. Longer training, broader
non-planted distributions, additional runtime budgets, capacity controls and
profile-menu coverage diagnostics remain necessary before a SOTA/generalization
claim. Checkpoints were fixed before evaluation; the descriptive ranking does
not turn these instances into a fresh confirmation set for future tuning.

The strict frozen-source audit passes: 1,146 final construction records,
7,394 complete schedules, 73,477 placed prefixes, 99 teacher snapshots,
36 checkpoints and nine solver baselines. It replays 512 unique rejection states
(19,941 repeated events); 374,579 other rejection events are not recomputed.
The one evaluation overrun is Transformer imitation, seed 47, larger balanced
instance (+0.488689 seconds); it counts as a failure. All neural/RL recovery
checks and deadline regression checks pass.

Descriptive tables, conditional optimum gaps and a comparison plot are stored
in the result folder's `analysis/`. The renderer's source and exact input hashes
are archived there separately from the frozen experiment sources. Render with
an environment containing NumPy and Matplotlib:

```bash
MPLCONFIGDIR=/tmp/onts-architecture-mpl /usr/bin/python3 \
  research/render_architecture_results.py \
  --output research/results/rl-architectures-20261011 --paper paper2
```

Paper 2 includes the architecture protocol, measured outcomes, table and figure;
its abstract and conclusion report the lack of an established RL benefit.
