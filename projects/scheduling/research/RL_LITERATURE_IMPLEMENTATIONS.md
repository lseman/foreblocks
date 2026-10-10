# Executable literature-inspired ONTS alternatives

The new alternatives are adaptations to ONTS, not reproductions of original FJSP,
routing or generic-IP algorithms. The original methods use different actions,
state semantics, objectives, solver interfaces and training budgets. Paper2 reports
these differences rather than attaching the original algorithm names to new results.

| Implementation | Transferred mechanism | Deliberate ONTS differences |
| --- | --- | --- |
| `rl_resched.ReschedActorCritic` | Task self-attention, edge-conditioned query/key/value cross-attention and self connections. | ONTS task/time entities and 15/9/6 features; physical-time sinusoidal positions; concrete repair candidates; LayerNorm; global scorer. No operation-chain RoPE or four-feature FJSP sufficiency claim. Common PPO replaces the paper's main REINFORCE optimizer. |
| `rl_mistar.MistarActorCritic` | Retrieved state/action memory, repetition bias and several sampled repair proposals. | Binary schedule agreement instead of operation-order similarity; pooled task-group embeddings instead of action-ID voting; logits bias rather than similarity reward penalty. Sequential repairs from a frozen current state, not parallel Nopt2 moves. |
| `rl_lns.RLLNSActorCritic` | Actual variable/constraint coefficient graph and factorized destruction. | Includes x, startup and SoC variables; signed normalized coefficients and dynamic row slack. Pools x/startup embeddings into task blocks, rather than selecting individual IP variables. Exact nonempty-conditioned distribution. Common PPO replaces customized actor–critic. |
| `rl_alns.AdaptiveLNS` | Reactive roulette weights, segmented rewards and annealing. | ONTS profile, restricted MILP and randomized profile operators; task severity; fixed cooling. A classical adaptive baseline, not neural RL. |
| `rl_alns.DRALNSActorCritic` | Learned joint operator, destruction severity and temperature choices, following DR-ALNS. | Candidate scorer over ONTS actions and ten search-status features; no routing repair. Softmax PPO rather than copying original action factorization. |

Primary references: [ReSched](https://arxiv.org/html/2603.07020v1),
[MIStar](https://papers.neurips.cc/paper_files/paper/2025/file/77629f2df5ef97ad967157c69cd8af7b-Paper-Conference.pdf),
[RL-LNS](https://papers.neurips.cc/paper_files/paper/2021/file/fc9e62695def29ccdb9eb3fed5b4c8c8-Paper.pdf),
[DR-ALNS](https://ojs.aaai.org/index.php/ICAPS/article/view/31507).
The [literature review](outputs/rl-mip-scheduling-review-2026-10-06.md) records the
broader landscape; its provenance distinguishes source facts from recommendations.

## Shared search semantics

`LiteratureSearchEnv` retains feasible `current` and `best` schedules separately.
The best output never decreases. Profile and restricted-MIP operators optimize
from current and retain its quality. A third operator generates per-task profiles
under random signed rewards using the exact per-task DP, and accepts replacements
only after full ONTS validation. Thus genuinely worse feasible working states are
possible. Temperature controls acceptance of their normalized objective losses.
Simply weakening acceptance of an always-monotone repair would not provide this
exploration mechanism.

Candidates combine pressure/utility/random task groups of sizes 1, 2, ceil(J/3)
and ceil(J/2), three repair types and temperatures 0, .003, .015. Duplicate
(group, operator, temperature) entries are removed. The default MILP allowance is
.05 seconds. PPO chooses the whole categorical candidate. RL-LNS instead directly
generates any nonempty task subset and uses fixed restricted-MIP repair with
monotone acceptance. These action interfaces differ, so an algorithm ranking is
not an isolated architecture ablation.

MIStar draws four IID candidate indices with replacement. All repairs start from
the same frozen current schedule; the best timely feasible candidate controls the
working-state transition. Its training action retains **every ordered draw**, even
duplicates and unexecuted expired suffixes. Joint log probability is the sum of
all four categorical log probabilities; entropy is four times base entropy. PPO
never substitutes the likelihood of the best selected candidate. Inference omits
likelihood recomputation and trajectory storage. Memory is bounded to 32 records
and frozen into each observation; retrieval uses up to four records. Repetition
bias applies to task groups across repair types, an explicit simplification.

Rewards are undiscounted best-so-far increments, not current-quality differences.
Every candidate and every accepted current state is fully validated. Eligibility
is guarded before acceptance; late schedules do not improve output. Clock guards
are cooperative: a single DP or solver call can finish late, so measured overruns
must be reported instead of claiming a hard real-time guarantee. A .05-second or
5%-budget reserve covers final bookkeeping, subject to measurement.

## Training and evaluation

Run from repository root into a fresh directory:

```bash
ONTS_SOLVER_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m projects.scheduling.research.rl_literature_study \
  --output /tmp/onts-literature-study --episodes 32 --budget 1.2
```

The default development run uses nine training instances, three development
instances and six larger instances, two seeds, and fourteen predeclared methods.
Its planted instance family is disclosed and content-disjoint from earlier RL
runs. Teacher generation tries up to six profile and six MILP candidates per
training instance. Positive gains supply at most three labels; RL-LNS receives
MILP-specific labels. ReSched/MIStar/DR-ALNS share categorical examples. Eight
imitation epochs initialize each policy. PPO then runs 32 fresh episodes, each
with a .35-second allowance and at most four decisions. There is no additional
teacher replay during PPO, so fixed imitation is an initialization control.
Short capped training and longer uncapped deployment are different horizons.

Teacher generation can be more expensive than inference; costs and model sizes
are recorded. Policies use width24, gamma=lambda=1, three PPO epochs per four
training episodes, batch size16 and entropy coefficient .001. MIStar's joint
entropy naturally scales with four proposals. Every final checkpoint is fixed
before evaluation; the development set is diagnostic, not used to select models.

The fourteen methods are IL/PPO for each of four policies, classical ALNS,
uniform annealing, four-proposal random search, random nonempty subsets with MIP
repair, the **existing companion profile portfolio**, and full MIP. Classical
ALNS and uniform annealing share the same arms/cooling. The companion portfolio
has its original controller and .1-second repair allowance, so it is an external
algorithm baseline. Its returned late improvement can be conservatively withheld
from credited quality. Full-MIP late results are withheld entirely. Compare
credited quality and actual returned schedule quality separately when necessary.

A complete run preserves exact instances, teacher candidate schedules, training
outputs/transitions, sixteen safe model checkpoints, final schedules, method
order, source archive, hashes, timing and exploratory paired bootstrap intervals.
The independent audit is:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m projects.scheduling.research.audit_rl_literature \
  --output projects/scheduling/research/results/rl-literature-20261006
```

The audit replays every full schedule and proposal, fixed outside-group rows,
current/best transitions, rewards, affordable eligibility, source/checkpoint hashes,
evaluation coverage, means and intervals. It reports actual overruns. Full-MIP
has no individual proposal timestamp; the companion clock uses its own setup-
adjusted allowance. Those limitations are recorded rather than inferred away.

## API

```python
from projects.scheduling.rl_alns import LiteratureSearchEnv, AdaptiveLNS, literature_rollout
from projects.scheduling.rl_resched import ReschedActorCritic
from projects.scheduling.rl_mistar import MistarActorCritic
from projects.scheduling.rl_lns import RLLNSActorCritic

# Use a checkpoint trained on compatible ONTS data.
policy = ReschedActorCritic.load(checkpoint_path)
env = LiteratureSearchEnv(instance, feasible_initial_schedule, budget=2.0)
literature_rollout(env, policy, collect=False)
schedule = env.best

# MIStar uses the same categorical engine; RL-LNS uses subset=True.
# Classical adaptive LNS has no training requirement.
env = LiteratureSearchEnv(instance, feasible_initial_schedule, budget=2.0)
literature_rollout(env, adaptive=AdaptiveLNS(), collect=False)
```

Tests cover nonmonotone feasible transitions with monotone best output, late/invalid
rejection, frozen same-base batches, exact joint likelihood, graph coefficient
sensitivity, task permutation, masked/isolated attention, memory bounds, gradients,
PPO updates and safe checkpoints. Results are development evidence; publication
claims require larger untouched tests, more seeds, multiple runtime budgets,
realistic energy competition and reproduced published ONTS comparators.

Completed short run (`results/rl-literature-20261006`): 252 evaluations, 256
training outputs, 216 teacher repairs and 16 checkpoints. All returned schedules
pass independent feasibility and nondegradation replay; there are no evaluation
overruns. Larger-instance normalized improvements are 0.191446 for MIStar PPO,
0.190884 for DR-ALNS PPO and 0.189891 for the companion heuristic. Both learned
methods' paired differences from imitation and the heuristic include zero.
RL-LNS PPO reaches 0.026727; it needs repair-budget and neighborhood diagnosis
before architecture conclusions. This is preliminary development evidence.

The archived auditor remains unchanged. `audit_corrected.py` in the result
folder fixes two auditor assumptions: tiny strict objective improvements may
change the best schedule; evaluation setup can tighten the internal deadline.
`audit-correction.json` records the changes. Eligible proposals and acceptance
timestamps are checked against the conservative maximum cutoff, but exact
setup-adjusted cutoffs and best-update timestamps are unavailable. Full-MIP
proposal times are unavailable; companion credit is checked against history and
returned quality. See `independent-audit.json` for these explicit limitations.
A deliberately corrupted reward fails replay (`audit-negative-control.json`).
Use the corrected auditor for this completed frozen study:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python \
  projects/scheduling/research/results/rl-literature-20261006/audit_corrected.py \
  --output projects/scheduling/research/results/rl-literature-20261006
```
