# Recent neural scheduling methods and an ONTS improvement path

Search cutoff: **5 October 2026**. This review prioritizes primary proceedings,
author manuscripts and author-linked code. It distinguishes published results
from preprints, and reported benchmark performance from proposed ONTS transfers.

**Recommendation:** retain the validated MILP repair and replace the current
changed-variable imitation objective with learning to rank complete candidate
neighborhoods. Next, test constraint-family semantics and search history. Use
an adaptive mixture as a strong comparator. A larger GNN alone is not the best
supported first intervention.

The recent literature contains strong methods for MILP and job-shop scheduling,
but none of the shortlisted papers demonstrates superiority on our exact
recurring job/time, solar-and-battery formulation. The ranking below concerns
transferability, not a universal state-of-the-art leaderboard.

## Most relevant published work

| Work | Verified status | Mechanism worth testing | Official code | ONTS priority |
|---|---|---|---|---|
| TLNS | LION 2025; proceedings first online 2 Jan 2026 | Contrastive neighborhood quality; lightweight global graph attention; nested repair | [NetSysOpt/TLNS](https://github.com/NetSysOpt/TLNS) | High: training; later: nested search |
| Constraint Matters | ICLR 2026 | Numerical instance graph plus abstract variable/constraint categories | [Liwow/Constraint_Matters](https://github.com/Liwow/Constraint_Matters) | High: semantic representation |
| MIStar | NeurIPS 2025 | Memory-enhanced complete-schedule improvement policy | [L2S-FJSP](https://github.com/wplus0077-dev/L2S-FJSP) | High: search history |
| BOPO | ICML 2025 | Best-anchored, objective-guided preference training | [L-Z-7/BOPO](https://github.com/L-Z-7/BOPO) | High: outcome preferences |
| ReSched | ICLR 2026 | Simplified residual scheduling state; edge-aware Transformer | [XiangjieXiao/ReSched](https://github.com/XiangjieXiao/ReSched) | Medium: attention ablation |
| BTBS-LNS | ICLR 2025 | Variable–constraint–objective graph; revisiting incorrectly fixed variables | Not verified in this search | Medium: stagnation recovery |
| Balans | IJCAI 2025 | Online bandit selection among complementary LNS operators | [coin-or/balans](https://github.com/coin-or/balans) | High: adaptive non-neural comparator |

### TLNS: the closest training and solver-search reference

TLNS searches a broad outer neighborhood through smaller inner neighborhoods,
with presolve and a lightweight graph Transformer. Its contrastive variant
learns from neighborhood quality rather than only a single successful variable
assignment. The paper evaluates public MILP benchmarks and reports improvements
over single-layer LNS and solver baselines under its experimental conditions.
Its long evaluation and teacher budgets do not establish effectiveness at our
one-to-three-second deadlines. Presolve/postsolve mappings and backend APIs also
matter for faithful nested-search replication.

The first transfer should be comparative neighborhood supervision. Nested repair
and global attention should be separate experiments. The conference year is
2025, while the authoritative proceedings publication date is 2 January 2026.
[Publisher](https://link.springer.com/chapter/10.1007/978-3-032-09156-7_15),
[author manuscript](https://arxiv.org/html/2412.08206v1).

### Constraint Matters: represent what each constraint means

This ICLR 2026 method combines numerical formulation information with abstract
variable/constraint categories to predict critical tight constraints. Its model
reduction turns selected inequalities into equalities. That is a restriction of
the search space, and wrong predictions can exclude feasible or better solutions.
Its implementation includes language-model category embeddings and solver/GPU
dependencies.

For ONTS, transfer category information first: startup, run length, execution
window, periodicity, instantaneous power and battery transition. Lightweight
learned category embeddings are an adaptation to test, not a replication of the
paper's multimodal pipeline. Preserve every original constraint and keep the
validated incumbent. Avoid predicted equality fixing as a supposed equivalent
reformulation.
[Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/25dd41a34a3fee27ca94ebd34186d22c-Abstract-Conference.html),
[revised manuscript](https://arxiv.org/html/2508.18742v3).

### MIStar: search memory rather than repeated blind decisions

MIStar learns improvement heuristics for flexible job-shop schedules using a
heterogeneous disjunctive graph, memory of visited states, local moves and
parallel proposal evaluation. The published experiments compare neural and
handcrafted scheduling methods. This is conceptually closer to our improvement
loop than a purely constructive scheduler.

Transfer recent neighborhood overlap, last repair outcomes, stagnation and
remaining time into selection. Its operation/machine moves do not directly map
to ONTS job/time binaries. Parallel proposal evaluation must be charged to the
same CPU and wall-clock allowance as its controls.
[NeurIPS 2025 paper](https://papers.neurips.cc/paper_files/paper/2025/file/77629f2df5ef97ad967157c69cd8af7b-Paper-Conference.pdf).

### BOPO: learn objective preferences

BOPO uses best-anchored preference pairs with objective-guided weighting, tested
on JSP, FJSP and TSP. It is an architecture-independent training contribution,
rather than a specific GNN that can simply replace ours.

For ONTS, use repaired objective outcomes to compare alternatives from the same
incumbent. A candidate-neighborhood ranker is a BOPO-inspired adaptation; it is
not BOPO itself. BOPO's sequence/action likelihood formulas should not be copied
onto independent variable labels without defining the actual neighborhood
policy distribution.
[ICML 2025 proceedings](https://proceedings.mlr.press/v267/liao25a.html).

### ReSched: recent Transformer scheduling architecture

ReSched uses a residual-subproblem formulation with four state features and
scheduling-specific attention. The ICLR 2026 experiments report improvements
over neural FJSP baselines and transfer to related scheduling variants. It is a
strong contemporary architecture comparator, with published author code.

Its simplified state depends on the FJSP formulation. ONTS still needs battery
state, startup counts, active run boundaries and periodic deadlines; four
features are not automatically sufficient. Best-of-many sampled schedules and
large synthetic training sets also require explicit compute accounting. Test
sparse typed attention before a dense Transformer over every MILP variable and
constraint.
[Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/467e109270678b3e2b3c2e7fa6b3d575-Abstract-Conference.html),
[author manuscript](https://arxiv.org/abs/2603.07020).

### BTBS-LNS: objective context and recovery

BTBS-LNS combines a variable–constraint–objective graph with attention and an
additional branching policy that revisits fixed decisions. Its integer
binarization machinery is less relevant to our already binary execution and
startup variables.

Test a global objective/budget context and a recovery operator that releases
decisions outside a repeatedly unsuccessful neighborhood. Stronger supervision
and additional search must be accounted for; the paper's reported wins do not
guarantee gains under our budgets.
[ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/7eca17ef54789b0663cab421f2e9dbf5-Paper-Conference.pdf).

### Balans: a necessary adaptive comparator

Balans selects complementary destroy/repair operators with an online bandit and
requires no offline neural training. Its modular implementation integrates MIP
solvers and adaptive LNS components. The published evaluation uses substantially
longer initialization, repair and instance budgets than our pilot study.

Adapt a small portfolio containing random, heuristic, learned, whole-run and
energy-bottleneck neighborhoods. Compare uniform mixing and a validation-selected
static mixture before a bandit. Online learning may receive too few trials at
three seconds. Improvement per second is a proposed ONTS utility, not a claim
that Balans uses that exact reward. An adaptive portfolio is not a GNN
contribution by itself.
[IJCAI 2025 proceedings](https://www.ijcai.org/proceedings/2025/286).

## Newer preprints and extensions

- **SPL-LNS**, submitted 22 August 2025: locally informed sampling, exploratory
  search states and hindsight relabeling. It distinguishes the current state
  from the best incumbent. Faithful solution-pool sampling needs a backend
  beyond SciPy's single returned solution or additional solves. Published venue
  and official code were not verified in this search. Useful later for escaping
  stagnation; training/repair costs are material.
  [Primary manuscript](https://arxiv.org/html/2508.16171v1).
- **Regularized LNS**, submitted 1 June 2026: regularized local optimization as
  a structured learning layer, with Gibbs-sampling and Fenchel–Young connections.
  Experiments include assignment and stochastic vehicle scheduling. Exact Gibbs
  claims require local marginal oracles and ergodicity; practical perturbation
  introduces structural bias. Our time-limited MILP repair is not an exact
  marginal oracle. Relevant research direction, not a near-term drop-in solver.
  [Primary manuscript](https://arxiv.org/html/2606.02294v1).
- **GOAL**, submitted 18 May 2026: objective-conditioned relational graph
  diffusion for several scheduling variants. The paper reports promising
  small-instance results; official implementation was not verified. Its
  precedence decisions and decoder differ from ONTS's recurring execution and
  battery constraints. Treat it as a watchlist item rather than our principal
  baseline.
  [Primary manuscript](https://arxiv.org/html/2605.19119v1).
- **DCAN**, ICLR 2026: preference-conditioned dual attention for multi-objective
  FJSP. Relevant only if we formally add competing objectives; battery and energy
  are currently constraints. Author code and accepted status are verified, but
  the complete OpenReview PDF was inaccessible to the reviewer, so no numerical
  superiority claim is used here.
  [Paper](https://openreview.net/forum?id=YAgOaYedLQ),
  [author code](https://github.com/ai-for-decision-making-tue/Neural_Multi-Objective_CO_for_FJSP).
- **Stochastic FJSP**, AAAI 2025: attention over stochastic scenarios augments
  scheduling policies. Potential inspiration for uncertain solar forecasts;
  this would require a new stochastic problem definition and validation contract.
  [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34870),
  [author code](https://github.com/ai-for-decision-making-tue/NCO-for-Stochastic-FJSP).
- **Curriculum GNN scheduling**, submitted 23 September 2026: very recent
  evidence on growing training sizes. Its reported training costs are far beyond
  our local study budget. Relevant to generalization, with preprint status and
  benchmark-specific findings.
  [Primary preprint](https://arxiv.org/abs/2609.28085).

For experimental design, **DynaSchedBench, ICML 2026** advocates calibrated
dynamic scheduling difficulty and shows that elaborate agents can fail to beat
strong dispatching rules. Its lesson for ONTS is difficulty stratification and
strong controls, not a direct transfer of its dynamic job-shop generator.
[Official proceedings](https://proceedings.mlr.press/v306/cao26z.html).

## What our implementation currently learns

The current source was inspected through the codebase graph and exact snippets;
coverage metadata matches for the four relevant modules.

1. [learning.py](../../learning.py) samples random neighborhoods and keeps only
   improving repairs. Its target is the variables that changed in the repaired
   solution, not the released neighborhood. It records gain but does not use gain
   in the BCE objective. Unsuccessful alternatives do not become training examples.
2. [neighborhood.py](../../neighborhood.py) already uses actual coefficients,
   incumbent values, numerical slacks and future-average solar. It has generic
   constraint features and mean aggregation. It is not missing all constraint
   information; the proposed additions are explicit family semantics and history.
3. [hybrid.py](../../hybrid.py) ranks scores and releases contiguous fixed-size
   blocks to a cell budget. Successful coordinated moves may need complete runs,
   startup relocation or simultaneous changes across competing jobs.

The completed study used 12 training identities. Neural LNS averaged +3.39%
versus MILP at three seconds on 12 IID identities, with adjusted p=0.1714, and
slightly lost to random/heuristic LNS. These observations motivate hypotheses;
they do not prove whether supervision, data volume or representation caused the
result. [Completed evidence](../results/RESULTS_REVIEW.md).

## Concrete next design

This section is our synthesis and proposed ONTS adaptation, not a method already
shown to work by the cited papers.

**First intervention: candidate-neighborhood ranking.** Generate a small diverse
set of masks from one incumbent, varying operator and coupled temporal region.
During training, repair each candidate with the same cap; save gain, actual
runtime, status and stagnation, including zero-gain outcomes. Learn a pairwise or
listwise ranker over incumbent–mask pairs. At inference, encode the state once,
score candidate masks cheaply, and solve only the chosen candidate. Do not spend
multiple repair calls selecting a neighborhood unless controls receive the same
budget.

Maintain separate gain and cost predictions or compare equal-cap realized gain
initially. A raw gain/time ratio can overfavor tiny cheap improvements and miss
an expensive escape. Time-limited zero gain is a censored observation, not proof
that a neighborhood cannot help. Use exploration and assess preference noise.

**Second intervention: temporal and energy coordination.** Release complete run
segments or startup relocation regions, with a bounded expansion to related
periodicity and energy competitors. Add explicit row categories and sparse
edge-aware aggregation. Auxiliary tightness prediction may help representation,
but must not silently impose equalities. Expansion is a proposal heuristic;
feasibility still comes from the original MILP and complete validator.

**Third intervention: history and selection.** Include recent mask overlap,
repair outcomes, time since improvement and remaining budget. Test static mixed
selection and a small contextual bandit; permit adaptive size/time only after
separating their effects. Keep the best validated schedule independently of any
exploratory current state. Do not promise the mixture matches its best constituent.

## Evaluation and paper claim

Use fresh training, validation and held-out identities. The completed test set
must remain an archived evaluation, not a tuning set. Preserve the current GNN,
random LNS, heuristic LNS and unrestricted MILP as controls. Add a simple non-GNN
candidate ranker, uniform candidate selection, static mixing and adaptive mixing.
Isolate ranking, semantics and history in ablations with comparable parameter
counts and proposal opportunities.

Charge initialization, graph construction, candidate generation, inference and
repair to the same wall-clock deadline. Include both short and longer budgets
when new compute is allocated. Report all-run feasibility and conditional
objective separately; use instance-level paired uncertainty. Keep the existing
AUC observability caveat until equivalent solver incumbent logging is available.
For backend changes, compare methods using the same backend and MIP-start access.

A plausible paper contribution is an **empirically demonstrated ONTS mechanism
that coordinates temporal and energy-coupled neighborhoods under tight
deadlines**. A combination of attention, preferences and bandits is an engineering
design until experiments establish why it helps. No improvement or
state-of-the-art claim follows from this literature search alone.
