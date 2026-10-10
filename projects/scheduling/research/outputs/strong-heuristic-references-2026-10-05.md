# Strong heuristic and neural MILP comparators

Search cutoff: 5 October 2026. This is a targeted primary-source supplement to the existing neural scheduling review. Published results on other MILP families do not establish ONTS superiority. No external code was executed for this review.

## Most important missing controls

| Comparator | Why it matters | Portable ONTS mechanism | Execution requirements |
|---|---|---|---|
| Local branching | Changes can occur anywhere within a Hamming radius; contiguous destruction need not reach equivalent moves | Add a Hamming-ball constraint around the incumbent and solve with the original utility objective | One extra sparse row; existing MILP backend can support an adaptation |
| LB-RELAX / RINS | Uses optimization information, rather than only utility or random release scores | Release variables that disagree with an LP reference; LB-RELAX obtains that reference inside a Hamming ball | LP solve charged to the same wall-clock budget; preserve all original constraints |
| Proximity search | Searches for a small change that crosses an objective-improvement threshold | Max-utility cutoff plus Hamming-distance minimization, then optional utility polish | Alternate objective and cutoff row; restricted bounds are not global bounds |
| SCIP defaults / aggressive heuristics | A real solver baseline already includes sophisticated primal heuristics and adaptive LNS | Export the identical audited formulation and use the same incumbent, threads and deadline | SCIP/PySCIPOpt integration and solver version pinning |
| Full Balans / ParBalans | Its diverse solver-informed operators are stronger controls than a UCB selector over four existing masks | Use published operator portfolio, or label a smaller adaptation explicitly | Different backend; parallel versions require equal-core accounting |

The ONTS adaptations above are research recommendations, not reported implementations or replications. For local branching, use

\[
\sum_{x^*_i=0}x_i+\sum_{x^*_i=1}(1-x_i)\le k.
\]

Counting only job-time decisions is a reasonable ONTS distance; counting all integer variables, including startup indicators, is a different neighborhood and must be disclosed. Battery variables remain continuous and all energy/run/startup constraints remain active. Maximization changes the direction of proximity cutoffs. Run closure for LP release scores is a domain adaptation, not literal RINS.

## Primary references and inspected code

### LB-RELAX and CL-LNS: older, essential comparators

**Local Branching Relaxation Heuristics for Integer Linear Programs**, CPAIOR 2023, proposes using an LP relaxation of the local-branching subproblem to select released variables. The author project explains its relationship to contrastively trained CL-LNS. [Author project](https://taoanhuang.github.io/project/LNS_ILP), [paper](https://arxiv.org/abs/2212.08183).

The official **CL-LNS** repository is archived and documents SCIP, Ecole patches and older PyTorch environments. Its reference data collection uses expensive local-branching teachers; that recipe cannot be equated with our subsecond teachers. [Official repository](https://github.com/facebookresearch/CL-LNS).

In `LNS.py`, the inspected `LOCAL...RELAXATION` path constructs a binary Hamming ball, solves its LP, and selects either nonagreeing variables or variables ranked by absolute incumbent/LP difference. A randomized variant samples using clipped differences; multiple teacher samples invoke separate repairs. Reference limits include 300 seconds for the LP and 120 seconds for candidate repairs, far above our caps. A short-budget port should therefore be called **LB-RELAX-inspired** and measured independently. [Inspected source](https://raw.githubusercontent.com/facebookresearch/CL-LNS/main/LNS.py).

### Solver-integrated ALNS, RINS and classical neighborhoods

Hendel's **Adaptive large neighborhood search for mixed integer programming**, Mathematical Programming Computation 2021, describes a common-budget adaptive portfolio. The article states that SCIP's default ALNS includes techniques such as local branching, proximity and DINS alongside established RINS/RENS/crossover. Thus beating our custom `heuristic` selector is not equivalent to beating default SCIP heuristics. [Primary article](https://link.springer.com/article/10.1007/s12532-021-00209-7).

SCIP's inspected RINS implementation combines incumbent and LP optimum. Its plugin documentation distinguishes diving from sub-MIP heuristics and warns against recursive subsolver heuristics. Standalone SciPy masks do not inherit the solver's LP-node state, propagation or tree information. [RINS implementation documentation](https://scipopt.org/doc-9.2.3/html/heur__rins_8c.php), [SCIP heuristic integration](https://www.scipopt.org/doc/html/HEUR.php).

### LLM-LNS: ICML 2025, author code verified

**Large Language Model-driven Large Neighborhood Search for Large-Scale MILP Problems** is an ICML 2025 paper that evolves prompts and heuristic code. It reports improvements over several learned and classical competitors on its benchmarks; those are author-reported results, not ONTS evidence. [Conference record](https://proceedings.mlr.press/v267/ye25j.html).

The author's MIT-licensed repository contains problem-specific scripts for independent set, maximum independent K-set, vertex cover and set cover. Running heuristic evolution requires configuring an LLM endpoint and API key. Transfer requires an ONTS evaluator and generation budget; using generated code as a fixed heuristic still requires train-only selection and fresh tests. No paid API calls were made. [Author repository](https://github.com/thuiar/LLM-LNS).

### Apollo-MILP: ICLR 2025, author code verified

**Apollo-MILP** alternates predicted assignments and solver correction, using reference solutions and uncertainty to choose fixings. It targets solution prediction/reduction, a different decision from ranking a fixed set of neighborhoods. Useful transferable ideas are correction before committing fixings and explicit uncertainty. Empirical optimality retention is not an unconditional theorem for learned ONTS restrictions. [Paper](https://arxiv.org/abs/2503.01129).

The author repository lists Python 3.9, PyTorch 1.12.1, PyG 2.5.3, PySCIPOpt 4.2.0, Gurobi 9.5.2 and Ecole 0.8.1; it now points to integration in MILP-X. A compatible environment, solver license where needed, ONTS data and retraining are prerequisites. [Author repository](https://github.com/MIRALab-USTC/Apollo-MILP), [inspected requirements](https://raw.githubusercontent.com/MIRALab-USTC/Apollo-MILP/main/README.md).

### Fix-propagate-repair: 2025 journal issue

**A fix-propagate-repair heuristic for mixed integer programming** was published online October 2024 in a 2025 MPC issue. It adds a WalkSAT-inspired repair stage to propagation-based diving and emphasizes portfolio robustness. Its main role is finding feasible starts quickly rather than choosing an incumbent-improvement mask. ONTS has bounded binaries and continuous battery state, but a full propagation engine is a substantial implementation. Public source was not verified in this search. [Primary article](https://link.springer.com/article/10.1007/s12532-024-00269-5).

### Fresh July 2026 reference: approximate LP information

**Fix-and-propagate heuristics using low-precision first-order LP solutions for large-scale mixed-integer linear optimization**, published 6 July 2026, studies PDLP-based approximate LP guidance, including large energy-system models. The authors explicitly state that source is not publicly available; compiled binaries are obtainable on request. Its GPU/million-variable setting and hour-scale experiments differ greatly from our seconds-scale CPU study. Portable lesson: LP guidance need not be solved to high accuracy, but approximate points must not be represented as certified global bounds. [Primary article, methods and data-availability statement](https://link.springer.com/article/10.1007/s12532-026-00312-7).

### May 2026 GNN fix-and-optimize preprint

**Learning to reoptimize: a GNN-aided fix-and-optimize approach and an application to the Lot Sizing problem**, arXiv 2605.27339, studies disrupted solutions and repair followed by learned neighborhood reoptimization. It is relevant to future dynamic ONTS scheduling, but does not directly validate static scheduling superiority. Author code and a peer-reviewed venue were not verified. [Preprint](https://arxiv.org/abs/2605.27339).

## Decision for our next comparison

Prioritize a same-backend LP-disagreement control, local branching and proximity search, followed by solver defaults with explicit backend attribution. Pair every method on the same initial incumbent and independent instance, charge LP/inference/model-copy overhead, report cold-start performance separately, and select radii/repair budgets using training/validation data only. Tune a modern adaptive portfolio on validation before comparing it with the ranker; do not equate our four-mask bandit with Balans. [Balans official code and operators](https://github.com/coin-or/balans), [peer-reviewed record](https://www.ijcai.org/proceedings/2025/286).

The current significant result against uniform selection establishes only that narrow comparison. Broader superiority remains open until these controls, appropriate budgets and fresh confirmatory instances are evaluated. A paper can already describe a validated semantic neural heuristic; it cannot yet claim state-of-the-art ONTS performance.
