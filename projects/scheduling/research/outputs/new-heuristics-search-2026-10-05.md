# New heuristic candidates for the ONTS paper

Search date: 5 October 2026. This note supplements
[strong-heuristic-references](strong-heuristic-references-2026-10-05.md),
[neural-scheduling-review](neural-scheduling-review-2026-10-05.md) and
[sota-claim-audit](sota-claim-audit-2026-10-05.md). It lists only methods those
files do not already cover. LP disagreement, RINS, local branching and the VNS
portfolio are already implemented in `classical.py` and are excluded here.
No external code was run. Performance reported in other papers is not ONTS evidence.

## Why the current method plateaus

The confirmatory study found a +0.28% gain over uniform selection, with no
established gain over the heuristic selector. Every current operator is
the same move: release a set of cells, then call HiGHS for an exact repair.
Better ranking over that pool has little headroom. Larger gains more likely come
from **new move types** whose cost structure differs: LP-free local search,
whole-job profile moves using ONTS decomposition, and proximity objectives.

## Ranked candidates

| # | Method | Source (year) | ONTS mechanism | Cost to build | Expected value |
|---|---|---|---|---|---|
| 1 | **Job-profile pricing neighborhoods** (our proposal) | ONTS BC&P pricing DP, C&OR 2023; constellation B&P, C&OR 2026 | Release 1–3 whole jobs. Re-optimize each job's activation profile with the existing per-job DP (startups, windows, period, run length), pricing energy with per-time multipliers from the LP or from battery slack. Then replay the battery exactly | Medium: DP is per job, O(T·run) | High. This move is structural, and a cell mask approximates it poorly. Each move needs no MILP call |
| 2 | **Weighted constraint local search (Feasibility Jump / Local-MIP)** | FJ, MPC 2024 (MIP 2022 competition winner); Local-MIP, CP 2024 | Single-cell and run-shift "jump" moves, plus Local-MIP's *lift* (use slack in satisfied rows to raise the objective) and *breakthrough* moves. Constraint weights adapt online. Use it as a cold-start initializer and as a cheap polishing operator between repairs | Medium. A pure NumPy kernel over the sparse rows already in `milp.py` | High for time to first feasible schedule and at short budgets. It also gives a non-MILP baseline that reviewers will expect |
| 3 | **Proximity search** | Fischetti & Monaci, 2014; listed but not implemented | Add a cutoff row obj ≥ z* + δ and minimize Hamming distance to the incumbent | Low. One row, one objective swap in `milp.py` | Medium. Complements local branching, which is already implemented |
| 4 | **Predict-and-search trust region (ConPaS / CoCo-MILP style)** | ConPaS, ICML 2024; CoCo-MILP, 2025; SRG, arXiv 2603.24033, 2026 | Reuse the ranker's GNN encoder to predict cell marginals. Fix confident cells inside a Hamming trust region Δ around the prediction and solve. Train contrastively: positives are high-objective schedules, negatives are low-objective or infeasible ones | Medium. Encoder already exists | Medium. This is SatGNN's own mechanism (trust region), so it is the most direct comparison with the published ONTS neural method |
| 5 | **Learned or adaptive local-branching radius** | Liu, Fischetti & Lodi, *Revisiting local branching with an ML lens* (2112.02195); RL-adaptive LB for log-truck scheduling (2607.20901, July 2026) | Predict k, or control it with a bandit, for the `local_branch` operator from search-memory features already in `SearchState` | Low | Medium. Reuses existing infrastructure. Gives the learning component a new decision beyond choosing masks |
| 6 | **LLM-evolved operators (EoH / ReEvo / MCTS-AHD / LLM-LNS)** | LLM-LNS, ICML 2025; MCTS-AHD, 2025; Hercules, 2025 | Evolve Python scoring functions for `make_candidates` or the DP pricing rule on training instances only, then freeze and test on fresh instances | Medium. Needs an API budget and an evaluator harness | Uncertain. Topical, but needs careful train/test separation. Good as an ablation, risky as the main claim |
| 7 | **GPU primal heuristic fusion (cuOpt FP+FJ+F&P; ZIB CHAP)** | Çördük et al., arXiv 2510.20499, NeurIPS 2025; CHAP, Land-Doig 2026 winner | Use as an external solver baseline (cuOpt heuristics-only mode) on the identical exported model | Low if only run as a baseline | Baseline only. ONTS instances are small, so GPU overhead probably dominates; report this honestly |

### 1. Job-profile pricing neighborhoods (recommended main contribution)

The ONTS decomposition literature already gives an exact DP for each job's
feasible activation profile. These are the pricing subproblems in the 2023
BC&P and 2026 constellation B&P papers. Using that DP as an **LNS move**
is not a reproduction of those exact methods, and we did not find it published:

1. Choose jobs to release. The learned ranker can score jobs instead of cell masks.
2. Compute energy prices π_t: use LP duals of the power/SoC rows when available,
   otherwise a battery-slack heuristic (π_t high where SoC is near its minimum).
3. For each released job j, run DP to maximize Σ_t (u_j − π_t·q_j) x_jt subject to
   j's own constraints. Insert the profile and replay the battery exactly with
   `validation.py`. If the replay fails, raise π on the violating times and retry
   (subgradient step), or fall back to an exact MILP repair over those jobs only.
4. Accept only validated improvements. This keeps the existing correctness gate.

Why it is promising: a whole-job release is exactly the case where cell masks
must release O(T) cells and pay for a full MILP. The DP costs milliseconds, so
far more moves fit in a 3 s budget. It also gives the paper an ONTS-specific
algorithmic idea rather than a generic MILP-LNS one.

Ablations: π source (LP duals, slack heuristic, zero), number of jobs released,
DP move with fallback versus MILP repair of the same job set.

### 2. Feasibility Jump / Local-MIP kernel

FJ minimizes weighted violation with Lagrangian-like weight updates and needs no LP.
Local-MIP (CP 2024) adds lift moves (feasible objective improvement using
row slack) and breakthrough moves, plus two-level scoring. It reports
better results than FJ, HiGHS, SCIP and CPLEX on MIPLIB, and results competitive with Gurobi. For ONTS:

- Variables: x_jt. Startups and SoC are derived. Replay the battery after each
  accepted move, or keep an incremental prefix of SoC.
- Moves: flip a cell, shift a run by ±1, extend or shrink a run, swap two jobs' runs.
- Uses: (a) cold start without a MILP solve, (b) polishing after each repair,
  (c) a standalone non-MILP baseline.

The open-source Local-MIP artifact (Dagstuhl, CP 2024) can also run directly
on our exported MPS file as an external baseline.

## Satellite-scheduling context to cite (not direct comparators)

- **EOS-Bench** (arXiv 2604.25782, April 2026, Yin et al.): Earth-observation
  scheduling benchmark with 13,900 instances, comparing MIP, greedy rules,
  GA/SA/ACO and a learned method. It shows the field is moving to standardized
  benchmarks. ONTS has none, so a released generator + instance set + validator is
  itself a contribution.
- **Realistic EO constellation scheduling** (NeurIPS 2025, arXiv 2510.26297): benchmark plus method.
- **Carruthers DRL scheduler** (ICAPS 2026): DRL deployed as the operational
  scheduler on a NASA mission, with activity-block macro actions and power/thermal action masking.
  Motivates run-level (macro) actions instead of cell-level actions.
- **DR-ALNS for imaging satellite scheduling** (JEIT, doi 10.11999/JEIT251009):
  DRL picks ALNS destroy/repair operators. It is the closest analogue to our
  operator-selection learning. The page could not be parsed, so read its details before citing.

## Suggested paper repositioning

The current evidence supports "a validated neural LNS with small gains over
uniform selection". To make a stronger and defensible claim:

1. Add #1 (job-profile pricing moves) and #2 (FJ/Local-MIP kernel) to the operator
   pool. Keep the ranker/bandit as the selector over the larger pool.
2. Add #3 and the SatGNN-style trust region (#4) as baselines. Pin SatGNN code.
3. Run external baselines on the exported model: SCIP default, Local-MIP binary,
   cuOpt heuristics-only (if a GPU is available), all under the same deadline.
4. Select on development instances, then freeze a new confirmatory protocol,
   as `CONFIRMATORY.md` already does.

## Sources

- Feasibility Jump: https://opus4.kobv.de/opus4-mpc/frontdoor/index/index/year/2024/docId/240
- Local-MIP (CP 2024): https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2024.19 ; software artifact https://drops.dagstuhl.de/entities/artifact/10.4230/artifacts.22500
- Local-ILP: https://arxiv.org/abs/2305.00188
- GPU primal heuristics (cuOpt): https://arxiv.org/abs/2510.20499
- 2026 Land-Doig competition (CHAP): https://www.zib.de/news/zib-team-wins-2026-land-doig-mip-competition
- ConPaS: https://proceedings.mlr.press/v235/huang24f.html ; CoCo-MILP: https://arxiv.org/pdf/2511.09209 ; SRG: https://arxiv.org/abs/2603.24033
- Revisiting local branching with ML: https://arxiv.org/abs/2112.02195 ; adaptive LB with RL: https://arxiv.org/abs/2607.20901
- LLM-LNS: https://proceedings.mlr.press/v267/ye25j.html ; MCTS-AHD: https://arxiv.org/abs/2501.08603
- EOS-Bench: https://arxiv.org/abs/2604.25782 ; EO constellation (NeurIPS 2025): https://arxiv.org/abs/2510.26297
- Carruthers DRL (ICAPS 2026): https://ojs.aaai.org/index.php/ICAPS/article/view/42871
- DR-ALNS satellite: https://jeit.ac.cn/en/article/doi/10.11999/JEIT251009
- ONTS BC&P: https://www.sciencedirect.com/science/article/abs/pii/S0305054823001569 ; constellation B&P: https://www.sciencedirect.com/science/article/abs/pii/S0305054825002886
