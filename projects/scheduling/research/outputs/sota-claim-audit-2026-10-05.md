# ONTS state-of-the-art claim audit

Review date: 5 October 2026. Scope: primary references and public repository
documentation, plus the frozen confirmatory results. No external method was
executed in this audit; public code availability is not replication evidence.

## Current decision

The present scheduler is a modern ONTS matheuristic, but **state-of-the-art
performance and superiority over the default heuristic are unestablished**.
The confirmatory study establishes a 0.284% mean advantage over uniform candidate
selection in a fixed planted IID/OOD mixture at three seconds with common warm
starts. Its simultaneous interval against the heuristic crosses zero; the median
heuristic comparison is negative, with 19 wins, 40 losses and one tie. Increasing
the number of seed runs does not create new independent instances, and a
significant result against uniform selection does not establish superiority over
the best published method. [Frozen results](../results/ranker-confirmatory/RESULTS_REVIEW.md).

The earlier generic neural scheduling review omitted directly relevant ONTS
methods. This is a substantive literature coverage gap, not evidence that those
methods will beat the new ranker. Contemporary attention, history and preference
training are useful design ideas; borrowing them does not establish novelty or
a performance ranking.

## Direct ONTS references that must enter the comparison

| Reference | Verified scope and availability | Required action |
|---|---|---|
| **SatGNN: Graph neural networks for the offline nanosatellite task scheduling problem**, Applied Soft Computing 186C, January 2026, DOI 10.1016/j.asoc.2025.114139 | Direct ONTS: nonpreemptive execution, startup counts, windows, periodicity and energy management. Author manuscript v4 revised 17 October 2025. Official repository README identifies paper experiments and evaluation scripts. | Treat as the principal published neural comparator. Pin code/data/checkpoints; map battery equations, boundary behavior and objective before evaluating. |
| **Explainable column-generation-based genetic algorithm for knapsack-like energy aware nanosatellite task scheduling**, Applied Soft Computing 144, September 2023, DOI 10.1016/j.asoc.2023.110475 | Direct ONTS CGbGA: task-profile generation using dynamic programming and population search. Complete official implementation was not verified by this reviewer. | Obtain author code or document a tested paper-faithful reimplementation; do not label an unrelated generic GA as CGbGA. |
| **Improving energy aware nanosatellite task scheduling by a branch-cut-and-price algorithm**, Computers & Operations Research 158, October 2023, DOI 10.1016/j.cor.2023.106292 | Direct ONTS decomposition, dynamic-programming pricing, cuts and pseudocost branching. Published optimality/bound evidence is relevant beyond default monolithic MILP. | Include exact/decomposition incumbent and bound comparisons where reproducible; separate valid-cut audit from search improvements. |
| **A branch-and-price algorithm for energy aware task scheduling of constellations of nanosatellites**, Computers & Operations Research 185, January 2026, DOI 10.1016/j.cor.2025.107259 | Constellation model with satellite-specific and shared tasks; structured pricing/decomposition. | Relevant recent exact-method context. Benchmark only after defining and validating the single-satellite restriction. |
| **Nested branch-and-price for multi-mode nanosatellite task scheduling with interior-point regularization and GPU acceleration**, EJOR, DOI 10.1016/j.ejor.2025.05.020 | Multi-mode ONTS with strengthened formulations and GPU pricing. | Verify one-mode reduction before using as a comparator; disclose GPU/CPU and formulation differences. |

Primary sources: [SatGNN journal](https://www.sciencedirect.com/science/article/pii/S1568494625014528),
[SatGNN author manuscript](https://arxiv.org/abs/2303.13773v4),
[SatGNN official code](https://github.com/gos-ufsc/sat-gnn),
[CGbGA journal](https://www.sciencedirect.com/science/article/pii/S1568494623004933),
[BC&P journal](https://www.sciencedirect.com/science/article/abs/pii/S0305054823001569),
[constellation B&P journal](https://www.sciencedirect.com/science/article/pii/S0305054825002886),
[multi-mode NB&P journal](https://www.sciencedirect.com/science/article/abs/pii/S0377221725003881).

SatGNN's official repository links a public dataset and explains the generation,
training and evaluation pipeline. The earlier author URL redirects to
`gos-ufsc/sat-gnn`; pin the destination revision rather than assume the old URL is
a separate implementation. The journal's reported percentage improvements are
against its own SCIP setup, not our HiGHS run or our default heuristic. Cross-paper
percentages cannot rank these implementations. [Repository](https://github.com/gos-ufsc/sat-gnn).

## Adjacent energy scheduling: useful ideas, different claims

### SatGNN mechanism and parity audit: manuscript evidence

The manuscript describes confidence-selected partial warm starts, hard
early-fixing, and a Hamming trust region. It predicts both execution and startup
binaries. Early-fixing and trust-region restrictions do not retain unrestricted
optimality guarantees. Its solar/battery dynamics use equality updates, bounded
SoC and an instantaneous discharge-power cap; surplus treatment must be checked
against our spill-capable battery replay. It permits a late-started run to persist
to the horizon and uses sliding startup windows for period bounds. Indexing and
final SoC bounds still need exact source verification. Its data has horizon 125,
600 training identities with 9/13/18 jobs and 120 validation/test identities with
20/22/24 jobs; teacher collection uses five minutes and 500 solutions per instance.
[Author manuscript, sections 3, 5.4 and 6.1](https://arxiv.org/html/2303.13773v4).

Web raw/blob/API attempts initially failed and sandboxed curl could not resolve
the host. The installed GitHub connector subsequently provided exact source.
The following findings supersede the earlier code-access limitation. The
manuscript paragraph above retains its own evidence attribution.

### SatGNN pinned source audit

Audited revision: **`1fe490a44b420843a5b6fb8e6b9c1c4aae016f9f`**, default-branch
commit dated 18 March 2025. This predates manuscript v4; it must not be assumed
to include every final-publication revision. External repository coverage is
unavailable in the workspace graph; direct complete reads of the seven files
listed below supplied the evidence. No external code was executed.

- **Model parity:** both `Instance.to_scip` and `Instance.to_gurobipy` actually
  use C17 **`b[t] / v_bat >= i[t]`**, not the manuscript's current equality.
  This permits curtailment. SoC equality updates use initial state at t=0 and
  bounds on every post-step state, including the final step. Defaults are
  initial SoC .7, lower limit 0, efficiency .9, voltage 3.6, capacity 5 and
  battery usage 5. Window indices are `[win_min, win_max)`. Startup count,
  sliding minimum/maximum startup periods and terminal truncated runs are
  explicit. Default `new_inequalities=False`; the optional sixth cut is present
  and needs the same validity audit as the earlier original-model cut.
  [Pinned problem.py](https://github.com/gos-ufsc/sat-gnn/blob/1fe490a44b420843a5b6fb8e6b9c1c4aae016f9f/src/problem.py).
- **Decoder/search:** confidence sorts absolute distance from .5, thresholds
  selected predictions at .5, then maps them to binary-variable names. Warm
  starting uses SCIP partial solutions; early fixing calls `fixVar`; trust
  region adds absolute-difference auxiliaries and a summed deviation limit.
  `_get_prediction` constructs a Gurobi model and graph before neural inference.
  Returned runtime is SCIP `getSolvingTime()`, which excludes this model/graph
  preparation and inference. Equal end-to-end deadlines require an external
  timer rather than copying that runtime metric. The common unrestricted
  solver is SCIP; default solver controls must match that backend.
  [Pinned solver.py](https://github.com/gos-ufsc/sat-gnn/blob/1fe490a44b420843a5b6fb8e6b9c1c4aae016f9f/src/solver.py).
- **Evaluation budgets and identities:** `evaluate.py` defaults to **120
  seconds**. Test identities use indices 20–39 and glob `125_2*`; validation
  uses 0–19. Paths for raw/interim/results are hardcoded under `/home/bruno`.
  The shipped test script does not override the deadline. Its selected models
  are W&B run IDs `xae9giql` and `gbmooeqh`, with distinct warm-start/fixing/
  region sizes. The sweep script explores N=500–2000 and region radii 1,2,5.
  Reusing final test-selected sweep parameters would contaminate a new test;
  choose parameters on independent development data.
  [Pinned evaluate.py](https://github.com/gos-ufsc/sat-gnn/blob/1fe490a44b420843a5b6fb8e6b9c1c4aae016f9f/evaluate.py),
  [test commands](https://github.com/gos-ufsc/sat-gnn/blob/1fe490a44b420843a5b6fb8e6b9c1c4aae016f9f/run_evaluation_test.sh),
  [sweep commands](https://github.com/gos-ufsc/sat-gnn/blob/1fe490a44b420843a5b6fb8e6b9c1c4aae016f9f/run_heuristic_experiments.sh).
- **Reproduction dependencies:** neural checkpoints are restored from W&B
  `model_best.pth` at runtime, not bundled in the audited repository tree;
  their availability and hashes were not verified. Requirements pin PyTorch
  1.13.1 and DGL 1.0.1 with CUDA 11.7, import Gurobi, and omit PySCIPOpt despite
  its import in solver/model code. Training uses 150 epochs in this source,
  unlike the manuscript's stated 100-epoch budget. Source/data/checkpoint and
  solver-version differences must be recorded before claiming replication.
  [Pinned requirements](https://github.com/gos-ufsc/sat-gnn/blob/1fe490a44b420843a5b6fb8e6b9c1c4aae016f9f/requirements.txt),
  [training script](https://github.com/gos-ufsc/sat-gnn/blob/1fe490a44b420843a5b6fb8e6b9c1c4aae016f9f/train_best_models.py).

This source audit makes a direct comparison plausible. It establishes neither
checkpoint availability nor executed performance. Our original energy semantics
should remain intact; the manuscript equality is not justification to silently
change the benchmark model.

**HELIOS**, Journal of Systems Architecture 178, September 2026,
DOI 10.1016/j.sysarc.2026.103885, makes temporal and energy readiness joint
conditions, uses a table-driven runtime and limited-preemptive ILP planning.
Its primary article links [public code](https://gitlab.telecom-paris.fr/software/helios).
It is a current energy-aware comparator for a matching real-time model, not a
drop-in ONTS utility maximizer. Energy readiness can inspire proposals without
changing the feasibility oracle. [Primary article](https://www.sciencedirect.com/science/article/pii/S1383762126002031).

**CyEnSe: Cyclic energy-aware scheduling for energy-harvested embedded systems**
studies periodic energy arrivals, a heuristic and an ILP, with First-Fit as a
control. Cyclic execution and energy-storage assumptions require a formal mapping
to ONTS before its results can support a comparison. It motivates a strong
cyclic/whole-run placement heuristic rather than a generic task-by-task greedy
baseline. [Primary article](https://www.sciencedirect.com/science/article/pii/S0141933121005597).

**Interweaving Real-Time Jobs with Energy Harvesting to Maximize Throughput**,
Algorithmica 2025, DOI 10.1007/s00453-025-01331-x, gives approximation results in
a model where harvesting and task execution are mutually exclusive and processing
times are uniform. ONTS battery/solar operation is different. Its guarantees
cannot be imported into time-limited MILP repair. [Primary article](https://link.springer.com/article/10.1007/s00453-025-01331-x).

Generic FJSP papers, energy-aware manufacturing and Earth-observation request
selection remain architecture references unless a constraint-preserving mapping
is supplied. They do not replace the direct ONTS baseline suite.

## Minimum credible next evaluation

1. **Model parity first.** Publish an equation-level contract: objective units,
   startup/periodicity convention, legal windows, final-run truncation, initial
   and terminal battery state, efficiency, charge/discharge/power limits and
   concurrency. Verify external solver outputs independently. An adapter that
   changes these conditions defines a separate problem; report it separately.
2. **Strong baseline family.** Retain original corrected MILP, default heuristic
   LNS, random LNS, uniform candidate selection and old imitation. Add SatGNN,
   CGbGA, and BC&P or the strongest reproducible ONTS decomposition. Compare
   validation-selected static operator mixtures and adaptive LNS controls. Add a
   cheap non-GNN candidate ranker trained on the same outcomes, which tests
   whether representation complexity earns its inference cost. A restricted
   comparison with only available implementations must state that scope.
3. **Default solver controls.** Include native default primal heuristics in the
   same solver backend as neural integration, and a transparent external default
   solver comparison. Separate solver/version advantage from neighborhood
   selection advantage; give methods equal MIP-start access and thread limits.
4. **Fresh benchmark identities.** Freeze development data before a new test.
   Use public ONTS/FloripaSat identities where compatible, plus fresh unplanted
   and feasible/infeasible cases with solar/eclipse and battery stress. Include
   larger horizons and job counts. Existing held-out identities are now archived
   evidence, not new tuning data. Hash content to detect renamed duplicates and
   audit overlap with external training datasets/checkpoints.
5. **Two clearly separated tracks.** Common validated warm starts isolate
   improvement selection. Cold starts assess a complete scheduler, including
   time to the first feasible schedule. Count initialization, graph building,
   proposals, inference, repairs and logging inside the deadline. Feasibility
   failures must remain visible rather than disappear from objective means.
6. **Budget curves.** Preserve the three-second result, then preregister longer
   deadlines selected for the intended application, e.g. 10, 30 and 60 seconds.
   A bandit needs enough rounds to adapt; a graph model needs enough budget to
   amortize inference. Do not choose the winning deadline retrospectively.
7. **Mechanism ablations.** Retrain semantics/history/ranking ablations on
   matched training identities and teacher opportunities. Compare whole-run and
   energy-coupled proposals independently of the selector. Track parameter count,
   repair count, proposal/inference cost and training wall time. Removing a
   feature only at inference measures distribution shift, not its training value.
8. **Frozen inference.** Instances are independent units; seeds are repeated
   measurements. Prespecify the primary population, estimand, practical margin,
   comparator family and multiplicity adjustment. Report paired uncertainty,
   median effects, win/loss counts and strata descriptively. Use variance from
   development data for power planning, then freeze sample size. More data cannot
   guarantee the desired effect, and repeated testing until p<0.05 is invalid.

## Scope-specific defensible claims

- **Supported now:** an independently validated ONTS neighborhood ranker with a
  small mean objective improvement over uniform selection in the frozen
  synthetic, warm-start, three-second setting.
- **Unestablished now:** improvement over the default heuristic, practical >1%
  superiority, typical-instance superiority, cold-start advantage, public ONTS
  benchmark superiority, and ONTS or neural scheduling state of the art.
- **Possible after new evidence:** better deadline objective than named,
  reproduced ONTS baselines on a named benchmark and budget range, with
  simultaneous uncertainty and complete failure accounting. Prefer this exact
  claim to an unrestricted SOTA label.

This audit proposes evaluation requirements; it reports no new performance
experiment and does not change the frozen confirmatory algorithm or results.
