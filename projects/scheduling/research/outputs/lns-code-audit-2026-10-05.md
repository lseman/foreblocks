# Author-code audit: LNS baselines for ONTS
Audit date: 2026-10-05. Static inspection only: no third-party training, solver run, checkpoint loading, or performance replication was performed. Files were fetched with the GitHub connector at the exact commits below. Initial web access had cache misses; connector reads resolved them.

## Revisions and scope

| Repository | Pinned revision | Commit date | Files inspected |
|---|---|---|---|
| coin-or/balans | `9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af` | 2026-08-13 | solver.py, solver_scip.py, base_instance.py, base_state.py, utils.py, destroy/local_branching.py, destroy/rins.py, repair/repair.py, configs/default.json, pyproject.toml |
| NetSysOpt/TLNS | `2998677d23c40591586cb05a0a3beea770235572` | 2026-03-24 | Inference.py, data_collection.py, GCN.py, helper.py, train.py, run.sh; complete tree |
| Liwow/Constraint_Matters | `2adbadc6df748b109561db1ffc3307b2e90668f1` | 2026-04-21 | model_test.py, model_test_SCIP.py, gp_tools.py, model.py, config.py, LICENSE, README; complete tree |

These are current repository snapshots, not necessarily the precise revisions used for the published tables. Commit dates do not establish paper publication dates. The supplied comparisons establish a small gain over uniform selection, not superiority over this baseline set.

## Balans: a stronger nonneural control

`Balans.solve` constructs an ALNS search and delegates operator selection and acceptance to the ALNS package. The constructor defaults to a Softmax MABSelector with four outcome rewards [8,4,2,1] and HillClimbing; simulated annealing and other acceptance policies are supported. This is materially different from ONTS's UCB reward based on realized gain per runtime. Maximization is converted to minimization internally and restored for output. Global deadline setup is present. [solver.py:347–500](https://github.com/coin-or/balans/blob/9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af/balans/solver.py#L347)

The checked JSON selects crossover, mutation, RINS, RENS, proximity and four local-branching radii. Its settings are total 60 s, initialization 10 s, ordinary repair 12 s and local branching 15 s. The constructor fallback is 300 s, so reporting “the Balans default” without recording how configuration was supplied is ambiguous. [default.json](https://github.com/coin-or/balans/blob/9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af/balans/configs/default.json)

`_local_branching` samples a radius between 5% and its configured maximum fraction of binary variables. This is a Hamming-ball constraint rather than a predefined release mask. [local_branching.py:8–25](https://github.com/coin-or/balans/blob/9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af/balans/destroy/local_branching.py#L8)
`_rins` uses disagreement with a cached root LP solution and samples indices; the NumPy choice call omits replace=False, then converts to a set, so realized cardinality may be smaller than requested. [rins.py:8–29](https://github.com/coin-or/balans/blob/9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af/balans/destroy/rins.py#L8)

Repair delegates to state.solve_and_update. [repair.py](https://github.com/coin-or/balans/blob/9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af/balans/repair/repair.py)
`_Instance.solve` caps repair time against remaining time, retains the previous state if no solution is returned, and reevaluates the original objective after objective-changing moves. Root LP extraction precedes initialization. [base_instance.py:45–156](https://github.com/coin-or/balans/blob/9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af/balans/base_instance.py#L45)
The SCIP backend fixes every supplied variable outside the skip set, including continuous variables. Root LP solve_and_undo has no explicit deadline cap. These details require checking when transferring battery SoC variables and enforcing a hard outer budget. [solver_scip.py:71–114,278–312](https://github.com/coin-or/balans/blob/9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af/balans/solver_scip.py#L71)

License is Apache-2.0; packaged dependencies include ALNS, MABWiser, NumPy, PySCIPOpt and gurobipy, without exact version pins. A SCIP-only run should still record installation dependencies and solver version. [pyproject.toml](https://github.com/coin-or/balans/blob/9dbf48fb8d14711ce3bc6dc2f60e39b2adbe55af/pyproject.toml)

## TLNS: actual nested solve and teacher costs

`double_LNS_scip` selects an outer release set, fixes its complement, writes a presolved model, then repeatedly repairs inner release sets. Transformed variables map back by names. Stagnation expands radii. Initial feasibility uses Gurobi SolutionLimit=1 without TimeLimit; model loading and original graph extraction precede the recorded timer. Inner solves have a cap, but inner stopping checks penalties rather than the global deadline. The executable hardcodes TLNS total 1000 s / inner 5 s and single LNS total 1000 s / repair 50 s, bypassing CLI timelimit for these methods. The mapping block also contains a self-comparison of a lower bound, a static concern requiring execution-level verification. [Inference.py:19–270,488–503](https://github.com/NetSysOpt/TLNS/blob/2998677d23c40591586cb05a0a3beea770235572/Inference.py#L19)

Selection samples variables without replacement using sigmoid scores plus a uniform floor; it does not rank a finite candidate-neighborhood pool. Current incumbent values are appended to variable features. [helper.py:33–51](https://github.com/NetSysOpt/TLNS/blob/2998677d23c40591586cb05a0a3beea770235572/helper.py#L33)
SGT combines fast global attention with a bipartite GNN and variable output head. The forward path forces its mask to cuda:0 despite CPU device fallbacks elsewhere. This is a portability defect visible statically. [GCN.py:575–801](https://github.com/NetSysOpt/TLNS/blob/2998677d23c40591586cb05a0a3beea770235572/GCN.py#L575)

The teacher performs local branching with a pool of up to 500 solutions, then 500 perturbed neighborhood repairs capped at 60 s each for every improving state. The local-branching default cap is 2000 s. Thus teacher generation may consume hours per state; a short ONTS teacher is an adaptation, not a matched replication. [data_collection.py:18–149](https://github.com/NetSysOpt/TLNS/blob/2998677d23c40591586cb05a0a3beea770235572/data_collection.py#L18)
Contrastive training separates masks above 60% and below 10% of reference improvement; imitation also exists. The published script accesses undeclared args.weightnorm, supplies five arguments to a four-argument train function and hardcodes a CUDA BCE weight. These are static reproduction blockers, not evidence the paper results are wrong. [train.py:15–187](https://github.com/NetSysOpt/TLNS/blob/2998677d23c40591586cb05a0a3beea770235572/train.py#L15)

The complete pinned tree contains no LICENSE or environment lockfile. Dependencies advertised by authors include PyTorch, PyG, SCIP, Gurobi and pytorch-metric-learning. Reimplementing ideas with citation avoids assuming permission to redistribute source. [pinned tree](https://github.com/NetSysOpt/TLNS/tree/2998677d23c40591586cb05a0a3beea770235572)

## Constraint Matters: newer restriction differs from a simple equality replacement

The current code retains original inequalities and adds indicator equalities for selected rows: z=0 implies lhs=rhs, with sum(z) ≤ Delta_c (default 5). Up to that many predictions may remain slack. Therefore “replace all predicted inequalities with equalities” misses the released implementation's trust region. The restricted model may exclude the true optimum, and its solver bound is not an original-model global bound. [model_test.py:95–200](https://github.com/Liwow/Constraint_Matters/blob/2adbadc6df748b109561db1ffc3307b2e90668f1/model_test.py#L95)
The SCIP version adds opposite-side indicators to existing inequalities and the same indicator budget. [model_test_SCIP.py:76–115](https://github.com/Liwow/Constraint_Matters/blob/2adbadc6df748b109561db1ffc3307b2e90668f1/model_test_SCIP.py#L76)
A second variable neighborhood limits summed absolute deviations from predicted binary values. [gp_tools.py:22–50](https://github.com/Liwow/Constraint_Matters/blob/2adbadc6df748b109561db1ffc3307b2e90668f1/gp_tools.py#L22)
The common configured solver budget is 800 s with one thread. [config.py:21–23](https://github.com/Liwow/Constraint_Matters/blob/2adbadc6df748b109561db1ffc3307b2e90668f1/config.py#L21)

The neural model connects structural features with class-specific semantic anchors. AnchorGNN hardcodes API mode true; text embedding code has blank API credentials/base URL and an optional local T5 path. The Gurobi evaluation imports GCN.postion_get, while GCN.py is absent from the complete pinned tree. Environment YAML mentioned by the README is also absent. The source therefore needs preparation before reproduction; no external embedding calls were made in this audit. [model.py:484–602](https://github.com/Liwow/Constraint_Matters/blob/2adbadc6df748b109561db1ffc3307b2e90668f1/model.py#L484), [model_test.py:1–3](https://github.com/Liwow/Constraint_Matters/blob/2adbadc6df748b109561db1ffc3307b2e90668f1/model_test.py#L1), [tree](https://github.com/Liwow/Constraint_Matters/tree/2adbadc6df748b109561db1ffc3307b2e90668f1)
Its license is MIT. [LICENSE](https://github.com/Liwow/Constraint_Matters/blob/2adbadc6df748b109561db1ffc3307b2e90668f1/LICENSE)

## Implementation and comparison recommendations

These are proposed ONTS adaptations, not tested improvements:

1. Add independent local branching and LP-disagreement RINS baselines. Hamming distance should concern scheduling x variables; allow auxiliary startup and continuous SoC values to adjust. Keep every original row and validate returned schedules.
2. Add a genuine ALNS portfolio control: multiple neighborhood radii, both random and LP-guided operators, and outcome-category Softmax or AlphaUCB selection. Record acceptance policy. Calling ONTS's four-operator UCB mode “Balans” would be inaccurate.
3. Put all model building, LP solving, inference, presolve and initialization inside one measured deadline. Compare native solver defaults and documented aggressive settings at identical threads, backend and wall time; separately report backend changes.
4. If implementing nested LNS, use a safe original-variable mapping and cap every inner repair against remaining time. Preserve a validated incumbent on failures. Test mapping independently before neural integration.
5. Consider semantic row selection only as a reversible restricted repair. Preserve the incumbent as fallback, distinguish its bounds, and tune indicator allowances on development data. ONTS energy balance equalities and run/startup logic need different handling from packing rows.
6. Train prospective neighborhood preferences using solver-measured outcomes including failures and zero gains. Match teacher generation cost across learned variants or disclose unequal training compute; small-budget ranking is not TLNS teacher replication.
7. Freeze variants and hyperparameters before a new test set. Report cluster-aware paired uncertainty over instance identities, training seeds, practical margins and multiplicity. More literature citations cannot turn an inconclusive heuristic comparison into statistical superiority.

## What this audit establishes

The author repositories provide concrete stronger controls and transferable mechanisms. They do not establish an ONTS result. A defensible SOTA statement needs measured superiority against the relevant reproducible baselines under a declared ONTS distribution and resource budget, with separate evidence for native cross-backend solvers.

