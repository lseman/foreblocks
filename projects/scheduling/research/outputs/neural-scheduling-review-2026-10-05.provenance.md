# Provenance: neural scheduling literature review

Review date and cutoff: 2026-10-05. All links were searched or inspected during
this session. No new optimization or model-training experiment was run.

## Workflow and scope

The installed `literature-review` skill specifies primary-source synthesis,
researcher/verifier/reviewer agents and an `outputs/` artifact with a provenance
sidecar. Its `/lit` expansion was not exposed as an executable interface here;
the work used the available web tools and three research/verification/review
agents directly. The parent synthesized and cross-checked key venue/date claims.

Search covered recent 2024–2026 learned MILP neighborhoods, GNN/Transformer
job-shop scheduling, neural improvement policies, preference learning,
energy/battery and satellite scheduling, and newer preprints. Queries included:

- `2025 2026 learning large neighborhood search mixed integer programming graph neural networks`
- `2025 2026 neural combinatorial optimization scheduling graph neural network state of the art job shop`
- `2025 learning to improve mixed integer programming large neighborhood search contrastive`
- `2026 large neighborhood search learning integer`
- `2026 scheduling neural ICLR`
- `2025 MILP neighborhood online graph`
- `2026 neural combinatorial scheduling`
- Exact-title searches for Constraint Matters, BTBS-LNS, Balans and SPL-LNS.
- Satellite/GNN/energy and battery/scheduling/GNN queries for 2025–2026.

This is a targeted implementability review, not an exhaustive bibliometric
survey. Search-result snippets were used for discovery; final technical
recommendations rely on primary papers, proceedings or author-linked code.
Secondary mirrors and promotional percentage claims did not establish rankings.

## Primary evidence ledger

| Work | Evidence inspected | Status and limit |
|---|---|---|
| TLNS | [Springer](https://link.springer.com/chapter/10.1007/978-3-032-09156-7_15), [manuscript](https://arxiv.org/html/2412.08206v1), [author code](https://github.com/NetSysOpt/TLNS) | LION 2025, publication 2 Jan 2026; manuscript/appendices support mechanism and budget cautions |
| Constraint Matters | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/25dd41a34a3fee27ca94ebd34186d22c-Abstract-Conference.html), [v3 manuscript](https://arxiv.org/html/2508.18742v3), [author code](https://github.com/Liwow/Constraint_Matters) | Published ICLR 2026; equality restrictions distinguished from equivalent reformulation |
| MIStar | [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2025/file/77629f2df5ef97ad967157c69cd8af7b-Paper-Conference.pdf), [author code](https://github.com/wplus0077-dev/L2S-FJSP) | Published 2025; flexible job-shop results, not ONTS |
| BOPO | [ICML proceedings](https://proceedings.mlr.press/v267/liao25a.html), [author code](https://github.com/L-Z-7/BOPO) | Published 2025; neighborhood-ranking proposal is an adaptation |
| ReSched | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/467e109270678b3e2b3c2e7fa6b3d575-Abstract-Conference.html), [manuscript](https://arxiv.org/abs/2603.07020), [author code](https://github.com/XiangjieXiao/ReSched) | Published 2026; state simplification relies on FJSP structure |
| BTBS-LNS | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7eca17ef54789b0663cab421f2e9dbf5-Abstract-Conference.html), [paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/7eca17ef54789b0663cab421f2e9dbf5-Paper-Conference.pdf) | Published 2025; official code not verified |
| Balans | [IJCAI proceedings](https://www.ijcai.org/proceedings/2025/286), [author code](https://github.com/coin-or/balans) | Published 2025; long-budget evaluations; not a neural model |
| SPL-LNS | [manuscript](https://arxiv.org/html/2508.16171v1) | Submitted 22 Aug 2025; published venue and official code not verified |
| Regularized LNS | [abstract](https://arxiv.org/abs/2606.02294), [full text](https://arxiv.org/html/2606.02294v1) | Submitted 1 Jun 2026; oracle/ergodicity assumptions inspected; implementation not verified |
| GOAL | [abstract](https://arxiv.org/abs/2605.19119), [full text](https://arxiv.org/html/2605.19119v1) | Submitted 18 May 2026; official code not verified; small and different formulations |
| DCAN | [OpenReview](https://openreview.net/forum?id=YAgOaYedLQ), [author code](https://github.com/ai-for-decision-making-tue/Neural_Multi-Objective_CO_for_FJSP) | ICLR 2026 accepted status/code verified; full OpenReview PDF inaccessible, numerical claims omitted |
| Stochastic FJSP | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34870), [paper](https://ojs.aaai.org/index.php/AAAI/article/download/34870/37025), [author code](https://github.com/ai-for-decision-making-tue/NCO-for-Stochastic-FJSP) | Published 11 Apr 2025; uncertainty is outside current deterministic scope |
| Curriculum GNN scheduling | [preprint](https://arxiv.org/abs/2609.28085) | Submitted 23 Sep 2026; review treats it as preprint evidence |
| DynaSchedBench | [ICML proceedings](https://proceedings.mlr.press/v306/cao26z.html), [author code](https://github.com/dsbx7/DynaSchedBench) | Published 2026; supports benchmarking design, not solver transfer |

Additional discovery included ConPaS (ICML 2024), DOAGNN (WWW 2025), 2026
energy-aware cloud scheduling and satellite insertion/preference policies.
They were not promoted above the shortlist where their formulation, available
evidence or intended use was less direct. A publisher page for the 2026
GRIP-MAPPO satellite paper could not be opened; no detailed claim from it is used
in the recommendations. GeoNest (Sep 2026) was screened as a different geometric
packing problem. Absence from the shortlist is not evidence of inferior results.

## Local implementation evidence

Codebase graph project: `home-seman-dev-foreblocks`.
Generation: `2026-10-05T21:44:03Z`; Tier 2 verification.
`check_index_coverage(paths=[...])` returned `no_recorded_issue` and
`metadata_match` for:

- `projects/scheduling/learning.py`
- `projects/scheduling/neighborhood.py`
- `projects/scheduling/hybrid.py`
- `projects/scheduling/milp.py`

Full snippets were inspected for `train_neighborhood_policy`, `GraphBuilder`,
`NeighborhoodPolicy` and `select_neighborhood`. Outbound calls from training
were traced. Graph coverage remains best-effort; conclusions concern these
inspected functions rather than the entire repository.

Local performance numbers come from
`projects/scheduling/research/results/local-study/analysis.json` and
`projects/scheduling/research/results/RESULTS_REVIEW.md`, not literature sources.
The completed frozen study and its optimizer sources were not modified by this
review.

## Inference boundaries

Candidate-neighborhood ranking, ONTS-specific run/energy closure, lightweight
constraint-family embeddings, combined gain/cost prediction and a small mixed
portfolio are proposed adaptations. No cited paper is presented as having
already validated them on ONTS. No code from external repositories was executed,
and no faithful external-method replication is claimed.

The critical reviewer checked short-budget transfer, constraint-restriction
risks, supervision attribution, censored repair outcomes and novelty. The
recommendation is a falsifiable experiment sequence, not evidence of improved
performance or a state-of-the-art result.
