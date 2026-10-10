# Improving ONTS learned search: representation, neighborhoods and training

The current experiments establish comparisons with our implemented controls, not
with published ONTS implementations. Their negative comparisons motivate changes;
they do not establish that ONTS RL is inferior to literature methods on matched
benchmarks. A publication claim requires that matched comparison.

## Primary-source findings

| Source | Verified method | ONTS adaptation and boundary |
| --- | --- | --- |
| [CL-LNS, ICML 2023](https://proceedings.mlr.press/v202/huang23g.html) | Variable–constraint bipartite GAT scores destruction sets. Contrastive supervision uses good local-branching neighborhoods and repaired poor perturbations. | Expose concrete task subsets and supervise multiple useful choices. Our task–time graph compresses ONTS structure; it is not their MILP graph. Our listwise outcome loss is a different loss. CL-LNS is supervised learning, not RL. |
| [DANIEL, primary manuscript](https://arxiv.org/abs/2305.05119) | Dual operation/machine attention represents precedence and machine competition; PPO chooses operation–machine pairs. Evaluation includes greedy decisions and best-of-100 sampled complete schedules. | Model explicit entity interactions; distinguish greedy deployment from stochastic training. A single ONTS search trajectory cannot be compared with best-of-100 scheduling without matching compute. Its graph is not a task–time bipartite graph. |
| [ReSched, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/467e109270678b3e2b3c2e7fa6b3d575-Abstract-Conference.html) | Duration edges enter attention queries, keys and values. The residual FJSP state uses operation availability, minimum duration, machine availability and pairwise duration. Main training is REINFORCE, with a PPO variant. | Put incumbent activity and resource interactions into attention values as well as scores. Preserve ONTS windows, starts, durations, periodicity and battery information; its four-information FJSP state is not a proven ONTS state reduction. |
| [MIStar, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/77629f2df5ef97ad967157c69cd8af7b-Abstract-Conference.html) | A heterogeneous complete-schedule graph, retrieved historical state–action information and multiple evaluated local actions support improvement search. | Remember failed subsets and later test richer retrieval. Our exact-subset mask is a simpler mechanism. Best-of-several repair changes action selection and consumes extra runtime; it cannot silently be inserted into categorical PPO. |
| [BTBS-LNS, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/7eca17ef54789b0663cab421f2e9dbf5-Paper-Conference.pdf) | Variable/constraint/objective tripartite attention, a branching policy and bound tightening address generic MIP search. | A full constraint/objective graph is a later alternative if the compressed graph fails. Binary ONTS activity does not need generic-integer binarization; copying every component is not justified. |
| [SPL-LNS, 2025 preprint](https://arxiv.org/abs/2508.16171) | Sampling feasible repairs, annealing and hindsight destruction labels support nonmonotone search while retaining a best-so-far solution. | Later test current-state versus best-incumbent separation and revalidated hindsight task sets. This would change our transition/reward accounting. It is not an off-the-shelf on-policy RL update. |
| [Satellite GNN + MCTS, IWPSS 2025 manuscript](https://arxiv.org/abs/2408.15041) | A graph/RL observation scheduler adds Monte Carlo tree search after learning. | Budgeted lookahead is a later alternative to one-step neighborhood decisions. Its Earth-observation visibility and maneuver constraints differ from recurring ONTS tasks and battery dynamics; published numbers do not transfer. |

The applicable ONTS comparator is [SatGNN](https://www.sciencedirect.com/science/article/pii/S1568494625014528).
It learns feasibility/optimality-related information for ONTS matheuristics. It
does not establish an RL scheduling baseline for our new controller. Its published
benchmark and model semantics need a separate reproduction before comparison.
The [authors' public repository](https://github.com/gos-ufsc/sat-gnn) documents
instance generation, model training and heuristic evaluation and links its
FloripaSat dataset. Reproduction is therefore a concrete next experiment: pin
the source revision, audit constraint equivalence, preserve the original solver
configuration and retrain/evaluate on compatible splits. The repository does not
supply evidence that our present policy already matches the published method.

## Implemented adaptation

`rl_bipartite.py` supplies a compressed graph with task nodes and time nodes.
Eligibility edges use the validated half-open execution windows. Six edge
attributes expose incumbent activity, starts, last release, last changes,
other-task load and task draw. Time nodes retain solar, usage, battery charge,
slack, future solar summary and time position. Task nodes retain the fifteen
features from the preceding portfolio. Two directional cross-attention rounds
put projected edge attributes in both attention keys and values.

Candidate readout pools exact released task–slot pairs. It scores every singleton
and pair, three pressure/utility/random triple proposals and eight original MILP
actions. Thus the policy selects concrete task groups rather than choosing only
a rule. It still enumerates candidate groups; it does not construct unrestricted
subsets autoregressively. Enumeration is quadratic in task count.

Failed profile subsets are temporarily masked at the unchanged incumbent. A
changed incumbent clears the mask. This is diversification, not a local-optimality
certificate: the profile repair is a bounded heuristic. All results retain the
existing independent feasibility, objective and deadline acceptance checks.

Offline supervision evaluates all candidate actions at four training-incumbent
stages. The listwise target is the softmax of within-state normalized measured
improvement rates, retaining useful tied actions. PPO optimizes undiscounted
objective increments, with replay of teacher examples after each update batch.
This is our adaptation; it reproduces none of the cited complete algorithms.

## Frozen experiment and what its controls establish

The study `rl_bipartite_study.py` uses fresh content-disjoint instances: 24 training,
3 validation, 12 development and 6 larger cases. Two policy seeds receive the same
teacher examples. Final checkpoints are fixed before evaluation; greedy and
stochastic graph deployment are separately declared. Online budgets are 0.6 s in
training and 1.2 s in evaluation. The result directory preserves exact tensor
instances, source hashes/archive, teachers, transitions, checkpoints and schedules.

| Comparison | What it measures | What it cannot isolate |
| --- | --- | --- |
| Graph hybrid / graph-off hybrid | Added edge information and cross-node messages, with the same readout and parameter shapes | Message passing versus dynamic edge attributes separately |
| Graph hybrid / flat hybrid | Practical architecture quality under matched wall time, actions and teachers | Parameter count, readout and latency separately |
| Graph hybrid / graph IL | PPO-plus-teacher-replay continuation | PPO alone, extra teacher optimization alone |
| Direct bandit / template bandit | Expanded concrete groups plus failed-subset memory | Action space versus memory separately |
| Graph greedy / graph stochastic | Inference selection with identical weights | Whether greedy graph IL would achieve the same result |
| New graph / previous checkpoint | Overall new recipe versus the preceding trained controller on the same new evaluation instances | Representation alone; training size and budget also differ |

Read `results/rl-bipartite-20261006/REPORT.md` for observed outcomes. No positive
result is assumed in the design. Small-sample instance bootstrap intervals remain
exploratory and marginal without multiplicity correction.

## Next decisions after observing the study

1. If the graph improves matched graph-off performance, separately retrain
   edge-only and message-only controls before identifying the mechanism.
2. If only greedy inference improves quality, compare greedy graph IL and
   replay-only continuation before claiming an RL benefit.
3. If direct random/bandit search improves as much as neural selection, prioritize
   the neighborhood engine and reduce neural overhead; a larger network is not
   supported by that outcome.
4. If short-budget plateaus persist, test learned variable-size destruction and
   best-versus-current search states, with validated repair and revised rewards.
5. Fix one deployment policy and primary comparison, then evaluate fresh external
   ONTS instances, more independent training seeds and multiple runtime budgets
   against a reproduced published comparator.

The publication target is better ONTS objective under comparable constraints,
initialization and total online compute. Literature-inspired architecture names
cannot substitute for that experiment.

## Observed outcome

The completed primary experiment has 420 runs; its separately declared
continuation diagnosis adds 126. All 546 schedules pass independent replay.
Greedy graph hybrid improves the preceding controller's larger-case normalized
gain from 0.175927 to 0.185890, with paired difference interval
[0.001421, 0.021923]. It remains below the heuristic mean 0.188011. Retrained
graph-off controls do not establish an edge/message benefit, and matched
teacher-only replay does not establish an additional PPO benefit. Greedy
replay-only reaches 0.187674 on larger cases; that supplementary mean is
descriptive rather than a selected deployment claim.

These results favor the next experiments on variable-size neighborhood generation,
stronger multi-step labels and inference cost, rather than assuming that a bigger
bipartite network solves the remaining gap. See
[the results review](../results/rl-bipartite-20261006/RESULTS_REVIEW.md) for controls,
limits and source-backed next steps.

## Implemented follow-up: compact subsets and verified hindsight

The next implementation is [RL_SUBSET.md](../RL_SUBSET.md), with a separately
frozen study in `results/rl-subset-20261006`. Wu et al.'s binary-factorized destroy
actions motivate variable-size task groups. Our policy conditions the Bernoulli
set on nonemptiness and uses its exact joint probability for PPO. It does not
reproduce their customized actor–critic training or general-IP variable graph.

SPL-LNS's hindsight direction motivates changed-task candidates from a four-move
trajectory. Each candidate is rerun from the original incumbent and retained only
when a complete validator confirms a positive feasible gain. This differs from
the paper's variable-level Hamming-neighborhood labels and its nonmonotone,
annealed top-k repair-pool search. The new experiment separately evaluates
one-step labels, added hindsight labels, teacher replay and PPO continuation.

Primary sources rechecked on 6 October 2026:
[Wu et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/file/fc9e62695def29ccdb9eb3fed5b4c8c8-Paper.pdf),
[SPL-LNS, sections Method and Training with Hindsight Relabeling](https://arxiv.org/html/2508.16171v1).
Architecture and training adaptations are implemented hypotheses; measured effects
belong to the new frozen study, not to the published sources.

## Implemented follow-up: conditional ordered groups and rollout baselines

The [ordered-group extension](../RL_SEQUENCE.md) replaces conditionally
independent task bits with repair/cardinality heads and task pointers that read
the selected prefix. [Kool et al., ICLR 2019](https://arxiv.org/abs/1803.08475)
provide a primary-source example of autoregressive combinatorial decoding.
[POMO, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html)
motivates training with related-rollout baselines. Our experiment uses four
rollouts from the same instance/incumbent and other-rollout terminal returns
as a leave-one-out baseline. It does not force diverse first choices or use
augmentation/best-of-N deployment. The sequence likelihood keeps latent order,
operator choice and cardinality for valid PPO replay. Gain-only supervision,
PPO, leave-one-out REINFORCE and replay-only continuation receive explicit
matched controls. These are ONTS adaptations to be tested, not reproduced
performance claims from either published routing study.
