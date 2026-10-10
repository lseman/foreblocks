# Provenance

Date: 6 October 2026. Primary-source method sections were read through web tools.
CL-LNS, BTBS-LNS and RL-LNS were checked in proceedings PDFs; DANIEL, ReSched,
MIStar and SPL-LNS were checked in primary manuscripts. Proceedings records were
used to distinguish conference years from later arXiv uploads. SatGNN was checked
through its publisher record and abstract.
The public SatGNN repository was checked through its GitHub README (the earlier
author URL redirects to `gos-ufsc/sat-gnn`). The satellite GNN/MCTS manuscript was
checked through its primary arXiv record, revised November 2025; its search idea
is recorded as future work, not an implemented method or ONTS comparison.

The literature-review workflow used three read-only roles: researcher, verifier
and reviewer. Verification corrected the distinction between supervised CL-LNS
and RL, ReSched's main REINFORCE training, edge-feature placement, DANIEL's
best-of-100 inference, and SPL-LNS's hindsight supervision. The methodological
review identified combined interventions in the frozen control comparisons;
those limits appear in the review and second manuscript.

Code inspection followed the codebase-memory skill. Coverage checks for existing
dependencies and the new reviewed files reported no recorded issue. Exact source
was read for material behavior; graph traces contained unrelated name-resolution
edges and were not treated as authoritative call evidence.

The review separates verified prior methods, our implemented adaptations and
future untested hypotheses. No cited scheduling makespan result is transferred
numerically to ONTS. No public ONTS comparator has yet been reproduced here.

Compact-subset follow-up, 6 October 2026: re-opened Wu et al.'s primary NeurIPS
2021 PDF (factorized destroy decisions, variable subset size) and SPL-LNS v1
(Method; Training with Hindsight Relabeling). The implementation uses
nonempty-set-conditioned likelihoods, task-level groups, mean task–time messages
and rerun-validated group labels. These adaptations are our experiment design,
not results asserted by either paper. SatGNN's official README was rechecked:
it identifies its public dataset and separate training and heuristic-evaluation
scripts. No SatGNN run or matched published-baseline result is claimed here.

Ordered-group follow-up, 6 October 2026: checked Kool, van Hoof and Welling,
`arXiv:1803.08475v3` (ICLR 2019) and the primary POMO NeurIPS 2020 PDF at
`https://proceedings.neurips.cc/paper_files/paper/2020/file/f231f2107df69eab0a3862d50018a9b2-Paper.pdf`.
POMO's forced different initial nodes and shared-mean baseline are distinguished
from our sampled initial choices and leave-one-out baseline. No routing results,
symmetry augmentation or performance claims are transferred to ONTS. The new
ordered latent likelihood and mixed repair menu are implemented ONTS hypotheses;
the archived matched study determines their measured effects.
