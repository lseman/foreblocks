# DARTS Search Pipeline

This page maps the conceptual DARTS workflow onto the actual ForeBlocks modules.

## Why this pipeline exists

ForeBlocks does not treat DARTS as a single opaque training call. The search stack is deliberately staged so you can:

- prune bad candidates early with cheap signals
- spend short differentiable-search budgets only on promoted candidates
- derive a discrete architecture explicitly
- retrain the final model from scratch when desired

That structure makes it easier to debug, benchmark, and reason about the search than a monolithic NAS loop.

## Phase-to-module map

| Phase | Main code location | Role |
| --- | --- | --- |
| search-space definition | `foreblocks/darts/config.py` | defines architecture modes, op pools, and search budgets |
| candidate generation | `foreblocks/darts/search/orchestrator.py` | samples candidate configs and coordinates evaluation |
| zero-cost ranking | `foreblocks/darts/search/zero_cost.py` | computes cheap pre-training metrics |
| bilevel search training | `foreblocks/darts/training/darts_loop.py` | runs the mixed-architecture DARTS phase |
| multi-fidelity orchestration | `foreblocks/darts/search/multi_fidelity.py` | promotes top candidates and manages staged budgets |
| final fixed-model retraining | `foreblocks/darts/training/final_trainer.py` | retrains the derived architecture |
| result analysis | `foreblocks/darts/evaluation/analyzer.py` | builds search-result summaries and plots |
| public entry surface | `foreblocks/darts/trainer.py` | exposes the staged workflow as `DARTSTrainer` methods |

## Public control flow

The high-level path is:

1. instantiate `DARTSTrainer`
2. call `multi_fidelity_search(...)`
3. inspect the returned candidates and final metrics
4. save the best final model

Internally, the call chain is roughly:

```text
DARTSTrainer.multi_fidelity_search(...)
  -> search/multi_fidelity.py
      -> search/orchestrator.py
      -> search/zero_cost.py
      -> training/darts_loop.py
      -> trainer.derive_final_architecture(...)
      -> training/final_trainer.py
```

## Result artifacts by phase

### Candidate generation and zero-cost ranking

Produces:

- raw candidates
- cheap ranking metrics
- promoted `top_k` candidates

This is the best phase for debugging search-space quality.

### Bilevel DARTS phase

Produces:

- mixed searched model
- training and validation losses
- alpha histories
- intermediate metrics

This is the best phase for debugging collapse, instability, or bad promotion thresholds.

### Final retraining phase

Produces:

- the fixed `final_model`
- final metrics
- training info

This is the artifact that should matter most for downstream use.

## The important design choice: fixed-model retraining

ForeBlocks explicitly separates search-time mixed architectures from final deployment artifacts. That means the model you save after `derive_final_architecture(...)` is not just the same mixed model with a different label; it is a discrete architecture intended to be retrained and validated on its own merits.

This separation matters because:

- mixed-model performance can overestimate the final discrete architecture
- retraining reveals whether the search signal really transfers
- saving and sharing the final model becomes cleaner

## Stats collection and benchmarking

When `collect_stats=True`, the multi-fidelity pipeline stores timing and worker-parallelism artifacts. This is useful when you want to compare:

- how expensive phase 1 really is
- whether more workers help in practice
- whether candidate generation or search training is the true bottleneck

Treat this as an experiment-benchmarking feature, not something you need for every search run.

## Related pages

- [DARTS Guide](../darts.md)
- [Run A DARTS Search](../tutorials/darts-multifidelity-search.md)
- [Repository Map](../reference/repository-map.md)
