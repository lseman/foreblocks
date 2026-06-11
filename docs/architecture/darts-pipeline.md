---
title: DARTS Search Pipeline
description: Conceptual DARTS workflow mapped onto actual ForeBlocks modules.
editLink: true
---

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
| search-space definition | `darts/config.py` | defines architecture modes, op pools, and search budgets |
| candidate generation | `darts/search/orchestrator.py` | samples candidate configs and coordinates evaluation |
| zero-cost ranking | `darts/search/zero_cost.py` | computes cheap pre-training metrics |
| bilevel search training | `darts/training/darts_loop.py` | runs the mixed-architecture DARTS phase |
| multi-fidelity orchestration | `darts/search/multi_fidelity.py` | promotes top candidates and manages staged budgets |
| final fixed-model retraining | `darts/training/final_trainer.py` | retrains the derived architecture |
| result analysis | `darts/evaluation/analyzer.py` | builds search-result summaries and plots |
| public entry surface | `darts/trainer.py` | exposes the staged workflow as `DARTSTrainer` methods |

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
