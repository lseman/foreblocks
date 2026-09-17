# Tree Module Structure

This document defines the intended header organization for `tree/`.

## Goals

- Keep public includes stable while improving internal readability.
- Reduce "where does this live?" ambiguity for split/binner/model code.
- Make future header splitting incremental and low risk.

## Current Implementation Status

The current `tree/` implementation is scalar-output only.

- `UnifiedTree` is a scalar tree implementation.
- `ForeForest` trains against 1-D targets and returns one prediction per row.
- The code in this branch should not be treated as supporting vector leaves, multiclass softmax trees, or distributional Gaussian objectives.

## Current Module Groups

- `core`
  - `foretree/core/data_binner.hpp`
  - `foretree/core/histogram_primitives.hpp` (hist config, stats, layouts, quantile utils)
  - `foretree/core/binning_strategies.hpp` (uniform/quantile/kmeans/grad/adaptive strategies)
  - `foretree/core/gradient_hist_system.hpp` (system orchestration and API)
- `split`
  - `foretree/split/split_aux.hpp` (optional helper namespace: `foretree::splitaux`)
  - `foretree/split/split_helpers.hpp`
  - `foretree/split/split_finder.hpp`
  - `foretree/split/split_engine.hpp`
- `tree model`
  - `foretree/tree/neural.hpp`
  - `foretree/tree/tree_types.hpp` (tree config + node/hist types)
  - `foretree/tree/unified_tree.hpp` (UnifiedTree implementation/entrypoint)
- `ensemble`
  - `foretree/ensemble/forest.hpp`

## Facade Headers

Use these for new call sites:

- `foretree/core.hpp`
- `foretree/split.hpp`
- `foretree/tree.hpp`
- `foretree/tree/unified_tree.hpp`
- `foretree/ensemble.hpp`
- `foretree/all.hpp`

These facades are thin includes only; behavior is unchanged.
`foretree/split.hpp` intentionally excludes `foretree/split/split_aux.hpp` because it exposes optional helper internals.

## Include Rules

- Prefer canonical `foretree/...` headers.
- Keep source files including the smallest needed facade or concrete header.
- Avoid broad includes (`foretree/all.hpp`) in performance-critical translation units.

## Suggested Phase 2 (Optional)

- Further split `unified_tree.hpp` internals into:
  - `tree_fit.hpp`
  - `tree_predict.hpp`
- Split `foretree/ensemble/forest.hpp` into:
  - `forest_config.hpp`
  - `forest_fit.hpp`
  - `forest_predict.hpp`
