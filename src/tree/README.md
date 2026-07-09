# Isolated Tree Build

This folder isolates the tree modules from the main project build.

By default it only configures the tree core (`foretree_core` interface target)
and does not build Python modules unless enabled.

Header organization and module grouping notes:
- `docs/structure.md`
- facade headers under `include/foretree/`

## Current Scope

The tree and forest implementations in this folder are currently scalar-output only.

- `UnifiedTree` stores one gradient, one hessian, and one leaf value per node.
- `ForeForest` expects 1-D targets for `fit_complete`.
- `predict` and `predict_margin` return one scalar value per row.
- Supported boosted objectives are the scalar objectives exposed in the bindings today: squared error, binary logloss, binary focal loss, and Huber error.

Multiclass softmax, vector-leaf trees, and distributional objectives such as Gaussian/NGBoost-style outputs are not part of the current implementation in this branch.

## Configure

```bash
cmake -S tree -B tree/build \
  -DTREE_BUILD_FORETREE=OFF \
  -DTREE_BUILD_FOREFOREST=OFF
```

## Build

```bash
cmake --build tree/build -j
```

## Optional toggles

- `-DTREE_BUILD_FORETREE=OFF`
- `-DTREE_BUILD_FOREFOREST=OFF`
- `-DTREE_ENABLE_TBB=OFF`
- `-DTREE_ENABLE_STDEXEC=ON -DTREE_STDEXEC_INCLUDE_DIR=/path/to/stdexec/include`
