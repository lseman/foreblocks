# Tree Module Structure

This document defines the intended header organization for `tree/`.

## Goals

- Keep public includes stable while improving internal readability.
- Reduce "where does this live?" ambiguity for split/binner/model code.
- Make future header splitting incremental and low risk.

## Current Implementation Status

The current `tree/` implementation is **scalar-output only**:

- `UnifiedTree` is a scalar tree implementation.
- `ForeForest` trains against 1-D targets and returns one prediction per row.
- The code in this branch should **not** be treated as supporting vector leaves, multiclass softmax trees, or distributional Gaussian objectives.

## Module Architecture

```
foretree/
│
├── all.hpp              # Master include (includes all facades)
├── core.hpp             # Core facade → binning + histograms + dataset
├── split.hpp            # Split engine facade → finders + helpers
├── tree.hpp             # Tree types facade → config + unified tree
├── ensemble.hpp         # Ensemble facade → ForeForest
│
├── core/                # Core tree functionality
│   ├── dataset.hpp                  # QuantizedDataset (row-major, u8/u16)
│   ├── histogram_primitives.hpp     # HistogramConfig, VariableBinLayout, HistogramAccumulator
│   ├── binning_strategies.hpp       # 7 binning strategies
│   ├── data_binner.hpp              # DataBinner (binning utility)
│   ├── gradient_hist_system.hpp     # GradientHistogramSystem (orchestrator)
│   ├── parallel_executor.hpp        # Thread pool executor
│   ├── ordered_categorical.hpp      # Ordered target statistics
│   │
│   └── tree/                    # Tree-specific implementations
│       ├── tree_types.hpp                # TreeConfig, TrainingNode, ModelNode
│       ├── unified_tree.hpp              # UnifiedTree (training + inference)
│       ├── packed_tree.hpp               # PackedTree (inference representation)
│       ├── packed_tree_builder.hpp       # PackedTreeBuilder
│       ├── growth_policy.hpp             # GrowthPolicy (leaf/level/oblivious)
│       ├── training_context.hpp          # TreeTrainingContext, TreeTrainingArena
│       ├── row_partitioner.hpp           # RowPartitioner
│       └── neural.hpp                    # Neural leaf definitions
│
├── split/                 # Split finding and evaluation
│   ├── split_engine.hpp              # SplitEngine, HistogramBackend, Splitter
│   ├── split_finder.hpp              # Axis/Categorical/Oblique/PairSplitFinder
│   ├── split_aux.hpp                 # Optional helper utilities (NOT in split.hpp)
│   └── split_helpers.hpp             # SplitContext, Candidate, SplitHyper
│
├── ensemble/              # Ensemble methods
│   └── forest.hpp                    # ForeForest (GBDT, Bagging, FWBoost)
│
└── gpu/                   # GPU/CUDA implementations
    ├── cuda_histogram.hpp            # CudaHistogramEngine declarations
    └── neural_leaf.hpp               # GpuNeuralLeafConfig declarations
```

## Data Flow

```
Raw Data (X, y, g, h)
       │
       ▼
┌─────────────────────┐
│  GradientHistSystem │ ← fit_bins() → HistogramConfig + VariableBinLayout
└─────────┬───────────┘
          │ binned data + histogram config
          ▼
┌─────────────────────┐
│    UnifiedTree::fit │ ← GrowthPolicy → parallel node expansion
└─────────┬───────────┘
          │ trained tree (TrainingNode graph)
          ▼
┌─────────────────────┐
│ PackedTreeBuilder   │ ← convert to structure-of-arrays
└─────────┬───────────┘
          │ packed tree (inference-ready)
          ▼
┌─────────────────────┐
│  ForeForest::predict│ ← sum over all trees → scalar prediction
└─────────────────────┘
```

## Facade Headers

Use these for new call sites:

| Facade | Includes | Purpose |
|--------|----------|---------|
| `foretree/core.hpp` | dataset, histogram_primitives, binning_strategies, data_binner, gradient_hist_system, parallel_executor, ordered_categorical | Core tree functionality |
| `foretree/split.hpp` | split_engine, split_finder, split_helpers | Split finding (excludes split_aux) |
| `foretree/tree.hpp` | tree_types, unified_tree, packed_tree, packed_tree_builder, growth_policy, training_context, row_partitioner, neural | Tree types and training |
| `foretree/ensemble.hpp` | ensemble/forest.hpp | Ensemble methods |
| `foretree/all.hpp` | all facades | Full library include |

**Note**: `foretree/split.hpp` intentionally excludes `foretree/split/split_aux.hpp` because it exposes optional helper internals.

## Include Rules

1. **Prefer canonical `foretree/...` headers** — always use the facade or concrete header under the `foretree/` namespace.
2. **Keep source files including the smallest needed facade or concrete header** — avoid broad includes in performance-critical translation units.
3. **Avoid `foretree/all.hpp` in performance-critical code** — use specific facades to reduce compile time.
4. **GPU headers are optional** — guarded by `#ifdef FORETREE_HAS_CUDA`.

## Suggested Phase 2 (Optional)

Further split large implementation files:

### unified_tree.hpp → tree_fit.hpp + tree_predict.hpp

Split the training and inference code paths:

```
unified_tree.hpp          →  tree_fit.hpp (training logic)
                            tree_predict.hpp (inference logic)
                            unified_tree.hpp (thin facade including both)
```

**Benefits**: Cleaner separation of concerns, faster compile times for inference-only users.

### forest.hpp → forest_config.hpp + forest_fit.hpp + forest_predict.hpp

Split the ensemble implementation:

```
ensemble/forest.hpp       →  forest_config.hpp (ForeForestConfig enum/class)
                            forest_fit.hpp (training loop, boosting logic)
                            forest_predict.hpp (prediction aggregation)
                            ensemble/forest.hpp (thin facade including all three)
```

**Benefits**: Users who only need prediction can include just `forest_predict.hpp`.

## Testing Strategy

### C++ Unit Tests (`tests/`)

Each test targets a specific component:

| Test File | Component | What It Validates |
|-----------|-----------|-------------------|
| `test_histogram_primitives.cpp` | histogram_primitives | Accumulation, collision handling |
| `test_dataset_representation.cpp` | dataset | QuantizedDataset serialization |
| `test_parallel_tree_paths.cpp` | growth_policy | Parallel tree growth correctness |
| `test_split_active_features.cpp` | split_finder | Split finding on active feature subsets |
| `test_row_partitioner.cpp` | row_partitioner | Row partitioning for parallel growth |
| `test_growth_policy.cpp` | growth_policy | Leaf-wise, level-wise, oblivious growth |
| `test_feature_major_histogram.cpp` | histogram_primitives | Feature-major histogram layout |
| `test_packed_tree.cpp` | packed_tree | Inference representation correctness |
| `test_ordered_categorical.cpp` | ordered_categorical | Ordered target statistics |
| `test_pair_interaction_split.cpp` | split_finder | 2D quadrant-based splits |

### Python Benchmarks (`tests/`)

| Benchmark | Compares Against | Key Metrics |
|-----------|-----------------|-------------|
| `bench_sota_options.py` | sklearn, XGBoost, LightGBM, CatBoost | Accuracy, training time, inference latency |
| `bench_unifiedtree.py` | standalone tree variants | Single-tree accuracy and speed |
| `bench_constraints.py` | monotone/constrained models | Constraint satisfaction, accuracy degradation |

## Future Roadmap

1. **Multiclass support**: K-1 output trees with softmax over classes
2. **Distributional objectives**: Gaussian, Poisson, and other exponential family targets
3. **ONNX export**: Production deployment via ONNX TreeEnsemble operator
4. **Treelite export**: C code generation for embedded deployment
5. **Header Phase 2**: Split unified_tree.hpp and forest.hpp as described above
