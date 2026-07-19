# ForeTree - High-Performance C++ Tree-Based Models with GPU Support

ForeTree is a high-performance, C++23 implementation of tree-based machine learning models including decision trees, random forests, and gradient boosting machines. It features histogram-based splitting with gradient-aware binning strategies, multiple tree growth policies (leaf-wise, level-wise, oblivious), advanced split types (axis-aligned, categorical partition, oblique/k-feature, pair interaction), parallel execution, GPU acceleration via CUDA, and Python bindings via nanobind.

## Overview

ForeTree provides:

- **Histogram-Based Binning**: 7 binning strategies (uniform, quantile, kmeans, gradient-aware, two-stage, adaptive, categorical gradient)
- **Gradient Histogram System**: Optimized gradient/hessian histogram computation with variable bin allocation
- **Multiple Growth Policies**: Leaf-wise (XGBoost), level-wise (sklearn), and oblivious (CatBoost) strategies
- **Advanced Split Types**: Axis-aligned, categorical partition, oblique (k-feature hyperplane), and pair interaction splits
- **Ensemble Modes**: Bagging (Random Forest), GBDT, and Frank-Wolfe boosting (FWBoost)
- **Advanced Features**: GOSS, DART, cost-complexity pruning, TreeSHAP, monotone constraints, EFB
- **GPU Acceleration**: CUDA-based histogram computation, split search, and neural leaf prediction
- **Unified Tree Representation**: Packed tree for memory-efficient inference
- **Python Bindings**: nanobind integration for Python usage

## Directory Structure

```
tree/
├── CMakeLists.txt                # CMake build configuration
├── README.md                     # This file
├── docs/                         # Documentation
│   └── structure.md              # Header organization guide
├── include/foretree/             # C++ header files
│   ├── all.hpp                   # Master include (includes all foretree modules)
│   ├── core.hpp                  # Core functionality facade
│   ├── split.hpp                 # Split engine facade
│   ├── ensemble.hpp              # Ensemble facade
│   ├── tree.hpp                  # Tree types facade
│   │
│   ├── core/                     # Core tree functionality
│   │   ├── dataset.hpp                   # QuantizedDataset (row-major, u8/u16)
│   │   ├── histogram_primitives.hpp      # HistogramConfig, VariableBinLayout, HistogramAccumulator
│   │   ├── binning_strategies.hpp        # 7 binning strategies
│   │   ├── data_binner.hpp               # DataBinner (binning utility)
│   │   ├── gradient_hist_system.hpp      # GradientHistogramSystem (orchestrator)
│   │   ├── parallel_executor.hpp         # Thread pool executor
│   │   ├── ordered_categorical.hpp       # Ordered target statistics
│   │   │
│   │   └── tree/                     # Tree-specific implementations
│   │       ├── tree_types.hpp                # TreeConfig, TrainingNode, ModelNode
│   │       ├── unified_tree.hpp              # UnifiedTree (training + inference)
│   │       ├── packed_tree.hpp               # PackedTree (inference representation)
│   │       ├── packed_tree_builder.hpp       # PackedTreeBuilder
│   │       ├── growth_policy.hpp             # GrowthPolicy (leaf/level/oblivious)
│   │       ├── training_context.hpp          # TreeTrainingContext, TreeTrainingArena
│   │       ├── row_partitioner.hpp           # RowPartitioner
│   │       └── neural.hpp                    # Neural leaf definitions
│   │
│   ├── split/                    # Split finding and evaluation
│   │   ├── split_engine.hpp              # SplitEngine, HistogramBackend, Splitter
│   │   ├── split_finder.hpp              # Axis/Categorical/Oblique/PairSplitFinder
│   │   ├── split_aux.hpp                 # Optional helper utilities
│   │   └── split_helpers.hpp             # SplitContext, Candidate, SplitHyper
│   │
│   └── gpu/                        # GPU/CUDA implementations
│       ├── cuda_histogram.hpp            # CudaHistogramEngine declarations
│       └── neural_leaf.hpp               # GpuNeuralLeafConfig declarations
│
├── src/                            # C++ source files
│   ├── pybind/                     # Python bindings via nanobind
│   │   ├── foretree.cpp              # Core + tree Python bindings
│   │   └── foreforest.cpp            # ForeForest Python bindings
│   └── gpu/
│       ├── cuda_histogram.cu         # CUDA histogram kernels
│       └── neural_leaf.cu            # CUDA neural leaf kernels
│
├── tests/                          # Test suite
│   ├── test_histogram_primitives.cpp # Histogram accumulation tests
│   ├── test_dataset_representation.cpp
│   ├── test_parallel_tree_paths.cpp
│   ├── test_split_active_features.cpp
│   ├── test_row_partitioner.cpp
│   ├── test_growth_policy.cpp
│   ├── test_feature_major_histogram.cpp
│   ├── test_packed_tree.cpp
│   ├── test_ordered_categorical.cpp
│   ├── test_pair_interaction_split.cpp
│   └── Python benchmarks:
│       ├── bench_sota_options.py     # Full benchmark vs sklearn/xgboost/lightgbm
│       ├── bench_unifiedtree.py
│       └── bench_constraints.py
│
├── build/                          # Build artifacts (CMake build directory)
│   ├── foretree.cpython-*.so        # Python module
│   ├── foreforest.cpython-*.so      # Python module
│   └── libforetree_cuda.a           # CUDA static library
```

## Core API

### C++ Usage

```cpp
#include <foretree/all.hpp>

using namespace foretree;

// Configure histogram binning
HistogramConfig hist_cfg;
hist_cfg.method = HistogramConfig::Method::Adaptive;
hist_cfg.max_bins = 256;

// Build gradient histogram system from raw data
GradientHistogramSystem ghs(hist_cfg);
ghs.fit_bins(X_data, N, P, g_data, h_data);

// Configure tree
tree::TreeConfig tree_cfg;
tree_cfg.max_depth = 10;
tree_cfg.max_leaves = 63;
tree_cfg.growth = tree::TreeConfig::Growth::LeafWise;
tree_cfg.lambda_ = 1.0;

// Train a single tree
UnifiedTree tree(tree_cfg, &ghs);
tree.fit(X_binned, N, P, g_data, h_data);

// Predict (returns (N,) for scalar output)
std::vector<double> pred = tree.predict(X_binned, N, P);

// Configure and train a forest
ForeForestConfig ff_cfg;
ff_cfg.mode = ForeForestConfig::Mode::GBDT;
ff_cfg.n_estimators = 300;
ff_cfg.learning_rate = 0.05;
ff_cfg.objective = ForeForestConfig::Objective::SquaredError;

ForeForest forest(ff_cfg);
forest.fit_complete(X_train, N, P, y_train);
std::vector<double> pred = forest.predict(X_test, N, P);
```

### Python Usage (via nanobind)

```python
import numpy as np
import foreforest

# Prepare data (numpy arrays, float64 for X and y)
X_train = np.random.rand(10000, 16).astype(np.float64)
y_train = np.random.rand(10000).astype(np.float64)
X_test = np.random.rand(1000, 16).astype(np.float64)

# Configure forest
cfg = foreforest.ForeForestConfig()
cfg.mode = foreforest.Mode.GBDT
cfg.n_estimators = 300
cfg.learning_rate = 0.05
cfg.objective = foreforest.Objective.SquaredError

# Build and train
model = foreforest.ForeForest(cfg)
model.fit_complete(X_train, y_train)

# Predict
pred = model.predict(X_test)            # (N,) for regression
prob = model.predict(X_test)            # (N,) for binary classification

# TreeSHAP contributions
contrib = model.predict_contrib(X_test) # (N, P+1)
```

See `tests/bench_sota_options.py` for full working examples including categorical features, oblique splits, DART, GOSS, and comparison benchmarks against XGBoost, LightGBM, and CatBoost.

## Key Features

### 1. Histogram-Based Splitting
- **Feature-Major Histograms**: Efficient histogram computation organized by features
- **Gradient Histogram System**: Optimized gradient and hessian histogram computation
- **Histogram Primitives**: Low-level histogram operations optimized for performance
- **Variable Bin Allocation**: Per-feature bin counts proportional to information content

### 2. Advanced Binning Strategies
- **7 Strategies**: uniform, quantile, kmeans, gradient-aware, two-stage, adaptive, categorical gradient
- **Data Binner**: Flexible data binning utilities with node-level overrides
- **Ordered Categorical**: Specialized handling for ordered categorical features
- **Feature Importance Weighting**: Adaptive bin counts based on feature importance

### 3. Tree Growth Policies
- **Leaf-Wise**: Priority-queue based best-first growth (XGBoost-style)
- **Level-Wise**: BFS level-by-level growth (scikit-learn HistGBDT-style)
- **Oblivious**: All nodes at same depth use same split (CatBoost-style)
- **Parallel Growth**: Concurrent node expansion with work stealing

### 4. Advanced Split Types
- **Axis-Aligned**: Standard histogram-based split on a single feature
- **Categorical Partition**: Optimal binary grouping of categories
- **Oblique**: Ridge regression on k features to find hyperplane splits
- **Pair Interaction**: 2D quadrant-based splits on feature pairs

### 5. Ensemble Methods
- **Bagging**: Random forest with row/column subsampling
- **GBDT**: Gradient boosting with GOSS and DART support
- **FWBoost**: Frank-Wolfe boosting (LPBoost-inspired) with line search
- **Objectives**: squared error, binary logloss, focal loss, Huber, quantile

### 6. Memory-Efficient Representation
- **QuantizedDataset**: uint8/uint16 feature codes with lazy column-major cache
- **PackedTree**: Structure-of-arrays inference representation
- **HistogramPool**: Efficient histogram memory management

### 7. GPU Acceleration (CUDA)
- **CudaHistogramEngine**: GPU-accelerated histogram computation and split search
- **Neural Leaf**: GPU support for neural network-based leaf predictions
- **Joint Histograms**: 2D feature pair histogram computation on GPU

### 8. Interpretability
- **TreeSHAP**: Shapley value feature attribution (scalar output)
- **Feature Importance**: gain, cover, and frequency metrics
- **Cost-Complexity Pruning**: Post-pruning via ccp_alpha

## Build Instructions

### Prerequisites

- CMake 3.15+
- C++23 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- CUDA Toolkit 11.0+ (optional, for GPU support)
- Python 3.9+ with numpy (for Python bindings)

### Build Steps

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `FORETREE_BUILD_TESTS` | OFF | Build C++ test suite |
| `FORETREE_ENABLE_CUDA` | OFF | Enable CUDA histogram backend |
| `FORETREE_ENABLE_TBB` | ON | Use TBB for parallel execution (falls back to std::thread) |
| `FORETREE_ENABLE_STDEXEC` | OFF | Use stdexec NVIDIA execution framework |

### Python Build

```bash
# Build and install Python bindings (in-tree build)
pip install -e .

# Or build standalone
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFORETREE_BUILD_TESTS=ON
make -j$(nproc)
```

## Testing

The test suite is currently disabled in the default build configuration. Enable it with:

```bash
cmake .. -DFORETREE_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

Individual C++ tests cover:
- `test_histogram_primitives.cpp`: Histogram accumulation, collision handling
- `test_dataset_representation.cpp`: QuantizedDataset serialization
- `test_parallel_tree_paths.cpp`: Parallel tree growth correctness
- `test_split_active_features.cpp`: Split finding on active feature subsets
- `test_row_partitioner.cpp`: Row partitioning for parallel growth
- `test_growth_policy.cpp`: Leaf-wise, level-wise, oblivious growth
- `test_feature_major_histogram.cpp`: Feature-major histogram layout
- `test_packed_tree.cpp`: Inference representation correctness
- `test_ordered_categorical.cpp`: Ordered target statistics
- `test_pair_interaction_split.cpp`: 2D quadrant-based splits

Python benchmarks in `tests/` compare ForeForest against sklearn HistGradientBoosting, XGBoost, LightGBM, and CatBoost.

## Capabilities

| Feature | Description |
|---------|-------------|
| Binning strategies | uniform, quantile, kmeans, gradient-aware, two-stage, adaptive, categorical gradient |
| Tree growth | leaf-wise (priority queue), level-wise (BFS), oblivious (same split per depth) |
| Split types | axis-aligned, categorical partition, oblique (k-feature hyperplane), pair interaction |
| Ensemble modes | Bagging (Random Forest), GBDT, Frank-Wolfe boosting (FWBoost) |
| Boosting features | GOSS, DART, early stopping, column/row subsampling |
| Objectives | squared error, binary logloss, binary focal loss, Huber, quantile regression |
| Regularization | L2 (lambda), L1 (alpha), gamma (min gain), max delta step, depth penalty |
| Constraints | monotone constraints, interaction constraints, max categories |
| Feature engineering | EFB (Exclusive Feature Bundling), ordered target statistics |
| Interpretability | TreeSHAP contributions, feature importance (gain/cover/frequency) |
| Pruning | Cost-complexity pruning (ccp_alpha) |
| GPU support | CUDA histogram computation, GPU neural leaf prediction |
| Multiclass | K-1 output trees |

## Performance Optimizations

1. **Feature-Major Histograms**: Column-oriented histogram layout for better cache locality during split search
2. **Variable Bin Allocation**: Per-feature bin counts proportional to feature information content
3. **AVX2-Accelerated Accumulation**: 4x unrolled histogram accumulation loops with SIMD
4. **GPU Histogram**: CUDA kernel for histogram building and split search on NVIDIA GPUs
5. **Packed Tree**: Structure-of-arrays inference representation eliminating node object overhead
6. **Thread Pool**: Work-stealing executor for parallel tree growth
7. **GOSS**: Gradient-based sampling reduces computation while preserving split quality

## Dependencies

- **CMake 3.15+**
- **C++23 compiler** (GCC 13+, Clang 16+, MSVC 2022+)
- **nanobind v2.9.2** (via CPM, for Python bindings)
- **fmt** (via CPM, for formatting)
- **Eigen 3.x** (via CPM, for matrix operations)
- **CUDA Toolkit 11.0+** (optional, `FORETREE_ENABLE_CUDA`)
- **TBB** (optional, for parallel execution; falls back to `std::thread`)
- **stdexec** (optional, NVIDIA execution framework)

## License

Please refer to the project license file for licensing information.
