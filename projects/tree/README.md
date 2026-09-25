# ForeTree - High-Performance C++ Tree-Based Models with GPU Support

ForeTree is a high-performance, C++23 implementation of tree-based machine learning models including decision trees, random forests, and gradient boosting machines. It features histogram-based splitting with gradient-aware binning strategies, multiple tree growth policies (leaf-wise, level-wise, oblivious), advanced split types (axis-aligned, categorical partition, oblique/k-feature, pair interaction), parallel execution, GPU acceleration via CUDA, and Python bindings via nanobind.

## Overview

ForeTree provides:

- **Histogram-Based Binning**: 7 binning strategies (uniform, quantile, kmeans, gradient-aware, two-stage, adaptive, categorical gradient)
- **Gradient Histogram System**: Optimized gradient/hessian histogram computation with variable bin allocation
- **Multiple Growth Policies**: Leaf-wise (XGBoost), level-wise (sklearn), and oblivious (CatBoost) strategies
- **Advanced Split Types**: Axis-aligned, categorical partition, oblique (k-feature hyperplane), and pair interaction splits
- **Ensemble Modes**: Bagging (Random Forest), GBDT, and Frank-Wolfe boosting (FWBoost)
- **Anomaly Detection**: Isolation Forest and Extended Isolation Forest (hyperplane splits), multithreaded
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

**Quantized training** (opt-in, like LightGBM's `use_quantized_grad`): per
tree, gradients and hessians are stochastically rounded to integer levels and
CPU histograms are built with packed integer sums. On a 150k x 40 binary task
this fits ~14% faster than exact CPU histograms (and matches the CUDA path)
with the same accuracy at 8 bits. Split gains use the quantized sums; node
totals and leaf values stay exact. Boosting modes only.

```python
cfg.quantized_gradients = True
cfg.quantized_gradient_bits = 8   # 2..8; 4 bits lost accuracy in our tests
```

### Isolation Forest (Python)

```python
import foreforest

iso = foreforest.IsolationForest(
    n_estimators=200,
    max_samples=256,       # rows per tree (psi)
    extension_level=0,     # 0: classic IF; k > 0: hyperplanes over k+1 features (Extended IF); -1: all features
    contamination=-1.0,    # < 0: "auto" (anomaly score 0.5 is the boundary); in (0, 0.5]: training quantile
    random_state=0,
).fit(X_train)             # float64 (N, P); NaN = missing

s = iso.anomaly_score(X)       # 2^(-E[h(x)] / c(psi)) in (0, 1], higher = more anomalous
iso.score_samples(X)           # -s (scikit-learn convention: higher = more normal)
iso.decision_function(X)       # score_samples - offset, negative for outliers
iso.predict(X)                 # +1 inlier / -1 outlier
```

C++: `foretree::IsolationForest` in `foretree/ensemble/isolation_forest.hpp`
(header-only; `fit`, `anomaly_score`, `score_samples`, `decision_function`,
`predict`, `mean_path_length`).

See `tests/bench_sota_options.py` for full working examples including categorical features, oblique splits, DART, GOSS, and comparison benchmarks against XGBoost, LightGBM, and CatBoost.

## Key Features

### 1. Histogram-Based Splitting
- **Feature-Major Histograms**: Efficient histogram computation organized by features
- **Gradient Histogram System**: Optimized gradient and hessian histogram computation
- **Histogram Primitives**: Low-level histogram operations optimized for performance
- **Variable Bin Allocation**: Per-feature bin counts proportional to information content

### 2. Advanced Binning Strategies
- **7 Strategies**: uniform, quantile, kmeans, gradient-aware, two-stage, adaptive, categorical gradient
- **Default**: hessian-weighted quantile bins (`HistogramConfig::method = "quantile"`), fitted one feature per thread
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
- **Isolation Forest**: unsupervised anomaly scores; classic axis splits or Extended IF hyperplanes, NaN-aware

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

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| CMake | 3.20 | Required for nanobind and CPM support |
| C++ compiler | GCC 13+, Clang 16+, MSVC 2022+ | C++23 required |
| Python | 3.12+ | For Python bindings (nanobind) |
| numpy | any | Required for Python usage |
| CUDA Toolkit | 11.0+ (optional) | Only if `TREE_ENABLE_CUDA_BACKEND=ON` |
| TBB | any (optional) | Falls back to `std::thread` if unavailable |
| Eigen3 | 3.x | Via CPM auto-download |
| fmt | any | Via CPM auto-download |

### Build Steps

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release          # builds foretree + foreforest Python modules
make -j$(nproc)
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TREE_BUILD_FORETREE` | ON | Build the `foretree` Python module |
| `TREE_BUILD_FOREFOREST` | ON | Build the `foreforest` Python module |
| `TREE_BUILD_TESTS` | OFF | Build the C++ tests and register them with CTest |
| `TREE_ENABLE_CUDA_BACKEND` | ON | Build the CUDA histogram backend |
| `TREE_ENABLE_TBB` | ON | Use TBB if available |
| `TREE_ENABLE_STDEXEC` | ON | Fetch stdexec for `foreforest` |
| `FORETREE_EXEC_USE_CUDA` | OFF | Use the nvexec CUDA scheduler (needs stdexec) |

## Testing

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DTREE_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure                         # C++ tests (asserts stay on in Release)
PYTHONPATH=. pytest ../tests/test_foreforest_python.py   # binding regressions
```

C++ tests cover histogram primitives and the feature-major kernel, dataset
representation, parallel tree paths, split finding, row partitioning, growth
policies, packed trees, ordered categoricals, pair-interaction splits, the
Isolation Forest (`test_isolation_forest.cpp`), and the CUDA histograms against
a CPU reference (`test_cuda_histogram*.cu`, skipped without a GPU).

Python benchmarks in `tests/` compare ForeForest against sklearn HistGradientBoosting, XGBoost, LightGBM, and CatBoost.

## Capabilities

| Feature | Description |
|---------|-------------|
| Binning strategies | uniform, quantile, kmeans, gradient-aware, two-stage, adaptive, categorical gradient |
| Tree growth | leaf-wise (priority queue), level-wise (BFS), oblivious (same split per depth) |
| Split types | axis-aligned, categorical partition, oblique (k-feature hyperplane), pair interaction |
| Ensemble modes | Bagging (Random Forest), GBDT, Frank-Wolfe boosting (FWBoost) |
| Anomaly detection | Isolation Forest, Extended Isolation Forest |
| Boosting features | GOSS, DART, early stopping, column/row subsampling, quantized-gradient training |
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

| Dependency | Type | Source |
|------------|------|--------|
| nanobind v2.9.2 | Build | CPM auto-download from wjakob/nanobind |
| fmt | Build | CPM auto-download |
| Eigen 3.x | Build | CPM auto-download |
| CUDA Toolkit 11.0+ | Optional | System install, `TREE_ENABLE_CUDA_BACKEND` |
| TBB | Optional | System install (`find_package(TBB)`), falls back to std::thread |
| stdexec | Optional | CPM auto-download from NVIDIA/stdexec |

## Troubleshooting

### Common Build Issues

**"Python 3.12+ not found"**
The bindings require Python 3.12+. Set the interpreter explicitly:
```bash
cmake .. -DPython3_EXECUTABLE=$(which python3.12)
```

**"TBB not found"**
TBB is optional — the build falls back to `std::thread`. To use TBB:
```bash
# Ubuntu/Debian
sudo apt install libtbb-dev
# CentOS/RHEL
sudo dnf install tbb-devel
```

**"CUDA not found"**
Ensure CUDA Toolkit 11.0+ is installed and `nvcc` is in PATH:
```bash
export CUDA_HOME=/usr/local/cuda
cmake .. -DTREE_ENABLE_CUDA_BACKEND=ON
```

**"stdexec download fails"**
The stdexec repo is large (~2GB). Ensure sufficient disk space and network connectivity. Disable if not needed:
```bash
cmake .. -DTREE_ENABLE_STDEXEC=OFF
```

**"CPM.cmake fetch fails"**
CPM auto-downloads dependencies. If the network is unreliable, set `CPM_USE_LOCAL_PACKAGES`:
```bash
cmake .. -DCPM_USE_LOCAL_PACKAGES=ON
```

### Runtime Issues

**ImportError: dynamic module does not define module export function**
This usually means the `.so` was built for a different Python version. Rebuild with the correct Python:
```bash
rm -rf build/
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python)
make -j$(nproc)
```

**CUDA out of memory**
Reduce batch size in benchmark scripts or disable CUDA backend. The GPU histogram engine uses approximately `N × P × max_bins × 16 bytes` for gradient+histogram buffers.

## License

Please refer to the project license file for licensing information.
