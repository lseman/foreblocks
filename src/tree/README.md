# Tree - High-Performance C++ Tree-Based Models with GPU Support

Tree is a high-performance, C++ implementation of tree-based machine learning models including decision trees, random forests, and gradient boosting machines. It features advanced histogram-based splitting, parallel execution, GPU acceleration via CUDA, and Python bindings via pybind11.

## Overview

Tree provides:

- **Histogram-Based Binning**: Advanced data binning strategies for efficient tree construction
- **Gradient Histogram System**: Optimized gradient histogram computation for gradient boosting
- **Parallel Execution**: Multi-threaded parallel execution for tree growth and training
- **GPU Acceleration**: CUDA-based histogram computation and neural leaf support
- **Unified Tree Representation**: Packed tree representation for memory efficiency
- **Flexible Growth Policies**: Configurable tree growth strategies and stopping criteria
- **Python Bindings**: pybind11 integration for Python usage

## Directory Structure

```
tree/
├── CMakeLists.txt                # CMake build configuration
├── .clangd                       # Clangd configuration for IDE support
├── test_headers.cpp              # Header test file
├── docs/                         # Documentation
│
├── include/foretree/             # C++ header files
│   ├── all.hpp                   # Main include file (includes all foretree modules)
│   ├── core.hpp                  # Core functionality includes
│   ├── split.hpp                 # Split engine includes
│   ├── ensemble.hpp              # Ensemble includes
│   ├── tree.hpp                  # Tree includes
│   │
│   ├── core/                     # Core tree functionality
│   │   ├── dataset.hpp                   # Dataset representation and management
│   │   ├── data_binner.hpp               # Data binning utilities
│   │   ├── binning_strategies.hpp        # Binning strategy implementations
│   │   ├── histogram_primitives.hpp      # Histogram primitive operations
│   │   ├── histogram_kernel.hpp          # Histogram kernel implementations
│   │   ├── gradient_hist_system.hpp      # Gradient histogram system
│   │   ├── parallel_executor.hpp         # Parallel execution utilities
│   │   ├── ordered_categorical.hpp       # Ordered categorical feature handling
│   │   │
│   │   ├── tree/                     # Tree-specific implementations
│   │   │   ├── tree_types.hpp                # Tree type definitions
│   │   │   ├── unified_tree.hpp              # Unified tree representation
│   │   │   ├── packed_tree.hpp               # Packed tree memory-efficient representation
│   │   │   ├── packed_tree_builder.hpp       # Packed tree builder
│   │   │   ├── row_partitioner.hpp           # Row partitioning utilities
│   │   │   ├── growth_policy.hpp             # Tree growth policies and stopping criteria
│   │   │   ├── training_context.hpp          # Training context management
│   │   │   └── neural.hpp                    # Neural leaf support
│   │   │
│   │   ├── split/                    # Split finding and evaluation
│   │   │   ├── split_engine.hpp              # Split engine interface
│   │   │   ├── split_finder.hpp              # Split finder implementation
│   │   │   ├── split_aux.hpp                 # Split auxiliary utilities
│   │   │   └── split_helpers.hpp             # Split helper functions
│   │   │
│   │   └── ensemble/                 # Ensemble methods
│   │       └── forest.hpp                    # Random forest implementation
│   │
│   └── gpu/                        # GPU/CUDA implementations
│       ├── cuda_histogram.hpp            # CUDA histogram computation
│       └── neural_leaf.hpp               # GPU neural leaf support
│
├── src/                            # C++ source files
│   └── pybind/                     # Python bindings via pybind11
│       ├── foretree.cpp              # Main foretree Python bindings
│       └── foreforest.cpp            # Forest Python bindings
│
├── tests/                          # Test suite
│   ├── test_histogram_primitives.cpp # Histogram primitives tests
│   ├── test_dataset_representation.cpp # Dataset representation tests
│   ├── test_parallel_tree_paths.cpp  # Parallel tree paths tests
│   ├── test_split_active_features.cpp # Split active features tests
│   ├── test_row_partitioner.cpp      # Row partitioner tests
│   ├── test_growth_policy.cpp        # Growth policy tests
│   ├── test_feature_major_histogram.cpp # Feature-major histogram tests
│   ├── test_packed_tree.cpp          # Packed tree tests
│   ├── test_ordered_categorical.cpp  # Ordered categorical tests
│   └── test_pair_interaction_split.cpp # Pair interaction split tests
│
├── build/                          # Build artifacts (CMake build directory)
└── README.md                       # This file

## Core API

### C++ Usage

```cpp
#include <foretree/all.hpp>

using namespace foretree;

// Create dataset
Dataset dataset = load_dataset("data.csv");

// Configure tree growth policy
GrowthPolicy policy;
policy.max_depth = 10;
policy.min_samples_split = 10;
policy.min_samples_leaf = 5;

// Create tree trainer
TreeTrainer trainer(policy);

// Train decision tree
UnifiedTree tree = trainer.train(dataset);

// Create forest
ForestConfig config;
config.n_trees = 100;
Forest forest(config);

// Train random forest
forest.train(dataset);
```

### Python Usage (via pybind11)

```python
import foretree

# Create dataset
dataset = foretree.Dataset.from_csv("data.csv")

# Train decision tree
tree = foretree.TreeTrainer(max_depth=10, min_samples_split=10).train(dataset)

# Train random forest
forest = foretree.Forest(n_trees=100, max_depth=10)
forest.train(dataset)

# Make predictions
predictions = forest.predict(dataset)
```

## Key Features

### 1. Histogram-Based Splitting
- **Feature-Major Histograms**: Efficient histogram computation organized by features
- **Gradient Histogram System**: Optimized gradient and hessian histogram computation
- **Histogram Primitives**: Low-level histogram operations optimized for performance

### 2. Advanced Binning Strategies
- **Data Binner**: Flexible data binning utilities
- **Binning Strategies**: Multiple binning strategies including equal-width, equal-frequency, and custom strategies
- **Ordered Categorical**: Specialized handling for ordered categorical features

### 3. Parallel Execution
- **Parallel Executor**: Multi-threaded parallel execution framework
- **Parallel Tree Paths**: Parallel traversal of tree paths during training
- **Row Partitioner**: Efficient row partitioning for parallel tree growth

### 4. Memory-Efficient Tree Representation
- **Packed Tree**: Memory-efficient packed tree representation
- **Packed Tree Builder**: Efficient construction of packed trees
- **Unified Tree**: Unified tree representation for flexibility

### 5. GPU Acceleration (CUDA)
- **CUDA Histogram**: GPU-accelerated histogram computation
- **Neural Leaf Support**: GPU support for neural leaf implementations
- **Parallel GPU Execution**: Parallel histogram computation on GPU

### 6. Flexible Growth Policies
- **Growth Policy**: Configurable tree growth strategies
- **Stopping Criteria**: Multiple stopping criteria (max depth, min samples, min impurity reduction)
- **Training Context**: Management of training state and context

### 7. Split Finding Algorithms
- **Split Engine**: Core split finding engine
- **Split Finder**: Implementation of split finding algorithms
- **Pair Interaction Split**: Support for feature interaction splits

## Build Instructions

### Prerequisites

- CMake 3.15+
- C++17 or C++20 compliant compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CUDA Toolkit (for GPU support)
- pybind11 (for Python bindings)

### Build Steps

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

### Python Build

```bash
# Build Python bindings
pip install -e .
# or
python -m build
```

## Dependencies

- CMake 3.15+
- C++17 or C++20 compiler
- CUDA Toolkit 11.0+ (optional, for GPU support)
- pybind11 (for Python bindings)
- Eigen (for matrix operations, if applicable)

## Testing

Run the test suite:

```bash
cd build
ctest --output-on-failure
```

Individual test files:
- `test_histogram_primitives.cpp`: Histogram primitive operations
- `test_dataset_representation.cpp`: Dataset management
- `test_parallel_tree_paths.cpp`: Parallel tree path execution
- `test_split_active_features.cpp`: Split finding with active features
- `test_row_partitioner.cpp`: Row partitioning utilities
- `test_growth_policy.cpp`: Tree growth policies
- `test_feature_major_histogram.cpp`: Feature-major histogram computation
- `test_packed_tree.cpp`: Packed tree representation
- `test_ordered_categorical.cpp`: Ordered categorical feature handling
- `test_pair_interaction_split.cpp`: Pair interaction split finding

## Performance Optimizations

1. **Feature-Major Histograms**: Organizes histogram computation by features for better cache locality
2. **Parallel Execution**: Multi-threaded tree growth and histogram computation
3. **GPU Acceleration**: CUDA-based histogram computation for large datasets
4. **Packed Tree Representation**: Memory-efficient tree storage with reduced overhead
5. **Gradient Histogram System**: Optimized gradient and hessian accumulation

## License

Please refer to the project license file for licensing information.
