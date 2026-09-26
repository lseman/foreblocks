# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Split `unified_tree.hpp` into `unified_tree.hpp` + `unified_tree_fit.tpp` + `unified_tree_predict.tpp` for improved compile times and maintainability.

## [0.1.0] - 2025-09-26

### Added

- **Histogram-based binning**: 7 binning strategies (uniform, quantile, kmeans, gradient-aware, two-stage, adaptive, categorical gradient)
- **Gradient histogram system**: Optimized gradient/hessian histogram computation with variable bin allocation
- **Multiple growth policies**: Leaf-wise (XGBoost-style), level-wise (sklearn-style), and oblivious (CatBoost-style) tree growth
- **Advanced split types**: Axis-aligned, categorical partition, oblique (k-feature hyperplane), and pair interaction splits
- **Ensemble modes**: Bagging (Random Forest), GBDT, and Frank-Wolfe boosting
- **GPU acceleration**: CUDA histogram computation via `CudaHistogramEngine`
- **Neural leaf support**: GPU-accelerated neural network-based leaf predictions
- **Python bindings**: `foretree` (decision trees) and `foreforest` (ensembles) via nanobind
- **Isolation Forest**: Outlier detection via `IsolationForest`
- **TreeSHAP**: Shapley value feature attribution for scalar-output trees
- **Feature importance**: Gain, cover, and frequency metrics
- **Cost-Complexity Pruning**: Post-pruning via `ccp_alpha`
- **2D pair interaction histograms**: GPU-accelerated 2D feature pair histogram computation
- **Model export**: Treelite and ONNX export utilities via `foretree_export.py`
- **Neural Oblique Decision Tree (NODT)**: PyTorch-based NODT implementation via `neural_odst.py`

### Features

- C++23 standard
- TBB parallel execution with `std::thread` fallback
- Eigen3 for linear algebra
- fmt for formatting
- CMake build system with CPM dependency manager
- Comprehensive test suite (C++ and Python)
- GitHub Actions CI/CD

[Unreleased]: https://github.com/semant/foreblocks/compare/tree/v0.1.0...HEAD
[0.1.0]: https://github.com/semant/foreblocks/releases/tree/tree/v0.1.0
