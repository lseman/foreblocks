#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace foretree {

struct PackedTree;

} // namespace foretree

namespace foretree::cuda {

// Configuration for neural-leaf MLP forward pass on device.
struct NeuralLeafConfig {
    int layer_count = 0;
    std::vector<int> layer_sizes;    // [input_dim, hidden1, ..., output_dim]
    std::vector<float> weights;      // row-major weights, concatenated layers
    std::vector<float> biases;       // concatenated biases
    // Activation type per layer (0=none, 1=relu, 2=sigmoid, 3=tanh)
    std::vector<uint8_t> activations;
};

// Prediction engine for GPU-accelerated inference on packed tree forests.
// Accepts binned data and walks each tree in parallel across samples.
class GpuPredictionEngine {
public:
    // Construct from a PackedTree. Copies all tree data to device memory.
    explicit GpuPredictionEngine(const PackedTree& tree,
                                 const NeuralLeafConfig* neural_config = nullptr);

    // Construct from numpy arrays (tuple/list of 23 arrays matching PackedTree field order):
    // 0:features(int), 1:thresholds(int), 2:split_values(double), 3:split_kinds(uint8),
    // 4:missing_left(uint8), 5:left_children(int), 6:right_children(int),
    // 7:leaf_flags(uint8), 8:cover(double),
    // 9:categorical_offsets(int), 10:categorical_counts(int), 11:categorical_bins(int),
    // 12:pair_features_a(int), 13:pair_features_b(int), 14:pair_thresholds_a(int),
    // 15:pair_thresholds_b(int), 16:pair_quadrant_masks(uint8),
    // 17:oblique_offsets(int), 18:oblique_counts(int), 19:oblique_features(int),
    // 20:oblique_weights(double), 21:oblique_thresholds(double), 22:leaf_values(double)
    explicit GpuPredictionEngine(
        std::span<const std::span<const int>> int_arrays,
        std::span<const std::span<const double>> double_arrays,
        std::span<const std::span<const uint8_t>> uint8_arrays,
        int num_features,
        int outputs = 1);

    ~GpuPredictionEngine();

    GpuPredictionEngine(GpuPredictionEngine&&) noexcept;
    GpuPredictionEngine& operator=(GpuPredictionEngine&&) noexcept;
    GpuPredictionEngine(const GpuPredictionEngine&) = delete;
    GpuPredictionEngine& operator=(const GpuPredictionEngine&) = delete;

    // Predict on binned data (uint8_t codes). Output is N * outputs.
    // For binary/single-output: output = N floats (one per sample).
    // For multiclass K: output is N*K floats (K trees, row-major).
    std::vector<double> predict_binned_uint8(std::span<const uint8_t> codes) const;

    // Predict on binned data (uint16_t codes).
    // Overload accepting explicit num_samples (avoids integer division issues).
    std::vector<double> predict_binned_uint16(std::span<const uint16_t> codes) const;
    std::vector<double> predict_binned_uint16(std::span<const uint16_t> codes, int num_samples) const;

    // Predict on raw double data. Uses neural leaf MLP when available.
    // codes may be empty if neural_config is provided (raw-only mode).
    std::vector<double> predict_raw(std::span<const double> raw_data,
                                    std::span<const uint8_t> codes = {}) const;

    // Overload with uint16_t codes + raw data.
    std::vector<double> predict_raw_uint16(std::span<const double> raw_data,
                                           std::span<const uint16_t> codes) const;

    // Construct from raw arrays (for Python binding).
    GpuPredictionEngine(const std::vector<std::span<const int>>& int_arrays,
                        const std::vector<std::span<const double>>& double_arrays,
                        const std::vector<std::span<const uint8_t>>& uint8_arrays,
                        int num_features, int outputs = 1);

    [[nodiscard]] int outputs() const noexcept;
    [[nodiscard]] int trees() const noexcept;
    [[nodiscard]] size_t device_bytes() const noexcept;

    // Impl is a pimpl; defined in gpu_prediction.cu
    // Made public so the .cu file's helper functions can access it.
    struct Impl;  // defined in gpu_prediction.cu
    std::unique_ptr<Impl> impl_;
};

bool is_available() noexcept;

} // namespace foretree::cuda
