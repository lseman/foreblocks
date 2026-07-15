#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "foretree/core/dataset.hpp"

namespace foretree::cuda {

struct HistogramResult {
    std::vector<float> gradients;
    std::vector<float> hessians;
    std::vector<uint32_t> counts;
};

struct FeaturePair {
    uint32_t first = 0;
    uint32_t second = 0;
};

// Flattened as [pair][reduced_bin_a][reduced_bin_b], followed by one missing
// cell per pair. stride == reduced_bins * reduced_bins + 1.
struct JointHistogramResult {
    int reduced_bins = 0;
    size_t pair_count = 0;
    std::vector<float> gradients;
    std::vector<float> hessians;
    std::vector<uint32_t> counts;

    [[nodiscard]] size_t stride() const noexcept {
        return static_cast<size_t>(reduced_bins * reduced_bins + 1);
    }
};

struct SplitCandidate {
    float gain = -1.0F;
    uint32_t feature = 0;
    uint16_t bin = 0;
    bool missing_left = false;
};

class CudaHistogramEngine {
public:
    explicit CudaHistogramEngine(const QuantizedDataset& dataset);
    ~CudaHistogramEngine();

    CudaHistogramEngine(CudaHistogramEngine&&) noexcept;
    CudaHistogramEngine& operator=(CudaHistogramEngine&&) noexcept;
    CudaHistogramEngine(const CudaHistogramEngine&) = delete;
    CudaHistogramEngine& operator=(const CudaHistogramEngine&) = delete;

    void set_gradients(std::span<const float> gradients, std::span<const float> hessians);
    void compute_squared_error_gradients(std::span<const float> labels, std::span<const float> predictions);
    void compute_binary_logloss_gradients(std::span<const float> labels, std::span<const float> margins);

    HistogramResult build_histogram(std::span<const uint32_t> rows = {});
    JointHistogramResult build_joint_histograms(std::span<const FeaturePair> pairs, int reduced_bins,
                                                std::span<const uint32_t> rows = {});
    SplitCandidate find_best_split(float parent_gradient, float parent_hessian, float lambda,
                                   uint32_t minimum_samples_leaf, float minimum_child_weight);
    std::vector<uint8_t> route_rows(const SplitCandidate& split);

    [[nodiscard]] int rows() const noexcept;
    [[nodiscard]] int features() const noexcept;
    [[nodiscard]] size_t device_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool is_available() noexcept;

} // namespace foretree::cuda
