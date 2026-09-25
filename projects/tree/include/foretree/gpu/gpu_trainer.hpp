#pragma once

// Whole-tree GPU training for scalar GBDT (squared error / binary logloss).
//
// Everything that scales with the number of rows stays on the device for the
// whole fit: bin codes, labels, weights, margins, gradients, the row order that
// encodes each leaf's rows, and node histograms. Per tree the device computes
// gradients, builds histograms (the smaller child directly, the larger one by
// subtraction), finds the best axis split per feature for both missing
// directions, and partitions rows. The host only runs the best-first leaf
// queue and downloads a few bytes per split (the per-feature candidates).
//
// Gradients are converted to 64-bit fixed point for histogram accumulation:
// integer atomics make histograms exact and deterministic (independent of
// thread scheduling) and sibling subtraction exact.

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "foretree/core/dataset.hpp"

namespace foretree::cuda {

enum class GpuObjective { SquaredError, BinaryLogloss };

struct GpuTreeParams {
    int max_leaves = 31;
    int max_depth = 6;  // <= 0: unlimited
    int min_samples_leaf = 1;
    double min_child_weight = 1e-3;
    double lambda = 1.0;
    double alpha = 0.0;
    double gamma = 0.0;
    int missing_policy = 0;  // 0: learn, 1: always left, 2: always right
};

// Axis split on bin codes: code <= threshold goes left; the missing code
// follows missing_left. Leaves own rows [row_begin, row_end) of the trainer's
// current row order.
struct GpuTreeNode {
    int feature = -1;
    int threshold = -1;
    bool missing_left = true;
    int left = -1;
    int right = -1;
    bool is_leaf = true;
    int depth = 0;
    int count = 0;
    double G = 0.0;
    double H = 0.0;
    double gain = 0.0;
    int row_begin = 0;
    int row_end = 0;
};

class GpuTreeTrainer {
public:
    // `weights` may be empty (all ones). Margins start at `base_score`.
    GpuTreeTrainer(const QuantizedDataset& dataset, std::span<const double> labels,
                   std::span<const double> weights, GpuObjective objective, double base_score);
    ~GpuTreeTrainer();
    GpuTreeTrainer(const GpuTreeTrainer&) = delete;
    GpuTreeTrainer& operator=(const GpuTreeTrainer&) = delete;

    // Gradients from the current margins, then one tree. Node 0 is the root.
    std::vector<GpuTreeNode> grow(const GpuTreeParams& params);

    // margins[row] += scale * value[node] for the rows of every leaf of the
    // last grown tree (`values` is indexed by node id; internal nodes ignored).
    void update_margins(const std::vector<GpuTreeNode>& nodes, const std::vector<double>& values, double scale);

    // Current training margins (device -> host copy).
    [[nodiscard]] std::vector<double> margins() const;

    [[nodiscard]] int rows() const noexcept;
    [[nodiscard]] int features() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace foretree::cuda
