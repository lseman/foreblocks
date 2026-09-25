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

// Multiclass: softmax over `num_classes` explicit class margins plus an
// implicit class with margin 0 (labels 0..num_classes, one tree per explicit
// class and round), matching the CPU trainer.
enum class GpuObjective { SquaredError, BinaryLogloss, Multiclass };

struct GpuTreeParams {
    int max_leaves = 31;
    int max_depth = 6;  // <= 0: unlimited
    int min_samples_leaf = 1;
    double min_child_weight = 1e-3;
    double lambda = 1.0;
    double alpha = 0.0;
    double gamma = 0.0;
    int missing_policy = 0;  // 0: learn, 1: always left, 2: always right
    // Column subsampling: per tree, `tree_feature_percent`% of the features
    // (or `feature_bagging_k` of them, optionally with replacement); per node,
    // `node_feature_percent`% of the tree's features. Sampled on the host from
    // `seed`.
    int tree_feature_percent = 100;
    int node_feature_percent = 100;
    int feature_bagging_k = -1;
    bool feature_bagging_with_replacement = false;
    uint64_t seed = 0;
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
    // `num_classes` is the number of explicit class margins (1 unless
    // multiclass).
    GpuTreeTrainer(const QuantizedDataset& dataset, std::span<const double> labels,
                   std::span<const double> weights, GpuObjective objective, double base_score,
                   int num_classes = 1);
    ~GpuTreeTrainer();
    GpuTreeTrainer(const GpuTreeTrainer&) = delete;
    GpuTreeTrainer& operator=(const GpuTreeTrainer&) = delete;

    // Gradients (for `class_index`) from the current margins, then one tree on
    // `rows` (all rows when empty). Node 0 is the root.
    std::vector<GpuTreeNode> grow(const GpuTreeParams& params, int class_index = 0,
                                  std::span<const int> rows = {});

    // margins[class_index][row] += scale * value[leaf of row] for every row
    // (`values` is indexed by node id; internal nodes ignored). Uses the leaf
    // row ranges when the last tree grew on all rows, else traverses the tree.
    void update_margins(const std::vector<GpuTreeNode>& nodes, const std::vector<double>& values, double scale,
                        int class_index = 0);

    // Current training margins, row-major [row][class] (device -> host copy).
    [[nodiscard]] std::vector<double> margins() const;

    [[nodiscard]] int rows() const noexcept;
    [[nodiscard]] int features() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace foretree::cuda
