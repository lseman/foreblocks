#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace foretree {

// Immutable-after-build inference representation. It deliberately contains no
// row ranges, gradients, histograms, sampling state, or training caches.
struct PackedTree {
    int root = -1;
    int outputs = 1;
    std::vector<int> features;
    std::vector<int> thresholds;
    std::vector<double> split_values;
    std::vector<uint8_t> split_kinds;
    std::vector<uint8_t> missing_left;
    std::vector<int> left_children;
    std::vector<int> right_children;
    std::vector<uint8_t> leaf_flags;
    std::vector<double> cover;
    std::vector<int> categorical_offsets;
    std::vector<int> categorical_counts;
    std::vector<int> categorical_bins;
    std::vector<int> pair_features_a;
    std::vector<int> pair_features_b;
    std::vector<int> pair_thresholds_a;
    std::vector<int> pair_thresholds_b;
    std::vector<uint8_t> pair_quadrant_masks;
    std::vector<int> oblique_offsets;
    std::vector<int> oblique_counts;
    std::vector<int> oblique_features;
    std::vector<double> oblique_weights;
    std::vector<double> oblique_thresholds;
    std::vector<double> leaf_values;

    [[nodiscard]] bool empty() const noexcept {
        return root < 0;
    }
    [[nodiscard]] size_t node_count() const noexcept {
        return leaf_flags.size();
    }
};

} // namespace foretree
