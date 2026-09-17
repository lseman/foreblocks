#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "foretree/tree/packed_tree.hpp"
#include "foretree/tree/tree_types.hpp"

namespace foretree {

struct PackedTreeArrays {
    std::vector<int>& features;
    std::vector<int>& thresholds;
    std::vector<double>& split_values;
    std::vector<uint8_t>& split_kinds;
    std::vector<uint8_t>& missing_left;
    std::vector<int>& left_children;
    std::vector<int>& right_children;
    std::vector<uint8_t>& leaf_flags;
    std::vector<double>& cover;
    std::vector<int>& categorical_offsets;
    std::vector<int>& categorical_counts;
    std::vector<int>& categorical_bins;
    std::vector<int>& pair_features_a;
    std::vector<int>& pair_features_b;
    std::vector<int>& pair_thresholds_a;
    std::vector<int>& pair_thresholds_b;
    std::vector<uint8_t>& pair_quadrant_masks;
    std::vector<int>& oblique_offsets;
    std::vector<int>& oblique_counts;
    std::vector<int>& oblique_features;
    std::vector<double>& oblique_weights;
    std::vector<double>& oblique_thresholds;
    std::vector<double>& leaf_values;
};

class PackedTreeBuilder {
public:
    static PackedTree build(const std::vector<Node>& nodes, int outputs) {
        PackedTree tree;
        tree.outputs = outputs;
        tree.root = build(nodes, outputs, PackedTreeArrays{tree.features,
                                                           tree.thresholds,
                                                           tree.split_values,
                                                           tree.split_kinds,
                                                           tree.missing_left,
                                                           tree.left_children,
                                                           tree.right_children,
                                                           tree.leaf_flags,
                                                           tree.cover,
                                                           tree.categorical_offsets,
                                                           tree.categorical_counts,
                                                           tree.categorical_bins,
                                                           tree.pair_features_a,
                                                           tree.pair_features_b,
                                                           tree.pair_thresholds_a,
                                                           tree.pair_thresholds_b,
                                                           tree.pair_quadrant_masks,
                                                           tree.oblique_offsets,
                                                           tree.oblique_counts,
                                                           tree.oblique_features,
                                                           tree.oblique_weights,
                                                           tree.oblique_thresholds,
                                                           tree.leaf_values});
        return tree;
    }

    static int build(const std::vector<Node>& nodes, int outputs, PackedTreeArrays arrays) {
        int maximum_id = -1;
        for (const Node& node : nodes)
            maximum_id = std::max(maximum_id, node.id);
        const int node_count = maximum_id + 1;

        arrays.features.assign(node_count, -1);
        arrays.thresholds.assign(node_count, -1);
        arrays.split_values.assign(node_count, std::numeric_limits<double>::quiet_NaN());
        arrays.split_kinds.assign(node_count, static_cast<uint8_t>(splitx::SplitKind::Axis));
        arrays.missing_left.assign(node_count, 0);
        arrays.left_children.assign(node_count, -1);
        arrays.right_children.assign(node_count, -1);
        arrays.leaf_flags.assign(node_count, 1);
        arrays.cover.assign(node_count, 0.0);
        arrays.categorical_offsets.assign(node_count, -1);
        arrays.categorical_counts.assign(node_count, 0);
        arrays.categorical_bins.clear();
        arrays.pair_features_a.assign(node_count, -1);
        arrays.pair_features_b.assign(node_count, -1);
        arrays.pair_thresholds_a.assign(node_count, -1);
        arrays.pair_thresholds_b.assign(node_count, -1);
        arrays.pair_quadrant_masks.assign(node_count, 0);
        arrays.oblique_offsets.assign(node_count, -1);
        arrays.oblique_counts.assign(node_count, 0);
        arrays.oblique_features.clear();
        arrays.oblique_weights.clear();
        arrays.oblique_thresholds.assign(node_count, std::numeric_limits<double>::quiet_NaN());
        arrays.leaf_values.assign(static_cast<size_t>(node_count) * static_cast<size_t>(outputs), 0.0);

        for (const Node& node : nodes) {
            const size_t id = static_cast<size_t>(node.id);
            arrays.cover[id] = std::max(0, node.C);
            if (node.is_leaf) {
                const size_t offset = id * static_cast<size_t>(outputs);
                for (int output = 0; output < outputs; ++output) {
                    arrays.leaf_values[offset + static_cast<size_t>(output)] =
                        static_cast<size_t>(output) < node.leaf_values.size()
                            ? node.leaf_values[static_cast<size_t>(output)]
                            : 0.0;
                }
                continue;
            }

            arrays.leaf_flags[id] = 0;
            arrays.features[id] = node.feature;
            arrays.thresholds[id] = node.thr;
            arrays.split_values[id] = node.split_value;
            arrays.split_kinds[id] = static_cast<uint8_t>(node.split_kind);
            arrays.missing_left[id] = node.miss_left ? 1u : 0u;
            arrays.left_children[id] = node.left;
            arrays.right_children[id] = node.right;
            if (node.split_kind == splitx::SplitKind::CategoricalPartition && !node.categorical_left_bins.empty()) {
                arrays.categorical_offsets[id] = static_cast<int>(arrays.categorical_bins.size());
                arrays.categorical_counts[id] = static_cast<int>(node.categorical_left_bins.size());
                arrays.categorical_bins.insert(arrays.categorical_bins.end(), node.categorical_left_bins.begin(),
                                               node.categorical_left_bins.end());
            } else if (node.split_kind == splitx::SplitKind::PairInteraction) {
                arrays.pair_features_a[id] = node.pair_feature_a;
                arrays.pair_features_b[id] = node.pair_feature_b;
                arrays.pair_thresholds_a[id] = node.pair_threshold_a;
                arrays.pair_thresholds_b[id] = node.pair_threshold_b;
                arrays.pair_quadrant_masks[id] = node.pair_quadrant_mask;
                arrays.missing_left[id] = node.pair_missing_left ? 1u : 0u;
            } else if (node.split_kind == splitx::SplitKind::Oblique && !node.oblique_features.empty()) {
                arrays.oblique_offsets[id] = static_cast<int>(arrays.oblique_features.size());
                arrays.oblique_counts[id] = static_cast<int>(node.oblique_features.size());
                arrays.oblique_features.insert(arrays.oblique_features.end(), node.oblique_features.begin(),
                                               node.oblique_features.end());
                arrays.oblique_weights.insert(arrays.oblique_weights.end(), node.oblique_weights.begin(),
                                              node.oblique_weights.end());
                arrays.oblique_thresholds[id] = node.oblique_threshold;
                arrays.missing_left[id] = node.oblique_missing_left ? 1u : 0u;
            }
        }
        return nodes.empty() ? -1 : nodes.front().id;
    }
};

} // namespace foretree
