#include <cassert>
#include <vector>

#include "foretree/tree/packed_tree_builder.hpp"

int main() {
    std::vector<foretree::Node> nodes(3);
    nodes[0].id = 0;
    nodes[0].is_leaf = false;
    nodes[0].feature = 2;
    nodes[0].thr = 4;
    nodes[0].left = 1;
    nodes[0].right = 2;
    nodes[0].C = 12;
    nodes[1].id = 1;
    nodes[1].leaf_values = {-1.5};
    nodes[1].C = 5;
    nodes[2].id = 2;
    nodes[2].leaf_values = {2.25};
    nodes[2].C = 7;

    const foretree::PackedTree packed = foretree::PackedTreeBuilder::build(nodes, 1);
    assert(packed.root == 0);
    assert(packed.node_count() == 3);
    assert(packed.features[0] == 2);
    assert(packed.left_children[0] == 1);
    assert(packed.right_children[0] == 2);
    assert(packed.leaf_values[1] == -1.5);
    assert(packed.leaf_values[2] == 2.25);

    nodes[0].split_kind = foretree::splitx::SplitKind::PairInteraction;
    nodes[0].pair_feature_a = 0;
    nodes[0].pair_feature_b = 1;
    nodes[0].pair_threshold_a = 2;
    nodes[0].pair_threshold_b = 3;
    nodes[0].pair_quadrant_mask = 9;
    nodes[0].pair_missing_left = false;
    const foretree::PackedTree pair_packed = foretree::PackedTreeBuilder::build(nodes, 1);
    assert(pair_packed.split_kinds[0] == static_cast<uint8_t>(foretree::splitx::SplitKind::PairInteraction));
    assert(pair_packed.pair_features_a[0] == 0);
    assert(pair_packed.pair_features_b[0] == 1);
    assert(pair_packed.pair_thresholds_a[0] == 2);
    assert(pair_packed.pair_thresholds_b[0] == 3);
    assert(pair_packed.pair_quadrant_masks[0] == 9);
    assert(pair_packed.missing_left[0] == 0);
}
