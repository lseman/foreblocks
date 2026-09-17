#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "foretree/split/split_finder.hpp"

int main() {
    constexpr int repetitions = 64;
    constexpr int rows = 4 * repetitions;
    std::vector<uint16_t> codes;
    std::vector<double> gradients;
    std::vector<double> hessians(rows, 1.0);
    std::vector<int> row_index(rows);
    codes.reserve(static_cast<size_t>(rows) * 2);
    gradients.reserve(rows);
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                codes.push_back(static_cast<uint16_t>(a));
                codes.push_back(static_cast<uint16_t>(b));
                gradients.push_back(a == b ? -1.0 : 1.0);
            }
        }
    }
    std::iota(row_index.begin(), row_index.end(), 0);

    const std::vector<int> finite_bins = {2, 2};
    const std::vector<int> missing_bins = {2, 2};
    const std::vector<int> active_features = {0, 1};
    foretree::splitx::SplitContext context;
    context.P = 2;
    context.N = rows;
    context.Gp = 0.0;
    context.Hp = rows;
    context.Cp = rows;
    context.variable_bins = true;
    context.finite_bins_per_feat = finite_bins.data();
    context.missing_ids_per_feat = missing_bins.data();
    context.active_features = &active_features;
    context.Xb.codes16 = codes.data();
    context.row_index = row_index.data();
    context.row_g = gradients.data();
    context.row_h = hessians.data();
    context.hyp.min_samples_leaf_ = 1;
    context.hyp.min_child_weight_ = 0.0;

    foretree::PairInteractionConfig config;
    config.interaction_bins = 2;
    config.min_node_rows = 2;
    config.axis_guard_factor = 1.0;
    const foretree::PairInteractionSplitFinder finder;
    const auto split = finder.best_pair_interaction(context, config);
    assert(split.kind == foretree::splitx::SplitKind::PairInteraction);
    assert(std::isfinite(split.gain) && split.gain > 0.0);
    assert(split.pair_feature_a == 0);
    assert(split.pair_feature_b == 1);
    assert(split.pair_threshold_a == 0);
    assert(split.pair_threshold_b == 0);
    assert(split.pair_quadrant_mask == 9); // equal quadrants (XNOR side)
}
