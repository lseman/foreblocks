#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "foretree/core/histogram_primitives.hpp"

namespace {

void expect_near(double actual, double expected) {
    assert(std::abs(actual - expected) < 1e-12);
}

void test_contiguous_collisions_and_counts() {
    foretree::VariableBinLayout layout;
    layout.initialize({3, 2, 4, 2});

    // Four rows deliberately collide in feature zero/bin one.
    const std::vector<uint16_t> codes = {
        1, 0, 2, 1, 1, 1, 2, 0, 1, 0, 3, 1, 1, 1, 3, 0,
    };
    const std::vector<float> gradients = {1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> hessians = {0.5F, 1.0F, 1.5F, 2.0F};

    foretree::HistogramAccumulator accumulator(&layout);
    accumulator.accumulate_samples<true>(codes.data(), gradients.data(), hessians.data(), nullptr, 4, 4);

    const auto& g = accumulator.gradients();
    const auto& h = accumulator.hessians();
    const auto& c = accumulator.counts();
    const size_t collision = layout.get_offset(0, 1);
    expect_near(g[collision], 10.0);
    expect_near(h[collision], 5.0);
    assert(c[collision] == 4);

    expect_near(g[layout.get_offset(2, 2)], 3.0);
    expect_near(h[layout.get_offset(2, 2)], 1.5);
    assert(c[layout.get_offset(2, 2)] == 2);

    const std::vector<uint8_t> compact_codes(codes.begin(), codes.end());
    foretree::HistogramAccumulator compact_accumulator(&layout);
    compact_accumulator.accumulate_samples<true>(compact_codes.data(), gradients.data(), hessians.data(), nullptr, 4,
                                                 4);
    assert(compact_accumulator.gradients() == g);
    assert(compact_accumulator.hessians() == h);
    assert(compact_accumulator.counts() == c);
}

void test_indexed_rows_and_invalid_indices() {
    foretree::VariableBinLayout layout;
    layout.initialize({2, 2, 2, 2});

    const std::vector<uint16_t> codes = {
        0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0,
    };
    const std::vector<double> gradients = {1, 2, 3, 4, 5};
    const std::vector<double> hessians = {10, 20, 30, 40, 50};
    const int indices[] = {4, -1, 1, 4, 2};

    foretree::HistogramAccumulator accumulator(&layout);
    accumulator.accumulate_samples<true>(codes.data(), gradients.data(), hessians.data(), indices, 5, 4);

    const auto& g = accumulator.gradients();
    const auto& h = accumulator.hessians();
    const auto& c = accumulator.counts();
    const size_t feature_zero_bin_one = layout.get_offset(0, 1);
    expect_near(g[feature_zero_bin_one], 12.0);
    expect_near(h[feature_zero_bin_one], 120.0);
    assert(c[feature_zero_bin_one] == 3);
}

} // namespace

int main() {
    test_contiguous_collisions_and_counts();
    test_indexed_rows_and_invalid_indices();
}
