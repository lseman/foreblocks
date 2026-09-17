#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "foretree/core/histogram_kernel.hpp"

int main() {
    // Three rows, two columns in feature-major order.
    const std::vector<uint8_t> codes = {0, 1, 0, 1, 0, 1};
    const std::vector<double> gradients = {1.0, 2.0, 3.0};
    const std::vector<double> hessians(3, 1.0);
    const std::vector<int> features = {0, 1};
    const std::vector<size_t> offsets = {0, 3, 6};
    const std::vector<int> missing = {2, 2};
    std::vector<double> histogram_gradients(6, 0.0);
    std::vector<double> histogram_hessians(6, 0.0);
    std::vector<int> histogram_counts(6, 0);
    foretree::ParallelExecutor executor(2);

    foretree::dispatch_feature_major_histogram(
        true, std::span<const uint8_t>(codes), 3, 3, [](int position) { return position; },
        std::span<const int>(features), std::span<const size_t>(offsets), std::span<const int>(missing),
        std::span<const double>(gradients), std::span<const double>(hessians),
        foretree::HistogramOutputView{histogram_gradients, histogram_hessians, histogram_counts}, executor);

    assert(histogram_gradients == std::vector<double>({4.0, 2.0, 0.0, 2.0, 4.0, 0.0}));
    assert(histogram_counts == std::vector<int>({2, 1, 0, 1, 2, 0}));
    assert(histogram_hessians == std::vector<double>({2.0, 1.0, 0.0, 1.0, 2.0, 0.0}));
}
