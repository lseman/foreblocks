// CUDA histograms must match a CPU reference for every feature (the launch
// once covered only feature 0), for row subsets, and for the joint histogram's
// missing cell.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "foretree/gpu/cuda_histogram.hpp"

int main() {
    if (!foretree::cuda::is_available())
        return 0;

    const int rows = 50000, features = 37;
    const uint8_t maximum_code = 63;
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> code(0, maximum_code);
    std::normal_distribution<float> gauss(0.0F, 1.0F);
    std::vector<uint8_t> codes(static_cast<size_t>(rows) * features);
    for (auto& c : codes)
        c = static_cast<uint8_t>(code(rng));  // code == maximum_code is the missing bin
    std::vector<float> g(rows), h(rows);
    for (int i = 0; i < rows; ++i) {
        g[i] = gauss(rng);
        h[i] = 0.25F + 0.1F * std::abs(gauss(rng));
    }
    auto dataset = foretree::QuantizedDataset::from_u8(rows, features, codes, maximum_code);
    foretree::cuda::CudaHistogramEngine engine(dataset);
    engine.set_gradients(g, h);

    std::vector<uint32_t> subset;
    for (int i = 0; i < rows; i += 3)
        subset.push_back(static_cast<uint32_t>(i));

    for (const bool use_subset : {false, true}) {
        const auto hist = use_subset ? engine.build_histogram(subset) : engine.build_histogram();
        const int bins = maximum_code + 1;
        assert(hist.gradients.size() == static_cast<size_t>(features) * bins);
        std::vector<double> rg(hist.gradients.size()), rh(hist.gradients.size());
        std::vector<uint32_t> rc(hist.gradients.size());
        auto add = [&](int row) {
            for (int f = 0; f < features; ++f) {
                const size_t cell = static_cast<size_t>(f) * bins + codes[static_cast<size_t>(row) * features + f];
                rg[cell] += g[row];
                rh[cell] += h[row];
                ++rc[cell];
            }
        };
        if (use_subset)
            for (uint32_t r : subset) add(static_cast<int>(r));
        else
            for (int r = 0; r < rows; ++r) add(r);
        for (size_t cell = 0; cell < rg.size(); ++cell) {
            assert(hist.counts[cell] == rc[cell]);
            assert(std::abs(hist.gradients[cell] - rg[cell]) < 1e-2);
            assert(std::abs(hist.hessians[cell] - rh[cell]) < 1e-2);
        }
    }

    // Joint histogram: every pair is filled, and rows missing on either
    // feature land in the trailing missing cell.
    const std::vector<foretree::cuda::FeaturePair> pairs = {{0, 1}, {2, 5}, {7, 36}, {10, 11}, {35, 36}};
    const int reduced = 8;
    const auto joint = engine.build_joint_histograms(pairs, reduced);
    const size_t stride = joint.stride();
    for (size_t p = 0; p < pairs.size(); ++p) {
        uint64_t total = 0, missing = 0;
        for (size_t cell = 0; cell < stride; ++cell)
            total += joint.counts[p * stride + cell];
        for (int r = 0; r < rows; ++r)
            missing += codes[static_cast<size_t>(r) * features + pairs[p].first] == maximum_code ||
                       codes[static_cast<size_t>(r) * features + pairs[p].second] == maximum_code;
        assert(total == static_cast<uint64_t>(rows));
        assert(joint.counts[p * stride + stride - 1] == missing);
    }
    return 0;
}
