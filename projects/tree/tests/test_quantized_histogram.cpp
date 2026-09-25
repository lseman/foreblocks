// Quantized histogram kernel: with gradients that are exact multiples of the
// scales, it must reproduce the double kernel exactly, for small nodes and for
// the (feature x row block) path used by large nodes.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "foretree/core/histogram_kernel.hpp"

int main() {
    const int N = 120000, F = 5, bins = 64;
    const double g_scale = 0.01, h_scale = 0.002;
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> code(0, bins - 1), g_level(-127, 127), h_level(0, 255);
    std::vector<uint16_t> codes(static_cast<size_t>(N) * F);  // feature-major
    for (auto& c : codes) c = static_cast<uint16_t>(code(rng));  // code == bins-1 is the missing bin
    std::vector<double> g(N), h(N);
    std::vector<int32_t> packed(N);
    for (int i = 0; i < N; ++i) {
        const int gl = g_level(rng), hl = h_level(rng);
        g[i] = gl * g_scale;
        h[i] = hl * h_scale;
        packed[i] = foretree::pack_quantized(gl, hl);
    }
    std::vector<int> features(F);
    std::iota(features.begin(), features.end(), 0);
    std::vector<size_t> offsets(F);
    std::vector<int> missing(F, bins - 1);
    for (int f = 0; f < F; ++f) offsets[f] = static_cast<size_t>(f) * bins;
    foretree::ParallelExecutor executor(8);
    const foretree::QuantizedGradients quantized{packed.data(), g_scale, h_scale};

    for (int n : {1, 700, 5000, N}) {
        std::vector<int> rows(N);
        std::iota(rows.begin(), rows.end(), 0);
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(n);
        std::sort(rows.begin(), rows.end());
        auto row_at = [&](int s) { return rows[s]; };

        std::vector<double> G(F * bins, 0.0), H(F * bins, 0.0), Gq(F * bins, 0.0), Hq(F * bins, 0.0);
        std::vector<int> C(F * bins, 0), Cq(F * bins, 0);
        foretree::dispatch_feature_major_histogram(
            false, std::span<const uint16_t>(codes), N, n, row_at, std::span<const int>(features),
            std::span<const size_t>(offsets), std::span<const int>(missing), std::span<const double>(g),
            std::span<const double>(h), foretree::HistogramOutputView{G, H, C}, executor);
        foretree::QuantizedFeatureMajorHistogramKernel<uint16_t>::build(
            std::span<const uint16_t>(codes), N, n, row_at, std::span<const int>(features),
            std::span<const size_t>(offsets), std::span<const int>(missing), quantized,
            foretree::HistogramOutputView{Gq, Hq, Cq}, executor);
        for (size_t k = 0; k < G.size(); ++k) {
            assert(C[k] == Cq[k]);
            assert(std::abs(G[k] - Gq[k]) < 1e-9 * (1.0 + std::abs(G[k])));
            assert(std::abs(H[k] - Hq[k]) < 1e-9 * (1.0 + std::abs(H[k])));
        }
    }
    return 0;
}
