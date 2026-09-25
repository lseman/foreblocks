// Row-wise histogram kernel (used for large datasets) must equal the
// feature-major kernel: small nodes (feature groups), large nodes (row blocks),
// a feature subset, missing codes, unit and non-unit hessians.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "foretree/core/histogram_kernel.hpp"

int main() {
    const int N = 60000, P = 9, bins = 32;  // codes 0..30 finite, 31 missing
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> code(0, bins - 1);
    std::vector<uint8_t> row_major(static_cast<size_t>(N) * P), col_major(static_cast<size_t>(N) * P);
    for (int i = 0; i < N; ++i)
        for (int f = 0; f < P; ++f) {
            const auto c = static_cast<uint8_t>(code(rng));
            row_major[static_cast<size_t>(i) * P + f] = c;
            col_major[static_cast<size_t>(f) * N + i] = c;
        }
    std::vector<double> g(N), h(N), ones(N, 1.0);
    for (int i = 0; i < N; ++i) {
        g[i] = std::normal_distribution<double>()(rng);
        h[i] = 0.1 + 0.01 * (i % 13);
    }
    std::vector<size_t> offsets(P);
    std::vector<int> missing(P, bins - 1);
    for (int f = 0; f < P; ++f) offsets[f] = static_cast<size_t>(f) * bins;
    const std::vector<int> subset = {0, 2, 3, 5, 8};
    foretree::ParallelExecutor executor(8);

    for (const bool unit : {false, true}) {
        const auto& hess = unit ? ones : h;
        for (int n : {300, 5000, 60000}) {
            std::vector<int> rows(N);
            std::iota(rows.begin(), rows.end(), 0);
            std::shuffle(rows.begin(), rows.end(), rng);
            rows.resize(n);
            std::sort(rows.begin(), rows.end());
            auto row_at = [&](int s) { return rows[s]; };
            std::vector<double> G1(P * bins, 0), H1(P * bins, 0), G2(P * bins, 0), H2(P * bins, 0);
            std::vector<int> C1(P * bins, 0), C2(P * bins, 0);
            foretree::dispatch_feature_major_histogram(
                unit, std::span<const uint8_t>(col_major), N, n, row_at, std::span<const int>(subset),
                std::span<const size_t>(offsets), std::span<const int>(missing), std::span<const double>(g),
                std::span<const double>(hess), foretree::HistogramOutputView{G1, H1, C1}, executor);
            foretree::dispatch_row_wise_histogram(
                unit, std::span<const uint8_t>(row_major), P, n, row_at, std::span<const int>(subset),
                std::span<const size_t>(offsets), std::span<const int>(missing), std::span<const double>(g),
                std::span<const double>(hess), foretree::HistogramOutputView{G2, H2, C2}, executor);
            for (size_t k = 0; k < G1.size(); ++k) {
                assert(C1[k] == C2[k]);
                assert(std::abs(G1[k] - G2[k]) < 1e-9 * (1.0 + std::abs(G1[k])));
                assert(std::abs(H1[k] - H2[k]) < 1e-9 * (1.0 + std::abs(H1[k])));
            }
        }
    }
    return 0;
}
