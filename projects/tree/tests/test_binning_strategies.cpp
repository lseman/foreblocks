// Invariants every binning strategy must hold, because bin codes are compared
// with thresholds (code <= t goes left) and edges are binary-searched:
// edges strictly increasing, codes monotone in the value, a continuous
// feature keeps most of its resolution, and each category of a low-cardinality
// feature gets its own bin.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "foretree/core/gradient_hist_system.hpp"

namespace {

using foretree::GradientHistogramSystem;
using foretree::HistogramConfig;

struct Binned {
    std::vector<double> edges;
    std::vector<int> codes;
    int missing = 0;
};

Binned bin(const std::string& method, const std::vector<double>& x, const std::vector<double>& g) {
    const int n = static_cast<int>(x.size());
    std::vector<double> h(x.size(), 1.0);
    HistogramConfig cfg;
    cfg.method = method;
    GradientHistogramSystem ghs(cfg);
    ghs.fit_bins(x.data(), n, 1, g.data(), h.data());
    Binned out;
    out.edges = ghs.feature_bins(0).edges;
    out.missing = ghs.total_bins(0) - 1;
    auto codes = ghs.prebin_dataset_compact(x.data(), n, 1);
    codes->visit_codes([&](auto c) { out.codes.assign(c.begin(), c.end()); });
    return out;
}

int failures = 0;
void expect(bool ok, const std::string& method, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL [%s] %s\n", method.c_str(), what);
        ++failures;
    }
}

}  // namespace

int main() {
    const std::vector<std::string> methods = {"quantile", "hist",      "kmeans",
                                              "two_stage", "adaptive", "grad_aware",
                                              "categorical_gradient"};
    std::mt19937 rng(0);

    // Continuous feature; gradients from a noisy smooth target.
    const int n = 20000;
    std::vector<double> x(n), g(n);
    for (int i = 0; i < n; ++i) {
        x[i] = std::normal_distribution<double>()(rng);
        g[i] = -(std::sin(x[i]) + std::normal_distribution<double>(0, 0.5)(rng));
    }
    // Low-cardinality categorical feature whose category effects are not
    // ordered like the category values.
    const double effect[6] = {0.0, -1.6, 0.3, 2.2, -0.5, 1.0};
    std::vector<double> xc(12000), gc(12000);
    for (int i = 0; i < 12000; ++i) {
        xc[i] = i % 6;
        gc[i] = -(effect[i % 6] + std::normal_distribution<double>(0, 0.3)(rng));
    }

    for (const auto& method : methods) {
        const Binned b = bin(method, x, g);
        bool increasing = true;
        for (size_t i = 1; i < b.edges.size(); ++i)
            increasing = increasing && b.edges[i] > b.edges[i - 1];
        expect(increasing, method, "continuous: edges strictly increasing");

        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int c) { return x[a] < x[c]; });
        bool monotone = true;
        for (int k = 1; k < n; ++k)
            monotone = monotone && b.codes[order[k]] >= b.codes[order[k - 1]];
        expect(monotone, method, "continuous: codes monotone in value");

        std::map<int, int> count;
        for (int c : b.codes) count[c]++;
        int largest = 0;
        for (auto& [code, k] : count) largest = std::max(largest, k);
        expect(count.size() >= 16, method, "continuous: at least 16 bins used");
        expect(largest <= n / 4, method, "continuous: no bin holds a quarter of the rows");

        const Binned bc = bin(method, xc, gc);
        std::map<int, std::set<int>> categories_per_code;
        for (int i = 0; i < 12000; ++i) categories_per_code[bc.codes[i]].insert(static_cast<int>(xc[i]));
        bool separate = categories_per_code.size() == 6;
        for (auto& [code, cats] : categories_per_code) separate = separate && cats.size() == 1;
        expect(separate, method, "categorical: one bin per category");
    }
    if (failures)
        std::fprintf(stderr, "%d invariant violations\n", failures);
    assert(failures == 0);
    return 0;
}
