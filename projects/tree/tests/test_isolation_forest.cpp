// Isolation Forest: scores, determinism, contamination, missing values and
// the Extended IF split.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "foretree/ensemble/isolation_forest.hpp"

namespace {

// n inliers ~ N(0, 1)^p, followed by n_out outliers at distance ~6.
std::vector<double> make_data(int n, int n_out, int p, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> x(static_cast<size_t>(n + n_out) * p);
    for (int i = 0; i < n + n_out; ++i)
        for (int j = 0; j < p; ++j)
            x[static_cast<size_t>(i) * p + j] = gauss(rng) + (i >= n ? 6.0 : 0.0);
    return x;
}

double mean(const std::vector<double>& v, int begin, int end) {
    double s = 0.0;
    for (int i = begin; i < end; ++i)
        s += v[static_cast<size_t>(i)];
    return s / (end - begin);
}

void test_average_path_length() {
    using foretree::IsolationForest;
    assert(IsolationForest::average_path_length(1) == 0.0);
    assert(IsolationForest::average_path_length(2) == 1.0);
    assert(std::abs(IsolationForest::average_path_length(256) - 10.2448) < 1e-3);
}

void test_outliers_score_higher(int extension_level) {
    const int n = 2000, n_out = 40, p = 6;
    const auto x = make_data(n, n_out, p, 1);
    foretree::IsolationForestConfig cfg;
    cfg.extension_level = extension_level;
    foretree::IsolationForest forest(cfg);
    forest.fit(x.data(), n + n_out, p);
    const auto s = forest.anomaly_score(x.data(), n + n_out, p);
    for (double v : s)
        assert(v > 0.0 && v <= 1.0);
    const double inlier = mean(s, 0, n), outlier = mean(s, n, n + n_out);
    assert(outlier > 0.6 && inlier < 0.5 && outlier > inlier + 0.15);
    // Every outlier ranks above the median inlier.
    std::vector<double> inliers(s.begin(), s.begin() + n);
    std::nth_element(inliers.begin(), inliers.begin() + n / 2, inliers.end());
    for (int i = n; i < n + n_out; ++i)
        assert(s[static_cast<size_t>(i)] > inliers[static_cast<size_t>(n / 2)]);
    assert(forest.max_depth() == 8);  // ceil(log2(256))
}

void test_deterministic_and_seed_sensitive() {
    const auto x = make_data(500, 10, 3, 2);
    auto fit_scores = [&](uint64_t seed) {
        foretree::IsolationForestConfig cfg;
        cfg.rng_seed = seed;
        foretree::IsolationForest forest(cfg);
        forest.fit(x.data(), 510, 3);
        return forest.score_samples(x.data(), 510, 3);
    };
    assert(fit_scores(7) == fit_scores(7));
    assert(fit_scores(7) != fit_scores(8));
}

void test_contamination_threshold() {
    const int n = 1000, n_out = 20, p = 4;
    const auto x = make_data(n, n_out, p, 3);
    foretree::IsolationForestConfig cfg;
    cfg.contamination = 0.02;
    foretree::IsolationForest forest(cfg);
    forest.fit(x.data(), n + n_out, p);
    const auto labels = forest.predict(x.data(), n + n_out, p);
    const auto flagged = std::count(labels.begin(), labels.end(), -1);
    assert(std::abs(static_cast<double>(flagged) - 0.02 * (n + n_out)) <= 2.0);
    int caught = 0;
    for (int i = n; i < n + n_out; ++i)
        caught += labels[static_cast<size_t>(i)] == -1;
    assert(caught >= n_out - 2);
}

void test_missing_values_and_constant_features() {
    const int n = 800, n_out = 16, p = 5;
    auto x = make_data(n, n_out, p, 4);
    std::mt19937 rng(5);
    std::bernoulli_distribution missing(0.15);
    for (double& v : x)
        if (missing(rng))
            v = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < n + n_out; ++i)
        x[static_cast<size_t>(i) * p + 4] = 1.0;  // constant column is never split on
    for (int level : {0, 2}) {
        foretree::IsolationForestConfig cfg;
        cfg.extension_level = level;
        foretree::IsolationForest forest(cfg);
        forest.fit(x.data(), n + n_out, p);
        const auto s = forest.anomaly_score(x.data(), n + n_out, p);
        for (double v : s)
            assert(std::isfinite(v));
        assert(mean(s, n, n + n_out) > mean(s, 0, n) + 0.1);
    }
}

void test_input_validation() {
    foretree::IsolationForest forest;
    const std::vector<double> x = {0.0, 1.0};
    bool threw = false;
    try {
        (void)forest.score_samples(x.data(), 1, 2);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    forest.fit(x.data(), 1, 2);  // one row: a single leaf, still valid
    threw = false;
    try {
        (void)forest.score_samples(x.data(), 2, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    test_average_path_length();
    test_outliers_score_higher(0);
    test_outliers_score_higher(1);
    test_outliers_score_higher(-1);
    test_deterministic_and_seed_sensitive();
    test_contamination_threshold();
    test_missing_values_and_constant_features();
    test_input_validation();
    return 0;
}
