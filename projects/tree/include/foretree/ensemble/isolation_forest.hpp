#pragma once
// Isolation Forest (Liu, Ting & Zhou, 2008) with the Extended Isolation Forest
// hyperplane splits of Hariri, Kind & Brunner (2019).
//
// Each tree isolates a random subsample of `max_samples` rows. A node splits
// either on one random feature at a threshold drawn uniformly in the node's
// range (extension_level = 0, classic IF), or on a random hyperplane through a
// point drawn uniformly in the node's bounding box, over
// `extension_level + 1` random features (Extended IF; removes the axis-aligned
// artifacts of the classic score). Trees stop at depth ceil(log2(max_samples)).
//
// The anomaly score of x is s(x) = 2^(-E[h(x)] / c(psi)), where h(x) is the
// path length to x's leaf plus c(leaf size), and c(n) is the average path
// length of an unsuccessful BST search over n points. s is near 1 for
// anomalies and well below 0.5 for inliers. `score_samples` returns -s
// (scikit-learn convention: higher means more normal).
//
// Missing values (NaN): an axis split sends them to a side chosen at fit time
// with probability proportional to that side's share of the node's finite
// values; a hyperplane split imputes the split point for missing coordinates,
// so they do not move the row.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "foretree/core/parallel_executor.hpp"

namespace foretree {

struct IsolationForestConfig {
    int n_estimators = 200;
    // Rows per tree (psi). <= 0 uses min(256, N).
    int max_samples = 256;
    // Fraction (0, 1] of the features each tree may split on.
    double max_features = 1.0;
    // < 0 uses ceil(log2(psi)).
    int max_depth = -1;
    // 0: axis-aligned splits (classic IF). k > 0: hyperplanes over k + 1
    // features (Extended IF). -1: fully extended (all available features).
    int extension_level = 0;
    // Sample rows with replacement.
    bool bootstrap = false;
    // < 0 ("auto"): offset -0.5, i.e. an anomaly score of 0.5 is the boundary,
    // as in the original paper. In (0, 0.5]: the offset is the `contamination`
    // quantile of the training scores.
    double contamination = -1.0;
    uint64_t rng_seed = 42;
};

class IsolationForest {
public:
    explicit IsolationForest(IsolationForestConfig cfg = {}) : cfg_(cfg) {}

    void fit(const double* X, int N, int P) {
        if (!X || N <= 0 || P <= 0)
            throw std::invalid_argument("IsolationForest::fit: empty input");
        if (cfg_.n_estimators <= 0)
            throw std::invalid_argument("IsolationForest: n_estimators must be positive");
        if (!(cfg_.max_features > 0.0 && cfg_.max_features <= 1.0))
            throw std::invalid_argument("IsolationForest: max_features must be in (0, 1]");
        if (cfg_.contamination >= 0.0 && !(cfg_.contamination > 0.0 && cfg_.contamination <= 0.5))
            throw std::invalid_argument("IsolationForest: contamination must be in (0, 0.5] or < 0 for auto");

        P_ = P;
        psi_ = cfg_.max_samples > 0 ? std::min(cfg_.max_samples, cfg_.bootstrap ? cfg_.max_samples : N)
                                    : std::min(256, N);
        psi_ = std::max(psi_, 1);
        max_depth_ = cfg_.max_depth >= 0
                         ? cfg_.max_depth
                         : static_cast<int>(std::ceil(std::log2(std::max(2, psi_))));
        c_psi_ = average_path_length(psi_);
        n_tree_features_ =
            std::clamp(static_cast<int>(std::ceil(cfg_.max_features * P_)), 1, P_);

        trees_.assign(static_cast<size_t>(cfg_.n_estimators), Tree{});
        executor().parallel_for(0, cfg_.n_estimators, 1, [&](int begin, int end) {
            for (int t = begin; t < end; ++t)
                trees_[static_cast<size_t>(t)] = build_tree_(X, N, t);
        });

        fitted_ = true;
        if (cfg_.contamination < 0.0) {
            offset_ = -0.5;
        } else {
            std::vector<double> train = score_samples(X, N, P);
            const size_t k = static_cast<size_t>(
                std::clamp(cfg_.contamination * static_cast<double>(N), 0.0, static_cast<double>(N - 1)));
            std::nth_element(train.begin(), train.begin() + static_cast<std::ptrdiff_t>(k), train.end());
            offset_ = train[k];
        }
    }

    // Mean path length E[h(x)] over the trees (including the c(n) leaf term).
    [[nodiscard]] std::vector<double> mean_path_length(const double* X, int N, int P) const {
        check_input_(X, N, P);
        std::vector<double> out(static_cast<size_t>(N), 0.0);
        const double inv_trees = 1.0 / static_cast<double>(trees_.size());
        executor().parallel_for(0, N, 256, [&](int begin, int end) {
            for (int i = begin; i < end; ++i) {
                const double* x = X + static_cast<size_t>(i) * static_cast<size_t>(P_);
                double total = 0.0;
                for (const Tree& tree : trees_)
                    total += path_length_(tree, x);
                out[static_cast<size_t>(i)] = total * inv_trees;
            }
        });
        return out;
    }

    // s(x) in (0, 1]; higher is more anomalous.
    [[nodiscard]] std::vector<double> anomaly_score(const double* X, int N, int P) const {
        std::vector<double> out = mean_path_length(X, N, P);
        for (double& v : out)
            v = std::exp2(-v / c_psi_);
        return out;
    }

    // -s(x): higher is more normal (scikit-learn convention).
    [[nodiscard]] std::vector<double> score_samples(const double* X, int N, int P) const {
        std::vector<double> out = anomaly_score(X, N, P);
        for (double& v : out)
            v = -v;
        return out;
    }

    // score_samples - offset: negative for outliers.
    [[nodiscard]] std::vector<double> decision_function(const double* X, int N, int P) const {
        std::vector<double> out = score_samples(X, N, P);
        for (double& v : out)
            v -= offset_;
        return out;
    }

    // +1 inlier, -1 outlier.
    [[nodiscard]] std::vector<int> predict(const double* X, int N, int P) const {
        const std::vector<double> d = decision_function(X, N, P);
        std::vector<int> out(d.size());
        for (size_t i = 0; i < d.size(); ++i)
            out[i] = d[i] < 0.0 ? -1 : 1;
        return out;
    }

    // Average path length of an unsuccessful BST search among n points.
    [[nodiscard]] static double average_path_length(double n) noexcept {
        if (n <= 1.0)
            return 0.0;
        if (n <= 2.0)
            return 1.0;
        constexpr double kEulerGamma = 0.5772156649015329;
        return 2.0 * (std::log(n - 1.0) + kEulerGamma) - 2.0 * (n - 1.0) / n;
    }

    [[nodiscard]] double offset() const noexcept { return offset_; }
    [[nodiscard]] int n_trees() const noexcept { return static_cast<int>(trees_.size()); }
    [[nodiscard]] int max_samples() const noexcept { return psi_; }
    [[nodiscard]] int max_depth() const noexcept { return max_depth_; }
    [[nodiscard]] bool fitted() const noexcept { return fitted_; }
    [[nodiscard]] const IsolationForestConfig& config() const noexcept { return cfg_; }

private:
    struct Node {
        int left = -1;       // -1 marks a leaf
        int right = -1;
        int feature = -1;    // axis split feature; -1 for hyperplane splits
        int hp_begin = 0;    // hyperplane terms [hp_begin, hp_end)
        int hp_end = 0;
        double threshold = 0.0;  // axis: x < t goes left; hyperplane: n.x < t goes left
        double leaf_term = 0.0;  // leaves: depth + c(size)
        bool missing_left = false;
    };

    struct Tree {
        std::vector<Node> nodes;
        std::vector<int> hp_features;
        std::vector<double> hp_weights;
        std::vector<double> hp_points;  // split point coordinate per term (imputes NaN)
    };

    static ParallelExecutor& executor() { return *default_parallel_executor(); }

    void check_input_(const double* X, int N, int P) const {
        if (!fitted_)
            throw std::runtime_error("IsolationForest: call fit first");
        if (P != P_)
            throw std::invalid_argument("IsolationForest: feature count differs from fit");
        if (N > 0 && !X)
            throw std::invalid_argument("IsolationForest: null input");
    }

    static uint64_t splitmix64(uint64_t x) noexcept {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    double path_length_(const Tree& tree, const double* x) const {
        int index = 0;
        for (;;) {
            const Node& node = tree.nodes[static_cast<size_t>(index)];
            if (node.left < 0)
                return node.leaf_term;
            bool go_left;
            if (node.feature >= 0) {
                const double v = x[node.feature];
                go_left = std::isnan(v) ? node.missing_left : v < node.threshold;
            } else {
                double dot = 0.0;
                for (int k = node.hp_begin; k < node.hp_end; ++k) {
                    const double v = x[tree.hp_features[static_cast<size_t>(k)]];
                    dot += tree.hp_weights[static_cast<size_t>(k)] *
                           (std::isnan(v) ? tree.hp_points[static_cast<size_t>(k)] : v);
                }
                go_left = dot < node.threshold;
            }
            index = go_left ? node.left : node.right;
        }
    }

    Tree build_tree_(const double* X, int N, int t) const {
        std::mt19937_64 rng(splitmix64(cfg_.rng_seed ^ splitmix64(static_cast<uint64_t>(t) + 1)));
        const auto P = static_cast<size_t>(P_);

        // Row subsample.
        std::vector<int> rows(static_cast<size_t>(psi_));
        if (cfg_.bootstrap) {
            std::uniform_int_distribution<int> pick(0, N - 1);
            for (int& r : rows)
                r = pick(rng);
        } else if (psi_ == N) {
            std::iota(rows.begin(), rows.end(), 0);
        } else {
            // Floyd's algorithm: psi distinct rows in O(psi).
            std::vector<int> chosen;
            chosen.reserve(static_cast<size_t>(psi_));
            std::vector<char> taken;
            const bool dense = static_cast<int64_t>(N) <= 64LL * psi_;
            if (dense)
                taken.assign(static_cast<size_t>(N), 0);
            std::vector<int> seen;
            for (int j = N - psi_; j < N; ++j) {
                const int r = std::uniform_int_distribution<int>(0, j)(rng);
                bool present;
                if (dense) {
                    present = taken[static_cast<size_t>(r)] != 0;
                } else {
                    present = std::find(seen.begin(), seen.end(), r) != seen.end();
                }
                const int value = present ? j : r;
                if (dense)
                    taken[static_cast<size_t>(value)] = 1;
                else
                    seen.push_back(value);
                chosen.push_back(value);
            }
            rows = std::move(chosen);
        }

        // Feature subset for this tree.
        std::vector<int> features(P);
        std::iota(features.begin(), features.end(), 0);
        if (n_tree_features_ < P_) {
            std::shuffle(features.begin(), features.end(), rng);
            features.resize(static_cast<size_t>(n_tree_features_));
        }
        const int n_features = static_cast<int>(features.size());
        const int hp_dims = cfg_.extension_level < 0
                                ? n_features
                                : std::min(n_features, cfg_.extension_level + 1);

        Tree tree;
        tree.nodes.reserve(static_cast<size_t>(2 * psi_));
        struct Work {
            int node, lo, hi, depth;
        };
        std::vector<Work> stack;
        tree.nodes.emplace_back();
        stack.push_back({0, 0, psi_, 0});
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::vector<int> candidates = features;
        std::vector<double> lows(static_cast<size_t>(hp_dims)), highs(static_cast<size_t>(hp_dims));
        std::vector<int> dims(static_cast<size_t>(hp_dims));
        std::vector<double> projection(static_cast<size_t>(psi_));

        auto value = [&](int row, int feature) { return X[static_cast<size_t>(row) * P + static_cast<size_t>(feature)]; };
        auto finite_range = [&](int lo, int hi, int feature, double& mn, double& mx) {
            mn = std::numeric_limits<double>::infinity();
            mx = -mn;
            for (int i = lo; i < hi; ++i) {
                const double v = value(rows[static_cast<size_t>(i)], feature);
                if (std::isnan(v))
                    continue;
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            return mn < mx;
        };

        while (!stack.empty()) {
            const Work w = stack.back();
            stack.pop_back();
            const int size = w.hi - w.lo;
            auto make_leaf = [&] {
                tree.nodes[static_cast<size_t>(w.node)].leaf_term =
                    static_cast<double>(w.depth) + average_path_length(static_cast<double>(size));
            };
            if (size <= 1 || w.depth >= max_depth_) {
                make_leaf();
                continue;
            }

            Node split;
            int mid = w.lo;
            if (hp_dims <= 1) {
                // Classic IF: a random feature that is not constant on the node.
                std::shuffle(candidates.begin(), candidates.end(), rng);
                int feature = -1;
                double mn = 0.0, mx = 0.0;
                for (int f : candidates) {
                    if (finite_range(w.lo, w.hi, f, mn, mx)) {
                        feature = f;
                        break;
                    }
                }
                if (feature < 0) {
                    make_leaf();
                    continue;
                }
                double threshold = mn + unit(rng) * (mx - mn);
                if (threshold <= mn)
                    threshold = std::nextafter(mn, mx);  // keep both sides non-empty
                int finite_left = 0, finite = 0;
                for (int i = w.lo; i < w.hi; ++i) {
                    const double v = value(rows[static_cast<size_t>(i)], feature);
                    if (!std::isnan(v)) {
                        ++finite;
                        finite_left += v < threshold;
                    }
                }
                split.feature = feature;
                split.threshold = threshold;
                split.missing_left = unit(rng) * finite < finite_left;
                mid = static_cast<int>(
                    std::partition(rows.begin() + w.lo, rows.begin() + w.hi,
                                   [&](int r) {
                                       const double v = value(r, feature);
                                       return std::isnan(v) ? split.missing_left : v < threshold;
                                   }) -
                    rows.begin());
            } else {
                // Extended IF: hyperplane over hp_dims random non-constant features.
                std::shuffle(candidates.begin(), candidates.end(), rng);
                int found = 0;
                for (int f : candidates) {
                    double mn, mx;
                    if (!finite_range(w.lo, w.hi, f, mn, mx))
                        continue;
                    dims[static_cast<size_t>(found)] = f;
                    lows[static_cast<size_t>(found)] = mn;
                    highs[static_cast<size_t>(found)] = mx;
                    if (++found == hp_dims)
                        break;
                }
                if (found == 0) {
                    make_leaf();
                    continue;
                }
                split.hp_begin = static_cast<int>(tree.hp_features.size());
                double threshold = 0.0;
                for (int k = 0; k < found; ++k) {
                    // Weights scaled by the node's range so every feature
                    // matters regardless of its units.
                    const double span = highs[static_cast<size_t>(k)] - lows[static_cast<size_t>(k)];
                    const double weight = gauss(rng) / span;
                    const double point = lows[static_cast<size_t>(k)] + unit(rng) * span;
                    tree.hp_features.push_back(dims[static_cast<size_t>(k)]);
                    tree.hp_weights.push_back(weight);
                    tree.hp_points.push_back(point);
                    threshold += weight * point;
                }
                split.hp_end = static_cast<int>(tree.hp_features.size());
                split.threshold = threshold;
                for (int i = w.lo; i < w.hi; ++i) {
                    const int r = rows[static_cast<size_t>(i)];
                    double dot = 0.0;
                    for (int k = split.hp_begin; k < split.hp_end; ++k) {
                        const double v = value(r, tree.hp_features[static_cast<size_t>(k)]);
                        dot += tree.hp_weights[static_cast<size_t>(k)] *
                               (std::isnan(v) ? tree.hp_points[static_cast<size_t>(k)] : v);
                    }
                    projection[static_cast<size_t>(i - w.lo)] = dot;
                }
                // Stable partition by projection (projection is indexed by position).
                std::vector<int> left, right;
                for (int i = w.lo; i < w.hi; ++i)
                    (projection[static_cast<size_t>(i - w.lo)] < threshold ? left : right)
                        .push_back(rows[static_cast<size_t>(i)]);
                std::copy(left.begin(), left.end(), rows.begin() + w.lo);
                std::copy(right.begin(), right.end(), rows.begin() + w.lo + static_cast<int>(left.size()));
                mid = w.lo + static_cast<int>(left.size());
            }

            const int left = static_cast<int>(tree.nodes.size());
            split.left = left;
            split.right = left + 1;
            tree.nodes[static_cast<size_t>(w.node)] = split;
            tree.nodes.emplace_back();
            tree.nodes.emplace_back();
            stack.push_back({left, w.lo, mid, w.depth + 1});
            stack.push_back({left + 1, mid, w.hi, w.depth + 1});
        }
        return tree;
    }

    IsolationForestConfig cfg_;
    std::vector<Tree> trees_;
    int P_ = 0;
    int psi_ = 0;
    int max_depth_ = 0;
    int n_tree_features_ = 0;
    double c_psi_ = 1.0;
    double offset_ = -0.5;
    bool fitted_ = false;
};

} // namespace foretree
