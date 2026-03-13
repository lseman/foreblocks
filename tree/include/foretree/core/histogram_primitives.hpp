#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace foretree {

struct HistogramConfig {
    std::string method =
        "kmeans";  // "hist"(uniform) | "quantile" | "kmeans" | "grad_aware" | "two_stage" | "adaptive"
    int max_bins = 256;
    bool use_missing_bin = true;

    // quantile / grad-aware
    int coarse_bins = 64;       // base for grad_aware auto-tuning
    bool density_aware = true;  // reserved for future

    // Adaptive binning parameters
    int min_bins = 8;              // minimum bins per feature
    int target_bins = 32;          // target bins for "normal" features
    bool adaptive_binning = true;  // enable per-feature adaptive bin counts
    double importance_threshold = 0.1;  // features above this get more bins
    double complexity_threshold = 0.7;  // features above this get more bins

    // Feature importance weighting
    bool use_feature_importance = false;
    std::vector<double>
        feature_importance_weights;  // if provided, override auto-detection

    // sketch-ish knobs (reserved for future approximations)
    double subsample_ratio = 0.3;
    int min_sketch_size = 10000;

    // threading
    bool use_parallel = false;
    int max_workers = 8;

    // rng (for any sampling we might add)
    uint64_t rng_seed = 42;

    // regularization-esque eps
    double eps = 1e-12;

    int total_bins() const { return max_bins + (use_missing_bin ? 1 : 0); }
    int missing_bin_id() const { return use_missing_bin ? max_bins : -1; }
};

struct FeatureStats {
    double variance = 0.0;
    double gradient_variance = 0.0;
    double gradient_range = 0.0;
    double value_range = 0.0;
    int unique_count = 0;
    double complexity_score = 0.0;
    double importance_score = 0.0;
    bool is_categorical = false;

    // Computed bin allocation
    int suggested_bins = 32;
    std::string allocation_reason = "default";
};

class VariableBinLayout {
private:
    std::vector<size_t> feature_offsets_;
    std::vector<uint16_t> bins_per_feature_;
    size_t total_histogram_size_ = 0;

public:
    void initialize(const std::vector<int>& bins_per_feat) {
        const size_t P = bins_per_feat.size();
        feature_offsets_.resize(P + 1);
        bins_per_feature_.resize(P);

        feature_offsets_[0] = 0;
        for (size_t i = 0; i < P; ++i) {
            bins_per_feature_[i] = static_cast<uint16_t>(bins_per_feat[i]);
            feature_offsets_[i + 1] = feature_offsets_[i] + bins_per_feat[i];
        }
        total_histogram_size_ = feature_offsets_[P];
    }

    // O(1) offset calculation
    inline size_t get_offset(int feature, int bin) const {
        return feature_offsets_[feature] + bin;
    }

    inline size_t feature_offset(int feature) const {
        return feature_offsets_[feature];
    }

    inline uint16_t bins_for_feature(int feature) const {
        return bins_per_feature_[feature];
    }

    inline size_t total_size() const { return total_histogram_size_; }

    inline int num_features() const {
        return static_cast<int>(bins_per_feature_.size());
    }
};

class HistogramAccumulator {
private:
    const VariableBinLayout* layout_;
    std::vector<double> hist_g_, hist_h_;
    std::vector<int> hist_c_;

public:
    explicit HistogramAccumulator(const VariableBinLayout* layout)
        : layout_(layout) {
        if (layout_) {
            const size_t size = layout_->total_size();
            hist_g_.resize(size, 0.0);
            hist_h_.resize(size, 0.0);
            hist_c_.resize(size, 0);
        }
    }

    void clear() {
        std::fill(hist_g_.begin(), hist_g_.end(), 0.0);
        std::fill(hist_h_.begin(), hist_h_.end(), 0.0);
        std::fill(hist_c_.begin(), hist_c_.end(), 0);
    }

    template <bool WITH_COUNTS, typename Gdouble, typename Hdouble>
    void accumulate_samples(const uint16_t* codes, const Gdouble* g,
                            const Hdouble* h, const int* sample_indices,
                            int n_samples, int P) {
        if (!layout_ || !codes || hist_g_.size() != layout_->total_size())
            return;

        auto accumulate_one = [&](int i) {
            const uint16_t* row = codes + static_cast<size_t>(i) * P;
            const double gi = static_cast<double>(g[i]);
            const double hi = static_cast<double>(h[i]);

            for (int j = 0; j < P; ++j) {
                const uint16_t bin = row[j];
                const uint16_t max_bins = layout_->bins_for_feature(j);
                if (bin >= max_bins) continue;

                const size_t offset = layout_->get_offset(j, bin);
                if (offset >= hist_g_.size()) continue;

                hist_g_[offset] += gi;
                hist_h_[offset] += hi;
                if constexpr (WITH_COUNTS) {
                    hist_c_[offset] += 1;
                }
            }
        };

        if (!sample_indices) {
            for (int i = 0; i < n_samples; ++i) accumulate_one(i);
        } else {
            for (int t = 0; t < n_samples; ++t) {
                const int sample_idx = sample_indices[t];
                if (sample_idx >= 0) {
                    accumulate_one(sample_idx);
                }
            }
        }
    }

    const std::vector<double>& gradients() const { return hist_g_; }
    const std::vector<double>& hessians() const { return hist_h_; }
    const std::vector<int>& counts() const { return hist_c_; }

    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    take_results() {
        return {std::move(hist_g_), std::move(hist_h_), std::move(hist_c_)};
    }

    std::pair<std::vector<double>, std::vector<double>> take_gh_results() {
        return {std::move(hist_g_), std::move(hist_h_)};
    }
};

class StreamingQuantileBuilder {
private:
    struct Bucket {
        double value;
        double weight;
        int count;

        Bucket(double v, double w, int c) : value(v), weight(w), count(c) {}
    };

    std::vector<Bucket> buckets_;
    double total_weight_ = 0.0;
    int max_buckets_;

public:
    explicit StreamingQuantileBuilder(int max_buckets = 1000)
        : max_buckets_(max_buckets) {}

    void add_point(double value, double weight = 1.0) {
        if (!std::isfinite(value) || !std::isfinite(weight) || weight <= 0.0)
            return;

        auto it = std::lower_bound(
            buckets_.begin(), buckets_.end(), value,
            [](const Bucket& b, double v) { return b.value < v; });

        if (it != buckets_.end() && std::abs(it->value - value) < 1e-12) {
            it->weight += weight;
            it->count += 1;
        } else {
            buckets_.emplace(it, value, weight, 1);
        }

        total_weight_ += weight;

        if (static_cast<int>(buckets_.size()) > max_buckets_) {
            compress();
        }
    }

    std::vector<double> get_quantile_edges(int n_bins) const {
        if (buckets_.empty()) return {0.0, 1.0};
        if (n_bins <= 1)
            return {buckets_.front().value - 1e-12,
                    buckets_.back().value + 1e-12};

        std::vector<double> edges(n_bins + 1);
        edges[0] = buckets_.front().value - 1e-12;
        edges[n_bins] = buckets_.back().value + 1e-12;

        std::vector<double> cum_weights(buckets_.size());
        cum_weights[0] = buckets_[0].weight;
        for (size_t i = 1; i < buckets_.size(); ++i) {
            cum_weights[i] = cum_weights[i - 1] + buckets_[i].weight;
        }

        for (int i = 1; i < n_bins; ++i) {
            const double target =
                (static_cast<double>(i) / n_bins) * total_weight_;
            auto it = std::lower_bound(cum_weights.begin(), cum_weights.end(),
                                       target);
            const size_t idx = (it == cum_weights.end())
                                   ? (buckets_.size() - 1)
                                   : (it - cum_weights.begin());
            edges[i] = buckets_[idx].value;
        }

        for (int i = 1; i <= n_bins; ++i) {
            if (edges[i] <= edges[i - 1]) {
                edges[i] = edges[i - 1] + 1e-12;
            }
        }

        return edges;
    }

private:
    void compress() {
        if (buckets_.size() <= 2) return;

        std::vector<Bucket> new_buckets;
        new_buckets.reserve(max_buckets_);

        const size_t keep = std::max<size_t>(2, static_cast<size_t>(max_buckets_));
        const size_t step = std::max<size_t>(1, buckets_.size() / keep);

        for (size_t i = 0; i < buckets_.size(); i += step) {
            new_buckets.push_back(buckets_[i]);
        }

        if (new_buckets.front().value != buckets_.front().value) {
            new_buckets.insert(new_buckets.begin(), buckets_.front());
        }
        if (new_buckets.back().value != buckets_.back().value) {
            new_buckets.push_back(buckets_.back());
        }

        buckets_ = std::move(new_buckets);
    }
};

inline FeatureStats analyze_feature_importance(
    const std::vector<double>& values, const std::vector<double>& gradients,
    const std::vector<double>& hessians, const HistogramConfig& cfg) {
    FeatureStats stats;
    (void)hessians;

    if (values.empty()) {
        stats.suggested_bins = cfg.min_bins;
        stats.allocation_reason = "empty_feature";
        return stats;
    }

    double val_min = std::numeric_limits<double>::max();
    double val_max = std::numeric_limits<double>::lowest();
    double val_sum = 0.0, val_sq = 0.0;
    double grad_sum = 0.0, grad_sq = 0.0;
    double grad_min = std::numeric_limits<double>::max();
    double grad_max = std::numeric_limits<double>::lowest();

    std::vector<double> unique_vals;
    unique_vals.reserve(std::min(values.size(), size_t{1000}));

    size_t finite_count = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double v = values[i];
        const double g = (i < gradients.size()) ? gradients[i] : 0.0;

        if (std::isfinite(v) && std::isfinite(g)) {
            finite_count++;
            val_min = std::min(val_min, v);
            val_max = std::max(val_max, v);
            val_sum += v;
            val_sq += v * v;

            grad_min = std::min(grad_min, g);
            grad_max = std::max(grad_max, g);
            grad_sum += g;
            grad_sq += g * g;

            if (unique_vals.size() < 1000) unique_vals.push_back(v);
        }
    }

    if (finite_count == 0) {
        stats.suggested_bins = cfg.min_bins;
        stats.allocation_reason = "no_finite_values";
        return stats;
    }

    const double n = static_cast<double>(finite_count);
    const double val_mean = val_sum / n;
    const double grad_mean = grad_sum / n;

    stats.variance = (n > 1) ? (val_sq - val_sum * val_mean) / (n - 1) : 0.0;
    stats.gradient_variance =
        (n > 1) ? (grad_sq - grad_sum * grad_mean) / (n - 1) : 0.0;
    stats.value_range = val_max - val_min;
    stats.gradient_range = grad_max - grad_min;

    std::sort(unique_vals.begin(), unique_vals.end());
    unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()),
                      unique_vals.end());
    stats.unique_count = static_cast<int>(unique_vals.size());

    stats.is_categorical =
        (stats.unique_count <= std::min(cfg.max_bins / 4, 32));

    if (finite_count >= 10) {
        const size_t sample_size = std::min(finite_count, size_t{1000});
        double complexity_sum = 0.0;
        int complexity_count = 0;

        for (size_t i = 2; i < sample_size && i < values.size(); ++i) {
            const double g_curr = gradients[i];
            const double g_prev = gradients[i - 1];
            const double g_prev2 = gradients[i - 2];

            if (std::isfinite(g_curr) && std::isfinite(g_prev) &&
                std::isfinite(g_prev2)) {
                const double second_diff = g_curr - 2.0 * g_prev + g_prev2;
                complexity_sum += std::abs(second_diff);
                complexity_count++;
            }
        }

        if (complexity_count > 0) {
            stats.complexity_score = complexity_sum / complexity_count;
        }
    }

    const double norm_grad_var =
        stats.gradient_variance / (1.0 + std::abs(grad_mean));
    const double norm_val_range =
        stats.value_range / (1.0 + std::abs(val_mean));
    stats.importance_score = std::sqrt(norm_grad_var * norm_val_range);

    if (stats.is_categorical) {
        stats.suggested_bins = std::min(stats.unique_count, cfg.max_bins);
        stats.allocation_reason = "categorical";
    } else if (stats.importance_score > cfg.importance_threshold &&
               stats.complexity_score > cfg.complexity_threshold) {
        const double factor =
            1.5 + 0.5 * stats.importance_score + 0.3 * stats.complexity_score;
        stats.suggested_bins = std::min(
            cfg.max_bins,
            std::max(cfg.min_bins, static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_importance_complex";
    } else if (stats.importance_score > cfg.importance_threshold) {
        const double factor = 1.2 + 0.4 * stats.importance_score;
        stats.suggested_bins = std::min(
            cfg.max_bins,
            std::max(cfg.min_bins, static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_importance";
    } else if (stats.complexity_score > cfg.complexity_threshold) {
        const double factor = 1.1 + 0.3 * stats.complexity_score;
        stats.suggested_bins = std::min(
            cfg.max_bins,
            std::max(cfg.min_bins, static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_complexity";
    } else if (stats.unique_count < cfg.min_bins) {
        stats.suggested_bins = std::max(2, stats.unique_count);
        stats.allocation_reason = "few_unique_values";
    } else {
        stats.suggested_bins = cfg.target_bins;
        stats.allocation_reason = "normal";
    }

    return stats;
}

struct FeatureBins {
    std::vector<double> edges;  // strictly increasing, size = nb+1
    bool is_uniform = false;
    std::string strategy =
        "uniform";  // "uniform"|"quantile"|"categorical"|"kmeans"|"grad_aware"|"two_stage"|"adaptive"
    double lo = 0.0;
    double width = 1.0;

    FeatureStats stats;

    int n_bins() const {
        return static_cast<int>(edges.empty() ? 0 : (edges.size() - 1));
    }
};

inline void _check_uniform(FeatureBins& b, double tol = 1e-9) {
    const int nb = b.n_bins();
    if (nb <= 1) {
        b.is_uniform = true;
        if (!b.edges.empty()) {
            b.lo = b.edges.front();
            b.width = (nb == 1 && b.edges.size() >= 2) ? (b.edges[1] - b.edges[0]) : 1.0;
        } else {
            b.lo = 0.0;
            b.width = 1.0;
        }
        return;
    }
    double total = 0.0, max_dev = 0.0;
    for (int k = 0; k < nb; ++k) total += (b.edges[k + 1] - b.edges[k]);
    if (total <= 0.0) {
        b.is_uniform = false;
        return;
    }

    const double meanw = total / nb;
    for (int k = 0; k < nb; ++k) {
        const double w = (b.edges[k + 1] - b.edges[k]);
        max_dev = std::max(max_dev, std::abs(w - meanw));
    }
    b.is_uniform = (max_dev <= tol * std::max(1.0, std::abs(meanw)));
    if (b.is_uniform) {
        b.lo = b.edges.front();
        b.width = meanw;
    }
}

inline std::vector<double> _midpoint_edges_of_unique(
    const std::vector<double>& uniq) {
    if (uniq.empty()) return {0.0, 1.0};
    if (uniq.size() == 1) {
        const double x = uniq[0];
        return {x - 1e-12, x + 1e-12};
    }
    std::vector<double> e(uniq.size() + 1);
    e.front() = uniq.front() - 1e-12;
    e.back() = uniq.back() + 1e-12;
    for (size_t i = 1; i < uniq.size(); ++i)
        e[i] = 0.5 * (uniq[i - 1] + uniq[i]);
    return e;
}

inline std::vector<double> exact_quantile_edges(const std::vector<double>& vals,
                                                int nb) {
    if (vals.empty()) return {0.0, 1.0};
    nb = std::max(1, nb);
    std::vector<double> s = vals;
    std::sort(s.begin(), s.end());

    std::vector<double> e;
    e.reserve(nb + 1);
    const size_t n = s.size();
    for (int i = 0; i <= nb; ++i) {
        const double q = static_cast<double>(i) / nb;
        const double pos = q * (n - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        const double w = pos - lo;
        const double v = (1.0 - w) * s[lo] + w * s[hi];
        e.push_back(v);
    }
    return e;
}

inline std::vector<double> weighted_quantile_edges(
    const std::vector<double>& vals, const std::vector<double>& wts, int nb) {
    if (vals.empty()) return {0.0, 1.0};
    nb = std::max(1, nb);

    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(vals.size());

    for (size_t i = 0; i < vals.size(); ++i) {
        const double v = vals[i];
        const double w = (i < wts.size() ? wts[i] : 1.0);
        if (std::isfinite(v) && std::isfinite(w) && w > 0.0) {
            pairs.emplace_back(v, w);
        }
    }

    if (pairs.empty()) return {0.0, 1.0};

    std::sort(pairs.begin(), pairs.end());

    std::vector<double> cum_weights(pairs.size());
    cum_weights[0] = pairs[0].second;
    for (size_t i = 1; i < pairs.size(); ++i) {
        cum_weights[i] = cum_weights[i - 1] + pairs[i].second;
    }
    double total_weight = cum_weights.back();

    std::vector<double> edges(nb + 1);
    edges[0] = pairs.front().first - 1e-12;
    edges[nb] = pairs.back().first + 1e-12;

    for (int i = 1; i < nb; ++i) {
        const double target = (double(i) / nb) * total_weight;
        auto it = std::lower_bound(cum_weights.begin(), cum_weights.end(), target);
        size_t idx = (it == cum_weights.end()) ? (pairs.size() - 1)
                                               : (it - cum_weights.begin());
        edges[i] = pairs[idx].first;
    }

    for (int i = 1; i <= nb; ++i) {
        if (edges[i] <= edges[i - 1]) {
            edges[i] = edges[i - 1] + 1e-12;
        }
    }

    return edges;
}

inline double gradient_complexity(const std::vector<double>& /*v_sorted*/,
                                  const std::vector<double>& g_sorted) {
    const size_t n = g_sorted.size();
    if (n < 3) return 0.45;

    const size_t max_samples = 1000;
    const size_t step = std::max(size_t{1}, n / max_samples);

    double acc = 0.0;
    size_t count = 0;

    for (size_t i = 2 * step; i < n; i += step) {
        const double d =
            g_sorted[i] - 2.0 * g_sorted[i - step] + g_sorted[i - 2 * step];
        acc += std::abs(d);
        count++;
    }

    const double local = (count > 0 ? acc / count : 0.0);
    const double smooth = 1.0 / (1.0 + local);
    const double comp = 0.20 * (1.0 - smooth);
    return std::min(2.0, std::max(0.1, 0.45 + comp));
}

inline std::vector<double> downsample_edges(const std::vector<double>& edges,
                                            int new_nb) {
    const int nb = static_cast<int>(edges.empty() ? 0 : (edges.size() - 1));
    if (nb <= 0 || new_nb <= 0) return {0.0, 1.0};
    if (new_nb >= nb) return edges;

    std::vector<double> out;
    out.reserve(static_cast<size_t>(new_nb) + 1);
    const int E = static_cast<int>(edges.size());
    for (int k = 0; k <= new_nb; ++k) {
        const double pos = static_cast<double>(k) * static_cast<double>(nb) /
                           static_cast<double>(new_nb);
        int i = static_cast<int>(std::floor(pos));
        double t = pos - i;
        if (i < 0) {
            i = 0;
            t = 0.0;
        }
        if (i >= E - 1) {
            i = E - 2;
            t = 1.0;
        }
        const double v = (1.0 - t) * edges[static_cast<size_t>(i)] +
                         t * edges[static_cast<size_t>(i + 1)];
        out.push_back(v);
    }
    return out;
}

} // namespace foretree
