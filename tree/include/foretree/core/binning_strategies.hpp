#pragma once

#include "foretree/core/histogram_primitives.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace foretree {

struct IBinningStrategy {
    virtual ~IBinningStrategy() = default;
    virtual FeatureBins create_bins(const std::vector<double>& values,
                                    const std::vector<double>& gradients,
                                    const std::vector<double>& hessians,
                                    const HistogramConfig& cfg) = 0;
};

inline int categorical_unique_limit(const HistogramConfig& cfg) {
    return std::max(2, std::min(cfg.max_bins / 4, 32));
}

inline bool categorical_from_unique_count(int unique_count, const HistogramConfig& cfg) {
    return unique_count > 0 && unique_count <= categorical_unique_limit(cfg);
}

inline std::vector<double> resample_edges_linear(const std::vector<double>& edges, int new_nb) {
    const int nb = static_cast<int>(edges.empty() ? 0 : (edges.size() - 1));
    if (nb <= 0 || new_nb <= 0) return {0.0, 1.0};
    if (new_nb == nb) return edges;

    std::vector<double> out(static_cast<size_t>(new_nb) + 1);
    for (int k = 0; k <= new_nb; ++k) {
        const double pos = static_cast<double>(k) * static_cast<double>(nb) / static_cast<double>(new_nb);
        int i = static_cast<int>(std::floor(pos));
        double t = pos - static_cast<double>(i);
        if (i < 0) {
            i = 0;
            t = 0.0;
        }
        if (i >= nb) {
            i = nb - 1;
            t = 1.0;
        }
        out[static_cast<size_t>(k)] =
            (1.0 - t) * edges[static_cast<size_t>(i)] + t * edges[static_cast<size_t>(i + 1)];
    }
    for (size_t i = 1; i < out.size(); ++i) {
        if (!(out[i] > out[i - 1])) {
            out[i] = std::nextafter(out[i - 1], std::numeric_limits<double>::infinity());
        }
    }
    return out;
}

inline void finalize_feature_bins(FeatureBins& fb, const HistogramConfig& cfg, int max_bins_for_feature,
                                  bool enforce_min_bins = true) {
    if (fb.edges.size() < 2) fb.edges = {0.0, 1.0};

    const int max_bins = std::max(1, max_bins_for_feature);
    const int cfg_min_bins = std::clamp(cfg.min_bins, 1, max_bins);
    const int suggested_min = enforce_min_bins ? cfg_min_bins : 1;

    int nb = fb.n_bins();
    if (nb > max_bins) {
        fb.edges = downsample_edges(fb.edges, max_bins);
        nb = fb.n_bins();
    }
    if (enforce_min_bins && nb < cfg_min_bins) {
        fb.edges = resample_edges_linear(fb.edges, cfg_min_bins);
        nb = fb.n_bins();
    }

    if (fb.stats.suggested_bins <= 0) {
        fb.stats.suggested_bins = std::clamp(nb, suggested_min, max_bins);
    } else {
        fb.stats.suggested_bins = std::clamp(fb.stats.suggested_bins, suggested_min, max_bins);
    }
    _check_uniform(fb);
}

inline bool apply_common_categorical_precheck(FeatureBins& fb, const std::vector<double>& finite_values,
                                              const HistogramConfig& cfg, int max_bins_for_feature) {
    if (finite_values.empty()) return false;
    std::vector<double> unique_vals = finite_values;
    std::sort(unique_vals.begin(), unique_vals.end());
    unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());

    const int unique_count = static_cast<int>(unique_vals.size());
    if (!categorical_from_unique_count(unique_count, cfg)) return false;

    fb.strategy = "categorical";
    fb.stats.is_categorical = true;
    fb.stats.unique_count = unique_count;
    fb.stats.allocation_reason = "categorical_precheck";
    fb.stats.suggested_bins = std::min(unique_count, std::max(1, max_bins_for_feature));
    fb.edges = _midpoint_edges_of_unique(unique_vals);
    finalize_feature_bins(fb, cfg, max_bins_for_feature, false);
    return true;
}

struct UniformBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& /*gradients*/,
                            const std::vector<double>& /*hessians*/,
                            const HistogramConfig& cfg) override {
        const int max_bins_for_feature = std::max(1, cfg.max_bins);
        FeatureBins fb;
        fb.strategy = "uniform";

        std::vector<double> finite_values;
        finite_values.reserve(values.size());
        double vmin = std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        for (double v : values) {
            if (!std::isfinite(v)) continue;
            finite_values.push_back(v);
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }

        if (finite_values.empty()) {
            fb.edges = {0.0, 1.0};
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "empty_feature";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        if (apply_common_categorical_precheck(fb, finite_values, cfg, max_bins_for_feature)) {
            return fb;
        }

        if (!std::isfinite(vmin) || !std::isfinite(vmax) || !(vmax > vmin)) {
            fb.edges = {vmin - 1e-12, vmin + 1e-12};
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "constant_feature";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        const int nb = max_bins_for_feature;
        fb.edges.resize(nb + 1);
        for (int i = 0; i <= nb; ++i) {
            fb.edges[i] = vmin + (vmax - vmin) * (double(i) / nb);
        }
        fb.stats.suggested_bins = nb;
        fb.stats.allocation_reason = "uniform_range";
        finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
        return fb;
    }
};

struct QuantileBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& /*gradients*/,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override {
        const int max_bins_for_feature = std::max(1, cfg.max_bins);
        std::vector<double> v, w;
        v.reserve(values.size());
        w.reserve(values.size());

        for (size_t i = 0; i < values.size(); ++i) {
            const double vi = values[i];
            const double wi = (i < hessians.size() ? hessians[i] : 1.0);
            if (std::isfinite(vi) && std::isfinite(wi)) {
                v.push_back(vi);
                w.push_back(std::max(cfg.eps, wi));
            }
        }
        FeatureBins fb;
        fb.strategy = "quantile";
        if (v.empty()) {
            fb.edges = {0.0, 1.0};
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "empty_feature";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        if (apply_common_categorical_precheck(fb, v, cfg, max_bins_for_feature)) {
            return fb;
        }

        std::vector<double> u = v;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        if (static_cast<int>(u.size()) <= max_bins_for_feature) {
            fb.edges = _midpoint_edges_of_unique(u);
            fb.stats.unique_count = static_cast<int>(u.size());
            fb.stats.suggested_bins = fb.n_bins();
            fb.stats.allocation_reason = "all_unique_midpoints";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        fb.edges = weighted_quantile_edges(v, w, max_bins_for_feature);
        fb.stats.suggested_bins = max_bins_for_feature;
        fb.stats.unique_count = static_cast<int>(u.size());
        fb.stats.allocation_reason = "weighted_quantile";
        finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
        return fb;
    }
};

struct KMeansBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& /*gradients*/,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override
    {
        const int max_bins_for_feature = std::max(1, cfg.max_bins);
        FeatureBins fb;
        fb.strategy = "kmeans";

        std::vector<std::pair<double,double>> vw;
        std::vector<double> finite_values;
        vw.reserve(values.size());
        finite_values.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const double v = values[i];
            const double w = (i < hessians.size() ? hessians[i] : 1.0);
            if (std::isfinite(v) && std::isfinite(w) && w > 0.0) {
                vw.emplace_back(v, std::max(cfg.eps, w));
                finite_values.push_back(v);
            }
        }
        if (vw.empty()) {
            fb.edges = {0.0, 1.0};
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "empty_feature";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        if (apply_common_categorical_precheck(fb, finite_values, cfg, max_bins_for_feature)) {
            return fb;
        }

        std::sort(vw.begin(), vw.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });

        {
            std::vector<double> uniq;
            uniq.reserve(vw.size());
            uniq.push_back(vw[0].first);
            for (size_t i = 1; i < vw.size(); ++i) {
                if (vw[i].first != uniq.back()) uniq.push_back(vw[i].first);
            }
            fb.stats.unique_count = static_cast<int>(uniq.size());
            if ((int)uniq.size() <= std::min(max_bins_for_feature, 32)) {
                fb.edges = _midpoint_edges_of_unique(uniq);
                fb.stats.suggested_bins = fb.n_bins();
                fb.stats.allocation_reason = "small_unique_midpoints";
                finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
                return fb;
            }
        }

        const int K_req = max_bins_for_feature;
        int K = K_req;
        {
            int uniq_count = 1;
            for (size_t i = 1; i < vw.size(); ++i)
                if (vw[i].first != vw[i-1].first) ++uniq_count;
            K = std::min(K, uniq_count);
        }
        if (K <= 1) {
            const double vmin = vw.front().first;
            const double vmax = vw.back().first;
            if (!(vmax > vmin)) {
                fb.edges = {vmin - 1e-12, vmin + 1e-12};
            } else {
                fb.edges = {vmin, vmax};
            }
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "single_cluster";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        std::vector<double> centers(K, 0.0);
        {
            const double total_w = std::accumulate(vw.begin(), vw.end(), 0.0,
                                [](double a, const auto& p){ return a + p.second; });
            std::vector<double> cumw(vw.size());
            cumw[0] = vw[0].second;
            for (size_t i = 1; i < vw.size(); ++i) cumw[i] = cumw[i-1] + vw[i].second;
            for (int k = 0; k < K; ++k) {
                const double target = (double(k) + 0.5) * (total_w / K);
                auto it = std::lower_bound(cumw.begin(), cumw.end(), target);
                size_t idx = (it == cumw.end() ? vw.size()-1 : size_t(it - cumw.begin()));
                centers[k] = vw[idx].first;
            }
            for (int k = 1; k < K; ++k) {
                if (!(centers[k] > centers[k-1])) {
                    centers[k] = std::nextafter(centers[k-1], std::numeric_limits<double>::infinity());
                }
            }
        }

        const int max_iters = 100;
        const double tol = 1e-7;
        std::vector<int> assign(vw.size(), 0);

        auto assign_step = [&](const std::vector<double>& c) {
            bool changed = false;
            for (size_t i = 0; i < vw.size(); ++i) {
                int best = 0;
                double bestd = std::abs(vw[i].first - c[0]);
                for (int k = 1; k < K; ++k) {
                    const double d = std::abs(vw[i].first - c[k]);
                    if (d < bestd) { bestd = d; best = k; }
                }
                if (assign[i] != best) { assign[i] = best; changed = true; }
            }
            return changed;
        };

        auto update_step = [&](std::vector<double>& c) {
            std::vector<double> num(K, 0.0), den(K, 0.0);
            for (size_t i = 0; i < vw.size(); ++i) {
                const int k = assign[i];
                num[k] += vw[i].first * vw[i].second;
                den[k] += vw[i].second;
            }

            for (int k = 0; k < K; ++k) {
                if (den[k] <= 0.0) {
                    const double q = (double(k) + 0.5) / (double)K;
                    const double total_w = std::accumulate(vw.begin(), vw.end(), 0.0,
                        [](double a,const auto& p){ return a + p.second; });
                    const double target = q * total_w;
                    double accw = 0.0;
                    size_t idx = 0;
                    for (; idx < vw.size(); ++idx) {
                        accw += vw[idx].second;
                        if (accw >= target) break;
                    }
                    c[k] = vw[std::min(idx, vw.size()-1)].first;
                } else {
                    c[k] = num[k] / den[k];
                }
            }

            std::sort(c.begin(), c.end());
            for (int k = 1; k < K; ++k) {
                if (!(c[k] > c[k-1])) {
                    c[k] = std::nextafter(c[k-1], std::numeric_limits<double>::infinity());
                }
            }
        };

        assign_step(centers);
        for (int it = 0; it < max_iters; ++it) {
            std::vector<double> oldc = centers;
            update_step(centers);
            const bool changed = assign_step(centers);
            double delta = 0.0;
            for (int k = 0; k < K; ++k) delta += std::abs(centers[k] - oldc[k]);
            if (!changed || delta < tol) break;
        }

        const double vmin = vw.front().first;
        const double vmax = vw.back().first;
        std::sort(centers.begin(), centers.end());
        centers.erase(std::unique(centers.begin(), centers.end(),
                         [](double a,double b){ return !(b>a); }), centers.end());
        int Kf = std::max(1, (int)centers.size());

        if (Kf == 1) {
            const double c0 = centers[0];
            fb.edges = {std::min(vmin, c0) - 1e-12, std::max(vmax, c0) + 1e-12};
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "collapsed_centers";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        fb.edges.resize((size_t)Kf + 1);
        fb.edges.front() = vmin - 1e-12;
        fb.edges.back()  = vmax + 1e-12;
        for (int i = 1; i < Kf; ++i) {
            fb.edges[(size_t)i] = 0.5 * (centers[i-1] + centers[i]);
        }

        for (size_t i = 1; i < fb.edges.size(); ++i) {
            if (!(fb.edges[i] > fb.edges[i-1])) {
                fb.edges[i] = std::nextafter(fb.edges[i-1], std::numeric_limits<double>::infinity());
            }
        }

        fb.stats.suggested_bins = Kf;
        fb.stats.allocation_reason = "kmeans_centroids";
        finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
        return fb;
    }
};

struct GradientAwareBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& gradients,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override {
        const int max_bins_for_feature = std::max(1, cfg.max_bins);
        std::vector<double> v, g, h;
        v.reserve(values.size());
        g.reserve(values.size());
        h.reserve(values.size());

        for (size_t i = 0; i < values.size(); ++i) {
            const double vi = values[i];
            const double gi = (i < gradients.size() ? gradients[i] : 0.0);
            const double hi = (i < hessians.size() ? hessians[i] : 1.0);
            if (std::isfinite(vi) && std::isfinite(gi) && std::isfinite(hi)) {
                v.push_back(vi);
                g.push_back(gi);
                h.push_back(std::max(cfg.eps, hi));
            }
        }
        FeatureBins fb;
        fb.strategy = "grad_aware";
        if (v.empty()) {
            fb.edges = {0.0, 1.0};
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "empty_feature";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        if (apply_common_categorical_precheck(fb, v, cfg, max_bins_for_feature)) {
            return fb;
        }

        std::vector<double> u = v;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        fb.stats.unique_count = static_cast<int>(u.size());
        if (static_cast<int>(u.size()) <= std::min(max_bins_for_feature, 32)) {
            fb.edges = _midpoint_edges_of_unique(u);
            fb.stats.suggested_bins = fb.n_bins();
            fb.stats.allocation_reason = "small_unique_midpoints";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        std::vector<size_t> ord(v.size());
        std::iota(ord.begin(), ord.end(), size_t{0});
        std::sort(ord.begin(), ord.end(),
                  [&](size_t a, size_t b) { return v[a] < v[b]; });

        std::vector<double> v_s, g_s, h_s;
        v_s.reserve(v.size());
        g_s.reserve(v.size());
        h_s.reserve(v.size());
        for (size_t k : ord) {
            v_s.push_back(v[k]);
            g_s.push_back(g[k]);
            h_s.push_back(h[k]);
        }

        const double comp = gradient_complexity(v_s, g_s);
        const int min_bins_for_feature = std::clamp(cfg.min_bins, 1, max_bins_for_feature);
        const int nb = std::clamp(
            std::max(16, static_cast<int>(std::lround(static_cast<double>(cfg.coarse_bins) * comp))),
            min_bins_for_feature, max_bins_for_feature);

        std::vector<double> w(v_s.size());
        for (size_t i = 0; i < v_s.size(); ++i)
            w[i] = std::max(cfg.eps, h_s[i]);
        fb.edges = weighted_quantile_edges(v_s, w, nb);
        fb.stats.suggested_bins = nb;
        fb.stats.allocation_reason = "gradient_aware_weighted_quantile";
        finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
        return fb;
    }
};

struct TwoStageBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& gradients,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override {
        const int max_bins_for_feature = std::max(1, cfg.max_bins);
        const int min_bins_for_feature = std::clamp(cfg.min_bins, 1, max_bins_for_feature);

        std::vector<double> v, g, h;
        v.reserve(values.size());
        g.reserve(values.size());
        h.reserve(values.size());

        for (size_t i = 0; i < values.size(); ++i) {
            const double vi = values[i];
            const double gi = (i < gradients.size() ? gradients[i] : 0.0);
            const double hi = (i < hessians.size() ? hessians[i] : 1.0);
            if (std::isfinite(vi) && std::isfinite(gi) && std::isfinite(hi)) {
                v.push_back(vi);
                g.push_back(gi);
                h.push_back(std::max(cfg.eps, hi));
            }
        }

        FeatureBins fb;
        fb.strategy = "two_stage";

        if (v.empty()) {
            fb.edges = {0.0, 1.0};
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "empty_feature";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        if (apply_common_categorical_precheck(fb, v, cfg, max_bins_for_feature)) {
            return fb;
        }

        std::vector<double> uniq = v;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        fb.stats.unique_count = static_cast<int>(uniq.size());
        if (static_cast<int>(uniq.size()) <= std::min(max_bins_for_feature, 32)) {
            fb.edges = _midpoint_edges_of_unique(uniq);
            fb.stats.suggested_bins = fb.n_bins();
            fb.stats.allocation_reason = "small_unique_midpoints";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        std::vector<size_t> ord(v.size());
        std::iota(ord.begin(), ord.end(), size_t{0});
        std::sort(ord.begin(), ord.end(),
                  [&](size_t a, size_t b) { return v[a] < v[b]; });

        std::vector<double> v_s, g_s, h_s;
        v_s.reserve(v.size());
        g_s.reserve(v.size());
        h_s.reserve(v.size());
        for (size_t k : ord) {
            v_s.push_back(v[k]);
            g_s.push_back(g[k]);
            h_s.push_back(h[k]);
        }

        const double comp = gradient_complexity(v_s, g_s);
        const int coarse_default = std::max(8, std::min(cfg.coarse_bins, 64));
        const int target_default =
            std::max(min_bins_for_feature,
                     std::min(max_bins_for_feature,
                              static_cast<int>(std::lround(static_cast<double>(cfg.target_bins) * comp))));
        const int target_bins = std::clamp(target_default, min_bins_for_feature, max_bins_for_feature);
        const int coarse_bins = std::clamp(coarse_default, 2, target_bins);

        std::vector<double> coarse_w(v.size());
        for (size_t i = 0; i < v.size(); ++i) coarse_w[i] = std::max(cfg.eps, h[i]);
        std::vector<double> coarse_edges = weighted_quantile_edges(v, coarse_w, coarse_bins);

        const int coarse_nb = std::max(1, static_cast<int>(coarse_edges.size()) - 1);
        if (coarse_nb <= 1 || target_bins <= coarse_nb) {
            fb.edges = weighted_quantile_edges(v, coarse_w, target_bins);
            fb.stats.suggested_bins = target_bins;
            fb.stats.allocation_reason = "two_stage_fallback_single_pass";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        std::vector<std::vector<double>> bin_values(static_cast<size_t>(coarse_nb));
        std::vector<std::vector<double>> bin_weights(static_cast<size_t>(coarse_nb));
        std::vector<double> bin_scores(static_cast<size_t>(coarse_nb), 0.0);
        std::vector<double> bin_hsum(static_cast<size_t>(coarse_nb), 0.0);

        for (size_t i = 0; i < v.size(); ++i) {
            const auto it = std::upper_bound(coarse_edges.begin() + 1, coarse_edges.end() - 1, v[i]);
            int b = static_cast<int>(it - coarse_edges.begin());
            b = std::clamp(b, 0, coarse_nb - 1);
            const double hi = std::max(cfg.eps, h[i]);
            const double wi = hi * (1.0 + std::abs(g[i]));
            bin_values[static_cast<size_t>(b)].push_back(v[i]);
            bin_weights[static_cast<size_t>(b)].push_back(std::max(cfg.eps, wi));
            bin_scores[static_cast<size_t>(b)] += std::abs(g[i]) * hi;
            bin_hsum[static_cast<size_t>(b)] += hi;
        }

        double total_score = 0.0;
        for (int b = 0; b < coarse_nb; ++b) {
            if (bin_hsum[static_cast<size_t>(b)] > 0.0) {
                const double normalized =
                    bin_scores[static_cast<size_t>(b)] / (bin_hsum[static_cast<size_t>(b)] + cfg.eps);
                bin_scores[static_cast<size_t>(b)] =
                    normalized * std::sqrt(bin_hsum[static_cast<size_t>(b)] + cfg.eps);
            } else {
                bin_scores[static_cast<size_t>(b)] = 0.0;
            }
            total_score += bin_scores[static_cast<size_t>(b)];
        }

        const int extra = std::max(0, target_bins - coarse_nb);
        std::vector<int> add(static_cast<size_t>(coarse_nb), 0);
        if (extra > 0) {
            if (total_score <= 0.0) {
                for (int t = 0; t < extra; ++t) add[static_cast<size_t>(t % coarse_nb)]++;
            } else {
                std::vector<double> rem(static_cast<size_t>(coarse_nb), 0.0);
                int assigned = 0;
                for (int b = 0; b < coarse_nb; ++b) {
                    const double exact = static_cast<double>(extra) *
                                         (bin_scores[static_cast<size_t>(b)] / total_score);
                    const int whole = static_cast<int>(std::floor(exact));
                    add[static_cast<size_t>(b)] = whole;
                    rem[static_cast<size_t>(b)] = exact - static_cast<double>(whole);
                    assigned += whole;
                }
                int left = extra - assigned;
                while (left-- > 0) {
                    int best_b = 0;
                    double best_r = rem[0];
                    for (int b = 1; b < coarse_nb; ++b) {
                        if (rem[static_cast<size_t>(b)] > best_r) {
                            best_r = rem[static_cast<size_t>(b)];
                            best_b = b;
                        }
                    }
                    add[static_cast<size_t>(best_b)]++;
                    rem[static_cast<size_t>(best_b)] = -1.0;
                }
            }
        }

        fb.edges.clear();
        fb.edges.reserve(static_cast<size_t>(target_bins) + 1);
        fb.edges.push_back(coarse_edges.front());
        for (int b = 0; b < coarse_nb; ++b) {
            const int add_b = add[static_cast<size_t>(b)];
            if (add_b > 0 && !bin_values[static_cast<size_t>(b)].empty()) {
                std::vector<double> local =
                    weighted_quantile_edges(bin_values[static_cast<size_t>(b)],
                                            bin_weights[static_cast<size_t>(b)],
                                            add_b + 1);
                for (int t = 1; t <= add_b; ++t) {
                    double edge = local[static_cast<size_t>(t)];
                    if (!(edge > fb.edges.back())) {
                        edge = std::nextafter(fb.edges.back(), std::numeric_limits<double>::infinity());
                    }
                    const double ub = coarse_edges[static_cast<size_t>(b + 1)];
                    if (edge >= ub) break;
                    fb.edges.push_back(edge);
                }
            }

            double upper = coarse_edges[static_cast<size_t>(b + 1)];
            if (!(upper > fb.edges.back())) {
                upper = std::nextafter(fb.edges.back(), std::numeric_limits<double>::infinity());
            }
            fb.edges.push_back(upper);
        }

        fb.stats.suggested_bins = target_bins;
        fb.stats.allocation_reason = "two_stage_coarse_refine";
        finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
        return fb;
    }
};

struct AdaptiveBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& gradients,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override {
        const int max_bins_for_feature = std::max(1, cfg.max_bins);
        std::vector<double> v, g, h;
        v.reserve(values.size());
        g.reserve(values.size());
        h.reserve(values.size());

        for (size_t i = 0; i < values.size(); ++i) {
            const double vi = values[i];
            const double gi = (i < gradients.size() ? gradients[i] : 0.0);
            const double hi = (i < hessians.size() ? hessians[i] : 1.0);
            if (std::isfinite(vi) && std::isfinite(gi) && std::isfinite(hi)) {
                v.push_back(vi);
                g.push_back(gi);
                h.push_back(std::max(cfg.eps, hi));
            }
        }

        FeatureBins fb;
        fb.strategy = "adaptive";

        if (v.empty()) {
            fb.edges = {0.0, 1.0};
            fb.stats.suggested_bins = 1;
            fb.stats.allocation_reason = "empty_feature";
            finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
            return fb;
        }

        if (apply_common_categorical_precheck(fb, v, cfg, max_bins_for_feature)) {
            return fb;
        }

        fb.stats = analyze_feature_importance(v, g, h, cfg);
        const int min_bins_for_feature = std::clamp(cfg.min_bins, 1, max_bins_for_feature);
        const int target_bins = std::clamp(fb.stats.suggested_bins, min_bins_for_feature, max_bins_for_feature);

        if (fb.stats.is_categorical) {
            std::vector<double> u = v;
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end()), u.end());
            fb.strategy = "categorical";
            fb.stats.unique_count = static_cast<int>(u.size());
            fb.stats.suggested_bins = std::min(static_cast<int>(u.size()), max_bins_for_feature);
            fb.edges = _midpoint_edges_of_unique(u);
            finalize_feature_bins(fb, cfg, max_bins_for_feature, false);
            return fb;
        }

        std::vector<double> w(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            const double base_weight = h[i];
            const double grad_weight =
                (fb.stats.importance_score > cfg.importance_threshold)
                    ? (1.0 + std::abs(g[i]))
                    : 1.0;
            w[i] = base_weight * grad_weight;
        }

        fb.edges = weighted_quantile_edges(v, w, target_bins);
        fb.stats.suggested_bins = target_bins;
        finalize_feature_bins(fb, cfg, max_bins_for_feature, true);
        return fb;
    }
};

} // namespace foretree
