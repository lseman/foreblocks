// tree/include/foretree/core/gradient_hist_system.hpp
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Adjust include if your path differs
#include "foretree/core/data_binner.hpp"  // DataBinner, EdgeSet, _strict_increasing
#include "foretree/core/histogram_primitives.hpp"
#include "foretree/core/binning_strategies.hpp"

namespace foretree {

// ============================ Main Histogram System ===========================

class GradientHistogramSystem {
   public:
    explicit GradientHistogramSystem(HistogramConfig cfg)
        : cfg_(std::move(cfg)), rng_(cfg_.rng_seed), P_(0), N_(0) {}

    // Fit bins per feature using the chosen strategy
    void fit_bins(const double* X, int N, int P, const double* g,
                  const double* h) {
        if (N <= 0 || P <= 0)
            throw std::invalid_argument("fit_bins: invalid N or P");
        if (!X || !g || !h) throw std::invalid_argument("fit_bins: null input");
        N_ = N;
        P_ = P;

        std::unique_ptr<IBinningStrategy> strat;
        if (cfg_.method == "quantile")
            strat = std::make_unique<QuantileBinner>();
        else if (cfg_.method == "hist")
            strat = std::make_unique<UniformBinner>();   // fixed: "hist" = uniform
        else if (cfg_.method == "two_stage")
            strat = std::make_unique<TwoStageBinner>();
        else if (cfg_.method == "adaptive")
            strat = std::make_unique<AdaptiveBinner>();
        else if (cfg_.method == "kmeans")
            strat = std::make_unique<KMeansBinner>();
        else
            strat = std::make_unique<GradientAwareBinner>();

        feature_bins_.assign(P_, FeatureBins{});

        auto process_feature = [&](int j) {
            std::vector<double> col(N_), gj(N_), hj(N_);
            for (int i = 0; i < N_; ++i) {
                const size_t off =
                    static_cast<size_t>(i) * static_cast<size_t>(P_) +
                    static_cast<size_t>(j);
                col[i] = X[off];
                gj[i] = g[i];
                hj[i] = h[i];
            }
            FeatureBins fb = strat->create_bins(col, gj, hj, cfg_);

            int max_bins_for_feature;
            if (cfg_.method == "adaptive" && cfg_.adaptive_binning) {
                max_bins_for_feature = fb.stats.suggested_bins;
            } else {
                max_bins_for_feature = cfg_.max_bins;
            }

            const int nb = fb.n_bins();
            if (nb > max_bins_for_feature) {
                fb.edges = downsample_edges(fb.edges, max_bins_for_feature);
                _check_uniform(fb);
            }
            return fb;
        };

        if (!cfg_.use_parallel || P_ == 1) {
            for (int j = 0; j < P_; ++j) feature_bins_[j] = process_feature(j);
        } else {
            const int workers = std::max(1, std::min(cfg_.max_workers, P_));
            (void)workers;
            std::vector<std::future<FeatureBins>> futs;
            futs.reserve(P_);
            for (int j = 0; j < P_; ++j) {
                futs.emplace_back(std::async(
                    std::launch::async, [&, j] { return process_feature(j); }));
            }
            for (int j = 0; j < P_; ++j) feature_bins_[j] = futs[j].get();
        }

        // Setup variable bin layout and DataBinner
        std::vector<int> bins_per_feat(P_);
        for (int j = 0; j < P_; ++j) {
            const int finite = feature_bins_[j].n_bins();
            const bool use_miss = cfg_.use_missing_bin;
            bins_per_feat[j] = finite + (use_miss ? 1 : 0);  // include missing slot
        }
        layout_.initialize(bins_per_feat);

        std::vector<std::vector<double>> edges_per_feat(P_);
        for (int j = 0; j < P_; ++j) edges_per_feat[j] = feature_bins_[j].edges;

        EdgeSet es;
        es.edges_per_feat = std::move(edges_per_feat);
        binner_ = std::make_unique<DataBinner>(P_);
        binner_->register_edges("hist", std::move(es));

        codes_.reset();
        miss_id_ = binner_->missing_bin_id("hist");
    }

    // Prebin whole matrix X; caches codes internally
    std::pair<std::shared_ptr<std::vector<uint16_t>>, int> prebin_dataset(
        const double* X, int N, int P) {
        if (!binner_)
            throw std::runtime_error(
                "fit_bins must be called before prebin_dataset");
        if (N != N_ || P != P_)
            throw std::invalid_argument(
                "prebin_dataset: shape mismatch vs fit_bins");
        auto pr = binner_->prebin(X, N, P, "hist", -1);
        codes_ = pr.first;
        miss_id_ = pr.second;
        return pr;
    }

    // Prebin ANY matrix X using fitted edges, without touching the internal
    // cache
    std::pair<std::shared_ptr<std::vector<uint16_t>>, int> prebin_matrix(
        const double* X, int N, int P) const {
        if (!binner_)
            throw std::runtime_error(
                "fit_bins must be called before prebin_matrix");
        if (P != P_)
            throw std::invalid_argument(
                "prebin_matrix: P mismatch vs fit_bins");
        return binner_->prebin(X, N, P, "hist", -1);
    }

    // High-performance histogram builders using optimized accumulator
    template <class Gdouble = double, class Hdouble = double>
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    build_histograms_with_counts(const Gdouble* g, const Hdouble* h,
                                 const int* sample_indices = nullptr,
                                 int n_sub = 0) const {
        if (!codes_) throw std::runtime_error("build_histograms_with_counts: call prebin_dataset first");
        HistogramAccumulator acc(&layout_);
        acc.accumulate_samples<true>(codes_->data(), g, h, sample_indices,
                                     (sample_indices ? n_sub : N_), P_);
        return acc.take_results();
    }

    template <class Gdouble = double, class Hdouble = double>
    std::pair<std::vector<double>, std::vector<double>> build_histograms(
        const Gdouble* g, const Hdouble* h, const int* sample_indices = nullptr,
        int n_sub = 0) const {
        if (!codes_) throw std::runtime_error("build_histograms: call prebin_dataset first");
        HistogramAccumulator acc(&layout_);
        acc.accumulate_samples<false>(codes_->data(), g, h, sample_indices,
                                      (sample_indices ? n_sub : N_), P_);
        return acc.take_gh_results();
    }

    // Legacy methods for compatibility
    template <class Gdouble = double, class Hdouble = double>
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    build_histograms_fast_with_counts(const Gdouble* g, const Hdouble* h,
                                      const int* sample_indices = nullptr,
                                      int n_sub = 0) const {
        return build_histograms_with_counts(g, h, sample_indices, n_sub);
    }

    template <class Gdouble = double, class Hdouble = double>
    std::pair<std::vector<double>, std::vector<double>> build_histograms_fast(
        const Gdouble* g, const Hdouble* h, const int* sample_indices = nullptr,
        int n_sub = 0) const {
        return build_histograms(g, h, sample_indices, n_sub);
    }

    // Helper to extract histogram for a specific feature from the packed format
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    extract_feature_histogram(const std::vector<double>& Hg,
                              const std::vector<double>& Hh,
                              const std::vector<int>& C, int feature) const {
        if (!binner_ || feature < 0 || feature >= P_) {
            throw std::invalid_argument("Invalid feature index");
        }

        const size_t feat_offset = layout_.feature_offset(feature);
        const int feat_bins = layout_.bins_for_feature(feature);

        std::vector<double> feat_Hg(feat_bins);
        std::vector<double> feat_Hh(feat_bins);
        std::vector<int> feat_C(feat_bins);

        for (int b = 0; b < feat_bins; ++b) {
            const size_t idx = feat_offset + b;
            feat_Hg[b] = (idx < Hg.size()) ? Hg[idx] : 0.0;
            feat_Hh[b] = (idx < Hh.size()) ? Hh[idx] : 0.0;
            feat_C[b] = (idx < C.size()) ? C[idx] : 0;
        }

        return {std::move(feat_Hg), std::move(feat_Hh), std::move(feat_C)};
    }

    // Get feature histogram offsets for manual indexing
    std::vector<size_t> get_feature_offsets() const {
        std::vector<size_t> offsets(P_ + 1);
        for (int j = 0; j < P_; ++j) {
            offsets[j] = layout_.feature_offset(j);
        }
        offsets[P_] = layout_.total_size();  // Final offset
        return offsets;
    }

    // Accessors
    int P() const { return P_; }
    int N() const { return N_; }
    int missing_bin_id() const { return miss_id_; }

    // Per-feature accessors
    int finite_bins(int feature) const {
        return binner_ ? binner_->finite_bins("hist", feature) : cfg_.max_bins;
    }
    int total_bins(int feature) const {
        return binner_ ? binner_->total_bins("hist", feature)
                       : cfg_.total_bins();
    }
    int missing_bin_id(int feature) const {
        return binner_ ? binner_->missing_bin_id("hist", feature)
                       : cfg_.missing_bin_id();
    }

    // Legacy accessors (return max across features for compatibility)
    int finite_bins() const {
        return binner_ ? binner_->finite_bins("hist") : cfg_.max_bins;
    }
    int total_bins() const {
        return binner_ ? binner_->total_bins("hist") : cfg_.total_bins();
    }

    // Get all bin counts at once
    std::vector<int> all_finite_bins() const {
        if (!binner_) return std::vector<int>(P_, cfg_.max_bins);
        return binner_->finite_bins_per_feat("hist");
    }
    std::vector<int> all_total_bins() const {
        std::vector<int> result(P_);
        for (int j = 0; j < P_; ++j) {
            result[j] = total_bins(j);
        }
        return result;
    }

    // Feature analysis results
    const FeatureStats& feature_stats(int j) const {
        return feature_bins_.at(j).stats;
    }

    // Get summary of bin allocation decisions
    std::vector<std::pair<int, std::string>> get_bin_allocation_summary()
        const {
        std::vector<std::pair<int, std::string>> summary(P_);
        for (int j = 0; j < P_; ++j) {
            summary[j] = {finite_bins(j),
                          feature_bins_[j].stats.allocation_reason};
        }
        return summary;
    }

    const FeatureBins& feature_bins(int j) const { return feature_bins_.at(j); }
    const DataBinner* binner() const { return binner_.get(); }
    std::shared_ptr<std::vector<uint16_t>> codes_view() const { return codes_; }

   private:
    HistogramConfig cfg_;
    std::mt19937_64 rng_;

    int P_ = 0, N_ = 0;
    int miss_id_ = -1;

    VariableBinLayout layout_;
    std::vector<FeatureBins> feature_bins_;
    std::unique_ptr<DataBinner> binner_;
    std::shared_ptr<std::vector<uint16_t>> codes_;
};

}  // namespace foretree
