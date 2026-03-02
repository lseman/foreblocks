#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(TREE_ENABLE_STDEXEC)
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#ifdef FORETREE_EXEC_USE_CUDA
#include <nvexec/stream_context.cuh>
#endif
#endif

#include "foretree/core/gradient_hist_system.hpp"
#include "foretree/tree/unified_tree.hpp"

namespace foretree {

// ============================================================================
// Execution Context
// ============================================================================
namespace execx {

#if defined(TREE_ENABLE_STDEXEC)
struct Context {
#ifdef FORETREE_EXEC_USE_CUDA
    nvexec::stream_context cuda_ctx;
    auto                   scheduler() { return cuda_ctx.get_scheduler(); }
#else
    exec::static_thread_pool pool;
    explicit Context(unsigned threads = std::thread::hardware_concurrency()) : pool(threads ? threads : 1) {}
    auto scheduler() { return pool.get_scheduler(); }
#endif
};

template <class F>
inline void bulk(Context &ctx, int n, F &&fn) {
    if (n <= 0) return;
    if (n == 1) {
        fn(0);
        return;
    }

    auto snd = stdexec::bulk(stdexec::schedule(ctx.scheduler()),
                             stdexec::par, // <-- new required argument
                             static_cast<std::size_t>(n),
                             [f = std::forward<F>(fn)](std::size_t i) mutable { f(static_cast<int>(i)); });

    stdexec::sync_wait(std::move(snd));
}
template <class F>
inline void bulk_iota(Context &ctx, int n, F &&fn) {
    bulk(ctx, n, std::forward<F>(fn));
}
#else
struct Context {
    explicit Context(unsigned threads = std::thread::hardware_concurrency()) : threads_(threads ? threads : 1U) {}
    unsigned threads_ = 1U;
};

template <class F>
inline void bulk(Context &, int n, F &&fn) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) fn(i);
}

template <class F>
inline void bulk_iota(Context &ctx, int n, F &&fn) {
    bulk(ctx, n, std::forward<F>(fn));
}
#endif

} // namespace execx

// ============================================================================
// ForeForest Configuration
// ============================================================================
struct ForeForestConfig {
    enum class Mode { Bagging, GBDT, FWBoost } mode                = Mode::Bagging;
    enum class Objective { SquaredError, BinaryLogloss, BinaryFocalLoss, HuberError } objective = Objective::SquaredError;

    int      n_estimators  = 100;
    double   learning_rate = 0.1;
    uint64_t rng_seed      = 123456789ULL;

    double focal_gamma = 2.0;    // For BinaryFocalLoss
    double huber_delta = 1.0;    // For HuberError

    HistogramConfig  hist_cfg{};
    TreeConfig       tree_cfg{};
    NeuralLeafConfig neural_cfg{};

    double colsample_bytree = 1.0;
    double colsample_bynode = 1.0;

    // Bagging
    double rf_row_subsample   = 1.0;
    bool   rf_bootstrap       = true;
    bool   rf_bootstrap_dedup = false;
    bool   rf_parallel        = true;

    // Exclusive Feature Bundling (EFB) for sparse/high-cardinality inputs
    bool   efb_enabled          = false;
    double efb_sparse_threshold = 0.2;
    int    efb_min_nonzero      = 10;
    double efb_max_conflict_rate = 0.0; // Allowed conflict rate for EFB

    // Boosting
    double gbdt_row_subsample       = 1.0;
    bool   gbdt_use_subsample       = false;
    bool   early_stopping_enabled   = false;
    int    early_stopping_rounds    = 20;
    double early_stopping_min_delta = 0.0;

    // Frank-Wolfe / LPBoost-inspired boosting
    bool   fw_use_subsample      = true;
    double fw_row_subsample      = 0.8;
    double fw_nu                 = 0.05;
    int    fw_line_search_points = 12;
    double fw_alpha_max          = 3.0;
    double fw_alpha_tol          = 1e-8;

    // DART
    bool   dart_enabled      = true;
    double dart_drop_rate    = 0.1;
    int    dart_max_drop     = 3;
    bool   dart_normalize    = true;
    bool   dart_one_drop_min = true;

    int threads = 0;
};

// ============================================================================
// ForeForest
// ============================================================================
class ForeForest {
public:
    explicit ForeForest(ForeForestConfig cfg)
        : cfg_(std::move(cfg)), rng_(cfg_.rng_seed), ghs_(std::make_unique<GradientHistogramSystem>(cfg_.hist_cfg)),
#ifndef FORETREE_EXEC_USE_CUDA
          ctx_((unsigned)(cfg_.threads > 0 ? cfg_.threads : (int)std::thread::hardware_concurrency()))
#else
          ctx_()
#endif
    {
        validate_config_();
    }

    // Set raw matrix for exact splits (optional, different from neural matrix)
    void set_raw_matrix(const double *Xraw, const uint8_t *Xmiss_or_null) {
        Xraw_exact_  = Xraw;
        Xmiss_exact_ = Xmiss_or_null;
    }

    // Set raw matrix for neural leaves (typically the same as training data)
    void set_raw_for_neural(const double *Xraw, const uint8_t *Xmiss) {
        Xraw_neural_  = Xraw;
        Xmiss_neural_ = Xmiss;
    }

    void fit_complete(const double *X, int N, int P, const double *y) {
        fit_complete(X, N, P, y, nullptr, 0, 0, nullptr);
    }

    void fit_complete(const double *X, int N, int P, const double *y, const double *X_valid, int N_valid, int P_valid,
                      const double *y_valid) {
        if (!X || !y) throw std::invalid_argument("ForeForest: null X or y");
        if (N <= 0 || P <= 0) throw std::invalid_argument("ForeForest: invalid N or P");
        const bool has_valid = (X_valid != nullptr || y_valid != nullptr || N_valid > 0 || P_valid > 0);
        if (has_valid) {
            if (!X_valid || !y_valid) throw std::invalid_argument("ForeForest: validation X/y must both be provided");
            if (N_valid <= 0 || P_valid <= 0) throw std::invalid_argument("ForeForest: invalid validation N/P");
            if (P_valid != P) throw std::invalid_argument("ForeForest: validation P mismatch");
        }

        N_          = N;
        original_P_ = P;

        reset_efb_state_();
        maybe_build_efb_plan_(X, N, P);

        int                 P_eff       = P;
        const double       *X_fit       = X;
        const double       *X_valid_fit = X_valid;
        int                 P_valid_eff = P_valid;
        std::vector<double> X_fit_efb;
        std::vector<double> X_valid_efb;

        if (efb_active_) {
            X_fit_efb = transform_with_efb_(X, N, P);
            P_eff     = efb_output_dim_();
            X_fit     = X_fit_efb.data();

            if (has_valid) {
                X_valid_efb = transform_with_efb_(X_valid, N_valid, P_valid);
                X_valid_fit = X_valid_efb.data();
                P_valid_eff = P_eff;
            }
        }
        P_ = P_eff;

        // Store raw matrix for neural leaves (training data)
        set_raw_for_neural(X, nullptr);

        // Base score (raw margin for binary logloss, mean for squared error).
        base_score_ = compute_base_score_(y, N);

        // Fit binning
        std::vector<double> init_g(N), init_h(N);
        compute_grad_hess_(y, N, base_score_, init_g, init_h);
        ghs_->fit_bins(X_fit, N, P_eff, init_g.data(), init_h.data());

        // Pre-bin dataset
        auto pr  = ghs_->prebin_dataset(X_fit, N, P_eff);
        codes_   = pr.first;
        miss_id_ = pr.second;
        std::shared_ptr<std::vector<uint16_t>> valid_codes;
        if (has_valid) {
            auto pv     = ghs_->prebin_matrix(X_valid_fit, N_valid, P_valid_eff);
            valid_codes = pv.first;
        }

        // Gradient buffers
        std::vector<double> g(N), h(N, 1.0);

        trees_.clear();
        tree_weights_.clear();
        train_metric_history_.clear();
        valid_metric_history_.clear();
        best_iteration_ = 0;
        best_score_     = std::numeric_limits<double>::infinity();
        early_stopped_  = false;
        trees_.reserve(cfg_.n_estimators);
        tree_weights_.reserve(cfg_.n_estimators);
        train_metric_history_.reserve(cfg_.n_estimators);
        valid_metric_history_.reserve(cfg_.n_estimators);

        if (cfg_.mode == ForeForestConfig::Mode::Bagging) {
            train_bagging_(g, h, y);
            if (has_valid) {
                auto valid_margin = predict_from_binned_(*valid_codes, N_valid, P_eff, X_valid_fit);
                valid_metric_history_.push_back(compute_metric_from_margin_(valid_margin, y_valid, N_valid));
                best_iteration_ = size();
                best_score_     = valid_metric_history_.back();
            } else {
                best_iteration_ = size();
            }
        } else if (cfg_.mode == ForeForestConfig::Mode::GBDT) {
            train_gbdt_(g, h, y, valid_codes.get(), N_valid, y_valid, X_valid_fit);
        } else {
            train_fwboost_(g, h, y, valid_codes.get(), N_valid, y_valid, X_valid_fit);
        }
    }

    void fit_complete(const std::vector<double> &X, int N, int P, const std::vector<double> &y) {
        if ((int)X.size() != N * P) throw std::invalid_argument("ForeForest: X.size != N*P");
        if ((int)y.size() != N) throw std::invalid_argument("ForeForest: y.size != N");
        fit_complete(X.data(), N, P, y.data());
    }

    void fit_complete(const std::vector<double> &X, int N, int P, const std::vector<double> &y,
                      const std::vector<double> &X_valid, int N_valid, int P_valid,
                      const std::vector<double> &y_valid) {
        if ((int)X.size() != N * P) throw std::invalid_argument("ForeForest: X.size != N*P");
        if ((int)y.size() != N) throw std::invalid_argument("ForeForest: y.size != N");
        if ((int)X_valid.size() != N_valid * P_valid)
            throw std::invalid_argument("ForeForest: X_valid.size != N_valid*P_valid");
        if ((int)y_valid.size() != N_valid) throw std::invalid_argument("ForeForest: y_valid.size != N_valid");
        fit_complete(X.data(), N, P, y.data(), X_valid.data(), N_valid, P_valid, y_valid.data());
    }

    std::vector<double> predict(const double *X, int N, int P) const {
        if (!ghs_->binner()) throw std::runtime_error("ForeForest: model not fitted");
        if (efb_active_) {
            if (P != original_P_)
                throw std::invalid_argument("ForeForest: P mismatch (expected original feature count)");
            std::vector<double> X_eff = transform_with_efb_(X, N, P);
            auto                pr    = ghs_->prebin_matrix(X_eff.data(), N, P_);
            auto                out   = predict_from_binned_(*pr.first, N, P_, X_eff.data());
            return finalize_prediction_(std::move(out), /*apply_link=*/true);
        }
        if (P != P_) throw std::invalid_argument("ForeForest: P mismatch");

        auto pr  = ghs_->prebin_matrix(X, N, P);
        auto out = predict_from_binned_(*pr.first, N, P, X); // Pass X for neural leaves
        return finalize_prediction_(std::move(out), /*apply_link=*/true);
    }

    std::vector<double> predict(const std::vector<double> &X, int N, int P) const { return predict(X.data(), N, P); }

    std::vector<double> predict_margin(const double *X, int N, int P) const {
        if (!ghs_->binner()) throw std::runtime_error("ForeForest: model not fitted");
        if (efb_active_) {
            if (P != original_P_)
                throw std::invalid_argument("ForeForest: P mismatch (expected original feature count)");
            std::vector<double> X_eff = transform_with_efb_(X, N, P);
            auto                pr    = ghs_->prebin_matrix(X_eff.data(), N, P_);
            auto                out   = predict_from_binned_(*pr.first, N, P_, X_eff.data());
            return finalize_prediction_(std::move(out), /*apply_link=*/false);
        }
        if (P != P_) throw std::invalid_argument("ForeForest: P mismatch");

        auto pr  = ghs_->prebin_matrix(X, N, P);
        auto out = predict_from_binned_(*pr.first, N, P, X);
        return finalize_prediction_(std::move(out), /*apply_link=*/false);
    }

    std::vector<double> predict_margin(const std::vector<double> &X, int N, int P) const {
        return predict_margin(X.data(), N, P);
    }

    std::vector<double> predict_contrib(const double *X, int N, int P) const {
        if (!ghs_->binner()) throw std::runtime_error("ForeForest: model not fitted");
        if (efb_active_) {
            throw std::runtime_error("ForeForest: TreeSHAP is not supported when EFB is enabled");
        }
        if (P != P_) throw std::invalid_argument("ForeForest: P mismatch");

        auto pr = ghs_->prebin_matrix(X, N, P);
        std::vector<double> out(static_cast<size_t>(N) * static_cast<size_t>(P + 1), 0.0);
        std::vector<double> contrib;

        for (size_t t = 0; t < trees_.size(); ++t) {
            const double wt = tree_weights_[t];
            contrib = trees_[t].predict_contrib(*pr.first, N, P, X);
            for (size_t i = 0; i < out.size(); ++i) out[i] += wt * contrib[i];
        }

        if (cfg_.mode == ForeForestConfig::Mode::Bagging && !trees_.empty()) {
            const double invT = 1.0 / static_cast<double>(trees_.size());
            for (double &v : out) v *= invT;
        } else if (cfg_.mode == ForeForestConfig::Mode::GBDT || cfg_.mode == ForeForestConfig::Mode::FWBoost) {
            for (int i = 0; i < N; ++i) {
                out[static_cast<size_t>(i) * static_cast<size_t>(P + 1) + static_cast<size_t>(P)] += base_score_;
            }
        }

        return out;
    }

    std::vector<double> predict_contrib(const std::vector<double> &X, int N, int P) const {
        return predict_contrib(X.data(), N, P);
    }

    std::vector<double> feature_importance_gain() const {
        std::vector<double> agg(P_, 0.0);
        for (size_t t = 0; t < trees_.size(); ++t) {
            const double wt = tree_weights_[t];
            const auto  &g  = trees_[t].feature_importance_gain();
            const int    m  = std::min<int>(P_, (int)g.size());
            for (int j = 0; j < m; ++j) agg[j] += wt * g[j];
        }
        return agg;
    }

    int                        size() const { return (int)trees_.size(); }
    const std::vector<double> &train_metric_history() const { return train_metric_history_; }
    const std::vector<double> &valid_metric_history() const { return valid_metric_history_; }
    int                        best_iteration() const { return best_iteration_; }
    double                     best_score() const { return best_score_; }
    bool                       early_stopped() const { return early_stopped_; }
    std::string                eval_metric_name() const {
        if (cfg_.objective == ForeForestConfig::Objective::BinaryLogloss) return "logloss";
        if (cfg_.objective == ForeForestConfig::Objective::BinaryFocalLoss) return "focal_loss";
        if (cfg_.objective == ForeForestConfig::Objective::HuberError) return "huber_loss";
        return "mse";
    }

    void clear() {
        trees_.clear();
        tree_weights_.clear();
        codes_.reset();
        train_metric_history_.clear();
        valid_metric_history_.clear();
        best_iteration_ = 0;
        best_score_     = std::numeric_limits<double>::infinity();
        early_stopped_  = false;
        N_ = P_ = original_P_ = 0;
        base_score_           = 0.0;
        Xraw_exact_           = nullptr;
        Xmiss_exact_          = nullptr;
        Xraw_neural_          = nullptr;
        Xmiss_neural_         = nullptr;
        reset_efb_state_();
        ghs_ = std::make_unique<GradientHistogramSystem>(cfg_.hist_cfg);
    }

private:
    void validate_config_() {
        cfg_.dart_drop_rate = std::clamp(cfg_.dart_drop_rate, 0.0, 1.0);
        cfg_.dart_max_drop  = std::max(0, cfg_.dart_max_drop);

        if (cfg_.dart_enabled && cfg_.dart_max_drop == 0 && cfg_.dart_one_drop_min) { cfg_.dart_max_drop = 1; }

        cfg_.colsample_bytree         = std::clamp(cfg_.colsample_bytree, 0.0, 1.0);
        cfg_.colsample_bynode         = std::clamp(cfg_.colsample_bynode, 0.0, 1.0);
        cfg_.rf_row_subsample         = std::clamp(cfg_.rf_row_subsample, 0.0, 1.0);
        cfg_.gbdt_row_subsample       = std::clamp(cfg_.gbdt_row_subsample, 0.0, 1.0);
        cfg_.fw_row_subsample         = std::clamp(cfg_.fw_row_subsample, 0.0, 1.0);
        cfg_.fw_nu                    = std::max(0.0, cfg_.fw_nu);
        cfg_.fw_line_search_points    = std::max(2, cfg_.fw_line_search_points);
        cfg_.fw_alpha_max             = std::max(0.0, cfg_.fw_alpha_max);
        cfg_.fw_alpha_tol             = std::max(0.0, cfg_.fw_alpha_tol);
        cfg_.early_stopping_rounds    = std::max(1, cfg_.early_stopping_rounds);
        cfg_.early_stopping_min_delta = std::max(0.0, cfg_.early_stopping_min_delta);
        cfg_.efb_sparse_threshold     = std::clamp(cfg_.efb_sparse_threshold, 0.0, 1.0);
        cfg_.efb_min_nonzero          = std::max(1, cfg_.efb_min_nonzero);
        cfg_.efb_max_conflict_rate    = std::clamp(cfg_.efb_max_conflict_rate, 0.0, 1.0);
        bool is_binary = (cfg_.objective == ForeForestConfig::Objective::BinaryLogloss || cfg_.objective == ForeForestConfig::Objective::BinaryFocalLoss);
        if (is_binary && cfg_.mode != ForeForestConfig::Mode::GBDT) {
            throw std::invalid_argument("ForeForest: Binary objectives require GBDT mode");
        }
        if (cfg_.tree_cfg.goss.enabled && cfg_.mode != ForeForestConfig::Mode::GBDT) {
            throw std::invalid_argument("ForeForest: GOSS currently supports GBDT mode only");
        }
        if (cfg_.mode == ForeForestConfig::Mode::FWBoost &&
            cfg_.objective != ForeForestConfig::Objective::SquaredError) {
            throw std::invalid_argument("ForeForest: FWBoost currently supports SquaredError objective only");
        }
        if (cfg_.efb_enabled && cfg_.tree_cfg.split_mode != TreeConfig::SplitMode::Histogram) {
            throw std::invalid_argument("ForeForest: EFB currently supports histogram split mode only");
        }
        if (cfg_.efb_enabled && cfg_.tree_cfg.neural_leaf.enabled) {
            throw std::invalid_argument("ForeForest: EFB is not compatible with neural leaf mode");
        }
    }

    static inline bool efb_nonzero_(double v) { return std::isfinite(v) && std::abs(v) > 1e-15; }

    int efb_output_dim_() const { return (int)efb_passthrough_features_.size() + (int)efb_bundles_.size(); }

    void reset_efb_state_() {
        efb_active_ = false;
        efb_passthrough_features_.clear();
        efb_bundles_.clear();
        efb_bundle_offsets_.clear();
    }

    void maybe_build_efb_plan_(const double *X, int N, int P) {
        if (!cfg_.efb_enabled || !X || N <= 0 || P <= 0) return;

        struct Candidate {
            int              feature = -1;
            std::vector<int> active_rows;
            double           max_val = 0.0;
        };

        std::vector<Candidate> candidates;
        candidates.reserve((size_t)P);

        for (int j = 0; j < P; ++j) {
            std::vector<int> active;
            active.reserve((size_t)std::max(1, N / 16));

            bool has_nan = false;
            double f_max = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < N; ++i) {
                const double v = X[(size_t)i * (size_t)P + (size_t)j];
                if (std::isnan(v)) {
                    has_nan = true;
                    break;
                }
                if (efb_nonzero_(v)) {
                    active.push_back(i);
                    if (v > f_max) f_max = v;
                }
            }
            if (has_nan) continue;

            const int nnz = (int)active.size();
            if (nnz < cfg_.efb_min_nonzero) continue;

            const double density = (double)nnz / (double)N;
            if (density > cfg_.efb_sparse_threshold) continue;

            candidates.push_back(Candidate{j, std::move(active), f_max});
        }

        if (candidates.size() < 2) return;

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &a, const Candidate &b) { return a.active_rows.size() > b.active_rows.size(); });

        struct BundleState {
            std::vector<int>     features;
            std::vector<uint8_t> occupied;
            std::vector<double>  offsets;
            double               current_offset = 0.0;
        };

        std::vector<BundleState> bundles;
        bundles.reserve(candidates.size() / 2 + 1);

        const int max_allowed_conflicts = (int)std::floor(cfg_.efb_max_conflict_rate * N);

        for (const auto &cand : candidates) {
            int placed = -1;
            for (int b = 0; b < (int)bundles.size(); ++b) {
                int conflicts = 0;
                for (int r : cand.active_rows) {
                    if (bundles[(size_t)b].occupied[(size_t)r]) {
                        conflicts++;
                        if (conflicts > max_allowed_conflicts) break;
                    }
                }
                if (conflicts <= max_allowed_conflicts) {
                    placed = b;
                    break;
                }
            }

            if (placed < 0) {
                BundleState st;
                st.occupied.assign((size_t)N, (uint8_t)0);
                bundles.push_back(std::move(st));
                placed = (int)bundles.size() - 1;
            }

            auto &dst = bundles[(size_t)placed];
            dst.features.push_back(cand.feature);
            dst.offsets.push_back(dst.current_offset);
            
            // Advance the offset by the max value of this feature
            // (add a small epsilon to strictly separate feature value spaces)
            double shift_amount = std::max(0.0, cand.max_val);
            dst.current_offset += shift_amount + 1e-4;

            for (int r : cand.active_rows) dst.occupied[(size_t)r] = (uint8_t)1;
        }

        std::vector<uint8_t> is_bundled((size_t)P, (uint8_t)0);
        efb_bundles_.clear();
        efb_bundle_offsets_.clear();
        
        for (auto &b : bundles) {
            if (b.features.size() < 2) continue;
            efb_bundles_.push_back(std::move(b.features));
            efb_bundle_offsets_.push_back(std::move(b.offsets));
            for (int f : efb_bundles_.back()) is_bundled[(size_t)f] = (uint8_t)1;
        }

        if (efb_bundles_.empty()) return;

        efb_passthrough_features_.clear();
        efb_passthrough_features_.reserve((size_t)P);
        for (int j = 0; j < P; ++j) {
            if (!is_bundled[(size_t)j]) efb_passthrough_features_.push_back(j);
        }

        efb_active_ = true;
    }

    std::vector<double> transform_with_efb_(const double *X, int N, int P) const {
        if (!efb_active_) { return std::vector<double>(X, X + (size_t)N * (size_t)P); }
        if (P != original_P_) throw std::invalid_argument("ForeForest: EFB transform P mismatch");

        const int           Pout = efb_output_dim_();
        std::vector<double> Xt((size_t)N * (size_t)Pout, 0.0);

        for (int outj = 0; outj < (int)efb_passthrough_features_.size(); ++outj) {
            const int srcj = efb_passthrough_features_[(size_t)outj];
            for (int i = 0; i < N; ++i) {
                Xt[(size_t)i * (size_t)Pout + (size_t)outj] = X[(size_t)i * (size_t)P + (size_t)srcj];
            }
        }

        const int bundle_col0 = (int)efb_passthrough_features_.size();
        for (int b = 0; b < (int)efb_bundles_.size(); ++b) {
            const int outj = bundle_col0 + b;
            for (size_t fi = 0; fi < efb_bundles_[(size_t)b].size(); ++fi) {
                const int srcj = efb_bundles_[(size_t)b][fi];
                const double offset = efb_bundle_offsets_[(size_t)b][fi];
                
                for (int i = 0; i < N; ++i) {
                    const double v = X[(size_t)i * (size_t)P + (size_t)srcj];
                    // Only apply offset if the feature is active for this row
                    if (std::isnan(v) || !efb_nonzero_(v)) continue;
                    
                    // Note: If there is a "conflict" (two bundled features active simultaneously), 
                    // this simply sums them. The offsets ensure that their value spaces are disjoint,
                    // but the sum of two disjoint spaces might land in a weird intermediate bin.
                    // A small conflict rate (e.g. 0.01%) means this noise is mostly ignored during tree construction.
                    Xt[(size_t)i * (size_t)Pout + (size_t)outj] += (v + offset);
                }
            }
        }

        return Xt;
    }

    static inline double sigmoid_(double x) {
        if (x >= 0.0) {
            const double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        }
        const double z = std::exp(x);
        return z / (1.0 + z);
    }

    double compute_base_score_(const double *y, int n) const {
        if (cfg_.objective == ForeForestConfig::Objective::BinaryLogloss || cfg_.objective == ForeForestConfig::Objective::BinaryFocalLoss) {
            double p = std::accumulate(y, y + n, 0.0) / std::max(1, n);
            p        = std::clamp(p, 1e-12, 1.0 - 1e-12);
            return std::log(p / (1.0 - p));
        }
        if (cfg_.objective == ForeForestConfig::Objective::HuberError) {
            // A simple approximation for Huber base score is the median. We'll use mean for simplicity unless we want to sort.
            return std::accumulate(y, y + n, 0.0) / std::max(1, n);
        }
        return std::accumulate(y, y + n, 0.0) / std::max(1, n);
    }

    void compute_grad_hess_(const double *y, int n, const std::vector<double> &F_ref, std::vector<double> &g,
                            std::vector<double> &h) {
        if ((int)F_ref.size() != n) throw std::invalid_argument("ForeForest: F_ref size mismatch");
        if ((int)g.size() != n || (int)h.size() != n)
            throw std::invalid_argument("ForeForest: gradient/hessian buffer size mismatch");

        if (cfg_.objective == ForeForestConfig::Objective::BinaryLogloss) {
            execx::bulk_iota(ctx_, n, [&](int i) {
                const double p = sigmoid_(F_ref[i]);
                g[i]           = p - y[i];
                h[i]           = std::max(1e-12, p * (1.0 - p));
            });
            return;
        }
        
        if (cfg_.objective == ForeForestConfig::Objective::BinaryFocalLoss) {
            const double gamma = cfg_.focal_gamma;
            execx::bulk_iota(ctx_, n, [&](int i) {
                const double yi = y[i];
                const double p = sigmoid_(F_ref[i]);
                const double p_t = yi * p + (1.0 - yi) * (1.0 - p); // Probability of true class
                const double mod = std::pow(1.0 - p_t, gamma);      // Modulation factor
                
                // Gradient of Focal Loss wrt log-odds F_ref
                // This is a simplified derivation for standard focal loss.
                g[i] = mod * (p - yi) - gamma * std::pow(1.0 - p_t, gamma - 1.0) * p * (1.0 - p) * (yi * (1.0 - p) - (1.0 - yi) * p) * std::log(std::max(1e-12, p_t));
                // Approximation for Hessian to ensure positive definiteness and stability:
                h[i] = std::max(1e-5, mod * p * (1.0 - p)); 
            });
            return;
        }
        
        if (cfg_.objective == ForeForestConfig::Objective::HuberError) {
            const double delta = cfg_.huber_delta;
            execx::bulk_iota(ctx_, n, [&](int i) {
                const double d = F_ref[i] - y[i];
                if (std::abs(d) <= delta) {
                    g[i] = d;
                    h[i] = 1.0;
                } else {
                    g[i] = delta * (d > 0 ? 1.0 : -1.0);
                    // Hessian for Huber outside delta is 0, but we use a small epsilon for tree splitting stability
                    h[i] = 1e-4; 
                }
            });
            return;
        }

        execx::bulk_iota(ctx_, n, [&](int i) {
            g[i] = F_ref[i] - y[i];
            h[i] = 1.0;
        });
    }

    void compute_grad_hess_(const double *y, int n, double scalar_F, std::vector<double> &g,
                            std::vector<double> &h) {
        std::vector<double> F((size_t)n, scalar_F);
        compute_grad_hess_(y, n, F, g, h);
    }

    TreeConfig make_tree_cfg_() const {
        TreeConfig tc               = cfg_.tree_cfg;
        tc.colsample_bytree_percent = std::clamp((int)std::round(100.0 * cfg_.colsample_bytree), 1, 100);
        tc.colsample_bynode_percent = std::clamp((int)std::round(100.0 * cfg_.colsample_bynode), 1, 100);
        if (tc.goss.enabled && !tc.goss.scale_hessian) { tc.goss.scale_hessian = true; }
        return tc;
    }

    UnifiedTree build_tree_(const std::vector<double> &g, const std::vector<double> &h, const std::vector<int> &rows,
                            uint64_t tree_seed) {
        TreeConfig tc = make_tree_cfg_();
        tc.rng_seed   = tree_seed;

        UnifiedTree T(tc, ghs_.get());

        // Set matrices: exact splits use Xraw_exact_, neural leaves use Xraw_neural_
        if (Xraw_exact_) { T.set_raw_matrix(Xraw_exact_, Xmiss_exact_); }
        T.set_raw_for_neural(Xraw_neural_, Xmiss_neural_);
        T.set_neural_leaf_config(cfg_.neural_cfg);

        T.fit_with_row_ids(*codes_, N_, P_, g, h, rows);
        return T;
    }

    // ===================== Bagging ===========================
    void train_bagging_(std::vector<double> &g, std::vector<double> &h, const double *y) {
        execx::bulk_iota(ctx_, N_, [&](int i) {
            g[i] = -y[i];
            h[i] = 1.0;
        });

        const int                     M = cfg_.n_estimators;
        std::vector<std::vector<int>> boot_rows(M);
        std::vector<std::mt19937_64>  rngs(M);

        for (int t = 0; t < M; ++t) { rngs[t] = std::mt19937_64(cfg_.rng_seed + 0x9E3779B97F4A7C15ULL * (t + 1)); }

        auto gen_sample = [&](int t) {
            auto                              &rows = boot_rows[t];
            auto                              &rng  = rngs[t];
            std::uniform_int_distribution<int> J(0, N_ - 1);

            const int k = std::max(1, (int)std::round(cfg_.rf_row_subsample * N_));

            if (cfg_.rf_bootstrap) {
                rows.resize(k);
                for (int i = 0; i < k; ++i) rows[i] = J(rng);
                if (cfg_.rf_bootstrap_dedup) {
                    std::sort(rows.begin(), rows.end());
                    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                    if (rows.empty()) rows.push_back(J(rng));
                }
            } else {
                std::vector<int> all(N_);
                std::iota(all.begin(), all.end(), 0);
                std::shuffle(all.begin(), all.end(), rng);
                rows.assign(all.begin(), all.begin() + std::min(k, N_));
            }
        };

        if (cfg_.rf_parallel) {
            execx::bulk_iota(ctx_, M, gen_sample);
        } else {
            for (int t = 0; t < M; ++t) gen_sample(t);
        }

        trees_.resize(M);
        tree_weights_.assign(M, 1.0);

        auto build_one = [&](int t) {
            uint64_t seed = cfg_.rng_seed + 0x9E3779B97F4A7C15ULL * (t + 1);
            trees_[t]     = build_tree_(g, h, boot_rows[t], seed);
        };

        if (cfg_.rf_parallel) {
            execx::bulk_iota(ctx_, M, build_one);
        } else {
            for (int t = 0; t < M; ++t) build_one(t);
        }
    }

    // ===================== Boosting with DART ===========================
    void train_gbdt_(std::vector<double> &g, std::vector<double> &h, const double *y) {
        train_gbdt_(g, h, y, nullptr, 0, nullptr, nullptr);
    }

    void train_gbdt_(std::vector<double> &g, std::vector<double> &h, const double *y,
                     const std::vector<uint16_t> *Xb_valid, int N_valid, const double *y_valid,
                     const double *Xraw_valid) {
        const int           M = cfg_.n_estimators;
        std::vector<double> F(N_, base_score_);
        std::vector<double> pred_buffer(N_);
        const bool          has_valid = (Xb_valid && y_valid && N_valid > 0);
        std::vector<double> F_valid;
        std::vector<double> pred_valid_buffer;
        if (has_valid) {
            F_valid.assign((size_t)N_valid, base_score_);
            pred_valid_buffer.resize((size_t)N_valid);
        }
        int rounds_without_improve = 0;

        for (int m = 0; m < M; ++m) {
            std::vector<int>    dropped = select_dropout_trees_();
            std::vector<double> Fbase   = compute_base_prediction_(F, dropped, pred_buffer);

            const auto &F_ref = dropped.empty() ? F : Fbase;
            compute_grad_hess_(y, N_, F_ref, g, h);

            std::vector<int> rows = sample_rows_for_gbdt_();
            uint64_t         seed = cfg_.rng_seed + 0x9E3779B97F4A7C15ULL * (m + 1);
            UnifiedTree      T    = build_tree_(g, h, rows, seed);

            double wt_new = cfg_.learning_rate;
            if (cfg_.dart_enabled && cfg_.dart_normalize && !dropped.empty()) {
                wt_new = apply_dart_normalization_(dropped, F, pred_buffer, has_valid ? &F_valid : nullptr, Xb_valid,
                                                   N_valid, Xraw_valid, has_valid ? &pred_valid_buffer : nullptr);
            }

            trees_.push_back(std::move(T));
            tree_weights_.push_back(wt_new);

            predict_tree_on_binned_(trees_.back(), *codes_, N_, P_, Xraw_neural_, pred_buffer);
            for (int i = 0; i < N_; ++i) { F[i] += wt_new * pred_buffer[i]; }

            train_metric_history_.push_back(compute_metric_from_margin_(F, y, N_));

            if (has_valid) {
                predict_tree_on_binned_(trees_.back(), *Xb_valid, N_valid, P_, Xraw_valid, pred_valid_buffer);
                for (int i = 0; i < N_valid; ++i) { F_valid[(size_t)i] += wt_new * pred_valid_buffer[(size_t)i]; }

                const double valid_metric = compute_metric_from_margin_(F_valid, y_valid, N_valid);
                valid_metric_history_.push_back(valid_metric);

                const bool improved = (valid_metric + cfg_.early_stopping_min_delta) < best_score_;
                if (improved) {
                    best_score_            = valid_metric;
                    best_iteration_        = (int)trees_.size();
                    rounds_without_improve = 0;
                } else if (cfg_.early_stopping_enabled) {
                    ++rounds_without_improve;
                    if (rounds_without_improve >= cfg_.early_stopping_rounds) {
                        early_stopped_ = true;
                        break;
                    }
                }
            } else {
                best_iteration_ = (int)trees_.size();
            }
        }

        if (has_valid && best_iteration_ == 0 && !valid_metric_history_.empty()) {
            best_iteration_ = (int)trees_.size();
            best_score_     = valid_metric_history_.back();
        }

        if (has_valid && cfg_.early_stopping_enabled && best_iteration_ > 0 && best_iteration_ < (int)trees_.size()) {
            trees_.resize((size_t)best_iteration_);
            tree_weights_.resize((size_t)best_iteration_);
            if ((int)train_metric_history_.size() > best_iteration_)
                train_metric_history_.resize((size_t)best_iteration_);
            if ((int)valid_metric_history_.size() > best_iteration_)
                valid_metric_history_.resize((size_t)best_iteration_);
        }
    }

    std::vector<int> select_dropout_trees_() {
        std::vector<int> dropped;
        if (!cfg_.dart_enabled || trees_.empty()) return dropped;

        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (size_t t = 0; t < trees_.size(); ++t) {
            if (U(rng_) < cfg_.dart_drop_rate) { dropped.push_back((int)t); }
        }

        if ((int)dropped.size() > cfg_.dart_max_drop) {
            std::shuffle(dropped.begin(), dropped.end(), rng_);
            dropped.resize(cfg_.dart_max_drop);
        }

        if (dropped.empty() && cfg_.dart_one_drop_min && cfg_.dart_max_drop > 0) {
            std::uniform_int_distribution<int> D(0, (int)trees_.size() - 1);
            dropped.push_back(D(rng_));
        }

        return dropped;
    }

    std::vector<double> compute_base_prediction_(const std::vector<double> &F, const std::vector<int> &dropped,
                                                 std::vector<double> &pred_buffer) const {
        if (dropped.empty()) return F;

        std::vector<double> Fbase = F;
        for (int t : dropped) {
            predict_tree_on_binned_(trees_[t], *codes_, N_, P_, Xraw_neural_, pred_buffer);
            const double wt = tree_weights_[t];
            execx::bulk_iota(ctx_, N_, [&](int i) {
                Fbase[i] -= wt * pred_buffer[i];
            });
        }
        return Fbase;
    }

    std::vector<int> sample_rows_for_gbdt_() {
        std::vector<int> rows;
        if (cfg_.gbdt_use_subsample && cfg_.gbdt_row_subsample < 1.0) {
            std::uniform_real_distribution<double> U(0.0, 1.0);
            for (int i = 0; i < N_; ++i) {
                if (U(rng_) < cfg_.gbdt_row_subsample) { rows.push_back(i); }
            }
            if (rows.empty()) {
                std::uniform_int_distribution<int> D(0, N_ - 1);
                rows.push_back(D(rng_));
            }
        } else {
            rows.resize(N_);
            std::iota(rows.begin(), rows.end(), 0);
        }
        return rows;
    }

    std::vector<int> sample_rows_for_fw_() {
        std::vector<int> rows;
        if (cfg_.fw_use_subsample && cfg_.fw_row_subsample < 1.0) {
            std::uniform_real_distribution<double> U(0.0, 1.0);
            for (int i = 0; i < N_; ++i) {
                if (U(rng_) < cfg_.fw_row_subsample) { rows.push_back(i); }
            }
            if (rows.empty()) {
                std::uniform_int_distribution<int> D(0, N_ - 1);
                rows.push_back(D(rng_));
            }
        } else {
            rows.resize(N_);
            std::iota(rows.begin(), rows.end(), 0);
        }
        return rows;
    }

    double fw_line_search_alpha_(const std::vector<double> &F, const std::vector<double> &h_pred, const double *y,
                                 int n) const {
        if (!y || n <= 0 || (int)F.size() < n || (int)h_pred.size() < n) return 0.0;

        const int    points    = std::max(2, cfg_.fw_line_search_points);
        const double alpha_max = std::max(0.0, cfg_.fw_alpha_max);
        double       best_a    = 0.0;
        double       best_obj  = std::numeric_limits<double>::infinity();

        for (int k = 0; k < points; ++k) {
            const double a   = (points <= 1) ? 0.0 : (alpha_max * (double)k / (double)(points - 1));
            double       mse = 0.0;
            for (int i = 0; i < n; ++i) {
                const double d = (F[(size_t)i] + a * h_pred[(size_t)i]) - y[i];
                mse += d * d;
            }
            mse /= std::max(1, n);
            const double obj = mse + cfg_.fw_nu * std::abs(a);
            if (obj < best_obj) {
                best_obj = obj;
                best_a   = a;
            }
        }
        return best_a;
    }

    void train_fwboost_(std::vector<double> &g, std::vector<double> &h, const double *y) {
        train_fwboost_(g, h, y, nullptr, 0, nullptr, nullptr);
    }

    void train_fwboost_(std::vector<double> &g, std::vector<double> &h, const double *y,
                        const std::vector<uint16_t> *Xb_valid, int N_valid, const double *y_valid,
                        const double *Xraw_valid) {
        const int           M = cfg_.n_estimators;
        std::vector<double> F(N_, base_score_);
        std::vector<double> pred_buffer(N_);
        const bool          has_valid = (Xb_valid && y_valid && N_valid > 0);
        std::vector<double> F_valid;
        std::vector<double> pred_valid_buffer;
        if (has_valid) {
            F_valid.assign((size_t)N_valid, base_score_);
            pred_valid_buffer.resize((size_t)N_valid);
        }
        int rounds_without_improve = 0;

        for (int m = 0; m < M; ++m) {
            compute_grad_hess_(y, N_, F, g, h);

            std::vector<int> rows = sample_rows_for_fw_();
            uint64_t         seed = cfg_.rng_seed + 0x9E3779B97F4A7C15ULL * (m + 1);
            UnifiedTree      T    = build_tree_(g, h, rows, seed);

            predict_tree_on_binned_(T, *codes_, N_, P_, Xraw_neural_, pred_buffer);

            const double alpha = fw_line_search_alpha_(F, pred_buffer, y, N_);
            if (alpha <= cfg_.fw_alpha_tol) {
                early_stopped_ = true;
                break;
            }

            const double wt_new = cfg_.learning_rate * alpha;
            trees_.push_back(std::move(T));
            tree_weights_.push_back(wt_new);

            for (int i = 0; i < N_; ++i) { F[i] += wt_new * pred_buffer[(size_t)i]; }

            train_metric_history_.push_back(compute_metric_from_margin_(F, y, N_));

            if (has_valid) {
                predict_tree_on_binned_(trees_.back(), *Xb_valid, N_valid, P_, Xraw_valid, pred_valid_buffer);
                for (int i = 0; i < N_valid; ++i) { F_valid[(size_t)i] += wt_new * pred_valid_buffer[(size_t)i]; }

                const double valid_metric = compute_metric_from_margin_(F_valid, y_valid, N_valid);
                valid_metric_history_.push_back(valid_metric);

                const bool improved = (valid_metric + cfg_.early_stopping_min_delta) < best_score_;
                if (improved) {
                    best_score_            = valid_metric;
                    best_iteration_        = (int)trees_.size();
                    rounds_without_improve = 0;
                } else if (cfg_.early_stopping_enabled) {
                    ++rounds_without_improve;
                    if (rounds_without_improve >= cfg_.early_stopping_rounds) {
                        early_stopped_ = true;
                        break;
                    }
                }
            } else {
                best_iteration_ = (int)trees_.size();
            }
        }

        if (has_valid && best_iteration_ == 0 && !valid_metric_history_.empty()) {
            best_iteration_ = (int)trees_.size();
            best_score_     = valid_metric_history_.back();
        }

        if (has_valid && cfg_.early_stopping_enabled && best_iteration_ > 0 && best_iteration_ < (int)trees_.size()) {
            trees_.resize((size_t)best_iteration_);
            tree_weights_.resize((size_t)best_iteration_);
            if ((int)train_metric_history_.size() > best_iteration_)
                train_metric_history_.resize((size_t)best_iteration_);
            if ((int)valid_metric_history_.size() > best_iteration_)
                valid_metric_history_.resize((size_t)best_iteration_);
        }
    }

    double apply_dart_normalization_(const std::vector<int> &dropped, std::vector<double> &F,
                                     std::vector<double> &pred_buffer, std::vector<double> *F_valid = nullptr,
                                     const std::vector<uint16_t> *Xb_valid = nullptr, int N_valid = 0,
                                     const double        *Xraw_valid        = nullptr,
                                     std::vector<double> *pred_valid_buffer = nullptr) {
        const int    k              = (int)dropped.size();
        const double dropped_scale  = double(k) / double(k + 1);
        const double new_tree_scale = 1.0 / double(k + 1);
        const bool   use_valid      = (F_valid && Xb_valid && N_valid > 0 && pred_valid_buffer);

        for (int t : dropped) {
            predict_tree_on_binned_(trees_[t], *codes_, N_, P_, Xraw_neural_, pred_buffer);

            const double old_w = tree_weights_[t];
            const double new_w = old_w * dropped_scale;
            const double delta = new_w - old_w;

            for (int i = 0; i < N_; ++i) { F[i] += delta * pred_buffer[i]; }
            if (use_valid) {
                predict_tree_on_binned_(trees_[t], *Xb_valid, N_valid, P_, Xraw_valid, *pred_valid_buffer);
                for (int i = 0; i < N_valid; ++i) { (*F_valid)[(size_t)i] += delta * (*pred_valid_buffer)[(size_t)i]; }
            }

            tree_weights_[t] = new_w;
        }

        return cfg_.learning_rate * new_tree_scale;
    }

    double compute_metric_from_margin_(const std::vector<double> &margin, const double *y, int n) {
        if (!y || n <= 0 || (int)margin.size() < n) return std::numeric_limits<double>::infinity();
        if (cfg_.objective == ForeForestConfig::Objective::BinaryLogloss) {
            double loss = 0.0;
            // Parallel reduction would be ideal, but for now we keep the simple loop with an atomic or std::reduce in C++17.
            // Given n is large, we can just use a basic loop as it's not the main bottleneck yet.
            for (int i = 0; i < n; ++i) {
                const double yi = y[i];
                const double pi = std::clamp(sigmoid_(margin[(size_t)i]), 1e-12, 1.0 - 1e-12);
                loss += -(yi * std::log(pi) + (1.0 - yi) * std::log(1.0 - pi));
            }
            return loss / std::max(1, n);
        }
        
        if (cfg_.objective == ForeForestConfig::Objective::BinaryFocalLoss) {
            double loss = 0.0;
            const double gamma = cfg_.focal_gamma;
            for (int i = 0; i < n; ++i) {
                const double yi = y[i];
                const double pi = std::clamp(sigmoid_(margin[(size_t)i]), 1e-12, 1.0 - 1e-12);
                const double p_t = yi * pi + (1.0 - yi) * (1.0 - pi);
                loss += -std::pow(1.0 - p_t, gamma) * std::log(p_t);
            }
            return loss / std::max(1, n);
        }
        
        if (cfg_.objective == ForeForestConfig::Objective::HuberError) {
            double loss = 0.0;
            const double delta = cfg_.huber_delta;
            for (int i = 0; i < n; ++i) {
                const double d = std::abs(margin[(size_t)i] - y[i]);
                if (d <= delta) {
                    loss += 0.5 * d * d;
                } else {
                    loss += delta * (d - 0.5 * delta);
                }
            }
            return loss / std::max(1, n);
        }

        double mse = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = margin[(size_t)i] - y[i];
            mse += d * d;
        }
        return mse / std::max(1, n);
    }

    // ===================== Prediction ===========================
    std::vector<double> predict_from_binned_(const std::vector<uint16_t> &Xb, int N, int /*P*/,
                                             const double *Xraw_for_neural) const {
        std::vector<double> out(N, 0.0), pred(N);
        for (size_t t = 0; t < trees_.size(); ++t) {
            const double wt = tree_weights_[t];
            predict_tree_on_binned_(trees_[t], Xb, N, P_, Xraw_for_neural, pred);
            for (int i = 0; i < N; ++i) { out[i] += wt * pred[i]; }
        }

        if (cfg_.mode == ForeForestConfig::Mode::Bagging && !trees_.empty()) {
            const double invT = 1.0 / (double)trees_.size();
            for (double &v : out) v *= invT;
        }
        return out;
    }

    std::vector<double> finalize_prediction_(std::vector<double> out, bool apply_link) const {
        if (cfg_.mode == ForeForestConfig::Mode::GBDT || cfg_.mode == ForeForestConfig::Mode::FWBoost) {
            for (double &v : out) v += base_score_;
            if (apply_link && cfg_.objective == ForeForestConfig::Objective::BinaryLogloss) {
                for (double &v : out) v = sigmoid_(v);
            }
        }
        return out;
    }

    static void predict_tree_on_binned_(const UnifiedTree &T, const std::vector<uint16_t> &Xb, int N, int P,
                                        const double *Xraw_for_neural, std::vector<double> &dst) {
        auto v = T.predict(Xb, N, P, Xraw_for_neural);
        dst.assign(v.begin(), v.end());
    }

private:
    ForeForestConfig cfg_;
    std::mt19937_64  rng_;

    std::unique_ptr<GradientHistogramSystem> ghs_;
    std::shared_ptr<std::vector<uint16_t>>   codes_;
    int                                      miss_id_ = -1;

    // Separate raw matrices for different purposes
    const double  *Xraw_exact_   = nullptr; // For exact splits (optional, user-provided)
    const uint8_t *Xmiss_exact_  = nullptr;
    const double  *Xraw_neural_  = nullptr; // For neural leaves (typically training data)
    const uint8_t *Xmiss_neural_ = nullptr;

    std::vector<UnifiedTree> trees_;
    std::vector<double>      tree_weights_;
    std::vector<double>      train_metric_history_;
    std::vector<double>      valid_metric_history_;
    int                      best_iteration_ = 0;
    double                   best_score_     = std::numeric_limits<double>::infinity();
    bool                     early_stopped_  = false;

    int    N_ = 0, P_ = 0;
    int    original_P_ = 0;
    double base_score_ = 0.0;

    bool                          efb_active_ = false;
    std::vector<int>              efb_passthrough_features_;
    std::vector<std::vector<int>> efb_bundles_;
    std::vector<std::vector<double>> efb_bundle_offsets_;

    mutable execx::Context ctx_;
};

} // namespace foretree
